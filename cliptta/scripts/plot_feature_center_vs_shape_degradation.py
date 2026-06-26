#!/usr/bin/env python3
"""Separate CLIP feature degradation into center shift and shape distortion."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "work" / ".mplconfig"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import font_manager
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from run_closed_set_cifar_benchmarks import DEFAULT_DATA_ROOT, FAMILIES  # noqa: E402
from ttavlm.datasets import CORRUPTIONS, return_train_val_datasets  # noqa: E402
from ttavlm.models import return_base_model  # noqa: E402
from ttavlm.models.clip import tokenize  # noqa: E402


DEFAULT_TEMPLATE = "a photo of a {}"
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "complete"
    / "final_paper_experiments"
    / "05_qualitative_analysis"
    / "feature_center_vs_shape_degradation"
)
DEFAULT_FONT_ZIP = WORKSPACE_ROOT / "TimesNewerRoman.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=list(FAMILIES.keys()), default="cifar10")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--corruptions", nargs="+", default=["all"])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-model-name", type=str, default="clip-ViT-B/16")
    parser.add_argument("--template", type=str, default=DEFAULT_TEMPLATE)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--feature-budgets", nargs="+", type=float, default=[0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20])
    parser.add_argument("--logit-budgets", nargs="+", type=float, default=[0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20])
    parser.add_argument("--font-zip", type=Path, default=DEFAULT_FONT_ZIP)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    ensure_dir(path.parent)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def configure_font(font_zip: Path) -> None:
    font_name = "Times New Roman"
    if font_zip.exists():
        font_dir = REPO_ROOT / "work" / ".fonts" / font_zip.stem
        ensure_dir(font_dir)
        with zipfile.ZipFile(font_zip) as archive:
            for member in archive.namelist():
                if member.lower().endswith((".otf", ".ttf")):
                    target = font_dir / Path(member).name
                    if not target.exists():
                        target.write_bytes(archive.read(member))
        for font_path in sorted(font_dir.glob("*.[ot]tf")):
            font_manager.fontManager.addfont(str(font_path))
        font_name = "Times Newer Roman"

    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.serif": [font_name, "Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.dpi": 140,
        }
    )


def choose_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
        device = torch.device(device_name)
        torch.cuda.set_device(device)
        return device
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_corruptions(values: Sequence[str]) -> List[str]:
    if list(values) == ["all"]:
        return list(CORRUPTIONS)
    return list(values)


def stratified_indices(dataset: Dataset, max_samples_per_class: int | None, seed: int) -> List[int]:
    labels = getattr(dataset, "labels", None)
    if labels is None:
        labels = getattr(dataset, "targets", None)
    if labels is None:
        raise AttributeError("Dataset does not expose labels/targets.")
    labels_np = np.asarray(labels, dtype=np.int64)
    all_indices = np.arange(labels_np.shape[0], dtype=np.int64)
    if max_samples_per_class is None:
        return all_indices.tolist()
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    for class_id in sorted(np.unique(labels_np).astype(int).tolist()):
        class_indices = all_indices[labels_np == class_id]
        if class_indices.size <= max_samples_per_class:
            chosen = np.sort(class_indices)
        else:
            chosen = np.sort(rng.choice(class_indices, size=max_samples_per_class, replace=False))
        selected.extend(int(idx) for idx in chosen.tolist())
    selected.sort()
    return selected


def build_loader(dataset: Dataset, indices: Sequence[int], batch_size: int, workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def load_dataset(name: str, data_root: Path, transform, corruption: str | None, severity: int | None = None) -> Dataset:
    _, dataset = return_train_val_datasets(
        name=name,
        data_dir=str(data_root),
        train_transform=transform,
        val_transform=transform,
        shift=corruption,
        severity=severity,
    )
    return dataset


def batch_image_tensor(batch) -> Tensor:
    images = batch["image"]
    if isinstance(images, (list, tuple)):
        return images[0]
    return images


@torch.no_grad()
def extract_features(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[Tensor, Tensor]:
    model.eval()
    visual = model.visual if hasattr(model, "visual") else model
    if not hasattr(visual, "use_local"):
        visual.use_local = False
    dtype = getattr(visual, "dtype", torch.float32)
    features: List[Tensor] = []
    labels: List[Tensor] = []
    for batch in loader:
        images = batch_image_tensor(batch).to(device, non_blocking=True)
        outputs = model.encode_image(images.type(dtype)).float()
        outputs = F.normalize(outputs, dim=-1)
        features.append(outputs.cpu())
        labels.append(batch["target"].long().cpu())
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


@torch.no_grad()
def encode_text_features(model: torch.nn.Module, class_names: Sequence[str], template: str, device: torch.device) -> Tensor:
    prompts = [template.format(name.replace("_", " ")) for name in class_names]
    tokens = tokenize(prompts, truncate=True).to(device)
    text_features = model.encode_text(tokens).float()
    return F.normalize(text_features, dim=-1).cpu()


def top1(features: Tensor, labels: Tensor, text_features: Tensor) -> float:
    logits = features.float() @ text_features.float().t()
    predictions = logits.argmax(dim=1)
    return 100.0 * float(predictions.eq(labels).float().mean().item())


def top1_logits(logits: Tensor, labels: Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return 100.0 * float(predictions.eq(labels).float().mean().item())


def normalize_features(features: Tensor) -> Tensor:
    return F.normalize(features.float(), dim=-1)


def class_centered_residual(delta: Tensor, labels: Tensor, num_classes: int) -> Tensor:
    centered = delta.clone()
    for class_id in range(num_classes):
        mask = labels == class_id
        if mask.any():
            centered[mask] = centered[mask] - delta[mask].mean(dim=0, keepdim=True)
    return centered


def permute_within_class(values: Tensor, labels: Tensor, seed: int) -> Tensor:
    rng = np.random.default_rng(seed)
    output = torch.empty_like(values)
    labels_np = labels.cpu().numpy()
    for class_id in sorted(np.unique(labels_np).astype(int).tolist()):
        indices = np.where(labels_np == class_id)[0]
        permuted = rng.permutation(indices)
        output[torch.from_numpy(indices).long()] = values[torch.from_numpy(permuted).long()]
    return output


def zero_mean_random_shape_like(values: Tensor, labels: Tensor, seed: int) -> Tensor:
    generator = torch.Generator(device=values.device)
    generator.manual_seed(seed)
    noise = torch.randn(values.shape, generator=generator, dtype=values.dtype, device=values.device)
    return class_centered_residual(noise, labels, int(labels.max().item()) + 1)


def summarize_alpha(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, float], List[float]] = {}
    for row in rows:
        key = (str(row["axis"]), float(row["alpha"]))
        grouped.setdefault(key, []).append(float(row["top1"]))
    summary: List[Dict[str, object]] = []
    for (axis, alpha), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        arr = np.asarray(values, dtype=np.float64)
        summary.append(
            {
                "axis": axis,
                "alpha": alpha,
                "mean_top1": float(arr.mean()),
                "std_top1": float(arr.std(ddof=0)),
                "num_corruptions": int(arr.size),
            }
        )
    return summary


def summarize_budget(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, float], List[float]] = {}
    for row in rows:
        key = (str(row["axis"]), float(row["budget"]))
        grouped.setdefault(key, []).append(float(row["top1"]))
    summary: List[Dict[str, object]] = []
    for (axis, budget), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        arr = np.asarray(values, dtype=np.float64)
        summary.append(
            {
                "axis": axis,
                "budget": budget,
                "mean_top1": float(arr.mean()),
                "std_top1": float(arr.std(ddof=0)),
                "num_corruptions": int(arr.size),
            }
        )
    return summary


def find_drop_crossing(summary_rows: Sequence[Mapping[str, object]], axis: str, baseline_top1: float, target_drop: float) -> float | None:
    rows = sorted([row for row in summary_rows if row["axis"] == axis], key=lambda row: float(row["alpha"]))
    threshold = baseline_top1 - target_drop
    previous_alpha: float | None = None
    previous_top1: float | None = None
    for row in rows:
        alpha = float(row["alpha"])
        value = float(row["mean_top1"])
        if value <= threshold:
            if previous_alpha is None or previous_top1 is None:
                return alpha
            if previous_top1 == value:
                return alpha
            ratio = (previous_top1 - threshold) / (previous_top1 - value)
            return previous_alpha + ratio * (alpha - previous_alpha)
        previous_alpha = alpha
        previous_top1 = value
    return None


def plot_budget_summary(summary_rows: Sequence[Mapping[str, object]], output_root: Path, baseline_top1: float) -> None:
    labels = {
        "matched_feature_center_shift": "Center Shift",
        "matched_feature_shape_distortion": "Paired residual shape",
        "matched_feature_class_centered_shape": "Class-centered shape",
        "matched_feature_permuted_shape": "Geometry Distortion",
        "matched_feature_random_shape": "Random zero-mean shape",
        "matched_logit_bias": "Logit bias only",
        "matched_logit_residual": "Logit residual only",
    }
    colors = {
        "matched_feature_center_shift": "#d62728",
        "matched_feature_shape_distortion": "#1f77b4",
        "matched_feature_class_centered_shape": "#9467bd",
        "matched_feature_permuted_shape": "#ff7f0e",
        "matched_feature_random_shape": "#17becf",
        "matched_logit_bias": "#d62728",
        "matched_logit_residual": "#1f77b4",
    }
    for axes, filename, xlabel in [
        (
            ["matched_feature_center_shift", "matched_feature_shape_distortion"],
            "matched_feature_l2_budget_degradation.png",
            "Mean feature perturbation budget (L2)",
        ),
        (
            [
                "matched_feature_center_shift",
                "matched_feature_permuted_shape",
            ],
            "controlled_matched_feature_l2_budget_degradation.png",
            "Mean feature perturbation budget (L2)",
        ),
        (
            ["matched_logit_bias", "matched_logit_residual"],
            "matched_logit_l2_budget_degradation.png",
            "Mean logit perturbation budget (L2)",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        for axis in axes:
            rows = sorted([row for row in summary_rows if row["axis"] == axis], key=lambda row: float(row["budget"]))
            budgets = np.asarray([float(row["budget"]) for row in rows], dtype=np.float64)
            mean = np.asarray([float(row["mean_top1"]) for row in rows], dtype=np.float64)
            std = np.asarray([float(row["std_top1"]) for row in rows], dtype=np.float64)
            ax.plot(budgets, mean, marker="o", linewidth=2.4, markersize=5.2, color=colors[axis], label=labels[axis])
            ax.fill_between(budgets, mean - std, mean + std, color=colors[axis], alpha=0.10, linewidth=0)
        ax.axhline(baseline_top1, color="black", linestyle=":", linewidth=1.4, label="Clean CLIP")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Top-1 accuracy (%)")
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)
        ax.legend(frameon=True, facecolor="white", edgecolor="0.75")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(output_root / filename.replace(".png", f".{suffix}"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_summary(summary_rows: Sequence[Mapping[str, object]], output_root: Path, baseline_top1: float) -> None:
    labels = {
        "feature_center_shift": "Center shift only",
        "feature_shape_distortion": "Shape distortion only",
        "feature_combined": "Center + shape",
        "cbc_feature_center_shift": "CBC-aligned center shift only",
        "cbc_feature_shape_distortion": "CBC-aligned shape distortion only",
        "cbc_feature_combined": "CBC-aligned center + shape",
        "logit_bias": "Logit bias only",
        "logit_residual": "Logit residual only",
        "logit_combined": "Logit bias + residual",
        "cbc_logit_bias": "CBC logit bias only",
        "cbc_logit_residual": "CBC logit residual only",
        "cbc_logit_combined": "CBC logit bias + residual",
    }
    colors = {
        "feature_center_shift": "#d62728",
        "feature_shape_distortion": "#1f77b4",
        "feature_combined": "#2ca02c",
        "cbc_feature_center_shift": "#d62728",
        "cbc_feature_shape_distortion": "#1f77b4",
        "cbc_feature_combined": "#2ca02c",
        "logit_bias": "#d62728",
        "logit_residual": "#1f77b4",
        "logit_combined": "#2ca02c",
        "cbc_logit_bias": "#d62728",
        "cbc_logit_residual": "#1f77b4",
        "cbc_logit_combined": "#2ca02c",
    }
    linestyles = {
        "feature_center_shift": "-",
        "feature_shape_distortion": "-",
        "feature_combined": "--",
        "cbc_feature_center_shift": "-",
        "cbc_feature_shape_distortion": "-",
        "cbc_feature_combined": "--",
        "logit_bias": "-",
        "logit_residual": "-",
        "logit_combined": "--",
        "cbc_logit_bias": "-",
        "cbc_logit_residual": "-",
        "cbc_logit_combined": "--",
    }

    for prefix, axes, filename in [
        (
            "Feature-space decomposition",
            ["feature_center_shift", "feature_shape_distortion", "feature_combined"],
            "feature_center_vs_shape_degradation.png",
        ),
        (
            "Logit-space decomposition",
            ["logit_bias", "logit_residual", "logit_combined"],
            "logit_bias_vs_residual_degradation.png",
        ),
        (
            "CBC-aligned feature-space decomposition",
            ["cbc_feature_center_shift", "cbc_feature_shape_distortion", "cbc_feature_combined"],
            "cbc_aligned_feature_center_vs_shape_degradation.png",
        ),
        (
            "CBC-aligned logit-space decomposition",
            ["cbc_logit_bias", "cbc_logit_residual", "cbc_logit_combined"],
            "cbc_aligned_logit_bias_vs_residual_degradation.png",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(7.4, 5.2))
        for axis in axes:
            rows = sorted([row for row in summary_rows if row["axis"] == axis], key=lambda row: float(row["alpha"]))
            if not rows:
                continue
            alpha = np.asarray([float(row["alpha"]) for row in rows], dtype=np.float64)
            mean = np.asarray([float(row["mean_top1"]) for row in rows], dtype=np.float64)
            std = np.asarray([float(row["std_top1"]) for row in rows], dtype=np.float64)
            ax.plot(alpha, mean, marker="o", linewidth=2.2, markersize=5.0, color=colors[axis], linestyle=linestyles[axis], label=labels[axis])
            ax.fill_between(alpha, mean - std, mean + std, color=colors[axis], alpha=0.10, linewidth=0)
        ax.axhline(baseline_top1, color="black", linestyle=":", linewidth=1.4, label="Clean CLIP")
        ax.set_xlabel("Perturbation strength")
        ax.set_ylabel("Top-1 accuracy (%)")
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)
        ax.legend(frameon=True, facecolor="white", edgecolor="0.75")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(output_root / filename.replace(".png", f".{suffix}"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    rows = [row for row in summary_rows if row["alpha"] == 1.0 and row["axis"] in labels]
    rows = sorted(rows, key=lambda row: float(row["mean_top1"]))
    names = [labels[str(row["axis"])] for row in rows]
    values = [float(row["mean_top1"]) for row in rows]
    bar_colors = [colors[str(row["axis"])] for row in rows]
    ax.barh(names, values, color=bar_colors, alpha=0.82)
    ax.axvline(baseline_top1, color="black", linestyle=":", linewidth=1.4)
    ax.set_xlabel("Top-1 accuracy at full empirical perturbation (%)")
    ax.grid(axis="x", alpha=0.22, linestyle="--", linewidth=0.7)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_root / f"full_perturbation_axis_comparison.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    configure_font(args.font_zip)
    device = choose_device(args.device)
    output_root = args.output_root
    ensure_dir(output_root)

    corruptions = resolve_corruptions(args.corruptions)
    family_cfg = FAMILIES[args.family]
    clean_name = str(family_cfg["clean_dataset"])
    corrupt_name = str(family_cfg["corrupted_dataset"])

    model, transform = return_base_model(
        name=args.base_model_name,
        device=device,
        dataset=corrupt_name,
        path_to_weights=str(REPO_ROOT / "work"),
    )
    model.eval()

    clean_dataset = load_dataset(clean_name, args.data_root, transform, None, None)
    class_names = list(clean_dataset.class_names)
    indices = stratified_indices(clean_dataset, args.max_samples_per_class, args.seed)
    clean_loader = build_loader(clean_dataset, indices, args.batch_size, args.num_workers, device)
    clean_features, clean_labels = extract_features(model, clean_loader, device)
    text_features = encode_text_features(model, class_names, args.template, device)
    clean_logits = clean_features.float() @ text_features.float().t()
    clean_top1 = top1(clean_features, clean_labels, text_features)

    metadata = {
        "family": args.family,
        "clean_dataset": clean_name,
        "corrupted_dataset": corrupt_name,
        "severity": int(args.severity),
        "corruptions": corruptions,
        "base_model_name": args.base_model_name,
        "template": args.template,
        "num_samples": int(clean_features.shape[0]),
        "max_samples_per_class": args.max_samples_per_class,
        "alphas": [float(alpha) for alpha in args.alphas],
        "feature_budgets": [float(budget) for budget in args.feature_budgets],
        "logit_budgets": [float(budget) for budget in args.logit_budgets],
        "clean_top1": clean_top1,
        "feature_decomposition": (
            "For each corruption Y and clean feature X, center_shift=mean(Y)-mean(X), "
            "shape_distortion=(Y-X)-center_shift. Center-only applies alpha*center_shift to every clean feature; "
            "shape-only applies alpha*shape_distortion per sample."
        ),
        "logit_decomposition": (
            "For CLIP logits, bias=mean(Y_logits-X_logits) and residual=(Y_logits-X_logits)-bias. "
            "This is the decomposition most directly aligned with CBC's class-wise logit centering."
        ),
        "cbc_aligned_decomposition": (
            "CBC removes the absolute batch mean logit of corrupted samples, up to a class-independent scalar. "
            "The CBC-aligned center-shift axis therefore injects centered mean(corrupted_logits) into clean predictions. "
            "The corresponding residual axis uses corrupted logits/features after this bias is removed."
        ),
    }
    write_json(output_root / "metadata.json", metadata)

    per_corruption_rows: List[Dict[str, object]] = []
    matched_budget_rows: List[Dict[str, object]] = []
    diagnostics_rows: List[Dict[str, object]] = []
    mu_clean = clean_features.mean(dim=0, keepdim=True)

    for corruption in corruptions:
        print(f"[center-vs-shape] {corruption}", flush=True)
        corrupt_dataset = load_dataset(corrupt_name, args.data_root, transform, corruption, int(args.severity))
        corrupt_loader = build_loader(corrupt_dataset, indices, args.batch_size, args.num_workers, device)
        corrupt_features, corrupt_labels = extract_features(model, corrupt_loader, device)
        if corrupt_labels.shape != clean_labels.shape or not torch.equal(corrupt_labels, clean_labels):
            raise ValueError(f"Label mismatch between clean and {corruption}.")

        corrupt_logits = corrupt_features.float() @ text_features.float().t()
        mu_corrupt = corrupt_features.mean(dim=0, keepdim=True)
        center_shift = mu_corrupt - mu_clean
        shape_distortion = (corrupt_features - clean_features) - center_shift

        logit_delta = corrupt_logits - clean_logits
        logit_bias = logit_delta.mean(dim=0, keepdim=True)
        logit_residual = logit_delta - logit_bias
        cbc_logit_bias = corrupt_logits.mean(dim=0, keepdim=True)
        cbc_logit_bias = cbc_logit_bias - cbc_logit_bias.mean(dim=1, keepdim=True)
        cbc_feature_shift = torch.linalg.lstsq(
            text_features.float(),
            cbc_logit_bias.squeeze(0).float().unsqueeze(1),
        ).solution.squeeze(1)
        cbc_feature_residual = (corrupt_features - cbc_feature_shift.view(1, -1)) - clean_features
        cbc_logit_residual = (corrupt_logits - cbc_logit_bias) - clean_logits

        cbc_feature_shift_unit = cbc_feature_shift / (cbc_feature_shift.norm() + 1.0e-12)
        cbc_feature_residual_mean_norm = torch.norm(cbc_feature_residual, dim=1).mean()
        cbc_feature_residual_unit = cbc_feature_residual / (cbc_feature_residual_mean_norm + 1.0e-12)
        class_centered_shape = class_centered_residual(corrupt_features - clean_features, clean_labels, len(class_names))
        class_centered_shape_mean_norm = torch.norm(class_centered_shape, dim=1).mean()
        class_centered_shape_unit = class_centered_shape / (class_centered_shape_mean_norm + 1.0e-12)
        permuted_shape = permute_within_class(class_centered_shape, clean_labels, args.seed + len(corruption))
        permuted_shape_mean_norm = torch.norm(permuted_shape, dim=1).mean()
        permuted_shape_unit = permuted_shape / (permuted_shape_mean_norm + 1.0e-12)
        random_shape = zero_mean_random_shape_like(clean_features, clean_labels, args.seed + 17 * len(corruption))
        random_shape_mean_norm = torch.norm(random_shape, dim=1).mean()
        random_shape_unit = random_shape / (random_shape_mean_norm + 1.0e-12)
        cbc_logit_bias_unit = cbc_logit_bias / (cbc_logit_bias.norm(dim=1, keepdim=True) + 1.0e-12)
        cbc_logit_residual_mean_norm = torch.norm(cbc_logit_residual, dim=1).mean()
        cbc_logit_residual_unit = cbc_logit_residual / (cbc_logit_residual_mean_norm + 1.0e-12)

        actual_corrupt_top1 = top1(corrupt_features, clean_labels, text_features)
        diagnostics_rows.append(
            {
                "corruption": corruption,
                "clean_top1": clean_top1,
                "actual_corrupt_top1": actual_corrupt_top1,
                "feature_center_shift_norm": float(center_shift.norm().item()),
                "feature_shape_distortion_mean_norm": float(torch.norm(shape_distortion, dim=1).mean().item()),
                "feature_shape_distortion_center_norm": float(shape_distortion.mean(dim=0).norm().item()),
                "logit_bias_norm": float(logit_bias.norm().item()),
                "logit_residual_mean_norm": float(torch.norm(logit_residual, dim=1).mean().item()),
                "logit_residual_center_norm": float(logit_residual.mean(dim=0).norm().item()),
                "cbc_logit_bias_norm": float(cbc_logit_bias.norm().item()),
                "cbc_feature_shift_norm": float(cbc_feature_shift.norm().item()),
                "cbc_feature_residual_mean_norm": float(torch.norm(cbc_feature_residual, dim=1).mean().item()),
                "class_centered_shape_mean_norm": float(class_centered_shape_mean_norm.item()),
                "class_centered_shape_center_norm": float(class_centered_shape.mean(dim=0).norm().item()),
                "permuted_shape_mean_norm": float(permuted_shape_mean_norm.item()),
                "random_shape_mean_norm": float(random_shape_mean_norm.item()),
                "cbc_logit_residual_mean_norm": float(cbc_logit_residual_mean_norm.item()),
            }
        )

        for alpha in args.alphas:
            alpha_float = float(alpha)
            feature_center = normalize_features(clean_features + alpha_float * center_shift)
            feature_shape = normalize_features(clean_features + alpha_float * shape_distortion)
            feature_combined = normalize_features(clean_features + alpha_float * (center_shift + shape_distortion))
            cbc_feature_center = normalize_features(clean_features + alpha_float * cbc_feature_shift.view(1, -1))
            cbc_feature_shape = normalize_features(clean_features + alpha_float * cbc_feature_residual)
            cbc_feature_combined = normalize_features(clean_features + alpha_float * (cbc_feature_shift.view(1, -1) + cbc_feature_residual))

            logit_bias_only = clean_logits + alpha_float * logit_bias
            logit_residual_only = clean_logits + alpha_float * logit_residual
            logit_combined = clean_logits + alpha_float * (logit_bias + logit_residual)
            cbc_logit_bias_only = clean_logits + alpha_float * cbc_logit_bias
            cbc_logit_residual_only = clean_logits + alpha_float * cbc_logit_residual
            cbc_logit_combined = clean_logits + alpha_float * (corrupt_logits - clean_logits)

            values = {
                "feature_center_shift": top1(feature_center, clean_labels, text_features),
                "feature_shape_distortion": top1(feature_shape, clean_labels, text_features),
                "feature_combined": top1(feature_combined, clean_labels, text_features),
                "cbc_feature_center_shift": top1(cbc_feature_center, clean_labels, text_features),
                "cbc_feature_shape_distortion": top1(cbc_feature_shape, clean_labels, text_features),
                "cbc_feature_combined": top1(cbc_feature_combined, clean_labels, text_features),
                "logit_bias": top1_logits(logit_bias_only, clean_labels),
                "logit_residual": top1_logits(logit_residual_only, clean_labels),
                "logit_combined": top1_logits(logit_combined, clean_labels),
                "cbc_logit_bias": top1_logits(cbc_logit_bias_only, clean_labels),
                "cbc_logit_residual": top1_logits(cbc_logit_residual_only, clean_labels),
                "cbc_logit_combined": top1_logits(cbc_logit_combined, clean_labels),
            }
            for axis, value in values.items():
                per_corruption_rows.append(
                    {
                        "corruption": corruption,
                        "axis": axis,
                        "alpha": alpha_float,
                        "top1": value,
                        "drop_from_clean": clean_top1 - value,
                    }
                )

        for budget in args.feature_budgets:
            budget_float = float(budget)
            matched_feature_center = normalize_features(clean_features + budget_float * cbc_feature_shift_unit.view(1, -1))
            matched_feature_shape = normalize_features(clean_features + budget_float * cbc_feature_residual_unit)
            matched_class_centered_shape = normalize_features(clean_features + budget_float * class_centered_shape_unit)
            matched_permuted_shape = normalize_features(clean_features + budget_float * permuted_shape_unit)
            matched_random_shape = normalize_features(clean_features + budget_float * random_shape_unit)
            for axis, value in {
                "matched_feature_center_shift": top1(matched_feature_center, clean_labels, text_features),
                "matched_feature_shape_distortion": top1(matched_feature_shape, clean_labels, text_features),
                "matched_feature_class_centered_shape": top1(matched_class_centered_shape, clean_labels, text_features),
                "matched_feature_permuted_shape": top1(matched_permuted_shape, clean_labels, text_features),
                "matched_feature_random_shape": top1(matched_random_shape, clean_labels, text_features),
            }.items():
                matched_budget_rows.append(
                    {
                        "corruption": corruption,
                        "axis": axis,
                        "budget": budget_float,
                        "top1": value,
                        "drop_from_clean": clean_top1 - value,
                    }
                )

        for budget in args.logit_budgets:
            budget_float = float(budget)
            matched_logit_bias = clean_logits + budget_float * cbc_logit_bias_unit
            matched_logit_residual = clean_logits + budget_float * cbc_logit_residual_unit
            for axis, value in {
                "matched_logit_bias": top1_logits(matched_logit_bias, clean_labels),
                "matched_logit_residual": top1_logits(matched_logit_residual, clean_labels),
            }.items():
                matched_budget_rows.append(
                    {
                        "corruption": corruption,
                        "axis": axis,
                        "budget": budget_float,
                        "top1": value,
                        "drop_from_clean": clean_top1 - value,
                    }
                )

    summary_rows = summarize_alpha(per_corruption_rows)
    budget_summary_rows = summarize_budget(matched_budget_rows)
    plot_summary(summary_rows, output_root, clean_top1)
    plot_budget_summary(budget_summary_rows, output_root, clean_top1)

    crossing_rows: List[Dict[str, object]] = []
    for drop in [1.0, 2.0, 5.0, 10.0, 20.0]:
        for axis in [
            "feature_center_shift",
            "feature_shape_distortion",
            "cbc_feature_center_shift",
            "cbc_feature_shape_distortion",
            "logit_bias",
            "logit_residual",
            "cbc_logit_bias",
            "cbc_logit_residual",
        ]:
            crossing_rows.append(
                {
                    "axis": axis,
                    "target_drop": drop,
                    "alpha_at_drop": find_drop_crossing(summary_rows, axis, clean_top1, drop),
                }
            )

    write_csv(output_root / "per_corruption_axis_sweep.csv", per_corruption_rows)
    write_csv(output_root / "axis_sweep_summary.csv", summary_rows)
    write_csv(output_root / "matched_budget_sweep.csv", matched_budget_rows)
    write_csv(output_root / "matched_budget_summary.csv", budget_summary_rows)
    write_csv(output_root / "corruption_decomposition_diagnostics.csv", diagnostics_rows)
    write_csv(output_root / "drop_crossings.csv", crossing_rows)

    readme = """# Feature Center Shift vs Shape Distortion

This experiment starts from clean CLIP image features and injects empirical CIFAR-10-C perturbations along two separated axes.

- Feature center shift: the global mean feature displacement from clean to corrupted images is applied to every clean feature.
- Feature shape distortion: the sample-wise corruption residual is applied after removing the global mean displacement.
- Logit bias/residual: the same decomposition is repeated in CLIP class-logit space, directly matching CBC's class-wise bias view.
- CBC-aligned bias/residual: the class-wise corrupted batch mean logit is used as the center-shift axis, because this is the bias term CBC explicitly removes.
- The mean curves aggregate the 15 CIFAR-10-C severity-5 corruptions.

The feature-space plot is the direct distribution-level analysis. The logit-space plot is the CBC-aligned evidence-space analysis.

Important interpretation: the full empirical corruption residual can have a larger total norm than the mean shift, so the full-residual curve may explain more of the final corrupted accuracy drop. The matched-budget plots control for perturbation magnitude and show the per-unit sensitivity of CLIP predictions to center shift versus shape distortion.
"""
    (output_root / "README.md").write_text(readme)
    print(f"[center-vs-shape] complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
