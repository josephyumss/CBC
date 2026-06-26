#!/usr/bin/env python3
"""Visualize offline CBC global-centroid bias removal for selected corruptions."""

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
from mpl_toolkits.mplot3d import proj3d
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from run_closed_set_cifar_benchmarks import DEFAULT_DATA_ROOT, FAMILIES  # noqa: E402
from ttavlm.datasets import return_train_val_datasets  # noqa: E402
from ttavlm.models import return_base_model  # noqa: E402
from ttavlm.models.clip import tokenize  # noqa: E402


DEFAULT_TEMPLATE = "a photo of a {}"
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "complete"
    / "final_paper_experiments"
    / "05_qualitative_analysis"
    / "cbc_global_centroid_bias"
)
DEFAULT_FONT_ZIP = WORKSPACE_ROOT / "TimesNewerRoman.zip"
CLASS_COLORS = list(plt.cm.tab10.colors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=list(FAMILIES.keys()), default="cifar10")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--corruptions", nargs="+", default=["brightness", "shot_noise"])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-model-name", type=str, default="clip-ViT-B/16")
    parser.add_argument("--template", type=str, default=DEFAULT_TEMPLATE)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--elev", type=float, default=22.0)
    parser.add_argument("--azim", type=float, default=-58.0)
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
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    ensure_dir(path.parent)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
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
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
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


def build_loader(dataset: Dataset, batch_size: int, workers: int, device: torch.device) -> DataLoader:
    indices = list(range(len(dataset)))
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


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


def offline_cbc_equivalent_shift(features: Tensor, text_features: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    logits = features.float() @ text_features.float().t()
    mean_logits = logits.mean(dim=0)
    delta = torch.linalg.lstsq(text_features.float(), mean_logits.unsqueeze(1)).solution.squeeze(1)
    residual = text_features.float() @ delta - mean_logits
    shifted = features.float() - delta.view(1, -1)
    return shifted, delta, mean_logits, residual


def fit_pca_3d(arrays: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate(arrays, axis=0).astype(np.float64)
    mean = stacked.mean(axis=0, keepdims=True)
    centered = stacked - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:3].T
    return mean, basis


def project(features: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (features.astype(np.float64) - mean) @ basis


def set_3d_limits(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) * 0.60, 1.0e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def style_axis(ax, elev: float, azim: float) -> None:
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("PC1", labelpad=11, fontsize=20)
    ax.set_ylabel("PC2", labelpad=11, fontsize=20)
    ax.set_zlabel("PC3", labelpad=14, fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=14, pad=2)
    ax.grid(alpha=0.20, linestyle="--", linewidth=0.6)
    ax.xaxis.pane.set_alpha(0.03)
    ax.yaxis.pane.set_alpha(0.03)
    ax.zaxis.pane.set_alpha(0.03)


def draw_arrow(ax, start: np.ndarray, end: np.ndarray) -> None:
    delta = end - start
    ax.quiver(
        start[0],
        start[1],
        start[2],
        delta[0],
        delta[1],
        delta[2],
        color="#2ca02c",
        alpha=0.82,
        linewidth=1.4,
        arrow_length_ratio=0.18,
        normalize=False,
    )


def annotate_projected_point(ax, point: np.ndarray, label: str, color: str, y_offset: float, va: str) -> None:
    x_coord, y_coord, _ = proj3d.proj_transform(point[0], point[1], point[2], ax.get_proj())
    ax.annotate(
        label,
        xy=(x_coord, y_coord),
        xytext=(0, y_offset),
        textcoords="offset points",
        ha="center",
        va=va,
        fontsize=16,
        color=color,
    )


def plot_global_centroid_pca(
    *,
    output_dir: Path,
    corruption: str,
    features: Tensor,
    shifted_features: Tensor,
    text_features: Tensor,
    class_names: Sequence[str],
    elev: float,
    azim: float,
) -> None:
    raw_centroid = features.mean(dim=0, keepdim=True)
    cbc_centroid = shifted_features.mean(dim=0, keepdim=True)
    text_centroid = text_features.mean(dim=0, keepdim=True)
    pca_mean, pca_basis = fit_pca_3d(
        [
            features.numpy(),
            shifted_features.numpy(),
            raw_centroid.numpy(),
            cbc_centroid.numpy(),
            text_centroid.numpy(),
        ]
    )
    raw_3d = project(raw_centroid.numpy(), pca_mean, pca_basis)[0]
    cbc_3d = project(cbc_centroid.numpy(), pca_mean, pca_basis)[0]
    text_3d = project(text_centroid.numpy(), pca_mean, pca_basis)[0]
    feature_sample_3d = project(torch.cat([features, shifted_features], dim=0).numpy(), pca_mean, pca_basis)
    all_points = np.concatenate([feature_sample_3d, raw_3d[None, :], cbc_3d[None, :], text_3d[None, :]], axis=0)

    fig = plt.figure(figsize=(9.8, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*raw_3d, marker="o", s=200, color="#d62728", edgecolor="black", linewidth=0.75)
    ax.scatter(*cbc_3d, marker="D", s=180, color="#2ca02c", edgecolor="black", linewidth=0.75)
    ax.scatter(*text_3d, marker="*", s=250, color="#1f77b4", edgecolor="black", linewidth=0.75)
    draw_arrow(ax, raw_3d, cbc_3d)
    set_3d_limits(ax, all_points)
    style_axis(ax, elev=elev, azim=azim)
    annotate_projected_point(ax, raw_3d, "Corrupted Centroid", "#9b1d20", y_offset=-26, va="top")
    annotate_projected_point(ax, cbc_3d, "CBC-driven Centroid", "#1f7a3a", y_offset=26, va="bottom")
    annotate_projected_point(ax, text_3d, "Text Centroid", "#1f4e8c", y_offset=26, va="bottom")
    fig.subplots_adjust(left=0.02, right=0.70, bottom=0.04, top=0.98)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{corruption}_global_centroid_text_pca3d.{suffix}", dpi=300)
    plt.close(fig)


def plot_centroid_text_bars(
    *,
    output_dir: Path,
    corruption: str,
    raw_similarity: Tensor,
    cbc_similarity: Tensor,
    class_names: Sequence[str],
) -> None:
    x = np.arange(len(class_names))
    width = 0.38
    raw = torch.softmax(raw_similarity.float(), dim=0).numpy()
    cbc = torch.softmax(cbc_similarity.float(), dim=0).numpy()
    uniform_prob = 1.0 / len(class_names)
    min_prob = float(min(raw.min(), cbc.min(), uniform_prob))
    max_prob = float(max(raw.max(), cbc.max(), uniform_prob))
    margin = max((max_prob - min_prob) * 0.20, 2.0e-4)
    y_min = max(0.0, min_prob - margin)
    y_max = min(1.0, max_prob + margin)

    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    ax.bar(x - width / 2, raw, width=width, color="#d95f02", alpha=0.88, label="Corrupted Centroid")
    ax.bar(x + width / 2, cbc, width=width, color="#1b9e77", alpha=0.88, label="CBC Centroid")
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace("_", " ") for name in class_names], rotation=35, ha="right")
    ax.set_ylabel("Centroid-text probability")
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.7)
    ax.legend(frameon=True, facecolor="white", edgecolor="0.75", loc="best")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{corruption}_centroid_text_probabilities_bar.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    configure_font(args.font_zip)
    device = choose_device(args.device)
    output_root = args.output_root
    ensure_dir(output_root)

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
    text_features = encode_text_features(model, class_names, args.template, device)

    metadata = {
        "family": args.family,
        "clean_dataset": clean_name,
        "corrupted_dataset": corrupt_name,
        "severity": int(args.severity),
        "corruptions": list(args.corruptions),
        "base_model_name": args.base_model_name,
        "template": args.template,
        "cbc_setting": "offline/global: all test samples in one corruption are used to estimate and subtract the CBC class-logit bias.",
        "interpretation": (
            "CBC is applied in logit space as logits - mean_logits over all test samples. "
            "For feature-space visualization, we use the minimum-norm feature displacement "
            "whose projection onto text prototypes reproduces the centered class-logit bias."
        ),
    }
    write_json(output_root / "metadata.json", metadata)

    all_summary: List[Dict[str, object]] = []
    all_bar_rows: List[Dict[str, object]] = []
    for corruption in args.corruptions:
        print(f"[cbc-global-centroid] {corruption}", flush=True)
        corruption_dir = output_root / corruption
        ensure_dir(corruption_dir)

        dataset = load_dataset(corrupt_name, args.data_root, transform, corruption, int(args.severity))
        loader = build_loader(dataset, args.batch_size, args.num_workers, device)
        features, labels = extract_features(model, loader, device)
        shifted_features, delta, mean_logits, residual = offline_cbc_equivalent_shift(features, text_features)

        raw_centroid = features.mean(dim=0)
        cbc_centroid = shifted_features.mean(dim=0)
        raw_similarity = raw_centroid @ text_features.t()
        cbc_similarity = cbc_centroid @ text_features.t()
        raw_probability = torch.softmax(raw_similarity.float(), dim=0)
        cbc_probability = torch.softmax(cbc_similarity.float(), dim=0)

        plot_global_centroid_pca(
            output_dir=corruption_dir,
            corruption=corruption,
            features=features,
            shifted_features=shifted_features,
            text_features=text_features,
            class_names=class_names,
            elev=args.elev,
            azim=args.azim,
        )
        plot_centroid_text_bars(
            output_dir=corruption_dir,
            corruption=corruption,
            raw_similarity=raw_similarity,
            cbc_similarity=cbc_similarity,
            class_names=class_names,
        )

        bar_rows: List[Dict[str, object]] = []
        for class_id, class_name in enumerate(class_names):
            row = {
                "corruption": corruption,
                "class_id": class_id,
                "class_name": class_name,
                "corrupted_centroid_text_logit": float(raw_similarity[class_id].item()),
                "cbc_centroid_text_logit": float(cbc_similarity[class_id].item()),
                "corrupted_centroid_text_probability": float(raw_probability[class_id].item()),
                "cbc_centroid_text_probability": float(cbc_probability[class_id].item()),
                "mean_logit_removed": float(mean_logits[class_id].item()),
            }
            bar_rows.append(row)
            all_bar_rows.append(row)
        write_csv(corruption_dir / f"{corruption}_centroid_text_logits.csv", bar_rows)

        summary = {
            "corruption": corruption,
            "num_samples": int(features.shape[0]),
            "raw_logit_std": float(raw_similarity.std(unbiased=False).item()),
            "cbc_logit_std": float(cbc_similarity.std(unbiased=False).item()),
            "raw_logit_range": float((raw_similarity.max() - raw_similarity.min()).item()),
            "cbc_logit_range": float((cbc_similarity.max() - cbc_similarity.min()).item()),
            "cbc_logit_mean": float(cbc_similarity.mean().item()),
            "raw_probability_std": float(raw_probability.std(unbiased=False).item()),
            "cbc_probability_std": float(cbc_probability.std(unbiased=False).item()),
            "raw_probability_range": float((raw_probability.max() - raw_probability.min()).item()),
            "cbc_probability_range": float((cbc_probability.max() - cbc_probability.min()).item()),
            "least_squares_residual_l2": float(residual.norm().item()),
            "delta_norm": float(delta.norm().item()),
            "pca_png": str(corruption_dir / f"{corruption}_global_centroid_text_pca3d.png"),
            "bar_png": str(corruption_dir / f"{corruption}_centroid_text_probabilities_bar.png"),
            "bar_csv": str(corruption_dir / f"{corruption}_centroid_text_logits.csv"),
        }
        write_json(corruption_dir / "summary.json", summary)
        all_summary.append(summary)

    write_csv(output_root / "summary.csv", all_summary)
    write_csv(output_root / "centroid_text_logits_all.csv", all_bar_rows)
    readme = """# Offline CBC Global Centroid Bias Visualization

This folder contains qualitative visualizations for brightness and shot_noise.

- `*_global_centroid_text_pca3d.png`: corrupted global feature centroid, CBC-driven centroid, and the text centroid in one PCA-3D space.
- `*_centroid_text_probabilities_bar.png`: softmax probability distribution over text prototypes before/after CBC.
- Offline CBC estimates one class-logit bias using all test samples from the corruption and subtracts it from every sample.
- After CBC, the global centroid has nearly equal logits against all text prototypes; after softmax this appears as an almost uniform green probability bar around 0.1.
"""
    (output_root / "README.md").write_text(readme)
    print(f"[cbc-global-centroid] complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
