#!/usr/bin/env python3
"""Measure shape-only feature distribution similarity between clean and corrupted CIFAR-C."""

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
from matplotlib import colors as mpl_colors
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from run_closed_set_cifar_benchmarks import DEFAULT_DATA_ROOT, FAMILIES  # noqa: E402
from ttavlm.datasets import CORRUPTIONS, return_train_val_datasets  # noqa: E402
from ttavlm.models import return_base_model  # noqa: E402


DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "complete"
    / "final_paper_experiments"
    / "05_qualitative_analysis"
    / "feature_distribution_similarity"
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
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
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


def center_scale(features: Tensor) -> Tensor:
    centered = features.float() - features.float().mean(dim=0, keepdim=True)
    rms = torch.sqrt(torch.sum(centered * centered, dim=1).mean()).clamp_min(1.0e-12)
    return centered / rms


def class_residuals(features: Tensor, labels: Tensor, num_classes: int) -> Tensor:
    residuals = features.float().clone()
    for class_id in range(num_classes):
        mask = labels == class_id
        if mask.any():
            residuals[mask] -= features.float()[mask].mean(dim=0, keepdim=True)
    rms = torch.sqrt(torch.sum(residuals * residuals, dim=1).mean()).clamp_min(1.0e-12)
    return residuals / rms


def class_centroids(features: Tensor, labels: Tensor, num_classes: int) -> Tensor:
    centroids: List[Tensor] = []
    for class_id in range(num_classes):
        mask = labels == class_id
        if mask.any():
            centroids.append(features.float()[mask].mean(dim=0))
        else:
            centroids.append(torch.full_like(features[0].float(), float("nan")))
    return torch.stack(centroids, dim=0)


def normalized_centroid_geometry(features: Tensor, labels: Tensor, num_classes: int) -> Tensor:
    centroids = class_centroids(features, labels, num_classes)
    centered = centroids - centroids.mean(dim=0, keepdim=True)
    return centered / centered.norm().clamp_min(1.0e-12)


def covariance_matrix(features: Tensor) -> Tensor:
    features = features.float()
    return features.t() @ features / max(int(features.shape[0]) - 1, 1)


def frobenius_cosine(a: Tensor, b: Tensor) -> float:
    numerator = torch.sum(a.float() * b.float())
    denominator = torch.norm(a.float()).clamp_min(1.0e-12) * torch.norm(b.float()).clamp_min(1.0e-12)
    return float((numerator / denominator).item())


def feature_top1(features: Tensor, labels: Tensor, text_features: Tensor) -> float:
    logits = features.float() @ text_features.float().t()
    return 100.0 * float(logits.argmax(dim=1).eq(labels).float().mean().item())


def compute_similarity(clean_features: Tensor, corrupt_features: Tensor, labels: Tensor, num_classes: int) -> Dict[str, float]:
    clean_shape = center_scale(clean_features)
    corrupt_shape = center_scale(corrupt_features)
    clean_within = class_residuals(clean_features, labels, num_classes)
    corrupt_within = class_residuals(corrupt_features, labels, num_classes)
    clean_geometry = normalized_centroid_geometry(clean_features, labels, num_classes)
    corrupt_geometry = normalized_centroid_geometry(corrupt_features, labels, num_classes)

    clean_cov = covariance_matrix(clean_shape)
    corrupt_cov = covariance_matrix(corrupt_shape)
    clean_within_cov = covariance_matrix(clean_within)
    corrupt_within_cov = covariance_matrix(corrupt_within)

    global_covariance_similarity = frobenius_cosine(clean_cov, corrupt_cov)
    within_class_covariance_similarity = frobenius_cosine(clean_within_cov, corrupt_within_cov)
    class_centroid_geometry_similarity = frobenius_cosine(clean_geometry, corrupt_geometry)

    return {
        "global_covariance_similarity": global_covariance_similarity,
        "global_covariance_distance": 1.0 - global_covariance_similarity,
        "within_class_covariance_similarity": within_class_covariance_similarity,
        "within_class_covariance_distance": 1.0 - within_class_covariance_similarity,
        "class_centroid_geometry_similarity": class_centroid_geometry_similarity,
        "class_centroid_geometry_distance": 1.0 - class_centroid_geometry_similarity,
        "mean_center_l2": float((corrupt_features.mean(dim=0) - clean_features.mean(dim=0)).norm().item()),
    }


def plot_similarity(rows: Sequence[Mapping[str, object]], output_root: Path) -> None:
    ordered = sorted(rows, key=lambda row: float(row["global_covariance_similarity"]))
    names = [str(row["corruption"]).replace("_", " ") for row in ordered]
    values = np.asarray([float(row["global_covariance_similarity"]) for row in ordered], dtype=np.float64)
    sunset = mpl_colors.LinearSegmentedColormap.from_list(
        "cbc_sunset",
        ["#1b1b3a", "#3b1f5f", "#7a2e6d", "#c44e52", "#f28e2b", "#ffd166"],
    )
    norm = mpl_colors.Normalize(vmin=float(values.min()), vmax=float(values.max()))
    bar_colors = [sunset(norm(value)) for value in values]

    fig, ax = plt.subplots(figsize=(8.8, 5.9))
    bars = ax.barh(names, values, color=bar_colors, alpha=0.96, edgecolor="white", linewidth=0.7)
    ax.axvline(float(np.mean(values)), color="black", linestyle=":", linewidth=1.4, label="Mean")
    ax.set_xlabel("Shape similarity to clean distribution")
    ax.set_xlim(max(0.0, float(values.min()) - 0.035), min(1.0, float(values.max()) + 0.025))
    ax.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.7)
    ax.legend(frameon=True, facecolor="white", edgecolor="0.75", loc="lower right")
    for bar, value in zip(bars, values):
        threshold = float(values.min() + 0.55 * (values.max() - values.min()))
        text_color = "#1f1f1f" if value > threshold else "white"
        ax.text(
            value - 0.006,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.3f}",
            va="center",
            ha="right",
            fontsize=10,
            color=text_color,
            fontweight="bold",
        )
    scalar_map = plt.cm.ScalarMappable(cmap=sunset, norm=norm)
    scalar_map.set_array([])
    colorbar = fig.colorbar(scalar_map, ax=ax, fraction=0.045, pad=0.025)
    colorbar.set_label("Similarity")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_root / f"global_distribution_shape_similarity.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_heatmap(rows: Sequence[Mapping[str, object]], output_root: Path) -> None:
    metrics = [
        ("global_covariance_similarity", "Global covariance"),
        ("within_class_covariance_similarity", "Within-class covariance"),
        ("class_centroid_geometry_similarity", "Class-centroid geometry"),
    ]
    ordered = sorted(rows, key=lambda row: float(row["global_covariance_similarity"]))
    values = np.asarray([[float(row[key]) for key, _ in metrics] for row in ordered], dtype=np.float64)
    names = [str(row["corruption"]).replace("_", " ") for row in ordered]

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(values, cmap="viridis", vmin=max(0.0, values.min() - 0.02), vmax=1.0, aspect="auto")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([label for _, label in metrics], rotation=25, ha="right")
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            ax.text(col_idx, row_idx, f"{values[row_idx, col_idx]:.2f}", ha="center", va="center", color="white" if values[row_idx, col_idx] < 0.82 else "black", fontsize=9)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Similarity")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_root / f"distribution_shape_similarity_metrics_heatmap.{suffix}", dpi=300, bbox_inches="tight")
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
    num_classes = len(class_names)
    indices = stratified_indices(clean_dataset, args.max_samples_per_class, args.seed)
    clean_loader = build_loader(clean_dataset, indices, args.batch_size, args.num_workers, device)
    clean_features, clean_labels = extract_features(model, clean_loader, device)

    metadata = {
        "family": args.family,
        "clean_dataset": clean_name,
        "corrupted_dataset": corrupt_name,
        "severity": int(args.severity),
        "corruptions": corruptions,
        "base_model_name": args.base_model_name,
        "num_samples": int(clean_features.shape[0]),
        "max_samples_per_class": args.max_samples_per_class,
        "shape_normalization": (
            "Features are already CLIP L2-normalized. For shape-only comparison, "
            "we subtract the distribution mean and divide by RMS radius before computing covariance similarity."
        ),
        "main_metric": "global_covariance_similarity",
    }
    write_json(output_root / "metadata.json", metadata)

    rows: List[Dict[str, object]] = []
    for corruption in corruptions:
        print(f"[distribution-similarity] {corruption}", flush=True)
        corrupt_dataset = load_dataset(corrupt_name, args.data_root, transform, corruption, int(args.severity))
        corrupt_loader = build_loader(corrupt_dataset, indices, args.batch_size, args.num_workers, device)
        corrupt_features, corrupt_labels = extract_features(model, corrupt_loader, device)
        if corrupt_labels.shape != clean_labels.shape or not torch.equal(corrupt_labels, clean_labels):
            raise ValueError(f"Label mismatch between clean and {corruption}.")
        metrics = compute_similarity(clean_features, corrupt_features, clean_labels, num_classes)
        rows.append(
            {
                "corruption": corruption,
                "severity": int(args.severity),
                **metrics,
            }
        )

    write_csv(output_root / "distribution_shape_similarity_by_corruption.csv", rows)
    plot_similarity(rows, output_root)
    plot_metric_heatmap(rows, output_root)

    readme = """# Clean vs Corrupted Feature Distribution Shape Similarity

This analysis compares CLIP image-feature distributions between clean CIFAR-10 and each CIFAR-10-C corruption at severity 5.

- Features are first CLIP-normalized by the model.
- To compare shape rather than location, we subtract each distribution mean and divide by its RMS radius.
- The main metric is covariance Frobenius cosine similarity after this center/scale normalization.
- Additional diagnostics compare within-class covariance and class-centroid geometry.

Use `global_distribution_shape_similarity.png` as the main corruption-wise visualization.
"""
    (output_root / "README.md").write_text(readme)
    print(f"[distribution-similarity] complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
