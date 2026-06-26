#!/usr/bin/env python3
"""Plot CBC-equivalent class-centroid shifts with text prototypes in PCA."""

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
from matplotlib.lines import Line2D
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
    / "cbc_class_centroid_shift_3d"
)
DEFAULT_FONT_ZIP = WORKSPACE_ROOT / "TimesNewerRoman.zip"
CLASS_COLORS = list(plt.cm.tab10.colors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=list(FAMILIES.keys()), default="cifar10")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--shift-types", nargs="+", default=["brightness", "glass_blur"])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-model-name", type=str, default="clip-ViT-B/16")
    parser.add_argument("--template", type=str, default=DEFAULT_TEMPLATE)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-class", type=int, default=200)
    parser.add_argument("--centroid-size", type=float, default=90.0)
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
            "axes.labelsize": 17,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )


def choose_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
        device = torch.device("cuda:0" if device_name == "cuda" else device_name)
        torch.cuda.set_device(device)
        return device
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_indices(dataset: Dataset, max_samples_per_class: int | None, seed: int) -> List[int]:
    labels = getattr(dataset, "labels", None)
    if labels is None:
        labels = getattr(dataset, "targets", None)
    if labels is None:
        raise AttributeError("Dataset does not expose labels/targets.")
    labels_np = np.asarray(labels, dtype=np.int64)
    all_indices = np.arange(labels_np.shape[0], dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    for class_id in sorted(np.unique(labels_np).astype(int).tolist()):
        class_indices = all_indices[labels_np == class_id]
        if max_samples_per_class is None or class_indices.size <= max_samples_per_class:
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


def visual_model(model: torch.nn.Module) -> torch.nn.Module:
    visual = model.visual if hasattr(model, "visual") else model
    if not hasattr(visual, "use_local"):
        visual.use_local = False
    return visual


@torch.no_grad()
def extract_features(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[Tensor, Tensor]:
    model.eval()
    visual = visual_model(model)
    dtype = getattr(visual, "dtype", torch.float32)
    features: List[Tensor] = []
    labels: List[Tensor] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
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


def class_centroids(features: Tensor, labels: Tensor, num_classes: int) -> Tensor:
    centroids: List[Tensor] = []
    for class_id in range(num_classes):
        mask = labels == class_id
        if mask.any():
            centroids.append(features[mask].mean(dim=0))
        else:
            centroids.append(torch.full_like(features[0], float("nan")))
    return torch.stack(centroids, dim=0)


def cbc_equivalent_shift(features: Tensor, text_features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    logits = features.float() @ text_features.float().t()
    logit_bias = logits.mean(dim=0)
    # Solve text_features @ delta ~= logit_bias, i.e. (features - delta) @ text.T
    # has logits shifted by -logit_bias. This is the minimum-norm feature-space
    # displacement equivalent to CBC in the text-prototype subspace.
    delta = torch.linalg.lstsq(text_features.float(), logit_bias.unsqueeze(1)).solution.squeeze(1)
    residual = text_features.float() @ delta - logit_bias
    shifted = features.float() - delta.view(1, -1)
    return shifted, delta, residual


def fit_pca_3d(arrays: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate(arrays, axis=0).astype(np.float64)
    mean = stacked.mean(axis=0, keepdims=True)
    centered = stacked - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:3].T
    return mean, basis


def fit_pca_2d(arrays: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate(arrays, axis=0).astype(np.float64)
    mean = stacked.mean(axis=0, keepdims=True)
    centered = stacked - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    return mean, basis


def project(features: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (features.astype(np.float64) - mean) @ basis


def set_3d_limits(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) * 0.55, 1.0e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def style_axis(ax, elev: float, azim: float) -> None:
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("PC1", labelpad=9)
    ax.set_ylabel("PC2", labelpad=9)
    ax.set_zlabel("PC3", labelpad=9)
    ax.grid(alpha=0.22, linestyle="--", linewidth=0.6)
    ax.xaxis.pane.set_alpha(0.03)
    ax.yaxis.pane.set_alpha(0.03)
    ax.zaxis.pane.set_alpha(0.03)


def draw_arrow(ax, start: np.ndarray, end: np.ndarray, color: str) -> None:
    delta = end - start
    ax.quiver(
        start[0],
        start[1],
        start[2],
        delta[0],
        delta[1],
        delta[2],
        color=color,
        alpha=0.72,
        linewidth=1.1,
        arrow_length_ratio=0.16,
        normalize=False,
    )


def draw_arrow_2d(ax, start: np.ndarray, end: np.ndarray, color: str) -> None:
    delta = end - start
    ax.arrow(
        start[0],
        start[1],
        delta[0],
        delta[1],
        color=color,
        alpha=0.72,
        linewidth=1.05,
        length_includes_head=True,
        head_width=0.045,
        head_length=0.07,
        overhang=0.18,
    )


def set_2d_limits(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) * 0.58, 1.0e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)


def style_axis_2d(ax) -> None:
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.22, linestyle="--", linewidth=0.6)
    ax.set_aspect("equal", adjustable="box")


def plot_shift(
    *,
    output_dir: Path,
    corruption: str,
    class_names: Sequence[str],
    raw_centroids: np.ndarray,
    cbc_centroids: np.ndarray,
    text_features: np.ndarray,
    sample_points: np.ndarray,
    elev: float,
    azim: float,
    centroid_size: float,
) -> None:
    pca_mean, pca_basis = fit_pca_3d([sample_points, raw_centroids, cbc_centroids, text_features])
    raw_3d = project(raw_centroids, pca_mean, pca_basis)
    cbc_3d = project(cbc_centroids, pca_mean, pca_basis)
    text_3d = project(text_features, pca_mean, pca_basis)
    sample_3d = project(sample_points, pca_mean, pca_basis)
    all_points = np.concatenate([sample_3d, raw_3d, cbc_3d, text_3d], axis=0)

    fig = plt.figure(figsize=(9.0, 7.4))
    ax = fig.add_subplot(111, projection="3d")
    for class_id, class_name in enumerate(class_names):
        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
        ax.scatter(
            raw_3d[class_id, 0],
            raw_3d[class_id, 1],
            raw_3d[class_id, 2],
            marker="o",
            s=centroid_size,
            color=color,
            alpha=0.55,
            edgecolor="black",
            linewidth=0.35,
        )
        ax.scatter(
            cbc_3d[class_id, 0],
            cbc_3d[class_id, 1],
            cbc_3d[class_id, 2],
            marker="D",
            s=centroid_size * 0.88,
            color=color,
            alpha=0.95,
            edgecolor="black",
            linewidth=0.45,
        )
        ax.scatter(
            text_3d[class_id, 0],
            text_3d[class_id, 1],
            text_3d[class_id, 2],
            marker="*",
            s=centroid_size * 1.35,
            color=color,
            alpha=1.0,
            edgecolor="black",
            linewidth=0.45,
        )
        draw_arrow(ax, raw_3d[class_id], cbc_3d[class_id], color)
        ax.text(
            text_3d[class_id, 0],
            text_3d[class_id, 1],
            text_3d[class_id, 2],
            class_name.replace("_", " "),
            fontsize=8,
        )

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=8, label="Corrupted class centroid"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=8, label="After CBC-equivalent shift"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=11, label="Text prototype"),
    ]
    ax.legend(handles=handles, frameon=True, facecolor="white", edgecolor="0.7", loc="best")
    set_3d_limits(ax, all_points)
    style_axis(ax, elev=elev, azim=azim)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"cbc_class_centroid_shift_3d_{corruption}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    pca_mean_2d, pca_basis_2d = fit_pca_2d([sample_points, raw_centroids, cbc_centroids, text_features])
    raw_2d = project(raw_centroids, pca_mean_2d, pca_basis_2d)
    cbc_2d = project(cbc_centroids, pca_mean_2d, pca_basis_2d)
    text_2d = project(text_features, pca_mean_2d, pca_basis_2d)
    sample_2d = project(sample_points, pca_mean_2d, pca_basis_2d)
    all_points_2d = np.concatenate([sample_2d, raw_2d, cbc_2d, text_2d], axis=0)

    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    for class_id, class_name in enumerate(class_names):
        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
        ax.scatter(
            raw_2d[class_id, 0],
            raw_2d[class_id, 1],
            marker="o",
            s=centroid_size,
            color=color,
            alpha=0.55,
            edgecolor="black",
            linewidth=0.35,
            zorder=3,
        )
        ax.scatter(
            cbc_2d[class_id, 0],
            cbc_2d[class_id, 1],
            marker="D",
            s=centroid_size * 0.88,
            color=color,
            alpha=0.95,
            edgecolor="black",
            linewidth=0.45,
            zorder=4,
        )
        ax.scatter(
            text_2d[class_id, 0],
            text_2d[class_id, 1],
            marker="*",
            s=centroid_size * 1.35,
            color=color,
            alpha=1.0,
            edgecolor="black",
            linewidth=0.45,
            zorder=5,
        )
        draw_arrow_2d(ax, raw_2d[class_id], cbc_2d[class_id], color)
        ax.text(
            text_2d[class_id, 0],
            text_2d[class_id, 1],
            class_name.replace("_", " "),
            fontsize=9,
            zorder=6,
        )

    ax.legend(handles=handles, frameon=True, facecolor="white", edgecolor="0.7", loc="best")
    set_2d_limits(ax, all_points_2d)
    style_axis_2d(ax)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"cbc_class_centroid_shift_2d_{corruption}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    rows_2d: List[Dict[str, object]] = []
    for class_id, class_name in enumerate(class_names):
        rows_2d.append(
            {
                "corruption": corruption,
                "class_id": class_id,
                "class_name": class_name,
                "raw_pc1": float(raw_2d[class_id, 0]),
                "raw_pc2": float(raw_2d[class_id, 1]),
                "cbc_pc1": float(cbc_2d[class_id, 0]),
                "cbc_pc2": float(cbc_2d[class_id, 1]),
                "text_pc1": float(text_2d[class_id, 0]),
                "text_pc2": float(text_2d[class_id, 1]),
                "raw_to_cbc_pca_l2": float(np.linalg.norm(cbc_2d[class_id] - raw_2d[class_id])),
                "cbc_to_text_pca_l2": float(np.linalg.norm(text_2d[class_id] - cbc_2d[class_id])),
                "raw_to_text_pca_l2": float(np.linalg.norm(text_2d[class_id] - raw_2d[class_id])),
            }
        )
    write_csv(output_dir / f"cbc_class_centroid_shift_2d_{corruption}.csv", rows_2d)

    rows: List[Dict[str, object]] = []
    for class_id, class_name in enumerate(class_names):
        rows.append(
            {
                "corruption": corruption,
                "class_id": class_id,
                "class_name": class_name,
                "raw_pc1": float(raw_3d[class_id, 0]),
                "raw_pc2": float(raw_3d[class_id, 1]),
                "raw_pc3": float(raw_3d[class_id, 2]),
                "cbc_pc1": float(cbc_3d[class_id, 0]),
                "cbc_pc2": float(cbc_3d[class_id, 1]),
                "cbc_pc3": float(cbc_3d[class_id, 2]),
                "text_pc1": float(text_3d[class_id, 0]),
                "text_pc2": float(text_3d[class_id, 1]),
                "text_pc3": float(text_3d[class_id, 2]),
                "raw_to_cbc_pca_l2": float(np.linalg.norm(cbc_3d[class_id] - raw_3d[class_id])),
                "cbc_to_text_pca_l2": float(np.linalg.norm(text_3d[class_id] - cbc_3d[class_id])),
                "raw_to_text_pca_l2": float(np.linalg.norm(text_3d[class_id] - raw_3d[class_id])),
            }
        )
    write_csv(output_dir / f"cbc_class_centroid_shift_3d_{corruption}.csv", rows)


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

    model, transform = return_base_model(args.base_model_name, device, dataset=corrupt_name)
    model.eval()
    clean_dataset = load_dataset(clean_name, args.data_root, transform, None, None)
    class_names = list(clean_dataset.class_names)
    text_features = encode_text_features(model, class_names, args.template, device)

    metadata = {
        "family": args.family,
        "severity": int(args.severity),
        "shift_types": list(args.shift_types),
        "base_model_name": args.base_model_name,
        "template": args.template,
        "batch_size": int(args.batch_size),
        "max_samples_per_class": args.max_samples_per_class,
        "seed": int(args.seed),
        "interpretation": (
            "CBC is a logit-space operation. For visualization, the class centroids are shifted "
            "by the minimum-norm feature displacement delta satisfying text_features @ delta ~= mean_logits."
        ),
    }
    write_json(output_root / "metadata.json", metadata)

    summary_rows: List[Dict[str, object]] = []
    for corruption in args.shift_types:
        print(f"[cbc-centroid] {corruption}", flush=True)
        dataset = load_dataset(corrupt_name, args.data_root, transform, corruption, int(args.severity))
        indices = stratified_indices(dataset, args.max_samples_per_class, args.seed)
        loader = build_loader(dataset, indices, args.batch_size, args.num_workers, device)
        features, labels = extract_features(model, loader, device)
        shifted_features, delta, residual = cbc_equivalent_shift(features, text_features)

        raw_centroids = class_centroids(features, labels, len(class_names))
        cbc_centroids = class_centroids(shifted_features, labels, len(class_names))

        corruption_dir = output_root / corruption
        ensure_dir(corruption_dir)
        plot_shift(
            output_dir=corruption_dir,
            corruption=corruption,
            class_names=class_names,
            raw_centroids=raw_centroids.numpy(),
            cbc_centroids=cbc_centroids.numpy(),
            text_features=text_features.numpy(),
            sample_points=torch.cat([features, shifted_features], dim=0).numpy(),
            elev=args.elev,
            azim=args.azim,
            centroid_size=args.centroid_size,
        )

        summary = {
            "corruption": corruption,
            "num_samples": int(features.shape[0]),
            "delta_norm": float(delta.norm().item()),
            "logit_bias_l2": float((features @ text_features.t()).mean(dim=0).norm().item()),
            "least_squares_residual_l2": float(residual.norm().item()),
            "output_png": str(corruption_dir / f"cbc_class_centroid_shift_3d_{corruption}.png"),
            "output_pdf": str(corruption_dir / f"cbc_class_centroid_shift_3d_{corruption}.pdf"),
            "output_2d_png": str(corruption_dir / f"cbc_class_centroid_shift_2d_{corruption}.png"),
            "output_2d_pdf": str(corruption_dir / f"cbc_class_centroid_shift_2d_{corruption}.pdf"),
            "centroid_csv": str(corruption_dir / f"cbc_class_centroid_shift_3d_{corruption}.csv"),
            "centroid_2d_csv": str(corruption_dir / f"cbc_class_centroid_shift_2d_{corruption}.csv"),
        }
        write_json(corruption_dir / "summary.json", summary)
        summary_rows.append(summary)

    write_csv(output_root / "summary.csv", summary_rows)
    readme = """# CBC Class-Centroid Shift in PCA

- Circles: corrupted image feature class centroids.
- Diamonds: class centroids after CBC-equivalent feature shift.
- Stars: class text prototypes from the default prompt.
- Arrows: per-class centroid movement induced by the shared CBC-equivalent bias.
- CBC itself is logit-space centering; this figure uses the minimum-norm feature displacement whose projection onto text prototypes reproduces the CBC logit shift.
- Each corruption folder includes both 3D and 2D PCA versions.
"""
    (output_root / "README.md").write_text(readme)
    print(f"[cbc-centroid] complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
