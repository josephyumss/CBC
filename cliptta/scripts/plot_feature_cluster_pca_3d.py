#!/usr/bin/env python3
"""Visualize clean/corrupted CLIP feature clusters and class/text centroids in PCA-3D."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "work" / ".mplconfig"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader, Dataset, Subset

from run_closed_set_cifar_benchmarks import DEFAULT_DATA_ROOT, FAMILIES, REPO_ROOT
from ttavlm.datasets import CORRUPTIONS, return_train_val_datasets
from ttavlm.models import return_base_model
from ttavlm.models.clip import tokenize


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "work" / "feature_cluster_pca_3d"
DEFAULT_FONT_ZIP = REPO_ROOT.parent / "TimesNewerRoman.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=list(FAMILIES.keys()), default="cifar10")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--shift-types", nargs="+", default=["all"])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-model-name", type=str, default="clip-ViT-B/16")
    parser.add_argument("--template", type=str, default="a photo of a {}")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--point-size", type=float, default=9.0)
    parser.add_argument("--point-alpha", type=float, default=0.45)
    parser.add_argument("--centroid-size", type=float, default=78.0)
    parser.add_argument("--elev", type=float, default=21.0)
    parser.add_argument("--azim", type=float, default=-56.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--write-sample-csv", action="store_true")
    parser.add_argument("--font-zip", type=Path, default=DEFAULT_FONT_ZIP)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def configure_plot_font(font_zip: Path) -> None:
    font_name = "Times Newer Roman"
    extracted_dir = REPO_ROOT / "work" / ".fonts" / font_zip.stem
    ensure_dir(extracted_dir)
    if font_zip.exists():
        with zipfile.ZipFile(font_zip) as archive:
            for member in archive.namelist():
                lower = member.lower()
                if lower.endswith(".otf") or lower.endswith(".ttf"):
                    target = extracted_dir / Path(member).name
                    if not target.exists():
                        target.write_bytes(archive.read(member))
        for font_path in sorted(extracted_dir.glob("*.[ot]tf")):
            font_manager.fontManager.addfont(str(font_path))

    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.serif": [font_name, "Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 18.0,
            "xtick.labelsize": 14.0,
            "ytick.labelsize": 14.0,
            "legend.fontsize": 13.0,
        }
    )


def resolve_shift_types(values: Sequence[str]) -> List[str]:
    if list(values) == ["all"]:
        return list(CORRUPTIONS)
    return list(values)


def choose_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False.")
        device = torch.device(device_name)
        torch.cuda.set_device(device)
        return device
    return torch.device(device_name)


def encode_text_features(
    class_names: Sequence[str],
    template: str,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    texts = [template.format(name.replace("_", " ")) for name in class_names]
    tokenized = tokenize(texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokenized).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    return text_features.cpu()


def stratified_indices(dataset: Dataset, max_samples_per_class: int | None, seed: int) -> List[int]:
    labels = getattr(dataset, "labels", None)
    if labels is None:
        labels = getattr(dataset, "targets", None)
    if labels is None:
        raise AttributeError("Dataset does not expose `labels` or `targets` for stratified sampling.")

    labels_np = np.asarray(labels, dtype=np.int64)
    all_indices = np.arange(labels_np.shape[0], dtype=np.int64)
    if max_samples_per_class is None:
        return all_indices.tolist()

    rng = np.random.default_rng(seed)
    selected: List[int] = []
    for class_id in sorted(int(v) for v in np.unique(labels_np)):
        class_indices = all_indices[labels_np == class_id]
        if class_indices.size <= max_samples_per_class:
            chosen = class_indices
        else:
            chosen = np.sort(rng.choice(class_indices, size=max_samples_per_class, replace=False))
        selected.extend(int(idx) for idx in chosen.tolist())
    selected.sort()
    return selected


def build_loader(dataset: Dataset, indices: Sequence[int], batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    subset = Subset(dataset, list(indices))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def extract_feature_tensor(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feats: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    visual = model.visual if hasattr(model, "visual") else model
    if not hasattr(visual, "use_local"):
        visual.use_local = False
    dtype = getattr(visual, "dtype", None)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            batch_labels = batch["target"]
            if dtype is not None:
                outputs = visual(images.type(dtype))
            else:
                outputs = visual(images)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            outputs = outputs.float()
            outputs = outputs / outputs.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
            feats.append(outputs.cpu())
            labels.append(batch_labels.cpu())

    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def compute_class_centroids(features: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    centroids: List[torch.Tensor] = []
    for class_id in range(num_classes):
        mask = labels == class_id
        if not bool(mask.any()):
            raise ValueError(f"Class {class_id} is missing from the sampled features.")
        center = features[mask].mean(dim=0)
        center = center / center.norm().clamp_min(1.0e-12)
        centroids.append(center)
    return torch.stack(centroids, dim=0)


def fit_pca_3d(*arrays: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
    mean = stacked.mean(axis=0, keepdims=True)
    centered = stacked - mean
    covariance = centered.T @ centered
    covariance /= max(centered.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1][:3]
    basis = eigvecs[:, order]
    return mean, basis


def project_pca(features: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (features.astype(np.float32, copy=False) - mean) @ basis


def class_colors(num_classes: int) -> List[Tuple[float, float, float, float]]:
    if num_classes <= 10:
        cmap = plt.get_cmap("tab10")
    elif num_classes <= 20:
        cmap = plt.get_cmap("tab20")
    else:
        cmap = plt.get_cmap("gist_ncar")
    return [cmap(i / max(num_classes - 1, 1)) for i in range(num_classes)]


def format_display_label(name: str) -> str:
    return name.replace("_", " ").title()


def set_3d_limits(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = np.maximum(maxs - mins, 1.0e-6)
    max_span = float(spans.max())
    centers = (mins + maxs) / 2.0
    half = max_span * 0.55
    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(centers[2] - half, centers[2] + half)


def style_3d_axis(ax, elev: float, azim: float) -> None:
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("PC1", labelpad=10)
    ax.set_ylabel("PC2", labelpad=10)
    ax.set_zlabel("PC3", labelpad=14)
    ax.tick_params(axis="both", which="major", labelsize=13, pad=2)
    ax.tick_params(axis="z", which="major", labelsize=13, pad=2)
    ax.grid(True, alpha=0.25)


def save_legend_only(handles: Sequence[Line2D], labels: Sequence[str], output_path: Path, *, ncol: int = 1) -> None:
    rows = max(1, int(np.ceil(len(labels) / max(ncol, 1))))
    fig_width = max(3.8, 2.45 * max(ncol, 1))
    fig_height = max(1.5, 0.72 * rows)
    fig = plt.figure(figsize=(fig_width, fig_height))
    legend = fig.legend(
        handles,
        labels,
        loc="center",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.92,
        borderpad=0.62,
        handletextpad=0.55,
        labelspacing=0.4,
        ncol=ncol,
        fontsize=22,
    )
    fig.canvas.draw()
    bbox = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(output_path, dpi=220, bbox_inches=bbox.expanded(1.08, 1.12), transparent=True)
    plt.close(fig)


def build_class_legend_handles(
    class_names: Sequence[str],
    colors: Sequence[Tuple[float, float, float, float]],
) -> Tuple[List[Line2D], List[str]]:
    handles: List[Line2D] = []
    labels: List[str] = []
    for class_id, class_name in enumerate(class_names):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=colors[class_id],
                markeredgecolor="black",
                markeredgewidth=0.55,
                markersize=13.5,
            )
        )
        labels.append(format_display_label(class_name))
    return handles, labels


def plot_feature_scatter(
    *,
    projected: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    colors: Sequence[Tuple[float, float, float, float]],
    output_path: Path,
    legend_output_path: Path | None,
    title: str,
    point_size: float,
    point_alpha: float,
    elev: float,
    azim: float,
) -> None:
    fig = plt.figure(figsize=(8.8, 7.1))
    ax = fig.add_subplot(111, projection="3d")
    del title
    for class_id, class_name in enumerate(class_names):
        mask = labels == class_id
        if not np.any(mask):
            continue
        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            projected[mask, 2],
            s=point_size,
            alpha=point_alpha,
            color=colors[class_id],
            edgecolors="none",
            label=format_display_label(class_name),
        )
    set_3d_limits(ax, projected)
    style_3d_axis(ax, elev=elev, azim=azim)
    if legend_output_path is not None and len(class_names) <= 12:
        handles, labels_for_legend = build_class_legend_handles(class_names, colors)
        save_legend_only(handles, labels_for_legend, legend_output_path, ncol=len(labels_for_legend))
    fig.subplots_adjust(left=0.03, right=0.95, bottom=0.05, top=0.98)
    fig.savefig(output_path, dpi=220, pad_inches=0.12)
    plt.close(fig)


def plot_centroids(
    *,
    clean_centroids: np.ndarray,
    corrupt_centroids: np.ndarray,
    text_centroids: np.ndarray,
    class_names: Sequence[str],
    colors: Sequence[Tuple[float, float, float, float]],
    output_path: Path,
    legend_output_path: Path | None,
    title: str,
    centroid_size: float,
    elev: float,
    azim: float,
) -> None:
    fig = plt.figure(figsize=(9.4, 7.4))
    ax = fig.add_subplot(111, projection="3d")
    del title

    for class_id, class_name in enumerate(class_names):
        color = colors[class_id]
        clean_pt = clean_centroids[class_id]
        corrupt_pt = corrupt_centroids[class_id]
        text_pt = text_centroids[class_id]

        ax.plot(
            [clean_pt[0], corrupt_pt[0]],
            [clean_pt[1], corrupt_pt[1]],
            [clean_pt[2], corrupt_pt[2]],
            color=color,
            linewidth=1.5,
            alpha=0.55,
        )
        ax.scatter(clean_pt[0], clean_pt[1], clean_pt[2], marker="o", s=centroid_size, color=color, alpha=0.95)
        ax.scatter(corrupt_pt[0], corrupt_pt[1], corrupt_pt[2], marker="^", s=centroid_size, color=color, alpha=0.95)
        ax.scatter(
            text_pt[0],
            text_pt[1],
            text_pt[2],
            marker="s",
            s=centroid_size * 1.05,
            color=color,
            alpha=0.98,
            edgecolors="black",
            linewidths=0.5,
        )
        if len(class_names) <= 12:
            ax.text(
                text_pt[0],
                text_pt[1],
                text_pt[2],
                f" {format_display_label(class_name)}",
                color=color,
                fontsize=9.0,
            )

    all_points = np.concatenate([clean_centroids, corrupt_centroids, text_centroids], axis=0)
    set_3d_limits(ax, all_points)
    style_3d_axis(ax, elev=elev, azim=azim)

    marker_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666", markersize=8, label="Clean Centroid"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#666666", markersize=8, label="Corrupted Centroid"),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#666666",
            markeredgecolor="black",
            markersize=8,
            label="Text Prototype",
        ),
    ]
    if legend_output_path is not None:
        save_legend_only(
            marker_handles,
            [handle.get_label() for handle in marker_handles],
            legend_output_path,
            ncol=1,
        )
    fig.subplots_adjust(left=0.03, right=0.95, bottom=0.05, top=0.98)
    fig.savefig(output_path, dpi=220, pad_inches=0.12)
    plt.close(fig)


def write_centroid_csv(
    path: Path,
    class_names: Sequence[str],
    clean_centroids: np.ndarray,
    corrupt_centroids: np.ndarray,
    text_centroids: np.ndarray,
) -> None:
    lines = ["class_id,class_name,point_type,pc1,pc2,pc3"]
    for class_id, class_name in enumerate(class_names):
        entries = [
            ("clean_centroid", clean_centroids[class_id]),
            ("corrupted_centroid", corrupt_centroids[class_id]),
            ("text_prototype", text_centroids[class_id]),
        ]
        for point_type, point in entries:
            lines.append(
                f"{class_id},{class_name},{point_type},{point[0]:.8f},{point[1]:.8f},{point[2]:.8f}"
            )
    path.write_text("\n".join(lines) + "\n")


def write_sample_csv(
    path: Path,
    projected: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    point_type: str,
) -> None:
    lines = ["point_type,class_id,class_name,pc1,pc2,pc3"]
    for idx in range(projected.shape[0]):
        class_id = int(labels[idx])
        point = projected[idx]
        lines.append(
            f"{point_type},{class_id},{class_names[class_id]},{point[0]:.8f},{point[1]:.8f},{point[2]:.8f}"
        )
    path.write_text("\n".join(lines) + "\n")


def metadata_payload(args: argparse.Namespace, indices: Sequence[int], class_names: Sequence[str]) -> Dict[str, object]:
    return {
        "family": args.family,
        "severity": int(args.severity),
        "shift_types": resolve_shift_types(args.shift_types),
        "template": args.template,
        "base_model_name": args.base_model_name,
        "device": args.device,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "seed": int(args.seed),
        "max_samples_per_class": args.max_samples_per_class,
        "num_indices": len(indices),
        "num_classes": len(class_names),
    }


def main() -> None:
    args = parse_args()
    configure_plot_font(args.font_zip)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    shifts = resolve_shift_types(args.shift_types)
    family_cfg = FAMILIES[args.family]
    clean_name = str(family_cfg["clean_dataset"])
    corrupt_name = str(family_cfg["corrupted_dataset"])

    output_root = args.output_root / args.family / f"severity_{args.severity}"
    cache_dir = output_root / "cache"
    ensure_dir(cache_dir)

    model, transform = return_base_model(
        name=args.base_model_name,
        device=device,
        dataset=corrupt_name,
        path_to_weights=str(REPO_ROOT / "work"),
    )
    model.eval()

    _, clean_dataset = return_train_val_datasets(
        name=clean_name,
        data_dir=str(args.data_root),
        train_transform=transform,
        val_transform=transform,
    )
    class_names = list(clean_dataset.class_names)
    colors = class_colors(len(class_names))
    indices = stratified_indices(clean_dataset, args.max_samples_per_class, args.seed)

    clean_loader = build_loader(clean_dataset, indices, args.batch_size, args.num_workers, device)
    sample_tag = "all" if args.max_samples_per_class is None else f"max{args.max_samples_per_class}pc"
    clean_cache = cache_dir / f"clean_{sample_tag}.pt"
    if clean_cache.exists():
        clean_payload = torch.load(clean_cache, map_location="cpu")
        clean_features = clean_payload["features"]
        clean_labels = clean_payload["labels"]
    else:
        clean_features, clean_labels = extract_feature_tensor(model, clean_loader, device)
        torch.save({"features": clean_features, "labels": clean_labels}, clean_cache)

    text_features = encode_text_features(class_names, args.template, model, device)

    clean_centroids = compute_class_centroids(clean_features, clean_labels, len(class_names))
    clean_np = clean_features.numpy().astype(np.float32, copy=False)
    clean_labels_np = clean_labels.numpy().astype(np.int64, copy=False)
    text_np = text_features.numpy().astype(np.float32, copy=False)
    clean_centroids_np = clean_centroids.numpy().astype(np.float32, copy=False)

    metadata = metadata_payload(args, indices, class_names)
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    for corruption in shifts:
        corruption_dir = output_root / corruption
        summary_path = corruption_dir / "summary.json"
        if args.skip_existing and summary_path.exists():
            continue

        ensure_dir(corruption_dir)
        _, corrupt_dataset = return_train_val_datasets(
            name=corrupt_name,
            data_dir=str(args.data_root),
            train_transform=transform,
            val_transform=transform,
            shift=corruption,
            severity=args.severity,
        )
        corrupt_loader = build_loader(corrupt_dataset, indices, args.batch_size, args.num_workers, device)
        corrupt_cache = cache_dir / f"{corruption}_{sample_tag}.pt"
        if corrupt_cache.exists():
            corrupt_payload = torch.load(corrupt_cache, map_location="cpu")
            corrupt_features = corrupt_payload["features"]
            corrupt_labels = corrupt_payload["labels"]
        else:
            corrupt_features, corrupt_labels = extract_feature_tensor(model, corrupt_loader, device)
            torch.save({"features": corrupt_features, "labels": corrupt_labels}, corrupt_cache)

        if clean_labels.shape != corrupt_labels.shape or not torch.equal(clean_labels, corrupt_labels):
            raise ValueError(f"Label mismatch between clean and {corruption} subsets; paired comparison is invalid.")

        corrupt_centroids = compute_class_centroids(corrupt_features, corrupt_labels, len(class_names))
        corrupt_np = corrupt_features.numpy().astype(np.float32, copy=False)
        corrupt_labels_np = corrupt_labels.numpy().astype(np.int64, copy=False)
        corrupt_centroids_np = corrupt_centroids.numpy().astype(np.float32, copy=False)

        pca_mean, pca_basis = fit_pca_3d(clean_np, corrupt_np, clean_centroids_np, corrupt_centroids_np, text_np)
        clean_proj = project_pca(clean_np, pca_mean, pca_basis)
        corrupt_proj = project_pca(corrupt_np, pca_mean, pca_basis)
        clean_centroids_proj = project_pca(clean_centroids_np, pca_mean, pca_basis)
        corrupt_centroids_proj = project_pca(corrupt_centroids_np, pca_mean, pca_basis)
        text_proj = project_pca(text_np, pca_mean, pca_basis)

        plot_feature_scatter(
            projected=clean_proj,
            labels=clean_labels_np,
            class_names=class_names,
            colors=colors,
            output_path=corruption_dir / "clean_features_pca3d.png",
            legend_output_path=corruption_dir / "class_legend.png",
            title=f"Clean CLIP features\nfamily={args.family}, reference for {corruption}, severity={args.severity}",
            point_size=args.point_size,
            point_alpha=args.point_alpha,
            elev=args.elev,
            azim=args.azim,
        )
        plot_feature_scatter(
            projected=corrupt_proj,
            labels=corrupt_labels_np,
            class_names=class_names,
            colors=colors,
            output_path=corruption_dir / "corrupted_features_pca3d.png",
            legend_output_path=None,
            title=f"Corrupted CLIP features\ncorruption={corruption}, severity={args.severity}",
            point_size=args.point_size,
            point_alpha=args.point_alpha,
            elev=args.elev,
            azim=args.azim,
        )
        plot_centroids(
            clean_centroids=clean_centroids_proj,
            corrupt_centroids=corrupt_centroids_proj,
            text_centroids=text_proj,
            class_names=class_names,
            colors=colors,
            output_path=corruption_dir / "centroids_and_text_pca3d.png",
            legend_output_path=corruption_dir / "centroids_marker_legend.png",
            title=f"Centroid alignment in shared PCA space\ncorruption={corruption}, severity={args.severity}",
            centroid_size=args.centroid_size,
            elev=args.elev,
            azim=args.azim,
        )

        write_centroid_csv(
            corruption_dir / "centroids_pca3d.csv",
            class_names,
            clean_centroids_proj,
            corrupt_centroids_proj,
            text_proj,
        )
        if args.write_sample_csv:
            write_sample_csv(corruption_dir / "clean_points_pca3d.csv", clean_proj, clean_labels_np, class_names, "clean")
            write_sample_csv(
                corruption_dir / "corrupted_points_pca3d.csv",
                corrupt_proj,
                corrupt_labels_np,
                class_names,
                "corrupted",
            )

        summary = {
            "corruption": corruption,
            "severity": int(args.severity),
            "num_samples": int(clean_proj.shape[0]),
            "num_classes": len(class_names),
            "files": {
                "clean_plot": str(corruption_dir / "clean_features_pca3d.png"),
                "corrupted_plot": str(corruption_dir / "corrupted_features_pca3d.png"),
                "centroid_plot": str(corruption_dir / "centroids_and_text_pca3d.png"),
                "centroid_csv": str(corruption_dir / "centroids_pca3d.csv"),
            },
            "pca_basis_norms": [float(np.linalg.norm(pca_basis[:, i])) for i in range(pca_basis.shape[1])],
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
