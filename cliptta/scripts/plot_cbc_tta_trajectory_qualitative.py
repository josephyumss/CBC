#!/usr/bin/env python3
"""Qualitative CBC and pseudo-label TTA trajectory visualizations for CIFAR-10-C.

The script produces three diagnostic views per corruption:

1. CBC's immediate shift in 3D class-logit evidence PCA space.
2. Step-wise 3D logit-evidence trajectories under a controlled STAMP-style
   pseudo-label visual adapter, with and without CBC pseudo labels.
3. Step-wise 3D feature-centroid trajectories against clean feature centroids.
4. Per-step 3D feature distribution frames that visualize restoration.
5. Step-wise alignment/shape metrics that show whether adaptation moves
   corrupted features toward text prototypes and restores clean-like clusters.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import zipfile
from copy import deepcopy
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
import torch.nn.functional as F
from matplotlib import font_manager
from matplotlib.lines import Line2D
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from run_closed_set_cifar_benchmarks import DEFAULT_DATA_ROOT  # noqa: E402
from ttavlm.datasets import return_train_val_datasets  # noqa: E402
from ttavlm.models import return_base_model  # noqa: E402
from ttavlm.models.clip import tokenize  # noqa: E402


DEFAULT_TEMPLATE = "a photo of a {}"
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "complete"
    / "final_paper_experiments"
    / "05_qualitative_analysis"
    / "cbc_tta_trajectory_stamp"
)
DEFAULT_FONT_ZIP = WORKSPACE_ROOT / "TimesNewerRoman.zip"
CLASS_COLORS = list(plt.cm.tab10.colors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--corruptions", nargs="+", default=["brightness", "glass_blur"])
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--base-model-name", type=str, default="clip-ViT-B/16")
    parser.add_argument("--template", type=str, default=DEFAULT_TEMPLATE)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--samples-per-class", type=int, default=80)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logit-scale", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--font-zip", type=Path, default=DEFAULT_FONT_ZIP)
    parser.add_argument("--elev", type=float, default=22.0)
    parser.add_argument("--azim", type=float, default=-58.0)
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
        extracted_dir = REPO_ROOT / "work" / ".fonts" / font_zip.stem
        ensure_dir(extracted_dir)
        with zipfile.ZipFile(font_zip) as archive:
            for member in archive.namelist():
                if member.lower().endswith((".otf", ".ttf")):
                    target = extracted_dir / Path(member).name
                    if not target.exists():
                        target.write_bytes(archive.read(member))
        for font_path in sorted(extracted_dir.glob("*.[ot]tf")):
            font_manager.fontManager.addfont(str(font_path))
        font_name = "Times Newer Roman"

    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.serif": [font_name, "Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "figure.dpi": 140,
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


def load_cifar10_dataset(data_root: Path, transform) -> Dataset:
    _, dataset = return_train_val_datasets(
        name="cifar10",
        data_dir=str(data_root),
        train_transform=transform,
        val_transform=transform,
    )
    return dataset


def load_cifar10c_dataset(data_root: Path, transform, corruption: str, severity: int) -> Dataset:
    _, dataset = return_train_val_datasets(
        name="cifar10c",
        data_dir=str(data_root),
        train_transform=transform,
        val_transform=transform,
        shift=corruption,
        severity=int(severity),
    )
    return dataset


def stratified_indices(dataset: Dataset, samples_per_class: int | None, seed: int) -> List[int]:
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
        if samples_per_class is None or class_indices.size <= samples_per_class:
            chosen = np.sort(class_indices)
        else:
            chosen = np.sort(rng.choice(class_indices, size=samples_per_class, replace=False))
        selected.extend(int(idx) for idx in chosen.tolist())
    selected.sort()
    return selected


def build_loader(
    dataset: Dataset,
    indices: Sequence[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(1234)
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=generator if shuffle else None,
    )


def encode_text_features(
    model: torch.nn.Module,
    class_names: Sequence[str],
    template: str,
    device: torch.device,
) -> Tensor:
    prompts = [template.format(name.replace("_", " ")) for name in class_names]
    tokens = tokenize(prompts, truncate=True).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens).float()
        text_features = F.normalize(text_features, dim=-1)
    return text_features


def visual_model(model: torch.nn.Module) -> torch.nn.Module:
    visual = model.visual if hasattr(model, "visual") else model
    if not hasattr(visual, "use_local"):
        visual.use_local = False
    return visual


def encode_images(model: torch.nn.Module, images: Tensor, device: torch.device) -> Tensor:
    visual = visual_model(model)
    dtype = getattr(visual, "dtype", torch.float32)
    features = model.encode_image(images.to(device, non_blocking=True).type(dtype)).float()
    return F.normalize(features, dim=-1)


@torch.no_grad()
def extract_features_and_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    text_features: Tensor,
    device: torch.device,
    logit_scale: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    model.eval()
    features: List[Tensor] = []
    logits: List[Tensor] = []
    labels: List[Tensor] = []
    for batch in loader:
        images = batch["image"]
        batch_labels = batch["target"].long()
        batch_features = encode_images(model, images, device)
        batch_logits = logit_scale * batch_features @ text_features.t()
        features.append(batch_features.cpu())
        logits.append(batch_logits.cpu())
        labels.append(batch_labels.cpu())
    return torch.cat(features), torch.cat(logits), torch.cat(labels)


def center_logits(logits: Tensor) -> Tensor:
    return logits - logits.mean(dim=0, keepdim=True)


def pca_project(arrays: Sequence[np.ndarray], dims: int = 2) -> List[np.ndarray]:
    stacked = np.concatenate(arrays, axis=0).astype(np.float64)
    mean = stacked.mean(axis=0, keepdims=True)
    centered = stacked - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:dims].T
    return [(arr.astype(np.float64) - mean) @ components for arr in arrays]


def setup_3d_axis(ax, elev: float, azim: float) -> None:
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("PC1", labelpad=10)
    ax.set_ylabel("PC2", labelpad=10)
    ax.set_zlabel("PC3", labelpad=10)
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.6)
    ax.xaxis.pane.set_alpha(0.03)
    ax.yaxis.pane.set_alpha(0.03)
    ax.zaxis.pane.set_alpha(0.03)


def equalize_3d_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 1.0e-6)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def draw_3d_segment(ax, start: np.ndarray, end: np.ndarray, color: str, alpha: float = 0.55, lw: float = 1.0) -> None:
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
        color=color,
        alpha=alpha,
        linewidth=lw,
    )
    delta = end - start
    ax.quiver(
        start[0],
        start[1],
        start[2],
        delta[0],
        delta[1],
        delta[2],
        color=color,
        alpha=min(alpha + 0.1, 0.9),
        linewidth=lw,
        arrow_length_ratio=0.18,
        normalize=False,
    )


def class_centroids(values: Tensor, labels: Tensor, num_classes: int) -> Tensor:
    centroids: List[Tensor] = []
    for class_id in range(num_classes):
        mask = labels == class_id
        if mask.any():
            centroids.append(values[mask].mean(dim=0))
        else:
            centroids.append(torch.full_like(values[0], float("nan")))
    return torch.stack(centroids)


def centroid_alignment(features: Tensor, labels: Tensor, text_features_cpu: Tensor) -> float:
    centroids = F.normalize(class_centroids(features, labels, text_features_cpu.shape[0]), dim=-1)
    valid = torch.isfinite(centroids).all(dim=1)
    sims = (centroids[valid] * text_features_cpu[valid]).sum(dim=1)
    return float(100.0 * sims.mean().item())


def clean_centroid_similarity(features: Tensor, labels: Tensor, clean_centroids: Tensor) -> float:
    centroids = F.normalize(class_centroids(features, labels, clean_centroids.shape[0]), dim=-1)
    clean = F.normalize(clean_centroids, dim=-1)
    valid = torch.isfinite(centroids).all(dim=1) & torch.isfinite(clean).all(dim=1)
    sims = (centroids[valid] * clean[valid]).sum(dim=1)
    return float(100.0 * sims.mean().item())


def clean_centroid_distance(features: Tensor, labels: Tensor, clean_centroids: Tensor) -> float:
    centroids = class_centroids(features, labels, clean_centroids.shape[0])
    valid = torch.isfinite(centroids).all(dim=1) & torch.isfinite(clean_centroids).all(dim=1)
    distances = torch.norm(centroids[valid] - clean_centroids[valid], dim=1)
    return float(distances.mean().item())


def mean_intraclass_spread(features: Tensor, labels: Tensor, num_classes: int) -> float:
    spreads: List[float] = []
    for class_id in range(num_classes):
        mask = labels == class_id
        if int(mask.sum().item()) <= 1:
            continue
        cls_features = features[mask]
        centroid = cls_features.mean(dim=0, keepdim=True)
        spreads.append(float(((cls_features - centroid) ** 2).sum(dim=1).mean().item()))
    return float(np.mean(spreads)) if spreads else float("nan")


def inter_intra_ratio(features: Tensor, labels: Tensor, num_classes: int) -> float:
    centroids = class_centroids(features, labels, num_classes)
    valid = torch.isfinite(centroids).all(dim=1)
    centroids = centroids[valid]
    if centroids.shape[0] <= 1:
        return float("nan")
    inter = torch.pdist(centroids, p=2).mean().item()
    intra = mean_intraclass_spread(features, labels, num_classes)
    return float(inter / (np.sqrt(intra) + 1.0e-12))


def top1_accuracy(logits: Tensor, labels: Tensor) -> float:
    return 100.0 * float(logits.argmax(dim=1).eq(labels).float().mean().item())


def configure_trainable_visual_norms(model: torch.nn.Module) -> List[str]:
    for param in model.parameters():
        param.requires_grad_(False)
    trainable_names: List[str] = []
    for name, param in visual_model(model).named_parameters():
        lower = name.lower()
        if "ln" in lower or "norm" in lower:
            param.requires_grad_(True)
            trainable_names.append(f"visual.{name}")
    if not trainable_names:
        for name, param in visual_model(model).named_parameters():
            param.requires_grad_(True)
            trainable_names.append(f"visual.{name}")
    return trainable_names


def adaptation_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    text_features: Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    logit_scale: float,
    use_cbc_pseudo_labels: bool,
) -> Tuple[float, float]:
    model.train()
    losses: List[float] = []
    pseudo_correct: List[float] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["target"].long().to(device, non_blocking=True)
        with torch.no_grad():
            pseudo_features = encode_images(model, images, device)
            pseudo_logits = logit_scale * pseudo_features @ text_features.t()
            used_logits = center_logits(pseudo_logits) if use_cbc_pseudo_labels else pseudo_logits
            pseudo_labels = used_logits.argmax(dim=1)
            pseudo_correct.append(float(pseudo_labels.eq(labels).float().mean().item()))

        features = encode_images(model, images, device)
        logits = logit_scale * features @ text_features.t()
        loss = F.cross_entropy(logits, pseudo_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)), 100.0 * float(np.mean(pseudo_correct))


def plot_logit_shift(
    output_dir: Path,
    corruption: str,
    logits: Tensor,
    labels: Tensor,
    class_names: Sequence[str],
    elev: float,
    azim: float,
) -> None:
    num_classes = len(class_names)
    raw_centroids = class_centroids(logits, labels, num_classes).numpy()
    cbc_centroids = class_centroids(center_logits(logits), labels, num_classes).numpy()
    ideal = np.eye(num_classes, dtype=np.float64) * float(np.nanmax(np.abs(cbc_centroids)))
    raw_3d, cbc_3d, ideal_3d = pca_project([raw_centroids, cbc_centroids, ideal], dims=3)

    fig = plt.figure(figsize=(8.0, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    for class_id, class_name in enumerate(class_names):
        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
        ax.scatter(*raw_3d[class_id], marker="o", s=55, color=color, alpha=0.45)
        ax.scatter(*cbc_3d[class_id], marker="D", s=62, color=color, edgecolor="black", linewidth=0.5)
        ax.scatter(*ideal_3d[class_id], marker="*", s=95, color=color, edgecolor="black", linewidth=0.4)
        draw_3d_segment(ax, raw_3d[class_id], cbc_3d[class_id], color=color, alpha=0.75, lw=1.25)
        ax.text(*cbc_3d[class_id], class_name.replace("_", " "), fontsize=8)
    setup_3d_axis(ax, elev=elev, azim=azim)
    equalize_3d_axes(ax, np.concatenate([raw_3d, cbc_3d, ideal_3d], axis=0))
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=8, label="CLIP logits"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=8, label="After CBC"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=10, label="Ideal class evidence"),
    ]
    ax.legend(handles=handles, frameon=True, facecolor="white", edgecolor="0.7")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"cbc_logit_shift_3d_{corruption}.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def record_state(
    *,
    method: str,
    step: int,
    corruption: str,
    features: Tensor,
    logits: Tensor,
    labels: Tensor,
    text_features_cpu: Tensor,
    clean_centroids_cpu: Tensor,
    clean_intraclass_spread: float,
    loss: float | None,
    pseudo_top1: float | None,
) -> Dict[str, float | int | str]:
    num_classes = text_features_cpu.shape[0]
    spread = mean_intraclass_spread(features, labels, num_classes)
    return {
        "corruption": corruption,
        "method": method,
        "step": step,
        "loss": float("nan") if loss is None else float(loss),
        "pseudo_top1": float("nan") if pseudo_top1 is None else float(pseudo_top1),
        "top1": top1_accuracy(logits, labels),
        "mean_centroid_text_cosine_x100": centroid_alignment(features, labels, text_features_cpu),
        "mean_clean_centroid_cosine_x100": clean_centroid_similarity(features, labels, clean_centroids_cpu),
        "mean_clean_centroid_l2": clean_centroid_distance(features, labels, clean_centroids_cpu),
        "mean_intraclass_spread": spread,
        "intraclass_spread_gap_to_clean": abs(spread - clean_intraclass_spread),
        "inter_intra_ratio": inter_intra_ratio(features, labels, num_classes),
    }


def run_visual_adapter_trajectory(
    *,
    base_model: torch.nn.Module,
    base_visual_state: Mapping[str, Tensor],
    loader_eval: DataLoader,
    loader_train: DataLoader,
    text_features: Tensor,
    labels: Tensor,
    text_features_cpu: Tensor,
    clean_centroids_cpu: Tensor,
    clean_intraclass_spread: float,
    device: torch.device,
    corruption: str,
    steps: int,
    lr: float,
    weight_decay: float,
    logit_scale: float,
) -> Tuple[List[Dict[str, object]], Dict[str, List[Tensor]], Dict[str, List[Tensor]], Dict[str, List[Tensor]], List[str]]:
    metric_rows: List[Dict[str, object]] = []
    feature_paths: Dict[str, List[Tensor]] = {}
    feature_samples: Dict[str, List[Tensor]] = {}
    logit_paths: Dict[str, List[Tensor]] = {}
    trainable_names: List[str] = []

    for method, use_cbc in [("TTA w/o CBC", False), ("TTA + CBC", True)]:
        base_model.load_state_dict({"visual." + key: value for key, value in base_visual_state.items()}, strict=False)
        names = configure_trainable_visual_norms(base_model)
        if not trainable_names:
            trainable_names = names
        optimizer = torch.optim.AdamW(
            [param for param in base_model.parameters() if param.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )
        features, logits, _ = extract_features_and_logits(base_model, loader_eval, text_features, device, logit_scale)
        feature_paths[method] = [class_centroids(features, labels, text_features_cpu.shape[0])]
        feature_samples[method] = [features]
        logit_paths[method] = [class_centroids(logits, labels, text_features_cpu.shape[0])]
        metric_rows.append(
            record_state(
                method=method,
                step=0,
                corruption=corruption,
                features=features,
                logits=logits,
                labels=labels,
                text_features_cpu=text_features_cpu,
                clean_centroids_cpu=clean_centroids_cpu,
                clean_intraclass_spread=clean_intraclass_spread,
                loss=None,
                pseudo_top1=None,
            )
        )
        for step in range(1, steps + 1):
            loss, pseudo_top1 = adaptation_epoch(
                base_model,
                loader_train,
                text_features,
                optimizer,
                device,
                logit_scale=logit_scale,
                use_cbc_pseudo_labels=use_cbc,
            )
            features, logits, _ = extract_features_and_logits(base_model, loader_eval, text_features, device, logit_scale)
            feature_paths[method].append(class_centroids(features, labels, text_features_cpu.shape[0]))
            feature_samples[method].append(features)
            logit_paths[method].append(class_centroids(logits, labels, text_features_cpu.shape[0]))
            metric_rows.append(
                record_state(
                    method=method,
                    step=step,
                    corruption=corruption,
                    features=features,
                    logits=logits,
                    labels=labels,
                    text_features_cpu=text_features_cpu,
                    clean_centroids_cpu=clean_centroids_cpu,
                    clean_intraclass_spread=clean_intraclass_spread,
                    loss=loss,
                    pseudo_top1=pseudo_top1,
                )
            )
    return metric_rows, feature_paths, feature_samples, logit_paths, trainable_names


def plot_feature_trajectory(
    output_dir: Path,
    corruption: str,
    paths: Mapping[str, Sequence[Tensor]],
    clean_centroids_cpu: Tensor,
    class_names: Sequence[str],
    elev: float,
    azim: float,
) -> List[Dict[str, object]]:
    arrays: List[np.ndarray] = []
    keys: List[Tuple[str, int]] = []
    for method, centroid_steps in paths.items():
        for step, centroids in enumerate(centroid_steps):
            arrays.append(centroids.numpy())
            keys.append((method, step))
    arrays.append(clean_centroids_cpu.numpy())
    projected = pca_project(arrays, dims=3)
    projection_by_key = {key: value for key, value in zip(keys, projected[:-1])}
    clean_3d = projected[-1]

    fig = plt.figure(figsize=(8.2, 6.9))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(clean_3d[:, 0], clean_3d[:, 1], clean_3d[:, 2], marker="*", s=125, color="black", label="Clean feature centroids", zorder=5)
    for class_id, class_name in enumerate(class_names):
        ax.text(clean_3d[class_id, 0], clean_3d[class_id, 1], clean_3d[class_id, 2], class_name.replace("_", " "), fontsize=8)

    style = {
        "TTA w/o CBC": {"color": "#1f77b4", "marker": "o", "label": "TTA w/o CBC"},
        "TTA + CBC": {"color": "#d62728", "marker": "D", "label": "TTA + CBC"},
    }
    centroid_rows: List[Dict[str, object]] = []
    for method, centroid_steps in paths.items():
        method_style = style[method]
        for class_id in range(len(class_names)):
            coords = np.stack([projection_by_key[(method, step)][class_id] for step in range(len(centroid_steps))])
            alpha = 0.28 if method == "TTA w/o CBC" else 0.45
            ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=method_style["color"], alpha=alpha, linewidth=1.0)
            ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color="white", edgecolor=method_style["color"], s=28, linewidth=0.9)
            ax.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], color=method_style["color"], marker=method_style["marker"], s=42, alpha=0.9)
            for step in range(1, len(coords)):
                draw_3d_segment(ax, coords[step - 1], coords[step], color=method_style["color"], alpha=0.32, lw=0.65)
            for step, coord in enumerate(coords):
                centroid_rows.append(
                    {
                        "corruption": corruption,
                        "method": method,
                        "step": step,
                        "class_id": class_id,
                        "class_name": class_names[class_id],
                        "pc1": float(coord[0]),
                        "pc2": float(coord[1]),
                        "pc3": float(coord[2]),
                    }
                )

    handles = [
        Line2D([0], [0], marker="*", color="black", linestyle="", markersize=10, label="Clean feature centroids"),
        Line2D([0], [0], color=style["TTA w/o CBC"]["color"], marker="o", markersize=7, label="TTA w/o CBC"),
        Line2D([0], [0], color=style["TTA + CBC"]["color"], marker="D", markersize=7, label="TTA + CBC"),
    ]
    ax.legend(handles=handles, frameon=True, facecolor="white", edgecolor="0.7", loc="best")
    setup_3d_axis(ax, elev=elev, azim=azim)
    equalize_3d_axes(ax, np.concatenate([clean_3d] + [projection_by_key[key] for key in keys], axis=0))
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"tta_feature_trajectory_3d_{corruption}.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    return centroid_rows


def plot_logit_trajectory(
    output_dir: Path,
    corruption: str,
    logit_paths: Mapping[str, Sequence[Tensor]],
    class_names: Sequence[str],
    elev: float,
    azim: float,
) -> List[Dict[str, object]]:
    num_classes = len(class_names)
    arrays: List[np.ndarray] = []
    keys: List[Tuple[str, int]] = []
    for method, centroid_steps in logit_paths.items():
        for step, centroids in enumerate(centroid_steps):
            arrays.append(center_logits(centroids).numpy())
            keys.append((method, step))
    scale = max(float(np.nanmax(np.abs(np.concatenate(arrays, axis=0)))), 1.0)
    ideal = np.eye(num_classes, dtype=np.float64) * scale
    arrays.append(ideal)
    projected = pca_project(arrays, dims=3)
    projection_by_key = {key: value for key, value in zip(keys, projected[:-1])}
    ideal_3d = projected[-1]

    fig = plt.figure(figsize=(8.2, 6.9))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(ideal_3d[:, 0], ideal_3d[:, 1], ideal_3d[:, 2], marker="*", s=125, color="black", zorder=5)
    for class_id, class_name in enumerate(class_names):
        ax.text(ideal_3d[class_id, 0], ideal_3d[class_id, 1], ideal_3d[class_id, 2], class_name.replace("_", " "), fontsize=8)

    style = {
        "TTA w/o CBC": {"color": "#1f77b4", "marker": "o"},
        "TTA + CBC": {"color": "#d62728", "marker": "D"},
    }
    rows: List[Dict[str, object]] = []
    for method, centroid_steps in logit_paths.items():
        method_style = style[method]
        for class_id in range(num_classes):
            coords = np.stack([projection_by_key[(method, step)][class_id] for step in range(len(centroid_steps))])
            alpha = 0.25 if method == "TTA w/o CBC" else 0.42
            ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=method_style["color"], alpha=alpha, linewidth=1.0)
            ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color="white", edgecolor=method_style["color"], s=28, linewidth=0.9)
            ax.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], color=method_style["color"], marker=method_style["marker"], s=42, alpha=0.9)
            for step in range(1, len(coords)):
                draw_3d_segment(ax, coords[step - 1], coords[step], color=method_style["color"], alpha=alpha, lw=0.65)
            for step, coord in enumerate(coords):
                rows.append(
                    {
                        "corruption": corruption,
                        "method": method,
                        "step": step,
                        "class_id": class_id,
                        "class_name": class_names[class_id],
                        "pc1": float(coord[0]),
                        "pc2": float(coord[1]),
                        "pc3": float(coord[2]),
                    }
                )
    handles = [
        Line2D([0], [0], marker="*", color="black", linestyle="", markersize=10, label="Ideal class evidence"),
        Line2D([0], [0], color=style["TTA w/o CBC"]["color"], marker="o", markersize=7, label="TTA w/o CBC"),
        Line2D([0], [0], color=style["TTA + CBC"]["color"], marker="D", markersize=7, label="TTA + CBC"),
    ]
    ax.legend(handles=handles, frameon=True, facecolor="white", edgecolor="0.7", loc="best")
    setup_3d_axis(ax, elev=elev, azim=azim)
    equalize_3d_axes(ax, np.concatenate([ideal_3d] + [projection_by_key[key] for key in keys], axis=0))
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"tta_logit_trajectory_3d_{corruption}.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    return rows


def plot_feature_distribution_3d_steps(
    output_dir: Path,
    corruption: str,
    feature_samples: Mapping[str, Sequence[Tensor]],
    clean_centroids_cpu: Tensor,
    labels: Tensor,
    class_names: Sequence[str],
    elev: float,
    azim: float,
) -> None:
    methods = ["TTA w/o CBC", "TTA + CBC"]
    all_feature_arrays: List[np.ndarray] = []
    keys: List[Tuple[str, int]] = []
    for method in methods:
        for step, features in enumerate(feature_samples[method]):
            all_feature_arrays.append(features.numpy())
            keys.append((method, step))
    all_feature_arrays.append(clean_centroids_cpu.numpy())
    projected = pca_project(all_feature_arrays, dims=3)
    projected_by_key = {key: value for key, value in zip(keys, projected[:-1])}
    clean_3d = projected[-1]
    all_points = np.concatenate(projected, axis=0)
    labels_np = labels.numpy()
    num_steps = len(feature_samples[methods[0]])
    selected_steps = sorted(set([0, num_steps // 2, num_steps - 1]))

    def draw_panel(ax, method: str, step: int, *, show_labels: bool) -> None:
        points = projected_by_key[(method, step)]
        for class_id, class_name in enumerate(class_names):
            mask = labels_np == class_id
            color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
            ax.scatter(
                points[mask, 0],
                points[mask, 1],
                points[mask, 2],
                s=9,
                color=color,
                alpha=0.18,
                depthshade=False,
            )
            centroid = points[mask].mean(axis=0)
            ax.scatter(*centroid, s=48, marker="o", color=color, edgecolor="black", linewidth=0.35, depthshade=False)
            ax.scatter(*clean_3d[class_id], s=70, marker="*", color=color, edgecolor="black", linewidth=0.35, depthshade=False)
            if show_labels:
                ax.text(*clean_3d[class_id], class_name.replace("_", " "), fontsize=7)
        ax.set_title(f"{method}, step {step}", fontsize=14)
        setup_3d_axis(ax, elev=elev, azim=azim)
        equalize_3d_axes(ax, all_points)

    summary_fig = plt.figure(figsize=(15.0, 8.6))
    panel_idx = 1
    for method in methods:
        for step in selected_steps:
            ax = summary_fig.add_subplot(2, len(selected_steps), panel_idx, projection="3d")
            draw_panel(ax, method, step, show_labels=panel_idx == 1)
            panel_idx += 1
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=7, label="Corrupted samples / centroid"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=9, label="Clean centroid"),
    ]
    summary_fig.legend(handles=handles, frameon=True, facecolor="white", edgecolor="0.7", loc="lower center", ncol=2)
    summary_fig.tight_layout(rect=(0, 0.06, 1, 1))
    for suffix in ("png", "pdf"):
        summary_fig.savefig(output_dir / f"tta_feature_distribution_steps_3d_{corruption}.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(summary_fig)

    frames_dir = output_dir / "feature_distribution_3d_frames"
    ensure_dir(frames_dir)
    for step in range(num_steps):
        fig = plt.figure(figsize=(11.5, 5.3))
        for panel_idx, method in enumerate(methods, start=1):
            ax = fig.add_subplot(1, 2, panel_idx, projection="3d")
            draw_panel(ax, method, step, show_labels=panel_idx == 1)
        fig.tight_layout()
        fig.savefig(frames_dir / f"{corruption}_step_{step:02d}.png", bbox_inches="tight", dpi=240)
        plt.close(fig)


def plot_stepwise_metrics(output_dir: Path, corruption: str, metric_rows: Sequence[Mapping[str, object]]) -> None:
    methods = ["TTA w/o CBC", "TTA + CBC"]
    colors = {"TTA w/o CBC": "#1f77b4", "TTA + CBC": "#d62728"}
    metrics = [
        ("mean_centroid_text_cosine_x100", "Centroid-text cosine ×100"),
        ("top1", "Top-1 (%)"),
        ("mean_clean_centroid_cosine_x100", "Clean-centroid cosine ×100"),
        ("inter_intra_ratio", "Inter/Intra ratio"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0))
    axes_flat = axes.ravel()
    for ax, (metric, ylabel) in zip(axes_flat, metrics):
        for method in methods:
            rows = [row for row in metric_rows if row["method"] == method]
            steps = [int(row["step"]) for row in rows]
            values = [float(row[metric]) for row in rows]
            ax.plot(steps, values, marker="o", linewidth=2.0, markersize=5.5, color=colors[method], label=method)
        ax.set_xlabel("Adaptation step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.24, linestyle="--", linewidth=0.7)
    axes_flat[0].legend(frameon=True, facecolor="white", edgecolor="0.7", loc="best")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"tta_stepwise_metrics_{corruption}.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    configure_font(args.font_zip)
    output_root = args.output_root
    ensure_dir(output_root)
    device = choose_device(args.device)

    model, transform = return_base_model(args.base_model_name, device, dataset="cifar10c")
    class_dataset = load_cifar10_dataset(args.data_root, transform)
    class_names = list(class_dataset.class_names)
    text_features = encode_text_features(model, class_names, args.template, device)
    text_features_cpu = text_features.detach().cpu()
    base_visual_state = deepcopy(visual_model(model).state_dict())

    metadata = {
        "base_model": args.base_model_name,
        "template": args.template,
        "severity": args.severity,
        "corruptions": args.corruptions,
        "samples_per_class": args.samples_per_class,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "logit_scale": args.logit_scale,
        "elev": args.elev,
        "azim": args.azim,
        "tta_adapter": "controlled STAMP-style visual-LN pseudo-label adapter",
        "note": "CBC is applied only to pseudo-label logits; the final feature trajectory comes from visual encoder updates.",
    }
    write_json(output_root / "metadata.json", metadata)

    all_summary_rows: List[Dict[str, object]] = []
    trainable_names: List[str] = []
    for corruption in args.corruptions:
        print(f"[trajectory] running corruption={corruption}", flush=True)
        corruption_dir = output_root / corruption
        ensure_dir(corruption_dir)
        dataset = load_cifar10c_dataset(args.data_root, transform, corruption, args.severity)
        indices = stratified_indices(dataset, args.samples_per_class, args.seed)
        clean_dataset = load_cifar10_dataset(args.data_root, transform)
        loader_eval = build_loader(dataset, indices, args.batch_size, args.num_workers, device, shuffle=False)
        loader_train = build_loader(dataset, indices, args.batch_size, args.num_workers, device, shuffle=True)
        loader_clean = build_loader(clean_dataset, indices, args.batch_size, args.num_workers, device, shuffle=False)

        model.load_state_dict({"visual." + key: value for key, value in base_visual_state.items()}, strict=False)
        clean_features, _, clean_labels = extract_features_and_logits(model, loader_clean, text_features, device, args.logit_scale)
        clean_centroids = class_centroids(clean_features, clean_labels, len(class_names))
        clean_spread = mean_intraclass_spread(clean_features, clean_labels, len(class_names))
        features, logits, labels = extract_features_and_logits(model, loader_eval, text_features, device, args.logit_scale)
        plot_logit_shift(corruption_dir, corruption, logits, labels, class_names, args.elev, args.azim)

        raw_pseudo_top1 = top1_accuracy(logits, labels)
        cbc_pseudo_top1 = top1_accuracy(center_logits(logits), labels)
        metric_rows, feature_paths, feature_samples, logit_paths, names = run_visual_adapter_trajectory(
            base_model=model,
            base_visual_state=base_visual_state,
            loader_eval=loader_eval,
            loader_train=loader_train,
            text_features=text_features,
            labels=labels,
            text_features_cpu=text_features_cpu,
            clean_centroids_cpu=clean_centroids,
            clean_intraclass_spread=clean_spread,
            device=device,
            corruption=corruption,
            steps=args.steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            logit_scale=args.logit_scale,
        )
        if not trainable_names:
            trainable_names = names
        centroid_rows = plot_feature_trajectory(
            corruption_dir, corruption, feature_paths, clean_centroids, class_names, args.elev, args.azim
        )
        logit_rows = plot_logit_trajectory(corruption_dir, corruption, logit_paths, class_names, args.elev, args.azim)
        plot_feature_distribution_3d_steps(
            corruption_dir, corruption, feature_samples, clean_centroids, labels, class_names, args.elev, args.azim
        )
        plot_stepwise_metrics(corruption_dir, corruption, metric_rows)
        write_csv(corruption_dir / f"trajectory_metrics_{corruption}.csv", metric_rows)
        write_csv(corruption_dir / f"centroid_paths_{corruption}.csv", centroid_rows)
        write_csv(corruption_dir / f"logit_paths_{corruption}.csv", logit_rows)

        first_cbc = next(row for row in metric_rows if row["method"] == "TTA + CBC" and row["step"] == 0)
        last_cbc = next(row for row in metric_rows if row["method"] == "TTA + CBC" and row["step"] == args.steps)
        first_raw = next(row for row in metric_rows if row["method"] == "TTA w/o CBC" and row["step"] == 0)
        last_raw = next(row for row in metric_rows if row["method"] == "TTA w/o CBC" and row["step"] == args.steps)
        all_summary_rows.extend(
            [
                {
                    "corruption": corruption,
                    "method": "CBC pseudo-label shift",
                    "initial_raw_pseudo_top1": raw_pseudo_top1,
                    "initial_cbc_pseudo_top1": cbc_pseudo_top1,
                    "delta": cbc_pseudo_top1 - raw_pseudo_top1,
                },
                {
                    "corruption": corruption,
                    "method": "TTA w/o CBC",
                    "step0_top1": first_raw["top1"],
                    "final_top1": last_raw["top1"],
                    "step0_alignment": first_raw["mean_centroid_text_cosine_x100"],
                    "final_alignment": last_raw["mean_centroid_text_cosine_x100"],
                    "step0_clean_centroid_cosine": first_raw["mean_clean_centroid_cosine_x100"],
                    "final_clean_centroid_cosine": last_raw["mean_clean_centroid_cosine_x100"],
                    "step0_inter_intra": first_raw["inter_intra_ratio"],
                    "final_inter_intra": last_raw["inter_intra_ratio"],
                },
                {
                    "corruption": corruption,
                    "method": "TTA + CBC",
                    "step0_top1": first_cbc["top1"],
                    "final_top1": last_cbc["top1"],
                    "step0_alignment": first_cbc["mean_centroid_text_cosine_x100"],
                    "final_alignment": last_cbc["mean_centroid_text_cosine_x100"],
                    "step0_clean_centroid_cosine": first_cbc["mean_clean_centroid_cosine_x100"],
                    "final_clean_centroid_cosine": last_cbc["mean_clean_centroid_cosine_x100"],
                    "step0_inter_intra": first_cbc["inter_intra_ratio"],
                    "final_inter_intra": last_cbc["inter_intra_ratio"],
                },
            ]
        )

    write_csv(output_root / "summary.csv", all_summary_rows)
    write_json(output_root / "trainable_parameters.json", {"trainable": trainable_names, "count": len(trainable_names)})
    readme = """# CBC + TTA Qualitative Trajectory

- `cbc_logit_shift_3d_*.png`: 3D class-logit evidence shift from CLIP to CBC.
- `tta_logit_trajectory_3d_*.png`: step-wise 3D class-evidence paths toward ideal class evidence.
- `tta_feature_trajectory_3d_*.png`: step-wise 3D feature-centroid paths relative to clean feature centroids.
- `tta_feature_distribution_steps_3d_*.png`: selected-step 3D sample distribution restoration panels.
- `feature_distribution_3d_frames/*.png`: every-step 3D sample distribution frames.
- `tta_stepwise_metrics_*.png`: step-wise top-1, centroid-text alignment, and inter/intra cluster ratio.
- CBC is used only to form pseudo labels; feature movement is produced by visual encoder adaptation.
- The adapter updates CLIP visual LayerNorm parameters to isolate pseudo-label guidance while keeping the experiment stable.
"""
    (output_root / "README.md").write_text(readme)
    print(f"[trajectory] complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
