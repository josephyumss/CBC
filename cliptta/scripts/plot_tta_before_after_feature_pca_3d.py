#!/usr/bin/env python3
"""Plot pre/post-adaptation feature distributions for non-CLIPTTA TTA methods."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import traceback
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

import ttavlm.configuration as config  # noqa: E402
import ttavlm.lib as lib  # noqa: E402
from run_closed_set_cifar_benchmarks import (  # noqa: E402
    BASELINE_BUILDERS,
    DEFAULT_DATA_ROOT,
    EXTRA_BUILDERS,
    FAMILIES,
)
from ttavlm.datasets import return_train_val_datasets  # noqa: E402
from ttavlm.methods import return_tta_model  # noqa: E402
from ttavlm.models import return_base_model  # noqa: E402
from ttavlm.models.clip import tokenize  # noqa: E402
from ttavlm.transforms import TransformList  # noqa: E402


DEFAULT_TEMPLATE = "a photo of a {}"
DEFAULT_METHODS = [
    "clipartt",
    "unient",
    "stamp",
    "tda",
    "adacontrast",
    "watt",
    "watt_otsu",
    "watt_unient",
    "etta",
]
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "complete"
    / "final_paper_experiments"
    / "05_qualitative_analysis"
    / "tta_before_after_feature_pca3d"
)
DEFAULT_FONT_ZIP = WORKSPACE_ROOT / "TimesNewerRoman.zip"
CLASS_COLORS = list(plt.cm.tab10.colors)
NO_VISUAL_ENCODER_UPDATE_METHODS = {
    "tda": "TDA updates positive/negative feature caches, not the visual encoder; feature distribution is expected to remain unchanged.",
    "etta": "ETTA updates a cache-enhanced classifier, not the visual encoder; feature distribution is expected to remain unchanged.",
}
SKIP_ADAPTATION_FOR_FEATURE_PLOT = {"etta"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=list(FAMILIES.keys()), default="cifar10")
    parser.add_argument("--corruptions", nargs="+", default=["brightness", "glass_blur"])
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-model-name", type=str, default="clip-ViT-B/16")
    parser.add_argument("--template", type=str, default=DEFAULT_TEMPLATE)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--extract-batch-size", type=int, default=256)
    parser.add_argument("--adapt-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--samples-per-class", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--point-size", type=float, default=7.0)
    parser.add_argument("--point-alpha-before", type=float, default=0.12)
    parser.add_argument("--point-alpha-after", type=float, default=0.30)
    parser.add_argument("--centroid-size", type=float, default=82.0)
    parser.add_argument("--elev", type=float, default=22.0)
    parser.add_argument("--azim", type=float, default=-58.0)
    parser.add_argument("--font-zip", type=Path, default=DEFAULT_FONT_ZIP)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
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
            "axes.labelsize": 17,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
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
    lib.fix_seed(int(seed))


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
    workers: int,
    device: torch.device,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=generator if shuffle else None,
    )


def batch_images(batch, device: torch.device) -> List[Tensor]:
    images = batch["image"]
    if isinstance(images, (list, tuple)):
        return [img.to(device, non_blocking=True) for img in images]
    return [images.to(device, non_blocking=True)]


def batch_image_tensor(batch) -> Tensor:
    images = batch["image"]
    if isinstance(images, (list, tuple)):
        return images[0]
    return images


@torch.no_grad()
def encode_text_features(model: torch.nn.Module, class_names: Sequence[str], template: str, device: torch.device) -> Tensor:
    prompts = [template.format(name.replace("_", " ")) for name in class_names]
    tokens = tokenize(prompts, truncate=True).to(device)
    text_features = model.encode_text(tokens).float()
    return F.normalize(text_features, dim=-1).cpu()


@torch.no_grad()
def extract_clip_features(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[Tensor, Tensor]:
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
def extract_tta_features(tta_model, loader: DataLoader, device: torch.device) -> Tuple[Tensor, Tensor]:
    tta_model.eval()
    features: List[Tensor] = []
    labels: List[Tensor] = []
    for batch in loader:
        images = batch_images(batch, device)
        outputs = tta_model.get_features(images)[0].float()
        outputs = F.normalize(outputs, dim=-1)
        features.append(outputs.cpu())
        labels.append(batch["target"].long().cpu())
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def class_centroids(features: Tensor, labels: Tensor, num_classes: int) -> Tensor:
    centroids: List[Tensor] = []
    for class_id in range(num_classes):
        mask = labels == class_id
        if mask.any():
            centroids.append(features[mask].mean(dim=0))
        else:
            centroids.append(torch.full_like(features[0], float("nan")))
    return torch.stack(centroids, dim=0)


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
    radius = max(float((maxs - mins).max()) * 0.55, 1.0e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def style_axis(ax, elev: float, azim: float) -> None:
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("PC1", labelpad=9)
    ax.set_ylabel("PC2", labelpad=9)
    ax.set_zlabel("PC3", labelpad=9)
    ax.grid(alpha=0.20, linestyle="--", linewidth=0.6)
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
        alpha=0.62,
        linewidth=0.9,
        arrow_length_ratio=0.16,
        normalize=False,
    )


def plain_top1(features: Tensor, labels: Tensor, text_features: Tensor) -> float:
    logits = features.float() @ text_features.float().t()
    return 100.0 * float(logits.argmax(dim=1).eq(labels).float().mean().item())


def plot_before_after(
    *,
    output_dir: Path,
    method: str,
    corruption: str,
    severity: int,
    class_names: Sequence[str],
    before_features: Tensor,
    after_features: Tensor,
    labels: Tensor,
    clean_centroids: Tensor,
    text_features: Tensor,
    point_size: float,
    point_alpha_before: float,
    point_alpha_after: float,
    centroid_size: float,
    elev: float,
    azim: float,
) -> List[Dict[str, object]]:
    num_classes = len(class_names)
    before_centroids = class_centroids(before_features, labels, num_classes)
    after_centroids = class_centroids(after_features, labels, num_classes)

    arrays = [
        before_features.numpy(),
        after_features.numpy(),
        before_centroids.numpy(),
        after_centroids.numpy(),
        clean_centroids.numpy(),
        text_features.numpy(),
    ]
    pca_mean, pca_basis = fit_pca_3d(arrays)
    before_3d = project(before_features.numpy(), pca_mean, pca_basis)
    after_3d = project(after_features.numpy(), pca_mean, pca_basis)
    before_centroids_3d = project(before_centroids.numpy(), pca_mean, pca_basis)
    after_centroids_3d = project(after_centroids.numpy(), pca_mean, pca_basis)
    clean_centroids_3d = project(clean_centroids.numpy(), pca_mean, pca_basis)
    text_3d = project(text_features.numpy(), pca_mean, pca_basis)
    labels_np = labels.numpy()
    all_points = np.concatenate(
        [before_3d, after_3d, before_centroids_3d, after_centroids_3d, clean_centroids_3d, text_3d],
        axis=0,
    )

    fig = plt.figure(figsize=(9.2, 7.4))
    ax = fig.add_subplot(111, projection="3d")
    for class_id, class_name in enumerate(class_names):
        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
        mask = labels_np == class_id
        if np.any(mask):
            ax.scatter(
                before_3d[mask, 0],
                before_3d[mask, 1],
                before_3d[mask, 2],
                marker="o",
                s=point_size,
                color=color,
                alpha=point_alpha_before,
                edgecolors="none",
                rasterized=True,
            )
            ax.scatter(
                after_3d[mask, 0],
                after_3d[mask, 1],
                after_3d[mask, 2],
                marker="^",
                s=point_size * 1.05,
                color=color,
                alpha=point_alpha_after,
                edgecolors="none",
                rasterized=True,
            )
        ax.scatter(
            before_centroids_3d[class_id, 0],
            before_centroids_3d[class_id, 1],
            before_centroids_3d[class_id, 2],
            marker="o",
            s=centroid_size,
            color=color,
            alpha=0.65,
            edgecolor="black",
            linewidth=0.35,
        )
        ax.scatter(
            after_centroids_3d[class_id, 0],
            after_centroids_3d[class_id, 1],
            after_centroids_3d[class_id, 2],
            marker="D",
            s=centroid_size * 0.92,
            color=color,
            alpha=0.98,
            edgecolor="black",
            linewidth=0.45,
        )
        draw_arrow(ax, before_centroids_3d[class_id], after_centroids_3d[class_id], color)
        ax.scatter(
            text_3d[class_id, 0],
            text_3d[class_id, 1],
            text_3d[class_id, 2],
            marker="*",
            s=centroid_size * 1.22,
            color=color,
            alpha=0.95,
            edgecolor="black",
            linewidth=0.35,
        )
        ax.text(
            after_centroids_3d[class_id, 0],
            after_centroids_3d[class_id, 1],
            after_centroids_3d[class_id, 2],
            class_name.replace("_", " "),
            fontsize=8,
        )

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=7, label="Before adaptation features"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markersize=7, label="After adaptation features"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=8, label="After class centroid"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=10, label="Text prototype"),
    ]
    ax.legend(handles=handles, frameon=True, facecolor="white", edgecolor="0.7", loc="best")
    set_3d_limits(ax, all_points)
    style_axis(ax, elev=elev, azim=azim)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{method}_{corruption}_before_after_features_pca3d.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    rows: List[Dict[str, object]] = []
    for class_id, class_name in enumerate(class_names):
        rows.append(
            {
                "method": method,
                "corruption": corruption,
                "severity": int(severity),
                "class_id": class_id,
                "class_name": class_name,
                "before_centroid_pc1": float(before_centroids_3d[class_id, 0]),
                "before_centroid_pc2": float(before_centroids_3d[class_id, 1]),
                "before_centroid_pc3": float(before_centroids_3d[class_id, 2]),
                "after_centroid_pc1": float(after_centroids_3d[class_id, 0]),
                "after_centroid_pc2": float(after_centroids_3d[class_id, 1]),
                "after_centroid_pc3": float(after_centroids_3d[class_id, 2]),
                "clean_centroid_pc1": float(clean_centroids_3d[class_id, 0]),
                "clean_centroid_pc2": float(clean_centroids_3d[class_id, 1]),
                "clean_centroid_pc3": float(clean_centroids_3d[class_id, 2]),
                "text_pc1": float(text_3d[class_id, 0]),
                "text_pc2": float(text_3d[class_id, 1]),
                "text_pc3": float(text_3d[class_id, 2]),
                "before_to_after_pca_l2": float(np.linalg.norm(after_centroids_3d[class_id] - before_centroids_3d[class_id])),
                "after_to_clean_pca_l2": float(np.linalg.norm(clean_centroids_3d[class_id] - after_centroids_3d[class_id])),
                "before_to_clean_pca_l2": float(np.linalg.norm(clean_centroids_3d[class_id] - before_centroids_3d[class_id])),
                "after_to_text_pca_l2": float(np.linalg.norm(text_3d[class_id] - after_centroids_3d[class_id])),
                "before_to_text_pca_l2": float(np.linalg.norm(text_3d[class_id] - before_centroids_3d[class_id])),
            }
        )
    write_csv(output_dir / f"{method}_{corruption}_centroids_pca3d.csv", rows)
    return rows


def fresh_ttavlm_args(argv: Sequence[str]) -> argparse.Namespace:
    saved_argv = sys.argv
    try:
        sys.argv = ["ttavlm.main", *argv]
        return config.argparser()
    finally:
        sys.argv = saved_argv


def method_tokens(method: str, family: str, family_cfg: Mapping[str, object]) -> List[str]:
    builders = dict(BASELINE_BUILDERS)
    builders.update(EXTRA_BUILDERS)
    if method not in builders:
        available = ", ".join(sorted(builders.keys()))
        raise ValueError(f"Unknown method {method!r}. Available: {available}")
    return builders[method](family, dict(family_cfg), True)


def method_batch_size(method: str, default_batch_size: int, tokens: Sequence[str]) -> int:
    batch_size = 1 if method == "tda" else int(default_batch_size)
    for idx, token in enumerate(tokens):
        if token == "--batch_size" and idx + 1 < len(tokens):
            batch_size = int(tokens[idx + 1])
    return batch_size


def build_runtime_args(
    *,
    args: argparse.Namespace,
    method: str,
    family: str,
    family_cfg: Mapping[str, object],
    dataset_name: str,
    corruption: str,
    save_root: Path,
) -> Tuple[argparse.Namespace, List[str], int]:
    tokens = method_tokens(method, family, family_cfg)
    batch_size = method_batch_size(method, args.adapt_batch_size, tokens)
    argv = [
        "--exp_name",
        f"{method}_{dataset_name}_{corruption}_before_after_pca3d",
        "--env",
        "closed_set_cifar",
        "--dataroot",
        str(args.data_root),
        "--save_root",
        str(save_root),
        "--dataset",
        dataset_name,
        "--base_model_name",
        args.base_model_name,
        "--steps",
        "10",
        "--workers",
        str(args.num_workers),
        "--batch_size",
        str(batch_size),
        "--seeds",
        str(args.seed),
        "--closed_set",
        "--shift_type",
        corruption,
        "--severity",
        str(args.severity),
        *tokens,
    ]
    runtime_args = fresh_ttavlm_args(argv)
    runtime_args.display_progress = False
    runtime_args.eval_max_batches = None
    runtime_args.max_iter = max(1, math.ceil(10000 / max(int(runtime_args.batch_size), 1)))
    runtime_args.use_tta = getattr(runtime_args, "use_tta", False)
    return runtime_args, tokens, int(runtime_args.batch_size)


def adapt_over_loader(tta_model, loader: DataLoader, device: torch.device) -> int:
    tta_model.train()
    steps = 0
    for batch in loader:
        images = batch_images(batch, device)
        labels = batch["target"].long().to(device, non_blocking=True)
        tta_model.forward(images, labels=labels)
        steps += 1
    return steps


def run_cache_only_method(
    *,
    method: str,
    before_features: Tensor,
    before_labels: Tensor,
) -> Tuple[Tensor, Tensor, Dict[str, object]]:
    return (
        before_features.clone(),
        before_labels.clone(),
        {
            "feature_update_type": "cache_only",
            "cache_only_note": NO_VISUAL_ENCODER_UPDATE_METHODS[method],
            "adaptation_batches": 0,
        },
    )


def run_repo_tta_method(
    *,
    args: argparse.Namespace,
    method: str,
    family: str,
    family_cfg: Mapping[str, object],
    dataset_name: str,
    corruption: str,
    class_names: Sequence[str],
    dataset: Dataset,
    indices: Sequence[int],
    device: torch.device,
    output_dir: Path,
) -> Tuple[Tensor, Tensor, Dict[str, object]]:
    runtime_args, tokens, batch_size = build_runtime_args(
        args=args,
        method=method,
        family=family,
        family_cfg=family_cfg,
        dataset_name=dataset_name,
        corruption=corruption,
        save_root=output_dir / "artifacts",
    )
    model, _ = return_base_model(
        name=args.base_model_name,
        device=device,
        dataset=dataset_name,
        path_to_weights=str(REPO_ROOT / "work"),
    )
    tta_model = return_tta_model(method, model, runtime_args, [args.template], list(class_names))
    tta_model.class_names = list(class_names)
    tta_model.reset()

    adapt_loader = build_loader(
        dataset,
        indices,
        batch_size=batch_size,
        workers=args.num_workers,
        device=device,
        shuffle=True,
        seed=args.seed,
    )
    eval_loader = build_loader(
        dataset,
        indices,
        batch_size=args.extract_batch_size,
        workers=args.num_workers,
        device=device,
        shuffle=False,
        seed=args.seed,
    )
    adaptation_batches = adapt_over_loader(tta_model, adapt_loader, device)
    after_features, after_labels = extract_tta_features(tta_model, eval_loader, device)
    meta = {
        "feature_update_type": "visual_encoder_or_method_state",
        "adaptation_batches": int(adaptation_batches),
        "runtime_batch_size": int(batch_size),
        "runtime_steps": int(runtime_args.steps),
        "method_cli_tokens": list(tokens),
    }
    if method in NO_VISUAL_ENCODER_UPDATE_METHODS:
        meta["feature_update_type"] = "cache_only"
        meta["cache_only_note"] = NO_VISUAL_ENCODER_UPDATE_METHODS[method]
    del tta_model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return after_features, after_labels, meta


def main() -> None:
    args = parse_args()
    invalid_methods = [method for method in args.methods if method == "cliptta"]
    if invalid_methods:
        raise SystemExit("This script intentionally excludes cliptta. Remove cliptta from --methods.")

    set_seed(args.seed)
    configure_font(args.font_zip)
    device = choose_device(args.device)
    output_root = args.output_root
    ensure_dir(output_root)

    family_cfg = FAMILIES[args.family]
    clean_name = str(family_cfg["clean_dataset"])
    corrupt_name = str(family_cfg["corrupted_dataset"])

    base_model, base_transform = return_base_model(
        name=args.base_model_name,
        device=device,
        dataset=corrupt_name,
        path_to_weights=str(REPO_ROOT / "work"),
    )
    base_model.eval()

    _, clean_dataset = return_train_val_datasets(
        name=clean_name,
        data_dir=str(args.data_root),
        train_transform=TransformList([base_transform]),
        val_transform=TransformList([base_transform]),
    )
    class_names = list(clean_dataset.class_names)
    text_features = encode_text_features(base_model, class_names, args.template, device)
    indices = stratified_indices(clean_dataset, args.samples_per_class, args.seed)
    clean_loader = build_loader(
        clean_dataset,
        indices,
        batch_size=args.extract_batch_size,
        workers=args.num_workers,
        device=device,
        shuffle=False,
        seed=args.seed,
    )
    clean_features, clean_labels = extract_clip_features(base_model, clean_loader, device)
    clean_centroids = class_centroids(clean_features, clean_labels, len(class_names))

    metadata = {
        "family": args.family,
        "clean_dataset": clean_name,
        "corrupted_dataset": corrupt_name,
        "corruptions": list(args.corruptions),
        "severity": int(args.severity),
        "methods": list(args.methods),
        "excluded_method": "cliptta",
        "template": args.template,
        "base_model_name": args.base_model_name,
        "samples_per_class": int(args.samples_per_class),
        "num_samples": len(indices),
        "seed": int(args.seed),
        "no_visual_encoder_update_methods": NO_VISUAL_ENCODER_UPDATE_METHODS,
    }
    write_json(output_root / "metadata.json", metadata)

    summary_rows: List[Dict[str, object]] = []
    centroid_rows_all: List[Dict[str, object]] = []

    for corruption in args.corruptions:
        print(f"[tta-pca] corruption={corruption}", flush=True)
        _, corrupt_dataset = return_train_val_datasets(
            name=corrupt_name,
            data_dir=str(args.data_root),
            train_transform=TransformList([base_transform]),
            val_transform=TransformList([base_transform]),
            shift=corruption,
            severity=int(args.severity),
        )
        eval_loader = build_loader(
            corrupt_dataset,
            indices,
            batch_size=args.extract_batch_size,
            workers=args.num_workers,
            device=device,
            shuffle=False,
            seed=args.seed,
        )
        before_features, before_labels = extract_clip_features(base_model, eval_loader, device)
        if before_labels.shape != clean_labels.shape or not torch.equal(before_labels, clean_labels):
            raise ValueError(f"Label mismatch between clean and corrupted subset for {corruption}.")

        for method in args.methods:
            print(f"[tta-pca]   method={method}", flush=True)
            method_dir = output_root / corruption / method
            ensure_dir(method_dir)
            summary_path = method_dir / "summary.json"
            if args.skip_existing and summary_path.exists():
                payload = json.loads(summary_path.read_text())
                summary_rows.append(payload)
                continue

            try:
                if method in SKIP_ADAPTATION_FOR_FEATURE_PLOT:
                    after_features, after_labels, method_meta = run_cache_only_method(
                        method=method,
                        before_features=before_features,
                        before_labels=before_labels,
                    )
                else:
                    after_features, after_labels, method_meta = run_repo_tta_method(
                        args=args,
                        method=method,
                        family=args.family,
                        family_cfg=family_cfg,
                        dataset_name=corrupt_name,
                        corruption=corruption,
                        class_names=class_names,
                        dataset=corrupt_dataset,
                        indices=indices,
                        device=device,
                        output_dir=method_dir,
                    )

                if after_labels.shape != before_labels.shape or not torch.equal(after_labels, before_labels):
                    raise ValueError(f"Label mismatch after adaptation for {method}/{corruption}.")

                rows = plot_before_after(
                    output_dir=method_dir,
                    method=method,
                    corruption=corruption,
                    severity=int(args.severity),
                    class_names=class_names,
                    before_features=before_features,
                    after_features=after_features,
                    labels=before_labels,
                    clean_centroids=clean_centroids,
                    text_features=text_features,
                    point_size=args.point_size,
                    point_alpha_before=args.point_alpha_before,
                    point_alpha_after=args.point_alpha_after,
                    centroid_size=args.centroid_size,
                    elev=args.elev,
                    azim=args.azim,
                )
                centroid_rows_all.extend(rows)

                feature_delta = torch.norm(after_features - before_features, dim=1)
                before_centroids = class_centroids(before_features, before_labels, len(class_names))
                after_centroids = class_centroids(after_features, after_labels, len(class_names))
                centroid_delta = torch.norm(after_centroids - before_centroids, dim=1)
                summary = {
                    "status": "success",
                    "method": method,
                    "corruption": corruption,
                    "severity": int(args.severity),
                    "num_samples": int(before_features.shape[0]),
                    "mean_feature_l2_delta": float(feature_delta.mean().item()),
                    "max_feature_l2_delta": float(feature_delta.max().item()),
                    "mean_class_centroid_l2_delta": float(centroid_delta.mean().item()),
                    "max_class_centroid_l2_delta": float(centroid_delta.max().item()),
                    "plain_text_head_top1_before": plain_top1(before_features, before_labels, text_features),
                    "plain_text_head_top1_after": plain_top1(after_features, after_labels, text_features),
                    "output_png": str(method_dir / f"{method}_{corruption}_before_after_features_pca3d.png"),
                    "output_pdf": str(method_dir / f"{method}_{corruption}_before_after_features_pca3d.pdf"),
                    "centroid_csv": str(method_dir / f"{method}_{corruption}_centroids_pca3d.csv"),
                    **method_meta,
                }
                write_json(summary_path, summary)
                summary_rows.append(summary)
            except Exception as exc:  # noqa: BLE001
                failure = {
                    "status": "failed",
                    "method": method,
                    "corruption": corruption,
                    "severity": int(args.severity),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                write_json(summary_path, failure)
                summary_rows.append(failure)
                print(f"[tta-pca][failed] {method}/{corruption}: {exc}", flush=True)
                if args.stop_on_error:
                    raise
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    write_csv(output_root / "summary.csv", summary_rows)
    write_csv(output_root / "all_centroids_pca3d.csv", centroid_rows_all)
    readme = """# TTA Before/After Feature PCA-3D

This folder visualizes feature distributions before and after adaptation for the non-CLIPTTA TTA methods used in the paper experiments.

- Circles: corrupted features before adaptation.
- Triangles: features after adaptation.
- Diamonds: class centroids after adaptation.
- Stars: default-prompt text prototypes.
- Arrows: class-centroid movement from before to after adaptation.
- TDA and ETTA are cache-based methods; they do not update the CLIP visual encoder, so their feature distributions are expected to overlap before/after.
"""
    (output_root / "README.md").write_text(readme)
    print(f"[tta-pca] complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
