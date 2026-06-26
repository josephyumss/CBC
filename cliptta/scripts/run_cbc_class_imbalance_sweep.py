#!/usr/bin/env python3
"""Evaluate CLIP and CLIP + CBC under per-corruption CIFAR-10 class imbalance."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ttavlm.datasets import CORRUPTIONS  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "work" / "cbc_clip_zero_shot_experiments"
DEFAULT_SOURCE_RUN = DEFAULT_OUTPUT_ROOT / "cbc_defaultprompt_claim_cifar10_20260615"
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="cbc_class_imbalance_cifar10c_s5")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-classes", nargs="+", default=["all"])
    parser.add_argument("--imbalance-pcts", nargs="+", type=float, default=[0, 25, 50, 75, 100])
    return parser.parse_args()


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


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


def cache_path(source_run: Path, severity: int, corruption: str) -> Path:
    return source_run / "cache" / "logits" / f"severity{severity}_{corruption}_all.pt"


def load_corruption_logits(source_run: Path, severity: int) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    payloads: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for corruption in CORRUPTIONS:
        path = cache_path(source_run, severity, corruption)
        if not path.exists():
            raise FileNotFoundError(f"Missing cached logits: {path}")
        payload = torch.load(path, map_location="cpu")
        logits = payload["logits"].float()
        labels = payload["labels"].long()
        payloads[corruption] = (logits, labels)
        log(f"loaded {corruption}: {labels.numel()} samples")
    return payloads


def resolve_target_classes(values: Sequence[str]) -> List[int]:
    if any(value.strip().lower() == "all" for value in values):
        return list(range(len(CIFAR10_CLASSES)))
    lookup = {name: index for index, name in enumerate(CIFAR10_CLASSES)}
    class_indices: List[int] = []
    for value in values:
        if value.isdigit():
            class_index = int(value)
        else:
            normalized = value.strip().replace("_", " ").lower()
            if normalized not in lookup:
                raise ValueError(f"Unknown CIFAR-10 class: {value}")
            class_index = lookup[normalized]
        if class_index < 0 or class_index >= len(CIFAR10_CLASSES):
            raise ValueError(f"CIFAR-10 class index out of range: {class_index}")
        class_indices.append(class_index)
    return class_indices


def class_counts(labels: torch.Tensor) -> Dict[int, int]:
    return {
        class_index: int(labels.eq(class_index).sum().item())
        for class_index in range(len(CIFAR10_CLASSES))
    }


def select_target_dominant_indices(
    *,
    labels: torch.Tensor,
    target_class: int,
    imbalance_pct: float,
    seed: int,
    corruption_index: int,
) -> torch.Tensor:
    """Keep all target samples and downsample non-target classes within one corruption.

    0% is the original per-corruption 10,000-image set. 100% keeps only the
    selected target class, yielding 1,000 images for CIFAR-10-C.
    """

    keep_fraction = max(0.0, min(1.0, 1.0 - float(imbalance_pct) / 100.0))
    generator = torch.Generator()
    generator.manual_seed(
        int(seed)
        + int(corruption_index) * 100_003
        + int(target_class) * 1_009
        + int(round(float(imbalance_pct) * 10))
    )
    selected: List[torch.Tensor] = []
    for class_index in range(len(CIFAR10_CLASSES)):
        class_indices = torch.nonzero(labels.eq(class_index), as_tuple=False).flatten()
        if class_index == target_class:
            keep_count = int(class_indices.numel())
        else:
            keep_count = int(round(float(class_indices.numel()) * keep_fraction))
        if keep_count <= 0:
            continue
        perm = torch.randperm(int(class_indices.numel()), generator=generator)[:keep_count]
        selected.append(class_indices[perm])
    indices = torch.cat(selected, dim=0)
    return torch.sort(indices).values


def evaluate_clip_cbc(logits: torch.Tensor, labels: torch.Tensor, batch_size: int, seed: int) -> Dict[str, object]:
    num_samples = int(labels.numel())
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    order = torch.randperm(num_samples, generator=generator)

    num_classes = int(logits.shape[1])
    raw_pred = logits.argmax(dim=1).long()
    cbc_pred = torch.empty_like(raw_pred)
    running_sum = torch.zeros((1, num_classes), dtype=torch.float64)
    running_count = 0

    for start in range(0, num_samples, int(batch_size)):
        idx = order[start : start + int(batch_size)]
        batch_logits = logits[idx].float()
        if batch_logits.shape[0] > 1:
            center = batch_logits.mean(dim=0, keepdim=True)
        elif running_count > 0:
            center = (running_sum / float(running_count)).to(dtype=batch_logits.dtype)
        else:
            center = torch.zeros((1, num_classes), dtype=batch_logits.dtype)
        cbc_pred[idx] = (batch_logits - center).argmax(dim=1).long()
        running_sum += batch_logits.double().sum(dim=0, keepdim=True)
        running_count += int(batch_logits.shape[0])

    raw_top1 = 100.0 * float(raw_pred.eq(labels).float().mean().item())
    cbc_top1 = 100.0 * float(cbc_pred.eq(labels).float().mean().item())
    return {
        "clip_top1": raw_top1,
        "cbc_top1": cbc_top1,
        "delta_top1": cbc_top1 - raw_top1,
        "num_samples": num_samples,
    }


def summarize_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    group_keys: Sequence[str],
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], List[Mapping[str, object]]] = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        grouped.setdefault(key, []).append(row)

    summaries: List[Dict[str, object]] = []
    for key, matching in sorted(grouped.items(), key=lambda item: item[0]):
        clip_values = np.asarray([float(row["clip_top1"]) for row in matching], dtype=np.float64)
        cbc_values = np.asarray([float(row["cbc_top1"]) for row in matching], dtype=np.float64)
        delta_values = np.asarray([float(row["delta_top1"]) for row in matching], dtype=np.float64)
        shares = np.asarray([float(row["target_share_pct"]) for row in matching], dtype=np.float64)
        counts = np.asarray([float(row["num_samples"]) for row in matching], dtype=np.float64)
        summary = {group_key: group_value for group_key, group_value in zip(group_keys, key)}
        summary.update(
            {
                "mean_target_share_pct": float(shares.mean()),
                "mean_clip_top1": float(clip_values.mean()),
                "std_clip_top1": float(clip_values.std(ddof=0)),
                "mean_cbc_top1": float(cbc_values.mean()),
                "std_cbc_top1": float(cbc_values.std(ddof=0)),
                "mean_delta_top1": float(delta_values.mean()),
                "std_delta_top1": float(delta_values.std(ddof=0)),
                "mean_num_samples": float(counts.mean()),
                "num_conditions": len(matching),
            }
        )
        summaries.append(summary)
    return summaries


def plot_average(rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    x_values = [float(row["imbalance_pct"]) for row in rows]
    clip_values = [float(row["mean_clip_top1"]) for row in rows]
    cbc_values = [float(row["mean_cbc_top1"]) for row in rows]
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(x_values, clip_values, marker="o", linewidth=2.0, label="CLIP")
    plt.plot(x_values, cbc_values, marker="o", linewidth=2.0, label="CLIP + CBC")
    plt.xlabel("Target-class imbalance (%)")
    plt.ylabel("Top-1 accuracy (%)")
    plt.title("CIFAR-10-C severity 5 per-corruption class-imbalance sweep")
    plt.grid(True, alpha=0.28)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_root = args.output_root / args.run_name
    tables_dir = output_root / "tables"
    plots_dir = output_root / "plots"
    ensure_dir(output_root)

    log(f"output_root={output_root}")
    if torch.cuda.is_available():
        log(f"cuda visible; device count={torch.cuda.device_count()}; current={torch.cuda.current_device()}")
    else:
        log("cuda not available; using cached CPU logits only")

    corruption_payloads = load_corruption_logits(args.source_run, int(args.severity))
    target_classes = resolve_target_classes(args.target_classes)
    original_counts = {
        corruption: {CIFAR10_CLASSES[key]: value for key, value in class_counts(labels).items()}
        for corruption, (_, labels) in corruption_payloads.items()
    }

    metadata = {
        "dataset": "CIFAR-10-C",
        "severity": int(args.severity),
        "corruptions": list(CORRUPTIONS),
        "source_run": str(args.source_run),
        "template": "a photo of a {}",
        "methods": ["CLIP", "CLIP + CBC"],
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "target_classes": [CIFAR10_CLASSES[index] for index in target_classes],
        "imbalance_pcts": [float(value) for value in args.imbalance_pcts],
        "imbalance_definition": "For each corruption independently, keep all samples from one target class and downsample every non-target class by the imbalance percentage; 0% is the original 10,000-image corruption set and 100% is target-class only (1,000 images).",
        "original_class_counts_by_corruption": original_counts,
    }
    write_json(output_root / "metadata.json", metadata)

    rows: List[Dict[str, object]] = []
    for corruption_index, corruption in enumerate(CORRUPTIONS):
        logits, labels = corruption_payloads[corruption]
        for target_class in target_classes:
            target_name = CIFAR10_CLASSES[target_class]
            for imbalance_pct in args.imbalance_pcts:
                log(f"evaluating corruption={corruption}, target={target_name}, imbalance={imbalance_pct:g}%")
                indices = select_target_dominant_indices(
                    labels=labels,
                    target_class=target_class,
                    imbalance_pct=float(imbalance_pct),
                    seed=int(args.seed),
                    corruption_index=corruption_index,
                )
                subset_logits = logits[indices]
                subset_labels = labels[indices]
                metrics = evaluate_clip_cbc(
                    logits=subset_logits,
                    labels=subset_labels,
                    batch_size=int(args.batch_size),
                    seed=int(args.seed),
                )
                target_share = 100.0 * float(subset_labels.eq(target_class).float().mean().item())
                row = {
                    "corruption": corruption,
                    "target_class": target_name,
                    "target_class_index": int(target_class),
                    "imbalance_pct": float(imbalance_pct),
                    "target_share_pct": target_share,
                    "clip_top1": metrics["clip_top1"],
                    "cbc_top1": metrics["cbc_top1"],
                    "delta_top1": metrics["delta_top1"],
                    "num_samples": metrics["num_samples"],
                    "batch_size": int(args.batch_size),
                }
                rows.append(row)
                log(
                    f"done corruption={corruption}, target={target_name}, imbalance={imbalance_pct:g}%: "
                    f"CLIP={float(metrics['clip_top1']):.3f}, CBC={float(metrics['cbc_top1']):.3f}, "
                    f"target_share={target_share:.1f}%, n={int(metrics['num_samples'])}"
                )
                write_csv(tables_dir / "class_imbalance_by_corruption_target.csv", rows)

    average_rows = summarize_rows(rows, group_keys=["imbalance_pct"])
    by_corruption_rows = summarize_rows(rows, group_keys=["corruption", "imbalance_pct"])
    by_target_rows = summarize_rows(rows, group_keys=["target_class", "target_class_index", "imbalance_pct"])
    write_csv(tables_dir / "class_imbalance_average.csv", average_rows)
    write_csv(tables_dir / "class_imbalance_by_corruption.csv", by_corruption_rows)
    write_csv(tables_dir / "class_imbalance_by_target.csv", by_target_rows)
    write_json(
        output_root / "final_summary.json",
        {
            "average_rows": average_rows,
            "by_corruption_rows": by_corruption_rows,
            "by_target_rows": by_target_rows,
            "by_corruption_target_rows": rows,
        },
    )
    plot_average(average_rows, plots_dir / "class_imbalance_sweep.png")
    log("per-corruption class imbalance sweep complete")
    log(f"average table: {tables_dir / 'class_imbalance_average.csv'}")


if __name__ == "__main__":
    main()
