#!/usr/bin/env python3
"""Run CLIP zero-shot vs default-prompt CBC experiments on CIFAR-10-C."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
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
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from run_closed_set_cifar_benchmarks import DEFAULT_DATA_ROOT, FAMILIES  # noqa: E402
from ttavlm.datasets import CORRUPTIONS, return_train_val_datasets  # noqa: E402
from ttavlm.datasets.cifar10c import load_cifar_c_severity_slice  # noqa: E402
from ttavlm.models import return_base_model  # noqa: E402
from ttavlm.models.clip import tokenize  # noqa: E402


DEFAULT_TEMPLATE = "a photo of a {}"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "work" / "cbc_clip_zero_shot_experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["cifar10"], default="cifar10")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="cbc_default_prompt_clip_zero_shot")
    parser.add_argument("--base-model-name", type=str, default="clip-ViT-B/16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batch-sweep", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-corruption-severity", type=int, default=5)
    parser.add_argument("--mixed-severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--severity-sweep", nargs="+", type=int, default=list(range(0, 11)))
    parser.add_argument("--pca-severity", type=int, default=5)
    parser.add_argument("--pca-batch-size", type=int, default=256)
    parser.add_argument("--max-samples-per-condition", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
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


def choose_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
        device = torch.device("cuda:0" if device_name == "cuda" else device_name)
        torch.cuda.set_device(device)
        return device
    return torch.device(device_name)


def maybe_subset(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(int(max_samples))))


def dataset_attr(dataset: Dataset, name: str):
    if hasattr(dataset, name):
        return getattr(dataset, name)
    if isinstance(dataset, Subset):
        return dataset_attr(dataset.dataset, name)
    raise AttributeError(name)


def batch_to_labels(batch: Mapping[str, object]) -> torch.Tensor:
    labels = batch["target"]
    if isinstance(labels, torch.Tensor):
        return labels.long()
    return torch.as_tensor(labels, dtype=torch.long)


def build_loader(dataset: Dataset, batch_size: int, workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def encode_text_features(
    *,
    model: torch.nn.Module,
    class_names: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    prompts = [DEFAULT_TEMPLATE.format(name.replace("_", " ")) for name in class_names]
    tokens = tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens).float()
        text_features = F.normalize(text_features, dim=-1)
    return text_features.cpu()


def extract_logits(
    *,
    model: torch.nn.Module,
    dataset: Dataset,
    text_features: torch.Tensor,
    batch_size: int,
    workers: int,
    device: torch.device,
    desc: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = build_loader(dataset, batch_size, workers, device)
    text_features_gpu = text_features.to(device)
    logits: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    total = len(dataset)
    seen = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=True)
            batch_labels = batch_to_labels(batch)
            image_features = model.encode_image(images).float()
            image_features = F.normalize(image_features, dim=-1)
            batch_logits = image_features @ text_features_gpu.t()
            logits.append(batch_logits.cpu())
            labels.append(batch_labels.cpu())
            seen += int(batch_labels.numel())
            if batch_index == 0 or seen == total or batch_index % 20 == 0:
                log(f"{desc}: encoded {seen}/{total}")
    return torch.cat(logits, dim=0), torch.cat(labels, dim=0)


def load_dataset(
    *,
    dataset_name: str,
    data_root: Path,
    transform,
    shift_type: str | None = None,
    severity: int | None = None,
    max_samples: int | None = None,
) -> Dataset:
    _, dataset = return_train_val_datasets(
        name=dataset_name,
        data_dir=str(data_root),
        train_transform=transform,
        val_transform=transform,
        shift=shift_type,
        severity=severity,
    )
    return maybe_subset(dataset, max_samples)


def cache_name(condition: str, max_samples: int | None) -> str:
    suffix = "all" if max_samples is None else f"first{max_samples}"
    return f"{condition}_{suffix}.pt"


def load_or_compute_logits(
    *,
    cache_dir: Path,
    condition: str,
    model: torch.nn.Module,
    dataset: Dataset,
    text_features: torch.Tensor,
    batch_size: int,
    workers: int,
    device: torch.device,
    max_samples: int | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ensure_dir(cache_dir)
    cache_path = cache_dir / cache_name(condition, max_samples)
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        log(f"cache hit: {condition}")
        return payload["logits"].float(), payload["labels"].long()

    log(f"cache miss: {condition}")
    logits, labels = extract_logits(
        model=model,
        dataset=dataset,
        text_features=text_features,
        batch_size=batch_size,
        workers=workers,
        device=device,
        desc=condition,
    )
    torch.save({"logits": logits.float(), "labels": labels.long()}, cache_path)
    return logits.float(), labels.long()


def evaluate_clip_cbc(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    seed: int,
) -> Dict[str, object]:
    num_samples = int(labels.numel())
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    order = torch.randperm(num_samples, generator=generator)

    num_classes = int(logits.shape[1])
    running_sum = torch.zeros((1, num_classes), dtype=torch.float64)
    running_count = 0
    raw_pred = logits.argmax(dim=1).long()
    cbc_pred = torch.empty_like(raw_pred)

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

    raw_correct = raw_pred.eq(labels)
    cbc_correct = cbc_pred.eq(labels)
    raw_acc = 100.0 * float(raw_correct.float().mean().item())
    cbc_acc = 100.0 * float(cbc_correct.float().mean().item())
    return {
        "raw_top1": raw_acc,
        "cbc_top1": cbc_acc,
        "delta_top1": cbc_acc - raw_acc,
        "num_samples": num_samples,
        "batch_size": int(batch_size),
        "raw_pred": raw_pred,
        "cbc_pred": cbc_pred,
    }


def strip_preds(metrics: Mapping[str, object]) -> Dict[str, object]:
    return {key: value for key, value in metrics.items() if not key.endswith("_pred")}


def grouped_rows(
    *,
    labels: torch.Tensor,
    group_names: Sequence[str],
    raw_pred: torch.Tensor,
    cbc_pred: torch.Tensor,
    group_key: str,
    base_row: Mapping[str, object],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    groups = np.asarray(group_names)
    for group in sorted(set(group_names)):
        mask_np = groups == group
        mask = torch.from_numpy(mask_np)
        group_labels = labels[mask]
        group_raw = raw_pred[mask]
        group_cbc = cbc_pred[mask]
        raw_acc = 100.0 * float(group_raw.eq(group_labels).float().mean().item())
        cbc_acc = 100.0 * float(group_cbc.eq(group_labels).float().mean().item())
        row = dict(base_row)
        row.update(
            {
                group_key: group,
                "raw_top1": raw_acc,
                "cbc_top1": cbc_acc,
                "delta_top1": cbc_acc - raw_acc,
                "num_samples": int(group_labels.numel()),
            }
        )
        rows.append(row)
    return rows


def plot_metric_lines(
    *,
    rows: Sequence[Mapping[str, object]],
    x_key: str,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str = "Top-1 accuracy (%)",
) -> None:
    ensure_dir(output_path.parent)
    x_values = [float(row[x_key]) for row in rows]
    raw_values = [float(row["raw_top1"]) for row in rows]
    cbc_values = [float(row["cbc_top1"]) for row in rows]
    plt.figure(figsize=(7.3, 4.8))
    plt.plot(x_values, raw_values, marker="o", linewidth=2.0, label="CLIP")
    plt.plot(x_values, cbc_values, marker="o", linewidth=2.0, label="CLIP + CBC")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.28)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_batch_sweep(
    *,
    rows: Sequence[Mapping[str, object]],
    output_path: Path,
    title: str,
) -> None:
    ensure_dir(output_path.parent)
    batch_sizes = [int(row["batch_size"]) for row in rows]
    cbc_values = [float(row["cbc_top1"]) for row in rows]
    raw_baseline = float(rows[0]["raw_top1"])
    plt.figure(figsize=(7.3, 4.8))
    plt.plot(batch_sizes, cbc_values, marker="o", linewidth=2.0, label="CLIP + CBC")
    plt.axhline(raw_baseline, linestyle="--", linewidth=2.0, color="tab:gray", label=f"CLIP ({raw_baseline:.2f})")
    plt.xscale("log", base=2)
    plt.xticks(batch_sizes, [str(size) for size in batch_sizes])
    plt.xlabel("Batch size")
    plt.ylabel("Top-1 accuracy (%)")
    plt.title(title)
    plt.grid(True, alpha=0.28, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def concat_conditions(payloads: Sequence[Tuple[torch.Tensor, torch.Tensor, str]]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    logits = torch.cat([payload[0] for payload in payloads], dim=0)
    labels = torch.cat([payload[1] for payload in payloads], dim=0)
    groups: List[str] = []
    for condition_logits, _, group in payloads:
        groups.extend([group] * int(condition_logits.shape[0]))
    return logits, labels, groups


def compute_single_corruption_rows(
    *,
    corruption_logits: Mapping[str, Tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    seed: int,
    severity: int,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    metrics_by_corruption: Dict[str, Dict[str, object]] = {}
    for corruption in CORRUPTIONS:
        logits, labels = corruption_logits[corruption]
        metrics = evaluate_clip_cbc(logits=logits, labels=labels, batch_size=batch_size, seed=seed)
        metrics_by_corruption[corruption] = metrics
        row = {
            "experiment": "single_corruption",
            "severity": int(severity),
            "corruption": corruption,
            **strip_preds(metrics),
        }
        rows.append(row)
    return rows, metrics_by_corruption


def load_raw_clean_image(data_root: Path, sample_index: int) -> Image.Image:
    dataset = CIFAR10(root=str(data_root / "CIFAR-10"), train=False, download=False)
    return Image.fromarray(dataset.data[int(sample_index)]).convert("RGB")


def load_raw_corruption_image(data_root: Path, corruption: str, severity: int, sample_index: int) -> Image.Image:
    data = load_cifar_c_severity_slice(
        data_root / "CIFAR-10-C",
        corruption,
        int(severity),
        10000,
    )
    return Image.fromarray(data[int(sample_index)]).convert("RGB")


def save_corruption_grid(
    *,
    data_root: Path,
    output_path: Path,
    sample_index: int,
    severity: int,
    class_name: str,
) -> None:
    ensure_dir(output_path.parent)
    entries: List[Tuple[str, Image.Image]] = [("clean", load_raw_clean_image(data_root, sample_index))]
    entries.extend(
        (corruption, load_raw_corruption_image(data_root, corruption, severity, sample_index))
        for corruption in CORRUPTIONS
    )
    cell = 132
    image_size = 96
    label_height = 30
    cols = 4
    rows = int(np.ceil(len(entries) / cols))
    canvas = Image.new("RGB", (cols * cell, rows * cell + 38), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 11)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    draw.text((8, 8), f"CIFAR-10 sample index={sample_index}, label={class_name}, severity={severity}", fill="black", font=title_font)
    y_offset = 38
    for idx, (name, image) in enumerate(entries):
        row = idx // cols
        col = idx % cols
        x = col * cell
        y = y_offset + row * cell
        resized = image.resize((image_size, image_size), resample=Image.Resampling.BICUBIC)
        canvas.paste(resized, (x + (cell - image_size) // 2, y + 6))
        display = name.replace("_", " ")
        draw.text((x + 6, y + image_size + 10), display[:20], fill="black", font=font)
    canvas.save(output_path)


def choose_visualization_sample(
    *,
    metrics_by_corruption: Mapping[str, Mapping[str, object]],
    labels: torch.Tensor,
) -> Dict[str, object]:
    fix_counts = torch.zeros_like(labels, dtype=torch.long)
    regress_counts = torch.zeros_like(labels, dtype=torch.long)
    for metrics in metrics_by_corruption.values():
        raw_pred = metrics["raw_pred"]
        cbc_pred = metrics["cbc_pred"]
        fixed = raw_pred.ne(labels) & cbc_pred.eq(labels)
        regressed = raw_pred.eq(labels) & cbc_pred.ne(labels)
        fix_counts += fixed.long()
        regress_counts += regressed.long()
    net = fix_counts - regress_counts
    sample_index = int(torch.argmax(net).item())
    return {
        "sample_index": sample_index,
        "fix_count": int(fix_counts[sample_index].item()),
        "regress_count": int(regress_counts[sample_index].item()),
        "net_fix_count": int(net[sample_index].item()),
    }


def run_pca_script(
    *,
    output_root: Path,
    data_root: Path,
    corruptions: Sequence[str],
    severity: int,
    batch_size: int,
    workers: int,
    seed: int,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "plot_feature_cluster_pca_3d.py"),
        "--family",
        "cifar10",
        "--severity",
        str(int(severity)),
        "--shift-types",
        *list(corruptions),
        "--data-root",
        str(data_root),
        "--output-root",
        str(output_root),
        "--device",
        "cuda:0",
        "--batch-size",
        str(int(batch_size)),
        "--num-workers",
        str(int(workers)),
        "--seed",
        str(int(seed)),
        "--skip-existing",
    ]
    log("running PCA visualizer: " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_root = args.output_root / args.run_name
    ensure_dir(output_root)
    cache_dir = output_root / "cache" / "logits"
    plots_dir = output_root / "plots"
    tables_dir = output_root / "tables"
    visuals_dir = output_root / "visualizations"

    metadata = {
        "family": args.family,
        "template": DEFAULT_TEMPLATE,
        "cbc_prompt": "default",
        "base_model_name": args.base_model_name,
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "feature_batch_size": int(args.feature_batch_size),
        "mixed_corruption_severity": int(args.mixed_corruption_severity),
        "mixed_severities": [int(value) for value in args.mixed_severities],
        "severity_sweep": [int(value) for value in args.severity_sweep],
        "corruptions": list(CORRUPTIONS),
        "note": "Severity 0 is clean CIFAR-10. Severities above 5 use the repository CIFAR-C extrapolation loader.",
    }
    write_json(output_root / "metadata.json", metadata)

    device = choose_device(args.device)
    log(f"using device={device}; output_root={output_root}")

    family_cfg = FAMILIES[args.family]
    clean_dataset_name = str(family_cfg["clean_dataset"])
    corrupted_dataset_name = str(family_cfg["corrupted_dataset"])

    model, transform = return_base_model(
        name=args.base_model_name,
        device=device,
        dataset=corrupted_dataset_name,
        path_to_weights=str(REPO_ROOT / "work"),
    )
    if hasattr(model, "visual") and not hasattr(model.visual, "use_local"):
        model.visual.use_local = False
    model.eval()

    clean_reference = load_dataset(
        dataset_name=clean_dataset_name,
        data_root=args.data_root,
        transform=transform,
        max_samples=args.max_samples_per_condition,
    )
    class_names = list(dataset_attr(clean_reference, "class_names"))
    text_features = encode_text_features(model=model, class_names=class_names, device=device)

    clean_logits, clean_labels = load_or_compute_logits(
        cache_dir=cache_dir,
        condition="clean",
        model=model,
        dataset=clean_reference,
        text_features=text_features,
        batch_size=args.feature_batch_size,
        workers=args.workers,
        device=device,
        max_samples=args.max_samples_per_condition,
    )

    severity5_payloads: List[Tuple[torch.Tensor, torch.Tensor, str]] = []
    severity5_by_corruption: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for corruption in CORRUPTIONS:
        condition = f"severity{int(args.mixed_corruption_severity)}_{corruption}"
        dataset = load_dataset(
            dataset_name=corrupted_dataset_name,
            data_root=args.data_root,
            transform=transform,
            shift_type=corruption,
            severity=int(args.mixed_corruption_severity),
            max_samples=args.max_samples_per_condition,
        )
        logits, labels = load_or_compute_logits(
            cache_dir=cache_dir,
            condition=condition,
            model=model,
            dataset=dataset,
            text_features=text_features,
            batch_size=args.feature_batch_size,
            workers=args.workers,
            device=device,
            max_samples=args.max_samples_per_condition,
        )
        severity5_payloads.append((logits, labels, corruption))
        severity5_by_corruption[corruption] = (logits, labels)

    log("experiment 1: mixed corruptions")
    mixed_corruption_logits, mixed_corruption_labels, mixed_corruption_groups = concat_conditions(severity5_payloads)
    mixed_corruption_metrics = evaluate_clip_cbc(
        logits=mixed_corruption_logits,
        labels=mixed_corruption_labels,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    mixed_corruption_summary = {
        "experiment": "mixed_corruptions",
        "severity": int(args.mixed_corruption_severity),
        **strip_preds(mixed_corruption_metrics),
    }
    write_json(output_root / "mixed_corruptions_summary.json", mixed_corruption_summary)
    write_csv(tables_dir / "mixed_corruptions_summary.csv", [mixed_corruption_summary])
    write_csv(
        tables_dir / "mixed_corruptions_by_corruption.csv",
        grouped_rows(
            labels=mixed_corruption_labels,
            group_names=mixed_corruption_groups,
            raw_pred=mixed_corruption_metrics["raw_pred"],
            cbc_pred=mixed_corruption_metrics["cbc_pred"],
            group_key="corruption",
            base_row={"experiment": "mixed_corruptions", "severity": int(args.mixed_corruption_severity)},
        ),
    )

    log("experiment 2: mixed severities")
    mixed_severity_payloads: List[Tuple[torch.Tensor, torch.Tensor, str]] = []
    for severity in args.mixed_severities:
        for corruption in CORRUPTIONS:
            condition = f"severity{int(severity)}_{corruption}"
            if int(severity) == int(args.mixed_corruption_severity) and corruption in severity5_by_corruption:
                logits, labels = severity5_by_corruption[corruption]
            else:
                dataset = load_dataset(
                    dataset_name=corrupted_dataset_name,
                    data_root=args.data_root,
                    transform=transform,
                    shift_type=corruption,
                    severity=int(severity),
                    max_samples=args.max_samples_per_condition,
                )
                logits, labels = load_or_compute_logits(
                    cache_dir=cache_dir,
                    condition=condition,
                    model=model,
                    dataset=dataset,
                    text_features=text_features,
                    batch_size=args.feature_batch_size,
                    workers=args.workers,
                    device=device,
                    max_samples=args.max_samples_per_condition,
                )
            mixed_severity_payloads.append((logits, labels, str(int(severity))))

    mixed_severity_logits, mixed_severity_labels, mixed_severity_groups = concat_conditions(mixed_severity_payloads)
    mixed_severity_metrics = evaluate_clip_cbc(
        logits=mixed_severity_logits,
        labels=mixed_severity_labels,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    mixed_severity_summary = {
        "experiment": "mixed_severities",
        "severities": " ".join(str(int(value)) for value in args.mixed_severities),
        **strip_preds(mixed_severity_metrics),
    }
    write_json(output_root / "mixed_severities_summary.json", mixed_severity_summary)
    write_csv(tables_dir / "mixed_severities_summary.csv", [mixed_severity_summary])
    write_csv(
        tables_dir / "mixed_severities_by_severity.csv",
        grouped_rows(
            labels=mixed_severity_labels,
            group_names=mixed_severity_groups,
            raw_pred=mixed_severity_metrics["raw_pred"],
            cbc_pred=mixed_severity_metrics["cbc_pred"],
            group_key="severity",
            base_row={"experiment": "mixed_severities"},
        ),
    )

    log("experiment 3: severity sweep")
    severity_sweep_rows: List[Dict[str, object]] = []
    for severity in args.severity_sweep:
        severity = int(severity)
        if severity == 0:
            logits = clean_logits
            labels = clean_labels
            num_conditions = 1
        elif severity == int(args.mixed_corruption_severity):
            logits, labels, _ = concat_conditions(severity5_payloads)
            num_conditions = len(CORRUPTIONS)
        else:
            payloads: List[Tuple[torch.Tensor, torch.Tensor, str]] = []
            for corruption in CORRUPTIONS:
                condition = f"severity{severity}_{corruption}"
                dataset = load_dataset(
                    dataset_name=corrupted_dataset_name,
                    data_root=args.data_root,
                    transform=transform,
                    shift_type=corruption,
                    severity=severity,
                    max_samples=args.max_samples_per_condition,
                )
                condition_logits, condition_labels = load_or_compute_logits(
                    cache_dir=cache_dir,
                    condition=condition,
                    model=model,
                    dataset=dataset,
                    text_features=text_features,
                    batch_size=args.feature_batch_size,
                    workers=args.workers,
                    device=device,
                    max_samples=args.max_samples_per_condition,
                )
                payloads.append((condition_logits, condition_labels, corruption))
            logits, labels, _ = concat_conditions(payloads)
            num_conditions = len(CORRUPTIONS)

        metrics = evaluate_clip_cbc(logits=logits, labels=labels, batch_size=args.batch_size, seed=args.seed)
        row = {
            "experiment": "severity_sweep",
            "severity": severity,
            "num_conditions": int(num_conditions),
            **strip_preds(metrics),
        }
        severity_sweep_rows.append(row)
        log(f"severity {severity}: CLIP={row['raw_top1']:.3f}, CBC={row['cbc_top1']:.3f}, delta={row['delta_top1']:.3f}")
    write_csv(tables_dir / "severity_sweep.csv", severity_sweep_rows)
    plot_metric_lines(
        rows=severity_sweep_rows,
        x_key="severity",
        output_path=plots_dir / "severity_sweep.png",
        title="CIFAR-10-C severity sweep",
        xlabel="Severity",
    )

    log("experiment 4: batch-size sweep")
    batch_sweep_rows: List[Dict[str, object]] = []
    for batch_size in args.batch_sweep:
        metrics = evaluate_clip_cbc(
            logits=mixed_corruption_logits,
            labels=mixed_corruption_labels,
            batch_size=int(batch_size),
            seed=args.seed,
        )
        row = {
            "experiment": "batch_size_sweep",
            "severity": int(args.mixed_corruption_severity),
            **strip_preds(metrics),
        }
        batch_sweep_rows.append(row)
        log(f"batch {batch_size}: CLIP={row['raw_top1']:.3f}, CBC={row['cbc_top1']:.3f}, delta={row['delta_top1']:.3f}")
    write_csv(tables_dir / "batch_size_sweep.csv", batch_sweep_rows)
    plot_batch_sweep(
        rows=batch_sweep_rows,
        output_path=plots_dir / "batch_size_sweep.png",
        title=f"CIFAR-10-C batch-size sweep, severity {int(args.mixed_corruption_severity)}",
    )

    log("experiment 5: sample grid and PCA corruptions")
    single_rows, single_metrics = compute_single_corruption_rows(
        corruption_logits=severity5_by_corruption,
        batch_size=args.batch_size,
        seed=args.seed,
        severity=int(args.pca_severity),
    )
    write_csv(tables_dir / "single_corruption_severity5.csv", single_rows)
    best_row = max(single_rows, key=lambda row: float(row["delta_top1"]))
    worst_row = min(single_rows, key=lambda row: float(row["delta_top1"]))
    best_corruption = str(best_row["corruption"])
    worst_corruption = str(worst_row["corruption"])
    sample_info = choose_visualization_sample(
        metrics_by_corruption=single_metrics,
        labels=severity5_by_corruption[CORRUPTIONS[0]][1],
    )
    sample_index = int(sample_info["sample_index"])
    sample_label = int(severity5_by_corruption[CORRUPTIONS[0]][1][sample_index].item())
    sample_payload = {
        **sample_info,
        "label_id": sample_label,
        "label_name": class_names[sample_label],
        "severity": int(args.pca_severity),
        "best_corruption": best_corruption,
        "best_delta_top1": float(best_row["delta_top1"]),
        "worst_corruption": worst_corruption,
        "worst_delta_top1": float(worst_row["delta_top1"]),
    }
    write_json(visuals_dir / "sample_and_pca_selection.json", sample_payload)
    save_corruption_grid(
        data_root=args.data_root,
        output_path=visuals_dir / "sample_15_corruptions_grid.png",
        sample_index=sample_index,
        severity=int(args.pca_severity),
        class_name=class_names[sample_label],
    )
    run_pca_script(
        output_root=visuals_dir / "feature_cluster_pca_3d",
        data_root=args.data_root,
        corruptions=[best_corruption, worst_corruption],
        severity=int(args.pca_severity),
        batch_size=int(args.pca_batch_size),
        workers=int(args.workers),
        seed=int(args.seed),
    )

    final_summary = {
        "mixed_corruptions": mixed_corruption_summary,
        "mixed_severities": mixed_severity_summary,
        "severity_sweep_csv": str(tables_dir / "severity_sweep.csv"),
        "severity_sweep_plot": str(plots_dir / "severity_sweep.png"),
        "batch_size_sweep_csv": str(tables_dir / "batch_size_sweep.csv"),
        "batch_size_sweep_plot": str(plots_dir / "batch_size_sweep.png"),
        "sample_grid": str(visuals_dir / "sample_15_corruptions_grid.png"),
        "pca_output_root": str(visuals_dir / "feature_cluster_pca_3d" / "cifar10" / f"severity_{int(args.pca_severity)}"),
        "best_corruption": best_corruption,
        "worst_corruption": worst_corruption,
    }
    write_json(output_root / "final_summary.json", final_summary)
    log("all requested experiments completed successfully")
    log(f"results saved to {output_root}")


if __name__ == "__main__":
    main()
