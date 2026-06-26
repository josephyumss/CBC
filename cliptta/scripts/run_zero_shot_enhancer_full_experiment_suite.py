#!/usr/bin/env python3
"""Run CLIP/CBC and zero-shot enhancement baselines across CBC experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "work" / ".mplconfig"),
)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_zero_shot_enhancement_baselines import (  # noqa: E402
    DEFAULT_EXTERNAL_ROOT,
    build_loader,
    batch_to_labels,
    choose_device,
    cupl_prompts,
    default_prompts,
    encode_prompt_list,
    evaluate_cbc,
    evaluate_frolic,
    evaluate_inmap,
    evaluate_logits,
    load_cifar10c_dataset,
    log,
    text_features_from_prompts,
    waffle_prompts,
)
from run_closed_set_cifar_benchmarks import DEFAULT_DATA_ROOT, FAMILIES  # noqa: E402
from ttavlm.datasets import CORRUPTIONS, return_train_val_datasets  # noqa: E402
from ttavlm.models import return_base_model  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "work" / "zero_shot_enhancement_full_suite"
DEFAULT_TEMPLATE = "a photo of a {}"
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
    parser.add_argument("--family", choices=["cifar10"], default="cifar10")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="cifar10_zero_shot_enhancer_full_suite")
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--base-model-name", type=str, default="clip-ViT-B/16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--severity-sweep", nargs="+", type=int, default=list(range(0, 11)))
    parser.add_argument("--batch-sweep", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    parser.add_argument("--imbalance-pcts", nargs="+", type=float, default=[0, 25, 50, 75, 100])
    parser.add_argument("--skip-imbalance", action="store_true")
    parser.add_argument("--include-transductive-imbalance", action="store_true")
    parser.add_argument("--inmap-iters", type=int, default=2000)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument("--inmap-lr", type=float, default=10.0)
    parser.add_argument("--inmap-tau-t", type=float, default=0.01)
    parser.add_argument("--inmap-tau-i", type=float, default=0.04)
    parser.add_argument("--inmap-alpha", type=float, default=0.6)
    parser.add_argument("--waffle-count", type=int, default=15)
    parser.add_argument("--waffle-reps", type=int, default=7)
    parser.add_argument("--calip-beta2", type=float, default=2.0)
    parser.add_argument("--calip-beta3", type=float, default=0.1)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


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


def load_clean_dataset(data_root: Path, transform) -> Dataset:
    _, dataset = return_train_val_datasets(
        name="cifar10",
        data_dir=str(data_root),
        train_transform=transform,
        val_transform=transform,
    )
    return dataset


def cache_key(kind: str, severity: int | None = None, corruption: str | None = None) -> str:
    if kind == "clean":
        return "clean"
    return f"severity{int(severity)}_{corruption}"


def extract_or_load_features(
    *,
    cache_dir: Path,
    key: str,
    model,
    dataset: Dataset,
    batch_size: int,
    workers: int,
    device: torch.device,
    skip_existing: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    path = cache_dir / f"{key}_features.pt"
    if skip_existing and path.exists():
        payload = torch.load(path, map_location="cpu")
        log(f"feature cache hit: {key}")
        return payload["features"].float(), payload["labels"].long()
    model.visual.use_local = False
    features: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    loader = build_loader(dataset, batch_size, workers, device)
    total = len(dataset)
    seen = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=True)
            batch_labels = batch_to_labels(batch)
            image_features = model.encode_image(images).float()
            image_features = F.normalize(image_features, dim=-1)
            features.append(image_features.cpu())
            labels.append(batch_labels.cpu())
            seen += int(batch_labels.numel())
            if batch_index == 0 or seen == total or batch_index % 20 == 0:
                log(f"features {key}: {seen}/{total}")
    payload = {"features": torch.cat(features, dim=0).float(), "labels": torch.cat(labels, dim=0).long()}
    ensure_dir(path.parent)
    torch.save(payload, path)
    return payload["features"], payload["labels"]


def extract_or_load_calip_logits(
    *,
    cache_dir: Path,
    key: str,
    model,
    dataset: Dataset,
    text_features: torch.Tensor,
    batch_size: int,
    workers: int,
    device: torch.device,
    beta2: float,
    beta3: float,
    skip_existing: bool,
) -> torch.Tensor:
    path = cache_dir / f"{key}_calip_logits.pt"
    if skip_existing and path.exists():
        log(f"CALIP cache hit: {key}")
        return torch.load(path, map_location="cpu")["logits"].float()
    text = text_features.to(device=device, dtype=torch.float32)
    logits_all: List[torch.Tensor] = []
    model.visual.use_local = True
    loader = build_loader(dataset, batch_size, workers, device)
    total = len(dataset)
    seen = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=True)
            total_features = model.encode_image(images).float()
            total_features = F.normalize(total_features, dim=-1)
            global_features = total_features[:, 0, :]
            spatial_features = total_features[:, 1:, :]
            clip_logits = 100.0 * global_features @ text.t()
            attention = torch.einsum("bld,cd->blc", spatial_features, text) * 2.0
            attention_text = F.softmax(attention, dim=1)
            attention_visual = F.softmax(attention, dim=2)
            text_aware = torch.einsum("bld,blc->bdc", spatial_features, attention_text)
            logits1 = 100.0 * torch.einsum("bd,bdc->bc", global_features, text_aware)
            visual_aware = torch.einsum("blc,cd->bld", attention_visual, text)
            visual_aware = visual_aware.mean(dim=1) + visual_aware.max(dim=1).values
            logits2 = 100.0 * visual_aware @ text.t()
            logits = clip_logits + float(beta2) * logits1 + float(beta3) * logits2
            logits_all.append(logits.cpu())
            seen += int(images.shape[0])
            if batch_index == 0 or seen == total or batch_index % 20 == 0:
                log(f"CALIP logits {key}: {seen}/{total}")
    model.visual.use_local = False
    payload = {"logits": torch.cat(logits_all, dim=0).float()}
    ensure_dir(path.parent)
    torch.save(payload, path)
    torch.cuda.empty_cache()
    return payload["logits"]


def subset_indices(labels: torch.Tensor, target_class: int, imbalance_pct: float, seed: int) -> torch.Tensor:
    keep_fraction = max(0.0, min(1.0, 1.0 - float(imbalance_pct) / 100.0))
    generator = torch.Generator()
    generator.manual_seed(int(seed) + target_class * 1009 + int(round(float(imbalance_pct) * 10)))
    selected: List[torch.Tensor] = []
    for class_index in range(len(CIFAR10_CLASSES)):
        class_indices = torch.nonzero(labels.eq(class_index), as_tuple=False).flatten()
        keep_count = int(class_indices.numel()) if class_index == target_class else int(round(float(class_indices.numel()) * keep_fraction))
        if keep_count <= 0:
            continue
        perm = torch.randperm(int(class_indices.numel()), generator=generator)[:keep_count]
        selected.append(class_indices[perm])
    return torch.sort(torch.cat(selected, dim=0)).values


def build_method_rows(
    *,
    setting: str,
    features: torch.Tensor,
    labels: torch.Tensor,
    default_text: torch.Tensor,
    cupl_text: torch.Tensor,
    calip_logits: torch.Tensor,
    waffle_texts: Sequence[torch.Tensor],
    cupl_prompt_features: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    include_transductive: bool,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    clip_logits = features @ default_text.t()
    clip_top1 = evaluate_logits(clip_logits, labels)
    cbc_top1 = evaluate_cbc(clip_logits, labels, args.batch_size, args.seed)
    rows.append({"setting": setting, "method": "CLIP", "top1": clip_top1, "delta_vs_clip": 0.0, "num_samples": int(labels.numel())})
    rows.append({"setting": setting, "method": "CLIP + CBC", "top1": cbc_top1, "delta_vs_clip": cbc_top1 - clip_top1, "num_samples": int(labels.numel()), "batch_size": int(args.batch_size)})
    calip_top1 = evaluate_logits(calip_logits, labels)
    rows.append({"setting": setting, "method": "CALIP", "top1": calip_top1, "delta_vs_clip": calip_top1 - clip_top1, "num_samples": int(labels.numel())})
    waffle_scores = [evaluate_logits(features @ waffle_text.t(), labels) for waffle_text in waffle_texts]
    waffle_top1 = float(np.mean(waffle_scores))
    rows.append({"setting": setting, "method": "WaffleCLIP", "top1": waffle_top1, "top1_std": float(np.std(waffle_scores, ddof=0)), "delta_vs_clip": waffle_top1 - clip_top1, "num_samples": int(labels.numel())})
    cupl_top1 = evaluate_logits(features @ cupl_text.t(), labels)
    rows.append({"setting": setting, "method": "CuPL", "top1": cupl_top1, "delta_vs_clip": cupl_top1 - clip_top1, "num_samples": int(labels.numel())})
    if include_transductive:
        inmap_top1 = evaluate_inmap(features=features, labels=labels, text_classifier=default_text, args=args, device=device)
        rows.append({"setting": setting, "method": "InMaP", "top1": inmap_top1, "delta_vs_clip": inmap_top1 - clip_top1, "num_samples": int(labels.numel())})
        frolic_top1 = evaluate_frolic(features=features, labels=labels, text_classifier=default_text, cupl_prompt_features=cupl_prompt_features, args=args, device=device)
        rows.append({"setting": setting, "method": "Frolic", "top1": frolic_top1, "delta_vs_clip": frolic_top1 - clip_top1, "num_samples": int(labels.numel())})
    return rows


def mean_rows(rows: Sequence[Mapping[str, object]], group_keys: Sequence[str]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], List[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    summaries: List[Dict[str, object]] = []
    for key, items in sorted(grouped.items(), key=lambda item: item[0]):
        values = np.asarray([float(item["top1"]) for item in items], dtype=np.float64)
        deltas = np.asarray([float(item["delta_vs_clip"]) for item in items], dtype=np.float64)
        summary = {group_key: value for group_key, value in zip(group_keys, key)}
        summary.update({"mean_top1": float(values.mean()), "std_top1": float(values.std(ddof=0)), "mean_delta_vs_clip": float(deltas.mean()), "num_conditions": len(items)})
        summaries.append(summary)
    return summaries


def main() -> None:
    args = parse_args()
    output_root = args.output_root / args.run_name
    cache_dir = output_root / "cache"
    tables_dir = output_root / "tables"
    ensure_dir(output_root)
    write_json(
        output_root / "metadata.json",
        json_safe(vars(args) | {"template": DEFAULT_TEMPLATE, "corruptions": list(CORRUPTIONS)}),
    )
    device = choose_device(args.device)
    log(f"output_root={output_root}")
    log(f"device={device}")

    family_cfg = FAMILIES[args.family]
    model, transform = return_base_model(
        name=args.base_model_name,
        device=device,
        dataset=str(family_cfg["corrupted_dataset"]),
        path_to_weights=str(REPO_ROOT / "work"),
    )
    if hasattr(model, "visual") and not hasattr(model.visual, "use_local"):
        model.visual.use_local = False
    model.eval()

    clean_dataset = load_clean_dataset(args.data_root, transform)
    class_names = list(clean_dataset.class_names)
    default_text = text_features_from_prompts(model=model, prompts_by_class=default_prompts(class_names), device=device)
    cupl_prompt_lists = cupl_prompts(class_names, args.external_root)
    cupl_text = text_features_from_prompts(model=model, prompts_by_class=cupl_prompt_lists, device=device)
    cupl_prompt_features = torch.stack([encode_prompt_list(model, prompts, device).float() for prompts in cupl_prompt_lists], dim=0)
    waffle_texts = []
    for rep in range(int(args.waffle_reps)):
        prompts = waffle_prompts(class_names=class_names, external_root=args.external_root, seed=rep, waffle_count=int(args.waffle_count))
        waffle_texts.append(text_features_from_prompts(model=model, prompts_by_class=prompts, device=device))

    condition_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def get_condition(kind: str, severity: int | None = None, corruption: str | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = cache_key(kind, severity, corruption)
        if key in condition_cache:
            return condition_cache[key]
        if kind == "clean":
            dataset = clean_dataset
        else:
            dataset = load_cifar10c_dataset(args.data_root, transform, str(corruption), int(severity))
        features, labels = extract_or_load_features(
            cache_dir=cache_dir,
            key=key,
            model=model,
            dataset=dataset,
            batch_size=int(args.feature_batch_size),
            workers=int(args.workers),
            device=device,
            skip_existing=bool(args.skip_existing),
        )
        calip_logits = extract_or_load_calip_logits(
            cache_dir=cache_dir,
            key=key,
            model=model,
            dataset=dataset,
            text_features=default_text,
            batch_size=int(args.batch_size),
            workers=int(args.workers),
            device=device,
            beta2=float(args.calip_beta2),
            beta3=float(args.calip_beta3),
            skip_existing=bool(args.skip_existing),
        )
        condition_cache[key] = (features, labels, calip_logits)
        return condition_cache[key]

    all_rows: List[Dict[str, object]] = []

    log("experiment: clean")
    features, labels, calip_logits = get_condition("clean")
    all_rows.extend(build_method_rows(setting="clean", features=features, labels=labels, default_text=default_text, cupl_text=cupl_text, calip_logits=calip_logits, waffle_texts=waffle_texts, cupl_prompt_features=cupl_prompt_features, args=args, device=device, include_transductive=True))
    write_csv(tables_dir / "clean_comparison.csv", [row for row in all_rows if row["setting"] == "clean"])

    log("experiment: corruption-by-corruption severity 5")
    corruption_rows: List[Dict[str, object]] = []
    severity5_payloads: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]] = []
    for corruption in CORRUPTIONS:
        features, labels, calip_logits = get_condition("corrupted", args.severity, corruption)
        severity5_payloads.append((features, labels, calip_logits, corruption))
        rows = build_method_rows(setting=f"severity{args.severity}:{corruption}", features=features, labels=labels, default_text=default_text, cupl_text=cupl_text, calip_logits=calip_logits, waffle_texts=waffle_texts, cupl_prompt_features=cupl_prompt_features, args=args, device=device, include_transductive=True)
        corruption_rows.extend(rows)
        all_rows.extend(rows)
        write_csv(tables_dir / "corruption_by_corruption_s5.csv", corruption_rows)

    log("experiment: corrupted overall severity 5")
    mixed_features = torch.cat([item[0] for item in severity5_payloads], dim=0)
    mixed_labels = torch.cat([item[1] for item in severity5_payloads], dim=0)
    mixed_calip_logits = torch.cat([item[2] for item in severity5_payloads], dim=0)
    overall_rows = build_method_rows(setting=f"severity{args.severity}:overall", features=mixed_features, labels=mixed_labels, default_text=default_text, cupl_text=cupl_text, calip_logits=mixed_calip_logits, waffle_texts=waffle_texts, cupl_prompt_features=cupl_prompt_features, args=args, device=device, include_transductive=True)
    all_rows.extend(overall_rows)
    write_csv(tables_dir / "corrupted_overall_s5.csv", overall_rows)

    log("experiment: severity sweep")
    severity_rows: List[Dict[str, object]] = []
    for severity in args.severity_sweep:
        if int(severity) == 0:
            features, labels, calip_logits = get_condition("clean")
        elif int(severity) == int(args.severity):
            features, labels, calip_logits = mixed_features, mixed_labels, mixed_calip_logits
        else:
            payloads = [get_condition("corrupted", int(severity), corruption) for corruption in CORRUPTIONS]
            features = torch.cat([item[0] for item in payloads], dim=0)
            labels = torch.cat([item[1] for item in payloads], dim=0)
            calip_logits = torch.cat([item[2] for item in payloads], dim=0)
        rows = build_method_rows(setting=f"severity_sweep:{severity}", features=features, labels=labels, default_text=default_text, cupl_text=cupl_text, calip_logits=calip_logits, waffle_texts=waffle_texts, cupl_prompt_features=cupl_prompt_features, args=args, device=device, include_transductive=True)
        for row in rows:
            row["severity"] = int(severity)
        severity_rows.extend(rows)
        write_csv(tables_dir / "severity_sweep.csv", severity_rows)

    log("experiment: batch sweep")
    batch_rows: List[Dict[str, object]] = []
    fixed_rows = build_method_rows(setting=f"batch_sweep:{args.batch_size}", features=mixed_features, labels=mixed_labels, default_text=default_text, cupl_text=cupl_text, calip_logits=mixed_calip_logits, waffle_texts=waffle_texts, cupl_prompt_features=cupl_prompt_features, args=args, device=device, include_transductive=True)
    fixed_by_method = {str(row["method"]): row for row in fixed_rows}
    clip_logits = mixed_features @ default_text.t()
    clip_top1 = evaluate_logits(clip_logits, mixed_labels)
    for batch_size in args.batch_sweep:
        cbc_top1 = evaluate_cbc(clip_logits, mixed_labels, int(batch_size), int(args.seed))
        batch_rows.append({"batch_size": int(batch_size), "method": "CLIP", "top1": clip_top1, "delta_vs_clip": 0.0})
        batch_rows.append({"batch_size": int(batch_size), "method": "CLIP + CBC", "top1": cbc_top1, "delta_vs_clip": cbc_top1 - clip_top1})
        for method in ["CALIP", "WaffleCLIP", "CuPL", "InMaP", "Frolic"]:
            row = fixed_by_method[method]
            batch_rows.append({"batch_size": int(batch_size), "method": method, "top1": row["top1"], "delta_vs_clip": row["delta_vs_clip"], "note": "batch-independent score repeated for comparison"})
    write_csv(tables_dir / "batch_size_sweep.csv", batch_rows)

    if not args.skip_imbalance:
        log("experiment: class imbalance")
        imbalance_rows: List[Dict[str, object]] = []
        for corruption, features, labels, calip_logits in [(item[3], item[0], item[1], item[2]) for item in severity5_payloads]:
            for target_class, target_name in enumerate(CIFAR10_CLASSES):
                for pct in args.imbalance_pcts:
                    idx = subset_indices(labels, target_class, float(pct), int(args.seed))
                    rows = build_method_rows(
                        setting=f"imbalance:{corruption}:{target_name}:{pct:g}",
                        features=features[idx],
                        labels=labels[idx],
                        default_text=default_text,
                        cupl_text=cupl_text,
                        calip_logits=calip_logits[idx],
                        waffle_texts=waffle_texts,
                        cupl_prompt_features=cupl_prompt_features,
                        args=args,
                        device=device,
                        include_transductive=bool(args.include_transductive_imbalance),
                    )
                    target_share = 100.0 * float(labels[idx].eq(target_class).float().mean().item())
                    for row in rows:
                        row.update({"corruption": corruption, "target_class": target_name, "target_class_index": target_class, "imbalance_pct": float(pct), "target_share_pct": target_share})
                    imbalance_rows.extend(rows)
                    write_csv(tables_dir / "class_imbalance_by_corruption_target.csv", imbalance_rows)
        write_csv(tables_dir / "class_imbalance_average.csv", mean_rows(imbalance_rows, ["method", "imbalance_pct"]))

    write_csv(tables_dir / "all_results.csv", all_rows)
    write_json(output_root / "final_summary.json", {"tables_dir": str(tables_dir)})
    log("full zero-shot enhancer suite complete")


if __name__ == "__main__":
    main()
