#!/usr/bin/env python3
"""Evaluate default-prompt CLIP and CLIP + CBC on CIFAR-10 clean/corrupted."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "work" / ".mplconfig"),
)

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_cbc_clip_zero_shot_experiments import (  # noqa: E402
    CORRUPTIONS,
    DEFAULT_DATA_ROOT,
    DEFAULT_TEMPLATE,
    FAMILIES,
    build_loader,
    choose_device,
    concat_conditions,
    dataset_attr,
    encode_text_features,
    ensure_dir,
    evaluate_clip_cbc,
    load_dataset,
    load_or_compute_logits,
    now,
    strip_preds,
    write_csv,
    write_json,
)
from ttavlm.models import return_base_model  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("/home/josephyumss/workspace/AIProject/outputs/cbc_eval/cbc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["cifar10"], default="cifar10")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="cifar10c_clip_clean_corrupted")
    parser.add_argument("--base-model-name", type=str, default="clip-ViT-B/16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument("--corruptions-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def metric_rows(setting: str, metrics: Mapping[str, object]) -> List[Dict[str, object]]:
    base = strip_preds(metrics)
    return [
        {
            "setting": setting,
            "method": "CLIP",
            "top1": base["raw_top1"],
            "delta_vs_clip": 0.0,
            "num_samples": base["num_samples"],
            "batch_size": base["batch_size"],
        },
        {
            "setting": setting,
            "method": "CLIP + CBC",
            "top1": base["cbc_top1"],
            "delta_vs_clip": base["delta_top1"],
            "num_samples": base["num_samples"],
            "batch_size": base["batch_size"],
        },
    ]


def main() -> None:
    args = parse_args()
    if args.clean_only and args.corruptions_only:
        raise SystemExit("Choose at most one of --clean-only and --corruptions-only.")

    run_root = args.output_root / args.run_name / args.family / f"severity_{args.severity}"
    cache_dir = run_root / "cache" / "logits"
    ensure_dir(run_root)

    device = choose_device(args.device)
    log(f"using device={device}; output_root={run_root}")

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

    clean_dataset = load_dataset(
        dataset_name=clean_dataset_name,
        data_root=args.data_root,
        transform=transform,
    )
    class_names = list(dataset_attr(clean_dataset, "class_names"))
    text_features = encode_text_features(model=model, class_names=class_names, device=device)

    metadata = {
        "family": args.family,
        "severity": int(args.severity),
        "template": DEFAULT_TEMPLATE,
        "methods": ["CLIP", "CLIP + CBC"],
        "batch_size": int(args.batch_size),
        "feature_batch_size": int(args.feature_batch_size),
        "seed": int(args.seed),
        "data_root": str(args.data_root),
        "clean_only": bool(args.clean_only),
        "corruptions_only": bool(args.corruptions_only),
        "corruptions": list(CORRUPTIONS),
    }
    write_json(run_root / "metadata.json", metadata)

    all_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    if not args.corruptions_only:
        clean_logits, clean_labels = load_or_compute_logits(
            cache_dir=cache_dir,
            condition="clean",
            model=model,
            dataset=clean_dataset,
            text_features=text_features,
            batch_size=args.feature_batch_size,
            workers=args.workers,
            device=device,
            max_samples=None,
        )
        clean_metrics = evaluate_clip_cbc(
            logits=clean_logits,
            labels=clean_labels,
            batch_size=int(args.batch_size),
            seed=int(args.seed),
        )
        summary_rows.append({"setting": "clean", **strip_preds(clean_metrics)})
        all_rows.extend(metric_rows("clean", clean_metrics))
        log(
            "clean: "
            f"CLIP={clean_metrics['raw_top1']:.3f}, "
            f"CBC={clean_metrics['cbc_top1']:.3f}, "
            f"delta={clean_metrics['delta_top1']:.3f}"
        )

    if not args.clean_only:
        corruption_payloads: List[Tuple[torch.Tensor, torch.Tensor, str]] = []
        for corruption in CORRUPTIONS:
            dataset = load_dataset(
                dataset_name=corrupted_dataset_name,
                data_root=args.data_root,
                transform=transform,
                shift_type=corruption,
                severity=int(args.severity),
            )
            logits, labels = load_or_compute_logits(
                cache_dir=cache_dir,
                condition=f"severity{int(args.severity)}_{corruption}",
                model=model,
                dataset=dataset,
                text_features=text_features,
                batch_size=args.feature_batch_size,
                workers=args.workers,
                device=device,
                max_samples=None,
            )
            corruption_metrics = evaluate_clip_cbc(
                logits=logits,
                labels=labels,
                batch_size=int(args.batch_size),
                seed=int(args.seed),
            )
            summary_rows.append({"setting": f"corrupted:{corruption}", **strip_preds(corruption_metrics)})
            all_rows.extend(metric_rows(f"corrupted:{corruption}", corruption_metrics))
            corruption_payloads.append((logits, labels, corruption))
            log(
                f"corrupted:{corruption}: "
                f"CLIP={corruption_metrics['raw_top1']:.3f}, "
                f"CBC={corruption_metrics['cbc_top1']:.3f}, "
                f"delta={corruption_metrics['delta_top1']:.3f}"
            )

        mixed_logits, mixed_labels, _ = concat_conditions(corruption_payloads)
        mixed_metrics = evaluate_clip_cbc(
            logits=mixed_logits,
            labels=mixed_labels,
            batch_size=int(args.batch_size),
            seed=int(args.seed),
        )
        summary_rows.append({"setting": "corrupted:overall", **strip_preds(mixed_metrics)})
        all_rows.extend(metric_rows("corrupted:overall", mixed_metrics))
        log(
            "corrupted:overall: "
            f"CLIP={mixed_metrics['raw_top1']:.3f}, "
            f"CBC={mixed_metrics['cbc_top1']:.3f}, "
            f"delta={mixed_metrics['delta_top1']:.3f}"
        )

    write_csv(run_root / "all_results.csv", all_rows)
    write_csv(run_root / "summary.csv", summary_rows)
    write_json(run_root / "summary.json", {"summary": summary_rows, "results": all_rows})
    log(f"CLIP/CBC evaluation complete: {run_root}")


if __name__ == "__main__":
    main()
