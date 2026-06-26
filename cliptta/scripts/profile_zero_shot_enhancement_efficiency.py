#!/usr/bin/env python3
"""Profile FLOPs, peak CUDA memory, and throughput for zero-shot enhancers.

The benchmark measures CLIP, CLIP+CBC, CALIP, WaffleCLIP, CuPL, InMaP, and
Frolic on the same CIFAR-10-C sample subset. FLOPs are profiler-based estimates
for batch inference methods and analytical estimates for transductive optimizers.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "work" / ".mplconfig"),
)

import numpy as np
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity
from torch.utils.data import Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_zero_shot_enhancement_baselines import (  # noqa: E402
    DEFAULT_EXTERNAL_ROOT,
    batch_to_labels,
    build_loader,
    choose_device,
    cupl_prompts,
    default_prompts,
    encode_prompt_list,
    evaluate_frolic,
    evaluate_inmap,
    load_cifar10_class_names,
    load_cifar10c_dataset,
    log,
    text_features_from_prompts,
    waffle_prompts,
)
from run_closed_set_cifar_benchmarks import DEFAULT_DATA_ROOT, FAMILIES  # noqa: E402
from ttavlm.datasets import CORRUPTIONS  # noqa: E402
from ttavlm.models import return_base_model  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "work" / "zero_shot_enhancement_efficiency"
METHODS = ["clip", "cbc", "calip", "waffle", "cupl", "inmap", "frolic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["cifar10"], default="cifar10")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--corruptions", nargs="+", default=["all"])
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="cifar10c_s5_zero_shot_efficiency")
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--base-model-name", type=str, default="clip-ViT-B/16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-corruption", type=int, default=512)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--inmap-iters", type=int, default=2000)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument("--inmap-lr", type=float, default=10.0)
    parser.add_argument("--inmap-tau-t", type=float, default=0.01)
    parser.add_argument("--inmap-tau-i", type=float, default=0.04)
    parser.add_argument("--inmap-alpha", type=float, default=0.6)
    parser.add_argument("--waffle-count", type=int, default=15)
    parser.add_argument("--calip-beta2", type=float, default=2.0)
    parser.add_argument("--calip-beta3", type=float, default=0.1)
    parser.add_argument("--skip-flops", action="store_true")
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


def resolve_corruptions(values: Sequence[str]) -> List[str]:
    if len(values) == 1 and values[0] == "all":
        return list(CORRUPTIONS)
    unknown = [value for value in values if value not in CORRUPTIONS]
    if unknown:
        raise SystemExit(f"Unknown corruptions {unknown}. Available: all, {', '.join(CORRUPTIONS)}")
    return list(values)


def maybe_subset(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None or int(max_samples) >= len(dataset):
        return dataset
    return Subset(dataset, list(range(int(max_samples))))


def build_datasets(args: argparse.Namespace, transform, corruptions: Sequence[str]) -> List[Tuple[str, Dataset]]:
    datasets: List[Tuple[str, Dataset]] = []
    for corruption in corruptions:
        dataset = load_cifar10c_dataset(args.data_root, transform, corruption, int(args.severity))
        datasets.append((corruption, maybe_subset(dataset, args.max_samples_per_corruption)))
    return datasets


def iter_batches(
    *,
    datasets: Sequence[Tuple[str, Dataset]],
    batch_size: int,
    workers: int,
    device: torch.device,
) -> Iterable[Tuple[str, Mapping[str, object]]]:
    for corruption, dataset in datasets:
        loader = build_loader(dataset, int(batch_size), int(workers), device)
        for batch in loader:
            yield corruption, batch


def first_batch(
    *,
    datasets: Sequence[Tuple[str, Dataset]],
    batch_size: int,
    workers: int,
    device: torch.device,
) -> Mapping[str, object]:
    return next(iter(iter_batches(datasets=datasets, batch_size=batch_size, workers=workers, device=device)))[1]


def cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_cuda_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> Tuple[float, float]:
    if device.type != "cuda":
        return float("nan"), float("nan")
    allocated = torch.cuda.max_memory_allocated(device) / (1024.0**2)
    reserved = torch.cuda.max_memory_reserved(device) / (1024.0**2)
    return allocated, reserved


def profiler_activities(device: torch.device) -> List[ProfilerActivity]:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    return activities


def profile_flops(
    *,
    device: torch.device,
    fn: Callable[[], torch.Tensor],
) -> int:
    cuda_synchronize(device)
    try:
        with torch.profiler.profile(
            activities=profiler_activities(device),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=True,
        ) as prof:
            with torch.no_grad():
                _ = fn()
        cuda_synchronize(device)
        return int(sum(int(evt.flops or 0) for evt in prof.key_averages()))
    except Exception as exc:
        log(f"FLOPs profiler failed: {exc}")
        return 0


def encode_global_logits(model, images: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    model.visual.use_local = False
    features = model.encode_image(images).float()
    if features.ndim == 3:
        features = features[:, 0, :]
    features = F.normalize(features, dim=-1)
    return 100.0 * features @ text_features.t()


def encode_calip_logits(
    model,
    images: torch.Tensor,
    text_features: torch.Tensor,
    beta2: float,
    beta3: float,
) -> torch.Tensor:
    model.visual.use_local = True
    total_features = model.encode_image(images).float()
    total_features = F.normalize(total_features, dim=-1)
    global_features = total_features[:, 0, :]
    spatial_features = total_features[:, 1:, :]
    clip_logits = 100.0 * global_features @ text_features.t()
    attention = torch.einsum("bld,cd->blc", spatial_features, text_features) * 2.0
    attention_text = F.softmax(attention, dim=1)
    attention_visual = F.softmax(attention, dim=2)
    text_aware = torch.einsum("bld,blc->bdc", spatial_features, attention_text)
    logits1 = 100.0 * torch.einsum("bd,bdc->bc", global_features, text_aware)
    visual_aware = torch.einsum("blc,cd->bld", attention_visual, text_features)
    visual_aware = visual_aware.mean(dim=1) + visual_aware.max(dim=1).values
    logits2 = 100.0 * visual_aware @ text_features.t()
    return clip_logits + float(beta2) * logits1 + float(beta3) * logits2


def method_logits_fn(
    *,
    method: str,
    model,
    text_features: torch.Tensor,
    calip_beta2: float,
    calip_beta3: float,
) -> Callable[[torch.Tensor], torch.Tensor]:
    def run(images: torch.Tensor) -> torch.Tensor:
        if method == "calip":
            return encode_calip_logits(model, images, text_features, calip_beta2, calip_beta3)
        logits = encode_global_logits(model, images, text_features)
        if method == "cbc":
            center = logits.mean(dim=0, keepdim=True) if logits.shape[0] > 1 else torch.zeros_like(logits[:1])
            logits = logits - center
        return logits

    return run


def warmup_method(
    *,
    datasets: Sequence[Tuple[str, Dataset]],
    batch_size: int,
    workers: int,
    device: torch.device,
    run_logits: Callable[[torch.Tensor], torch.Tensor],
    warmup_batches: int,
) -> None:
    if warmup_batches <= 0:
        return
    seen_batches = 0
    with torch.no_grad():
        for _, batch in iter_batches(datasets=datasets, batch_size=batch_size, workers=workers, device=device):
            images = batch["image"].to(device, non_blocking=True)
            _ = run_logits(images)
            seen_batches += 1
            if seen_batches >= warmup_batches:
                break
    cuda_synchronize(device)


def profile_streaming_method(
    *,
    method: str,
    model,
    datasets: Sequence[Tuple[str, Dataset]],
    text_features: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    text = text_features.to(device=device, dtype=torch.float32)
    run_logits = method_logits_fn(
        method=method,
        model=model,
        text_features=text,
        calip_beta2=float(args.calip_beta2),
        calip_beta3=float(args.calip_beta3),
    )
    batch = first_batch(datasets=datasets, batch_size=int(args.batch_size), workers=int(args.workers), device=device)
    images = batch["image"].to(device, non_blocking=True)
    n_profile = int(images.shape[0])
    flops = 0 if args.skip_flops else profile_flops(device=device, fn=lambda: run_logits(images))

    warmup_method(
        datasets=datasets,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        device=device,
        run_logits=run_logits,
        warmup_batches=int(args.warmup_batches),
    )

    reset_cuda_peak(device)
    total = 0
    correct = 0
    started = time.perf_counter()
    with torch.no_grad():
        for _, item in iter_batches(datasets=datasets, batch_size=int(args.batch_size), workers=int(args.workers), device=device):
            batch_images = item["image"].to(device, non_blocking=True)
            labels = batch_to_labels(item).to(device, non_blocking=True)
            logits = run_logits(batch_images)
            correct += int(logits.argmax(dim=1).eq(labels).sum().item())
            total += int(labels.numel())
    cuda_synchronize(device)
    elapsed = time.perf_counter() - started
    allocated_mb, reserved_mb = peak_memory_mb(device)
    model.visual.use_local = False

    flops_per_sample = float(flops) / float(max(n_profile, 1)) if flops else float("nan")
    return {
        "method": method,
        "sample_count": total,
        "top1": 100.0 * float(correct) / float(max(total, 1)),
        "eval_sec": elapsed,
        "throughput_img_s": float(total) / elapsed if elapsed > 0 else float("nan"),
        "ms_per_image": 1000.0 * elapsed / float(max(total, 1)),
        "peak_allocated_mb": allocated_mb,
        "peak_reserved_mb": reserved_mb,
        "profiled_batch_size": n_profile,
        "flops_per_image": flops_per_sample,
        "estimated_total_flops": flops_per_sample * float(total) if flops else float("nan"),
        "flops_source": "torch_profiler_batch_extrapolated",
    }


def extract_features_timed(
    *,
    model,
    datasets: Sequence[Tuple[str, Dataset]],
    batch_size: int,
    workers: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    features: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    model.visual.use_local = False
    started = time.perf_counter()
    with torch.no_grad():
        for corruption, dataset in datasets:
            loader = build_loader(dataset, int(batch_size), int(workers), device)
            seen = 0
            for batch_index, batch in enumerate(loader):
                images = batch["image"].to(device, non_blocking=True)
                batch_labels = batch_to_labels(batch)
                image_features = model.encode_image(images).float()
                if image_features.ndim == 3:
                    image_features = image_features[:, 0, :]
                image_features = F.normalize(image_features, dim=-1)
                features.append(image_features.cpu())
                labels.append(batch_labels.cpu())
                seen += int(batch_labels.numel())
                if batch_index == 0 or seen == len(dataset) or batch_index % 20 == 0:
                    log(f"features {corruption}: {seen}/{len(dataset)}")
    cuda_synchronize(device)
    elapsed = time.perf_counter() - started
    return torch.cat(features, dim=0).float(), torch.cat(labels, dim=0).long(), elapsed


def estimate_image_only_flops_per_sample(
    *,
    model,
    datasets: Sequence[Tuple[str, Dataset]],
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    if args.skip_flops:
        return float("nan")
    batch = first_batch(datasets=datasets, batch_size=int(args.batch_size), workers=int(args.workers), device=device)
    images = batch["image"].to(device, non_blocking=True)
    n_profile = int(images.shape[0])

    def run() -> torch.Tensor:
        model.visual.use_local = False
        output = model.encode_image(images).float()
        if output.ndim == 3:
            output = output[:, 0, :]
        return output

    flops = profile_flops(device=device, fn=run)
    return float(flops) / float(max(n_profile, 1)) if flops else float("nan")


def matmul_flops(rows: int, inner: int, cols: int) -> float:
    return 2.0 * float(rows) * float(inner) * float(cols)


def estimate_inmap_flops(
    *,
    num_samples: int,
    feature_dim: int,
    num_classes: int,
    image_flops_per_sample: float,
    args: argparse.Namespace,
) -> float:
    feature_flops = image_flops_per_sample * float(num_samples) if np.isfinite(image_flops_per_sample) else float("nan")
    logits_and_final = 2.0 * matmul_flops(num_samples, feature_dim, num_classes)
    sinkhorn = float(args.sinkhorn_iters) * 6.0 * float(num_samples) * float(num_classes)
    image_opt = float(args.inmap_iters) * (
        matmul_flops(num_samples, feature_dim, num_classes)
        + matmul_flops(feature_dim, num_samples, num_classes)
        + 6.0 * float(num_samples) * float(num_classes)
    )
    return feature_flops + logits_and_final + sinkhorn + image_opt


def estimate_frolic_flops(
    *,
    num_samples: int,
    feature_dim: int,
    num_classes: int,
    prompt_count: int,
    image_flops_per_sample: float,
    args: argparse.Namespace,
) -> float:
    feature_flops = image_flops_per_sample * float(num_samples) if np.isfinite(image_flops_per_sample) else float("nan")
    llm_count = int(num_classes) * int(prompt_count)
    front_matter = (
        3.0 * matmul_flops(num_samples, feature_dim, num_classes)
        + 2.0 * float(num_samples + llm_count) * float(feature_dim)
        + 10.0 * float(num_samples) * float(num_classes)
    )
    sinkhorn = 2.0 * float(args.sinkhorn_iters) * 6.0 * float(num_samples) * float(num_classes)
    two_image_opts = 2.0 * float(args.inmap_iters) * (
        matmul_flops(num_samples, feature_dim, num_classes)
        + matmul_flops(feature_dim, num_samples, num_classes)
        + 6.0 * float(num_samples) * float(num_classes)
    )
    debias = 11.0 * (float(num_samples) * float(num_classes) * 4.0 + float(num_classes) ** 2)
    return feature_flops + front_matter + sinkhorn + two_image_opts + debias


def profile_transductive_method(
    *,
    method: str,
    model,
    datasets: Sequence[Tuple[str, Dataset]],
    default_text: torch.Tensor,
    cupl_prompt_features: torch.Tensor,
    image_flops_per_sample: float,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    reset_cuda_peak(device)
    features, labels, feature_sec = extract_features_timed(
        model=model,
        datasets=datasets,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        device=device,
    )
    adapt_started = time.perf_counter()
    if method == "inmap":
        top1 = evaluate_inmap(
            features=features,
            labels=labels,
            text_classifier=default_text,
            args=args,
            device=device,
        )
    elif method == "frolic":
        top1 = evaluate_frolic(
            features=features,
            labels=labels,
            text_classifier=default_text,
            cupl_prompt_features=cupl_prompt_features,
            args=args,
            device=device,
        )
    else:
        raise ValueError(method)
    cuda_synchronize(device)
    adapt_sec = time.perf_counter() - adapt_started
    total_sec = feature_sec + adapt_sec
    allocated_mb, reserved_mb = peak_memory_mb(device)
    num_samples = int(labels.numel())
    feature_dim = int(features.shape[1])
    num_classes = int(default_text.shape[0])
    prompt_count = int(cupl_prompt_features.shape[1])
    if method == "inmap":
        total_flops = estimate_inmap_flops(
            num_samples=num_samples,
            feature_dim=feature_dim,
            num_classes=num_classes,
            image_flops_per_sample=image_flops_per_sample,
            args=args,
        )
    else:
        total_flops = estimate_frolic_flops(
            num_samples=num_samples,
            feature_dim=feature_dim,
            num_classes=num_classes,
            prompt_count=prompt_count,
            image_flops_per_sample=image_flops_per_sample,
            args=args,
        )
    return {
        "method": method,
        "sample_count": num_samples,
        "top1": top1,
        "feature_extraction_sec": feature_sec,
        "adapt_sec": adapt_sec,
        "eval_sec": total_sec,
        "throughput_img_s": float(num_samples) / total_sec if total_sec > 0 else float("nan"),
        "ms_per_image": 1000.0 * total_sec / float(max(num_samples, 1)),
        "peak_allocated_mb": allocated_mb,
        "peak_reserved_mb": reserved_mb,
        "profiled_batch_size": int(args.batch_size),
        "flops_per_image": total_flops / float(max(num_samples, 1)),
        "estimated_total_flops": total_flops,
        "flops_source": "analytical_feature_extraction_plus_transductive_optimization",
    }


def main() -> None:
    args = parse_args()
    unknown = [method for method in args.methods if method not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown methods {unknown}. Available: {', '.join(METHODS)}")

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    output_root = args.output_root / args.run_name
    ensure_dir(output_root)
    device = choose_device(args.device)
    corruptions = resolve_corruptions(args.corruptions)

    write_json(
        output_root / "metadata.json",
        {
            "family": args.family,
            "dataset": "cifar10c",
            "severity": int(args.severity),
            "corruptions": corruptions,
            "max_samples_per_corruption": args.max_samples_per_corruption,
            "methods": args.methods,
            "batch_size": int(args.batch_size),
            "base_model_name": args.base_model_name,
            "flops_note": "torch profiler estimates for streaming methods; analytical estimates for InMaP/Frolic optimization.",
        },
    )
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

    datasets = build_datasets(args, transform, corruptions)
    class_names = load_cifar10_class_names(args.data_root, transform)
    default_text = text_features_from_prompts(model=model, prompts_by_class=default_prompts(class_names), device=device)
    cupl_prompt_lists = cupl_prompts(class_names, args.external_root)
    cupl_text = text_features_from_prompts(model=model, prompts_by_class=cupl_prompt_lists, device=device)
    cupl_prompt_features = torch.stack(
        [encode_prompt_list(model, prompts, device).float() for prompts in cupl_prompt_lists],
        dim=0,
    ).float()
    waffle_text = text_features_from_prompts(
        model=model,
        prompts_by_class=waffle_prompts(
            class_names=class_names,
            external_root=args.external_root,
            seed=int(args.seed),
            waffle_count=int(args.waffle_count),
        ),
        device=device,
    )

    image_flops_per_sample = estimate_image_only_flops_per_sample(
        model=model,
        datasets=datasets,
        args=args,
        device=device,
    )

    text_by_method = {
        "clip": default_text,
        "cbc": default_text,
        "calip": default_text,
        "waffle": waffle_text,
        "cupl": cupl_text,
    }
    display_names = {
        "clip": "CLIP",
        "cbc": "CLIP + CBC",
        "calip": "CALIP",
        "waffle": "WaffleCLIP",
        "cupl": "CuPL",
        "inmap": "InMaP",
        "frolic": "Frolic",
    }

    rows: List[Dict[str, object]] = []
    for method in args.methods:
        log(f"profiling {display_names[method]}")
        if method in text_by_method:
            row = profile_streaming_method(
                method=method,
                model=model,
                datasets=datasets,
                text_features=text_by_method[method],
                args=args,
                device=device,
            )
        else:
            row = profile_transductive_method(
                method=method,
                model=model,
                datasets=datasets,
                default_text=default_text,
                cupl_prompt_features=cupl_prompt_features,
                image_flops_per_sample=image_flops_per_sample,
                args=args,
                device=device,
            )
        row["method"] = display_names[method]
        row["severity"] = int(args.severity)
        row["num_corruptions"] = len(corruptions)
        rows.append(row)
        write_csv(output_root / "efficiency_results_partial.csv", rows)
        log(
            f"{display_names[method]}: throughput={float(row['throughput_img_s']):.2f} img/s, "
            f"peak={float(row['peak_allocated_mb']):.1f} MB, "
            f"flops/img={float(row['flops_per_image']):.3e}"
        )

    write_csv(output_root / "efficiency_results.csv", rows)
    write_json(output_root / "efficiency_results.json", {"rows": rows})
    log(f"wrote {output_root / 'efficiency_results.csv'}")


if __name__ == "__main__":
    main()
