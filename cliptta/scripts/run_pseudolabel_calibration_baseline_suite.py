#!/usr/bin/env python3
"""Benchmark pseudo-label-driven TTA baselines with and without calibration."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
ETTA_ROOT = WORKSPACE_ROOT / "ETTA"

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "work" / ".mplconfig"))

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ETTA_ROOT))

import clip as etta_clip  # noqa: E402

import ttavlm.configuration as config  # noqa: E402
import ttavlm.lib as lib  # noqa: E402
from ttavlm.datasets import CLEAN_DATASETS, CORRUPTIONS, get_template, return_train_val_datasets  # noqa: E402
from ttavlm.methods import return_tta_model  # noqa: E402
from ttavlm.models import return_base_model  # noqa: E402
from ttavlm.models.clip import clip as clip_tokenizer  # noqa: E402
from ttavlm.transforms import TransformList, add_tta_transform  # noqa: E402

from ETTA import update_cache as etta_update_cache, compute_cache_logits as etta_compute_cache_logits  # noqa: E402
from utils import avg_entropy as etta_avg_entropy, clip_classifier, softmax_entropy as etta_softmax_entropy  # noqa: E402

from run_closed_set_cifar_benchmarks import (  # noqa: E402
    BASELINE_BUILDERS,
    EXTRA_BUILDERS,
    FAMILIES as CLOSED_SET_FAMILIES,
)
from run_closed_set_cifar_etta import (  # noqa: E402
    CORRUPTIONS as ETTA_CORRUPTIONS,
    DATASET_TO_FAMILY as ETTA_DATASET_TO_FAMILY,
    FAMILIES as ETTA_FAMILIES,
    build_loader as build_etta_loader,
    set_seed as set_etta_seed,
)


DEFAULT_DATA_ROOT = Path("/home/josephyumss/data")
DEFAULT_OUTPUT_ROOT = Path("/home/josephyumss/workspace/AIProject/centroid_shifting")

GENERIC_CORRUPTION_TEMPLATES: Tuple[Tuple[str, str], ...] = (
    ("default", "a photo of a {}"),
    ("noisy", "a noisy photo of a {}"),
    ("corrupted", "a corrupted photo of a {}"),
    ("degraded", "a degraded photo of a {}"),
    ("distorted", "a distorted photo of a {}"),
    ("blurry", "a blurry photo of a {}"),
    ("low_resolution", "a low resolution photo of a {}"),
    ("bad", "a bad photo of a {}"),
    ("damaged", "a damaged photo of a {}"),
)

DEFAULT_METHODS: Tuple[str, ...] = (
    "clipartt",
    "unient",
    "stamp",
    "tda",
    "rotta",
    "sotta",
    "adacontrast",
    "watt",
    "watt_otsu",
    "watt_unient",
    "etta",
)

VARIANTS: Tuple[str, ...] = ("raw", "calibrated")

# Closed-set CIFAR benchmark paper settings used in the repo's official suite.
PAPER_BATCH_SIZES: Dict[str, int] = {
    "clipartt": 128,
    "rotta": 128,
    "stamp": 128,
    "sotta": 128,
    "adacontrast": 128,
    "watt": 128,
    "watt_otsu": 128,
    "watt_unient": 128,
    "unient": 128,
    "tda": 1,
}

TTAVLM_BUILDERS: Dict[str, Any] = dict(BASELINE_BUILDERS)
TTAVLM_BUILDERS.update(EXTRA_BUILDERS)
TTAVLM_BUILDERS.pop("etta", None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=list(CLOSED_SET_FAMILIES.keys()), default="cifar10")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--shift-types", nargs="+", default=["all"])
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="pseudolabel_calibration_baseline_suite")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--default-batch-size", type=int, default=128)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--display-progress", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--calibration-prompt-mode",
        type=str,
        choices=("default", "distorted", "generic_max"),
        default="distorted",
    )
    parser.add_argument("--etta-backbone", choices=("RN50", "ViT-B/16"), default="ViT-B/16")
    parser.add_argument("--etta-alpha", type=float, default=0.2)
    return parser.parse_args()


def resolve_methods(values: Sequence[str]) -> List[str]:
    if list(values) == ["all"]:
        return list(DEFAULT_METHODS)
    unknown = sorted(set(values) - set(DEFAULT_METHODS))
    if unknown:
        raise ValueError(f"Unsupported methods: {unknown}")
    return list(values)


def resolve_shift_types(values: Sequence[str]) -> List[str]:
    if list(values) == ["all"]:
        return list(CORRUPTIONS)
    return list(values)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
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


def write_json(path: Path, payload: Dict[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def mean(values: Sequence[float]) -> float:
    valid = [float(value) for value in values if value == value]
    if not valid:
        return float("nan")
    return float(sum(valid) / len(valid))


def std(values: Sequence[float]) -> float:
    valid = [float(value) for value in values if value == value]
    if not valid:
        return float("nan")
    return float(np.asarray(valid, dtype=np.float64).std(ddof=0))


def build_loader(
    *,
    dataset_name: str,
    data_root: Path,
    transform,
    shift_type: str,
    severity: int,
    batch_size: int,
    workers: int,
    seed: int,
) -> torch.utils.data.DataLoader:
    _, dataset = return_train_val_datasets(
        name=dataset_name,
        data_dir=str(data_root),
        train_transform=transform,
        val_transform=transform,
        shift=shift_type,
        severity=severity,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=False,
        drop_last=False,
        generator=generator,
    )


def encode_text_features(
    *,
    text_encoder: torch.nn.Module,
    class_names: Sequence[str],
    templates: Sequence[str],
    device: torch.device,
) -> Tensor:
    bank: List[Tensor] = []
    with torch.no_grad():
        for template in templates:
            prompts = [template.format(name.replace("_", " ")) for name in class_names]
            tokens = clip_tokenizer.tokenize(prompts).to(device)
            text_features = text_encoder.encode_text(tokens).float()
            text_features = F.normalize(text_features, dim=-1)
            bank.append(text_features)
    text_features = torch.stack(bank, dim=0).mean(dim=0)
    return F.normalize(text_features, dim=-1)


def parse_ttavlm_args(flags: Sequence[str]) -> argparse.Namespace:
    saved_argv = sys.argv
    try:
        sys.argv = ["ttavlm.main", *flags]
        return config.argparser()
    finally:
        sys.argv = saved_argv


def common_ttavlm_flags(
    *,
    user_args: argparse.Namespace,
    dataset_name: str,
    exp_name: str,
    save_root: Path,
    batch_size: int,
) -> List[str]:
    flags = [
        "--env",
        "closed_set_cifar",
        "--exp_name",
        exp_name,
        "--dataroot",
        str(user_args.data_root),
        "--save_root",
        str(save_root),
        "--dataset",
        dataset_name,
        "--base_model_name",
        "clip-ViT-B/16",
        "--steps",
        str(int(user_args.steps)),
        "--workers",
        str(int(user_args.workers)),
        "--batch_size",
        str(int(batch_size)),
        "--seeds",
        *[str(seed) for seed in user_args.seeds],
        "--closed_set",
    ]
    flags += ["--shift_type", "all" if dataset_name.endswith("c") else "original"]
    if user_args.display_progress:
        flags.append("--display_progress")
    return flags


def drop_flag(flags: Sequence[str], flag_name: str) -> List[str]:
    cleaned: List[str] = []
    skip_next = False
    for value in flags:
        if skip_next:
            skip_next = False
            continue
        if value == flag_name:
            skip_next = True
            continue
        cleaned.append(value)
    return cleaned


def build_ttavlm_runtime_args(
    *,
    user_args: argparse.Namespace,
    family: str,
    family_cfg: Mapping[str, object],
    method: str,
    dataset_name: str,
    save_root: Path,
) -> argparse.Namespace:
    if method not in TTAVLM_BUILDERS:
        raise ValueError(f"Unsupported ttavlm method: {method}")
    default_batch_size = int(user_args.default_batch_size)
    exp_name = f"{method}_{dataset_name}_calibration_suite"
    flags = common_ttavlm_flags(
        user_args=user_args,
        dataset_name=dataset_name,
        exp_name=exp_name,
        save_root=save_root,
        batch_size=default_batch_size,
    )
    method_flags = TTAVLM_BUILDERS[method](family, dict(family_cfg), dataset_name.endswith("c"))
    method_flags = drop_flag(method_flags, "--batch_size")
    method_flags = drop_flag(method_flags, "--template_type")
    flags.extend(method_flags)
    runtime_args = parse_ttavlm_args(flags)
    runtime_args.eval_max_batches = user_args.max_eval_batches
    runtime_args.severity = [int(user_args.severity)]
    runtime_args.paper_batch_size = PAPER_BATCH_SIZES.get(method)
    runtime_args.batch_size_matches_paper = (
        runtime_args.paper_batch_size is None or int(runtime_args.batch_size) == int(runtime_args.paper_batch_size)
    )
    if not runtime_args.batch_size_matches_paper:
        print(
            "[suite][WARN] "
            f"{method} is running with batch_size={int(runtime_args.batch_size)} "
            f"but the repo's closed-set CIFAR benchmark uses batch_size={int(runtime_args.paper_batch_size)}.",
            flush=True,
        )
    return runtime_args


def build_eval_transform(base_transform, runtime_args: argparse.Namespace):
    if runtime_args.use_tta:
        if runtime_args.adaptation == "zero":
            views = [base_transform] + [add_tta_transform(base_transform, 224, style="zero") for _ in range(runtime_args.n_augment)]
        else:
            views = [base_transform] + [add_tta_transform(base_transform, 224) for _ in range(runtime_args.n_augment)]
        return TransformList(views)
    return TransformList([base_transform])


class CenteringState:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = int(num_classes)
        self.reset()

    def reset(self) -> None:
        self.running_sum: Tensor | None = None
        self.running_count = 0

    def center_for(self, scores: Tensor) -> Tensor:
        if scores.shape[0] > 1:
            return scores.mean(dim=0, keepdim=True)
        if self.running_sum is None or self.running_count <= 0:
            return torch.zeros((1, self.num_classes), device=scores.device, dtype=scores.dtype)
        return (self.running_sum / float(self.running_count)).to(device=scores.device, dtype=scores.dtype)

    def commit(self, scores: Tensor) -> None:
        with torch.no_grad():
            detached = scores.detach().sum(dim=0, keepdim=True).cpu()
            if self.running_sum is None:
                self.running_sum = detached
            else:
                self.running_sum = self.running_sum + detached
            self.running_count += int(scores.shape[0])


class TTAVLMLogitCalibrator:
    def __init__(
        self,
        *,
        tta_model,
        class_names: Sequence[str],
        prompt_mode: str,
    ) -> None:
        self.tta_model = tta_model
        self.class_names = list(class_names)
        self.prompt_mode = prompt_mode
        self.num_classes = len(class_names)
        self.state = CenteringState(self.num_classes)
        self.original_get_logits = tta_model.get_logits
        self.current_center: Tensor | None = None
        self.last_default_scores: Tensor | None = None

        device = tta_model.class_prototypes.device
        text_encoder = tta_model.clip_text_encoder.module if hasattr(tta_model.clip_text_encoder, "module") else tta_model.clip_text_encoder
        self.prompt_features: Dict[str, Tensor] = {
            name: encode_text_features(
                text_encoder=text_encoder,
                class_names=self.class_names,
                templates=[template],
                device=device,
            )
            for name, template in GENERIC_CORRUPTION_TEMPLATES
        }
        self.prompt_stack = torch.stack([self.prompt_features[name] for name, _ in GENERIC_CORRUPTION_TEMPLATES], dim=1)

    def reset(self) -> None:
        self.state.reset()
        self.current_center = None
        self.last_default_scores = None

    def _is_default_prototype_call(self, class_prototypes: Tensor | None) -> bool:
        if class_prototypes is None:
            return True
        default = self.tta_model.class_prototypes
        if class_prototypes.shape != default.shape:
            return False
        return class_prototypes.data_ptr() == default.data_ptr()

    def _default_scores(self, image_features: Tensor) -> Tensor:
        features = image_features.float()
        if self.prompt_mode == "default":
            return features @ self.tta_model.class_prototypes.float().t()
        if self.prompt_mode in self.prompt_features:
            prototypes = self.prompt_features[self.prompt_mode].to(device=features.device, dtype=features.dtype)
            return features @ prototypes.t()
        if self.prompt_mode == "generic_max":
            stack = self.prompt_stack.to(device=features.device, dtype=features.dtype)
            return torch.einsum("nd,cpd->ncp", features, stack).max(dim=-1).values
        raise ValueError(f"Unsupported prompt mode: {self.prompt_mode}")

    def prepare_for_batch(self, image_features: List[Tensor]) -> None:
        default_scores = self._default_scores(image_features[0])
        self.current_center = self.state.center_for(default_scores)
        self.last_default_scores = default_scores.detach()

    def commit_batch(self) -> None:
        if self.last_default_scores is not None:
            self.state.commit(self.last_default_scores)
        self.current_center = None
        self.last_default_scores = None

    def raw_source_logits(self, image_features: List[Tensor]) -> Tensor:
        return self.original_get_logits(image_features, self.tta_model.class_prototypes)[0]

    def used_logits(self, image_features: List[Tensor]) -> Tensor:
        if self.current_center is None:
            self.prepare_for_batch(image_features)
        return self.tta_model.get_logits(image_features, self.tta_model.class_prototypes)[0]

    def _hook(self, this_model, image_features: List[Tensor], class_prototypes: Tensor = None, class_bias: Tensor = None) -> List[Tensor]:
        del this_model
        if self._is_default_prototype_call(class_prototypes):
            logits: List[Tensor] = []
            for features in image_features:
                raw = self._default_scores(features)
                center = self.current_center
                if center is None:
                    center = self.state.center_for(raw)
                logits.append(raw - center.to(device=raw.device, dtype=raw.dtype))
            return logits

        raw_logits = self.original_get_logits(image_features, class_prototypes, class_bias)
        centered_logits: List[Tensor] = []
        for raw in raw_logits:
            if raw.shape[0] > 1:
                centered_logits.append(raw - raw.mean(dim=0, keepdim=True))
            else:
                centered_logits.append(raw)
        return centered_logits

    def install(self) -> str:
        self.tta_model.get_logits = types.MethodType(self._hook, self.tta_model)
        return f"prompt={self.prompt_mode}; cbc=batch_or_stream"


def reduce_logits_to_probs(tta_model, logits: List[Tensor]) -> Tensor:
    if tta_model.tta_reduction == "logits":
        reduced = [tta_model.logit_scale * item for item in logits]
        return torch.stack(reduced, dim=0).mean(dim=0).softmax(dim=-1)
    if tta_model.tta_reduction == "probs":
        reduced = [(tta_model.logit_scale * item).softmax(dim=-1) for item in logits]
        return torch.stack(reduced, dim=0).mean(dim=0)
    raise NotImplementedError(f"Unsupported tta_reduction: {tta_model.tta_reduction}")


def evaluate_ttavlm_tta_model(
    *,
    tta_model,
    loader,
    calibrator: TTAVLMLogitCalibrator | None,
    display_progress: bool,
    eval_max_batches: int | None,
) -> Dict[str, float]:
    total_count = 0
    source_correct = 0.0
    used_correct = 0.0
    final_top1_correct = 0.0
    final_top5_correct = 0.0
    pseudo_changed = 0.0
    source_conf_sum = 0.0
    used_conf_sum = 0.0

    method_name = tta_model.__class__.__name__
    dataset_name = loader.dataset.__class__.__name__[:-7]
    shift_type = getattr(loader.dataset, "shift_type", "unknown")
    iterable = lib.track(loader, f"{method_name} running on {dataset_name} / {shift_type}") if display_progress else loader

    for batch_index, batch in enumerate(iterable):
        if eval_max_batches is not None and batch_index >= eval_max_batches:
            break

        images = [img.cuda(non_blocking=True) for img in batch["image"]]
        labels = batch["target"].cuda(non_blocking=True)

        with torch.no_grad():
            image_features = tta_model.get_features(images)
            if calibrator is not None:
                calibrator.prepare_for_batch(image_features)
                raw_logits = calibrator.raw_source_logits(image_features)
                used_logits = calibrator.used_logits(image_features)
            else:
                raw_logits = tta_model.get_logits(image_features, tta_model.class_prototypes)[0]
                used_logits = raw_logits

            source_probs = (tta_model.logit_scale * raw_logits).softmax(dim=-1)
            used_probs = (tta_model.logit_scale * used_logits).softmax(dim=-1)
            source_pred = source_probs.argmax(dim=-1)
            used_pred = used_probs.argmax(dim=-1)
            source_conf = source_probs.max(dim=-1).values
            used_conf = used_probs.max(dim=-1).values

            batch_count = int(labels.shape[0])
            total_count += batch_count
            source_correct += float(source_pred.eq(labels).float().sum().item())
            used_correct += float(used_pred.eq(labels).float().sum().item())
            pseudo_changed += float(source_pred.ne(used_pred).float().sum().item())
            source_conf_sum += float(source_conf.sum().item())
            used_conf_sum += float(used_conf.sum().item())

        logits, _ = tta_model.forward(images, labels=labels)

        with torch.no_grad():
            probs = reduce_logits_to_probs(tta_model, logits)
            pred = probs.argmax(dim=-1)
            topk = min(5, probs.shape[-1])
            topk_pred = probs.topk(topk, dim=-1, largest=True, sorted=True).indices
            final_top1_correct += float(pred.eq(labels).float().sum().item())
            final_top5_correct += float(topk_pred.eq(labels.unsqueeze(1)).any(dim=1).float().sum().item())

        if calibrator is not None:
            calibrator.commit_batch()

    denom = max(total_count, 1)
    return {
        "num_samples": float(total_count),
        "source_pseudolabel_top1": 100.0 * source_correct / denom,
        "used_pseudolabel_top1": 100.0 * used_correct / denom,
        "pseudo_label_delta": 100.0 * (used_correct - source_correct) / denom,
        "pseudo_label_changed": 100.0 * pseudo_changed / denom,
        "source_mean_confidence": source_conf_sum / denom,
        "used_mean_confidence": used_conf_sum / denom,
        "final_top1": 100.0 * final_top1_correct / denom,
        "final_top5": 100.0 * final_top5_correct / denom,
    }


def evaluate_ttavlm_method(
    *,
    user_args: argparse.Namespace,
    family_cfg: Mapping[str, Any],
    method: str,
    variant: str,
    shift_types: Sequence[str],
    method_root: Path,
) -> List[Dict[str, object]]:
    dataset_name = str(family_cfg["corrupted_dataset"])
    clean_dataset_name = str(CLEAN_DATASETS[dataset_name])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runtime_args = build_ttavlm_runtime_args(
        user_args=user_args,
        family=user_args.family,
        family_cfg=family_cfg,
        method=method,
        dataset_name=dataset_name,
        save_root=method_root / "artifacts",
    )
    base_model, base_transform = return_base_model(
        name="clip-ViT-B/16",
        device=device,
        dataset=dataset_name,
        path_to_weights=str(method_root / "artifacts"),
        segments=runtime_args.segments,
    )
    _, clean_val_dataset = return_train_val_datasets(
        name=clean_dataset_name,
        data_dir=str(user_args.data_root),
        train_transform=base_transform,
        val_transform=base_transform,
    )
    template = get_template(dataset_name, runtime_args.template_type)
    tta_model = return_tta_model(
        runtime_args.adaptation,
        base_model,
        runtime_args,
        template,
        clean_val_dataset.class_names,
    )
    calibrator = None
    hook_desc = "raw_default_prompt"
    if variant == "calibrated":
        calibrator = TTAVLMLogitCalibrator(
            tta_model=tta_model,
            class_names=clean_val_dataset.class_names,
            prompt_mode=user_args.calibration_prompt_mode,
        )
        hook_desc = calibrator.install()

    eval_transform = build_eval_transform(base_transform, runtime_args)
    rows: List[Dict[str, object]] = []
    for seed in user_args.seeds:
        lib.fix_seed(seed)
        for shift_type in shift_types:
            if not runtime_args.fully_continual:
                tta_model.reset()
            if calibrator is not None:
                calibrator.reset()
            loader = build_loader(
                dataset_name=dataset_name,
                data_root=user_args.data_root,
                transform=eval_transform,
                shift_type=shift_type,
                severity=int(user_args.severity),
                batch_size=int(runtime_args.batch_size),
                workers=int(user_args.workers),
                seed=int(seed),
            )
            metrics = evaluate_ttavlm_tta_model(
                tta_model=tta_model,
                loader=loader,
                calibrator=calibrator,
                display_progress=bool(user_args.display_progress),
                eval_max_batches=user_args.max_eval_batches,
            )
            rows.append(
                {
                    "method": method,
                    "variant": variant,
                    "seed": int(seed),
                    "corruption": shift_type,
                    "severity": int(user_args.severity),
                    "top1": metrics["final_top1"],
                    "top5": metrics["final_top5"],
                    "source_pseudolabel_top1": metrics["source_pseudolabel_top1"],
                    "used_pseudolabel_top1": metrics["used_pseudolabel_top1"],
                    "pseudo_label_delta": metrics["pseudo_label_delta"],
                    "pseudo_label_changed": metrics["pseudo_label_changed"],
                    "source_mean_confidence": metrics["source_mean_confidence"],
                    "used_mean_confidence": metrics["used_mean_confidence"],
                    "num_samples": int(metrics["num_samples"]),
                    "batch_size": int(runtime_args.batch_size),
                    "paper_batch_size": (
                        int(runtime_args.paper_batch_size) if runtime_args.paper_batch_size is not None else ""
                    ),
                    "batch_size_matches_paper": bool(runtime_args.batch_size_matches_paper),
                    "hook": hook_desc,
                    "status": "ok",
                }
            )

    del tta_model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def compute_etta_logits_from_features(images_all_features: Tensor, clip_weights: Tensor, alpha: float) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    images_all_features = images_all_features / images_all_features.norm(dim=-1, keepdim=True)

    clip_logits = torch.einsum("bd,cpd->bcp", images_all_features, clip_weights)
    sorted_logits, sorted_indices = torch.sort(clip_logits, dim=-1, descending=True)
    num_top_prompts = max(1, int(sorted_logits.size(-1) * alpha))
    top_indices = sorted_indices[:, :, :num_top_prompts]

    batch_size, num_classes, top_k = top_indices.shape
    feature_dim = clip_weights.shape[-1]
    expanded_weights = clip_weights.unsqueeze(0).expand(batch_size, -1, -1, -1)
    gather_index = top_indices.unsqueeze(-1).expand(batch_size, num_classes, top_k, feature_dim)
    top_clip_weights = torch.gather(expanded_weights, dim=2, index=gather_index)
    top_clip_weights = top_clip_weights.mean(dim=2)
    top_clip_weights = top_clip_weights / top_clip_weights.norm(dim=2, keepdim=True)

    filtered_logits = torch.einsum("bd,bcd->bc", images_all_features, top_clip_weights)
    pred = (100.0 * filtered_logits).argmax(dim=-1)
    return images_all_features, filtered_logits, pred, top_clip_weights


def build_etta_prompt_weights(
    *,
    classnames: Sequence[str],
    clip_model,
    prompt_mode: str,
) -> Tensor:
    if prompt_mode == "default":
        templates = ["a photo of a {}"]
    elif prompt_mode == "distorted":
        templates = ["a distorted photo of a {}"]
    elif prompt_mode == "generic_max":
        templates = [template for _, template in GENERIC_CORRUPTION_TEMPLATES]
    else:
        raise ValueError(f"Unsupported ETTA prompt mode: {prompt_mode}")
    return clip_classifier(
        classnames=[name.replace("_", " ") for name in classnames],
        template=templates,
        clip_model=clip_model,
        json_file_path=None,
    )


def evaluate_etta_loader(
    *,
    loader,
    clip_model,
    raw_clip_weights: Tensor,
    calibration_clip_weights: Tensor | None,
    alpha: float,
    calibrated: bool,
    eval_max_batches: int | None,
) -> Dict[str, float]:
    num_classes = int(raw_clip_weights.size(0))
    state = CenteringState(num_classes)

    cache: Dict[int, Dict[str, Tensor | int]] = {}
    for class_index in range(num_classes):
        cache[class_index] = {
            "sum": 0,
            "weight": torch.zeros_like(raw_clip_weights[0][0]).unsqueeze(0),
        }

    source_correct = 0.0
    used_correct = 0.0
    final_top1_correct = 0.0
    final_top5_correct = 0.0
    pseudo_changed = 0.0
    source_conf_sum = 0.0
    used_conf_sum = 0.0
    total_count = 0

    used_weight_bank = raw_clip_weights if calibration_clip_weights is None else calibration_clip_weights

    for batch_index, (images, target) in enumerate(loader):
        if eval_max_batches is not None and batch_index >= eval_max_batches:
            break

        with torch.no_grad():
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            images_all_features = clip_model.encode_image(images)
            images_all_features = images_all_features / images_all_features.norm(dim=-1, keepdim=True)

            source_image_features, source_logits, source_pred, _ = compute_etta_logits_from_features(
                images_all_features,
                raw_clip_weights,
                alpha,
            )
            del source_image_features

            used_image_features, used_logits_raw, used_pred_uncentered, used_top_weights = compute_etta_logits_from_features(
                images_all_features,
                used_weight_bank,
                alpha,
            )
            del used_pred_uncentered

            if calibrated:
                center = state.center_for(used_logits_raw)
                used_logits = used_logits_raw - center.to(device=used_logits_raw.device, dtype=used_logits_raw.dtype)
            else:
                used_logits = used_logits_raw

            source_probs = (100.0 * source_logits).softmax(dim=-1)
            used_probs = (100.0 * used_logits).softmax(dim=-1)
            source_pred = source_probs.argmax(dim=-1)
            used_pred = used_probs.argmax(dim=-1)

            batch_count = int(target.shape[0])
            total_count += batch_count
            source_correct += float(source_pred.eq(target).float().sum().item())
            used_correct += float(used_pred.eq(target).float().sum().item())
            pseudo_changed += float(source_pred.ne(used_pred).float().sum().item())
            source_conf_sum += float(source_probs.max(dim=-1).values.sum().item())
            used_conf_sum += float(used_probs.max(dim=-1).values.sum().item())

            for sample_index in range(batch_count):
                sample_feature = used_image_features[sample_index : sample_index + 1]
                sample_logits = used_logits[sample_index : sample_index + 1]
                sample_target = target[sample_index : sample_index + 1]
                sample_pred = int(used_pred[sample_index].item())
                sample_top_weight = used_top_weights[sample_index, sample_pred].unsqueeze(0)

                etta_update_cache(cache, sample_pred, sample_feature, sample_top_weight)
                cache_logits = etta_compute_cache_logits(sample_feature, cache)
                cache_prob = (100 * cache_logits).softmax(dim=1) + 1e-6
                cache_entropy = -(cache_prob * cache_prob.log()).sum(dim=1)
                clip_prob = (100 * sample_logits).softmax(dim=1) + 1e-6
                clip_entropy = -(clip_prob * clip_prob.log()).sum(dim=1)
                merged_logit = (
                    (clip_entropy / (clip_entropy + cache_entropy)) * cache_logits
                    + (cache_entropy / (clip_entropy + cache_entropy)) * sample_logits
                )
                final_pred = merged_logit.argmax(dim=-1)
                topk = min(5, merged_logit.shape[-1])
                topk_pred = merged_logit.topk(topk, dim=-1, largest=True, sorted=True).indices
                final_top1_correct += float(final_pred.eq(sample_target).float().sum().item())
                final_top5_correct += float(topk_pred.eq(sample_target.unsqueeze(1)).any(dim=1).float().sum().item())

        if calibrated:
            state.commit(used_logits_raw)

    denom = max(total_count, 1)
    return {
        "num_samples": float(total_count),
        "source_pseudolabel_top1": 100.0 * source_correct / denom,
        "used_pseudolabel_top1": 100.0 * used_correct / denom,
        "pseudo_label_delta": 100.0 * (used_correct - source_correct) / denom,
        "pseudo_label_changed": 100.0 * pseudo_changed / denom,
        "source_mean_confidence": source_conf_sum / denom,
        "used_mean_confidence": used_conf_sum / denom,
        "final_top1": 100.0 * final_top1_correct / denom,
        "final_top5": 100.0 * final_top5_correct / denom,
    }


def evaluate_etta_method(
    *,
    user_args: argparse.Namespace,
    method: str,
    variant: str,
    shift_types: Sequence[str],
    method_root: Path,
) -> List[Dict[str, object]]:
    del method_root
    if method != "etta":
        raise ValueError("ETTA evaluator called for non-ETTA method")
    dataset_name = str(CLOSED_SET_FAMILIES[user_args.family]["corrupted_dataset"])
    if dataset_name not in ETTA_DATASET_TO_FAMILY:
        raise ValueError(f"ETTA does not support dataset {dataset_name}")

    clip_model, preprocess = etta_clip.load(user_args.etta_backbone)
    clip_model.eval()

    family = ETTA_DATASET_TO_FAMILY[dataset_name]
    family_cfg = ETTA_FAMILIES[family]
    clean_root = user_args.data_root / str(family_cfg["clean_dir"])
    reference_dataset = family_cfg["dataset_cls"](
        root=str(clean_root),
        train=False,
        download=False,
    )
    classnames = [name.replace("_", " ") for name in reference_dataset.classes]

    raw_clip_weights = build_etta_prompt_weights(
        classnames=classnames,
        clip_model=clip_model,
        prompt_mode="default",
    )
    calibration_clip_weights = None
    hook_desc = "raw_default_prompt"
    if variant == "calibrated":
        calibration_clip_weights = build_etta_prompt_weights(
            classnames=classnames,
            clip_model=clip_model,
            prompt_mode=user_args.calibration_prompt_mode,
        )
        hook_desc = f"prompt={user_args.calibration_prompt_mode}; cbc=batch_or_stream"

    rows: List[Dict[str, object]] = []
    for seed in user_args.seeds:
        set_etta_seed(int(seed))
        for shift_type in shift_types:
            loader_args = argparse.Namespace(
                dataset=dataset_name,
                data_root=user_args.data_root,
                severity=int(user_args.severity),
                workers=int(user_args.workers),
                batch_size=int(user_args.default_batch_size),
            )
            loader = build_etta_loader(loader_args, shift_type, preprocess, int(seed))
            metrics = evaluate_etta_loader(
                loader=loader,
                clip_model=clip_model,
                raw_clip_weights=raw_clip_weights,
                calibration_clip_weights=calibration_clip_weights,
                alpha=float(user_args.etta_alpha),
                calibrated=(variant == "calibrated"),
                eval_max_batches=user_args.max_eval_batches,
            )
            rows.append(
                {
                    "method": "etta",
                    "variant": variant,
                    "seed": int(seed),
                    "corruption": shift_type,
                    "severity": int(user_args.severity),
                    "top1": metrics["final_top1"],
                    "top5": metrics["final_top5"],
                    "source_pseudolabel_top1": metrics["source_pseudolabel_top1"],
                    "used_pseudolabel_top1": metrics["used_pseudolabel_top1"],
                    "pseudo_label_delta": metrics["pseudo_label_delta"],
                    "pseudo_label_changed": metrics["pseudo_label_changed"],
                    "source_mean_confidence": metrics["source_mean_confidence"],
                    "used_mean_confidence": metrics["used_mean_confidence"],
                    "num_samples": int(metrics["num_samples"]),
                    "batch_size": int(user_args.default_batch_size),
                    "hook": hook_desc,
                    "status": "ok",
                }
            )

    del clip_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def evaluate_method_variant(
    *,
    user_args: argparse.Namespace,
    family_cfg: Mapping[str, Any],
    method: str,
    variant: str,
    shift_types: Sequence[str],
    variant_root: Path,
) -> List[Dict[str, object]]:
    if method == "etta":
        return evaluate_etta_method(
            user_args=user_args,
            method=method,
            variant=variant,
            shift_types=shift_types,
            method_root=variant_root,
        )
    return evaluate_ttavlm_method(
        user_args=user_args,
        family_cfg=family_cfg,
        method=method,
        variant=variant,
        shift_types=shift_types,
        method_root=variant_root,
    )


def summarize(rows: List[Dict[str, object]], methods: Sequence[str]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    grouped: Dict[str, Dict[Tuple[str, str], List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["corruption"])][(str(row["method"]), str(row["variant"]))].append(row)

    comparison_rows: List[Dict[str, object]] = []
    for corruption, by_method_variant in sorted(grouped.items()):
        row: Dict[str, object] = {"corruption": corruption}
        for method in methods:
            raw_items = by_method_variant.get((method, "raw"), [])
            cal_items = by_method_variant.get((method, "calibrated"), [])
            row[f"{method}_raw_top1"] = mean([float(item["top1"]) for item in raw_items])
            row[f"{method}_calibrated_top1"] = mean([float(item["top1"]) for item in cal_items])
            row[f"{method}_delta"] = row[f"{method}_calibrated_top1"] - row[f"{method}_raw_top1"]
            row[f"{method}_raw_pseudo_top1"] = mean([float(item["used_pseudolabel_top1"]) for item in raw_items])
            row[f"{method}_calibrated_pseudo_top1"] = mean([float(item["used_pseudolabel_top1"]) for item in cal_items])
            row[f"{method}_pseudo_delta_gain"] = (
                row[f"{method}_calibrated_pseudo_top1"] - row[f"{method}_raw_pseudo_top1"]
            )
        comparison_rows.append(row)

    summary_rows: List[Dict[str, object]] = []
    for method in methods:
        method_row: Dict[str, object] = {"method": method, "num_corruptions": len(comparison_rows)}
        raw_top1 = [float(row[f"{method}_raw_top1"]) for row in comparison_rows]
        cal_top1 = [float(row[f"{method}_calibrated_top1"]) for row in comparison_rows]
        delta_top1 = [float(row[f"{method}_delta"]) for row in comparison_rows]
        raw_pseudo = [float(row[f"{method}_raw_pseudo_top1"]) for row in comparison_rows]
        cal_pseudo = [float(row[f"{method}_calibrated_pseudo_top1"]) for row in comparison_rows]
        pseudo_gain = [float(row[f"{method}_pseudo_delta_gain"]) for row in comparison_rows]
        method_row["mean_raw_top1"] = mean(raw_top1)
        method_row["mean_calibrated_top1"] = mean(cal_top1)
        method_row["mean_delta_top1"] = mean(delta_top1)
        method_row["std_delta_top1"] = std(delta_top1)
        method_row["wins"] = int(sum(delta > 0.0 for delta in delta_top1))
        method_row["mean_raw_pseudo_top1"] = mean(raw_pseudo)
        method_row["mean_calibrated_pseudo_top1"] = mean(cal_pseudo)
        method_row["mean_pseudo_gain"] = mean(pseudo_gain)
        summary_rows.append(method_row)
    return comparison_rows, summary_rows


def main() -> None:
    args = parse_args()
    args.methods = resolve_methods(args.methods)
    shift_types = resolve_shift_types(args.shift_types)
    family_cfg = CLOSED_SET_FAMILIES[args.family]

    run_root = args.output_root / args.run_name / args.family / f"severity_{args.severity}"
    ensure_dir(run_root)
    lib.setup_logger(str(run_root / "run.log"))

    metadata = {
        "family": args.family,
        "severity": int(args.severity),
        "shift_types": shift_types,
        "methods": list(args.methods),
        "variants": list(args.variants),
        "workers": int(args.workers),
        "seeds": list(args.seeds),
        "steps": int(args.steps),
        "max_eval_batches": args.max_eval_batches,
        "data_root": str(args.data_root),
        "calibration_prompt_mode": args.calibration_prompt_mode,
        "etta_backbone": args.etta_backbone,
        "etta_alpha": float(args.etta_alpha),
        "note": (
            "Raw keeps the original logits. Calibrated uses prompt-aware batchwise centering only."
        ),
    }
    write_json(run_root / "metadata.json", metadata)
    if args.dry_run:
        lib.LOGGER.info("Dry run configuration:\n%s", json.dumps(metadata, indent=2))
        return

    all_rows: List[Dict[str, object]] = []
    for method in args.methods:
        method_root = run_root / method
        ensure_dir(method_root)
        for variant in args.variants:
            variant_root = method_root / variant
            ensure_dir(variant_root / "artifacts")
            result_path = variant_root / "results.csv"
            if args.skip_existing and result_path.exists():
                with result_path.open() as handle:
                    method_rows = list(csv.DictReader(handle))
            else:
                lib.LOGGER.info("Running method=%s variant=%s", method, variant)
                method_rows = evaluate_method_variant(
                    user_args=args,
                    family_cfg=family_cfg,
                    method=method,
                    variant=variant,
                    shift_types=shift_types,
                    variant_root=variant_root,
                )
                write_csv(result_path, method_rows)
            all_rows.extend(method_rows)

    comparison_rows, summary_rows = summarize(all_rows, args.methods)
    write_csv(run_root / "all_results.csv", all_rows)
    write_csv(run_root / "comparison_by_corruption.csv", comparison_rows)
    write_csv(run_root / "summary.csv", summary_rows)
    write_json(run_root / "summary.json", {"summary": summary_rows, "comparison": comparison_rows})
    lib.LOGGER.info("Baseline calibration suite complete: %s", run_root)
    lib.LOGGER.info("Summary: %s", summary_rows)


if __name__ == "__main__":
    main()
