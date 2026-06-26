#!/usr/bin/env python3
"""Evaluate additional online/offline CLIP zero-shot baselines from caches.

This script intentionally reuses cached CLIP image features/logits so that
screening extra baselines does not require re-encoding CIFAR-10-C images.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "work" / ".mplconfig"),
)

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_zero_shot_enhancement_baselines import (  # noqa: E402
    choose_device,
    default_prompts,
    evaluate_cbc,
    text_features_from_prompts,
    waffle_prompts,
)
from run_closed_set_cifar_benchmarks import FAMILIES  # noqa: E402
from ttavlm.datasets import CORRUPTIONS, return_train_val_datasets  # noqa: E402
from ttavlm.datasets.utils import all_templates, templates as select_templates  # noqa: E402
from ttavlm.models import return_base_model  # noqa: E402


DEFAULT_COMPLETE_ROOT = WORKSPACE_ROOT / "complete"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "work" / "additional_zero_shot_baseline_screening"
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
    parser.add_argument("--complete-root", type=Path, default=DEFAULT_COMPLETE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="cifar10c_s5_additional_baselines")
    parser.add_argument("--external-root", type=Path, default=WORKSPACE_ROOT / "external_zero_shot_baselines")
    parser.add_argument("--base-model-name", type=str, default="clip-ViT-B/16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lp-k", type=int, default=20)
    parser.add_argument("--lp-alpha", type=float, default=0.5)
    parser.add_argument("--lp-iters", type=int, default=10)
    parser.add_argument("--skip-label-prop", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    ensure(path.parent)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_feature_cache(path: Path) -> Tuple[torch.Tensor, torch.Tensor, List[str] | None]:
    payload = torch.load(path, map_location="cpu")
    return payload["features"].float(), payload["labels"].long(), payload.get("corruptions")


def load_logit_cache(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    return payload["logits"].float(), payload["labels"].long()


def accuracy(logits_or_probs: torch.Tensor, labels: torch.Tensor) -> float:
    return 100.0 * float(logits_or_probs.argmax(dim=1).eq(labels.cpu()).float().mean().item())


def prompt_lists_from_templates(template_list: Sequence[str]) -> List[List[str]]:
    return [[template.format(name.replace("_", " ")) for template in template_list] for name in CIFAR10_CLASSES]


def prompt_tensor_from_templates(model, template_list: Sequence[str], device: torch.device) -> torch.Tensor:
    class_prompt_features: List[torch.Tensor] = []
    for class_name in CIFAR10_CLASSES:
        prompts = [[template.format(class_name.replace("_", " "))] for template in template_list]
        prompt_features = [
            text_features_from_prompts(model=model, prompts_by_class=[prompt], device=device).squeeze(0)
            for prompt in prompts
        ]
        class_prompt_features.append(torch.stack(prompt_features, dim=0).float())
    return torch.stack(class_prompt_features, dim=0).float()


def score_with_weights(
    features: torch.Tensor,
    class_weights: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    weights = class_weights.to(device=device, dtype=torch.float32)
    logits: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            x = features[start : start + batch_size].to(device=device, dtype=torch.float32)
            logits.append((100.0 * x @ weights.t()).cpu())
    return torch.cat(logits, dim=0)


def score_logit_ensemble(
    features: torch.Tensor,
    prompt_features: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    adaptive: bool = False,
    adaptive_tau: float = 0.1,
) -> torch.Tensor:
    text = prompt_features.to(device=device, dtype=torch.float32)
    logits: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            x = features[start : start + batch_size].to(device=device, dtype=torch.float32)
            per_prompt = 100.0 * torch.einsum("bd,cpd->bcp", x, text)
            if adaptive:
                probs = per_prompt.softmax(dim=1)
                confidence = probs.max(dim=1).values
                weights = F.softmax(confidence / float(adaptive_tau), dim=1)
                out = torch.einsum("bp,bcp->bc", weights, per_prompt)
            else:
                out = per_prompt.mean(dim=2)
            logits.append(out.cpu())
    return torch.cat(logits, dim=0)


def sinkhorn(scores: torch.Tensor, tau: float = 0.01, iterations: int = 20, device: torch.device | None = None) -> torch.Tensor:
    work_device = device or torch.device("cpu")
    probs_chunks: List[torch.Tensor] = []
    for start in range(0, scores.shape[0], 65536):
        probs_chunks.append(F.softmax(scores[start : start + 65536].to(work_device) / float(tau), dim=1).cpu())
    probs = torch.cat(probs_chunks, dim=0).to(work_device)
    rows, cols = probs.shape
    probs = probs / float(rows)
    for _ in range(int(iterations)):
        probs = probs / probs.sum(dim=0, keepdim=True).clamp_min(1e-12)
        probs = probs / float(cols)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
        probs = probs / float(rows)
    return (probs * float(rows)).cpu()


def prior_normalize(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    probs = F.softmax(logits.float() / float(temperature), dim=1)
    prior = probs.mean(dim=0, keepdim=True).clamp_min(1e-12)
    adjusted = torch.log(probs.clamp_min(1e-12)) - torch.log(prior)
    return adjusted


def label_propagation_group(
    features: torch.Tensor,
    logits: torch.Tensor,
    *,
    device: torch.device,
    k: int,
    alpha: float,
    iterations: int,
    chunk_size: int = 512,
) -> torch.Tensor:
    x = F.normalize(features.float(), dim=1).to(device)
    p0 = F.softmax(logits.float().to(device) / 10.0, dim=1)
    num_samples = x.shape[0]
    p = p0.clone()
    neighbors = []
    weights = []
    with torch.no_grad():
        for start in range(0, num_samples, chunk_size):
            sim = x[start : start + chunk_size] @ x.t()
            top_values, top_indices = torch.topk(sim, k=int(k) + 1, dim=1)
            top_values = top_values[:, 1:]
            top_indices = top_indices[:, 1:]
            weights.append(F.softmax(top_values / 0.05, dim=1))
            neighbors.append(top_indices)
    for _ in range(int(iterations)):
        next_p = torch.empty_like(p)
        for start, (idx, w) in zip(range(0, num_samples, chunk_size), zip(neighbors, weights)):
            next_p[start : start + idx.shape[0]] = (w.unsqueeze(-1) * p[idx]).sum(dim=1)
        p = float(alpha) * next_p + (1.0 - float(alpha)) * p0
        p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return p.cpu()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    device = choose_device(args.device)
    output_root = args.output_root / args.run_name
    ensure(output_root)

    complete = args.complete_root
    feature_cache = complete / "zero_shot_enhancement_baselines/cache/cifar10c_severity5_all_global_features.pt"
    clean_feature_cache = complete / "clip_zero_shot_claim/visualizations/feature_cluster_pca_3d/cifar10/severity_5/cache/clean_all.pt"
    logit_cache = complete / "clip_zero_shot_claim/cache/logits"
    features, labels, groups = load_feature_cache(feature_cache)
    clean_features, clean_labels, _ = load_feature_cache(clean_feature_cache)
    if groups is None:
        groups = []
        for corruption in CORRUPTIONS:
            groups.extend([corruption] * 10000)

    family_cfg = FAMILIES["cifar10"]
    model, _ = return_base_model(
        name=args.base_model_name,
        device=device,
        dataset=str(family_cfg["corrupted_dataset"]),
        path_to_weights=str(REPO_ROOT / "work"),
    )
    model.eval()
    if hasattr(model, "visual") and not hasattr(model.visual, "use_local"):
        model.visual.use_local = False

    log("encoding text baselines")
    default_text = text_features_from_prompts(model=model, prompts_by_class=default_prompts(CIFAR10_CLASSES), device=device)
    corrupted_text = text_features_from_prompts(
        model=model,
        prompts_by_class=prompt_lists_from_templates(["a corrupted photo of a {}"]),
        device=device,
    )
    select_prompt_tensor = prompt_tensor_from_templates(model, select_templates, device)
    all_prompt_tensor = prompt_tensor_from_templates(model, all_templates, device)
    waffle_prompt_tensor = torch.stack(
        [
            text_features_from_prompts(
                model=model,
                prompts_by_class=waffle_prompts(
                    class_names=CIFAR10_CLASSES,
                    external_root=args.external_root,
                    seed=int(args.seed) + rep,
                    waffle_count=15,
                ),
                device=device,
            )
            for rep in range(3)
        ],
        dim=0,
    ).mean(dim=0)
    waffle_plus_all_tensor = F.normalize(
        torch.cat([waffle_prompt_tensor.unsqueeze(1), all_prompt_tensor], dim=1).mean(dim=1),
        dim=-1,
    )

    online_specs: List[Tuple[str, str, torch.Tensor | None, torch.Tensor | None, bool]] = [
        ("CLIP", "reference_default_prompt", default_text, None, False),
        ("CorruptedPrompt", "generic_corruption_prompt", corrupted_text, None, False),
        ("PromptEns-8", "CLIP_select_prompt_feature_average", F.normalize(select_prompt_tensor.mean(dim=1), dim=-1), None, False),
        ("PromptEns-80", "CLIP_all_prompt_feature_average", F.normalize(all_prompt_tensor.mean(dim=1), dim=-1), None, False),
        ("LogitEns-8", "CLIP_select_prompt_logit_average", None, select_prompt_tensor, False),
        ("LogitEns-80", "CLIP_all_prompt_logit_average", None, all_prompt_tensor, False),
        ("AdaptivePromptEns-80", "AutoCLIP_style_per_image_prompt_weighting", None, all_prompt_tensor, True),
        ("Waffle+PromptEns", "WaffleCLIP_descriptors_plus_CLIP_prompt_ensemble", waffle_plus_all_tensor, None, False),
    ]
    online_rows: List[Dict[str, object]] = []
    clean_rows: List[Dict[str, object]] = []
    for name, note, weights, prompt_tensor, adaptive in online_specs:
        log(f"online {name}")
        if prompt_tensor is not None:
            logits = score_logit_ensemble(
                features,
                prompt_tensor,
                device=device,
                batch_size=int(args.batch_size),
                adaptive=adaptive,
            )
            clean_logits = score_logit_ensemble(
                clean_features,
                prompt_tensor,
                device=device,
                batch_size=int(args.batch_size),
                adaptive=adaptive,
            )
        else:
            assert weights is not None
            logits = score_with_weights(features, weights, device=device, batch_size=int(args.batch_size))
            clean_logits = score_with_weights(clean_features, weights, device=device, batch_size=int(args.batch_size))
        clip_top1 = accuracy(score_with_weights(features, default_text, device=device, batch_size=int(args.batch_size)), labels)
        clean_clip_top1 = accuracy(score_with_weights(clean_features, default_text, device=device, batch_size=int(args.batch_size)), clean_labels)
        overall = accuracy(logits, labels)
        online_rows.append(
            {
                "setting": "severity5:overall",
                "method": name,
                "top1": overall,
                "delta_vs_clip": overall - clip_top1,
                "num_samples": int(labels.numel()),
                "category": "online",
                "note": note,
            }
        )
        clean_top1 = accuracy(clean_logits, clean_labels)
        clean_rows.append(
            {
                "setting": "clean",
                "method": name,
                "top1": clean_top1,
                "delta_vs_clip": clean_top1 - clean_clip_top1,
                "num_samples": int(clean_labels.numel()),
                "category": "online",
                "note": note,
            }
        )
        by_corr: List[Dict[str, object]] = []
        offset = 0
        for corruption in CORRUPTIONS:
            n = 10000
            sub_logits = logits[offset : offset + n]
            sub_labels = labels[offset : offset + n]
            clip_logits, _ = load_logit_cache(logit_cache / f"severity5_{corruption}_all.pt")
            clip_score = accuracy(clip_logits, sub_labels)
            top1 = accuracy(sub_logits, sub_labels)
            by_corr.append(
                {
                    "setting": f"severity5:{corruption}",
                    "corruption": corruption,
                    "method": name,
                    "top1": top1,
                    "delta_vs_clip": top1 - clip_score,
                    "num_samples": n,
                    "category": "online",
                    "note": note,
                }
            )
            offset += n
        write_csv(output_root / f"online_{name.lower().replace('+', '_plus_').replace('-', '_')}_by_corruption.csv", by_corr)
    write_csv(output_root / "online_additional_overall_s5.csv", online_rows)
    write_csv(output_root / "online_additional_clean.csv", clean_rows)

    log("offline baselines")
    default_logits = score_with_weights(features, default_text, device=device, batch_size=int(args.batch_size))
    clip_top1 = accuracy(default_logits, labels)
    offline_rows: List[Dict[str, object]] = [
        {
            "setting": "severity5:overall",
            "method": "CLIP",
            "top1": clip_top1,
            "delta_vs_clip": 0.0,
            "num_samples": int(labels.numel()),
            "category": "offline_reference",
            "note": "default CLIP logits",
        },
        {
            "setting": "severity5:overall",
            "method": "CBC Offline",
            "top1": accuracy(default_logits - default_logits.mean(dim=0, keepdim=True), labels),
            "delta_vs_clip": accuracy(default_logits - default_logits.mean(dim=0, keepdim=True), labels) - clip_top1,
            "num_samples": int(labels.numel()),
            "category": "offline",
            "note": "global logit center subtraction",
        },
    ]
    sinkhorn_probs = sinkhorn(default_logits, tau=0.01, iterations=20, device=device)
    top1 = accuracy(sinkhorn_probs, labels)
    offline_rows.append(
        {
            "setting": "severity5:overall",
            "method": "Sinkhorn-OT",
            "top1": top1,
            "delta_vs_clip": top1 - clip_top1,
            "num_samples": int(labels.numel()),
            "category": "offline",
            "note": "balanced optimal-transport assignment on all logits",
        }
    )
    prior_logits = prior_normalize(default_logits, temperature=1.0)
    top1 = accuracy(prior_logits, labels)
    offline_rows.append(
        {
            "setting": "severity5:overall",
            "method": "PriorNorm",
            "top1": top1,
            "delta_vs_clip": top1 - clip_top1,
            "num_samples": int(labels.numel()),
            "category": "offline",
            "note": "global predicted-prior normalization",
        }
    )
    if not args.skip_label_prop:
        lp_correct = 0
        lp_count = 0
        by_corr_lp: List[Dict[str, object]] = []
        for corruption_index, corruption in enumerate(CORRUPTIONS):
            log(f"label propagation {corruption}")
            start = corruption_index * 10000
            end = start + 10000
            probs = label_propagation_group(
                features[start:end],
                default_logits[start:end],
                device=device,
                k=int(args.lp_k),
                alpha=float(args.lp_alpha),
                iterations=int(args.lp_iters),
            )
            sub_labels = labels[start:end]
            top1_corr = accuracy(probs, sub_labels)
            lp_correct += int(probs.argmax(dim=1).eq(sub_labels).sum().item())
            lp_count += int(sub_labels.numel())
            clip_corr = accuracy(default_logits[start:end], sub_labels)
            by_corr_lp.append(
                {
                    "setting": f"severity5:{corruption}",
                    "corruption": corruption,
                    "method": "LabelProp-kNN",
                    "top1": top1_corr,
                    "delta_vs_clip": top1_corr - clip_corr,
                    "num_samples": int(sub_labels.numel()),
                    "category": "offline",
                    "note": f"k={args.lp_k}, alpha={args.lp_alpha}, iters={args.lp_iters}",
                }
            )
        top1 = 100.0 * float(lp_correct) / float(max(lp_count, 1))
        offline_rows.append(
            {
                "setting": "severity5:overall",
                "method": "LabelProp-kNN",
                "top1": top1,
                "delta_vs_clip": top1 - clip_top1,
                "num_samples": int(lp_count),
                "category": "offline",
                "note": f"per-corruption kNN propagation, k={args.lp_k}, alpha={args.lp_alpha}, iters={args.lp_iters}",
            }
        )
        write_csv(output_root / "offline_labelprop_by_corruption_s5.csv", by_corr_lp)
    write_csv(output_root / "offline_additional_overall_s5.csv", offline_rows)
    write_json(
        output_root / "metadata.json",
        {
            "feature_cache": str(feature_cache),
            "clean_feature_cache": str(clean_feature_cache),
            "logit_cache": str(logit_cache),
            "online_methods": [row[0] for row in online_specs],
            "offline_methods": [row["method"] for row in offline_rows],
            "note": "Additional baseline screening; methods marked style/local are not claimed as exact official reproductions.",
        },
    )
    log(f"wrote {output_root}")


if __name__ == "__main__":
    main()
