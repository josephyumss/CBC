#!/usr/bin/env python3
"""Evaluate CLIP zero-shot enhancement baselines on CIFAR-10-C."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "work" / ".mplconfig"),
)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from run_closed_set_cifar_benchmarks import DEFAULT_DATA_ROOT, FAMILIES  # noqa: E402
from ttavlm.datasets import CORRUPTIONS, return_train_val_datasets  # noqa: E402
from ttavlm.models import return_base_model  # noqa: E402
from ttavlm.models.clip import tokenize  # noqa: E402


DEFAULT_TEMPLATE = "a photo of a {}"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "work" / "zero_shot_enhancement_baselines"
DEFAULT_EXTERNAL_ROOT = WORKSPACE_ROOT / "external_zero_shot_baselines"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["cifar10"], default="cifar10")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="cifar10c_s5_zero_shot_enhancers")
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--base-model-name", type=str, default="clip-ViT-B/16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--max-samples-per-corruption", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


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


def build_loader(dataset: Dataset, batch_size: int, workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def batch_to_labels(batch: Mapping[str, object]) -> torch.Tensor:
    labels = batch["target"]
    if isinstance(labels, torch.Tensor):
        return labels.long()
    return torch.as_tensor(labels, dtype=torch.long)


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


def maybe_subset(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return torch.utils.data.Subset(dataset, list(range(int(max_samples))))


def load_cifar10_class_names(data_root: Path, transform) -> List[str]:
    _, dataset = return_train_val_datasets(
        name="cifar10",
        data_dir=str(data_root),
        train_transform=transform,
        val_transform=transform,
    )
    return list(dataset.class_names)


def encode_prompt_list(model: torch.nn.Module, prompts: Sequence[str], device: torch.device) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(prompts), 256):
            batch_prompts = list(prompts[start : start + 256])
            tokens = tokenize(batch_prompts, truncate=True).to(device)
            features = model.encode_text(tokens).float()
            features = F.normalize(features, dim=-1)
            chunks.append(features.cpu())
    return torch.cat(chunks, dim=0)


def text_features_from_prompts(
    *,
    model: torch.nn.Module,
    prompts_by_class: Sequence[Sequence[str]],
    device: torch.device,
) -> torch.Tensor:
    class_features: List[torch.Tensor] = []
    for prompts in prompts_by_class:
        features = encode_prompt_list(model, prompts, device)
        class_feature = F.normalize(features.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
        class_features.append(class_feature)
    return torch.stack(class_features, dim=0).float()


def default_prompts(class_names: Sequence[str]) -> List[List[str]]:
    return [[DEFAULT_TEMPLATE.format(name.replace("_", " "))] for name in class_names]


def cupl_prompts(class_names: Sequence[str], external_root: Path) -> List[List[str]]:
    prompt_path = external_root / "CuPL" / "all_prompts" / "full_prompts" / "cifar10_prompts_full.json"
    with prompt_path.open() as handle:
        prompt_data = json.load(handle)
    prompts: List[List[str]] = []
    for class_name in class_names:
        key = class_name.replace("_", " ")
        if key not in prompt_data:
            raise KeyError(f"CuPL prompt file is missing class {key}")
        prompts.append(list(prompt_data[key]))
    return prompts


def make_descriptor_sentence(descriptor: str) -> str:
    descriptor = descriptor.strip()
    if descriptor.startswith("a") or descriptor.startswith("an"):
        return f"which is {descriptor}"
    if descriptor.startswith(("has", "often", "typically", "may", "can")):
        return f"which {descriptor}"
    if descriptor.startswith("used"):
        return f"which is {descriptor}"
    return f"which has {descriptor}"


def waffle_prompts(
    *,
    class_names: Sequence[str],
    external_root: Path,
    seed: int,
    waffle_count: int,
) -> List[List[str]]:
    rng = np.random.default_rng(seed)
    word_path = external_root / "WaffleCLIP" / "word_list.pkl"
    with word_path.open("rb") as handle:
        word_list = pickle.load(handle)
    word_list = [str(word) for word in word_list if str(word).strip()]
    avg_num_words = max(1, int(np.round(np.mean([len(name.replace("_", " ").split()) for name in class_names]))))
    avg_word_length = max(1, int(np.round(np.mean([np.mean([len(part) for part in name.replace("_", " ").split()]) for name in class_names]))))
    clipped_words = [word[:avg_word_length] for word in word_list if len(word) > 0]
    character_list = np.array(list("abcdefghijklmnopqrstuvwxyz"))
    num_spaces = int(np.round(np.mean([name.count("_") + name.count(" ") for name in class_names]))) + 1
    num_chars = int(np.ceil(np.mean([max(len(part) for part in name.replace("_", " ").split()) for name in class_names])))
    num_chars += num_spaces - (num_chars % num_spaces)
    sample_key_parts = ["a" * (num_chars // num_spaces) for _ in range(num_spaces)]
    sample_key = " ".join(sample_key_parts)

    shared_descriptors: List[str] = []
    for _ in range(int(waffle_count)):
        sampled_words = rng.choice(clipped_words, size=avg_num_words, replace=False)
        shared_descriptors.append(" ".join(str(word) for word in sampled_words))
        noise_chars: List[str] = []
        for char in sample_key:
            if char == " ":
                noise_chars.append(", ")
            else:
                noise_chars.append(str(rng.choice(character_list)))
        shared_descriptors.append("".join(noise_chars))

    prompts_by_class: List[List[str]] = []
    for class_name in class_names:
        display = class_name.replace("_", " ")
        prompts_by_class.append(
            [f"A photo of a {display}, {make_descriptor_sentence(descriptor)}." for descriptor in shared_descriptors]
        )
    return prompts_by_class


def extract_or_load_global_features(
    *,
    cache_dir: Path,
    model: torch.nn.Module,
    transform,
    data_root: Path,
    severity: int,
    batch_size: int,
    workers: int,
    device: torch.device,
    skip_existing: bool,
    max_samples_per_corruption: int | None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    sample_tag = "all" if max_samples_per_corruption is None else f"first{int(max_samples_per_corruption)}"
    cache_path = cache_dir / f"cifar10c_severity{severity}_{sample_tag}_global_features.pt"
    if skip_existing and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        log(f"loaded global feature cache: {cache_path}")
        return payload["features"].float(), payload["labels"].long(), list(payload["corruptions"])

    features: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    groups: List[str] = []
    model.visual.use_local = False
    model.eval()
    for corruption in CORRUPTIONS:
        dataset = maybe_subset(
            load_cifar10c_dataset(data_root, transform, corruption, severity),
            max_samples_per_corruption,
        )
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
                groups.extend([corruption] * int(batch_labels.numel()))
                seen += int(batch_labels.numel())
                if batch_index == 0 or seen == total or batch_index % 20 == 0:
                    log(f"global features {corruption}: {seen}/{total}")

    feature_tensor = torch.cat(features, dim=0).float()
    label_tensor = torch.cat(labels, dim=0).long()
    ensure_dir(cache_path.parent)
    torch.save({"features": feature_tensor, "labels": label_tensor, "corruptions": groups}, cache_path)
    log(f"saved global feature cache: {cache_path}")
    return feature_tensor, label_tensor, groups


def evaluate_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return 100.0 * float(logits.argmax(dim=1).eq(labels).float().mean().item())


def evaluate_cbc(logits: torch.Tensor, labels: torch.Tensor, batch_size: int, seed: int) -> float:
    num_samples = int(labels.numel())
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    order = torch.randperm(num_samples, generator=generator)
    num_classes = int(logits.shape[1])
    running_sum = torch.zeros((1, num_classes), dtype=torch.float64)
    running_count = 0
    cbc_pred = torch.empty(num_samples, dtype=torch.long)
    for start in range(0, num_samples, int(batch_size)):
        idx = order[start : start + int(batch_size)]
        batch_logits = logits[idx].float()
        if batch_logits.shape[0] > 1:
            center = batch_logits.mean(dim=0, keepdim=True)
        elif running_count > 0:
            center = (running_sum / float(running_count)).to(dtype=batch_logits.dtype)
        else:
            center = torch.zeros((1, num_classes), dtype=batch_logits.dtype)
        cbc_pred[idx] = (batch_logits - center).argmax(dim=1).cpu()
        running_sum += batch_logits.double().sum(dim=0, keepdim=True).cpu()
        running_count += int(batch_logits.shape[0])
    return 100.0 * float(cbc_pred.eq(labels.cpu()).float().mean().item())


def sinkhorn(scores: torch.Tensor, tau_t: float = 0.01, gamma: float = 0.0, iterations: int = 20) -> torch.Tensor:
    row, col = scores.shape
    probs = F.softmax(scores / tau_t, dim=1)
    probs = probs / float(row)
    if gamma > 0:
        target = torch.sum(probs, dim=0, keepdim=True)
        target = target ** gamma
        target = target / torch.sum(target)
    else:
        target = None
    for _ in range(iterations):
        probs = probs / probs.sum(dim=0, keepdim=True).clamp_min(1e-12)
        if target is None:
            probs = probs / float(col)
        else:
            probs = probs * target
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
        probs = probs / float(row)
    return probs * float(row)


def image_opt(
    features: torch.Tensor,
    init_classifier: torch.Tensor,
    pseudo_labels: torch.Tensor,
    *,
    lr: float,
    iterations: int,
    tau_i: float,
    alpha: float,
) -> torch.Tensor:
    num_samples = features.shape[0]
    values, indices = torch.max(pseudo_labels, dim=1)
    hard_mask = values > alpha
    pseudo = pseudo_labels.clone()
    pseudo[hard_mask, :] = 0
    pseudo[hard_mask, indices[hard_mask]] = 1
    base = features.t() @ pseudo
    classifier = init_classifier.clone()
    previous_norm = float("inf")
    current_lr = float(lr)
    for iteration in range(int(iterations)):
        prob = F.softmax(features @ classifier / tau_i, dim=1)
        grad = features.t() @ prob - base
        grad_norm = float(torch.norm(grad).item())
        if grad_norm > previous_norm:
            current_lr /= 2.0
        previous_norm = grad_norm
        classifier -= (current_lr / (float(num_samples) * tau_i)) * grad
        classifier = F.normalize(classifier, dim=0)
        if iteration == 0 or iteration + 1 == int(iterations) or (iteration + 1) % 500 == 0:
            log(f"image_opt iteration {iteration + 1}/{iterations}, grad_norm={grad_norm:.4f}, lr={current_lr:.5f}")
    return classifier


def evaluate_inmap(
    *,
    features: torch.Tensor,
    labels: torch.Tensor,
    text_classifier: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    x = features.to(device=device, dtype=torch.float32)
    y = labels.to(device=device)
    classifier = text_classifier.t().to(device=device, dtype=torch.float32)
    logits_t = x @ classifier
    log("InMaP: sinkhorn pseudo-label refinement")
    pseudo = sinkhorn(logits_t, tau_t=args.inmap_tau_t, gamma=0.0, iterations=args.sinkhorn_iters)
    log("InMaP: optimizing visual proxy")
    image_classifier = image_opt(
        x,
        classifier,
        pseudo,
        lr=args.inmap_lr,
        iterations=args.inmap_iters,
        tau_i=args.inmap_tau_i,
        alpha=args.inmap_alpha,
    )
    logits = x @ image_classifier
    acc = evaluate_logits(logits.detach().cpu(), y.cpu())
    del x, y, classifier, logits_t, pseudo, image_classifier, logits
    torch.cuda.empty_cache()
    return acc


def pp_estimate_eigen(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    _, num_classes = probs.size()
    matrix = torch.zeros((num_classes, num_classes), device=logits.device, dtype=logits.dtype)
    for class_index in range(num_classes):
        indices = (labels == class_index).nonzero(as_tuple=True)[0]
        if indices.numel() > 0:
            matrix[class_index] = probs[indices].mean(dim=0)
    matrix = matrix.t()
    vector = torch.full((num_classes,), 1.0 / float(num_classes), device=logits.device, dtype=logits.dtype)
    for _ in range(100):
        next_vector = torch.mv(matrix, vector)
        next_vector = next_vector / next_vector.norm(p=1).clamp_min(1e-12)
        next_vector = torch.clamp(next_vector, min=0)
        if torch.norm(next_vector - vector, p=1) < 1e-3:
            vector = next_vector
            break
        vector = next_vector
    return vector.clamp_min(1e-12)


def evaluate_frolic(
    *,
    features: torch.Tensor,
    labels: torch.Tensor,
    text_classifier: torch.Tensor,
    cupl_prompt_features: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    x = features.to(device=device, dtype=torch.float32)
    y = labels.to(device=device)
    clip_weights = text_classifier.t().to(device=device, dtype=torch.float32)
    llm_features = cupl_prompt_features.to(device=device, dtype=torch.float32)
    num_classes, prompt_count, feature_dim = llm_features.shape
    llm_weights_ave = F.normalize(llm_features.mean(dim=1), dim=-1)
    llm_weights = F.normalize(llm_features.reshape(-1, feature_dim), dim=-1)

    clip_logits = 100.0 * x @ clip_weights
    llm_ave_logits = 100.0 * x @ llm_weights_ave.t()
    log(f"Frolic: CLIP={evaluate_logits(clip_logits.cpu(), labels):.3f}, LLM-avg={evaluate_logits(llm_ave_logits.cpu(), labels):.3f}")

    lamda = 0.9
    used_samples = torch.cat((x, llm_weights), dim=0)
    variance = used_samples.var(dim=0, unbiased=True)
    inv_diag = 1.0 / ((1.0 - lamda) * variance + lamda).clamp_min(1e-6)
    x_part = 0.5 * ((x * inv_diag) * x).sum(dim=1)
    mu_part = 0.5 * ((llm_weights_ave * inv_diag) * llm_weights_ave).sum(dim=1)
    crs_part = (x * inv_diag) @ llm_weights_ave.t()
    lda_logits = crs_part - mu_part.unsqueeze(0) - x_part.unsqueeze(1)
    log(f"Frolic: LDA={evaluate_logits(lda_logits.detach().cpu(), labels):.3f}")

    log("Frolic: optimizing visual proxies")
    logits_v1 = x @ image_opt(
        x,
        clip_weights,
        sinkhorn(llm_ave_logits / 100.0, tau_t=args.inmap_tau_t, gamma=1.0, iterations=args.sinkhorn_iters),
        lr=args.inmap_lr,
        iterations=args.inmap_iters,
        tau_i=args.inmap_tau_i,
        alpha=args.inmap_alpha,
    )
    logits_v3 = x @ image_opt(
        x,
        clip_weights,
        sinkhorn(lda_logits, tau_t=args.inmap_tau_t, gamma=1.0, iterations=args.sinkhorn_iters),
        lr=args.inmap_lr,
        iterations=args.inmap_iters,
        tau_i=args.inmap_tau_i,
        alpha=args.inmap_alpha,
    )
    ensemble_logits = 100.0 * logits_v1 + 100.0 * logits_v3
    used_debias_logits = ensemble_logits.detach().clone()
    for iteration in range(11):
        values, _ = used_debias_logits.topk(2, dim=-1)
        diff = torch.abs(values[:, 0] - values[:, 1])
        selected = torch.where(diff > (1.0 / float(num_classes)))[0]
        if selected.numel() == 0:
            selected = torch.arange(used_debias_logits.shape[0], device=device)
        selected_logits = used_debias_logits[selected]
        selected_labels = selected_logits.argmax(dim=-1)
        prior = pp_estimate_eigen(selected_logits, selected_labels)
        ensemble_logits = ensemble_logits - torch.log(prior).unsqueeze(0)
        used_debias_logits = ensemble_logits.detach().clone()
        if iteration in {0, 5, 10}:
            log(f"Frolic debias iteration {iteration}/10, selected={int(selected.numel())}")
    acc = evaluate_logits(ensemble_logits.detach().cpu(), y.cpu())
    del x, y, clip_weights, llm_features, llm_weights_ave, llm_weights, clip_logits, llm_ave_logits
    del used_samples, variance, inv_diag, x_part, mu_part, crs_part, lda_logits, logits_v1, logits_v3, ensemble_logits
    torch.cuda.empty_cache()
    return acc


def evaluate_calip(
    *,
    model: torch.nn.Module,
    transform,
    data_root: Path,
    severity: int,
    text_features: torch.Tensor,
    labels_all: torch.Tensor,
    batch_size: int,
    workers: int,
    device: torch.device,
    beta2: float,
    beta3: float,
    max_samples_per_corruption: int | None,
) -> float:
    del labels_all
    text = text_features.to(device=device, dtype=torch.float32)
    total_correct = 0
    total_count = 0
    model.visual.use_local = True
    model.eval()
    with torch.no_grad():
        for corruption in CORRUPTIONS:
            dataset = maybe_subset(
                load_cifar10c_dataset(data_root, transform, corruption, severity),
                max_samples_per_corruption,
            )
            loader = build_loader(dataset, batch_size, workers, device)
            seen = 0
            for batch_index, batch in enumerate(loader):
                images = batch["image"].to(device, non_blocking=True)
                labels = batch_to_labels(batch).to(device, non_blocking=True)
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
                total_correct += int(logits.argmax(dim=1).eq(labels).sum().item())
                total_count += int(labels.numel())
                seen += int(labels.numel())
                if batch_index == 0 or seen == len(dataset) or batch_index % 20 == 0:
                    log(f"CALIP {corruption}: {seen}/{len(dataset)}")
    model.visual.use_local = False
    torch.cuda.empty_cache()
    return 100.0 * float(total_correct) / float(max(total_count, 1))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_root = args.output_root / args.run_name
    cache_dir = output_root / "cache"
    ensure_dir(output_root)
    metadata = {
        "family": args.family,
        "dataset": "cifar10c",
        "severity": int(args.severity),
        "corruptions": list(CORRUPTIONS),
        "num_images": len(CORRUPTIONS) * 10000,
        "max_samples_per_corruption": args.max_samples_per_corruption,
        "base_model_name": args.base_model_name,
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "external_root": str(args.external_root),
        "setting": "CIFAR-10-C severity 5, all 15 corruptions, ViT-B/16, label-free zero-shot inference",
    }
    write_json(output_root / "metadata.json", metadata)

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

    class_names = load_cifar10_class_names(args.data_root, transform)
    default_text = text_features_from_prompts(model=model, prompts_by_class=default_prompts(class_names), device=device)
    cupl_prompt_lists = cupl_prompts(class_names, args.external_root)
    cupl_text = text_features_from_prompts(model=model, prompts_by_class=cupl_prompt_lists, device=device)
    cupl_prompt_features = torch.stack(
        [encode_prompt_list(model, prompts, device).float() for prompts in cupl_prompt_lists],
        dim=0,
    )

    features, labels, groups = extract_or_load_global_features(
        cache_dir=cache_dir,
        model=model,
        transform=transform,
        data_root=args.data_root,
        severity=int(args.severity),
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        device=device,
        skip_existing=bool(args.skip_existing),
        max_samples_per_corruption=args.max_samples_per_corruption,
    )
    del groups
    rows: List[Dict[str, object]] = []

    log("evaluating CLIP and CLIP + CBC")
    clip_logits = features @ default_text.t()
    clip_top1 = evaluate_logits(clip_logits, labels)
    cbc_top1 = evaluate_cbc(clip_logits, labels, args.batch_size, args.seed)
    rows.append({"method": "CLIP", "top1": clip_top1, "delta_vs_clip": 0.0, "notes": "default prompt"})
    rows.append({"method": "CLIP + CBC", "top1": cbc_top1, "delta_vs_clip": cbc_top1 - clip_top1, "notes": "default prompt, batch=128"})
    write_csv(output_root / "partial_results.csv", rows)

    log("evaluating CALIP")
    calip_top1 = evaluate_calip(
        model=model,
        transform=transform,
        data_root=args.data_root,
        severity=int(args.severity),
        text_features=default_text,
        labels_all=labels,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        device=device,
        beta2=float(args.calip_beta2),
        beta3=float(args.calip_beta3),
        max_samples_per_corruption=args.max_samples_per_corruption,
    )
    rows.append({"method": "CALIP", "top1": calip_top1, "delta_vs_clip": calip_top1 - clip_top1, "notes": "official beta2=2.0 beta3=0.1"})
    write_csv(output_root / "partial_results.csv", rows)

    log("evaluating WaffleCLIP")
    waffle_scores: List[float] = []
    for rep in range(int(args.waffle_reps)):
        prompts = waffle_prompts(
            class_names=class_names,
            external_root=args.external_root,
            seed=rep,
            waffle_count=int(args.waffle_count),
        )
        waffle_text = text_features_from_prompts(model=model, prompts_by_class=prompts, device=device)
        score = evaluate_logits(features @ waffle_text.t(), labels)
        waffle_scores.append(score)
        log(f"WaffleCLIP rep {rep + 1}/{args.waffle_reps}: {score:.3f}")
    waffle_mean = float(np.mean(waffle_scores))
    waffle_std = float(np.std(waffle_scores, ddof=0))
    rows.append(
        {
            "method": "WaffleCLIP",
            "top1": waffle_mean,
            "top1_std": waffle_std,
            "delta_vs_clip": waffle_mean - clip_top1,
            "notes": f"waffle_count={args.waffle_count}, reps={args.waffle_reps}",
        }
    )
    write_csv(output_root / "partial_results.csv", rows)

    log("evaluating CuPL")
    cupl_top1 = evaluate_logits(features @ cupl_text.t(), labels)
    rows.append(
        {
            "method": "CuPL",
            "top1": cupl_top1,
            "delta_vs_clip": cupl_top1 - clip_top1,
            "notes": "provided CIFAR10 full prompts, 30/class",
        }
    )
    write_csv(output_root / "partial_results.csv", rows)

    log("evaluating InMaP")
    inmap_top1 = evaluate_inmap(
        features=features,
        labels=labels,
        text_classifier=default_text,
        args=args,
        device=device,
    )
    rows.append(
        {
            "method": "InMaP",
            "top1": inmap_top1,
            "delta_vs_clip": inmap_top1 - clip_top1,
            "notes": f"iters={args.inmap_iters}, sinkhorn={args.sinkhorn_iters}",
        }
    )
    write_csv(output_root / "partial_results.csv", rows)

    log("evaluating Frolic")
    frolic_top1 = evaluate_frolic(
        features=features,
        labels=labels,
        text_classifier=default_text,
        cupl_prompt_features=cupl_prompt_features,
        args=args,
        device=device,
    )
    rows.append(
        {
            "method": "Frolic",
            "top1": frolic_top1,
            "delta_vs_clip": frolic_top1 - clip_top1,
            "notes": "core repo implementation adapted with CuPL CIFAR10 prompts",
        }
    )

    for row in rows:
        row["top1"] = float(row["top1"])
        row["delta_vs_clip"] = float(row["delta_vs_clip"])
    write_csv(output_root / "comparison_results.csv", rows)
    write_json(output_root / "comparison_results.json", {"rows": rows})
    log("comparison complete")
    for row in rows:
        std = row.get("top1_std", "")
        std_text = f" ± {float(std):.3f}" if std != "" else ""
        log(f"{row['method']}: {float(row['top1']):.3f}{std_text} ({float(row['delta_vs_clip']):+.3f})")


if __name__ == "__main__":
    main()
