from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch import Tensor

import ttavlm.lib as lib
from ttavlm.methods.cliptta_otsu import CLIPTTA

Kwargs = Dict[str, Any]


class CLIPTTAReliabilityGate(CLIPTTA):
    def __init__(
        self,
        template: List[str],
        class_names: List[str],
        use_memory: bool = False,
        use_softmax_entropy: bool = False,
        use_scheduler: bool = False,
        use_tent: bool = False,
        use_clipartt: bool = False,
        K: int = 3,
        clipartt_temp: Optional[float] = 0.01,
        vision_gate_manifest_file: Optional[str] = None,
        reliability_gate_enable: bool = False,
        reliability_gate_low_threshold: float = 0.45,
        reliability_gate_high_threshold: float = 0.65,
        reliability_gate_diversity_threshold: float = 0.90,
        reliability_entropy_weight: float = 0.5,
        reliability_margin_weight: float = 0.5,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(
            template=template,
            class_names=class_names,
            use_memory=use_memory,
            use_softmax_entropy=use_softmax_entropy,
            use_scheduler=use_scheduler,
            use_tent=use_tent,
            use_clipartt=use_clipartt,
            K=K,
            clipartt_temp=clipartt_temp,
            **kwargs,
        )
        self.vision_gate_manifest_file = vision_gate_manifest_file
        self.reliability_gate_enable = bool(reliability_gate_enable and vision_gate_manifest_file)
        self.reliability_gate_low_threshold = float(reliability_gate_low_threshold)
        self.reliability_gate_high_threshold = float(reliability_gate_high_threshold)
        self.reliability_gate_diversity_threshold = float(reliability_gate_diversity_threshold)
        self.reliability_entropy_weight = float(reliability_entropy_weight)
        self.reliability_margin_weight = float(reliability_margin_weight)
        self._gate_history: List[Dict[str, Any]] = []
        self._current_gate_metrics: Optional[Dict[str, Any]] = None
        self._adaptation_batch_index = 0
        self._gate_manifest: Dict[str, Any] = {}
        self._broad_names: Set[str] = set(self._trainable_model_param_names)
        self._ln_names: Set[str] = set(self._collect_norm_param_names(self.model)).intersection(self._broad_names)
        self._medium_names: Set[str] = set(self._broad_names)
        self._load_gate_manifest()

    def _load_gate_manifest(self) -> None:
        if not self.reliability_gate_enable or self.vision_gate_manifest_file is None:
            return
        path = Path(self.vision_gate_manifest_file)
        with path.open() as handle:
            payload = json.load(handle)
        self._gate_manifest = payload if isinstance(payload, dict) else {}

        def _names(key: str, default: Sequence[str]) -> Set[str]:
            values = self._gate_manifest.get(key, default)
            return {str(value) for value in values}.intersection(self._broad_names)

        self._ln_names = _names("ln_names", list(self._ln_names))
        self._medium_names = _names("medium_names", list(self._broad_names))
        broad_names = _names("broad_names", list(self._broad_names))
        if broad_names:
            self._broad_names = broad_names
        if not self._ln_names:
            self._ln_names = set(self._collect_norm_param_names(self.model)).intersection(self._broad_names)
        if not self._medium_names:
            self._medium_names = set(self._ln_names)
        self._medium_names.update(self._ln_names)
        self._broad_names.update(self._medium_names)

    @staticmethod
    def _entropy_denominator(num_classes: int) -> float:
        return max(math.log(max(int(num_classes), 2)), 1.0e-12)

    def _compute_batch_gate_metrics(
        self,
        image_features: List[Tensor],
        class_prototypes: Tensor,
    ) -> Dict[str, Any]:
        logits = image_features[0].float() @ class_prototypes.float().t()
        scaled_logits = self.logit_scale * logits
        probs = scaled_logits.softmax(dim=-1)
        top_values = probs.topk(k=min(2, probs.shape[-1]), dim=-1).values
        if top_values.shape[-1] == 1:
            margins = top_values[:, 0]
        else:
            margins = top_values[:, 0] - top_values[:, 1]

        norm_denom = self._entropy_denominator(probs.shape[-1])
        mean_entropy = float((lib.softmax_entropy(scaled_logits) / norm_denom).mean().item())
        batch_marginal_entropy = float((lib.softmax_mean_entropy(scaled_logits) / norm_denom).item())
        mean_margin = float(margins.mean().item())
        mean_top1_prob = float(top_values[:, 0].mean().item())

        entropy_confidence = 1.0 - mean_entropy
        weight_sum = max(self.reliability_entropy_weight + self.reliability_margin_weight, 1.0e-12)
        reliability = (
            self.reliability_entropy_weight * entropy_confidence
            + self.reliability_margin_weight * mean_margin
        ) / weight_sum

        diversity_ok = batch_marginal_entropy >= self.reliability_gate_diversity_threshold
        if (not diversity_ok) or reliability < self.reliability_gate_low_threshold:
            tier = "ln_only"
        elif reliability < self.reliability_gate_high_threshold:
            tier = "medium"
        else:
            tier = "broad"

        return {
            "tier": tier,
            "reliability": reliability,
            "normalized_entropy": mean_entropy,
            "margin": mean_margin,
            "top1_prob": mean_top1_prob,
            "batch_marginal_entropy": batch_marginal_entropy,
            "diversity_ok": diversity_ok,
        }

    def _active_visual_names_for_tier(self, tier: str) -> Set[str]:
        if tier == "ln_only":
            return set(self._ln_names)
        if tier == "medium":
            return set(self._medium_names)
        return set(self._broad_names)

    def _apply_gate_gradient_mask(self, tier: str) -> Tuple[int, int]:
        active_names = self._active_visual_names_for_tier(tier)
        enabled = 0
        masked = 0
        for name, param in self.model.named_parameters():
            if name not in self._broad_names or param.grad is None:
                continue
            if name in active_names:
                enabled += 1
                continue
            param.grad = None
            masked += 1
        return enabled, masked

    def _gate_tensor_stats(self, tier: str) -> Tuple[int, int, int]:
        active_names = self._active_visual_names_for_tier(tier)
        enabled = len(active_names)
        enabled_non_ln = sum(1 for name in active_names if name not in self._ln_names)
        masked = len(self._broad_names - active_names)
        return enabled, enabled_non_ln, masked

    def _record_gate_decision(self, metrics: Dict[str, Any], step: int) -> None:
        enabled, enabled_non_ln, masked = self._gate_tensor_stats(str(metrics["tier"]))
        metrics["step"] = int(step)
        metrics["enabled_tensors"] = int(enabled)
        metrics["masked_tensors"] = int(masked)
        metrics["enabled_non_ln_tensors"] = int(enabled_non_ln)
        self._current_gate_metrics = metrics
        lib.LOGGER.info(
            "RELIABILITY_GATE batch=%d step=%d tier=%s reliability=%.4f entropy=%.4f margin=%.4f batch_entropy=%.4f top1_prob=%.4f enabled=%d masked=%d",
            self._adaptation_batch_index,
            int(step),
            metrics["tier"],
            float(metrics["reliability"]),
            float(metrics["normalized_entropy"]),
            float(metrics["margin"]),
            float(metrics["batch_marginal_entropy"]),
            float(metrics["top1_prob"]),
            int(enabled),
            int(masked),
        )

    def _gate_paths(self) -> Tuple[Path, Path]:
        save_root = Path(self.save_root)
        save_root.mkdir(parents=True, exist_ok=True)
        return (
            save_root / "reliability_gate_history.csv",
            save_root / "reliability_gate_summary.json",
        )

    def _flush_gate_artifacts(self) -> None:
        history_path, summary_path = self._gate_paths()
        fieldnames = [
            "batch_index",
            "step",
            "tier",
            "reliability",
            "normalized_entropy",
            "margin",
            "top1_prob",
            "batch_marginal_entropy",
            "diversity_ok",
            "enabled_tensors",
            "masked_tensors",
            "enabled_non_ln_tensors",
        ]
        with history_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._gate_history)

        if not self._gate_history:
            summary = {
                "num_batches": 0,
                "tier_counts": {"ln_only": 0, "medium": 0, "broad": 0},
            }
        else:
            tier_counts = {
                "ln_only": sum(1 for row in self._gate_history if row["tier"] == "ln_only"),
                "medium": sum(1 for row in self._gate_history if row["tier"] == "medium"),
                "broad": sum(1 for row in self._gate_history if row["tier"] == "broad"),
            }
            summary = {
                "num_batches": len(self._gate_history),
                "tier_counts": tier_counts,
                "tier_fractions": {
                    key: value / max(len(self._gate_history), 1) for key, value in tier_counts.items()
                },
                "mean_reliability": sum(float(row["reliability"]) for row in self._gate_history) / len(self._gate_history),
                "mean_normalized_entropy": sum(float(row["normalized_entropy"]) for row in self._gate_history) / len(self._gate_history),
                "mean_margin": sum(float(row["margin"]) for row in self._gate_history) / len(self._gate_history),
                "mean_top1_prob": sum(float(row["top1_prob"]) for row in self._gate_history) / len(self._gate_history),
                "mean_batch_marginal_entropy": sum(float(row["batch_marginal_entropy"]) for row in self._gate_history) / len(self._gate_history),
                "mean_enabled_tensors": sum(int(row["enabled_tensors"]) for row in self._gate_history) / len(self._gate_history),
                "mean_enabled_non_ln_tensors": sum(int(row["enabled_non_ln_tensors"]) for row in self._gate_history) / len(self._gate_history),
            }
        summary.update(
            {
                "low_threshold": self.reliability_gate_low_threshold,
                "high_threshold": self.reliability_gate_high_threshold,
                "diversity_threshold": self.reliability_gate_diversity_threshold,
                "entropy_weight": self.reliability_entropy_weight,
                "margin_weight": self.reliability_margin_weight,
            }
        )
        with summary_path.open("w") as handle:
            json.dump(summary, handle, indent=2)

    @torch.enable_grad()
    def _forward_and_adapt(
        self,
        images: List[Tensor],
        step: int,
    ) -> Tensor:
        self._last_backward_succeeded = False
        image_features = self.get_features(images)
        logits = self.get_logits(image_features)
        loss_device = image_features[0].device
        loss_reg = torch.zeros((), device=loss_device, dtype=torch.float32)
        loss_ood = torch.zeros((), device=loss_device, dtype=torch.float32)

        if self.beta_reg != 0:
            loss_reg = self._scaled_entropy(logits[0])

        if self.use_ood_loss or self.detect_ood:
            scores = self.get_scores(logits, image_features[0])
            if self.use_ood_loss:
                loss_ood = self.get_otsu_loss(scores)

        if self.detect_ood:
            images, image_features = self.filter_id(images, image_features, scores, self.alpha if self.update_alpha else None)

        if self.update_text:
            class_prototypes, _ = lib.get_text_features(self.class_names, self.template, self.clip_text_encoder, enable_grad=True)
        else:
            class_prototypes = self.class_prototypes

        if self.reliability_gate_enable and (step == 0 or self._current_gate_metrics is None):
            metrics = self._compute_batch_gate_metrics(image_features, class_prototypes)
            self._record_gate_decision(metrics, step)

        loss_tta = self.compute_loss_tta(image_features, class_prototypes)
        loss_relation = self.compute_relation_consistency_loss(image_features)

        if self.use_memory:
            if step == 0:
                with torch.no_grad():
                    logits = self.get_logits(image_features)
                    _, pred = logits[0].topk(1, 1, True, True)
                    scores = self.get_scores(logits, image_features, score_type=self.id_score_type)
                self.memory.update(images[0].cpu().detach(), pred[:, 0].cpu().detach(), scores.cpu().detach())

            images_mem, _, _ = self.memory.sample()
            image_features_mem = self.get_features([images_mem.to(image_features[0].device)])
            logits_mem = self.get_logits(image_features_mem)
            loss_tta += self.compute_loss_tta(image_features_mem, class_prototypes)
            loss_reg += self._scaled_entropy(logits_mem[0])

        loss = (
            self.beta_tta * loss_tta
            + self.beta_relation * loss_relation
            - self.beta_reg * loss_reg
            + self.beta_ood * loss_ood
        )

        if not torch.isfinite(loss):
            lib.LOGGER.warning("Skipping CLIPTTA reliability-gated update because the adaptation loss became non-finite.")
            self.optimizer.zero_grad(set_to_none=True)
            return loss.detach()

        loss.backward()
        if self.reliability_gate_enable and self._current_gate_metrics is not None:
            self._apply_gate_gradient_mask(str(self._current_gate_metrics["tier"]))
        self._last_backward_succeeded = True
        return loss.detach()

    def after_adaptation(self, **kwargs: Kwargs) -> None:
        super().after_adaptation(**kwargs)
        if not self.reliability_gate_enable or self._current_gate_metrics is None:
            return
        row = {
            "batch_index": int(self._adaptation_batch_index),
            **self._current_gate_metrics,
        }
        self._gate_history.append(row)
        self._adaptation_batch_index += 1
        self._flush_gate_artifacts()
        self._current_gate_metrics = None
