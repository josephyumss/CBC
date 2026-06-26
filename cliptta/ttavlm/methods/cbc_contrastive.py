from typing import Any, Dict, List, Tuple
from typing_extensions import TypeAlias

from functools import partial

import torch
from torch import Tensor
import torch.nn.functional as F

from ttavlm.methods.cliptta_otsu import CLIPTTA
import ttavlm.lib as lib


Kwargs: TypeAlias = Dict[str, Any]


class CBCContrastiveTTA(CLIPTTA):
    """CBC-guided contrastive test-time adaptation.

    One adaptation step consists of two coupled operations:
    1. Estimate and remove the current batch-level corruption bias from CLIP
       class logits to obtain CBC pseudo labels.
    2. Apply a contrastive/prototype objective on feature proxies whose batch
       center has been shifted toward the text-prototype center.

    Final predictions are made with CBC logits after the visual update. CBC is
    therefore responsible for distribution position, while the contrastive loss
    is responsible for adapting feature geometry.
    """

    def __init__(
        self,
        template: List[str],
        class_names: List[str],
        cbc_shift_weight: float = 1.0,
        cbc_contrastive_weight: float = 1.0,
        cbc_proto_weight: float = 1.0,
        cbc_diversity_weight: float = 0.2,
        cbc_target_mode: str = "classwise",
        cbc_confidence_threshold: float = 0.0,
        cbc_min_keep_ratio: float = 0.25,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(
            template=template,
            class_names=class_names,
            use_memory=False,
            use_softmax_entropy=False,
            use_scheduler=False,
            use_tent=False,
            use_clipartt=False,
            cliptta_target_mode="classwise",
            **kwargs,
        )
        self.cbc_shift_weight = float(cbc_shift_weight)
        self.cbc_contrastive_weight = float(cbc_contrastive_weight)
        self.cbc_proto_weight = float(cbc_proto_weight)
        self.cbc_diversity_weight = float(cbc_diversity_weight)
        self.cbc_target_mode = str(cbc_target_mode)
        self.cbc_confidence_threshold = float(cbc_confidence_threshold)
        self.cbc_min_keep_ratio = float(cbc_min_keep_ratio)

    def _center_logits(self, logits: Tensor) -> Tensor:
        if logits.shape[0] <= 1:
            return logits
        return logits - logits.mean(dim=0, keepdim=True)

    def compute_pseudo_label_logits(
        self,
        image_features: Tensor,
        class_prototypes: Tensor,
    ) -> Tensor:
        logits = image_features.float() @ class_prototypes.float().t()
        return self._center_logits(logits)

    def _shift_features_to_text_center(self, image_features: Tensor, class_prototypes: Tensor) -> Tensor:
        features = image_features.float()
        prototypes = class_prototypes.float()
        if features.shape[0] <= 1 or self.cbc_shift_weight == 0.0:
            return F.normalize(features, dim=-1)

        feature_center = features.mean(dim=0, keepdim=True)
        text_center = prototypes.mean(dim=0, keepdim=True)
        bias = feature_center - text_center
        shifted = features - self.cbc_shift_weight * bias.detach()
        return F.normalize(shifted, dim=-1)

    def _targets_from_pseudo_labels(self, labels: Tensor, device: torch.device) -> Tensor:
        batch_size = int(labels.shape[0])
        identity = torch.eye(batch_size, device=device, dtype=torch.float32)
        same_class = labels.unsqueeze(0).eq(labels.unsqueeze(1)).to(dtype=torch.float32)
        same_class = same_class / same_class.sum(dim=1, keepdim=True).clamp_min(1.0)

        if self.cbc_target_mode == "instance":
            return identity
        if self.cbc_target_mode == "classwise":
            return same_class
        if self.cbc_target_mode == "hybrid":
            targets = 0.5 * identity + 0.5 * same_class
            return targets / targets.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        raise ValueError(f"Unsupported cbc_target_mode: {self.cbc_target_mode}")

    def _confidence_keep_mask(self, cbc_logits: Tensor) -> Tensor:
        if self.cbc_confidence_threshold <= 0.0:
            return torch.ones(cbc_logits.shape[0], device=cbc_logits.device, dtype=torch.bool)

        probs = (self.logit_scale * cbc_logits).softmax(dim=-1)
        confidence = probs.max(dim=-1).values
        keep = confidence >= self.cbc_confidence_threshold
        min_keep = max(1, int(round(cbc_logits.shape[0] * self.cbc_min_keep_ratio)))
        if int(keep.sum().item()) < min_keep:
            keep_idx = torch.argsort(confidence, descending=True)[:min_keep]
            keep = torch.zeros_like(keep)
            keep[keep_idx] = True
        return keep

    def compute_loss_tta(self, image_features: List[Tensor], class_prototypes: Tensor) -> Tensor:
        image_features_fp32 = image_features[0].float()
        class_prototypes_fp32 = class_prototypes.float()

        cbc_logits = self.compute_pseudo_label_logits(image_features_fp32, class_prototypes_fp32)
        pseudo_labels = cbc_logits.argmax(dim=1)
        keep = self._confidence_keep_mask(cbc_logits)
        if int(keep.sum().item()) == 0:
            return torch.zeros((), device=image_features_fp32.device, dtype=torch.float32)

        shifted_features = self._shift_features_to_text_center(image_features_fp32, class_prototypes_fp32)
        shifted_kept = shifted_features[keep]
        labels_kept = pseudo_labels[keep]
        pred_text_features = class_prototypes_fp32[labels_kept]

        zero = torch.zeros((), device=image_features_fp32.device, dtype=torch.float32)
        loss_contrast = zero
        if self.cbc_contrastive_weight > 0.0 and shifted_kept.shape[0] > 1:
            logits_per_image = self.logit_scale * shifted_kept @ pred_text_features.t()
            logits_per_text = logits_per_image.t() if self.update_text else logits_per_image
            targets = self._targets_from_pseudo_labels(labels_kept, logits_per_image.device)
            loss_contrast = (
                self._soft_cross_entropy(logits_per_image, targets)
                + self._soft_cross_entropy(logits_per_text, targets)
            ) / 2

        loss_proto = zero
        if self.cbc_proto_weight > 0.0:
            proto_logits = self.logit_scale * shifted_kept @ class_prototypes_fp32.t()
            loss_proto = F.cross_entropy(proto_logits, labels_kept)

        loss_diversity = zero
        if self.cbc_diversity_weight > 0.0 and shifted_features.shape[0] > 1:
            shifted_logits = shifted_features @ class_prototypes_fp32.t()
            loss_diversity = lib.softmax_mean_entropy(self.logit_scale * shifted_logits)

        return (
            self.cbc_contrastive_weight * loss_contrast
            + self.cbc_proto_weight * loss_proto
            - self.cbc_diversity_weight * loss_diversity
        )

    @torch.enable_grad()
    def forward_and_adapt(
        self,
        images: List[Tensor],
        step: int,
        labels: Tensor = None,
    ) -> Tuple[List[Tensor], Tensor]:
        _ = self._forward_and_adapt(images, step)
        closure = partial(self._forward_and_adapt, images=images, step=step) if self.use_sam else None

        if self._last_backward_succeeded:
            trainable_params = self._optimizer_params_with_grads()
            if trainable_params:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
                if torch.isfinite(torch.as_tensor(grad_norm)):
                    self.optimizer.step(closure)
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.optimizer.zero_grad(set_to_none=True)

        if step == self.steps - 1:
            with torch.no_grad():
                class_prototypes = self.class_prototypes
                image_features = self.get_features(images)
                raw_logits = self.get_logits(image_features, class_prototypes)
                logits = [self._center_logits(raw_logits[0])]
                scores = self.get_scores(logits, image_features)
        else:
            logits, scores = None, None

        return logits, scores
