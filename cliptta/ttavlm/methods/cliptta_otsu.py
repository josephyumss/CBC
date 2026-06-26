from typing import Dict, Any, Optional, Tuple, List
from typing_extensions import TypeAlias

from functools import partial
import math

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from copy import deepcopy

from ttavlm.methods.abstract_model import AbstractOpenSetTTAModel
from ttavlm.models.clip import tokenize as clip_tokenize
from ttavlm.memory import CCM


import ttavlm.lib as lib

Kwargs: TypeAlias = Dict[str, Any]


class CLIPTTA(AbstractOpenSetTTAModel):
    """
    CLIPTTA adapts CLIP using the same loss as during the pre-training.
    """

    def __init__(
        self,
        template: List[str],
        class_names: List[str],
        use_memory: bool = False,
        use_softmax_entropy: bool = False,
        use_scheduler: bool = False,
        use_tent: bool = False,
        use_clipartt: bool = False,
        cliptta_target_mode: str = "instance",
        cliptta_instance_weight: float = 1.0,
        cliptta_classwise_weight: float = 1.0,
        cliptta_sameclass_vv_weight: float = 0.1,
        cliptta_proto_sym_weight: float = 0.25,
        cliptta_proto_sym_agreement_only: bool = False,
        use_relation_consistency: bool = False,
        beta_relation: float = 1.0,
        relation_temperature: float = 0.2,
        K: int = 3,
        clipartt_temp: Optional[float] = 0.01,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.template = template
        self.class_names = class_names
        self.use_memory = use_memory
        self.use_softmax_entropy = use_softmax_entropy
        self.use_scheduler = use_scheduler
        self.use_tent = use_tent
        self.use_clipartt = use_clipartt
        self.cliptta_target_mode = str(cliptta_target_mode)
        self.cliptta_instance_weight = max(float(cliptta_instance_weight), 0.0)
        self.cliptta_classwise_weight = max(float(cliptta_classwise_weight), 0.0)
        self.cliptta_sameclass_vv_weight = max(float(cliptta_sameclass_vv_weight), 0.0)
        self.cliptta_proto_sym_weight = max(float(cliptta_proto_sym_weight), 0.0)
        self.cliptta_proto_sym_agreement_only = bool(cliptta_proto_sym_agreement_only)
        self.use_relation_consistency = use_relation_consistency
        self.beta_relation = float(beta_relation)
        self.relation_temperature = max(float(relation_temperature), 1.0e-6)
        self.K = K
        self.clipartt_temp = clipartt_temp
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.max_grad_norm = 1.0
        self._last_backward_succeeded = False
        self._relation_warning_emitted = False
        self.pseudo_label_aux_class_prototypes: Optional[Tensor] = None
        self.pseudo_label_merge_mode = "none"
        self.pseudo_label_clean_weight = 1.0
        self.pseudo_label_aux_weight = 1.0
        self.pseudo_label_aux_scope = "always"
        self.pseudo_label_aux_apply_on_memory_loss = True
        self.memory_update_mode = "default"
        self._adaptation_batch_index = 0
        self._current_adaptation_step = 0
        self._pseudo_label_context = "main"
        self.dynamic_prompt_alpha_encoder: Optional[nn.Module] = None
        self.dynamic_prompt_alpha_mode = "severity_alpha"
        self.dynamic_prompt_alpha_feature_source = "global"
        self.dynamic_prompt_alpha_spatial_grid_size = 14
        self.dynamic_prompt_alpha_class_values: Optional[Tensor] = None
        self.dynamic_prompt_alpha_max_severity = 5.0
        self.dynamic_prompt_alpha_last_mean: Optional[float] = None
        self._pseudo_label_clean_weight_override: Optional[Tensor] = None
        self._pseudo_label_aux_weight_override: Optional[Tensor] = None

        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_iter)

        self.memory = None
        if use_memory:
            self.memory = CCM(num_shots=self.num_shots, num_classes=len(self.class_names), sample_size=self.sample_size)

        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_iter)

        if self.measure_improvement:
            self.model0 = deepcopy(self.model)
            for param in self.model0.parameters():
                param.detach()

    def _instance_targets(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.eye(batch_size, device=device, dtype=torch.float32)

    def _classwise_targets(self, pred: Tensor) -> Tensor:
        labels = pred[:, 0] if pred.ndim > 1 else pred
        targets = labels.unsqueeze(0).eq(labels.unsqueeze(1)).to(dtype=torch.float32)
        return targets / targets.sum(dim=1, keepdim=True).clamp_min(1.0)

    def _contrastive_targets(self, pred: Tensor, device: torch.device) -> Tensor:
        batch_size = int(pred.shape[0])
        if self.cliptta_target_mode == "instance":
            return self._instance_targets(batch_size, device)
        if self.cliptta_target_mode == "classwise":
            return self._classwise_targets(pred)
        if self.cliptta_target_mode == "hybrid":
            targets = torch.zeros((batch_size, batch_size), device=device, dtype=torch.float32)
            if self.cliptta_instance_weight > 0.0:
                targets = targets + self.cliptta_instance_weight * self._instance_targets(batch_size, device)
            if self.cliptta_classwise_weight > 0.0:
                targets = targets + self.cliptta_classwise_weight * self._classwise_targets(pred)
            if torch.all(targets == 0):
                raise ValueError("Hybrid CLIPTTA targets require a positive instance or classwise weight.")
            return targets / targets.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        if self.cliptta_target_mode == "prototype_sameclass_vv":
            raise RuntimeError(
                "prototype_sameclass_vv uses a dedicated loss branch and should not call _contrastive_targets."
            )
        if self.cliptta_target_mode == "prototype_sym":
            raise RuntimeError(
                "prototype_sym uses a dedicated loss branch and should not call _contrastive_targets."
            )
        if self.cliptta_target_mode == "instance_proto_sym":
            raise RuntimeError(
                "instance_proto_sym uses a dedicated loss branch and should not call _contrastive_targets."
            )
        raise ValueError(f"Unsupported cliptta_target_mode: {self.cliptta_target_mode}")

    def _soft_cross_entropy(self, logits: Tensor, targets: Tensor) -> Tensor:
        return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

    def _sameclass_vv_repulsion_loss(self, image_features: Tensor, pred: Tensor) -> Tensor:
        labels = pred[:, 0] if pred.ndim > 1 else pred
        zero = torch.zeros((), device=image_features.device, dtype=torch.float32)
        if labels.numel() < 2 or self.cliptta_sameclass_vv_weight <= 0.0:
            return zero

        similarities = image_features.float() @ image_features.float().t()
        same_class_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1))
        diagonal_mask = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
        repulsion_mask = same_class_mask & ~diagonal_mask
        if not repulsion_mask.any():
            return zero
        return similarities.masked_select(repulsion_mask).mean()

    def _prototype_sym_loss(
        self,
        *,
        logits: Tensor,
        image_features: Tensor,
        class_prototypes: Tensor,
        pred: Tensor,
    ) -> Tensor:
        labels = pred[:, 0] if pred.ndim > 1 else pred
        if self.cliptta_proto_sym_agreement_only:
            raw_labels = logits.argmax(dim=-1)
            keep_mask = raw_labels.eq(labels)
            if int(keep_mask.sum().item()) == 0:
                return torch.zeros((), device=logits.device, dtype=torch.float32)
            logits = logits[keep_mask]
            image_features = image_features[keep_mask]
            labels = labels[keep_mask]

        image_to_text_loss = F.cross_entropy(self.logit_scale * logits, labels)
        unique_labels = torch.unique(labels, sorted=True)
        text_logits = self.logit_scale * class_prototypes[unique_labels] @ image_features.t()
        text_targets = labels.unsqueeze(0).eq(unique_labels.unsqueeze(1)).to(dtype=torch.float32)
        text_targets = text_targets / text_targets.sum(dim=1, keepdim=True).clamp_min(1.0)
        text_to_image_loss = self._soft_cross_entropy(text_logits, text_targets)
        return 0.5 * (image_to_text_loss + text_to_image_loss)

    def _scaled_entropy(self, logits: Tensor) -> Tensor:
        """Compute entropy regularization in float32 for numerical stability."""
        return lib.softmax_mean_entropy((self.logit_scale * logits).float())

    def _optimizer_params_with_grads(self) -> List[Tensor]:
        params: List[Tensor] = []
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    params.append(param)
        return params

    def _has_finite_gradients(self) -> bool:
        for param in self._optimizer_params_with_grads():
            if not torch.isfinite(param.grad).all():
                return False
        return True

    def _relation_logits(self, features: Tensor) -> Optional[Tensor]:
        if features.shape[0] < 2:
            return None
        logits = features.float() @ features.float().t()
        diagonal_mask = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(diagonal_mask, float("-inf"))
        return logits / self.relation_temperature

    def compute_relation_consistency_loss(self, image_features: List[Tensor]) -> Tensor:
        loss_device = image_features[0].device
        zero = torch.zeros((), device=loss_device, dtype=torch.float32)
        if not self.use_relation_consistency:
            return zero
        if len(image_features) < 2:
            if not self._relation_warning_emitted:
                lib.LOGGER.warning(
                    "Relation consistency was enabled, but no augmented view is available. "
                    "Use --use_tta with n_augment >= 1 to activate the auxiliary loss."
                )
                self._relation_warning_emitted = True
            return zero

        teacher_logits = self._relation_logits(image_features[0])
        if teacher_logits is None:
            return zero
        teacher_probs = F.softmax(teacher_logits.detach(), dim=-1)

        losses: List[Tensor] = []
        for augmented_features in image_features[1:]:
            student_logits = self._relation_logits(augmented_features)
            if student_logits is None:
                continue
            losses.append(
                F.kl_div(
                    F.log_softmax(student_logits, dim=-1),
                    teacher_probs,
                    reduction="batchmean",
                )
            )

        if not losses:
            return zero
        return torch.stack(losses).mean()

    def compute_pseudo_label_logits(
        self,
        image_features: Tensor,
        class_prototypes: Tensor,
    ) -> Tensor:
        image_features_fp32 = image_features.float()
        clean_logits = image_features_fp32 @ class_prototypes.float().t()

        aux_class_prototypes = self.pseudo_label_aux_class_prototypes
        if (
            aux_class_prototypes is None
            or self.pseudo_label_merge_mode == "none"
            or not self._should_use_aux_pseudo_labels()
        ):
            return clean_logits

        aux_logits = image_features_fp32 @ aux_class_prototypes.float().t()
        clean_weight_override = self._pseudo_label_clean_weight_override
        aux_weight_override = self._pseudo_label_aux_weight_override
        if clean_weight_override is not None and aux_weight_override is not None:
            clean_weight = clean_weight_override.to(device=clean_logits.device, dtype=clean_logits.dtype)
            aux_weight = aux_weight_override.to(device=clean_logits.device, dtype=clean_logits.dtype)
            if clean_weight.ndim > 0 and clean_weight.shape[0] != clean_logits.shape[0]:
                clean_weight = clean_weight.mean()
            if aux_weight.ndim > 0 and aux_weight.shape[0] != aux_logits.shape[0]:
                aux_weight = aux_weight.mean()
        else:
            clean_weight = torch.as_tensor(
                float(self.pseudo_label_clean_weight),
                device=clean_logits.device,
                dtype=clean_logits.dtype,
            )
            aux_weight = torch.as_tensor(
                float(self.pseudo_label_aux_weight),
                device=clean_logits.device,
                dtype=clean_logits.dtype,
            )

        if self.pseudo_label_merge_mode == "sum":
            return clean_weight * clean_logits + aux_weight * aux_logits

        if self.pseudo_label_merge_mode == "prototype_interp":
            clean_proto = class_prototypes.float()
            aux_proto = aux_class_prototypes.float()
            if clean_weight.ndim == 0 and aux_weight.ndim == 0:
                mixed_prototypes = clean_weight * clean_proto + aux_weight * aux_proto
                mixed_prototypes = mixed_prototypes / mixed_prototypes.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
                return image_features_fp32 @ mixed_prototypes.t()

            if clean_weight.ndim == 0:
                clean_weight = clean_weight.expand(image_features_fp32.shape[0])
            if aux_weight.ndim == 0:
                aux_weight = aux_weight.expand(image_features_fp32.shape[0])

            clean_weight = clean_weight.reshape(image_features_fp32.shape[0], 1, 1)
            aux_weight = aux_weight.reshape(image_features_fp32.shape[0], 1, 1)
            mixed_prototypes = (
                clean_weight * clean_proto.unsqueeze(0)
                + aux_weight * aux_proto.unsqueeze(0)
            )
            mixed_prototypes = mixed_prototypes / mixed_prototypes.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
            return torch.einsum("bd,bcd->bc", image_features_fp32, mixed_prototypes)

        raise ValueError(f"Unsupported pseudo-label merge mode: {self.pseudo_label_merge_mode}")

    def _should_use_aux_pseudo_labels(self) -> bool:
        if self.pseudo_label_aux_scope == "always":
            return True
        if self.pseudo_label_aux_scope == "initial_step":
            if self._adaptation_batch_index != 0 or self._current_adaptation_step != 0:
                return False
            if self._pseudo_label_context == "memory_loss" and not self.pseudo_label_aux_apply_on_memory_loss:
                return False
            return True
        if self.pseudo_label_aux_scope == "none":
            return False
        raise ValueError(f"Unsupported pseudo_label_aux_scope: {self.pseudo_label_aux_scope}")

    def _compute_memory_update_inputs(
        self,
        image_features: Tensor,
        class_prototypes: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        pseudo_label_logits = self.compute_pseudo_label_logits(image_features, class_prototypes)
        _, pred = pseudo_label_logits.topk(1, 1, True, True)
        scores = self.get_scores([pseudo_label_logits], image_features, score_type=self.id_score_type)
        keep_mask = torch.ones_like(pred[:, 0], dtype=torch.bool)

        if (
            self.memory_update_mode == "initial_match_avg_conf"
            and self._adaptation_batch_index == 0
            and self._current_adaptation_step == 0
        ):
            aux_class_prototypes = self.pseudo_label_aux_class_prototypes
            if aux_class_prototypes is not None:
                image_features_fp32 = image_features.float()
                clean_logits = image_features_fp32 @ class_prototypes.float().t()
                aux_logits = image_features_fp32 @ aux_class_prototypes.float().t()
                clean_probs = clean_logits.softmax(dim=-1)
                aux_probs = aux_logits.softmax(dim=-1)
                clean_scores, clean_pred = clean_probs.max(dim=1)
                aux_scores, aux_pred = aux_probs.max(dim=1)
                keep_mask = clean_pred.eq(aux_pred)
                pred = aux_pred.unsqueeze(1)
                scores = 0.5 * (clean_scores + aux_scores)

        return pred[:, 0], scores, keep_mask

    def _dynamic_prompt_alpha_visual_model(self) -> nn.Module:
        return self.model.module if hasattr(self.model, "module") else self.model

    def _extract_dynamic_prompt_alpha_features(self, images: Tensor) -> Optional[Tensor]:
        encoder = self.dynamic_prompt_alpha_encoder
        if encoder is None:
            return None

        visual_model = self._dynamic_prompt_alpha_visual_model()
        if self.dynamic_prompt_alpha_feature_source == "global":
            image_features = visual_model(images.type(visual_model.dtype)).float()
            return image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)

        original_use_local = getattr(visual_model, "use_local", False)
        try:
            visual_model.use_local = True
            local_tokens = visual_model(images.type(visual_model.dtype)).float()
        finally:
            visual_model.use_local = original_use_local

        patch_tokens = local_tokens[:, 1:, :]
        num_patches = int(patch_tokens.shape[1])
        source_grid = int(round(math.sqrt(num_patches)))
        if source_grid * source_grid != num_patches:
            raise ValueError(f"Expected square patch grid, got {num_patches} patch tokens")
        patch_tokens = patch_tokens.transpose(1, 2).reshape(
            patch_tokens.shape[0],
            patch_tokens.shape[2],
            source_grid,
            source_grid,
        )
        spatial_features = F.normalize(patch_tokens, dim=1)
        target_grid = int(self.dynamic_prompt_alpha_spatial_grid_size)
        if source_grid != target_grid:
            spatial_features = F.adaptive_avg_pool2d(spatial_features, output_size=(target_grid, target_grid))
        return spatial_features

    def _predict_dynamic_prompt_alpha(self, images: Tensor) -> Optional[Tensor]:
        encoder = self.dynamic_prompt_alpha_encoder
        severity_class_values = self.dynamic_prompt_alpha_class_values
        if encoder is None:
            return None

        features = self._extract_dynamic_prompt_alpha_features(images)
        if features is None:
            return None

        with torch.no_grad():
            _, outputs = encoder(features.to(device=images.device, dtype=torch.float32))
            if self.dynamic_prompt_alpha_mode == "prompt_alpha":
                alpha = outputs.squeeze(1).sigmoid()
            else:
                if severity_class_values is None:
                    return None
                probs = outputs.softmax(dim=1)
                severity_values = severity_class_values.to(device=probs.device, dtype=probs.dtype)
                expected_severity = probs @ severity_values
                alpha = expected_severity / float(self.dynamic_prompt_alpha_max_severity)
        return alpha.clamp_(0.0, 1.0)

    def before_adaptation(
        self,
        images: List[Tensor],
        **kwargs: Kwargs,
    ):  # noqa: ANN201
        super().before_adaptation(images, **kwargs)
        self._pseudo_label_clean_weight_override = None
        self._pseudo_label_aux_weight_override = None
        self.dynamic_prompt_alpha_last_mean = None

        alpha = self._predict_dynamic_prompt_alpha(images[0])
        if alpha is None:
            return

        alpha = alpha.view(-1, 1)
        self._pseudo_label_clean_weight_override = 1.0 - alpha
        self._pseudo_label_aux_weight_override = alpha
        self.dynamic_prompt_alpha_last_mean = float(alpha.mean().item())

    @torch.enable_grad()
    def _forward_and_adapt(
        self,
        images: List[Tensor],
        step: int,
    ) -> Tensor:
        self._last_backward_succeeded = False
        self._current_adaptation_step = int(step)
        self._pseudo_label_context = "main"
        image_features = self.get_features(images)
        logits = self.get_logits(image_features)
        loss_device = image_features[0].device
        loss_reg = torch.zeros((), device=loss_device, dtype=torch.float32)
        loss_ood = torch.zeros((), device=loss_device, dtype=torch.float32)

        # Regularization loss
        if self.beta_reg != 0:
            loss_reg = self._scaled_entropy(logits[0])

        # OOD loss computation
        if self.use_ood_loss or self.detect_ood:
            scores = self.get_scores(logits, image_features[0])
            if self.use_ood_loss:
                loss_ood = self.get_otsu_loss(scores)

        # OOD-ID separation
        if self.detect_ood:
            images, image_features = self.filter_id(images, image_features, scores, self.alpha if self.update_alpha else None)

        # TTA loss computation
        if self.update_text:
            class_prototypes, _ = lib.get_text_features(self.class_names, self.template, self.clip_text_encoder, enable_grad=True)
        else:
            class_prototypes = self.class_prototypes

        # Compute TTA loss using samples from the batch
        loss_tta = self.compute_loss_tta(image_features, class_prototypes)
        loss_relation = self.compute_relation_consistency_loss(image_features)

        # Compute TTA loss using samples from the memory
        if self.use_memory:
            # Updating Memory
            if step == 0:
                with torch.no_grad():
                    self._pseudo_label_context = "memory_update"
                    pred, scores, keep_mask = self._compute_memory_update_inputs(image_features[0], class_prototypes)

                if keep_mask.any():
                    self.memory.update(
                        images[0][keep_mask].cpu().detach(),
                        pred[keep_mask].cpu().detach(),
                        scores[keep_mask].cpu().detach(),
                    )

            # Sampling from memory
            images, _, _ = self.memory.sample()
            image_features = self.get_features([images.to(image_features[0].device)])
            logits = self.get_logits(image_features)
            self._pseudo_label_context = "memory_loss"
            loss_tta += self.compute_loss_tta(image_features, class_prototypes)
            loss_reg += self._scaled_entropy(logits[0])

        # Final loss
        self._pseudo_label_context = "main"
        loss = (
            self.beta_tta * loss_tta
            + self.beta_relation * loss_relation
            - self.beta_reg * loss_reg
            + self.beta_ood * loss_ood
        )

        if not torch.isfinite(loss):
            lib.LOGGER.warning("Skipping CLIPTTA update because the adaptation loss became non-finite.")
            self.optimizer.zero_grad(set_to_none=True)
            return loss.detach()

        loss.backward()
        self._last_backward_succeeded = True
        return loss.detach()

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
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    lib.LOGGER.warning("Skipping CLIPTTA optimizer step because gradient norm became non-finite.")
                    self.optimizer.zero_grad(set_to_none=True)
                else:
                    self.optimizer.step(closure)
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                self.optimizer.zero_grad(set_to_none=True)
        else:
            self.optimizer.zero_grad(set_to_none=True)

        # Get final logits and OOD scores
        if step == self.steps - 1:
            with torch.no_grad():
                if self.update_text:
                    class_prototypes, _ = lib.get_text_features(self.class_names, self.template, self.clip_text_encoder)
                else:
                    class_prototypes = self.class_prototypes
                image_features = self.get_features(images)
                logits = self.get_logits(image_features, class_prototypes)
                scores = self.get_scores(logits, image_features)
        else:
            logits, scores = None, None

        return logits, scores

    def compute_loss_tta(self, image_features: List[Tensor], class_prototypes: Tensor) -> Tensor:
        image_features_fp32 = image_features[0].float()
        class_prototypes_fp32 = class_prototypes.float()

        # Compute pseudo-labels
        logits = image_features_fp32 @ class_prototypes_fp32.t()
        pseudo_label_logits = self.compute_pseudo_label_logits(image_features[0], class_prototypes)
        _, pred = pseudo_label_logits.topk(1, 1, True, True)
        pred_text_features = class_prototypes_fp32[pred[:, 0]]

        # Compute logits (image v.s. pseudo-captions, i.e. size is B x B)
        logits_per_image = self.logit_scale * image_features_fp32 @ pred_text_features.t()
        logits_per_text = logits_per_image.t() if self.update_text else logits_per_image

        # TTA loss
        if self.use_tent:
            loss_tta = lib.softmax_entropy(self.logit_scale * logits).mean(0)
        elif self.use_clipartt:
            _, pred = logits.topk(self.K, 1, True, True)
            if self.K == 1:
                text_features = class_prototypes_fp32[pred[:, 0]]
            else:
                text_prompts = lib.getprompt(self.K, pred.cpu().numpy(), self.class_names, self.template[0])
                pred_inputs = clip_tokenize(text_prompts).to(logits.device)

                # With the new prompts, compute the image-to-image and text-to-text similarities to get targets
                with torch.no_grad():
                    text_features = self.clip_text_encoder(pred_inputs)
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)
                    text_features = text_features.float()

            images_similarity = image_features_fp32 @ image_features_fp32.t()
            texts_similarity = text_features @ text_features.t()
            targets = F.softmax(((images_similarity + texts_similarity) / 2) / self.clipartt_temp, dim=-1)

            # Obtain new logits (image v.s. new prompt, i.e. size is B x B)
            predictions = (self.logit_scale * text_features @ image_features_fp32.t()).t()
            loss_tta = F.cross_entropy(predictions, targets)
        else:
            if self.use_softmax_entropy:
                loss_tta = (lib.softmax_entropy(logits_per_image).mean(0) + lib.softmax_entropy(logits_per_text).mean(0)) / 2
            elif self.cliptta_target_mode == "prototype_sameclass_vv":
                loss_tta = F.cross_entropy(self.logit_scale * logits, pred[:, 0])
                if self.cliptta_sameclass_vv_weight > 0.0:
                    loss_tta = loss_tta + self.cliptta_sameclass_vv_weight * self._sameclass_vv_repulsion_loss(
                        image_features_fp32,
                        pred,
                    )
            elif self.cliptta_target_mode == "prototype_sym":
                loss_tta = self._prototype_sym_loss(
                    logits=logits.float(),
                    image_features=image_features_fp32,
                    class_prototypes=class_prototypes_fp32,
                    pred=pred,
                )
            elif self.cliptta_target_mode == "instance_proto_sym":
                targets = self._instance_targets(pred.shape[0], logits_per_image.device)
                instance_loss = (
                    self._soft_cross_entropy(logits_per_image, targets)
                    + self._soft_cross_entropy(logits_per_text, targets)
                ) / 2
                proto_sym_loss = self._prototype_sym_loss(
                    logits=logits.float(),
                    image_features=image_features_fp32,
                    class_prototypes=class_prototypes_fp32,
                    pred=pred,
                )
                weight = self.cliptta_proto_sym_weight
                loss_tta = (instance_loss + weight * proto_sym_loss) / max(1.0 + weight, 1.0e-12)
            else:
                targets = self._contrastive_targets(pred, logits_per_image.device)
                loss_tta = (
                    self._soft_cross_entropy(logits_per_image, targets)
                    + self._soft_cross_entropy(logits_per_text, targets)
                ) / 2

        return loss_tta

    def _reset_extra(self) -> None:
        if self.use_memory:
            self.memory.reset()
        self._adaptation_batch_index = 0
        self._current_adaptation_step = 0
        self._pseudo_label_context = "main"
        self._pseudo_label_clean_weight_override = None
        self._pseudo_label_aux_weight_override = None
        self.dynamic_prompt_alpha_last_mean = None

    def after_adaptation(self, **kwargs: Kwargs) -> None:
        self._pseudo_label_clean_weight_override = None
        self._pseudo_label_aux_weight_override = None
        self._adaptation_batch_index += 1
        if self.use_scheduler:
            self.scheduler.step()
