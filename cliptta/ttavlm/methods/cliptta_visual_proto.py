from __future__ import annotations

from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

import ttavlm.lib as lib
from ttavlm.memory import ClasswiseVisualPrototypeMemory
from ttavlm.methods.cliptta_otsu import CLIPTTA
from ttavlm.models.clip import tokenize as clip_tokenize

Kwargs = Dict[str, Any]


class CLIPTTAVisualProto(CLIPTTA):
    """CLIPTTA variant that applies visual-prototype reranking to every sample."""

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
        visual_proto_bank_size: int = 32,
        visual_proto_lambda: float = 0.05,
        visual_proto_count_tau: float = 16.0,
        visual_proto_update_threshold: float = 0.85,
        visual_proto_low_conf_threshold: float = 0.5,
        visual_proto_high_conf_threshold: float = 0.75,
        visual_proto_margin_threshold: float = 0.15,
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
        self.visual_proto_bank_size = max(int(visual_proto_bank_size), 1)
        self.visual_proto_lambda = float(visual_proto_lambda)
        self.visual_proto_count_tau = float(visual_proto_count_tau)
        self.visual_proto_update_threshold = float(visual_proto_update_threshold)
        self.visual_proto_low_conf_threshold = float(visual_proto_low_conf_threshold)
        self.visual_proto_high_conf_threshold = float(visual_proto_high_conf_threshold)
        self.visual_proto_margin_threshold = max(float(visual_proto_margin_threshold), 1.0e-6)
        self.visual_proto_memory = ClasswiseVisualPrototypeMemory(
            num_classes=len(self.class_names),
            bank_size=self.visual_proto_bank_size,
        )

    def _visual_proto_alpha(self, counts: Tensor) -> Tensor:
        if self.visual_proto_lambda == 0.0:
            return torch.zeros_like(counts, dtype=torch.float32)
        counts = counts.float()
        if self.visual_proto_count_tau <= 0:
            weights = (counts > 0).float()
        else:
            weights = 1.0 - torch.exp(-counts / self.visual_proto_count_tau)
        return self.visual_proto_lambda * weights

    def _text_logits(
        self,
        image_features: Tensor,
        class_prototypes: Tensor,
    ) -> Tensor:
        return image_features.float() @ class_prototypes.float().t()

    def _text_predictions_and_scores(
        self,
        image_features: Tensor,
        class_prototypes: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        text_logits = self._text_logits(image_features, class_prototypes)
        text_probs = (self.logit_scale * text_logits).softmax(dim=-1)
        confidences, pred = text_probs.max(dim=1)
        scores = self.get_scores([text_logits], image_features, score_type=self.id_score_type)
        return pred, confidences, scores, text_logits

    def _sample_gate_from_text_probs(self, text_probs: Tensor) -> Tensor:
        batch_size = text_probs.shape[0]
        device = text_probs.device
        dtype = text_probs.dtype
        # Visual prototypes are now consulted for every sample once a prototype bank exists.
        return torch.ones(batch_size, device=device, dtype=dtype)

    def _combined_logits_for_views(
        self,
        image_features: List[Tensor],
        class_prototypes: Tensor,
    ) -> List[Tensor]:
        return [self.compute_pseudo_label_logits(features.float(), class_prototypes.float())[0] for features in image_features]

    def compute_pseudo_label_logits(
        self,
        image_features: Tensor,
        class_prototypes: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        text_logits = self._text_logits(image_features, class_prototypes)
        text_probs = (self.logit_scale * text_logits).softmax(dim=-1)
        proto_payload = self.visual_proto_memory.prototype_matrix(text_logits.device)
        if proto_payload is None:
            visual_logits = torch.zeros_like(text_logits)
            gate = torch.zeros_like(text_logits)
            return text_logits, text_logits, visual_logits, gate

        visual_prototypes, counts = proto_payload
        visual_logits = image_features.float() @ visual_prototypes.t()
        visual_probs = (self.logit_scale * visual_logits).softmax(dim=-1)
        class_alpha = self._visual_proto_alpha(counts).to(device=text_logits.device, dtype=text_logits.dtype)
        sample_gate = self._sample_gate_from_text_probs(text_probs).to(device=text_logits.device, dtype=text_logits.dtype)
        effective_gate = sample_gate.unsqueeze(1) * class_alpha.unsqueeze(0)
        combined_probs = text_probs + effective_gate * visual_probs
        combined_probs = combined_probs / combined_probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        combined_logits = combined_probs.clamp_min(1.0e-8).log() / float(self.logit_scale)
        return combined_logits, text_logits, visual_logits, effective_gate

    def _combined_predictions_and_scores(
        self,
        image_features: Tensor,
        class_prototypes: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        combined_logits, _, _, _ = self.compute_pseudo_label_logits(image_features, class_prototypes)
        probs = (self.logit_scale * combined_logits).softmax(dim=-1)
        confidences, pred = probs.max(dim=1)
        if self.id_score_type == "max_prob":
            scores = confidences
        else:
            scores = self.get_scores([combined_logits], image_features, score_type=self.id_score_type)
        return pred, confidences, scores

    def _update_visual_proto_memory(
        self,
        image_features: Tensor,
        class_prototypes: Tensor,
    ) -> None:
        with torch.no_grad():
            pred, confidences, _, _ = self._text_predictions_and_scores(image_features.float(), class_prototypes.float())
            keep = confidences >= self.visual_proto_update_threshold
            if not bool(keep.any()):
                return
            self.visual_proto_memory.update(
                image_features[keep],
                pred[keep],
                confidences[keep],
            )

    def compute_loss_tta(self, image_features: List[Tensor], class_prototypes: Tensor) -> Tensor:
        image_features_fp32 = image_features[0].float()
        class_prototypes_fp32 = class_prototypes.float()

        logits, _, _, _ = self.compute_pseudo_label_logits(image_features_fp32, class_prototypes_fp32)
        _, pred = logits.topk(1, 1, True, True)
        pred_text_features = class_prototypes_fp32[pred[:, 0]]

        logits_per_image = self.logit_scale * image_features_fp32 @ pred_text_features.t()
        logits_per_text = logits_per_image.t() if self.update_text else logits_per_image

        if self.use_tent:
            loss_tta = lib.softmax_entropy(self.logit_scale * logits).mean(0)
        elif self.use_clipartt:
            _, pred = logits.topk(self.K, 1, True, True)
            if self.K == 1:
                text_features = class_prototypes_fp32[pred[:, 0]]
            else:
                text_prompts = lib.getprompt(self.K, pred.cpu().numpy(), self.class_names, self.template[0])
                pred_inputs = clip_tokenize(text_prompts).to(logits.device)

                with torch.no_grad():
                    text_features = self.clip_text_encoder(pred_inputs)
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)
                    text_features = text_features.float()

            images_similarity = image_features_fp32 @ image_features_fp32.t()
            texts_similarity = text_features @ text_features.t()
            targets = F.softmax(((images_similarity + texts_similarity) / 2) / self.clipartt_temp, dim=-1)
            predictions = (self.logit_scale * text_features @ image_features_fp32.t()).t()
            loss_tta = F.cross_entropy(predictions, targets)
        else:
            if self.use_softmax_entropy:
                loss_tta = (lib.softmax_entropy(logits_per_image).mean(0) + lib.softmax_entropy(logits_per_text).mean(0)) / 2
            else:
                targets = torch.eye(logits_per_image.shape[0]).to(logits_per_image.device)
                loss_tta = (self.loss_fn(logits_per_image, targets).mean(0) + self.loss_fn(logits_per_text, targets).mean(0)) / 2

        return loss_tta

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

        loss_tta = self.compute_loss_tta(image_features, class_prototypes)
        loss_relation = self.compute_relation_consistency_loss(image_features)

        if self.use_memory:
            if step == 0:
                with torch.no_grad():
                    pred, _, scores_mem, _ = self._text_predictions_and_scores(image_features[0], class_prototypes)
                self.memory.update(images[0].cpu().detach(), pred.cpu().detach(), scores_mem.cpu().detach())

            images_mem, _, _ = self.memory.sample()
            image_features_mem = self.get_features([images_mem.to(image_features[0].device)])
            logits_mem = self._combined_logits_for_views(image_features_mem, class_prototypes)
            loss_tta += self.compute_loss_tta(image_features_mem, class_prototypes)
            loss_reg += self._scaled_entropy(logits_mem[0])

        loss = (
            self.beta_tta * loss_tta
            + self.beta_relation * loss_relation
            - self.beta_reg * loss_reg
            + self.beta_ood * loss_ood
        )

        if not torch.isfinite(loss):
            lib.LOGGER.warning("Skipping CLIPTTA visual-proto update because the adaptation loss became non-finite.")
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
                    lib.LOGGER.warning("Skipping CLIPTTA visual-proto optimizer step because gradient norm became non-finite.")
                    self.optimizer.zero_grad(set_to_none=True)
                else:
                    self.optimizer.step(closure)
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                self.optimizer.zero_grad(set_to_none=True)
        else:
            self.optimizer.zero_grad(set_to_none=True)

        if step == self.steps - 1:
            with torch.no_grad():
                if self.update_text:
                    class_prototypes, _ = lib.get_text_features(self.class_names, self.template, self.clip_text_encoder)
                else:
                    class_prototypes = self.class_prototypes
                image_features = self.get_features(images)
                self._update_visual_proto_memory(image_features[0], class_prototypes)
                logits = self._combined_logits_for_views(image_features, class_prototypes)
                scores = self.get_scores(logits, image_features)
        else:
            logits, scores = None, None

        return logits, scores

    def _reset_extra(self) -> None:
        super()._reset_extra()
        self.visual_proto_memory.reset()
