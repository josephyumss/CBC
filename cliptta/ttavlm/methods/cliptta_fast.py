from typing import Any, Dict, List, Tuple
from typing_extensions import TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor

from ttavlm.methods.abstract_model import AbstractOpenSetTTAModel
import ttavlm.lib as lib


Kwargs: TypeAlias = Dict[str, Any]


class CLIPTTAFast(AbstractOpenSetTTAModel):
    """
    CLIPTTA-style contrastive adaptation without model backpropagation.

    The method keeps CLIPTTA's image-to-pseudo-text contrastive signal, but uses
    it only to update a compact confident cache. At prediction time, cache
    affinities are added to the original CLIP logits, similar in spirit to TDA
    and ETTA state updates.
    """

    def __init__(
        self,
        template: List[str],
        class_names: List[str],
        fast_cache_alpha: float = 0.2,
        fast_cache_beta: float = 5.0,
        fast_update_ratio: float = 0.1,
        fast_entropy_threshold: float = 1.0,
        fast_use_text_affinity: bool = False,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.template = template
        self.class_names = class_names
        self.fast_cache_alpha = fast_cache_alpha
        self.fast_cache_beta = fast_cache_beta
        self.fast_update_ratio = fast_update_ratio
        self.fast_entropy_threshold = fast_entropy_threshold
        self.fast_use_text_affinity = fast_use_text_affinity
        self.cache: Dict[int, List[Dict[str, Tensor]]] = {}

        # This variant never backprops, so keep the encoders in inference mode.
        self.model.eval()
        self.model.requires_grad_(False)
        if self.clip_text_encoder is not None:
            self.clip_text_encoder.eval()
            self.clip_text_encoder.requires_grad_(False)

    def forward_and_adapt(
        self,
        images: List[Tensor],
        step: int,
        labels: Tensor = None,
    ) -> Tuple[List[Tensor], Tensor]:
        with torch.no_grad():
            image_features = self.get_features(images)
            base_logits = self.get_logits(image_features)[0]

            final_logits = base_logits + self.compute_cache_logits(image_features[0], base_logits)
            # Avoid using the current batch to correct itself. The cache update
            # becomes available from the next batch, reducing confirmation bias.
            self.update_contrastive_cache(image_features[0], base_logits)
            scores = self.get_scores([final_logits], image_features)

        return [final_logits], scores

    def update_contrastive_cache(
        self,
        image_features: Tensor,
        base_logits: Tensor,
    ) -> None:
        if image_features.shape[0] == 0:
            return

        scaled_logits = self.logit_scale * base_logits
        entropy = lib.softmax_entropy(scaled_logits)
        keep_count = max(1, int(image_features.shape[0] * self.fast_update_ratio))
        selected_idx = torch.argsort(entropy, descending=False)[:keep_count]
        selected_idx = selected_idx[entropy[selected_idx] <= self.fast_entropy_threshold]
        if selected_idx.numel() == 0:
            return

        selected_images = image_features[selected_idx]
        selected_logits = base_logits[selected_idx]
        _, selected_pred = selected_logits.topk(1, 1, True, True)
        selected_pred = selected_pred[:, 0]
        selected_text = self.class_prototypes[selected_pred]

        # Same BxB image-to-pseudo-text contrastive object as CLIPTTA, but used
        # as a confidence signal instead of a differentiable loss.
        contrast_logits = self.logit_scale * selected_images @ selected_text.t()
        contrast_prob = F.softmax(contrast_logits, dim=-1).diag()
        max_prob = F.softmax(scaled_logits[selected_idx], dim=-1).max(dim=-1).values
        confidence = contrast_prob * max_prob

        for row, class_index in enumerate(selected_pred.tolist()):
            entry = {
                "image": selected_images[row : row + 1].detach(),
                "text": selected_text[row : row + 1].detach(),
                "confidence": confidence[row : row + 1].detach(),
            }
            self.update_class_cache(class_index, entry)

    def update_class_cache(self, class_index: int, entry: Dict[str, Tensor]) -> None:
        class_cache = self.cache.setdefault(class_index, [])
        class_cache.append(entry)
        class_cache.sort(key=lambda item: float(item["confidence"].item()), reverse=True)
        del class_cache[self.num_shots :]

    def compute_cache_logits(
        self,
        image_features: Tensor,
        base_logits: Tensor,
    ) -> Tensor:
        if not self.cache:
            return torch.zeros_like(base_logits)

        cache_images = []
        cache_texts = []
        cache_labels = []
        cache_confidences = []
        for class_index in sorted(self.cache.keys()):
            for entry in self.cache[class_index]:
                cache_images.append(entry["image"])
                cache_texts.append(entry["text"])
                cache_labels.append(class_index)
                cache_confidences.append(entry["confidence"])

        compute_dtype = torch.float32
        query_features = image_features.to(dtype=compute_dtype)
        cache_image_keys = torch.cat(cache_images, dim=0).to(
            device=image_features.device,
            dtype=compute_dtype,
        )
        cache_text_keys = torch.cat(cache_texts, dim=0).to(
            device=image_features.device,
            dtype=compute_dtype,
        )
        cache_conf = torch.cat(cache_confidences, dim=0).to(
            device=image_features.device,
            dtype=compute_dtype,
        ).view(1, -1)

        image_affinity = query_features @ cache_image_keys.t()
        if self.fast_use_text_affinity:
            pred = base_logits.topk(1, 1, True, True).indices[:, 0]
            query_text = self.class_prototypes[pred].to(
                device=image_features.device,
                dtype=compute_dtype,
            )
            text_affinity = query_text @ cache_text_keys.t()
            affinity = (image_affinity + text_affinity) / 2.0
        else:
            affinity = image_affinity

        weights = torch.exp(self.fast_cache_beta * (affinity - 1.0)) * cache_conf
        values = F.one_hot(
            torch.tensor(cache_labels, dtype=torch.long, device=image_features.device),
            num_classes=self.class_prototypes.shape[0],
        ).to(weights.dtype)
        # Keep weak cache evidence weak. Normalizing by the total weight makes a
        # single low-similarity cache entry behave like a full one-hot logit.
        cache_logits = weights @ values
        cache_delta = (self.fast_cache_alpha / float(self.logit_scale)) * cache_logits
        return cache_delta.to(dtype=base_logits.dtype)

    def _reset_extra(self) -> None:
        self.cache = {}
