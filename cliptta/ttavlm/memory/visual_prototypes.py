from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


class ClasswiseVisualPrototypeMemory:
    """Keep a small high-confidence feature bank per class and expose class prototypes."""

    def __init__(
        self,
        num_classes: int,
        bank_size: int = 32,
    ) -> None:
        self.num_classes = int(num_classes)
        self.bank_size = max(int(bank_size), 1)
        self.reset()

    def reset(self) -> None:
        self.features: List[Optional[torch.Tensor]] = [None for _ in range(self.num_classes)]
        self.confidences: List[Optional[torch.Tensor]] = [None for _ in range(self.num_classes)]

    @property
    def per_class_counts(self) -> List[int]:
        counts: List[int] = []
        for features in self.features:
            counts.append(0 if features is None else int(features.shape[0]))
        return counts

    def update(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        confidences: torch.Tensor,
    ) -> None:
        if features.numel() == 0:
            return
        features = features.detach().cpu().float()
        labels = labels.detach().cpu().long()
        confidences = confidences.detach().cpu().float()

        for class_index in range(self.num_classes):
            mask = labels == class_index
            if not bool(mask.any()):
                continue

            class_features = features[mask]
            class_confidences = confidences[mask]
            stored_features = self.features[class_index]
            stored_confidences = self.confidences[class_index]

            if stored_features is None:
                merged_features = class_features
                merged_confidences = class_confidences
            else:
                merged_features = torch.cat([stored_features, class_features], dim=0)
                merged_confidences = torch.cat([stored_confidences, class_confidences], dim=0)

            if merged_confidences.shape[0] > self.bank_size:
                keep_indices = merged_confidences.topk(k=self.bank_size).indices
                merged_features = merged_features[keep_indices]
                merged_confidences = merged_confidences[keep_indices]

            self.features[class_index] = merged_features
            self.confidences[class_index] = merged_confidences

    def prototype_matrix(
        self,
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        first_feature = next((features for features in self.features if features is not None and features.shape[0] > 0), None)
        if first_feature is None:
            return None

        feature_dim = int(first_feature.shape[1])
        prototypes = torch.zeros((self.num_classes, feature_dim), dtype=torch.float32, device=device)
        counts = torch.zeros((self.num_classes,), dtype=torch.float32, device=device)

        for class_index, class_features in enumerate(self.features):
            if class_features is None or class_features.shape[0] == 0:
                continue
            prototype = class_features.mean(dim=0, keepdim=False).to(device=device, dtype=torch.float32)
            prototypes[class_index] = F.normalize(prototype.unsqueeze(0), dim=-1).squeeze(0)
            counts[class_index] = float(class_features.shape[0])

        return prototypes, counts
