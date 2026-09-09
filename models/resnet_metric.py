from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GeM(nn.Module):
    """Generalized-mean pooling with a learnable exponent."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(min=1.0, max=8.0)
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(p), 1).pow(1.0 / p)


class SubCenterArcMarginProduct(nn.Module):
    """ArcFace classification head with K prototypes (sub-centers) per class.

    During training ``labels`` enables the angular margin for the target class.
    During validation/export the head returns scaled cosine logits without a
    label-dependent margin, which keeps inference deterministic and exportable.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.30,
        subcenters: int = 2,
    ) -> None:
        super().__init__()
        if subcenters < 1:
            raise ValueError("subcenters 必须 >= 1。")
        if not 0.0 <= margin < math.pi / 2:
            raise ValueError("ArcFace margin 必须位于 [0, pi/2)。")
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.scale = float(scale)
        self.margin = float(margin)
        self.subcenters = int(subcenters)
        self.weight = nn.Parameter(torch.empty(num_classes * subcenters, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.threshold = math.cos(math.pi - self.margin)
        self.margin_correction = math.sin(math.pi - self.margin) * self.margin

    def cosine_logits(self, features: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine = cosine.view(-1, self.num_classes, self.subcenters)
        return cosine.max(dim=2).values.clamp(-1.0, 1.0)

    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        cosine = self.cosine_logits(features)
        if labels is None or self.margin == 0.0:
            return cosine * self.scale

        sine = torch.sqrt((1.0 - cosine.square()).clamp(min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.margin_correction)
        one_hot = F.one_hot(labels, num_classes=self.num_classes).to(dtype=cosine.dtype)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale


class ResNet50MetricClassifier(nn.Module):
    """ResNet50 + GAP/GeM + embedding + Linear/Sub-center ArcFace head."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.1,
        pooling: str = "gem",
        head_type: str = "arcface",
        embedding_dim: int = 512,
        gem_p: float = 3.0,
        arcface_scale: float = 30.0,
        arcface_margin: float = 0.30,
        arcface_subcenters: int = 2,
    ) -> None:
        super().__init__()
        if pooling not in {"gap", "gem"}:
            raise ValueError("metric_pooling 必须是 'gap' 或 'gem'。")
        if head_type not in {"linear", "arcface"}:
            raise ValueError("metric_head 必须是 'linear' 或 'arcface'。")

        resnet = self._build_resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pooling_name = pooling
        self.head_type = head_type
        self.pool = GeM(p=gem_p) if pooling == "gem" else nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Linear(2048, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(dropout),
        )
        if head_type == "arcface":
            self.head: nn.Module = SubCenterArcMarginProduct(
                in_features=embedding_dim,
                num_classes=num_classes,
                scale=arcface_scale,
                margin=arcface_margin,
                subcenters=arcface_subcenters,
            )
        else:
            self.head = nn.Linear(embedding_dim, num_classes)

    @staticmethod
    def _build_resnet50(pretrained: bool) -> nn.Module:
        if not pretrained:
            return models.resnet50(weights=None)
        try:
            return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except AttributeError:
            return models.resnet50(pretrained=True)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        return self.embedding(x)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.forward_features(x)
        if self.head_type == "arcface":
            return self.head(features, labels)
        return self.head(features)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

    def get_gradcam_target_layer(self) -> nn.Module:
        return self.backbone[-1][-1]
