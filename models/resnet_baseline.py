from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ResNet50Baseline(nn.Module):
    """
    纯净 ResNet50 baseline，用于和 ResNet50-Transformer 做消融对比。

    结构:
        Image -> ResNet50 backbone -> global average pooling -> linear classifier
    输入:
        num_classes: 输出类别数。
        pretrained: 是否加载 ImageNet 预训练权重。
        dropout: 分类头 Dropout 概率。
    输出:
        forward(x) 返回 [B, num_classes] 的 logits。
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = self._build_resnet50(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    @staticmethod
    def _build_resnet50(pretrained: bool) -> nn.Module:
        """
        作用:
            构建 torchvision ResNet50，并兼容新旧 torchvision 权重 API。
        输入:
            pretrained: 是否加载 ImageNet 预训练权重。
        输出:
            ResNet50 模型实例。
        """
        if not pretrained:
            return models.resnet50(weights=None)
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            return models.resnet50(weights=weights)
        except AttributeError:
            return models.resnet50(pretrained=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        作用:
            执行 ResNet50 baseline 前向传播。
        输入:
            x: 输入图像 Tensor，形状 [B, 3, H, W]。
        输出:
            分类 logits，形状 [B, num_classes]。
        """
        return self.backbone(x)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """
        作用:
            控制 ResNet50 backbone 是否参与训练，分类头始终保持可训练。
        输入:
            trainable: True 表示解冻 backbone，False 表示冻结 backbone。
        输出:
            无返回值；直接修改参数 requires_grad。
        """
        for name, param in self.backbone.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = trainable
