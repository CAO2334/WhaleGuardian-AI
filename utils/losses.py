from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    作用:
        实现 Focal Loss，用于降低易分类样本权重、增强模型对难样本和长尾类别的关注。
    输入:
        gamma: 难样本调节系数，越大越关注低置信度样本。
        alpha: 可选类别权重 Tensor，用于类别不平衡修正。
        reduction: 损失聚合方式，支持 mean/sum/none。
        eps: 数值稳定项。
    输出:
        forward(logits, targets) 返回标量损失或逐样本损失。
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        作用:
            根据模型 logits 和真实标签计算 Focal Loss。
        输入:
            logits: 模型输出，形状 [B, num_classes]。
            targets: 真实类别编号，形状 [B]。
        输出:
            根据 reduction 聚合后的损失 Tensor。
        """
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        targets = targets.long()
        pt = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1).clamp_min(self.eps)
        log_pt = log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        loss = -((1.0 - pt).pow(self.gamma)) * log_pt
        if self.alpha is not None:
            loss = self.alpha.gather(dim=0, index=targets) * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_class_balanced_alpha(df: pd.DataFrame, class_to_idx: Dict[str, int]) -> torch.Tensor:
    """
    作用:
        根据训练集每个类别样本数计算 Focal Loss 的类别平衡 alpha 权重。
    输入:
        df: 训练集 DataFrame，必须包含 species 列。
        class_to_idx: 物种名到类别编号映射。
    输出:
        alpha Tensor，形状 [num_classes]，均值归一化到 1 附近。
    """
    counts = df["species"].value_counts()
    alpha = torch.ones(len(class_to_idx), dtype=torch.float32)
    for class_name, idx in class_to_idx.items():
        alpha[idx] = 1.0 / math.sqrt(float(counts[class_name]))
    return alpha / alpha.mean()


def mixup_data(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    作用:
        在 batch 维度执行 Mixup 图像与标签混合。
    输入:
        images: 输入图像 Tensor，形状 [B, C, H, W]。
        labels: 原始类别标签，形状 [B]。
        alpha: Beta 分布参数；<=0 时关闭 Mixup。
    输出:
        (mixed_images, labels_a, labels_b, lambda)，用于后续混合损失计算。
    """
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[index]
    return mixed_images, labels, labels[index], lam


def mixup_criterion(
    criterion: nn.Module,
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    作用:
        计算 Mixup 损失，即两组标签损失按 lambda 加权求和。
    输入:
        criterion: 基础损失函数，如 FocalLoss 或 CrossEntropyLoss。
        logits: 模型输出 logits。
        labels_a: 原 batch 标签。
        labels_b: 打乱后的 batch 标签。
        lam: Mixup 混合系数。
    输出:
        Mixup 后的标量损失。
    """
    return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
