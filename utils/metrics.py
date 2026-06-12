from __future__ import annotations

import copy
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    作用:
        根据 logits 计算 Top-1 Accuracy。
    输入:
        logits: 模型输出，形状 [B, num_classes]。
        labels: 真实标签，形状 [B]。
    输出:
        Python float 类型的准确率。
    """
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def macro_f1_score(preds: List[int], labels: List[int], num_classes: int) -> float:
    """
    作用:
        计算 Macro F1，适合评估长尾类别表现。
    输入:
        preds: 预测类别编号列表。
        labels: 真实类别编号列表。
        num_classes: 类别总数。
    输出:
        Macro F1 浮点数；优先使用 sklearn，不可用时使用 NumPy fallback。
    """
    if len(labels) == 0:
        return 0.0
    try:
        from sklearn.metrics import f1_score

        return float(
            f1_score(
                labels,
                preds,
                labels=list(range(num_classes)),
                average="macro",
                zero_division=0,
            )
        )
    except ImportError:
        pass

    preds_np = np.asarray(preds, dtype=np.int64)
    labels_np = np.asarray(labels, dtype=np.int64)
    f1_values = []
    for class_idx in range(num_classes):
        tp = np.sum((preds_np == class_idx) & (labels_np == class_idx))
        fp = np.sum((preds_np == class_idx) & (labels_np != class_idx))
        fn = np.sum((preds_np != class_idx) & (labels_np == class_idx))
        denom = 2 * tp + fp + fn
        f1_values.append(float(2 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1_values))


class ModelEMA:
    """
    作用:
        维护模型参数的指数滑动平均副本，用于更稳定的验证和 checkpoint 保存。
    输入:
        model: 原始训练模型。
        decay: EMA 衰减系数，越接近 1 平滑越强。
    输出:
        module 属性保存 EMA 模型；update(model) 用当前模型参数更新 EMA。
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        作用:
            用当前训练模型参数更新 EMA 模型参数。
        输入:
            model: 当前训练模型。
        输出:
            无返回值；原地更新 self.module 的 state_dict。
        """
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for name, ema_value in ema_state.items():
            model_value = model_state[name].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(model_value.to(dtype=ema_value.dtype), alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)

    def to(self, device: torch.device) -> "ModelEMA":
        """
        作用:
            将 EMA 模型移动到指定设备。
        输入:
            device: CPU 或 CUDA 设备。
        输出:
            self，便于链式调用。
        """
        self.module.to(device)
        return self


def create_summary_writer(log_dir: Path) -> Optional[object]:
    """
    作用:
        创建 TensorBoard SummaryWriter；若未安装 tensorboard 则返回 None。
    输入:
        log_dir: TensorBoard 日志目录。
    输出:
        SummaryWriter 实例或 None。
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("提示: 当前环境缺少 tensorboard，跳过 TensorBoard 日志。安装命令: pip install tensorboard")
        return None
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard 日志目录: {log_dir.resolve()}")
    print(f"查看命令: tensorboard --logdir {log_dir.resolve()}")
    return writer


def log_epoch_metrics(
    writer: Optional[object],
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    val_macro_f1: float,
    optimizer: torch.optim.Optimizer,
) -> None:
    """
    作用:
        将一个 epoch 的训练/验证指标和各参数组学习率写入 TensorBoard。
    输入:
        writer: SummaryWriter 或 None。
        epoch: 当前 epoch。
        train_loss/train_acc: 训练损失和准确率。
        val_loss/val_acc/val_macro_f1: 验证指标。
        optimizer: 优化器，用于读取各参数组 lr。
    输出:
        无返回值；写入 TensorBoard 日志。
    """
    if writer is None:
        return
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)
    writer.add_scalar("F1/val_macro", val_macro_f1, epoch)
    for idx, group in enumerate(optimizer.param_groups):
        group_name = group.get("name", f"group_{idx}")
        writer.add_scalar(f"LR/{group_name}", group["lr"], epoch)
    writer.flush()
