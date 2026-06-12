from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import TrainConfig, parse_args
from data.dataset import (
    WhaleSpeciesDataset,
    build_dataloader,
    build_label_maps,
    build_transforms,
    count_group_overlap,
    normalize_species_column,
    resolve_data_paths,
    resolve_num_workers,
    split_train_val,
)
from models.resnet_baseline import ResNet50Baseline
from models.resnet_transformer import ResNet50_Transformer
from utils.losses import FocalLoss, compute_class_balanced_alpha, mixup_criterion, mixup_data
from utils.metrics import ModelEMA, accuracy_from_logits, create_summary_writer, log_epoch_metrics, macro_f1_score


def set_seed(seed: int) -> None:
    """
    作用:
        固定 Python、NumPy、PyTorch 的随机种子，降低训练结果的随机波动。
    输入:
        seed: 随机种子整数。
    输出:
        无返回值；函数会直接修改全局随机状态和 CuDNN 配置。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> AdamW:
    """
    作用:
        根据模型类型构造 AdamW 优化器，并为不同模块建立参数组，便于监控各层学习率。
    输入:
        model: 待训练模型，可以是 ResNet50Baseline 或 ResNet50_Transformer。
        cfg: 训练配置，提供 lr、weight_decay 等优化器超参数。
    输出:
        AdamW 优化器实例。
    """
    if isinstance(model, ResNet50_Transformer):
        token_params = [model.pos_embed]
        if model.pooling == "cls":
            token_params.append(model.cls_token)
        param_groups = [
            {"name": "backbone", "params": [p for p in model.cnn_backbone.parameters() if p.requires_grad], "lr": cfg.lr},
            {"name": "channel_project", "params": [p for p in model.channel_project.parameters() if p.requires_grad], "lr": cfg.lr},
            {"name": "transformer", "params": [p for p in model.transformer.parameters() if p.requires_grad], "lr": cfg.lr},
            {"name": "classifier", "params": [p for p in model.classifier.parameters() if p.requires_grad], "lr": cfg.lr},
            {"name": "tokens_and_pos", "params": [p for p in token_params if p.requires_grad], "lr": cfg.lr},
        ]
        if model.semantic_project is not None:
            param_groups.append(
                {"name": "semantic_project", "params": [p for p in model.semantic_project.parameters() if p.requires_grad], "lr": cfg.lr}
            )
    else:
        param_groups = [
            {"name": "model", "params": [p for p in model.parameters() if p.requires_grad], "lr": cfg.lr},
        ]
    param_groups = [group for group in param_groups if len(group["params"]) > 0]
    return AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)


def build_experiment_name(cfg: TrainConfig) -> str:
    """
    作用:
        根据关键训练配置自动生成实验名称，方便区分不同消融实验。
    输入:
        cfg: 训练配置；若 cfg.experiment_name 不为空，则直接使用用户指定名称。
    输出:
        实验名称字符串。
    """
    if cfg.experiment_name:
        return cfg.experiment_name

    parts = [cfg.model_type]
    if cfg.model_type == "transformer":
        parts.append(cfg.transformer_pooling)
        parts.append(cfg.backbone_stage)
        if cfg.token_pool_size > 0:
            parts.append(f"tp{cfg.token_pool_size}")
    parts.append(cfg.loss_type)
    parts.append("mixup" if cfg.mixup_alpha > 0 else "nomixup")
    parts.append("cutout" if cfg.cutout_p > 0 else "nocutout")
    parts.append("ema" if cfg.use_ema else "noema")
    return "_".join(parts)


def build_criterion(
    cfg: TrainConfig,
    train_df: pd.DataFrame,
    class_to_idx: Dict[str, int],
    device: torch.device,
) -> nn.Module:
    """
    作用:
        构造训练损失函数，支持 CrossEntropy 与 Focal Loss 两种模式。
    输入:
        cfg: 训练配置，决定 loss_type、focal_gamma 等参数。
        train_df: 训练集 DataFrame，用于统计类别频次并计算 Focal Loss alpha。
        class_to_idx: 物种名到类别编号的映射。
        device: 当前训练设备，用于把类别权重移动到 CPU/GPU。
    输出:
        PyTorch 损失函数模块。
    """
    if cfg.loss_type == "ce":
        return nn.CrossEntropyLoss()
    if cfg.loss_type == "focal":
        alpha = compute_class_balanced_alpha(train_df, class_to_idx).to(device)
        return FocalLoss(gamma=cfg.focal_gamma, alpha=alpha, reduction="mean")
    raise ValueError(f"不支持的 loss_type: {cfg.loss_type}")


def build_experiment_summary(
    cfg: TrainConfig,
    experiment_name: str,
    best_epoch: int,
    best_val_acc: float,
    best_val_macro_f1: float,
    best_path: Path,
    num_classes: int,
    train_size: int,
    val_size: int,
) -> Dict[str, object]:
    """
    作用:
        汇总一次训练实验的关键配置和最佳验证指标，便于保存为 metrics.json 或消融表。
    输入:
        cfg: 本次训练配置。
        experiment_name: 实验名称。
        best_epoch: 验证 Macro F1 最佳的 epoch。
        best_val_acc: 最佳 Macro F1 对应的验证准确率。
        best_val_macro_f1: 最佳验证 Macro F1。
        best_path: 最佳模型 checkpoint 保存路径。
        num_classes: 物种类别数。
        train_size: 训练样本数。
        val_size: 验证样本数。
    输出:
        字典形式的实验摘要。
    """
    is_transformer = cfg.model_type == "transformer"
    return {
        "experiment_name": experiment_name,
        "model_type": cfg.model_type,
        "loss_type": cfg.loss_type,
        "focal": cfg.loss_type == "focal",
        "mixup": cfg.mixup_alpha > 0,
        "cutout": cfg.cutout_p > 0,
        "transformer": is_transformer,
        "pooling": cfg.transformer_pooling if is_transformer else "gap",
        "cls_token": is_transformer and cfg.transformer_pooling == "cls",
        "token_pool_size": cfg.token_pool_size if is_transformer else 0,
        "multiscale": is_transformer and cfg.backbone_stage == "layer3_layer4",
        "ema": cfg.use_ema,
        "backbone_stage": cfg.backbone_stage if is_transformer else "resnet50",
        "split_strategy": cfg.split_strategy,
        "group_col": cfg.group_col,
        "epochs": cfg.epochs,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_macro_f1": best_val_macro_f1,
        "num_classes": num_classes,
        "train_size": train_size,
        "val_size": val_size,
        "best_model_path": str(best_path),
    }


def save_experiment_summary(output_dir: Path, summary: Dict[str, object]) -> None:
    """
    作用:
        保存单次实验摘要，并追加到消融实验汇总 CSV。
    输入:
        output_dir: 当前实验输出目录。
        summary: build_experiment_summary 生成的实验摘要字典。
    输出:
        无返回值；写入 metrics.json 和 ablation_results.csv。
    """
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    table_path = output_dir.parent / "ablation_results.csv" if output_dir.parent.name == "ablations" else output_dir / "ablation_results.csv"
    row = pd.DataFrame([summary])
    if table_path.exists():
        old = pd.read_csv(table_path)
        table = pd.concat([old, row], ignore_index=True)
    else:
        table = row
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    print(f"实验指标已保存: {metrics_path.resolve()}")
    print(f"消融汇总表已追加: {table_path.resolve()}")


def build_tensorboard_log_dir(cfg: TrainConfig, output_dir: Path, experiment_name: str) -> Path:
    """
    作用:
        构造带实验名和时间戳的 TensorBoard 日志目录，避免多次训练曲线名称混在一起。
    输入:
        cfg: 训练配置；cfg.log_dir 可指定日志根目录。
        output_dir: 当前实验输出目录。
        experiment_name: 当前实验名称。
    输出:
        TensorBoard run 目录，例如 outputs/runs/exp_name_20260420-031500。
    """
    base_log_dir = Path(cfg.log_dir) if cfg.log_dir else output_dir / "runs"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return base_log_dir / f"{experiment_name}_{timestamp}"


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    total_epochs: int,
    warmup_epochs: int | None = None,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    作用:
        构造学习率调度器，默认使用 Linear Warmup + CosineAnnealingLR。
    输入:
        optimizer: 需要调度学习率的优化器。
        cfg: 训练配置，提供 lr、warmup_epochs、warmup_start_lr。
        total_epochs: 当前训练阶段的总 epoch 数。
        warmup_epochs: 可选的 warmup 轮数；为空时使用 cfg.warmup_epochs。
    输出:
        PyTorch 学习率调度器。
    """
    total_epochs = max(1, total_epochs)
    if warmup_epochs is None:
        warmup_epochs = cfg.warmup_epochs
    warmup_epochs = min(max(0, warmup_epochs), max(0, total_epochs - 1))

    if warmup_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=cfg.lr * 0.01)

    start_factor = min(1.0, max(1e-8, cfg.warmup_start_lr / cfg.lr))
    warmup = LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=cfg.lr * 0.01)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def train_one_epoch(
    model: ResNet50_Transformer,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: object,
    device: torch.device,
    mixup_alpha: float,
    use_amp: bool,
    ema: ModelEMA | None = None,
) -> Tuple[float, float]:
    """
    作用:
        执行一个 epoch 的训练，包括 Mixup、前向传播、损失计算、反向传播、参数更新和 EMA 更新。
    输入:
        model: 待训练模型。
        loader: 训练集 DataLoader，输出 images 和 labels。
        criterion: 损失函数。
        optimizer: 优化器。
        scaler: AMP GradScaler，用于混合精度训练。
        device: CPU 或 CUDA 设备。
        mixup_alpha: Mixup 的 Beta 分布参数；<=0 时关闭 Mixup。
        use_amp: 是否启用 CUDA 混合精度。
        ema: 可选 EMA 模型包装器。
    输出:
        (平均训练损失, 平均训练准确率)。
    """
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.size(0)
        mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=mixup_alpha)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(mixed_images)
            loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (lam * (preds == labels_a).float() + (1.0 - lam) * (preds == labels_b).float()).mean().item()

        running_loss += loss.item() * batch_size
        running_acc += acc * batch_size
        total_samples += batch_size
        pbar.set_postfix(loss=running_loss / total_samples, acc=running_acc / total_samples)

    return running_loss / total_samples, running_acc / total_samples


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    num_classes: int,
) -> Tuple[float, float, float]:
    """
    作用:
        在验证集上评估模型，统计验证损失、Accuracy 和 Macro F1。
    输入:
        model: 待评估模型，通常为 EMA 模型或当前训练模型。
        loader: 验证集 DataLoader。
        criterion: 验证损失函数。
        device: CPU 或 CUDA 设备。
        use_amp: 是否启用混合精度推理。
        num_classes: 类别总数，用于计算 Macro F1。
    输出:
        (平均验证损失, 平均验证准确率, 验证 Macro F1)。
    """
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    pbar = tqdm(loader, desc="Valid", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.size(0)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        acc = accuracy_from_logits(logits, labels)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

        running_loss += loss.item() * batch_size
        running_acc += acc * batch_size
        total_samples += batch_size
        pbar.set_postfix(loss=running_loss / total_samples, acc=running_acc / total_samples)

    val_macro_f1 = macro_f1_score(all_preds, all_labels, num_classes=num_classes)
    return running_loss / total_samples, running_acc / total_samples, val_macro_f1


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    epoch: int,
    best_val_macro_f1: float,
    val_acc: float,
    class_to_idx: Dict[str, int],
    idx_to_class: Dict[int, str],
    cfg: TrainConfig,
) -> None:
    """
    作用:
        保存当前最佳模型 checkpoint，包含模型权重、优化器状态、调度器状态、类别映射和训练配置。
    输入:
        path: checkpoint 保存路径。
        model: 要保存的模型，通常是 EMA 模型或当前模型。
        optimizer: 当前优化器。
        scheduler: 当前学习率调度器。
        epoch: 当前 epoch。
        best_val_macro_f1: 当前最佳验证 Macro F1。
        val_acc: 最佳 Macro F1 对应的验证准确率。
        class_to_idx: 类别名到编号映射。
        idx_to_class: 编号到类别名映射。
        cfg: 训练配置。
    输出:
        无返回值；写入 .pth checkpoint 文件。
    """
    checkpoint = {
        "epoch": epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "val_acc_at_best_macro_f1": val_acc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "config": asdict(cfg),
    }
    torch.save(checkpoint, path)


def print_dataset_summary(df: pd.DataFrame, class_to_idx: Dict[str, int]) -> None:
    """
    作用:
        在控制台打印数据集基本统计信息，包括样本数、类别数、高频类别和低频类别。
    输入:
        df: 已清洗的训练 DataFrame，至少包含 species 列。
        class_to_idx: 类别名到编号映射。
    输出:
        无返回值；只打印文本信息。
    """
    counts = df["species"].value_counts()
    print(f"总样本数: {len(df)}")
    print(f"物种类别数: {len(class_to_idx)}")
    print("Top 10 高频类别:")
    print(counts.head(10).to_string())
    print("Tail 10 低频类别:")
    print(counts.tail(10).to_string())


def main() -> None:
    """
    作用:
        训练主入口，串联参数解析、数据集构建、模型创建、训练验证循环、最佳模型保存和实验摘要输出。
    输入:
        无显式输入；通过命令行参数和配置文件读取训练设置。
    输出:
        无返回值；生成 best_model.pth、class_to_idx.json、metrics.json、TensorBoard 日志等训练产物。
    """
    cfg = parse_args()
    set_seed(cfg.seed)
    experiment_name = build_experiment_name(cfg)

    train_csv, image_dir = resolve_data_paths(cfg)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.use_amp and device.type == "cuda"
    print(f"使用设备: {device}, AMP: {use_amp}")
    print(f"实验名称: {experiment_name}")
    print(f"CSV: {train_csv}")
    print(f"图片目录: {image_dir}")

    df = pd.read_csv(train_csv)
    df = normalize_species_column(df, fix_typos=cfg.fix_species_typos)
    class_to_idx, idx_to_class = build_label_maps(df["species"])
    print_dataset_summary(df, class_to_idx)

    train_df, val_df = split_train_val(
        df=df,
        label_col="species",
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        split_strategy=cfg.split_strategy,
        group_col=cfg.group_col,
    )
    print(f"训练集: {len(train_df)}  验证集: {len(val_df)}")
    if cfg.split_strategy == "group":
        overlap = count_group_overlap(train_df, val_df, cfg.group_col)
        print(f"Group split: {cfg.group_col} 训练/验证重叠数: {overlap}")

    with open(output_dir / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    train_tfms, val_tfms = build_transforms(image_size=cfg.image_size, cutout_p=cfg.cutout_p)
    train_dataset = WhaleSpeciesDataset(train_df, image_dir, class_to_idx, transforms=train_tfms)
    val_dataset = WhaleSpeciesDataset(val_df, image_dir, class_to_idx, transforms=val_tfms)

    pin_memory = device.type == "cuda"
    num_workers = resolve_num_workers(cfg, device)
    print(f"DataLoader workers: {num_workers}, persistent_workers: {num_workers > 0}")
    train_loader = build_dataloader(train_dataset, cfg.batch_size, True, num_workers, pin_memory, drop_last=True)
    val_loader = build_dataloader(val_dataset, cfg.batch_size, False, num_workers, pin_memory, drop_last=False)

    if cfg.model_type == "baseline":
        model = ResNet50Baseline(
            num_classes=len(class_to_idx),
            pretrained=cfg.pretrained,
            dropout=cfg.dropout,
        ).to(device)
    else:
        model = ResNet50_Transformer(
            num_classes=len(class_to_idx),
            image_size=cfg.image_size,
            transformer_dim=cfg.transformer_dim,
            transformer_depth=cfg.transformer_depth,
            transformer_heads=cfg.transformer_heads,
            transformer_mlp_ratio=cfg.transformer_mlp_ratio,
            pooling=cfg.transformer_pooling,
            dropout=cfg.dropout,
            pretrained=cfg.pretrained,
            backbone_stage=cfg.backbone_stage,
            token_pool_size=cfg.token_pool_size,
        ).to(device)
    print(f"模型类型: {cfg.model_type}, loss: {cfg.loss_type}")
    if cfg.model_type == "transformer":
        print(f"Transformer pooling: {cfg.transformer_pooling}")
        print(f"Token pooling: {cfg.token_pool_size if cfg.token_pool_size > 0 else 'disabled'}")
        print(f"Multi-scale fusion: {cfg.backbone_stage == 'layer3_layer4'}")

    if cfg.freeze_backbone_epochs > 0:
        model.set_backbone_trainable(False)
        print(f"前 {cfg.freeze_backbone_epochs} 个 epoch 冻结 ResNet50 backbone。")

    criterion = build_criterion(cfg, train_df, class_to_idx, device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, total_epochs=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = ModelEMA(model, decay=cfg.ema_decay).to(device) if cfg.use_ema else None
    if ema is not None:
        print(f"EMA 已启用，decay={cfg.ema_decay}")

    writer = create_summary_writer(build_tensorboard_log_dir(cfg, output_dir, experiment_name))
    if writer is not None:
        writer.add_text("experiment/name", experiment_name, 0)
        writer.add_text("experiment/config", json.dumps(asdict(cfg), ensure_ascii=False, indent=2), 0)
    best_val_macro_f1 = -1.0
    best_val_acc = 0.0
    best_epoch = 0
    best_path = output_dir / "best_model.pth"

    for epoch in range(1, cfg.epochs + 1):
        if cfg.freeze_backbone_epochs > 0 and epoch == cfg.freeze_backbone_epochs + 1:
            model.set_backbone_trainable(True)
            optimizer = build_optimizer(model, cfg)
            scheduler = build_scheduler(optimizer, cfg, total_epochs=max(1, cfg.epochs - epoch + 1), warmup_epochs=0)
            print("已解冻 ResNet50 backbone，并重建 optimizer/scheduler。")

        print(f"\nEpoch [{epoch}/{cfg.epochs}]  lr={optimizer.param_groups[0]['lr']:.6g}")
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            mixup_alpha=cfg.mixup_alpha,
            use_amp=use_amp,
            ema=ema,
        )

        eval_model = ema.module if ema is not None else model
        val_loss, val_acc, val_macro_f1 = validate_one_epoch(
            model=eval_model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            num_classes=len(class_to_idx),
        )

        log_epoch_metrics(writer, epoch, train_loss, train_acc, val_loss, val_acc, val_macro_f1, optimizer)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_macro_f1={val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(best_path, eval_model, optimizer, scheduler, epoch, best_val_macro_f1, val_acc, class_to_idx, idx_to_class, cfg)
            print(f"保存最佳模型: {best_path}  best_val_macro_f1={best_val_macro_f1:.4f}, val_acc={best_val_acc:.4f}")

    if writer is not None:
        writer.close()

    print(f"\n训练完成。最佳验证 Macro F1: {best_val_macro_f1:.4f}, 对应验证准确率: {best_val_acc:.4f}")
    print(f"最佳权重文件: {best_path.resolve()}")
    summary = build_experiment_summary(
        cfg=cfg,
        experiment_name=experiment_name,
        best_epoch=best_epoch,
        best_val_acc=best_val_acc,
        best_val_macro_f1=best_val_macro_f1,
        best_path=best_path,
        num_classes=len(class_to_idx),
        train_size=len(train_df),
        val_size=len(val_df),
    )
    save_experiment_summary(output_dir, summary)


if __name__ == "__main__":
    main()
