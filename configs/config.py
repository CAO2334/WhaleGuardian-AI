from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional


SPECIES_TYPO_MAP = {
    "bottlenose_dolpin": "bottlenose_dolphin",
    "kiler_whale": "killer_whale",
}


@dataclass
class TrainConfig:
    """
    作用:
        集中保存训练、模型、数据增强、评估和部署相关的超参数。
    输入:
        通过 dataclass 字段或 parse_args() 从命令行参数构造。
    输出:
        TrainConfig 实例，供 train.py 和其他脚本统一读取配置。
    """

    data_root: str = "archive"
    train_csv: Optional[str] = None
    image_dir: Optional[str] = None
    output_dir: str = "outputs"
    log_dir: Optional[str] = None
    image_size: int = 512
    val_ratio: float = 0.2
    split_strategy: str = "group"
    group_col: str = "individual_id"
    epochs: int = 20
    batch_size: int = 8
    num_workers: int = -1
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    warmup_start_lr: float = 1e-6
    model_type: str = "transformer"
    experiment_name: Optional[str] = None
    loss_type: str = "focal"
    transformer_dim: int = 512
    transformer_depth: int = 2
    transformer_heads: int = 8
    transformer_mlp_ratio: float = 4.0
    transformer_pooling: str = "cls"
    token_pool_size: int = 16
    dropout: float = 0.1
    backbone_stage: str = "layer3_layer4"
    focal_gamma: float = 2.0
    mixup_alpha: float = 0.4
    cutout_p: float = 0.5
    seed: int = 42
    use_amp: bool = True
    pretrained: bool = True
    fix_species_typos: bool = True
    freeze_backbone_epochs: int = 0
    use_ema: bool = True
    ema_decay: float = 0.999


def parse_args() -> TrainConfig:
    """
    作用:
        解析命令行参数，并转换为 TrainConfig 配置对象。
    输入:
        命令行参数，例如 --epochs、--batch-size、--backbone-stage。
    输出:
        TrainConfig 实例。
    """
    parser = argparse.ArgumentParser(description="AI护鲸使者 ResNet50-Transformer 物种分类训练")
    parser.add_argument("--data-root", default="archive", help="数据根目录，默认 archive")
    parser.add_argument("--train-csv", default=None, help="训练 CSV 路径；为空时使用 data_root/train.csv")
    parser.add_argument("--image-dir", default=None, help="训练图片目录；为空时使用 data_root/train_images")
    parser.add_argument("--output-dir", default="outputs", help="模型与日志输出目录")
    parser.add_argument("--log-dir", default=None, help="TensorBoard 日志目录；为空时使用 output_dir/runs")
    parser.add_argument("--image-size", type=int, default=512, help="输入图像尺寸，默认 512x512")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument(
        "--split-strategy",
        choices=("group", "stratified"),
        default="group",
        help="验证集划分策略；group 会按 individual_id 分组防止同一鲸鱼个体泄漏",
    )
    parser.add_argument("--group-col", default="individual_id", help="group 划分使用的分组列，默认 individual_id")
    parser.add_argument("--epochs", type=int, default=20, help="训练 epoch 数")
    parser.add_argument("--batch-size", type=int, default=8, help="批大小；显存不足时调小")
    parser.add_argument("--num-workers", type=int, default=-1, help="DataLoader 进程数；-1 表示自动，Linux/CUDA 默认 4，Windows 默认 0")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW 初始学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW 权重衰减")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="学习率预热 epoch 数；Transformer 训练建议 3-5")
    parser.add_argument("--warmup-start-lr", type=float, default=1e-6, help="Warmup 起始学习率")
    parser.add_argument("--model-type", choices=("transformer", "baseline"), default="transformer", help="模型类型：transformer 或纯 ResNet50 baseline")
    parser.add_argument("--experiment-name", default=None, help="实验名称；为空时根据关键配置自动生成")
    parser.add_argument("--loss-type", choices=("focal", "ce"), default="focal", help="损失函数类型：focal 或 ce")
    parser.add_argument("--transformer-dim", type=int, default=512, help="Transformer token 维度")
    parser.add_argument("--transformer-depth", type=int, default=2, help="Transformer Encoder 层数")
    parser.add_argument("--transformer-heads", type=int, default=8, help="多头注意力头数")
    parser.add_argument("--transformer-mlp-ratio", type=float, default=4.0, help="Transformer FFN 扩张倍率")
    parser.add_argument("--transformer-pooling", choices=("cls", "mean"), default="cls", help="Transformer 输出聚合方式，用于 CLS Token 消融")
    parser.add_argument("--token-pool-size", type=int, default=16, help="Transformer token 空间池化尺寸；16 表示 16x16，0 表示不池化")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 概率")
    parser.add_argument(
        "--backbone-stage",
        choices=("layer3", "layer4", "layer3_layer4"),
        default="layer3_layer4",
        help="ResNet 特征输出层；layer3_layer4 表示 layer3 token + layer4 全局语义融合",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal Loss gamma，越大越关注难样本")
    parser.add_argument("--mixup-alpha", type=float, default=0.4, help="Mixup Beta 分布参数；<=0 时关闭")
    parser.add_argument("--cutout-p", type=float, default=0.5, help="Cutout/CoarseDropout 概率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0, help="前 N 个 epoch 冻结 ResNet 主干")
    parser.add_argument("--no-amp", action="store_true", help="关闭 CUDA 混合精度")
    parser.add_argument("--no-pretrained", action="store_true", help="不加载 ImageNet 预训练 ResNet50")
    parser.add_argument("--no-fix-species-typos", action="store_true", help="不合并 Kaggle 中的物种拼写噪声")
    parser.add_argument("--no-ema", action="store_true", help="关闭 EMA 指数滑动平均模型")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA 衰减系数，常用 0.999 或 0.9999")
    args = parser.parse_args()

    return TrainConfig(
        data_root=args.data_root,
        train_csv=args.train_csv,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        split_strategy=args.split_strategy,
        group_col=args.group_col,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        warmup_start_lr=args.warmup_start_lr,
        model_type=args.model_type,
        experiment_name=args.experiment_name,
        loss_type=args.loss_type,
        transformer_dim=args.transformer_dim,
        transformer_depth=args.transformer_depth,
        transformer_heads=args.transformer_heads,
        transformer_mlp_ratio=args.transformer_mlp_ratio,
        transformer_pooling=args.transformer_pooling,
        token_pool_size=args.token_pool_size,
        dropout=args.dropout,
        backbone_stage=args.backbone_stage,
        focal_gamma=args.focal_gamma,
        mixup_alpha=args.mixup_alpha,
        cutout_p=args.cutout_p,
        seed=args.seed,
        use_amp=not args.no_amp,
        pretrained=not args.no_pretrained,
        fix_species_typos=not args.no_fix_species_typos,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        use_ema=not args.no_ema,
        ema_decay=args.ema_decay,
    )
