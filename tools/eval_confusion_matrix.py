"""
绘制鲸类物种分类混淆矩阵。

默认使用 archive/train.csv 按训练脚本相同方式切分验证集进行评估。
注意: Kaggle test_images 没有 species 真值，无法直接计算混淆矩阵。

示例:
    python tools/eval_confusion_matrix.py --checkpoint outputs/best_model.pth
    python tools/eval_confusion_matrix.py --eval-csv archive/train.csv --use-full-csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import TrainConfig
from data.dataset import (
    WhaleSpeciesDataset,
    build_label_maps,
    build_transforms,
    normalize_species_column,
    split_train_val,
)
from models.resnet_baseline import ResNet50Baseline
from models.resnet_transformer import ResNet50_Transformer
from utils.report_paths import timestamped_path


def parse_args() -> argparse.Namespace:
    """
    作用:
        解析混淆矩阵评估脚本参数。
    输入:
        命令行参数，如 --checkpoint、--eval-csv、--normalize。
    输出:
        argparse.Namespace 参数对象。
    """
    parser = argparse.ArgumentParser(description="绘制验证/测试集混淆矩阵")
    parser.add_argument("--checkpoint", default="outputs/best_model.pth", help="模型权重路径")
    parser.add_argument("--class-map", default="outputs/class_to_idx.json", help="class_to_idx.json 路径")
    parser.add_argument("--eval-csv", default="archive/train.csv", help="带 image/species 列的评估 CSV")
    parser.add_argument("--image-dir", default="archive/train_images", help="评估图片目录")
    parser.add_argument("--output", default=None, help="输出图片路径；为空时写入 outputs/reports/evaluation/ 并自动加时间戳")
    parser.add_argument("--model-type", choices=("auto", "transformer", "baseline"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=None, help="覆盖 checkpoint 中的 image_size")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="未使用 --use-full-csv 时的验证集比例")
    parser.add_argument("--split-strategy", choices=("group", "stratified"), default="group", help="验证集划分策略")
    parser.add_argument("--group-col", default="individual_id", help="group 划分使用的分组列")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-full-csv", action="store_true", help="直接使用 eval-csv 全量样本")
    parser.add_argument("--normalize", action="store_true", help="按真实类别行归一化显示比例")
    parser.add_argument("--max-samples", type=int, default=0, help="调试用，>0 时只评估前 N 张")
    return parser.parse_args()


def load_class_map(path: Path, df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    作用:
        加载类别映射；若文件不存在则根据评估 DataFrame 重建。
    输入:
        path: class_to_idx.json 路径。
        df: 包含 species 的 DataFrame。
    输出:
        (class_to_idx, idx_to_class)。
    """
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            class_to_idx = {str(k): int(v) for k, v in json.load(f).items()}
    else:
        class_to_idx, _ = build_label_maps(df["species"])
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class


def build_model(args: argparse.Namespace, checkpoint: dict, num_classes: int, device: torch.device) -> torch.nn.Module:
    """
    作用:
        根据 checkpoint 配置构造评估模型并加载权重。
    输入:
        args: 命令行参数。
        checkpoint: 已加载的 checkpoint 字典。
        num_classes: 类别数量。
        device: 评估设备。
    输出:
        eval 模式的 PyTorch 模型。
    """
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    image_size = args.image_size or int(cfg.get("image_size", 512))

    model_type = args.model_type
    if model_type == "auto":
        model_type = str(cfg.get("model_type", "transformer"))

    if model_type == "baseline":
        model = ResNet50Baseline(
            num_classes=num_classes,
            pretrained=False,
            dropout=float(cfg.get("dropout", 0.1)),
        )
    else:
        model = ResNet50_Transformer(
            num_classes=num_classes,
            image_size=image_size,
            transformer_dim=int(cfg.get("transformer_dim", 512)),
            transformer_depth=int(cfg.get("transformer_depth", 2)),
            transformer_heads=int(cfg.get("transformer_heads", 8)),
            transformer_mlp_ratio=float(cfg.get("transformer_mlp_ratio", 4.0)),
            pooling=str(cfg.get("transformer_pooling", "cls")),
            dropout=float(cfg.get("dropout", 0.1)),
            pretrained=False,
            backbone_stage=str(cfg.get("backbone_stage", "layer3")),
            token_pool_size=int(cfg.get("token_pool_size", 16)),
        )

    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def compute_confusion_matrix(labels: List[int], preds: List[int], num_classes: int) -> np.ndarray:
    """
    作用:
        根据真实标签和预测标签累计混淆矩阵。
    输入:
        labels: 真实类别编号列表。
        preds: 预测类别编号列表。
        num_classes: 类别总数。
    输出:
        NumPy 混淆矩阵，形状 [num_classes, num_classes]。
    """
    matrix = np.zeros((num_classes, num_classes), dtype=np.float64)
    for label, pred in zip(labels, preds):
        matrix[int(label), int(pred)] += 1
    return matrix


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[List[int], List[int]]:
    """
    作用:
        遍历评估 DataLoader，收集所有真实标签和预测标签。
    输入:
        model: 已加载的评估模型。
        loader: 评估 DataLoader。
        device: CPU 或 CUDA 设备。
    输出:
        (labels, preds) 两个列表。
    """
    all_labels: List[int] = []
    all_preds: List[int] = []
    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())
    return all_labels, all_preds


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: List[str],
    output_path: Path,
    normalize: bool,
) -> None:
    """
    作用:
        使用 seaborn/matplotlib 绘制并保存混淆矩阵图。
    输入:
        matrix: 混淆矩阵。
        class_names: 类别显示名称列表。
        output_path: 输出图片路径。
        normalize: 是否按真实类别行归一化。
    输出:
        无返回值；保存 PNG/JPG 图片。
    """
    if normalize:
        denom = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, np.maximum(denom, 1.0))
        fmt = ".2f"
        cbar_label = "Recall ratio"
    else:
        fmt = ".0f"
        cbar_label = "Count"

    figure_size = max(10, min(28, 0.42 * len(class_names)))
    plt.figure(figsize=(figure_size, figure_size))
    sns.heatmap(
        matrix,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={"label": cbar_label},
        fmt=fmt,
        annot=len(class_names) <= 35,
    )
    plt.xlabel("Predicted species")
    plt.ylabel("True species")
    plt.title("Whale Species Confusion Matrix")
    plt.xticks(rotation=60, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> None:
    """
    作用:
        混淆矩阵脚本主入口，负责加载数据、模型、收集预测并绘图。
    输入:
        无显式输入；通过命令行参数读取路径和评估配置。
    输出:
        无返回值；生成混淆矩阵图片。
    """
    args = parse_args()
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    class_map_path = PROJECT_ROOT / args.class_map
    eval_csv_path = PROJECT_ROOT / args.eval_csv
    image_dir = PROJECT_ROOT / args.image_dir
    output_path = PROJECT_ROOT / args.output if args.output else timestamped_path(
        PROJECT_ROOT / "outputs" / "reports" / "evaluation",
        "confusion_matrix",
        ".png",
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")
    if not eval_csv_path.exists():
        raise FileNotFoundError(f"找不到评估 CSV: {eval_csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"找不到图片目录: {image_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    image_size = args.image_size or int(cfg.get("image_size", 512))

    df = pd.read_csv(eval_csv_path)
    df = normalize_species_column(df, fix_typos=True)
    if not args.use_full_csv:
        _, df = split_train_val(
            df,
            label_col="species",
            val_ratio=args.val_ratio,
            seed=args.seed,
            split_strategy=args.split_strategy,
            group_col=args.group_col,
        )
    if args.max_samples > 0:
        df = df.head(args.max_samples).reset_index(drop=True)

    class_to_idx, idx_to_class = load_class_map(class_map_path, df)
    class_names = [idx_to_class[idx].replace("_", " ").title() for idx in range(len(class_to_idx))]

    _, val_tfms = build_transforms(image_size=image_size, cutout_p=0.0)
    dataset = WhaleSpeciesDataset(df, image_dir, class_to_idx, transforms=val_tfms)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args, checkpoint, num_classes=len(class_to_idx), device=device)
    labels, preds = collect_predictions(model, loader, device=device)
    matrix = compute_confusion_matrix(labels, preds, num_classes=len(class_to_idx))
    plot_confusion_matrix(matrix, class_names, output_path, normalize=args.normalize)
    print(f"混淆矩阵已保存: {output_path.resolve()}")


if __name__ == "__main__":
    main()
