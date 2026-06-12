"""
Happywhale 数据集长尾分布与模型分类表现分析工具。

默认只依赖 train.csv，输出类别长尾图和训练/验证划分统计。
如果额外传入 --checkpoint，则会在验证集上评估模型，输出各类别 F1 和最易混淆类别对。

示例:
    python tools/analyze_dataset.py
    python tools/analyze_dataset.py --checkpoint outputs/best_model.pth
    python tools/analyze_dataset.py --output-dir outputs/analysis --val-ratio 0.2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")

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

from data.dataset import (
    WhaleSpeciesDataset,
    build_label_maps,
    build_transforms,
    count_group_overlap,
    normalize_species_column,
    split_train_val,
)
from models.resnet_baseline import ResNet50Baseline
from models.resnet_transformer import ResNet50_Transformer


def parse_args() -> argparse.Namespace:
    """
    作用:
        解析数据集分析和可选 checkpoint 评估参数。
    输入:
        命令行参数，如 --csv、--output-dir、--checkpoint。
    输出:
        argparse.Namespace 参数对象。
    """
    parser = argparse.ArgumentParser(description="分析鲸类数据集长尾分布与可选模型分类表现")
    parser.add_argument("--csv", default="archive/train.csv", help="包含 image/species 列的 CSV")
    parser.add_argument("--image-dir", default="archive/train_images", help="图片目录；仅 checkpoint 评估时需要")
    parser.add_argument("--output-dir", default="outputs/analysis", help="分析结果输出目录")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例，与训练脚本保持一致")
    parser.add_argument(
        "--split-strategy",
        choices=("group", "stratified"),
        default="group",
        help="验证集划分策略；group 会按 individual_id 分组防止泄漏",
    )
    parser.add_argument("--group-col", default="individual_id", help="group 划分使用的分组列")
    parser.add_argument("--seed", type=int, default=42, help="划分随机种子，与训练脚本保持一致")
    parser.add_argument("--fix-species-typos", action="store_true", default=True, help="合并 Kaggle 物种拼写噪声")
    parser.add_argument("--no-fix-species-typos", dest="fix_species_typos", action="store_false")
    parser.add_argument("--checkpoint", default=None, help="可选：best_model.pth 路径，用于计算 per-class F1 和混淆类别对")
    parser.add_argument("--class-map", default="outputs/class_to_idx.json", help="class_to_idx.json 路径；缺失时根据 CSV 重建")
    parser.add_argument("--model-type", choices=("auto", "transformer", "baseline"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16, help="checkpoint 评估 batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="checkpoint 评估 DataLoader workers")
    parser.add_argument("--image-size", type=int, default=None, help="覆盖 checkpoint 中保存的 image_size")
    parser.add_argument("--use-full-csv", action="store_true", help="checkpoint 评估时直接使用 CSV 全量样本")
    parser.add_argument("--max-samples", type=int, default=0, help="调试用，>0 时 checkpoint 评估只跑前 N 张")
    parser.add_argument("--top-k-confusions", type=int, default=20, help="输出最严重混淆类别对数量")
    return parser.parse_args()


def safe_label(name: str) -> str:
    """
    作用:
        将下划线物种名转换为更适合图表显示的标题格式。
    输入:
        name: 原始物种名。
    输出:
        格式化后的显示名称。
    """
    return name.replace("_", " ").title()


def plot_species_count_bar(counts: pd.Series, output_path: Path) -> None:
    """
    作用:
        绘制每个物种样本数量柱状图。
    输入:
        counts: species -> count 的 Series。
        output_path: 输出图片路径。
    输出:
        无返回值；保存图片文件。
    """
    plot_df = counts.reset_index()
    plot_df.columns = ["species", "count"]
    plot_df["species"] = plot_df["species"].map(safe_label)

    height = max(7.0, 0.32 * len(plot_df))
    plt.figure(figsize=(12, height))
    sns.barplot(data=plot_df, y="species", x="count", hue="species", palette="viridis", legend=False)
    plt.xlabel("Image Count")
    plt.ylabel("Species")
    plt.title("Species Sample Count Distribution")
    plt.xscale("log")
    plt.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_long_tail_distribution(counts: pd.Series, output_path: Path) -> None:
    """
    作用:
        绘制按样本数排序后的长尾分布曲线。
    输入:
        counts: species -> count 的 Series。
        output_path: 输出图片路径。
    输出:
        无返回值；保存图片文件。
    """
    sorted_counts = counts.sort_values(ascending=False).to_numpy()
    ranks = np.arange(1, len(sorted_counts) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(ranks, sorted_counts, marker="o", linewidth=2)
    plt.fill_between(ranks, sorted_counts, alpha=0.18)
    plt.yscale("log")
    plt.xlabel("Species Rank By Sample Count")
    plt.ylabel("Image Count (log scale)")
    plt.title("Long-Tail Distribution Of Whale Species")
    plt.grid(True, linestyle="--", alpha=0.35)

    head_count = int(sorted_counts[0])
    tail_count = int(sorted_counts[-1])
    imbalance = head_count / max(tail_count, 1)
    plt.annotate(
        f"Max/Min = {imbalance:.1f}x",
        xy=(len(sorted_counts), tail_count),
        xytext=(max(1, len(sorted_counts) * 0.55), max(tail_count * 3, 2)),
        arrowprops={"arrowstyle": "->", "lw": 1.2},
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def build_split_stats(df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    作用:
        统计每个物种在全量、训练集、验证集中的样本数和 group 数。
    输入:
        df: 全量 DataFrame。
        train_df: 训练集 DataFrame。
        val_df: 验证集 DataFrame。
        group_col: 分组列名，通常为 individual_id。
    输出:
        包含每类样本/个体划分统计的 DataFrame。
    """
    total_counts = df["species"].value_counts()
    train_counts = train_df["species"].value_counts()
    val_counts = val_df["species"].value_counts()

    rows = []
    for species, total_count in total_counts.items():
        train_count = int(train_counts.get(species, 0))
        val_count = int(val_counts.get(species, 0))
        total_groups = df.loc[df["species"] == species, group_col].nunique() if group_col in df.columns else 0
        train_groups = train_df.loc[train_df["species"] == species, group_col].nunique() if group_col in train_df.columns else 0
        val_groups = val_df.loc[val_df["species"] == species, group_col].nunique() if group_col in val_df.columns else 0
        rows.append(
            {
                "species": species,
                "total_count": int(total_count),
                "train_count": train_count,
                "val_count": val_count,
                "total_groups": int(total_groups),
                "train_groups": int(train_groups),
                "val_groups": int(val_groups),
                "train_ratio": train_count / max(int(total_count), 1),
                "val_ratio": val_count / max(int(total_count), 1),
                "is_tail_20_percent": False,
            }
        )

    stats = pd.DataFrame(rows).sort_values("total_count", ascending=False).reset_index(drop=True)
    tail_count = max(1, int(np.ceil(len(stats) * 0.2)))
    stats.loc[stats.tail(tail_count).index, "is_tail_20_percent"] = True
    return stats


def save_split_leakage_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    group_col: str,
    split_strategy: str,
    output_dir: Path,
) -> Dict[str, object]:
    """
    作用:
        生成并保存训练/验证 group 泄漏检查报告。
    输入:
        train_df: 训练集 DataFrame。
        val_df: 验证集 DataFrame。
        group_col: 分组列名。
        split_strategy: 当前划分策略名称。
        output_dir: 输出目录。
    输出:
        泄漏检查报告字典。
    """
    overlap_count = count_group_overlap(train_df, val_df, group_col)
    if overlap_count > 0:
        overlap_groups = sorted(
            set(train_df[group_col].astype(str)).intersection(set(val_df[group_col].astype(str)))
        )
    else:
        overlap_groups = []
    report = {
        "split_strategy": split_strategy,
        "group_col": group_col,
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "train_groups": int(train_df[group_col].nunique()) if group_col in train_df.columns else None,
        "val_groups": int(val_df[group_col].nunique()) if group_col in val_df.columns else None,
        "overlap_group_count": int(overlap_count),
        "overlap_group_examples": overlap_groups[:20],
        "leakage_free": overlap_count == 0,
    }
    with open(output_dir / "split_leakage_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def save_dataset_summary(df: pd.DataFrame, output_dir: Path) -> Dict[str, object]:
    """
    作用:
        保存数据集总体摘要和完整类别计数表。
    输入:
        df: 已清洗的全量 DataFrame。
        output_dir: 输出目录。
    输出:
        数据集摘要字典。
    """
    counts = df["species"].value_counts()
    summary = {
        "num_images": int(len(df)),
        "num_species": int(counts.size),
        "max_class": str(counts.index[0]),
        "max_class_count": int(counts.iloc[0]),
        "min_class": str(counts.index[-1]),
        "min_class_count": int(counts.iloc[-1]),
        "imbalance_ratio_max_to_min": float(counts.iloc[0] / max(int(counts.iloc[-1]), 1)),
        "median_class_count": float(counts.median()),
        "mean_class_count": float(counts.mean()),
        "top10": {str(k): int(v) for k, v in counts.head(10).items()},
        "tail10": {str(k): int(v) for k, v in counts.tail(10).items()},
    }

    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    counts.rename_axis("species").reset_index(name="count").to_csv(
        output_dir / "species_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return summary


def load_class_map(path: Path, df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    作用:
        加载 class_to_idx.json；缺失时根据 DataFrame 重建类别映射。
    输入:
        path: 类别映射文件路径。
        df: 评估 DataFrame。
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


def build_model(
    checkpoint: Dict[str, object],
    model_type: str,
    image_size_override: int | None,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    """
    作用:
        根据 checkpoint 配置构造模型并加载权重，用于分析脚本中的验证集评估。
    输入:
        checkpoint: 已加载 checkpoint。
        model_type: 模型类型，支持 auto/transformer/baseline。
        image_size_override: 可选输入尺寸覆盖值。
        num_classes: 类别数量。
        device: 运行设备。
    输出:
        eval 模式的 PyTorch 模型。
    """
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    if model_type == "auto":
        model_type = str(cfg.get("model_type", "transformer"))

    image_size = image_size_override or int(cfg.get("image_size", 512))
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


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[List[int], List[int]]:
    """
    作用:
        遍历 DataLoader，收集模型预测和真实标签。
    输入:
        model: 待评估模型。
        loader: 评估 DataLoader。
        device: CPU 或 CUDA。
    输出:
        (真实标签列表, 预测标签列表)。
    """
    labels_all: List[int] = []
    preds_all: List[int] = []
    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().tolist()
        preds_all.extend(preds)
        labels_all.extend(labels.cpu().tolist())
    return labels_all, preds_all


def compute_confusion_matrix(labels: List[int], preds: List[int], num_classes: int) -> np.ndarray:
    """
    作用:
        根据标签和预测结果计算混淆矩阵。
    输入:
        labels: 真实类别编号列表。
        preds: 预测类别编号列表。
        num_classes: 类别总数。
    输出:
        混淆矩阵数组，形状 [num_classes, num_classes]。
    """
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, pred in zip(labels, preds):
        matrix[int(label), int(pred)] += 1
    return matrix


def compute_per_class_metrics(
    labels: List[int],
    preds: List[int],
    idx_to_class: Dict[int, str],
    output_dir: Path,
    top_k_confusions: int,
) -> None:
    """
    作用:
        计算每个类别的 Precision/Recall/F1，并统计最严重的混淆类别对。
    输入:
        labels: 真实类别编号列表。
        preds: 预测类别编号列表。
        idx_to_class: 类别编号到物种名映射。
        output_dir: 输出目录。
        top_k_confusions: 保存混淆类别对数量。
    输出:
        无返回值；保存 per_class_metrics.csv、per_class_f1_bar.png、confusion_top_pairs.csv/png。
    """
    num_classes = len(idx_to_class)
    matrix = compute_confusion_matrix(labels, preds, num_classes)

    rows = []
    for idx in range(num_classes):
        tp = int(matrix[idx, idx])
        fp = int(matrix[:, idx].sum() - tp)
        fn = int(matrix[idx, :].sum() - tp)
        support = int(matrix[idx, :].sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append(
            {
                "species": idx_to_class[idx],
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values("support", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(output_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
    plot_per_class_f1(metrics_df, output_dir / "per_class_f1_bar.png")

    confusion_rows = []
    supports = matrix.sum(axis=1)
    for true_idx in range(num_classes):
        for pred_idx in range(num_classes):
            if true_idx == pred_idx:
                continue
            count = int(matrix[true_idx, pred_idx])
            if count <= 0:
                continue
            confusion_rows.append(
                {
                    "true_species": idx_to_class[true_idx],
                    "pred_species": idx_to_class[pred_idx],
                    "count": count,
                    "true_support": int(supports[true_idx]),
                    "error_rate_in_true_class": count / max(int(supports[true_idx]), 1),
                }
            )

    confusion_df = pd.DataFrame(confusion_rows)
    if not confusion_df.empty:
        confusion_df = confusion_df.sort_values(["count", "error_rate_in_true_class"], ascending=False).head(top_k_confusions)
    else:
        confusion_df = pd.DataFrame(columns=["true_species", "pred_species", "count", "true_support", "error_rate_in_true_class"])
    confusion_df.to_csv(output_dir / "confusion_top_pairs.csv", index=False, encoding="utf-8-sig")
    plot_confusion_pairs(confusion_df, output_dir / "confusion_top_pairs.png")


def plot_per_class_f1(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """
    作用:
        绘制各类别 F1 柱状图。
    输入:
        metrics_df: 包含 species 和 f1 的 DataFrame。
        output_path: 输出图片路径。
    输出:
        无返回值；保存图片文件。
    """
    plot_df = metrics_df.sort_values("f1", ascending=True).copy()
    plot_df["species"] = plot_df["species"].map(safe_label)

    height = max(7.0, 0.32 * len(plot_df))
    plt.figure(figsize=(12, height))
    sns.barplot(data=plot_df, y="species", x="f1", hue="species", palette="mako", legend=False)
    plt.xlim(0, 1)
    plt.xlabel("F1 Score")
    plt.ylabel("Species")
    plt.title("Per-Class F1 Score")
    plt.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_confusion_pairs(confusion_df: pd.DataFrame, output_path: Path) -> None:
    """
    作用:
        绘制最易混淆类别对柱状图。
    输入:
        confusion_df: 包含 true_species、pred_species、count 的 DataFrame。
        output_path: 输出图片路径。
    输出:
        无返回值；保存图片文件。
    """
    plt.figure(figsize=(12, 7))
    if confusion_df.empty:
        plt.text(0.5, 0.5, "No off-diagonal confusion pairs", ha="center", va="center")
        plt.axis("off")
    else:
        plot_df = confusion_df.copy()
        plot_df["pair"] = plot_df.apply(
            lambda row: f"{safe_label(row['true_species'])} -> {safe_label(row['pred_species'])}",
            axis=1,
        )
        plot_df = plot_df.sort_values("count", ascending=True)
        sns.barplot(data=plot_df, y="pair", x="count", hue="pair", palette="flare", legend=False)
        plt.xlabel("Misclassified Count")
        plt.ylabel("True Species -> Predicted Species")
        plt.title("Top Confused Species Pairs")
        plt.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def evaluate_checkpoint(args: argparse.Namespace, df: pd.DataFrame, eval_df: pd.DataFrame, output_dir: Path) -> None:
    """
    作用:
        加载 checkpoint 并在验证/全量数据上计算类别级指标和混淆类别对。
    输入:
        args: 命令行参数。
        df: 全量 DataFrame，用于类别映射 fallback。
        eval_df: 实际评估 DataFrame。
        output_dir: 输出目录。
    输出:
        无返回值；保存模型评估分析文件。
    """
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    class_map_path = PROJECT_ROOT / args.class_map
    image_dir = PROJECT_ROOT / args.image_dir
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"找不到图片目录: {image_dir}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    image_size = args.image_size or int(cfg.get("image_size", 512))
    class_to_idx, idx_to_class = load_class_map(class_map_path, df)

    if args.max_samples > 0:
        eval_df = eval_df.head(args.max_samples).reset_index(drop=True)

    _, val_tfms = build_transforms(image_size=image_size, cutout_p=0.0)
    dataset = WhaleSpeciesDataset(eval_df, image_dir, class_to_idx, transforms=val_tfms)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint, args.model_type, args.image_size, len(class_to_idx), device)
    labels, preds = collect_predictions(model, loader, device)
    compute_per_class_metrics(labels, preds, idx_to_class, output_dir, args.top_k_confusions)
    print(f"模型类别指标已保存: {output_dir / 'per_class_metrics.csv'}")
    print(f"最易混淆类别对已保存: {output_dir / 'confusion_top_pairs.csv'}")


def main() -> None:
    """
    作用:
        数据分析脚本主入口，生成长尾分布图、划分统计、泄漏报告，并可选评估 checkpoint。
    输入:
        无显式输入；通过命令行参数读取 CSV、输出目录和 checkpoint。
    输出:
        无返回值；在输出目录生成图表和 CSV/JSON 分析文件。
    """
    args = parse_args()
    csv_path = PROJECT_ROOT / args.csv
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    df = normalize_species_column(df, fix_typos=args.fix_species_typos)
    train_df, val_df = split_train_val(
        df,
        label_col="species",
        val_ratio=args.val_ratio,
        seed=args.seed,
        split_strategy=args.split_strategy,
        group_col=args.group_col,
    )
    eval_df = df if args.use_full_csv else val_df

    counts = df["species"].value_counts()
    summary = save_dataset_summary(df, output_dir)
    split_stats = build_split_stats(df, train_df, val_df, args.group_col)
    split_stats.to_csv(output_dir / "train_val_split_stats.csv", index=False, encoding="utf-8-sig")
    leakage_report = save_split_leakage_report(train_df, val_df, args.group_col, args.split_strategy, output_dir)
    plot_species_count_bar(counts, output_dir / "species_count_bar.png")
    plot_long_tail_distribution(counts, output_dir / "long_tail_distribution.png")

    print("数据集长尾分析完成:")
    print(f"  样本数: {summary['num_images']}")
    print(f"  类别数: {summary['num_species']}")
    print(f"  最大/最小类别样本比: {summary['imbalance_ratio_max_to_min']:.1f}x")
    print(f"  划分策略: {args.split_strategy}")
    print(f"  {args.group_col} 训练/验证重叠数: {leakage_report['overlap_group_count']}")
    print(f"  输出目录: {output_dir.resolve()}")

    if args.checkpoint:
        evaluate_checkpoint(args, df, eval_df, output_dir)
    else:
        print("未传入 --checkpoint，已跳过 per-class F1 与混淆类别对评估。")


if __name__ == "__main__":
    main()
