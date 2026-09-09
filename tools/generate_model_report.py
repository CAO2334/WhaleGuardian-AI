"""Generate a complete report on a frozen validation or independent test CSV."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import WhaleSpeciesDataset, build_transforms, normalize_species_column
from models.factory import load_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在固定 CSV 上生成完整模型评估报告")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--class-map", required=True)
    parser.add_argument("--eval-csv", required=True, help="固定验证/测试划分 CSV，不会再次随机划分")
    parser.add_argument("--train-csv", default=None, help="训练划分 CSV，用于定义头部/尾部类别")
    parser.add_argument("--image-dir", default="archive/train_images")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", default="test")
    parser.add_argument(
        "--independent-test",
        action="store_true",
        help="确认该 CSV 是模型选择完成后才使用的冻结测试集",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--bootstrap-iters", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--group-col", default="individual_id")
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def expected_calibration_error(labels: np.ndarray, preds: np.ndarray, confidence: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if not np.any(mask):
            continue
        bin_accuracy = np.mean(preds[mask] == labels[mask])
        value += float(mask.mean() * abs(bin_accuracy - confidence[mask].mean()))
    return value


def observed_balanced_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    observed_labels = sorted(np.unique(labels).tolist())
    recalls = precision_recall_fscore_support(
        labels, preds, labels=observed_labels, zero_division=0
    )[1]
    return float(np.mean(recalls))


def bootstrap_intervals(
    frame: pd.DataFrame,
    labels: np.ndarray,
    preds: np.ndarray,
    group_col: str,
    iterations: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    if iterations <= 0:
        return {}
    rng = np.random.default_rng(seed)
    if group_col in frame.columns:
        groups = frame[group_col].astype(str).to_numpy()
        unique_groups = np.unique(groups)
        indices_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}

        def sample_indices() -> np.ndarray:
            sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            return np.concatenate([indices_by_group[group] for group in sampled_groups])
    else:
        all_indices = np.arange(len(frame))

        def sample_indices() -> np.ndarray:
            return rng.choice(all_indices, size=len(all_indices), replace=True)

    samples = {"accuracy": [], "macro_f1": [], "balanced_accuracy": []}
    for _ in range(iterations):
        selected = sample_indices()
        y_true = labels[selected]
        y_pred = preds[selected]
        samples["accuracy"].append(accuracy_score(y_true, y_pred))
        bootstrap_labels = sorted(np.unique(y_true).tolist())
        samples["macro_f1"].append(
            f1_score(y_true, y_pred, labels=bootstrap_labels, average="macro", zero_division=0)
        )
        samples["balanced_accuracy"].append(observed_balanced_accuracy(y_true, y_pred))

    result: Dict[str, Dict[str, float]] = {}
    for metric, values in samples.items():
        low, high = np.percentile(values, [2.5, 97.5])
        result[metric] = {"low": float(low), "high": float(high)}
    return result


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    labels_all: List[np.ndarray] = []
    probabilities_all: List[np.ndarray] = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for images, labels in tqdm(loader, desc="Final evaluation", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probabilities_all.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels_all.append(labels.numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    probabilities = np.concatenate(probabilities_all)
    labels = np.concatenate(labels_all)
    preds = probabilities.argmax(axis=1)
    return labels, preds, probabilities, elapsed


def save_confusion_plot(matrix: np.ndarray, names: List[str], output_path: Path) -> None:
    denom = np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    normalized = matrix / denom
    size = max(11, min(24, len(names) * 0.48))
    plt.figure(figsize=(size, size))
    sns.heatmap(
        normalized,
        cmap="Blues",
        xticklabels=[name.replace("_", " ") for name in names],
        yticklabels=[name.replace("_", " ") for name in names],
        vmin=0,
        vmax=1,
        annot=len(names) <= 35,
        fmt=".2f",
        square=True,
    )
    plt.xlabel("Predicted species")
    plt.ylabel("True species")
    plt.title("Row-normalized confusion matrix")
    plt.xticks(rotation=60, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_confidence_plots(labels: np.ndarray, preds: np.ndarray, confidence: np.ndarray, output_dir: Path) -> None:
    correct = preds == labels
    plt.figure(figsize=(9, 5.5))
    plt.hist(confidence[correct], bins=20, alpha=0.72, label="Correct", color="#14b8a6")
    if np.any(~correct):
        plt.hist(confidence[~correct], bins=20, alpha=0.72, label="Incorrect", color="#f97316")
    plt.xlabel("Top-1 confidence")
    plt.ylabel("Samples")
    plt.title("Confidence distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_histogram.png", dpi=220)
    plt.close()

    edges = np.linspace(0.0, 1.0, 11)
    xs, ys, sizes = [], [], []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if np.any(mask):
            xs.append(float(confidence[mask].mean()))
            ys.append(float(correct[mask].mean()))
            sizes.append(int(mask.sum()))
    plt.figure(figsize=(6.5, 6.5))
    plt.plot([0, 1], [0, 1], "--", color="#64748b", label="Perfect calibration")
    plt.plot(xs, ys, marker="o", color="#0f766e", label="Model")
    for x, y, size in zip(xs, ys, sizes):
        plt.annotate(str(size), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    plt.xlabel("Mean confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Reliability diagram (labels show bin size)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "reliability_diagram.png", dpi=220)
    plt.close()


def write_markdown(
    output_path: Path,
    metrics: Dict[str, object],
    split_name: str,
    tail_species: List[str],
    independent_test: bool,
) -> None:
    ci = metrics.get("bootstrap_95_ci", {})

    def ci_text(name: str) -> str:
        values = ci.get(name, {}) if isinstance(ci, dict) else {}
        if not values:
            return "N/A"
        return f"[{values['low']:.4f}, {values['high']:.4f}]"

    def metric_text(name: str) -> str:
        value = metrics.get(name)
        return "N/A（该集合无对应类别）" if value is None else f"{float(value):.4f}"

    status_note = (
        "> 本报告只读取固定 CSV；该独立测试集不参与模型选择或调参。"
        if independent_test
        else "> 本报告只读取固定 CSV，但未声明它是未参与选模的独立测试集；结果只能按其 split 名称解读。"
    )
    lines = [
        f"# {split_name} 模型评估报告",
        "",
        status_note,
        "",
        "## 总体结果",
        "",
        "| 指标 | 数值 | 95% bootstrap CI |",
        "|---|---:|---:|",
        f"| Accuracy | {metrics['accuracy']:.4f} | {ci_text('accuracy')} |",
        f"| Macro F1（测试集中出现的类别） | {metrics['macro_f1_present_classes']:.4f} | {ci_text('macro_f1')} |",
        f"| Balanced Accuracy | {metrics['balanced_accuracy']:.4f} | {ci_text('balanced_accuracy')} |",
        f"| Top-3 Accuracy | {metrics['top3_accuracy']:.4f} | - |",
        f"| NLL | {metrics['negative_log_likelihood']:.4f} | - |",
        f"| ECE (15 bins) | {metrics['ece_15_bins']:.4f} | - |",
        "",
        "## 长尾结果",
        "",
        f"- 头部/非尾部类别 Macro F1: {metric_text('head_macro_f1')}",
        f"- 尾部 20% 类别 Macro F1: {metric_text('tail_macro_f1')}",
        f"- 尾部类别: {', '.join(tail_species)}",
        "",
        "## 模型与运行",
        "",
        f"- 模型类型: `{metrics['model_type']}`",
        f"- 参数量: {metrics['parameter_count']:,}",
        f"- 评估样本: {metrics['num_samples']}",
        f"- 推理设备: `{metrics['device']}`",
        f"- 端到端评估吞吐量: {metrics['samples_per_second']:.2f} samples/s",
        "",
        "## 附件",
        "",
        "- `metrics.json`: 机器可读指标与置信区间",
        "- `predictions.csv`: 每张图片的真值、Top-1/Top-3 与置信度",
        "- `error_cases.csv`: 按置信度降序排列的误分类样本",
        "- `per_class_metrics.csv`: 每个物种 Precision/Recall/F1/Support",
        "- `confusion_matrix.csv` / `confusion_matrix_normalized.png`",
        "- `confidence_histogram.png` / `reliability_diagram.png`",
        "",
        "## 解读边界",
        "",
        "- 极少数类的 support 很小时，单类 F1 波动很大，必须同时报告 support。",
        "- softmax/ArcFace confidence 不是概率正确性的保证；ECE 与可靠性图用于揭示过度自信。",
        "- 该任务是 species 分类，不等同于 Happywhale 原竞赛的 individual_id 开集检索任务。",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve(args.checkpoint)
    class_map_path = resolve(args.class_map)
    eval_csv_path = resolve(args.eval_csv)
    train_csv_path = resolve(args.train_csv) if args.train_csv else None
    image_dir = resolve(args.image_dir)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (checkpoint_path, class_map_path, eval_csv_path, image_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint.get("config", {})
    image_size = args.image_size or int(cfg.get("image_size", 512))
    class_to_idx = {str(key): int(value) for key, value in json.loads(class_map_path.read_text(encoding="utf-8-sig")).items()}
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    class_names = [idx_to_class[index] for index in range(len(idx_to_class))]

    eval_df = normalize_species_column(pd.read_csv(eval_csv_path), fix_typos=True)
    if args.max_samples > 0:
        eval_df = eval_df.head(args.max_samples).reset_index(drop=True)
    unknown = sorted(set(eval_df["species"]) - set(class_to_idx))
    if unknown:
        raise ValueError(f"评估 CSV 含类别映射外的 species: {unknown}")

    _, eval_transform = build_transforms(image_size=image_size, cutout_p=0.0)
    dataset = WhaleSpeciesDataset(eval_df, image_dir, class_to_idx, transforms=eval_transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = load_model_from_checkpoint(checkpoint, num_classes=len(class_to_idx), device=device)
    labels, preds, probabilities, elapsed = predict(model, loader, device)
    confidence = probabilities.max(axis=1)
    top3 = np.argsort(probabilities, axis=1)[:, -3:][:, ::-1]
    present_labels = sorted(np.unique(labels).tolist())

    precision, recall, per_f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    per_class = pd.DataFrame(
        {
            "class_index": range(len(class_names)),
            "species": class_names,
            "precision": precision,
            "recall": recall,
            "f1": per_f1,
            "support": support.astype(int),
        }
    )

    if train_csv_path and train_csv_path.exists():
        train_df = normalize_species_column(pd.read_csv(train_csv_path), fix_typos=True)
        train_counts = train_df["species"].value_counts()
    else:
        train_counts = eval_df["species"].value_counts()
    tail_count = max(1, int(np.ceil(len(class_names) * 0.20)))
    tail_species = train_counts.reindex(class_names, fill_value=0).sort_values().head(tail_count).index.tolist()
    per_class["train_count"] = per_class["species"].map(train_counts).fillna(0).astype(int)
    per_class["is_tail_20_percent"] = per_class["species"].isin(tail_species)
    observed = per_class["support"] > 0
    tail_observed = observed & per_class["is_tail_20_percent"]
    head_observed = observed & ~per_class["is_tail_20_percent"]

    top3_correct = np.any(top3 == labels[:, None], axis=1)
    intervals = bootstrap_intervals(
        eval_df,
        labels,
        preds,
        args.group_col,
        args.bootstrap_iters,
        args.bootstrap_seed,
    )
    head_macro_f1 = float(per_class.loc[head_observed, "f1"].mean()) if head_observed.any() else None
    tail_macro_f1 = float(per_class.loc[tail_observed, "f1"].mean()) if tail_observed.any() else None
    metrics: Dict[str, object] = {
        "split_name": args.split_name,
        "eval_csv": str(eval_csv_path),
        "checkpoint": str(checkpoint_path),
        "independent_test": args.independent_test,
        "test_set_used_for_selection": False if args.independent_test else None,
        "num_samples": int(len(labels)),
        "num_classes_total": len(class_names),
        "num_classes_present": len(present_labels),
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": observed_balanced_accuracy(labels, preds),
        "macro_f1_present_classes": float(
            f1_score(labels, preds, labels=present_labels, average="macro", zero_division=0)
        ),
        "macro_f1_all_classes": float(
            f1_score(labels, preds, labels=list(range(len(class_names))), average="macro", zero_division=0)
        ),
        "top3_accuracy": float(top3_correct.mean()),
        "negative_log_likelihood": float(log_loss(labels, probabilities, labels=list(range(len(class_names))))),
        "ece_15_bins": expected_calibration_error(labels, preds, confidence, bins=15),
        "head_macro_f1": head_macro_f1,
        "tail_macro_f1": tail_macro_f1,
        "bootstrap_unit": args.group_col if args.group_col in eval_df.columns else "image",
        "bootstrap_iterations": args.bootstrap_iters,
        "bootstrap_95_ci": intervals,
        "model_type": str(cfg.get("model_type", "unknown")),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "checkpoint_size_mb": checkpoint_path.stat().st_size / (1024**2),
        "device": str(device),
        "elapsed_seconds": elapsed,
        "samples_per_second": len(labels) / max(elapsed, 1e-9),
    }

    predictions = eval_df.copy()
    predictions["true_index"] = labels
    predictions["pred_index"] = preds
    predictions["pred_species"] = [idx_to_class[int(index)] for index in preds]
    predictions["confidence"] = confidence
    predictions["correct"] = preds == labels
    for rank in range(3):
        predictions[f"top{rank + 1}_species"] = [idx_to_class[int(index)] for index in top3[:, rank]]
        predictions[f"top{rank + 1}_score"] = probabilities[np.arange(len(probabilities)), top3[:, rank]]

    matrix = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    predictions.loc[~predictions["correct"]].sort_values("confidence", ascending=False).to_csv(
        output_dir / "error_cases.csv", index=False, encoding="utf-8-sig"
    )
    per_class.sort_values(["is_tail_20_percent", "support"], ascending=[False, True]).to_csv(
        output_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        output_dir / "confusion_matrix.csv", encoding="utf-8-sig"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    save_confusion_plot(matrix, class_names, output_dir / "confusion_matrix_normalized.png")
    save_confidence_plots(labels, preds, confidence, output_dir)
    write_markdown(output_dir / "report.md", metrics, args.split_name, tail_species, args.independent_test)

    print(f"完整评估报告: {(output_dir / 'report.md').resolve()}")
    print(f"Accuracy={metrics['accuracy']:.4f}, Macro F1={metrics['macro_f1_present_classes']:.4f}")


if __name__ == "__main__":
    main()
