"""
根据 ablation_results.csv 生成消融实验 Markdown 报告和指标柱状图。

示例:
    python tools/generate_ablation_report.py
    python tools/generate_ablation_report.py --csv outputs/ablations/ablation_results.csv --output outputs/ablations/ablation_report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.report_paths import timestamped_path


def parse_args() -> argparse.Namespace:
    """
    作用:
        解析消融报告生成脚本参数。
    输入:
        命令行参数，如 --csv、--output、--plot。
    输出:
        argparse.Namespace 参数对象。
    """
    parser = argparse.ArgumentParser(description="生成消融实验报告")
    parser.add_argument("--csv", default=None, help="消融结果 CSV；为空时自动查找 outputs/ablations/ablation_results.csv 或 outputs/ablation_results.csv")
    parser.add_argument("--output", default=None, help="Markdown 报告输出路径；为空时写入 outputs/reports/ablation/ 并自动加时间戳")
    parser.add_argument("--plot", default=None, help="Macro F1 柱状图输出路径；为空时写入 outputs/reports/ablation/ 并自动加时间戳")
    return parser.parse_args()


def resolve_csv_path(csv_arg: str | None) -> Path:
    """
    作用:
        自动定位消融结果 CSV。
    输入:
        csv_arg: 用户显式传入的 CSV 路径，或 None。
    输出:
        存在的 CSV 路径。
    """
    candidates = []
    if csv_arg:
        candidates.append(PROJECT_ROOT / csv_arg)
    candidates.extend(
        [
            PROJECT_ROOT / "outputs" / "ablations" / "ablation_results.csv",
            PROJECT_ROOT / "outputs" / "ablation_results.csv",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("找不到 ablation_results.csv，请先完成至少一次训练或消融实验。")


def format_bool(value: object) -> str:
    """
    作用:
        将布尔值转换为报告表格中的中文“是/否”。
    输入:
        value: 布尔值或可转为布尔含义的对象。
    输出:
        '是' 或 '否'。
    """
    if isinstance(value, str):
        return "是" if value.lower() == "true" else "否"
    return "是" if bool(value) else "否"


def build_report_table(df: pd.DataFrame) -> List[str]:
    """
    作用:
        将实验结果 DataFrame 转换为 Markdown 表格行。
    输入:
        df: ablation_results.csv 读取后的 DataFrame。
    输出:
        Markdown 文本行列表。
    """
    lines = [
        "| 实验 | 模型 | Focal | Mixup | Cutout | Transformer | CLS | EMA | Acc | Macro F1 | 最佳 Epoch |",
        "|---|---|---|---|---|---|---|---|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        model_name = str(row.get("model_type", ""))
        if bool(row.get("multiscale", False)):
            model_name += "+multiscale"
        if int(row.get("token_pool_size", 0) or 0) > 0 and bool(row.get("transformer", False)):
            model_name += f"+tp{int(row['token_pool_size'])}"
        lines.append(
            "| {experiment} | {model} | {focal} | {mixup} | {cutout} | {transformer} | {cls} | {ema} | {acc:.4f} | {f1:.4f} | {epoch} |".format(
                experiment=row.get("experiment_name", ""),
                model=model_name,
                focal=format_bool(row.get("focal", False)),
                mixup=format_bool(row.get("mixup", False)),
                cutout=format_bool(row.get("cutout", False)),
                transformer=format_bool(row.get("transformer", False)),
                cls=format_bool(row.get("cls_token", False)),
                ema=format_bool(row.get("ema", False)),
                acc=float(row.get("best_val_acc", 0.0)),
                f1=float(row.get("best_val_macro_f1", 0.0)),
                epoch=int(row.get("best_epoch", 0)),
            )
        )
    return lines


def plot_macro_f1(df: pd.DataFrame, output_path: Path) -> None:
    """
    作用:
        绘制各实验 Macro F1 柱状图。
    输入:
        df: 消融结果 DataFrame。
        output_path: 图片输出路径。
    输出:
        无返回值；保存 PNG 图片。
    """
    plot_df = df.copy()
    plot_df = plot_df.sort_values("best_val_macro_f1", ascending=True)
    labels = plot_df["experiment_name"].astype(str).tolist()
    values = plot_df["best_val_macro_f1"].astype(float).tolist()

    height = max(4.8, 0.46 * len(plot_df) + 2.0)
    plt.figure(figsize=(12, height))
    plt.barh(labels, values, color="#2dd4bf")
    plt.xlim(0, 1)
    plt.xlabel("Validation Macro F1")
    plt.ylabel("Experiment")
    plt.title("Ablation Study - Macro F1")
    plt.grid(axis="x", linestyle="--", alpha=0.35)
    for idx, value in enumerate(values):
        plt.text(min(value + 0.01, 0.98), idx, f"{value:.4f}", va="center")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def write_report(df: pd.DataFrame, csv_path: Path, output_path: Path, plot_path: Path) -> None:
    """
    作用:
        写入 Markdown 消融实验报告。
    输入:
        df: 消融结果 DataFrame。
        csv_path: 原始 CSV 路径。
        output_path: Markdown 输出路径。
        plot_path: Macro F1 图路径。
    输出:
        无返回值；保存 Markdown 文件。
    """
    best_idx = df["best_val_macro_f1"].astype(float).idxmax()
    best = df.loc[best_idx]
    lines = [
        "# 消融实验报告",
        "",
        f"- 数据来源: `{csv_path.relative_to(PROJECT_ROOT)}`",
        f"- 实验数量: {len(df)}",
        f"- 最佳实验: `{best.get('experiment_name', '')}`",
        f"- 最佳验证 Accuracy: {float(best.get('best_val_acc', 0.0)):.4f}",
        f"- 最佳验证 Macro F1: {float(best.get('best_val_macro_f1', 0.0)):.4f}",
        f"- Macro F1 图: `{plot_path.relative_to(PROJECT_ROOT)}`",
        "",
        "## 结果表",
        "",
    ]
    lines.extend(build_report_table(df))
    lines.extend(
        [
            "",
            "## 结论填写建议",
            "",
            "- 对比 `ResNet50 CE` 与 `ResNet50 Focal`，说明 Focal Loss 对长尾类别是否有效。",
            "- 对比带/不带 Mixup、Cutout、EMA 的结果，说明训练策略对泛化的贡献。",
            "- 对比 `Transformer mean` 与 `Transformer cls`，说明 CLS Token 是否优于平均池化。",
            "- 对比 baseline 与 Transformer，说明全局空间关系建模是否提升 Macro F1。",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """
    作用:
        消融报告脚本主入口，读取 CSV、绘图并生成 Markdown 报告。
    输入:
        无显式输入；通过命令行参数读取路径。
    输出:
        无返回值；生成 ablation_report.md 和 ablation_macro_f1.png。
    """
    args = parse_args()
    csv_path = resolve_csv_path(args.csv)
    report_dir = PROJECT_ROOT / "outputs" / "reports" / "ablation"
    output_path = PROJECT_ROOT / args.output if args.output else timestamped_path(report_dir, "ablation_report", ".md")
    plot_path = PROJECT_ROOT / args.plot if args.plot else output_path.with_name(output_path.stem.replace("ablation_report", "ablation_macro_f1") + ".png")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV 为空，无法生成报告: {csv_path}")
    plot_macro_f1(df, plot_path)
    write_report(df, csv_path, output_path, plot_path)
    print(f"消融报告已保存: {output_path.resolve()}")
    print(f"Macro F1 图已保存: {plot_path.resolve()}")


if __name__ == "__main__":
    main()
