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
        "| 实验 | 模型 | runs | Focal | Mixup | Cutout | EMA | Val Acc (mean±std) | Val Macro F1 (mean±std) |",
        "|---|---|---:|---|---|---|---|---:|---:|",
    ]
    for _, row in df.iterrows():
        model_name = str(row.get("model_type", ""))
        if model_name == "metric":
            model_name += f"+{row.get('pooling', '')}+{row.get('metric_head', '')}"
        if bool(row.get("multiscale", False)):
            model_name += "+multiscale"
        if int(row.get("token_pool_size", 0) or 0) > 0 and bool(row.get("transformer", False)):
            model_name += f"+tp{int(row['token_pool_size'])}"
        lines.append(
            "| {experiment} | {model} | {runs} | {focal} | {mixup} | {cutout} | {ema} | {acc:.4f}±{acc_std:.4f} | {f1:.4f}±{f1_std:.4f} |".format(
                experiment=row.get("experiment_name", ""),
                model=model_name,
                runs=int(row.get("runs", 1)),
                focal=format_bool(row.get("focal", False)),
                mixup=format_bool(row.get("mixup", False)),
                cutout=format_bool(row.get("cutout", False)),
                ema=format_bool(row.get("ema", False)),
                acc=float(row.get("val_acc_mean", row.get("best_val_acc", 0.0))),
                acc_std=float(row.get("val_acc_std", 0.0)),
                f1=float(row.get("val_macro_f1_mean", row.get("best_val_macro_f1", 0.0))),
                f1_std=float(row.get("val_macro_f1_std", 0.0)),
            )
        )
    return lines


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated seeds without silently treating them as separate methods."""
    config_columns = [
        column
        for column in (
            "experiment_name", "model_type", "focal", "mixup", "cutout", "transformer",
            "cls_token", "ema", "multiscale", "token_pool_size", "pooling", "metric_head",
            "crop_scale_min",
        )
        if column in df.columns
    ]
    grouped = df.groupby("experiment_name", sort=False, dropna=False)
    rows = []
    for _, group in grouped:
        row = {column: group.iloc[0][column] for column in config_columns}
        row.update(
            {
                "runs": len(group),
                "val_acc_mean": float(group["best_val_acc"].mean()),
                "val_acc_std": float(group["best_val_acc"].std(ddof=0)),
                "val_macro_f1_mean": float(group["best_val_macro_f1"].mean()),
                "val_macro_f1_std": float(group["best_val_macro_f1"].std(ddof=0)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("val_macro_f1_mean", ascending=False).reset_index(drop=True)


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
    plot_df = df.copy().sort_values("val_macro_f1_mean", ascending=True)
    labels = plot_df["experiment_name"].astype(str).tolist()
    values = plot_df["val_macro_f1_mean"].astype(float).tolist()
    errors = plot_df["val_macro_f1_std"].astype(float).tolist()

    height = max(4.8, 0.46 * len(plot_df) + 2.0)
    plt.figure(figsize=(12, height))
    plt.barh(labels, values, xerr=errors, color="#2dd4bf", ecolor="#334155", capsize=3)
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
    best_idx = df["val_macro_f1_mean"].astype(float).idxmax()
    best = df.loc[best_idx]
    lines = [
        "# 消融实验报告",
        "",
        f"- 数据来源: `{csv_path.relative_to(PROJECT_ROOT)}`",
        f"- 实验配置数: {len(df)}",
        f"- 总训练次数: {int(df['runs'].sum())}",
        f"- 最佳实验: `{best.get('experiment_name', '')}`",
        f"- 最佳验证 Accuracy (mean±std): {float(best.get('val_acc_mean', 0.0)):.4f}±{float(best.get('val_acc_std', 0.0)):.4f}",
        f"- 最佳验证 Macro F1 (mean±std): {float(best.get('val_macro_f1_mean', 0.0)):.4f}±{float(best.get('val_macro_f1_std', 0.0)):.4f}",
        "- 选择规则: 仅按验证集多随机种子平均 Macro F1 排名；独立测试集不参与模型选择",
        f"- Macro F1 图: `{plot_path.relative_to(PROJECT_ROOT)}`",
        "",
        "## 结果表",
        "",
    ]
    lines.extend(build_report_table(df))
    controlled_suite = any("metric_" in str(name) for name in df["experiment_name"])
    if controlled_suite:
        suggestions = [
            "- `01 -> 02` 检验随机裁剪能否减轻背景/填充区域依赖。",
            "- `03 -> 04` 只改变 GAP/GeM，检验池化策略。",
            "- `04 -> 05` 只改变 Linear/Sub-center ArcFace，检验度量学习分类头。",
            "- `02 -> 06` 对比 ResNet50 与 Transformer，全局关系建模是否带来稳定收益。",
            "- `07` 保留历史最佳配方，作为新受控实验体系与旧结论的连接点。",
        ]
    else:
        suggestions = [
            "- 对比 ResNet50 CE 与 Focal，说明 Focal Loss 在当前长尾设定下是否有效。",
            "- 带 Mixup/Cutout/EMA 的旧实验同时改变多个因素，只能描述相关性，不能做单因素归因。",
            "- 对比 Transformer mean/CLS，说明聚合方式的观测差异。",
        ]
    lines.extend(["", "## 结论填写建议", "", *suggestions])
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
    raw_df = pd.read_csv(csv_path)
    if raw_df.empty:
        raise ValueError(f"CSV 为空，无法生成报告: {csv_path}")
    df = aggregate_results(raw_df)
    aggregate_csv = output_path.with_name(output_path.stem + "_aggregated.csv")
    aggregate_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(aggregate_csv, index=False, encoding="utf-8-sig")
    plot_macro_f1(df, plot_path)
    write_report(df, csv_path, output_path, plot_path)
    print(f"消融报告已保存: {output_path.resolve()}")
    print(f"多种子聚合表已保存: {aggregate_csv.resolve()}")
    print(f"Macro F1 图已保存: {plot_path.resolve()}")


if __name__ == "__main__":
    main()
