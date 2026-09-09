"""Select a model family by mean validation Macro F1, never by test score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按多随机种子验证 Macro F1 选择最佳实验")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    output_path = Path(args.output)
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"结果表为空: {csv_path}")
    required = {"experiment_name", "best_val_macro_f1", "best_model_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"结果表缺少列: {sorted(missing)}")
    if "seed" not in df.columns:
        df["seed"] = 0

    means = df.groupby("experiment_name")["best_val_macro_f1"].mean().sort_values(ascending=False)
    best_name = str(means.index[0])
    candidates = df[df["experiment_name"] == best_name].copy()
    mean_f1 = float(means.iloc[0])
    candidates["distance_to_mean"] = (candidates["best_val_macro_f1"] - mean_f1).abs()
    representative = candidates.sort_values(["distance_to_mean", "seed"], ascending=[True, True]).iloc[0]

    checkpoint = Path(str(representative["best_model_path"]))
    run_dir = checkpoint.parent
    payload = {
        "selection_metric": "mean_validation_macro_f1",
        "test_set_used_for_selection": False,
        "best_experiment": best_name,
        "num_seeds": int(len(candidates)),
        "mean_val_macro_f1": mean_f1,
        "std_val_macro_f1": float(candidates["best_val_macro_f1"].std(ddof=0)),
        "representative_seed": int(representative.get("seed", 0)),
        "representative_val_macro_f1": float(representative["best_val_macro_f1"]),
        "model_type": str(representative.get("model_type", "unknown")),
        "checkpoint": str(checkpoint),
        "representative_checkpoint": str(checkpoint),
        "class_map": str(run_dir / "class_to_idx.json"),
        "metrics": str(run_dir / "metrics.json"),
        "train_split": str(run_dir / "splits" / "train.csv"),
        "val_split": str(run_dir / "splits" / "val.csv"),
        "test_split": str(run_dir / "splits" / "test.csv"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
