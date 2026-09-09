"""将已完成消融实验的四个 checkpoint 导出为 Flask 可加载的集成 artifact。

示例::

    python tools/export_ensemble_artifacts.py \
      --run-root outputs/autodl_research_20260909-023128

导出的目录包含四个独立 ONNX 模型和 ``ensemble_manifest.json``。权重固定为
此前只在验证集搜索得到的 0.1/0.1/0.1/0.7，不读取测试标签。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPONENTS = [
    ("02_resnet50_ce_crop", 0.1),
    ("04_metric_linear_gem_crop", 0.1),
    ("06_transformer_mean_ce_crop", 0.1),
    ("07_previous_best_transformer_recipe", 0.7),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出四模型 Flask 概率集成 artifact")
    parser.add_argument("--run-root", required=True, help="完整消融运行目录")
    parser.add_argument("--output-dir", default=None, help="集成 artifact 输出目录，默认 <run-root>/ensemble_artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--force", action="store_true", help="覆盖已有组件 ONNX")
    return parser.parse_args()


def load_optional_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_manifest(
    path: Path,
    output_dir: Path,
    version: str,
    components: List[Dict[str, object]],
    run_root: Path,
) -> None:
    selection = load_optional_json(run_root / "reports" / "fast_ensemble" / "selection.json")
    independent_metrics = load_optional_json(
        run_root / "reports" / "fast_ensemble" / "independent_test" / "metrics.json"
    )
    payload = {
        "model_name": "whale_species_probability_ensemble",
        "version": version,
        "experiment_name": "validation_locked_probability_ensemble",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "onnxruntime",
        "selection_metric": "validation_macro_f1",
        "test_set_used_for_selection": False,
        "num_classes": 28,
        "image_size": 512,
        "components": components,
        "weights_sum": sum(float(component["weight"]) for component in components),
        "metrics": {
            "validation": {
                "accuracy": selection.get("val_accuracy"),
                "macro_f1": selection.get("val_macro_f1"),
            },
            "independent_test": independent_metrics,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = PROJECT_ROOT / run_root
    run_root = run_root.resolve()
    output_dir = Path(args.output_dir) if args.output_dir else run_root / "ensemble_artifacts"
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()
    version = args.version or f"ensemble_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    exported: List[Dict[str, object]] = []
    for name, weight in DEFAULT_COMPONENTS:
        seed_dir = run_root / "ablations" / name / f"seed_{args.seed}"
        checkpoint = seed_dir / "best_model.pth"
        class_map = seed_dir / "class_to_idx.json"
        metrics = seed_dir / "metrics.json"
        artifact = output_dir / name
        model_file = artifact / "model.onnx"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"找不到 {name} checkpoint: {checkpoint}")
        if not class_map.is_file():
            raise FileNotFoundError(f"找不到 {name} class_to_idx.json: {class_map}")
        if model_file.exists() and not args.force:
            print(f"[{name}] 已存在，跳过导出（需要覆盖时加 --force）。")
        else:
            command = [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "export_onnx.py"),
                "--checkpoint", str(checkpoint),
                "--class-map", str(class_map),
                "--metrics", str(metrics),
                "--artifact-dir", str(artifact),
                "--version", f"{version}_{name}",
                "--opset", str(args.opset),
            ]
            print("导出:", " ".join(command))
            subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
        if not model_file.is_file():
            raise FileNotFoundError(f"导出后仍找不到: {model_file}")
        exported.append({"name": name, "weight": weight, "artifact_dir": name})

    write_manifest(output_dir / "ensemble_manifest.json", output_dir, version, exported, run_root)
    print(f"集成 artifact 已生成: {output_dir}")
    print(f"Flask 可通过 WHALE_ENSEMBLE_ARTIFACT_DIR={output_dir} 加载。")


if __name__ == "__main__":
    main()
