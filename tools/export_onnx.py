"""
导出 AI护鲸使者 PyTorch checkpoint 为 ONNX。

示例:
    python tools/export_onnx.py --checkpoint outputs/best_model.pth --output whale_model.onnx
    python tools/export_onnx.py --checkpoint outputs/best_model.pth --model-type transformer --opset 17
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.factory import load_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    """
    作用:
        解析 ONNX 导出和 artifact 打包相关命令行参数。
    输入:
        命令行参数，如 --checkpoint、--artifact-dir、--version。
    输出:
        argparse.Namespace 参数对象。
    """
    parser = argparse.ArgumentParser(description="导出 best_model.pth 为 ONNX")
    parser.add_argument("--checkpoint", default="outputs/best_model.pth", help="PyTorch checkpoint 路径")
    parser.add_argument("--output", default=None, help="ONNX 输出路径；不传时普通导出为 whale_model.onnx，artifact 导出为 model.onnx")
    parser.add_argument("--artifact-dir", default=None, help="模型版本目录，例如 artifacts/whale_resnet50_transformer_v1")
    parser.add_argument("--model-name", default=None, help="模型名称；不传时根据 checkpoint 自动生成")
    parser.add_argument("--version", default="v1", help="模型版本号，写入 artifact 元数据")
    parser.add_argument("--metrics", default=None, help="可选 metrics.json 路径，会复制到 artifact 中")
    parser.add_argument("--class-map", default="outputs/class_to_idx.json", help="类别映射路径；artifact 导出时会复制为 class_to_idx.json")
    parser.add_argument("--model-type", choices=("auto", "transformer", "baseline", "metric"), default="auto", help="模型结构类型")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本")
    parser.add_argument("--batch-size", type=int, default=1, help="导出时 dummy input 的 batch size")
    parser.add_argument("--image-size", type=int, default=None, help="覆盖 checkpoint 中保存的 image_size")
    parser.add_argument("--no-check", action="store_true", help="跳过 torch.onnx.export 的模型检查")
    parser.add_argument("--no-runtime-check", action="store_true", help="跳过 ONNX Runtime 与 PyTorch 数值一致性检查")
    return parser.parse_args()


def resolve_export_paths(args: argparse.Namespace) -> Tuple[Path | None, Path]:
    """
    作用:
        根据用户参数确定 artifact 目录和 ONNX 输出路径。
    输入:
        args: 命令行参数对象。
    输出:
        (artifact_dir 或 None, onnx_output_path)。
    """
    artifact_dir = PROJECT_ROOT / args.artifact_dir if args.artifact_dir else None
    if artifact_dir is not None:
        output_path = PROJECT_ROOT / args.output if args.output else artifact_dir / "model.onnx"
    else:
        output_path = PROJECT_ROOT / args.output if args.output else PROJECT_ROOT / "whale_model.onnx"
    return artifact_dir, output_path


def infer_model_name(cfg: Dict[str, Any], model_type: str) -> str:
    """
    作用:
        根据 checkpoint 配置推断部署模型名称。
    输入:
        cfg: checkpoint 中保存的 config 字典。
        model_type: 用户指定或自动识别的模型类型。
    输出:
        模型名称字符串。
    """
    if model_type == "auto":
        model_type = str(cfg.get("model_type", "transformer"))
    if model_type == "baseline":
        return "whale_resnet50_baseline"
    if model_type == "metric":
        return "whale_resnet50_metric"
    return "whale_resnet50_transformer"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """
    作用:
        以 UTF-8 格式写入 JSON 文件。
    输入:
        path: 输出 JSON 路径。
        payload: 要写入的字典。
    输出:
        无返回值；在磁盘写入 JSON 文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def create_metrics_payload(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """
    作用:
        从 checkpoint 中提取基础训练指标，作为 artifact 的 metrics.json fallback。
    输入:
        checkpoint: torch.load 读取到的 checkpoint 字典。
    输出:
        指标字典，包含 epoch、best_val_macro_f1 等字段。
    """
    payload: Dict[str, Any] = {}
    for key in ("epoch", "best_val_macro_f1", "val_acc_at_best_macro_f1"):
        if key in checkpoint:
            value = checkpoint[key]
            if hasattr(value, "item"):
                value = value.item()
            payload[key] = value
    return payload


def validate_onnx_runtime(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
) -> Dict[str, Any]:
    """Compare ONNX Runtime logits with the exported PyTorch model."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("缺少 onnxruntime，无法做导出数值一致性检查。") from exc

    with torch.no_grad():
        torch_output = model(dummy_input).detach().cpu().numpy()
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_output = session.run(None, {"images": dummy_input.cpu().numpy()})[0]
    abs_diff = np.abs(torch_output - ort_output)
    passed = bool(np.allclose(torch_output, ort_output, rtol=1e-3, atol=1e-4))
    payload = {
        "passed": passed,
        "provider": session.get_providers()[0],
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "rtol": 1e-3,
        "atol": 1e-4,
    }
    if not passed:
        raise RuntimeError(f"ONNX Runtime 数值检查失败: {payload}")
    return payload


def package_artifact(
    artifact_dir: Path,
    output_path: Path,
    checkpoint: Dict[str, Any],
    class_map_path: Path,
    metrics_path: Path | None,
    model_name: str,
    version: str,
    image_size: int,
    model_type: str,
    opset: int,
) -> None:
    """
    作用:
        将 ONNX、类别映射、配置、指标和 manifest 打包成标准模型 artifact 目录。
    输入:
        artifact_dir: artifact 输出目录。
        output_path: 已导出的 ONNX 文件路径。
        checkpoint: PyTorch checkpoint 字典。
        class_map_path: 类别映射路径。
        metrics_path: 可选 metrics.json 路径。
        model_name/version: 模型名称和版本号。
        image_size/model_type/opset: 模型部署元数据。
    输出:
        无返回值；写入 artifact 目录中的多个文件。
    """
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    class_to_idx = checkpoint.get("class_to_idx", {}) if isinstance(checkpoint, dict) else {}
    artifact_dir.mkdir(parents=True, exist_ok=True)

    target_class_map = artifact_dir / "class_to_idx.json"
    if class_map_path.exists():
        shutil.copy2(class_map_path, target_class_map)
    elif class_to_idx:
        write_json(target_class_map, {str(k): int(v) for k, v in class_to_idx.items()})
    else:
        print("警告: 未找到类别映射，artifact 中不会生成 class_to_idx.json。")

    config_payload = dict(cfg)
    config_payload.update(
        {
            "model_name": model_name,
            "version": version,
            "model_type": model_type,
            "image_size": image_size,
            "onnx_opset": opset,
            "num_classes": len(class_to_idx) if class_to_idx else None,
        }
    )
    write_json(artifact_dir / "config.json", config_payload)

    target_metrics = artifact_dir / "metrics.json"
    if metrics_path is not None and metrics_path.exists():
        shutil.copy2(metrics_path, target_metrics)
    else:
        write_json(target_metrics, create_metrics_payload(checkpoint))

    manifest = {
        "model_name": model_name,
        "version": version,
        "artifact_name": artifact_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "onnxruntime",
        "model_file": output_path.name,
        "class_map_file": "class_to_idx.json",
        "config_file": "config.json",
        "metrics_file": "metrics.json",
        "onnx_validation_file": "onnx_validation.json",
        "num_classes": len(class_to_idx) if class_to_idx else None,
        "image_size": image_size,
        "model_type": model_type,
        "onnx_opset": opset,
    }
    write_json(artifact_dir / "manifest.json", manifest)


def build_model_from_checkpoint(
    checkpoint: Dict[str, object],
    model_type: str,
    image_size_override: int | None,
) -> torch.nn.Module:
    """
    作用:
        根据 checkpoint 中的配置重建 PyTorch 模型，并加载权重。
    输入:
        checkpoint: PyTorch checkpoint 字典。
        model_type: 'auto'、'transformer' 或 'baseline'。
        image_size_override: 可选输入尺寸覆盖值。
    输出:
        已加载权重并切换到 eval 模式的 PyTorch 模型。
    """
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    class_to_idx = checkpoint.get("class_to_idx", {}) if isinstance(checkpoint, dict) else {}
    if not class_to_idx:
        raise ValueError("checkpoint 中缺少 class_to_idx，无法确定 num_classes。")

    return load_model_from_checkpoint(
        checkpoint,
        num_classes=len(class_to_idx),
        model_type=model_type,
        image_size=image_size_override,
        device="cpu",
    )


def main() -> None:
    """
    作用:
        ONNX 导出脚本主入口，负责读取 checkpoint、重建模型、导出 ONNX 并可选打包 artifact。
    输入:
        无显式输入；通过命令行参数读取路径和导出设置。
    输出:
        无返回值；生成 ONNX 文件和可选 artifact 目录。
    """
    args = parse_args()
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    class_map_path = PROJECT_ROOT / args.class_map
    metrics_path = PROJECT_ROOT / args.metrics if args.metrics else None
    artifact_dir, output_path = resolve_export_paths(args)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")
    if not class_map_path.exists():
        print(f"警告: 未找到类别映射文件 {class_map_path}，ONNX 仍会导出，但部署推理需要 class_to_idx.json。")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    image_size = args.image_size or int(cfg.get("image_size", 512))
    resolved_model_type = args.model_type
    if resolved_model_type == "auto":
        resolved_model_type = str(cfg.get("model_type", "transformer"))
    model_name = args.model_name or infer_model_name(cfg, args.model_type)
    model = build_model_from_checkpoint(checkpoint, args.model_type, args.image_size)

    dummy_input = torch.randn(args.batch_size, 3, image_size, image_size, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        training=torch.onnx.TrainingMode.EVAL,
        verbose=False,
        # Keep deployment compatible with minimal AutoDL environments that do
        # not install the newer onnxscript-based dynamo exporter.
        dynamo=False,
    )

    if not args.no_runtime_check:
        validation = validate_onnx_runtime(model, dummy_input, output_path)
        validation_path = (
            artifact_dir / "onnx_validation.json"
            if artifact_dir is not None
            else output_path.with_name(output_path.stem + "_validation.json")
        )
        write_json(validation_path, validation)
        print(
            "ONNX Runtime parity: OK, "
            f"max_abs_diff={validation['max_abs_diff']:.6g}, provider={validation['provider']}"
        )

    if not args.no_check:
        try:
            import onnx

            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            print("ONNX checker: OK")
        except ImportError:
            print("提示: 未安装 onnx，跳过 onnx.checker。安装命令: pip install onnx")

    if artifact_dir is not None:
        package_artifact(
            artifact_dir=artifact_dir,
            output_path=output_path,
            checkpoint=checkpoint,
            class_map_path=class_map_path,
            metrics_path=metrics_path,
            model_name=model_name,
            version=args.version,
            image_size=image_size,
            model_type=resolved_model_type,
            opset=args.opset,
        )
        print(f"Artifact 已打包: {artifact_dir.resolve()}")

    print(f"ONNX 已导出: {output_path.resolve()}")
    print("输入名: images, 输出名: logits, 动态 batch 维度: batch_size")


if __name__ == "__main__":
    main()
