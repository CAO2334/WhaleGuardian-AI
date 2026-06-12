"""
ONNX Runtime 轻量级推理类。

示例:
    from deploy.onnx_inference import WhaleONNXPredictor

    predictor = WhaleONNXPredictor(
        onnx_path="whale_model.onnx",
        class_map_path="outputs/class_to_idx.json",
    )
    result = predictor.predict("archive/train_images/00021adfb725ed.jpg")
    print(result)
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import BinaryIO, Dict, List, Union

import numpy as np
from PIL import Image


ImageInput = Union[str, Path, BinaryIO, bytes]


class WhaleONNXPredictor:
    """
    作用:
        封装 ONNX Runtime 鲸类物种推理流程，包括 artifact 加载、图片预处理、Top-k 输出和低置信度判断。
    输入:
        onnx_path: 传统模式下的 ONNX 模型路径。
        class_map_path: 传统模式下的 class_to_idx.json 路径。
        image_size: 输入模型前的 resize 尺寸。
        providers: ONNX Runtime 执行后端列表。
        artifact_dir: 模型版本目录；传入后优先读取 model.onnx、class_to_idx.json、config.json。
        confidence_threshold: Top-1 低置信度阈值，低于该值标记为 uncertain。
    输出:
        predict(image) 返回物种预测、置信度、Top-3、providers 和不确定性判断。
    """

    def __init__(
        self,
        onnx_path: Union[str, Path] = "whale_model.onnx",
        class_map_path: Union[str, Path] = "outputs/class_to_idx.json",
        image_size: int = 512,
        providers: List[str] | None = None,
        artifact_dir: Union[str, Path, None] = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("缺少 onnxruntime，请先安装: pip install onnxruntime") from exc

        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self.manifest = self._load_optional_json(self.artifact_dir / "manifest.json") if self.artifact_dir else {}
        self.config = self._load_optional_json(self.artifact_dir / "config.json") if self.artifact_dir else {}
        self.metrics = self._load_optional_json(self.artifact_dir / "metrics.json") if self.artifact_dir else {}

        if self.artifact_dir is not None:
            model_file = str(self.manifest.get("model_file", "model.onnx"))
            class_map_file = str(self.manifest.get("class_map_file", "class_to_idx.json"))
            self.onnx_path = self.artifact_dir / model_file
            self.class_map_path = self.artifact_dir / class_map_file
            self.image_size = int(self.config.get("image_size", self.manifest.get("image_size", image_size)))
        else:
            self.onnx_path = Path(onnx_path)
            self.class_map_path = Path(class_map_path)
            self.image_size = image_size
        self.confidence_threshold = float(confidence_threshold)

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"找不到 ONNX 模型: {self.onnx_path}")
        if not self.class_map_path.exists():
            raise FileNotFoundError(f"找不到类别映射: {self.class_map_path}")

        self.class_to_idx = self._load_class_map(self.class_map_path)
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        available = ort.get_available_providers()
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.providers = self.session.get_providers()
        self.model_name = str(self.manifest.get("model_name", self.config.get("model_name", self.onnx_path.stem)))
        self.version = str(self.manifest.get("version", self.config.get("version", "unknown")))

    @staticmethod
    def _load_class_map(path: Path) -> Dict[str, int]:
        """
        作用:
            读取 class_to_idx.json，并确保类别编号为 int。
        输入:
            path: 类别映射 JSON 路径。
        输出:
            {species_name: class_index} 字典。
        """
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {str(name): int(idx) for name, idx in raw.items()}

    @staticmethod
    def _load_optional_json(path: Path) -> Dict[str, object]:
        """
        作用:
            安全读取可选 JSON 文件，不存在时返回空字典。
        输入:
            path: JSON 文件路径。
        输出:
            JSON 字典或空字典。
        """
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def info(self) -> Dict[str, object]:
        """
        作用:
            返回当前推理器加载的模型版本、路径、类别数、阈值和运行后端信息。
        输入:
            无。
        输出:
            模型健康检查信息字典。
        """
        return {
            "model_name": self.model_name,
            "version": self.version,
            "artifact_dir": str(self.artifact_dir) if self.artifact_dir is not None else "",
            "onnx_model_path": str(self.onnx_path),
            "class_map_path": str(self.class_map_path),
            "num_classes": len(self.class_to_idx),
            "image_size": self.image_size,
            "engine": "onnxruntime",
            "onnx_runtime": True,
            "providers": self.providers,
            "metrics": self.metrics,
            "confidence_threshold": self.confidence_threshold,
        }

    def preprocess(self, image: ImageInput) -> np.ndarray:
        """
        ONNX 推理预处理，与训练/Flask 推理保持一致:
            1. RGB
            2. Resize 到 image_size x image_size
            3. /255 转 float32
            4. ImageNet mean/std 归一化
            5. HWC -> CHW，并添加 batch 维度 [1, 3, H, W]
        """
        pil_image = self._read_image(image).convert("RGB")
        pil_image = pil_image.resize((self.image_size, self.image_size), Image.BILINEAR)

        array = np.asarray(pil_image).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        array = (array - mean) / std
        array = np.transpose(array, (2, 0, 1))
        return np.expand_dims(array, axis=0).astype(np.float32)

    @staticmethod
    def _read_image(image: ImageInput) -> Image.Image:
        """
        作用:
            统一读取不同输入形式的图片。
        输入:
            image: 文件路径、Path、二进制文件对象或 bytes。
        输出:
            PIL Image 对象。
        """
        if isinstance(image, (str, Path)):
            return Image.open(image)
        if isinstance(image, bytes):
            return Image.open(io.BytesIO(image))
        return Image.open(image)

    def predict(self, image: ImageInput, topk: int = 3) -> Dict[str, object]:
        """
        作用:
            对单张图片执行 ONNX 推理，输出 Top-k 预测和低置信度判断。
        输入:
            image: 图片路径、文件对象或 bytes。
            topk: 返回概率最高的类别数量。
        输出:
            包含 species、confidence、top3、is_uncertain、decision 等字段的字典。
        """
        input_array = self.preprocess(image)
        logits = self.session.run([self.output_name], {self.input_name: input_array})[0]
        probs = self._softmax(logits[0])
        top_indices = probs.argsort()[-topk:][::-1]

        topk_rows = []
        for idx in top_indices:
            species = self.idx_to_class[int(idx)]
            confidence = float(probs[int(idx)])
            topk_rows.append(
                {
                    "species": species,
                    "species_display": species.replace("_", " ").title(),
                    "confidence": confidence,
                    "confidence_percent": round(confidence * 100.0, 2),
                }
            )

        best = topk_rows[0]
        is_uncertain = best["confidence"] < self.confidence_threshold
        decision = "uncertain" if is_uncertain else "accepted"
        message = (
            "模型置信度较低，建议上传更清晰、主体更完整的鲸类图片，或人工复核 Top-3 结果。"
            if is_uncertain
            else "模型置信度达到阈值。"
        )
        return {
            "species": best["species"],
            "species_display": best["species_display"],
            "confidence": best["confidence"],
            "confidence_percent": best["confidence_percent"],
            "confidence_threshold": self.confidence_threshold,
            "confidence_threshold_percent": round(self.confidence_threshold * 100.0, 2),
            "is_uncertain": is_uncertain,
            "decision": decision,
            "message": message,
            "top3": topk_rows,
            "providers": self.providers,
        }

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """
        作用:
            对 logits 做数值稳定的 softmax，得到类别概率。
        输入:
            logits: 一维类别 logits 数组。
        输出:
            一维类别概率数组。
        """
        logits = logits.astype(np.float64)
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        return (exp / np.sum(exp)).astype(np.float32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ONNX 鲸类物种识别推理")
    parser.add_argument("--artifact-dir", default=None, help="模型 artifact 目录；传入后会自动读取 model.onnx/class_to_idx.json/config.json")
    parser.add_argument("--onnx", default="whale_model.onnx", help="ONNX 模型路径")
    parser.add_argument("--class-map", default="outputs/class_to_idx.json", help="class_to_idx.json 路径")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="低置信度阈值，低于该值标记为 uncertain")
    args = parser.parse_args()

    predictor = WhaleONNXPredictor(
        onnx_path=args.onnx,
        class_map_path=args.class_map,
        image_size=args.image_size,
        artifact_dir=args.artifact_dir,
        confidence_threshold=args.confidence_threshold,
    )
    print(predictor.predict(args.image))
