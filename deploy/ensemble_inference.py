"""ONNX Runtime 四模型概率集成推理。

集成目录由 ``tools/export_ensemble_artifacts.py`` 生成，结构如下::

    ensemble_artifacts/
      ensemble_manifest.json
      02_resnet50_ce_crop/model.onnx
      04_metric_linear_gem_crop/model.onnx
      06_transformer_mean_ce_crop/model.onnx
      07_previous_best_transformer_recipe/model.onnx

每个组件保持独立 artifact，manifest 中记录验证集锁定的 soft-voting 权重。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Union

import numpy as np

from deploy.onnx_inference import ImageInput, WhaleONNXPredictor


class WhaleEnsemblePredictor:
    """使用多个 ONNX artifact 对 28 类物种概率做加权平均。"""

    def __init__(
        self,
        artifact_dir: Union[str, Path],
        confidence_threshold: float = 0.5,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        manifest_path = self.artifact_dir / "ensemble_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"找不到集成 manifest: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)

        components = self.manifest.get("components")
        if not isinstance(components, list) or len(components) < 2:
            raise ValueError("ensemble_manifest.json 至少需要两个 components。")

        self.predictors: List[WhaleONNXPredictor] = []
        raw_weights: List[float] = []
        component_meta: List[Dict[str, object]] = []
        for component in components:
            if not isinstance(component, dict):
                raise ValueError("ensemble manifest 的 component 必须是对象。")
            relative_dir = component.get("artifact_dir")
            if not relative_dir:
                raise ValueError("ensemble component 缺少 artifact_dir。")
            component_path = Path(str(relative_dir))
            if not component_path.is_absolute():
                component_path = self.artifact_dir / component_path
            predictor = WhaleONNXPredictor(
                artifact_dir=component_path,
                confidence_threshold=confidence_threshold,
            )
            self.predictors.append(predictor)
            raw_weights.append(float(component.get("weight", 0.0)))
            component_meta.append(
                {
                    "name": str(component.get("name", component_path.name)),
                    "weight": float(component.get("weight", 0.0)),
                    "artifact_dir": str(component_path),
                    "experiment_name": predictor.info().get("experiment_name", ""),
                }
            )

        weights = np.asarray(raw_weights, dtype=np.float64)
        if np.any(weights < 0) or not np.isfinite(weights).all() or float(weights.sum()) <= 0:
            raise ValueError("ensemble 权重必须是非负有限数，且总和大于 0。")
        self.weights = (weights / weights.sum()).astype(np.float32)

        first = self.predictors[0]
        for predictor in self.predictors[1:]:
            if predictor.class_to_idx != first.class_to_idx:
                raise ValueError("集成组件的 class_to_idx 不一致。")
            if predictor.image_size != first.image_size:
                raise ValueError("集成组件的 image_size 不一致。")

        self.class_to_idx = first.class_to_idx
        self.idx_to_class = first.idx_to_class
        self.image_size = first.image_size
        self.confidence_threshold = float(confidence_threshold)
        self.providers = list(dict.fromkeys(provider for p in self.predictors for provider in p.providers))
        self.model_name = str(self.manifest.get("model_name", "whale_species_probability_ensemble"))
        self.version = str(self.manifest.get("version", "unknown"))
        self.experiment_name = str(
            self.manifest.get("experiment_name", "validation_locked_probability_ensemble")
        )
        self.component_meta = component_meta

    def info(self) -> Dict[str, object]:
        """返回健康检查和 API 使用的集成元信息。"""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "experiment_name": self.experiment_name,
            "artifact_dir": str(self.artifact_dir),
            "onnx_model_path": [str(p.onnx_path) for p in self.predictors],
            "class_map_path": [str(p.class_map_path) for p in self.predictors],
            "num_classes": len(self.class_to_idx),
            "image_size": self.image_size,
            "engine": "onnxruntime",
            "onnx_runtime": True,
            "providers": self.providers,
            "metrics": self.manifest.get("metrics", {}),
            "confidence_threshold": self.confidence_threshold,
            "model_count": len(self.predictors),
            "weights": {meta["name"]: float(weight) for meta, weight in zip(self.component_meta, self.weights)},
            "components": self.component_meta,
        }

    def predict(self, image: ImageInput, topk: int = 3) -> Dict[str, object]:
        """对单张图片执行四模型 soft-voting，并返回与单模型一致的 API 结构。"""
        if topk < 1:
            raise ValueError("topk 必须大于等于 1。")
        input_array = self.predictors[0].preprocess(image)
        probabilities = np.zeros(len(self.class_to_idx), dtype=np.float64)
        for weight, predictor in zip(self.weights, self.predictors):
            logits = predictor.session.run(
                [predictor.output_name],
                {predictor.input_name: input_array},
            )[0][0]
            probabilities += float(weight) * WhaleONNXPredictor._softmax(logits)

        top_indices = probabilities.argsort()[-topk:][::-1]
        topk_rows = []
        for idx in top_indices:
            species = self.idx_to_class[int(idx)]
            confidence = float(probabilities[int(idx)])
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
