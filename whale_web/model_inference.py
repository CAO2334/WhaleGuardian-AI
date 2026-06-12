"""
AI护鲸使者 Web 演示系统 - PyTorch 推理封装

职责:
    1. 加载 best_model.pth 和 class_to_idx.json。
    2. 构建与训练阶段一致的 ResNet50_Transformer。
    3. 将上传图片预处理为 [1, 3, 512, 512] Tensor。
    4. 输出 Top-3 物种概率，用于前端结果卡片与柱状图。
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import BinaryIO, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image


class WhaleSpeciesPredictor:
    """
    作用:
        旧版 PyTorch 推理封装，用于直接加载 best_model.pth 做 Web 推理。
    输入:
        model_path: PyTorch checkpoint 路径。
        class_map_path: class_to_idx.json 路径。
        project_root: 项目根目录，用于导入自定义模型类。
        image_size: 输入模型前的 resize 尺寸。
    输出:
        predict(file_obj) 返回物种预测、置信度、Top-3 和设备信息。
    注意:
        当前 Web 默认使用 deploy/onnx_inference.py，本类主要保留历史兼容。
    """

    def __init__(
        self,
        model_path: Path,
        class_map_path: Path,
        project_root: Path,
        image_size: int = 512,
    ) -> None:
        self.model_path = Path(model_path)
        self.class_map_path = Path(class_map_path)
        self.project_root = Path(project_root)
        self.image_size = image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = str(self.device)
        self.model = None
        self.idx_to_class: Dict[int, str] = {}
        self.status_message = "模型尚未加载。"

        self._try_load()

    @property
    def is_ready(self) -> bool:
        """
        作用:
            判断模型和类别映射是否已成功加载。
        输入:
            无。
        输出:
            True 表示可推理，False 表示不可推理。
        """
        return self.model is not None and bool(self.idx_to_class)

    def _try_load(self) -> None:
        """
        作用:
            尝试加载 checkpoint、类别映射和模型结构。
        输入:
            无显式输入；使用实例初始化时保存的路径。
        输出:
            无返回值；成功时设置 self.model 和 self.idx_to_class，失败时写入 status_message。
        """
        if not self.model_path.exists():
            self.status_message = f"未找到模型权重: {self.model_path}"
            return
        if not self.class_map_path.exists():
            self.status_message = f"未找到类别映射: {self.class_map_path}"
            return

        # train_resnet50_transformer.py 位于项目根目录。
        # Web 子目录启动时需要把项目根加入 sys.path，才能导入自定义模型类。
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

        try:
            from models.resnet_transformer import ResNet50_Transformer
        except Exception as exc:
            self.status_message = f"导入 ResNet50_Transformer 失败: {exc}"
            return

        class_to_idx = self._load_class_map(self.class_map_path)
        self.idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        num_classes = len(class_to_idx)

        checkpoint = torch.load(self.model_path, map_location=self.device)
        cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

        # 优先使用训练 checkpoint 中保存的结构参数，避免 Web 推理模型结构和训练结构不一致。
        model = ResNet50_Transformer(
            num_classes=num_classes,
            image_size=int(cfg.get("image_size", self.image_size)),
            transformer_dim=int(cfg.get("transformer_dim", 512)),
            transformer_depth=int(cfg.get("transformer_depth", 2)),
            transformer_heads=int(cfg.get("transformer_heads", 8)),
            transformer_mlp_ratio=float(cfg.get("transformer_mlp_ratio", 4.0)),
            dropout=float(cfg.get("dropout", 0.1)),
            pretrained=False,
            backbone_stage=str(cfg.get("backbone_stage", "layer3")),
        )

        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()

        self.model = model
        self.image_size = int(cfg.get("image_size", self.image_size))
        self.status_message = f"模型加载成功，共 {num_classes} 个物种类别，设备: {self.device_name}"

    @staticmethod
    def _load_class_map(class_map_path: Path) -> Dict[str, int]:
        """
        作用:
            读取类别映射 JSON。
        输入:
            class_map_path: class_to_idx.json 路径。
        输出:
            {species_name: class_index} 字典。
        """
        with open(class_map_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {str(name): int(idx) for name, idx in raw.items()}

    def predict(self, file_obj: BinaryIO) -> Dict[str, object]:
        """
        作用:
            对上传图片执行 PyTorch 推理并返回 Top-3 预测。
        输入:
            file_obj: 上传图片的二进制文件对象。
        输出:
            包含 species、confidence、top3、device 的结果字典。
        """
        if not self.is_ready:
            raise RuntimeError(self.status_message)

        image_tensor = self.preprocess_image(file_obj).to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)
            # Softmax 将 logits 转为概率分布，所有类别概率和为 1。
            probabilities = torch.softmax(logits, dim=1).squeeze(0)
            top_probs, top_indices = torch.topk(probabilities, k=min(3, probabilities.numel()))

        top3 = self._format_topk(top_indices.cpu().tolist(), top_probs.cpu().tolist())
        best = top3[0]

        return {
            "species": best["species"],
            "species_display": best["species_display"],
            "confidence": best["confidence"],
            "confidence_percent": best["confidence_percent"],
            "top3": top3,
            "device": self.device_name,
        }

    def preprocess_image(self, file_obj: BinaryIO) -> torch.Tensor:
        """
        将上传图片转为模型输入 Tensor。

        关键步骤:
            1. PIL 读取字节流，并强制转为 RGB，统一处理 PNG 透明通道和灰度图。
            2. Resize 到训练时的 512x512。
            3. 像素缩放到 [0, 1]。
            4. 使用 ImageNet mean/std 标准化，保持与 ResNet50 预训练分布一致。
            5. HWC -> CHW，并增加 batch 维度，得到 [1, 3, 512, 512]。
        """
        image_bytes = file_obj.read()
        if not image_bytes:
            raise ValueError("上传文件为空。")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        array = np.asarray(image).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        array = (array - mean) / std

        array = np.transpose(array, (2, 0, 1))
        tensor = torch.from_numpy(array).float().unsqueeze(0)
        return tensor

    def _format_topk(self, indices: List[int], probs: List[float]) -> List[Dict[str, object]]:
        """
        作用:
            将 top-k 类别编号和概率转换为前端友好的字典列表。
        输入:
            indices: 类别编号列表。
            probs: 对应概率列表。
        输出:
            包含 species、species_display、confidence、confidence_percent 的列表。
        """
        rows = []
        for idx, prob in zip(indices, probs):
            species = self.idx_to_class[int(idx)]
            rows.append(
                {
                    "species": species,
                    "species_display": self._display_species(species),
                    "confidence": float(prob),
                    "confidence_percent": round(float(prob) * 100.0, 2),
                }
            )
        return rows

    @staticmethod
    def _display_species(species: str) -> str:
        """
        作用:
            将下划线类别名转换为标题格式显示名。
        输入:
            species: 原始类别名，例如 humpback_whale。
        输出:
            展示名，例如 Humpback Whale。
        """
        # 模型类别名来自 CSV，例如 humpback_whale；前端展示时转成 Humpback Whale。
        return species.replace("_", " ").title()
