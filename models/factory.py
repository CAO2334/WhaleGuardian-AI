from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

from models.resnet_baseline import ResNet50Baseline
from models.resnet_metric import ResNet50MetricClassifier
from models.resnet_transformer import ResNet50_Transformer


def _as_mapping(config: object) -> Mapping[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return config
    raise TypeError("config 必须是 dataclass 或 mapping。")


def build_model(
    config: object,
    num_classes: int,
    *,
    pretrained: bool | None = None,
    model_type_override: str | None = None,
    image_size_override: int | None = None,
) -> nn.Module:
    """Single model-construction entry point shared by training and tools."""
    cfg = _as_mapping(config)
    model_type = model_type_override or str(cfg.get("model_type", "transformer"))
    use_pretrained = bool(cfg.get("pretrained", True)) if pretrained is None else pretrained
    image_size = image_size_override or int(cfg.get("image_size", 512))

    if model_type == "baseline":
        return ResNet50Baseline(
            num_classes=num_classes,
            pretrained=use_pretrained,
            dropout=float(cfg.get("dropout", 0.1)),
        )
    if model_type == "transformer":
        return ResNet50_Transformer(
            num_classes=num_classes,
            image_size=image_size,
            transformer_dim=int(cfg.get("transformer_dim", 512)),
            transformer_depth=int(cfg.get("transformer_depth", 2)),
            transformer_heads=int(cfg.get("transformer_heads", 8)),
            transformer_mlp_ratio=float(cfg.get("transformer_mlp_ratio", 4.0)),
            pooling=str(cfg.get("transformer_pooling", "cls")),
            dropout=float(cfg.get("dropout", 0.1)),
            pretrained=use_pretrained,
            backbone_stage=str(cfg.get("backbone_stage", "layer3")),
            token_pool_size=int(cfg.get("token_pool_size", 16)),
        )
    if model_type == "metric":
        return ResNet50MetricClassifier(
            num_classes=num_classes,
            pretrained=use_pretrained,
            dropout=float(cfg.get("dropout", 0.1)),
            pooling=str(cfg.get("metric_pooling", "gem")),
            head_type=str(cfg.get("metric_head", "arcface")),
            embedding_dim=int(cfg.get("embedding_dim", 512)),
            gem_p=float(cfg.get("gem_p", 3.0)),
            arcface_scale=float(cfg.get("arcface_scale", 30.0)),
            arcface_margin=float(cfg.get("arcface_margin", 0.30)),
            arcface_subcenters=int(cfg.get("arcface_subcenters", 2)),
        )
    raise ValueError(f"不支持的 model_type: {model_type}")


def load_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    num_classes: int | None = None,
    *,
    model_type: str = "auto",
    image_size: int | None = None,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Rebuild a model from its checkpoint while keeping legacy checkpoints valid."""
    cfg = checkpoint.get("config", {})
    class_to_idx = checkpoint.get("class_to_idx", {})
    resolved_num_classes = num_classes or len(class_to_idx)
    if resolved_num_classes <= 0:
        raise ValueError("无法确定 num_classes；请提供类别映射。")
    resolved_type = None if model_type == "auto" else model_type
    model = build_model(
        cfg,
        num_classes=resolved_num_classes,
        pretrained=False,
        model_type_override=resolved_type,
        image_size_override=image_size,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model
