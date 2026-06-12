"""
生成 ResNet50-Transformer 的最后一层自注意力热力图。

用途:
    - Grad-CAM 展示 CNN 局部关注区域。
    - Attention Map 展示 Transformer 中 CLS Token 对全局空间 token 的关联关注。

示例:
    python tools/generate_attention_map.py --image archive/train_images/00021adfb725ed.jpg
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.resnet_transformer import ResNet50_Transformer
from utils.report_paths import timestamped_path


def parse_args() -> argparse.Namespace:
    """
    作用:
        解析 Transformer Attention Map 生成脚本参数。
    输入:
        命令行参数，如 --image、--checkpoint、--output。
    输出:
        argparse.Namespace 参数对象。
    """
    parser = argparse.ArgumentParser(description="生成 Transformer Attention Map")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--checkpoint", default="outputs/best_model.pth", help="模型权重路径")
    parser.add_argument("--class-map", default="outputs/class_to_idx.json", help="class_to_idx.json 路径")
    parser.add_argument("--output", default=None, help="输出图片路径；为空时写入 outputs/reports/interpretability/ 并自动加时间戳")
    parser.add_argument("--image-size", type=int, default=None, help="覆盖 checkpoint 中的 image_size")
    return parser.parse_args()


def load_class_map(path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    作用:
        读取类别映射文件。
    输入:
        path: class_to_idx.json 路径。
    输出:
        (class_to_idx, idx_to_class)。
    """
    with open(path, "r", encoding="utf-8") as f:
        class_to_idx = {str(k): int(v) for k, v in json.load(f).items()}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class


def load_model(checkpoint_path: Path, num_classes: int, image_size: Optional[int], device: torch.device) -> ResNet50_Transformer:
    """
    作用:
        根据 checkpoint 重建 ResNet50-Transformer 并加载权重。
    输入:
        checkpoint_path: checkpoint 路径。
        num_classes: 类别数量。
        image_size: 可选输入尺寸覆盖值。
        device: 运行设备。
    输出:
        eval 模式的 ResNet50_Transformer。
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model = ResNet50_Transformer(
        num_classes=num_classes,
        image_size=image_size or int(cfg.get("image_size", 512)),
        transformer_dim=int(cfg.get("transformer_dim", 512)),
        transformer_depth=int(cfg.get("transformer_depth", 2)),
        transformer_heads=int(cfg.get("transformer_heads", 8)),
        transformer_mlp_ratio=float(cfg.get("transformer_mlp_ratio", 4.0)),
        pooling=str(cfg.get("transformer_pooling", "cls")),
        dropout=float(cfg.get("dropout", 0.1)),
        pretrained=False,
        backbone_stage=str(cfg.get("backbone_stage", "layer3")),
        token_pool_size=int(cfg.get("token_pool_size", 16)),
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: Path, image_size: int) -> Tuple[torch.Tensor, np.ndarray]:
    """
    作用:
        读取图片并转换为模型输入 Tensor，同时保留原始 RGB 图。
    输入:
        image_path: 图片路径。
        image_size: 模型输入尺寸。
    输出:
        (input_tensor, original_rgb_array)。
    """
    image = Image.open(image_path).convert("RGB")
    original = np.asarray(image).copy()
    resized = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, original


@torch.no_grad()
def generate_attention(
    model: ResNet50_Transformer,
    input_tensor: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, int, float]:
    """
    作用:
        捕获最后一层 Transformer attention，并转换为空间注意力热力图。
    输入:
        model: 已加载的 ResNet50_Transformer。
        input_tensor: 预处理后的单张图片 Tensor。
        device: 运行设备。
    输出:
        (归一化 attention_map, 预测类别编号, Top-1 置信度)。
    """
    model.enable_attention_capture(True)
    logits = model(input_tensor.to(device))
    probs = torch.softmax(logits, dim=1)
    pred_index = int(probs.argmax(dim=1).item())
    confidence = float(probs[0, pred_index].item())

    attn = model.get_last_attention_map()
    model.enable_attention_capture(False)
    if attn is None:
        raise RuntimeError("未捕获到 attention 权重，请确认模型 forward 已执行。")

    # attn: [B, heads, target_tokens, source_tokens]
    attn = attn[0].detach().cpu()
    token_h, token_w = model.last_token_hw
    spatial_token_count = token_h * token_w

    if model.pooling == "cls":
        # CLS Token 对所有空间 token 的注意力，越高代表该区域对全局分类表征越重要。
        spatial_attention = attn[:, 0, -spatial_token_count:].mean(dim=0)
    else:
        # Mean pooling 模型没有 CLS Token，退化为所有 query 对 source token 的平均关注。
        spatial_attention = attn[:, :, -spatial_token_count:].mean(dim=(0, 1))

    attention_map = spatial_attention.reshape(token_h, token_w).numpy()
    attention_map = attention_map - attention_map.min()
    attention_map = attention_map / (attention_map.max() + 1e-8)
    return attention_map, pred_index, confidence


def overlay_attention(original_rgb: np.ndarray, attention_map: np.ndarray) -> np.ndarray:
    """
    作用:
        将 attention map 映射为热力图并与原图叠加。
    输入:
        original_rgb: 原始 RGB 图像数组。
        attention_map: 归一化空间注意力图。
    输出:
        叠加后的 RGB 图像数组。
    """
    h, w = original_rgb.shape[:2]
    resized = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap = np.uint8(255 * resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.clip(0.58 * original_rgb.astype(np.float32) + 0.42 * heatmap.astype(np.float32), 0, 255).astype(np.uint8)


def save_comparison(original_rgb: np.ndarray, overlay_rgb: np.ndarray, output_path: Path, title: str) -> None:
    """
    作用:
        拼接原图和 Transformer Attention 叠加图，并保存到磁盘。
    输入:
        original_rgb: 原始 RGB 图。
        overlay_rgb: 注意力叠加图。
        output_path: 输出路径。
        title: 右侧图标题。
    输出:
        无返回值；保存图片文件。
    """
    h, w = original_rgb.shape[:2]
    canvas = np.zeros((h + 52, w * 2, 3), dtype=np.uint8)
    canvas[:, :, :] = (4, 17, 28)
    canvas[52:, :w] = original_rgb
    canvas[52:, w:] = overlay_rgb

    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.putText(canvas_bgr, "Original", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 251, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas_bgr, f"Transformer Attention: {title}", (w + 18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (103, 232, 249), 2, cv2.LINE_AA)
    cv2.imencode(".jpg", canvas_bgr)[1].tofile(str(output_path))


def main() -> None:
    """
    作用:
        Attention Map 脚本主入口，完成模型加载、图片预处理、注意力生成和结果保存。
    输入:
        无显式输入；通过命令行参数读取图片和模型路径。
    输出:
        无返回值；生成 attention_result 图片。
    """
    args = parse_args()
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    class_map_path = PROJECT_ROOT / args.class_map
    image_path = PROJECT_ROOT / args.image
    output_path = PROJECT_ROOT / args.output if args.output else timestamped_path(
        PROJECT_ROOT / "outputs" / "reports" / "interpretability",
        "attention",
        ".jpg",
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")
    if not class_map_path.exists():
        raise FileNotFoundError(f"找不到类别映射: {class_map_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"找不到图片: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    image_size = args.image_size or int(cfg.get("image_size", 512))

    _, idx_to_class = load_class_map(class_map_path)
    model = load_model(checkpoint_path, num_classes=len(idx_to_class), image_size=image_size, device=device)
    input_tensor, original_rgb = preprocess_image(image_path, image_size=image_size)
    attention_map, pred_index, confidence = generate_attention(model, input_tensor, device)
    overlay_rgb = overlay_attention(original_rgb, attention_map)
    species_name = idx_to_class[pred_index].replace("_", " ").title()
    save_comparison(original_rgb, overlay_rgb, output_path, f"{species_name} {confidence * 100:.2f}%")
    print(f"Attention Map 已保存: {output_path.resolve()}")


if __name__ == "__main__":
    main()
