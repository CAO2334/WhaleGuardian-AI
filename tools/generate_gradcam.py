"""
为 ResNet50-Transformer 生成 Grad-CAM 可解释性图。

示例:
    python tools/generate_gradcam.py --image archive/train_images/00021adfb725ed.jpg
    python tools/generate_gradcam.py --image xxx.jpg --target-class humpback_whale
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
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.resnet_transformer import ResNet50_Transformer
from utils.report_paths import timestamped_path


def parse_args() -> argparse.Namespace:
    """
    作用:
        解析 Grad-CAM 生成脚本参数。
    输入:
        命令行参数，如 --image、--checkpoint、--target-class。
    输出:
        argparse.Namespace 参数对象。
    """
    parser = argparse.ArgumentParser(description="生成 ResNet50-Transformer Grad-CAM 图")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--checkpoint", default="outputs/best_model.pth", help="模型权重路径")
    parser.add_argument("--class-map", default="outputs/class_to_idx.json", help="class_to_idx.json 路径")
    parser.add_argument("--output", default=None, help="输出图片路径；为空时写入 outputs/reports/interpretability/ 并自动加时间戳")
    parser.add_argument("--target-class", default=None, help="目标类别名；为空时使用模型 Top-1 类别")
    parser.add_argument("--image-size", type=int, default=None, help="覆盖 checkpoint 中的 image_size")
    return parser.parse_args()


def load_class_map(path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    作用:
        读取类别映射文件，构造正向和反向类别映射。
    输入:
        path: class_to_idx.json 路径。
    输出:
        (class_to_idx, idx_to_class)。
    """
    with open(path, "r", encoding="utf-8") as f:
        class_to_idx = {str(k): int(v) for k, v in json.load(f).items()}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class


def load_model(checkpoint_path: Path, num_classes: int, image_size: Optional[int], device: torch.device) -> torch.nn.Module:
    """
    作用:
        根据 checkpoint 重建 ResNet50-Transformer 并加载权重。
    输入:
        checkpoint_path: checkpoint 路径。
        num_classes: 类别数量。
        image_size: 可选输入尺寸覆盖值。
        device: 运行设备。
    输出:
        eval 模式的模型。
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
        读取并预处理输入图片，同时保留原始 RGB 图用于叠加热力图。
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


def get_target_layer(model: ResNet50_Transformer) -> torch.nn.Module:
    """
    作用:
        获取 Grad-CAM 需要挂钩的目标卷积层。
    输入:
        model: ResNet50_Transformer 模型。
    输出:
        目标 nn.Module。
    """
    return model.get_gradcam_target_layer()


def generate_cam(
    model: ResNet50_Transformer,
    input_tensor: torch.Tensor,
    target_index: Optional[int],
    device: torch.device,
) -> Tuple[np.ndarray, int, float]:
    """
    作用:
        通过前向/反向 hook 计算指定类别的 Grad-CAM 热力图。
    输入:
        model: 待解释模型。
        input_tensor: 预处理后的单张图片 Tensor。
        target_index: 目标类别编号；None 表示使用模型 Top-1。
        device: 运行设备。
    输出:
        (归一化 CAM 数组, 目标/预测类别编号, 置信度)。
    """
    activations = {}
    gradients = {}
    target_layer = get_target_layer(model)

    def forward_hook(_module, _inputs, output):
        activations["value"] = output

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    try:
        input_tensor = input_tensor.to(device)
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        if target_index is None:
            target_index = int(probs.argmax(dim=1).item())
        confidence = float(probs[0, target_index].item())

        model.zero_grad(set_to_none=True)
        score = logits[0, target_index]
        score.backward()

        acts = activations["value"].detach()
        grads = gradients["value"].detach()
        # Grad-CAM: 对梯度做全局平均池化得到每个通道的重要性权重。
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam, target_index, confidence
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


def overlay_cam(original_rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """
    作用:
        将 CAM 热力图 resize 到原图尺寸并与原图叠加。
    输入:
        original_rgb: 原始 RGB 图像数组。
        cam: 归一化 CAM 数组。
    输出:
        叠加后的 RGB 图像数组。
    """
    h, w = original_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.clip(0.55 * original_rgb.astype(np.float32) + 0.45 * heatmap.astype(np.float32), 0, 255).astype(np.uint8)
    return overlay


def save_comparison(original_rgb: np.ndarray, overlay_rgb: np.ndarray, output_path: Path, title: str) -> None:
    """
    作用:
        将原图和 Grad-CAM 叠加图拼接保存为对比图。
    输入:
        original_rgb: 原图 RGB 数组。
        overlay_rgb: 叠加图 RGB 数组。
        output_path: 输出路径。
        title: 标题文本。
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
    cv2.putText(canvas_bgr, f"Grad-CAM: {title}", (w + 18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (103, 232, 249), 2, cv2.LINE_AA)
    cv2.imencode(".jpg", canvas_bgr)[1].tofile(str(output_path))


def main() -> None:
    """
    作用:
        Grad-CAM 脚本主入口，负责加载模型、预处理图片、生成热力图并保存结果。
    输入:
        无显式输入；通过命令行参数读取图片和模型路径。
    输出:
        无返回值；生成 cam_result 图片。
    """
    args = parse_args()
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    class_map_path = PROJECT_ROOT / args.class_map
    image_path = PROJECT_ROOT / args.image
    output_path = PROJECT_ROOT / args.output if args.output else timestamped_path(
        PROJECT_ROOT / "outputs" / "reports" / "interpretability",
        "gradcam",
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

    class_to_idx, idx_to_class = load_class_map(class_map_path)
    target_index = None
    if args.target_class:
        if args.target_class not in class_to_idx:
            raise ValueError(f"未知 target-class: {args.target_class}")
        target_index = class_to_idx[args.target_class]

    model = load_model(checkpoint_path, num_classes=len(class_to_idx), image_size=image_size, device=device)
    input_tensor, original_rgb = preprocess_image(image_path, image_size=image_size)
    cam, pred_index, confidence = generate_cam(model, input_tensor, target_index=target_index, device=device)
    overlay_rgb = overlay_cam(original_rgb, cam)

    species_name = idx_to_class[pred_index].replace("_", " ").title()
    save_comparison(original_rgb, overlay_rgb, output_path, f"{species_name} {confidence * 100:.2f}%")
    print(f"Grad-CAM 已保存: {output_path.resolve()}")


if __name__ == "__main__":
    main()
