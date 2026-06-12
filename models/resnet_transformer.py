from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNetFeatureBackbone(nn.Module):
    """
    作用:
        ResNet50 特征主干，显式暴露 layer3/layer4，便于多尺度融合与 Grad-CAM。
    输入:
        resnet: torchvision 构建的 ResNet50 模型。
    输出:
        forward_features 返回 (layer3_feature, layer4_feature)；
        forward 返回指定 stage 的特征图。
    """

    def __init__(self, resnet: nn.Module) -> None:
        super().__init__()
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        作用:
            一次前向同时提取 layer3 和 layer4 特征。
        输入:
            x: 输入图像 Tensor，形状 [B, 3, H, W]。
        输出:
            (layer3, layer4) 两个特征图。
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        layer3 = self.layer3(x)
        layer4 = self.layer4(layer3)
        return layer3, layer4

    def forward(self, x: torch.Tensor, stage: str = "layer3") -> torch.Tensor:
        """
        作用:
            按需提取单个 stage 特征，避免 layer3 模式下额外计算 layer4。
        输入:
            x: 输入图像 Tensor。
            stage: 'layer3' 或 'layer4'。
        输出:
            指定 stage 的特征图。
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        layer3 = self.layer3(x)
        if stage == "layer3":
            return layer3
        if stage == "layer4":
            layer4 = self.layer4(layer3)
            return layer4
        raise ValueError("stage 必须是 'layer3' 或 'layer4'。")


class AttentionEncoderBlock(nn.Module):
    """
    自定义 Transformer Encoder Block。

    相比 nn.TransformerEncoderLayer，这里可以在需要可解释性时保存 Multi-Head Attention 权重，
    供 tools/generate_attention_map.py 可视化 CLS Token 对空间 token 的关注区域。
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.capture_attention = False
        self.last_attention: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        作用:
            执行一个 Transformer Encoder Block，并在开启 capture_attention 时保存注意力权重。
        输入:
            x: token 序列，形状 [B, N, D]。
        输出:
            编码后的 token 序列，形状 [B, N, D]。
        """
        normed = self.norm1(x)
        attn_out, attn_weights = self.self_attn(
            normed,
            normed,
            normed,
            need_weights=self.capture_attention,
            average_attn_weights=False,
        )
        if self.capture_attention and attn_weights is not None:
            self.last_attention = attn_weights.detach()
        else:
            self.last_attention = None

        x = x + self.dropout1(attn_out)
        ffn_in = self.norm2(x)
        ffn_out = self.linear2(self.dropout(F.gelu(self.linear1(ffn_in))))
        x = x + self.dropout2(ffn_out)
        return x


class AttentionTransformerEncoder(nn.Module):
    """
    作用:
        堆叠多个 AttentionEncoderBlock，形成可捕获注意力权重的 Transformer Encoder。
    输入:
        depth: Encoder block 层数。
        d_model: token 维度。
        nhead: 多头注意力头数。
        dim_feedforward: FFN 隐藏层维度。
        dropout: Dropout 概率。
    输出:
        forward 返回编码后的 token 序列；get_last_attention 返回最后一层注意力权重。
    """

    def __init__(
        self,
        depth: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                AttentionEncoderBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        作用:
            顺序通过所有 Transformer block，并在最后做 LayerNorm。
        输入:
            x: token 序列，形状 [B, N, D]。
        输出:
            编码后的 token 序列，形状 [B, N, D]。
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def enable_attention_capture(self, enabled: bool = True) -> None:
        """
        作用:
            开关所有 Transformer block 的注意力权重捕获功能。
        输入:
            enabled: True 表示保存 attention，False 表示不保存。
        输出:
            无返回值；修改内部 block 状态。
        """
        for layer in self.layers:
            layer.capture_attention = enabled

    def get_last_attention(self) -> Optional[torch.Tensor]:
        """
        作用:
            获取最后一个 Transformer block 保存的注意力权重。
        输入:
            无。
        输出:
            注意力 Tensor，形状 [B, heads, N, N]；未捕获时返回 None。
        """
        if not self.layers:
            return None
        return self.layers[-1].last_attention


class ResNet50_Transformer(nn.Module):
    """
    ResNet50-Transformer 串行混合架构。

    支持三种特征模式:
        layer3: 使用高分辨率 32x32 局部纹理特征。
        layer4: 使用低分辨率 16x16 高语义特征。
        layer3_layer4: layer3 token 进 Transformer，layer4 全局语义向量与 Transformer 表征拼接。

    支持 token pooling:
        token_pool_size=16 时，layer3 的 32x32 token 会先 AdaptiveAvgPool 到 16x16，
        把自注意力复杂度从 1024^2 降到 256^2，显著降低显存压力。
    输入:
        num_classes: 输出类别数。
        image_size: 输入图像尺寸，用于初始化位置编码大小。
        transformer_dim/depth/heads/mlp_ratio: Transformer 结构超参数。
        pooling: 'cls' 使用 CLS Token，'mean' 使用平均池化。
        dropout: Dropout 概率。
        pretrained: 是否加载 ImageNet 预训练 ResNet50。
        backbone_stage: 'layer3'、'layer4' 或 'layer3_layer4'。
        token_pool_size: token 空间池化尺寸，0 表示不池化。
    输出:
        forward(x) 返回分类 logits，形状 [B, num_classes]。
    """

    def __init__(
        self,
        num_classes: int,
        image_size: int = 512,
        transformer_dim: int = 512,
        transformer_depth: int = 2,
        transformer_heads: int = 8,
        transformer_mlp_ratio: float = 4.0,
        pooling: str = "cls",
        dropout: float = 0.1,
        pretrained: bool = True,
        backbone_stage: str = "layer3",
        token_pool_size: int = 16,
    ) -> None:
        super().__init__()
        if transformer_dim % transformer_heads != 0:
            raise ValueError("transformer_dim 必须能被 transformer_heads 整除。")
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling 必须是 'cls' 或 'mean'。")
        if backbone_stage not in {"layer3", "layer4", "layer3_layer4"}:
            raise ValueError("backbone_stage 必须是 'layer3'、'layer4' 或 'layer3_layer4'。")
        if token_pool_size < 0:
            raise ValueError("token_pool_size 必须 >= 0，设为 0 表示不做 token pooling。")

        self.transformer_dim = transformer_dim
        self.pooling = pooling
        self.backbone_stage = backbone_stage
        self.token_pool_size = token_pool_size
        self.use_multiscale = backbone_stage == "layer3_layer4"
        self.last_token_hw: Tuple[int, int] = (0, 0)

        resnet = self._build_resnet50(pretrained=pretrained)
        self.cnn_backbone = ResNetFeatureBackbone(resnet)

        token_channels, downsample_rate = self._resolve_token_feature_spec(backbone_stage)
        self.channel_project = nn.Conv2d(token_channels, transformer_dim, kernel_size=1)
        self.token_pool = nn.AdaptiveAvgPool2d((token_pool_size, token_pool_size)) if token_pool_size > 0 else nn.Identity()

        base_feature_size = token_pool_size if token_pool_size > 0 else max(1, image_size // downsample_rate)
        self.base_feature_hw = (base_feature_size, base_feature_size)
        max_tokens = base_feature_size * base_feature_size

        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, transformer_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.transformer = AttentionTransformerEncoder(
            depth=transformer_depth,
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=int(transformer_dim * transformer_mlp_ratio),
            dropout=dropout,
        )

        if self.use_multiscale:
            self.semantic_project = nn.Sequential(
                nn.Linear(2048, transformer_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            classifier_dim = transformer_dim * 2
        else:
            self.semantic_project = None
            classifier_dim = transformer_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_dim),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, num_classes),
        )

    @staticmethod
    def _build_resnet50(pretrained: bool) -> nn.Module:
        """
        作用:
            构建 ResNet50，并兼容 torchvision 新旧预训练权重接口。
        输入:
            pretrained: 是否加载 ImageNet 预训练权重。
        输出:
            ResNet50 模型实例。
        """
        if not pretrained:
            return models.resnet50(weights=None)
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            return models.resnet50(weights=weights)
        except AttributeError:
            return models.resnet50(pretrained=True)

    @staticmethod
    def _resolve_token_feature_spec(backbone_stage: str) -> Tuple[int, int]:
        """
        作用:
            根据 backbone_stage 确定进入 Transformer 的特征通道数和下采样倍率。
        输入:
            backbone_stage: 特征模式名称。
        输出:
            (token_channels, downsample_rate)。
        """
        if backbone_stage in {"layer3", "layer3_layer4"}:
            return 1024, 16
        if backbone_stage == "layer4":
            return 2048, 32
        raise ValueError("backbone_stage 必须是 'layer3'、'layer4' 或 'layer3_layer4'。")

    def _get_token_and_semantic_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        作用:
            根据当前结构模式提取 token 特征图和可选的 layer4 语义特征图。
        输入:
            x: 输入图像 Tensor。
        输出:
            (token_feature_map, semantic_feature_map 或 None)。
        """
        if self.backbone_stage == "layer3":
            return self.cnn_backbone(x, stage="layer3"), None
        if self.backbone_stage == "layer4":
            return self.cnn_backbone(x, stage="layer4"), None
        layer3, layer4 = self.cnn_backbone.forward_features(x)
        return layer3, layer4

    def _get_position_embedding(self, feat_h: int, feat_w: int) -> torch.Tensor:
        """
        作用:
            获取与当前 token 空间尺寸匹配的位置编码，尺寸不一致时进行二维插值。
        输入:
            feat_h: token 特征图高度。
            feat_w: token 特征图宽度。
        输出:
            位置编码 Tensor，形状 [1, feat_h * feat_w, transformer_dim]。
        """
        base_h, base_w = self.base_feature_hw
        if (feat_h, feat_w) == (base_h, base_w):
            return self.pos_embed
        pos = self.pos_embed.reshape(1, base_h, base_w, self.transformer_dim).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(feat_h, feat_w), mode="bicubic", align_corners=False)
        return pos.permute(0, 2, 3, 1).reshape(1, feat_h * feat_w, self.transformer_dim)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """
        作用:
            控制 ResNet50 特征主干是否参与训练。
        输入:
            trainable: True 表示解冻，False 表示冻结。
        输出:
            无返回值；直接修改参数 requires_grad。
        """
        for param in self.cnn_backbone.parameters():
            param.requires_grad = trainable

    def enable_attention_capture(self, enabled: bool = True) -> None:
        """
        作用:
            开关 Transformer 注意力权重捕获，供 Attention Map 可视化使用。
        输入:
            enabled: 是否捕获 attention。
        输出:
            无返回值。
        """
        self.transformer.enable_attention_capture(enabled)

    def get_last_attention_map(self) -> Optional[torch.Tensor]:
        """
        作用:
            获取最后一层 Transformer 的注意力权重。
        输入:
            无。
        输出:
            注意力 Tensor 或 None。
        """
        return self.transformer.get_last_attention()

    def get_gradcam_target_layer(self) -> nn.Module:
        """
        作用:
            返回适合 Grad-CAM 挂钩的最后一层卷积模块。
        输入:
            无。
        输出:
            ResNet layer3/layer4 的最后一个 Bottleneck 模块。
        """
        if self.backbone_stage == "layer4":
            return self.cnn_backbone.layer4[-1]
        if self.backbone_stage == "layer3_layer4":
            return self.cnn_backbone.layer4[-1]
        return self.cnn_backbone.layer3[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        作用:
            执行完整 ResNet50-Transformer 前向传播。
        输入:
            x: 输入图像 Tensor，形状 [B, 3, H, W]。
        输出:
            分类 logits，形状 [B, num_classes]。
        """
        feature_map, semantic_map = self._get_token_and_semantic_maps(x)
        feature_map = self.channel_project(feature_map)
        feature_map = self.token_pool(feature_map)
        b, _, h, w = feature_map.shape
        self.last_token_hw = (h, w)

        tokens = feature_map.flatten(2).transpose(1, 2)
        pos_embed = self._get_position_embedding(h, w).to(dtype=tokens.dtype, device=tokens.device)
        tokens = tokens + pos_embed

        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(b, -1, -1).to(dtype=tokens.dtype, device=tokens.device)
            tokens = torch.cat((cls_tokens, tokens), dim=1)

        encoded_tokens = self.transformer(tokens)
        if self.pooling == "cls":
            global_feature = encoded_tokens[:, 0]
        else:
            global_feature = encoded_tokens.mean(dim=1)

        if self.use_multiscale and semantic_map is not None and self.semantic_project is not None:
            semantic_feature = F.adaptive_avg_pool2d(semantic_map, output_size=1).flatten(1)
            semantic_feature = self.semantic_project(semantic_feature)
            global_feature = torch.cat([global_feature, semantic_feature], dim=1)

        return self.classifier(global_feature)
