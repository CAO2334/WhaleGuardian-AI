from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from configs.config import SPECIES_TYPO_MAP, TrainConfig

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from albumentations import Compose
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    A = None
    Compose = object
    ToTensorV2 = None


def resolve_data_paths(cfg: TrainConfig) -> Tuple[Path, Path]:
    """
    作用:
        根据训练配置解析 train.csv 和 train_images 目录路径，并检查它们是否存在。
    输入:
        cfg: 训练配置，包含 data_root、train_csv、image_dir。
    输出:
        (train_csv 路径, image_dir 路径)。
    """
    data_root = Path(cfg.data_root)
    train_csv = Path(cfg.train_csv) if cfg.train_csv else data_root / "train.csv"
    image_dir = Path(cfg.image_dir) if cfg.image_dir else data_root / "train_images"
    if not train_csv.exists():
        raise FileNotFoundError(f"找不到训练 CSV: {train_csv.resolve()}")
    if not image_dir.exists():
        raise FileNotFoundError(f"找不到训练图片目录: {image_dir.resolve()}")
    return train_csv, image_dir


def normalize_species_column(df: pd.DataFrame, fix_typos: bool = True) -> pd.DataFrame:
    """
    作用:
        校验并清洗 CSV 中的 species 列，必要时合并 Kaggle 数据中的物种拼写噪声。
    输入:
        df: 原始 CSV 读取后的 DataFrame。
        fix_typos: 是否修正已知拼写错误。
    输出:
        清洗后的 DataFrame 副本。
    """
    df = df.copy()
    if "species" not in df.columns:
        raise ValueError("train.csv 必须包含 species 列。")
    if "image" not in df.columns:
        raise ValueError("train.csv 必须包含 image 列。")
    df["species"] = df["species"].astype(str)
    if fix_typos:
        df["species"] = df["species"].replace(SPECIES_TYPO_MAP)
    return df


def build_label_maps(species: Iterable[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    作用:
        将物种字符串标签编码为连续整数类别，并构造反向映射。
    输入:
        species: 物种名称序列。
    输出:
        (class_to_idx, idx_to_class) 两个字典。
    """
    classes = sorted(set(species))
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class


def stratified_split(
    df: pd.DataFrame,
    label_col: str,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    作用:
        按类别做图片级分层随机划分，尽量让每个 species 都进入验证集。
    输入:
        df: 完整数据 DataFrame。
        label_col: 类别列名，通常为 species。
        val_ratio: 验证集比例。
        seed: 随机种子。
    输出:
        (训练集 DataFrame, 验证集 DataFrame)。
    注意:
        该方法不保证 individual_id 无泄漏；更严谨时应使用 group_stratified_split。
    """
    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    for _, group in df.groupby(label_col):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        if len(indices) <= 1:
            train_indices.extend(indices.tolist())
            continue
        val_count = max(1, int(round(len(indices) * val_ratio)))
        val_count = min(val_count, len(indices) - 1)
        val_indices.extend(indices[:val_count].tolist())
        train_indices.extend(indices[val_count:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return df.loc[train_indices].reset_index(drop=True), df.loc[val_indices].reset_index(drop=True)


def group_stratified_split(
    df: pd.DataFrame,
    label_col: str,
    group_col: str,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按 group_col 做验证集划分，确保同一个 individual_id 不会同时出现在训练和验证。

    实现思路:
        1. 先把样本聚合到 group 级别，每个 group 对应一个鲸鱼个体。
        2. 用 group 内出现最多的 species 作为该 group 的分层标签。
        3. 在每个 species 内按 group 粒度抽取验证集，尽量接近 val_ratio。

    这样比直接按图片随机分层更严格，可以降低“模型记住同一个体外观/背景”的泄漏风险。
    """
    if group_col not in df.columns:
        raise ValueError(f"group split 需要 CSV 包含 {group_col} 列。")
    if label_col not in df.columns:
        raise ValueError(f"group split 需要 CSV 包含 {label_col} 列。")

    rng = np.random.default_rng(seed)
    group_rows = []
    for group_id, group_df in df.groupby(group_col, sort=False):
        label = group_df[label_col].mode(dropna=False).iloc[0]
        group_rows.append(
            {
                "group_id": group_id,
                "label": label,
                "indices": group_df.index.to_numpy(),
                "size": len(group_df),
            }
        )

    groups = pd.DataFrame(group_rows)
    train_indices: List[int] = []
    val_indices: List[int] = []

    for _, label_groups in groups.groupby("label", sort=False):
        rows = label_groups.to_dict("records")
        rng.shuffle(rows)
        total_samples = int(sum(row["size"] for row in rows))

        if len(rows) <= 1:
            for row in rows:
                train_indices.extend(row["indices"].tolist())
            continue

        target_val_samples = max(1, int(round(total_samples * val_ratio)))
        current_val_samples = 0
        selected_val_groups = 0

        for row in rows:
            remaining_groups = len(rows) - selected_val_groups
            if remaining_groups <= 1:
                train_indices.extend(row["indices"].tolist())
                continue

            if current_val_samples < target_val_samples:
                val_indices.extend(row["indices"].tolist())
                current_val_samples += int(row["size"])
                selected_val_groups += 1
            else:
                train_indices.extend(row["indices"].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    if not val_indices:
        raise ValueError("group split 未能生成验证集，请检查 val_ratio 或 group 分布。")
    return df.loc[train_indices].reset_index(drop=True), df.loc[val_indices].reset_index(drop=True)


def split_train_val(
    df: pd.DataFrame,
    label_col: str,
    val_ratio: float,
    seed: int,
    split_strategy: str = "group",
    group_col: str = "individual_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    作用:
        统一训练/验证划分入口，根据 split_strategy 调用普通分层划分或 group 分层划分。
    输入:
        df: 完整数据 DataFrame。
        label_col: 类别列名。
        val_ratio: 验证集比例。
        seed: 随机种子。
        split_strategy: 'group' 或 'stratified'。
        group_col: group 划分使用的列名，默认 individual_id。
    输出:
        (训练集 DataFrame, 验证集 DataFrame)。
    """
    if split_strategy == "stratified":
        return stratified_split(df=df, label_col=label_col, val_ratio=val_ratio, seed=seed)
    if split_strategy == "group":
        return group_stratified_split(
            df=df,
            label_col=label_col,
            group_col=group_col,
            val_ratio=val_ratio,
            seed=seed,
        )
    raise ValueError("split_strategy 必须是 'group' 或 'stratified'。")


def count_group_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame, group_col: str) -> int:
    """
    作用:
        统计训练集和验证集在指定 group 列上的重叠数量，用于检查个体泄漏。
    输入:
        train_df: 训练集 DataFrame。
        val_df: 验证集 DataFrame。
        group_col: 分组列名，通常为 individual_id。
    输出:
        重叠 group 数量；如果列不存在返回 -1。
    """
    if group_col not in train_df.columns or group_col not in val_df.columns:
        return -1
    train_groups = set(train_df[group_col].astype(str))
    val_groups = set(val_df[group_col].astype(str))
    return len(train_groups.intersection(val_groups))


def read_rgb_image(image_path: Path) -> Optional[np.ndarray]:
    """
    作用:
        读取图片并转换为 RGB NumPy 数组，兼容 Windows 中文路径。
    输入:
        image_path: 图片文件路径。
    输出:
        RGB 图像数组，形状为 [H, W, 3]；读取失败时返回 None。
    """
    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class WhaleSpeciesDataset(Dataset):
    """
    作用:
        Happywhale 物种分类数据集类，根据 CSV 行读取图片并返回训练所需的 Tensor 和标签。
    输入:
        df: 包含 image 和 species 的 DataFrame。
        image_dir: 图片目录。
        class_to_idx: 物种名到类别编号映射。
        transforms: 图像增强/预处理流水线。
    输出:
        __getitem__ 返回 (image_tensor, label_tensor)。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Path,
        class_to_idx: Dict[str, int],
        transforms: Optional[object] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.class_to_idx = class_to_idx
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        image_path = self.image_dir / row["image"]
        image = read_rgb_image(image_path)
        if image is None:
            raise FileNotFoundError(f"图像读取失败: {image_path}")
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        label = self.class_to_idx[row["species"]]
        return image, torch.tensor(label, dtype=torch.long)


class NumpyTrainTransform:
    """
    作用:
        当 albumentations 不可用时的训练增强 fallback，基于 OpenCV/NumPy 实现。
    输入:
        image_size: 输出图像尺寸。
        cutout_p: Cutout 随机遮挡概率。
    输出:
        调用实例时返回 {"image": Tensor}。
    """

    def __init__(self, image_size: int, cutout_p: float) -> None:
        self.image_size = image_size
        self.cutout_p = cutout_p
        self.mean = np.array((0.485, 0.456, 0.406), dtype=np.float32)
        self.std = np.array((0.229, 0.224, 0.225), dtype=np.float32)

    def __call__(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        if random.random() < 0.5:
            angle = random.uniform(-15.0, 15.0)
            h, w = image.shape[:2]
            matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
        if random.random() < 0.7:
            alpha = 1.0 + random.uniform(-0.25, 0.25)
            beta = random.uniform(-0.25, 0.25) * 255.0
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        if random.random() < self.cutout_p:
            image = self._apply_cutout(image)
        return {"image": self._normalize_to_tensor(image)}

    def _apply_cutout(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        min_hole = max(8, int(self.image_size * 0.08))
        max_hole = max(min_hole + 1, int(self.image_size * 0.25))
        image = image.copy()
        for _ in range(random.randint(1, 2)):
            hole_h = random.randint(min_hole, max_hole)
            hole_w = random.randint(min_hole, max_hole)
            cy = random.randint(0, h - 1)
            cx = random.randint(0, w - 1)
            y1 = max(0, cy - hole_h // 2)
            y2 = min(h, cy + hole_h // 2)
            x1 = max(0, cx - hole_w // 2)
            x2 = min(w, cx + hole_w // 2)
            image[y1:y2, x1:x2, :] = 0
        return image

    def _normalize_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        return torch.from_numpy(image).float()


class NumpyValTransform:
    """
    作用:
        当 albumentations 不可用时的验证预处理 fallback，只做 resize 和归一化。
    输入:
        image_size: 输出图像尺寸。
    输出:
        调用实例时返回 {"image": Tensor}。
    """

    def __init__(self, image_size: int) -> None:
        self.image_size = image_size
        self.mean = np.array((0.485, 0.456, 0.406), dtype=np.float32)
        self.std = np.array((0.229, 0.224, 0.225), dtype=np.float32)

    def __call__(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        return {"image": torch.from_numpy(image).float()}


def make_cutout_transform(image_size: int, p: float):
    """
    作用:
        构造 albumentations 的 CoarseDropout/Cutout 增强，并兼容不同 albumentations 版本参数名。
    输入:
        image_size: 图像尺寸，用于确定遮挡块大小范围。
        p: Cutout 触发概率。
    输出:
        albumentations 变换对象。
    """
    min_hole = max(8, int(image_size * 0.08))
    max_hole = max(min_hole + 1, int(image_size * 0.25))
    try:
        return A.CoarseDropout(
            num_holes_range=(1, 2),
            hole_height_range=(min_hole, max_hole),
            hole_width_range=(min_hole, max_hole),
            fill=0,
            p=p,
        )
    except TypeError:
        return A.CoarseDropout(
            min_holes=1,
            max_holes=2,
            min_height=min_hole,
            max_height=max_hole,
            min_width=min_hole,
            max_width=max_hole,
            fill_value=0,
            p=p,
        )


def build_transforms(image_size: int, cutout_p: float) -> Tuple[object, object]:
    """
    作用:
        构建训练和验证图像预处理流水线。
    输入:
        image_size: 输入模型前统一 resize 的尺寸。
        cutout_p: 训练增强中 Cutout 的概率。
    输出:
        (训练 transforms, 验证 transforms)。
    """
    if A is None or ToTensorV2 is None:
        print("提示: 当前环境未安装 albumentations，已启用内置 OpenCV/Torch fallback 增强。")
        return NumpyTrainTransform(image_size=image_size, cutout_p=cutout_p), NumpyValTransform(image_size=image_size)

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    train_tfms = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
            make_cutout_transform(image_size=image_size, p=cutout_p),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2(),
        ]
    )
    val_tfms = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2(),
        ]
    )
    return train_tfms, val_tfms


def resolve_num_workers(cfg: TrainConfig, device: torch.device) -> int:
    """
    作用:
        根据系统平台和训练设备自动决定 DataLoader num_workers。
    输入:
        cfg: 训练配置，若 cfg.num_workers >= 0 则优先使用用户指定值。
        device: 当前训练设备。
    输出:
        DataLoader worker 进程数。
    """
    if cfg.num_workers >= 0:
        return cfg.num_workers
    if os.name == "nt":
        return 0
    if device.type == "cuda":
        cpu_count = os.cpu_count() or 4
        return min(8, max(4, cpu_count // 2))
    return 0


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool = False,
) -> DataLoader:
    """
    作用:
        构造 PyTorch DataLoader，并在多进程模式下启用 persistent_workers 和 prefetch。
    输入:
        dataset: Dataset 实例。
        batch_size: 批大小。
        shuffle: 是否打乱数据。
        num_workers: 数据加载进程数。
        pin_memory: 是否启用页锁定内存。
        drop_last: 是否丢弃最后不足 batch 的样本。
    输出:
        DataLoader 实例。
    """
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)
