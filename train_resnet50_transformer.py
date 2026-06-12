"""
兼容入口：训练代码已拆分到标准模块结构。

新训练入口:
    python train.py --data-root archive

保留本文件是为了兼容旧命令和 Web 推理中的历史导入:
    from train_resnet50_transformer import ResNet50_Transformer
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from configs.config import SPECIES_TYPO_MAP, TrainConfig, parse_args
from data.dataset import (
    NumpyTrainTransform,
    NumpyValTransform,
    WhaleSpeciesDataset,
    build_dataloader,
    build_label_maps,
    build_transforms,
    count_group_overlap,
    group_stratified_split,
    normalize_species_column,
    read_rgb_image,
    resolve_data_paths,
    resolve_num_workers,
    split_train_val,
    stratified_split,
)
from models.resnet_transformer import ResNet50_Transformer
from models.resnet_baseline import ResNet50Baseline
from train import (
    build_criterion,
    build_experiment_name,
    build_experiment_summary,
    build_optimizer,
    build_scheduler,
    main,
    print_dataset_summary,
    save_checkpoint,
    save_experiment_summary,
    set_seed,
    train_one_epoch,
    validate_one_epoch,
)
from utils.losses import FocalLoss, compute_class_balanced_alpha, mixup_criterion, mixup_data
from utils.metrics import (
    ModelEMA,
    accuracy_from_logits,
    create_summary_writer,
    log_epoch_metrics,
    macro_f1_score,
)

__all__ = [
    "SPECIES_TYPO_MAP",
    "TrainConfig",
    "parse_args",
    "NumpyTrainTransform",
    "NumpyValTransform",
    "WhaleSpeciesDataset",
    "build_dataloader",
    "build_label_maps",
    "build_transforms",
    "count_group_overlap",
    "group_stratified_split",
    "normalize_species_column",
    "read_rgb_image",
    "resolve_data_paths",
    "resolve_num_workers",
    "split_train_val",
    "stratified_split",
    "ResNet50_Transformer",
    "ResNet50Baseline",
    "build_criterion",
    "build_experiment_name",
    "build_experiment_summary",
    "build_optimizer",
    "build_scheduler",
    "main",
    "print_dataset_summary",
    "save_checkpoint",
    "save_experiment_summary",
    "set_seed",
    "train_one_epoch",
    "validate_one_epoch",
    "FocalLoss",
    "compute_class_balanced_alpha",
    "mixup_criterion",
    "mixup_data",
    "ModelEMA",
    "accuracy_from_logits",
    "create_summary_writer",
    "log_epoch_metrics",
    "macro_f1_score",
]


if __name__ == "__main__":
    main()
