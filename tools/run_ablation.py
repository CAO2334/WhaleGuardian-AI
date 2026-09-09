"""Run reproducible, controlled ablation suites.

Examples:
    python tools/run_ablation.py --dry-run --suite controlled
    python tools/run_ablation.py --run --suite controlled --seeds 42 123 3407
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AblationExperiment:
    name: str
    model_type: str
    loss_type: str = "ce"
    mixup_alpha: float = 0.0
    cutout_p: float = 0.0
    crop_scale_min: float = 1.0
    use_ema: bool = False
    transformer_pooling: str = "mean"
    metric_pooling: str = "gem"
    metric_head: str = "linear"


CONTROLLED_EXPERIMENTS = [
    AblationExperiment("01_resnet50_ce_resize", "baseline"),
    AblationExperiment("02_resnet50_ce_crop", "baseline", crop_scale_min=0.80),
    AblationExperiment("03_metric_linear_gap_crop", "metric", crop_scale_min=0.80, metric_pooling="gap"),
    AblationExperiment("04_metric_linear_gem_crop", "metric", crop_scale_min=0.80, metric_pooling="gem"),
    AblationExperiment(
        "05_metric_arcface_gem_crop",
        "metric",
        crop_scale_min=0.80,
        metric_pooling="gem",
        metric_head="arcface",
    ),
    AblationExperiment(
        "06_transformer_mean_ce_crop",
        "transformer",
        crop_scale_min=0.80,
        transformer_pooling="mean",
    ),
    AblationExperiment(
        "07_previous_best_transformer_recipe",
        "transformer",
        loss_type="focal",
        mixup_alpha=0.4,
        cutout_p=0.5,
        transformer_pooling="mean",
    ),
]


LEGACY_EXPERIMENTS = [
    AblationExperiment("01_resnet50_ce_plain", "baseline"),
    AblationExperiment("02_resnet50_focal_plain", "baseline", loss_type="focal"),
    AblationExperiment(
        "03_resnet50_focal_mixup_cutout_ema",
        "baseline",
        loss_type="focal",
        mixup_alpha=0.4,
        cutout_p=0.5,
        use_ema=True,
    ),
    AblationExperiment(
        "04_transformer_mean_focal_mixup_cutout",
        "transformer",
        loss_type="focal",
        mixup_alpha=0.4,
        cutout_p=0.5,
        transformer_pooling="mean",
    ),
    AblationExperiment(
        "05_transformer_cls_focal_mixup_cutout",
        "transformer",
        loss_type="focal",
        mixup_alpha=0.4,
        cutout_p=0.5,
        transformer_pooling="cls",
    ),
    AblationExperiment(
        "06_transformer_cls_focal_mixup_cutout_ema",
        "transformer",
        loss_type="focal",
        mixup_alpha=0.4,
        cutout_p=0.5,
        use_ema=True,
        transformer_pooling="cls",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行鲸类物种分类受控消融实验")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="真正开始训练")
    mode.add_argument("--dry-run", action="store_true", help="只打印命令")
    parser.add_argument("--suite", choices=("controlled", "legacy"), default="controlled")
    parser.add_argument("--data-root", default="archive")
    parser.add_argument("--output-root", default="outputs/ablations")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="训练随机种子，可传多个")
    parser.add_argument("--split-strategy", choices=("group", "stratified"), default="group")
    parser.add_argument("--group-col", default="individual_id")
    parser.add_argument("--token-pool-size", type=int, default=16)
    parser.add_argument(
        "--backbone-stage",
        choices=("layer3", "layer4", "layer3_layer4"),
        default="layer3_layer4",
    )
    parser.add_argument("--arcface-scale", type=float, default=30.0)
    parser.add_argument("--arcface-margin", type=float, default=0.30)
    parser.add_argument("--arcface-subcenters", type=int, default=2)
    parser.add_argument("--head-lr-multiplier", type=float, default=10.0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--skip-completed", action="store_true", help="metrics.json 已存在时跳过")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def get_suite(name: str) -> List[AblationExperiment]:
    return CONTROLLED_EXPERIMENTS if name == "controlled" else LEGACY_EXPERIMENTS


def build_command(args: argparse.Namespace, exp: AblationExperiment, seed: int) -> List[str]:
    output_root = Path(args.output_root)
    output_dir = output_root / exp.name / f"seed_{seed}"
    command = [
        args.python,
        str(PROJECT_ROOT / "train.py"),
        "--data-root", args.data_root,
        "--output-dir", str(output_dir),
        "--results-csv", str(output_root / "ablation_results.csv"),
        "--experiment-name", exp.name,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--image-size", str(args.image_size),
        "--num-workers", str(args.num_workers),
        "--val-ratio", str(args.val_ratio),
        "--test-ratio", str(args.test_ratio),
        "--split-seed", str(args.split_seed),
        "--seed", str(seed),
        "--split-strategy", args.split_strategy,
        "--group-col", args.group_col,
        "--model-type", exp.model_type,
        "--loss-type", exp.loss_type,
        "--mixup-alpha", str(exp.mixup_alpha),
        "--cutout-p", str(exp.cutout_p),
        "--crop-scale-min", str(exp.crop_scale_min),
    ]

    if exp.model_type == "transformer":
        command.extend([
            "--backbone-stage", args.backbone_stage,
            "--transformer-pooling", exp.transformer_pooling,
            "--token-pool-size", str(args.token_pool_size),
        ])
    if exp.model_type == "metric":
        command.extend([
            "--metric-pooling", exp.metric_pooling,
            "--metric-head", exp.metric_head,
            "--arcface-scale", str(args.arcface_scale),
            "--arcface-margin", str(args.arcface_margin),
            "--arcface-subcenters", str(args.arcface_subcenters),
            "--head-lr-multiplier", str(args.head_lr_multiplier),
        ])
    if not exp.use_ema:
        command.append("--no-ema")
    if args.deterministic:
        command.append("--deterministic")
    command.extend(args.extra_args)
    return command


def main() -> None:
    args = parse_args()
    selected = get_suite(args.suite)
    if args.only:
        wanted = set(args.only)
        selected = [exp for exp in selected if exp.name in wanted]
        missing = wanted - {exp.name for exp in selected}
        if missing:
            raise ValueError(f"{args.suite} suite 中不存在实验: {sorted(missing)}")

    should_run = args.run and not args.dry_run
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"实验套件: {args.suite}, 配置数: {len(selected)}, seeds: {args.seeds}")
    print("模式:", "执行训练" if should_run else "仅打印命令")

    for exp in selected:
        for seed in args.seeds:
            output_dir = Path(args.output_root) / exp.name / f"seed_{seed}"
            if should_run and args.skip_completed and (output_dir / "metrics.json").exists():
                print(f"\n[{exp.name} seed={seed}] 已完成，跳过。")
                continue
            command = build_command(args, exp, seed)
            printable = " ".join(f'"{part}"' if " " in part else part for part in command)
            print(f"\n[{exp.name} seed={seed}]\n{printable}")
            if should_run:
                subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)

    if not should_run:
        print("\n确认命令无误后，加 --run 开始训练。")


if __name__ == "__main__":
    main()
