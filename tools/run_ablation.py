"""
标准消融实验启动脚本。

默认只打印命令，不直接训练；确认参数后加 --run 顺序执行。

示例:
    python tools/run_ablation.py --dry-run --epochs 3 --batch-size 4
    python tools/run_ablation.py --run --epochs 20 --batch-size 4 --backbone-stage layer4

训练完成后，每个实验会写入独立目录:
    outputs/ablations/<experiment_name>/

汇总表会自动追加到:
    outputs/ablations/ablation_results.csv
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
    """
    作用:
        描述一组消融实验配置。
    输入:
        name/model_type/loss_type/mixup_alpha/cutout_p/use_ema/transformer_pooling。
    输出:
        不直接输出；供 build_command 生成训练命令。
    """

    name: str
    model_type: str
    loss_type: str
    mixup_alpha: float
    cutout_p: float
    use_ema: bool
    transformer_pooling: str = "cls"


EXPERIMENTS = [
    AblationExperiment(
        name="01_resnet50_ce_plain",
        model_type="baseline",
        loss_type="ce",
        mixup_alpha=0.0,
        cutout_p=0.0,
        use_ema=False,
    ),
    AblationExperiment(
        name="02_resnet50_focal_plain",
        model_type="baseline",
        loss_type="focal",
        mixup_alpha=0.0,
        cutout_p=0.0,
        use_ema=False,
    ),
    AblationExperiment(
        name="03_resnet50_focal_mixup_cutout_ema",
        model_type="baseline",
        loss_type="focal",
        mixup_alpha=0.4,
        cutout_p=0.5,
        use_ema=True,
    ),
    AblationExperiment(
        name="04_transformer_mean_focal_mixup_cutout",
        model_type="transformer",
        loss_type="focal",
        mixup_alpha=0.4,
        cutout_p=0.5,
        use_ema=False,
        transformer_pooling="mean",
    ),
    AblationExperiment(
        name="05_transformer_cls_focal_mixup_cutout",
        model_type="transformer",
        loss_type="focal",
        mixup_alpha=0.4,
        cutout_p=0.5,
        use_ema=False,
        transformer_pooling="cls",
    ),
    AblationExperiment(
        name="06_transformer_cls_focal_mixup_cutout_ema",
        model_type="transformer",
        loss_type="focal",
        mixup_alpha=0.4,
        cutout_p=0.5,
        use_ema=True,
        transformer_pooling="cls",
    ),
]


def parse_args() -> argparse.Namespace:
    """
    作用:
        解析消融实验批量运行参数。
    输入:
        命令行参数，如 --run、--epochs、--only。
    输出:
        argparse.Namespace 参数对象。
    """
    parser = argparse.ArgumentParser(description="顺序运行 AI护鲸使者标准消融实验")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="真正开始训练")
    mode.add_argument("--dry-run", action="store_true", help="只打印命令，不训练")
    parser.add_argument("--data-root", default="archive", help="数据根目录")
    parser.add_argument("--output-root", default="outputs/ablations", help="消融实验输出根目录")
    parser.add_argument("--epochs", type=int, default=20, help="每个实验训练轮数")
    parser.add_argument("--batch-size", type=int, default=4, help="批大小")
    parser.add_argument("--image-size", type=int, default=512, help="输入图像尺寸")
    parser.add_argument("--num-workers", type=int, default=-1, help="DataLoader workers")
    parser.add_argument("--split-strategy", choices=("group", "stratified"), default="group", help="训练/验证划分策略")
    parser.add_argument("--group-col", default="individual_id", help="group 划分使用的分组列")
    parser.add_argument("--token-pool-size", type=int, default=16, help="Transformer token pooling 尺寸")
    parser.add_argument(
        "--backbone-stage",
        choices=("layer3", "layer4", "layer3_layer4"),
        default="layer3_layer4",
        help="Transformer backbone 输出层",
    )
    parser.add_argument("--python", default=sys.executable, help="Python 解释器路径")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="只运行指定实验名，例如 --only 01_resnet50_ce_plain 06_transformer_cls_focal_mixup_cutout_ema",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="追加传给 train.py 的参数，必须放在命令最后，例如 --extra-args --no-pretrained",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace, exp: AblationExperiment) -> List[str]:
    """
    作用:
        将一组消融实验配置转换为可执行的 train.py 命令列表。
    输入:
        args: run_ablation.py 的命令行参数。
        exp: 单个消融实验配置。
    输出:
        subprocess.run 可直接执行的命令参数列表。
    """
    output_dir = Path(args.output_root) / exp.name
    command = [
        args.python,
        str(PROJECT_ROOT / "train.py"),
        "--data-root",
        args.data_root,
        "--output-dir",
        str(output_dir),
        "--experiment-name",
        exp.name,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--image-size",
        str(args.image_size),
        "--num-workers",
        str(args.num_workers),
        "--split-strategy",
        args.split_strategy,
        "--group-col",
        args.group_col,
        "--model-type",
        exp.model_type,
        "--loss-type",
        exp.loss_type,
        "--mixup-alpha",
        str(exp.mixup_alpha),
        "--cutout-p",
        str(exp.cutout_p),
    ]

    if exp.model_type == "transformer":
        command.extend(
            [
                "--backbone-stage",
                args.backbone_stage,
                "--transformer-pooling",
                exp.transformer_pooling,
                "--token-pool-size",
                str(args.token_pool_size),
            ]
        )

    if not exp.use_ema:
        command.append("--no-ema")

    command.extend(args.extra_args)
    return command


def main() -> None:
    """
    作用:
        消融实验脚本主入口，按选择的实验列表打印或顺序执行训练命令。
    输入:
        无显式输入；通过命令行参数控制 dry-run 或 run。
    输出:
        无返回值；执行训练子进程或打印命令。
    """
    args = parse_args()
    selected = EXPERIMENTS
    if args.only:
        wanted = set(args.only)
        selected = [exp for exp in EXPERIMENTS if exp.name in wanted]
        missing = wanted - {exp.name for exp in selected}
        if missing:
            raise ValueError(f"未知实验名: {sorted(missing)}")

    should_run = args.run and not args.dry_run
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"实验数量: {len(selected)}")
    print("模式:", "执行训练" if should_run else "仅打印命令")

    for exp in selected:
        command = build_command(args, exp)
        printable = " ".join(f'"{part}"' if " " in part else part for part in command)
        print(f"\n[{exp.name}]")
        print(printable)
        if should_run:
            subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)

    if not should_run:
        print("\n确认命令无误后，加 --run 开始训练。")


if __name__ == "__main__":
    main()
