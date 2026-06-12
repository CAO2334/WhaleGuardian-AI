"""
整理 outputs 根目录下散落的报告和图片，复制到 outputs/reports/archive_<timestamp>/。

默认只复制，不删除原文件，避免影响当前打开的文件或已有脚本路径。

示例:
    python tools/organize_outputs.py
    python tools/organize_outputs.py --move
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.report_paths import timestamp

REPORT_PATTERNS = {
    "ablation": ["ablation_report.md", "ablation_macro_f1.png"],
    "evaluation": ["confusion_matrix.png"],
    "interpretability": ["cam_result.jpg", "attention_result.jpg"],
}


def parse_args() -> argparse.Namespace:
    """
    作用:
        解析 outputs 整理脚本参数。
    输入:
        命令行参数，主要是 --move。
    输出:
        argparse.Namespace 参数对象。
    """
    parser = argparse.ArgumentParser(description="整理 outputs 中散落的报告和图片")
    parser.add_argument("--move", action="store_true", help="移动文件而不是复制文件；默认只复制，较安全")
    parser.add_argument("--output-root", default="outputs/reports", help="归档根目录")
    return parser.parse_args()


def collect_existing_files(outputs_dir: Path) -> Dict[str, List[Path]]:
    """
    作用:
        收集 outputs 根目录下已存在的报告/图片文件。
    输入:
        outputs_dir: outputs 目录路径。
    输出:
        {类别名: 文件路径列表} 字典。
    """
    collected: Dict[str, List[Path]] = {}
    for group, names in REPORT_PATTERNS.items():
        files = []
        for name in names:
            path = outputs_dir / name
            if path.exists():
                files.append(path)
        if files:
            collected[group] = files
    return collected


def organize_files(collected: Dict[str, List[Path]], archive_dir: Path, move: bool) -> int:
    """
    作用:
        将收集到的文件复制或移动到归档目录。
    输入:
        collected: collect_existing_files 返回的文件字典。
        archive_dir: 本次归档目录。
        move: True 表示移动，False 表示复制。
    输出:
        实际整理的文件数量。
    """
    count = 0
    for group, files in collected.items():
        target_dir = archive_dir / group
        target_dir.mkdir(parents=True, exist_ok=True)
        for src in files:
            dst = target_dir / src.name
            if move:
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(src, dst)
            count += 1
            action = "移动" if move else "复制"
            print(f"{action}: {src.relative_to(PROJECT_ROOT)} -> {dst.relative_to(PROJECT_ROOT)}")
    return count


def main() -> None:
    """
    作用:
        整理脚本主入口，按类别归档 outputs 根目录下的报告/图片。
    输入:
        无显式输入；通过命令行参数控制复制或移动。
    输出:
        无返回值；生成 outputs/reports/archive_<timestamp>/ 目录。
    """
    args = parse_args()
    outputs_dir = PROJECT_ROOT / "outputs"
    archive_dir = PROJECT_ROOT / args.output_root / f"archive_{timestamp()}"
    collected = collect_existing_files(outputs_dir)
    if not collected:
        print("没有发现需要整理的散落报告/图片。")
        return
    count = organize_files(collected, archive_dir, move=args.move)
    print(f"整理完成，共处理 {count} 个文件。归档目录: {archive_dir.resolve()}")


if __name__ == "__main__":
    main()
