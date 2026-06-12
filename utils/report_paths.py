from __future__ import annotations

from datetime import datetime
from pathlib import Path


def timestamp() -> str:
    """
    作用:
        生成适合文件名使用的时间戳。
    输入:
        无。
    输出:
        形如 20260420-143012 的字符串。
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def timestamped_path(directory: Path, stem: str, suffix: str) -> Path:
    """
    作用:
        构造带时间戳的输出路径，避免报告和图片被覆盖。
    输入:
        directory: 输出目录。
        stem: 文件名前缀。
        suffix: 文件后缀，例如 .png、.jpg、.md。
    输出:
        带时间戳的 Path。
    """
    return directory / f"{stem}_{timestamp()}{suffix}"
