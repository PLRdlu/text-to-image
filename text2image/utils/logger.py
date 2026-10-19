"""基于 loguru 的统一日志配置。"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_configured = False

_CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    """配置控制台 + 文件双通道日志（进程内只配置一次）。"""
    global _configured
    if _configured:
        return

    logger.remove()
    logger.add(sys.stderr, level=level, format=_CONSOLE_FORMAT, colorize=True)

    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "text2image_{time:YYYY-MM-DD}.log",
        level=level,
        rotation="10 MB",
        retention="14 days",
        encoding="utf-8",
    )
    _configured = True


def get_logger():
    """返回全局 loguru logger。"""
    return logger
