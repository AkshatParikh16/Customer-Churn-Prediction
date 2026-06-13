"""
src/churn/utils/logging.py
───────────────────────────
Centralised Loguru setup.
Import `setup_logging()` once at each entrypoint (train.py, main.py, etc.).
"""

from __future__ import annotations

import sys
from loguru import logger


def setup_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure Loguru for the current process."""
    logger.remove()  # remove default handler

    fmt = (
        '{"time":"{time:YYYY-MM-DDTHH:mm:ss}", "level":"{level}", "msg":"{message}"}'
        if json_logs
        else (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
        )
    )

    logger.add(sys.stderr, format=fmt, level=level, colorize=not json_logs)
    logger.add(
        "logs/churn_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="14 days",
        compression="gz",
        format=fmt,
        level="DEBUG",
        enqueue=True,  # thread-safe
    )
