"""Run-specific logging helpers."""

from __future__ import annotations

import logging
from pathlib import Path


def create_run_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"bodybuilder.run.{id(path)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger
