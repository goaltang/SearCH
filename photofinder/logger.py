"""Central logging setup for photofinder."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "photofinder.log"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure a rotating file handler + console handler for the package."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("photofinder")
    root.setLevel(level)

    # Avoid adding duplicate handlers when multiple modules import this.
    if root.handlers:
        return root

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(console_handler)
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``photofinder`` namespace."""
    return logging.getLogger(name)
