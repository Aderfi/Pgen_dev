"""Logging configuration for Pharmagen.

Call :func:`setup_logging` once at application start-up (``main.py``).
Subsequent ``logging.getLogger(__name__)`` calls in any module then inherit
the configured handlers and levels automatically.
"""

import logging
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

from src.config import get_settings


def setup_logging(
    name: str = "Pharmagen",
    level: int | None = None,
    console_level: int | None = None,
) -> logging.Logger:
    root = logging.getLogger()
    if root.handlers:
        return root  # already configured — idempotent

    log_file = get_settings().paths.logs / f"{name}_{datetime.now():%Y-%m-%d}.log"

    root.setLevel(logging.DEBUG)

    fmt_file = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fmt_console = logging.Formatter("%(message)s")

    file_handler = TimedRotatingFileHandler(
        log_file, when="midnight", backupCount=7, encoding="utf-8"
    )
    file_handler.setLevel(level or logging.WARNING)
    file_handler.setFormatter(fmt_file)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(console_level or level or logging.WARNING)
    stream_handler.setFormatter(fmt_console)

    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    for noisy_lib in ("matplotlib", "optuna", "numba"):
        logging.getLogger(noisy_lib).setLevel(logging.ERROR)

    return root
