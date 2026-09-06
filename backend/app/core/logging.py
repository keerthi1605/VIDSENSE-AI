"""
Basic logging setup.

Why not just use `print()`?
  - Logging lets us attach severity levels (INFO/WARNING/ERROR), which
    matters once background video processing runs unattended (Phase 8+)
    and we need to find failures in a log file rather than a terminal
    that's long gone.
  - `uvicorn` already configures its own loggers; we piggyback on the
    root logger config so our own log lines look consistent with its
    request logs.

This is intentionally minimal for Phase 1. It will likely grow (file
handlers, log rotation) once background jobs exist.
"""

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
