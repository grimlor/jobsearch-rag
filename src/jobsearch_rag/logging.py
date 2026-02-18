"""Logging configuration for jobsearch-rag.

Sets up standard logging to stderr with a consistent format.
Import the ``logger`` instance from this module throughout the codebase.

Call :func:`configure_file_logging` to add a timestamped file handler
under ``data/logs/`` for post-run diagnosis.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

# Create logger
logger = logging.getLogger("jobsearch-rag")
logger.setLevel(logging.INFO)

# Create handler to stderr
handler = logging.StreamHandler(sys.stderr)
handler.setLevel(logging.INFO)

# Create formatter (shared between stderr and file handlers)
_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)
handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(handler)

# Default log directory
DEFAULT_LOG_DIR = "data/logs"


def configure_file_logging(
    log_dir: str = DEFAULT_LOG_DIR,
    *,
    level: int = logging.INFO,
) -> logging.FileHandler:
    """Add a timestamped file handler to the logger.

    Creates ``log_dir`` if it does not exist.  Returns the handler so
    callers (or tests) can remove it later.

    Args:
        log_dir: Directory for log files.  Created automatically.
        level: Logging level for the file handler (default: INFO).

    Returns:
        The :class:`logging.FileHandler` that was added.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filename = log_path / f"jobsearch-rag_{timestamp}.log"

    file_handler = logging.FileHandler(str(filename), encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))

    # Ensure the logger captures messages at the lowest requested level
    if level < logger.level:
        logger.setLevel(level)

    logger.addHandler(file_handler)
    return file_handler


__all__ = ["configure_file_logging", "logger"]
