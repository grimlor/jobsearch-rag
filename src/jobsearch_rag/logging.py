"""Logging configuration for jobsearch-rag.

Sets up standard logging to stderr with a consistent format.
Import the ``logger`` instance from this module throughout the codebase.
"""

import logging
import sys

# Create logger
logger = logging.getLogger("jobsearch-rag")
logger.setLevel(logging.INFO)

# Create handler to stderr
handler = logging.StreamHandler(sys.stderr)
handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(handler)

__all__ = ["logger"]
