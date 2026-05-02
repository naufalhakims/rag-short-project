"""
Logging configuration — structured logging for retrieval hits, errors, and general events.
"""

import logging
import sys


def setup_logging() -> logging.Logger:
    """Configure and return the application-wide logger."""

    logger = logging.getLogger("pocket_librarian")
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers on reload
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # File handler for persistent logs
    file_handler = logging.FileHandler("app.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()
