"""
pipeline_logging.py
-------------------
Shared logging setup for the combined pipeline.

Sets up a logger that writes to both stdout (INFO+) and a timestamped file
in the logs/ directory.

Usage:
    from pipeline_logging import setup_logging, get_logger

    setup_logging()          # call once at pipeline start
    log = get_logger(__name__)
    log.info("Hello")
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path


_PIPELINE_LOGGER_NAME = "pipeline"
_FILE_HANDLER: logging.FileHandler | None = None


def setup_logging(
    logs_dir: Path | None = None,
    level: int = logging.INFO,
) -> Path:
    """Configure pipeline logging.

    Parameters
    ----------
    logs_dir : Path, optional
        Directory to write the log file to. If None, uses the logs/
        directory relative to this file's parent.
    level : int
        Root logging level. Default: INFO.

    Returns
    -------
    Path
        Path to the log file created.
    """
    global _FILE_HANDLER

    if logs_dir is None:
        logs_dir = Path(__file__).resolve().parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"pipeline_{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any existing handlers to avoid duplicates on re-init
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler
    _FILE_HANDLER = logging.FileHandler(log_path, encoding="utf-8")
    _FILE_HANDLER.setLevel(logging.DEBUG)   # file always gets DEBUG+
    _FILE_HANDLER.setFormatter(fmt)
    root.addHandler(_FILE_HANDLER)

    return log_path


def get_logger(name: str) -> logging.Logger:
    """Return a named child of the pipeline logger."""
    return logging.getLogger(name)


class StepTimer:
    """Context manager that logs the elapsed time of a named step."""

    def __init__(self, step_name: str, logger: logging.Logger | None = None):
        self.step_name = step_name
        self.logger = logger or logging.getLogger(_PIPELINE_LOGGER_NAME)
        self._start: float = 0.0

    def __enter__(self) -> "StepTimer":
        self._start = time.time()
        self.logger.info("  -> Starting: %s", self.step_name)
        return self

    def __exit__(self, *_) -> None:
        elapsed = time.time() - self._start
        self.logger.info("  <- Completed: %s (%.1fs)", self.step_name, elapsed)

    @property
    def elapsed(self) -> float:
        return time.time() - self._start
