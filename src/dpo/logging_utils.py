"""Shared logging setup for DPO pipeline modules.

Provides a `setup_logger()` helper that configures both file and stdout
handlers, writes to `./output/09_cliff_dpo/logs/{name}_{timestamp}.log` by default.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

DEFAULT_LOG_DIR = "./output/09_cliff_dpo/logs"


def setup_logger(
    name: str,
    log_dir: str = DEFAULT_LOG_DIR,
    level: int = logging.INFO,
    console: bool = True,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure a named logger with file + stdout handlers.

    Args:
        name: Logger name (e.g. "step1_rollout_gsm8k"). Also used in the log filename.
        log_dir: Directory where the log file will be created.
        level: Logging level.
        console: Whether to also log to stdout.
        log_file: Explicit log file path (overrides auto-generated one).
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = str(Path(log_dir) / f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # prevent duplicate handlers on re-init

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if console:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    logger.propagate = False
    logger.info(f"Logging to {log_file}")
    return logger


def parse_log_level(level_str: str) -> int:
    """Convert 'INFO'/'DEBUG'/... string to logging constant."""
    return getattr(logging, level_str.upper(), logging.INFO)
