from __future__ import annotations

import logging
import warnings
from logging import FileHandler, StreamHandler
from pathlib import Path
from datetime import datetime
import os
import sys

ROOT_LOGGER_NAME = "traj_lib"
ROOT_DIR = Path(__file__).resolve().parent.parent 
LOG_DIR  = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

MAX_LOG_FILES_PER_PAIR = 5
COLOR_ENABLED = (
    os.environ.get("NO_COLOR") is None and
    os.environ.get("TRAJLIB_COLOR", "1").lower() not in {"0", "false", "no"}
)

_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FILE = LOG_DIR / f"benchmark-{_ts}-{os.getpid()}.log"
model_name: str | None = None
dataset_name: str | None = None


class _Ansi:
    RED = "[31m"
    BOLD = "[1m"
    RESET = "[0m"


def _stream_supports_color(stream) -> bool:
    try:
        return hasattr(stream, "isatty") and stream.isatty()
    except Exception:
        return False


class ErrorRedFormatter(logging.Formatter):
    def __init__(self, fmt: str, datefmt: str | None = None, enable_color: bool = True, stream=None) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.enable_color = bool(enable_color and _stream_supports_color(stream or sys.stderr))

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if self.enable_color and record.levelno >= logging.ERROR:
            if record.levelno >= logging.CRITICAL:
                return f"{_Ansi.BOLD}{_Ansi.RED}{msg}{_Ansi.RESET}"
            return f"{_Ansi.RED}{msg}{_Ansi.RESET}"
        return msg


def _prune_old_logs_for_pair(model: str | None, dataset: str | None, keep: int = MAX_LOG_FILES_PER_PAIR) -> None:
    if not model or not dataset:
        return

    pattern = f"{model}-{dataset}-*.log"
    try:
        files = sorted(
            LOG_DIR.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except Exception as e:
        warnings.warn(f"List log files failed for pattern {pattern}: {e}")
        return

    for old in files[keep:]:
        try:
            old.unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            warnings.warn(f"Failed to remove old log file {old}: {e}")



def set_model_name(name: str) -> None:
    global model_name
    model_name = name


def set_dataset_name(name: str) -> None:
    global dataset_name
    dataset_name = name


def set_log_file_name() -> None:
    global LOG_FILE
    LOG_FILE = LOG_DIR / f"{model_name}-{dataset_name}-{_ts}-{os.getpid()}.log"
    _prune_old_logs_for_pair(model_name, dataset_name, MAX_LOG_FILES_PER_PAIR)



def _configure_root_logger() -> logging.Logger:
    root = logging.getLogger(ROOT_LOGGER_NAME)
    if root.handlers:
        return root

    root.setLevel(logging.DEBUG)

    root.propagate = False

    _prune_old_logs_for_pair(model_name, dataset_name, MAX_LOG_FILES_PER_PAIR)

    console = StreamHandler()
    console.setLevel(logging.INFO)

    console_fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    console_datefmt = "%H:%M:%S"

    if COLOR_ENABLED:
        console.setFormatter(ErrorRedFormatter(console_fmt, console_datefmt, enable_color=True, stream=getattr(console, 'stream', None)))
    else:
        console.setFormatter(logging.Formatter(console_fmt, console_datefmt))

    file = FileHandler(LOG_FILE, encoding="utf-8")
    file.setLevel(logging.DEBUG)
    file.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S"))

    root.addHandler(console)
    root.addHandler(file)

    logging.captureWarnings(True)
    warnings.filterwarnings("default") 

    return root



def get_logger(name: str | None = None,
               level: int = logging.DEBUG) -> logging.Logger:
    _configure_root_logger()

    def _qualify(n: str | None) -> str:
        if not n:
            return ROOT_LOGGER_NAME
        if n == ROOT_LOGGER_NAME or n.startswith(f"{ROOT_LOGGER_NAME}."):
            return n
        return f"{ROOT_LOGGER_NAME}.{n}"

    qualified_name = _qualify(name)
    lg = logging.getLogger(qualified_name)
    lg.setLevel(level)
    lg.handlers.clear()
    lg.propagate = True
    return lg
