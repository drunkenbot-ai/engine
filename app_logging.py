from __future__ import annotations

import logging
import faulthandler
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import threading
import traceback
from typing import Optional


DEFAULT_LOG_DIR = Path.home() / ".drunkenbot_ide" / "logs"
DEFAULT_LOG_PATH = DEFAULT_LOG_DIR / "drunkenbot_ide.log"
_FAULT_LOG_HANDLE: Optional[object] = None
_CONFIGURED_PATH: Optional[Path] = None


def setup_logging(log_path: Optional[Path] = None) -> Path:
    """Configure console and rotating file logging for the desktop app.

    Args:
        log_path: Optional explicit log file path.

    Returns:
        Path to the active log file.
    """

    global _CONFIGURED_PATH
    active_path = log_path or DEFAULT_LOG_PATH
    active_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if not any(getattr(handler, "_micro_llm_handler", False) for handler in root_logger.handlers):
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        file_handler = RotatingFileHandler(active_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler._micro_llm_handler = True  # type: ignore[attr-defined]
        root_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler._micro_llm_handler = True  # type: ignore[attr-defined]
        root_logger.addHandler(console_handler)

    logging.captureWarnings(True)
    sys.excepthook = _log_uncaught_exception
    threading.excepthook = _log_thread_exception
    _enable_fault_logging(active_path)
    for logger_name in ("datasets", "huggingface_hub", "urllib3", "filelock", "pyarrow"):
        logging.getLogger(logger_name).setLevel(logging.INFO)
    if _CONFIGURED_PATH is None:
        _CONFIGURED_PATH = active_path
        logging.getLogger(__name__).info("Logging initialized: %s", active_path)
    return active_path


def qt_message_handler(mode: object, context: object, message: str) -> None:
    """Route Qt runtime messages to the app log.

    Args:
        mode: Qt message type.
        context: Qt message context.
        message: Message text.
    """

    logger = logging.getLogger("qt")
    file_name = getattr(context, "file", "") or ""
    line = getattr(context, "line", 0) or 0
    location = f" ({file_name}:{line})" if file_name else ""
    logger.warning("%s%s", message, location)


def _log_uncaught_exception(exc_type: type[BaseException], exc_value: BaseException, exc_tb: object) -> None:
    """Log exceptions that reach the Python top level.

    Args:
        exc_type: Exception type.
        exc_value: Exception instance.
        exc_tb: Traceback object.
    """

    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    formatted = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    logging.getLogger(__name__).critical("Uncaught exception:\n%s", formatted)


def _log_thread_exception(args: threading.ExceptHookArgs) -> None:
    """Log uncaught exceptions from Python threads.

    Args:
        args: Thread exception hook arguments.
    """

    if args.exc_type is None or args.exc_value is None:
        return
    formatted = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
    logging.getLogger(__name__).critical("Uncaught thread exception in %s:\n%s", args.thread.name if args.thread else "-", formatted)


def _enable_fault_logging(active_path: Path) -> None:
    """Enable crash dumps for native faults when possible.

    Args:
        active_path: Main application log path.
    """

    global _FAULT_LOG_HANDLE
    if _FAULT_LOG_HANDLE is not None:
        return
    fault_path = active_path.with_name("drunkenbot_ide_faults.log")
    _FAULT_LOG_HANDLE = fault_path.open("a", encoding="utf-8")
    faulthandler.enable(file=_FAULT_LOG_HANDLE, all_threads=True)

