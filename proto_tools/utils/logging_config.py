"""
proto_tools/utils/logging_config.py

Provides centralized logging setup with file and console handlers,
automatic log directory management, suppression of noisy third-party loggers,
and integration with Python's warnings system.
"""

import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


class BioToolsOnlyFilter(logging.Filter):
    """Filter to only allow logs from proto_tools project packages."""
    def filter(self, record):
        # Include logs from proto_tools and tests
        allowed_prefixes = ("proto_tools", "tests")
        return record.name.startswith(allowed_prefixes)


class SelectiveLevelFormatter(logging.Formatter):
    """Formatter that shows level prefix only for WARNING and above."""

    def format(self, record):
        # For WARNING, ERROR, CRITICAL: add level prefix
        if record.levelno >= logging.WARNING:
            # Save original format
            original_fmt = self._style._fmt
            # Temporarily use format with level
            self._style._fmt = "%(levelname)s: %(message)s"
            result = super().format(record)
            # Restore original format
            self._style._fmt = original_fmt
            return result
        else:
            # For DEBUG and INFO: just the message
            return record.getMessage()


def _parse_log_level(level: Union[int, str]) -> int:
    """
    Parse log level from string or int, case-insensitive.

    Args:
        level (int | str): Log level as int (e.g., logging.INFO) or string (e.g., "INFO", "info", "Info")

    Returns:
        int: Integer log level

    Raises:
        ValueError: If string level is not recognized
    """
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        level_upper = level.upper()
        if hasattr(logging, level_upper):
            return getattr(logging, level_upper)
        raise ValueError(
            f"Unknown log level: '{level}'. "
            f"Valid levels: DEBUG, INFO, WARNING, ERROR, CRITICAL"
        )
    raise TypeError(f"level must be int or str, got {type(level)}")


def setup_logging(
    level: Union[int, str] = logging.INFO,
    log_dir: Optional[str] = None,
    log_filename: Optional[str] = None,
    log_to_file: Optional[bool] = None,
    log_to_console: bool = True,
    console_level: Optional[Union[int, str]] = None,
    file_level: Optional[Union[int, str]] = None,
    console_output_formatted: bool = False,
    log_file_header: Optional[str] = None,
) -> None:
    """
    Configure logging for proto_tools.

    Args:
        level (int | str): Default logging level for all handlers. Can be an int (e.g., logging.INFO)
            or a case-insensitive string (e.g., "INFO", "info", "Debug"). Default: INFO.
        log_dir (str | None): Directory for log files. Defaults to logs/ in project root
            or PROTO_LOG_DIR environment variable.
        log_filename (str | None): Custom filename for the log file. If None, uses timestamped filename
            like proto_tools_YYYYMMDD_HHMMSS.log. If provided, uses this exact filename.
        log_to_file (bool | None): Whether to enable file logging. Default: None (auto-detect).
            When None, file logging is enabled only if a ``pyproject.toml`` is found
            in the directory tree (i.e., running from a development repo) or
            ``PROTO_LOG_DIR`` is set. Pass True/False to override.
        log_to_console (bool): Whether to enable console logging to stdout (default: True).
        console_level (int | str | None): Override level for console handler. Accepts int or string (default: uses `level`).
        file_level (int | str | None): Override level for file handler. Accepts int or string (default: DEBUG for full capture).
        console_output_formatted (bool): If True, console output includes timestamp and metadata like file logs.
            If False (default), console output is print-like: DEBUG/INFO show only the message,
            while WARNING/ERROR/CRITICAL include the level prefix (e.g., "WARNING: message").
        log_file_header (str | None): Optional header text to write at the top of the log file before logging starts.
    """
    # Parse levels (supports case-insensitive strings)
    level = _parse_log_level(level)
    if console_level is not None:
        console_level = _parse_log_level(console_level)
    if file_level is not None:
        file_level = _parse_log_level(file_level)

    # Resolve log_to_file auto-detect (None → check environment)
    if log_to_file is None:
        if os.environ.get("PYTEST_RUNNING") == "1" and log_filename is None:
            # pytest: disable unless explicitly requested with log_filename
            log_to_file = False
        elif os.environ.get("PROTO_LOG_DIR"):
            # User explicitly set a log dir; they want file logging
            log_to_file = True
        else:
            # Enable file logging only in a dev repo (pyproject.toml present)
            log_to_file = any(
                (parent / "pyproject.toml").exists()
                for parent in [Path.cwd()] + list(Path.cwd().parents)
            )

    # Determine log directory
    if log_dir is None:
        current = Path.cwd()
        project_root = current
        for parent in [current] + list(current.parents):
            if (parent / "pyproject.toml").exists():
                project_root = parent
                break
        log_dir = os.environ.get(
            "PROTO_LOG_DIR",
            str(project_root / "logs")
        )
    log_path = Path(log_dir)

    # Create log directory if needed
    if log_to_file:
        log_path.mkdir(parents=True, exist_ok=True)

    # Configure root logger for proto_tools
    root_logger = logging.getLogger("proto_tools")
    root_logger.setLevel(logging.DEBUG)  # Capture all, handlers filter

    # Clear any existing handlers to prevent duplicates on reconfiguration
    root_logger.handlers.clear()

    # File handler format (with line numbers for debugging)
    file_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler format (default: selective level display, or full formatted output)
    if console_output_formatted:
        console_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        console_formatter = logging.Formatter(console_format, datefmt="%Y-%m-%d %H:%M:%S")
    else:
        # Use SelectiveLevelFormatter: shows level prefix only for WARNING and above
        console_formatter = SelectiveLevelFormatter("%(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler (stdout for print-like behavior, works in Jupyter)
    console_handler = None
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level or level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handler (timestamped or custom filename)
    file_handler = None
    if log_to_file:
        if log_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_path / f"proto_tools_{timestamp}.log"
        else:
            log_file = log_path / log_filename

        # Write header to log file if provided
        if log_file_header:
            with open(log_file, 'w') as f:
                f.write(log_file_header)
            # Use append mode for FileHandler so we don't overwrite the header
            file_handler = logging.FileHandler(log_file, mode='a')
        else:
            # Use write mode as before
            file_handler = logging.FileHandler(log_file, mode='w')

        file_handler.setLevel(file_level or logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        # Add filter to only log proto_tools messages to file
        file_handler.addFilter(BioToolsOnlyFilter())
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    noisy_loggers = [
        "transformers",
        "vortex",
        "StripedHyena",
        "esm",
        "torch",
        "urllib3",
        "httpx",
        "matplotlib",
        "PIL",
        "h5py",
        "numba",
        "filelock",
        "huggingface_hub",
        "datasets",
        "accelerate",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Capture warnings through the logging system
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.setLevel(logging.WARNING)

    # Add handlers to py.warnings logger so warnings go to same destinations
    if log_to_console and console_handler:
        warnings_logger.addHandler(console_handler)
    if log_to_file and file_handler:
        warnings_logger.addHandler(file_handler)

    # Filter noisy third-party deprecation/future warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"transformers.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"huggingface_hub.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"huggingface_hub.*")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a proto_tools module.

    Args:
        name (str): Logger name, typically __name__ from the calling module.
            Will be prefixed with 'proto_tools.' if not already.

    Returns:
        logging.Logger: Configured Logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.debug("Detailed debug info")
    """
    if not name.startswith("proto_tools"):
        name = f"proto_tools.{name}"
    return logging.getLogger(name)
