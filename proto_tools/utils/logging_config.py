r"""proto_tools/utils/logging_config.py.

Parent-side logging setup for proto_tools.

- **Library auto-configure**: adds a stderr console handler if none exists,
  silences noisy third-party loggers.
- **Spinner takeover**: :class:`SpinnerFromLogsHandler` is attached to the
  ``proto_tools`` namespace by :func:`install_spinner_handler` so any record
  flagged ``update_status=True`` (from parent-side wrapper code or re-emitted
  from a worker subprocess) updates the active spinner subtitle.
- **Logger class**: :func:`install_logger_class` registers :class:`ProtoLogger`
  globally so every ``logging.getLogger("proto_tools.X")`` accepts the
  ``update_status=True`` kwarg.

The subprocess-side bridge lives in
``standalone_helpers/proto_logging.py`` (copied into each tool's micromamba
venv at worker startup). It writes ``\\x00LOG\\x00<json>\\n`` lines on stderr;
:data:`_TAG_PREFIX` here is the parent-side counterpart and must match the
producer constant. The drain thread in
``proto_tools/utils/persistent_worker.py`` demultiplexes those lines and
re-emits them under ``proto_tools.worker.{toolkit}.{name}``.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

# NullHandler prevents "No handlers found" warnings when proto_tools is imported as a library without explicit logging config.
logging.getLogger("proto_tools").addHandler(logging.NullHandler())


# ============================================================================
# Noisy third-party loggers (shared between setup_logging and auto-configure)
# ============================================================================
_NOISY_LOGGERS = [
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


def _suppress_noisy_loggers() -> None:
    """Suppress noisy third-party loggers to WARNING level."""
    for logger_name in _NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


# ============================================================================
# Auto-configure: lazy, one-shot logging setup for standalone users
# ============================================================================
_state: dict[str, bool] = {"auto_configured": False}


def _auto_configure_logging() -> None:
    """Auto-configure minimal console logging if no handlers are set up.

    Called lazily on first tool invocation via the ``@tool`` wrapper.
    Adds a stderr handler at INFO level to the ``proto_tools`` logger
    only if no explicit handlers exist anywhere.

    This ensures "just works" output for standalone users (print-like
    INFO messages on stderr) while staying out of the way for library
    consumers who configure their own logging.
    """
    if _state["auto_configured"]:
        return
    _state["auto_configured"] = True

    pt_logger = logging.getLogger("proto_tools")

    # SpinnerFromLogsHandler doesn't emit normal console output, so its presence shouldn't suppress auto-configure of a stderr handler.
    real_handlers = [h for h in pt_logger.handlers if not isinstance(h, (logging.NullHandler, SpinnerFromLogsHandler))]
    if real_handlers:
        return

    # Root logger already configured downstream - our messages propagate via NullHandler.
    if logging.getLogger().handlers:
        return

    # No one configured logging - add minimal stderr handler
    handler = _BarAwareStreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    handler.setFormatter(SelectiveLevelFormatter("%(message)s"))
    handler.addFilter(_drop_update_status_records)
    pt_logger.addHandler(handler)
    pt_logger.setLevel(logging.DEBUG)

    _suppress_noisy_loggers()


def _drop_update_status_records(record: logging.LogRecord) -> bool:
    """Filter for console handlers: drop ``update_status`` records only while a bar shows them.

    These records are spinner-subtitle updates; while a progress bar is active it
    renders them (and SpinnerFromLogsHandler routes them there), so echoing them to
    the console would just duplicate the bar. With no active bar there's nothing
    displaying them, so keep them visible — this is the no-bar ``set_substatus``
    fallback path, and dropping it would silently swallow phase messages locally.
    File handlers don't install this filter, so they capture everything as an audit trail.
    """
    if not getattr(record, "update_status", False):
        return True
    from proto_tools.utils.progress import has_active_progress_bar

    return not has_active_progress_bar()


class _BarAwareStreamHandler(logging.StreamHandler):  # type: ignore[type-arg]
    r"""StreamHandler that routes through ``tqdm.write`` when a spinner is active.

    Plain ``StreamHandler.emit`` writes ``msg + terminator`` directly to the
    stream. When ``_AnimatedProgressBar`` is repainting the same stderr line
    via ``\r`` (verbose >= 2, DEBUG records streaming), the log line appends
    to the spinner frame and the next frame leaves the prior one stranded
    above — ghost frames accumulate. ``tqdm.write`` clears the bar line first,
    writes the message, then redraws the bar so log lines and the spinner
    don't collide.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Lazy import: progress -> tqdm is heavier and ``logging_config`` is
        # imported very early (proto_tools package init).
        from proto_tools.utils.progress import has_active_progress_bar

        if has_active_progress_bar():
            try:
                from tqdm import tqdm

                tqdm.write(self.format(record), file=self.stream)
                return
            except Exception:
                self.handleError(record)
                return
        super().emit(record)


# ============================================================================
# Formatting
# ============================================================================
class SelectiveLevelFormatter(logging.Formatter):
    """Formatter that shows level prefix only for WARNING and above."""

    def format(self, record: Any) -> Any:
        """Format a log record with colored level and structured output."""
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
        # For DEBUG and INFO: just the message
        return record.getMessage()


# ============================================================================
# Helpers
# ============================================================================
def _parse_log_level(level: int | str) -> int:
    """Parse log level from string or int, case-insensitive.

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
            return getattr(logging, level_upper)  # type: ignore[no-any-return]
        raise ValueError(f"Unknown log level: '{level}'. Valid levels: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    raise TypeError(f"level must be int or str, got {type(level)}")


# ============================================================================
# Explicit logging setup (power users)
# ============================================================================
def setup_logging(
    level: int | str = logging.INFO,
    log_dir: str | None = None,
    log_filename: str | None = None,
    log_to_file: bool | None = None,
    log_to_console: bool = True,
    console_level: int | str | None = None,
    file_level: int | str | None = None,
    console_output_formatted: bool = False,
    log_file_header: str | None = None,
) -> None:
    """Configure logging for proto_tools.

    Calling this function marks logging as explicitly configured, preventing
    the auto-configure logic from adding its own handler.

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
        log_to_console (bool): Whether to enable console logging to stderr (default: True).
        console_level (int | str | None): Override level for console handler. Accepts int or string (default: uses `level`).
        file_level (int | str | None): Override level for file handler. Accepts int or string (default: DEBUG for full capture).
        console_output_formatted (bool): If True, console output includes timestamp and metadata like file logs.
            If False (default), console output is print-like: DEBUG/INFO show only the message,
            while WARNING/ERROR/CRITICAL include the level prefix (e.g., "WARNING: message").
        log_file_header (str | None): Optional header text to write at the top of the log file before logging starts.
    """
    # Mark as explicitly configured - prevents auto-configure from running
    _state["auto_configured"] = True

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
                (parent / "pyproject.toml").exists() for parent in [Path.cwd(), *list(Path.cwd().parents)]
            )

    # Determine log directory
    if log_dir is None:
        current = Path.cwd()
        project_root = current
        for parent in [current, *list(current.parents)]:
            if (parent / "pyproject.toml").exists():
                project_root = parent
                break
        log_dir = os.environ.get("PROTO_LOG_DIR", str(project_root / "logs"))
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

    # Console handler (stderr - safe from subprocess IPC on stdout)
    console_handler = None
    if log_to_console:
        console_handler = _BarAwareStreamHandler(sys.stderr)
        console_handler.setLevel(console_level or level)
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(_drop_update_status_records)
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
            with open(log_file, "w") as f:
                f.write(log_file_header)
            # Use append mode for FileHandler so we don't overwrite the header
            file_handler = logging.FileHandler(log_file, mode="a")
        else:
            # Use write mode as before
            file_handler = logging.FileHandler(log_file, mode="w")

        file_handler.setLevel(file_level or logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        # Handler is attached to the proto_tools logger, which only sees
        # proto_tools.* records via propagation, so no extra filter is needed.
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    _suppress_noisy_loggers()

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

    # handlers.clear() above dropped the spinner handler installed at import; re-attach it so
    # update_status records (local worker-bridged or cloud-replayed) still reach the active bar.
    install_spinner_handler()


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a proto_tools module.

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


# ============================================================================
# Structured worker logging: ProtoLogger, spinner handler
# ============================================================================

# Sentinel that prefixes every JSON-tagged stderr line emitted by the producer
# side (``standalone_helpers.proto_logging._BridgeHandler``). Picked to be
# vanishingly unlikely to appear in free-form text output (NUL byte + literal
# "LOG" + NUL byte). Must match the producer constant.
_TAG_PREFIX = "\x00LOG\x00"


class ProtoLogger(logging.Logger):
    """``logging.Logger`` subclass that accepts an ``update_status`` keyword.

    The standard ``logging`` API doesn't allow arbitrary kwargs on log methods;
    only ``exc_info``, ``extra``, ``stack_info``, ``stacklevel`` are honored.
    This subclass adds ``update_status`` by overriding ``_log`` (which all
    public methods forward kwargs to) and translating the flag into the
    standard ``extra`` dict so it lands as a ``LogRecord`` attribute.

    Example:
        >>> logger = logging.getLogger("proto_tools.bioemu.sampler")
        >>> logger.info("Sampling chain A", update_status=True)
    """

    def _log(  # type: ignore[override]
        self,
        level: int,
        msg: object,
        args: Any,
        exc_info: Any = None,
        extra: dict[str, Any] | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        update_status: bool = False,
    ) -> None:
        """Override of ``Logger._log`` that recognizes ``update_status``."""
        if update_status:
            extra = {**(extra or {}), "update_status": True}
        super()._log(
            level,
            msg,
            args,
            exc_info=exc_info,
            extra=extra,
            stack_info=stack_info,
            stacklevel=stacklevel,
        )

    def update_status(self, msg: object, *args: Any) -> None:
        """Emit a status record that updates the spinner subtitle and is captured by file handlers but never shown on console.

        Use this for phase-transition messages inside long-running tools
        ("Loading weights", "Moving to GPU", "Folding 1 complex"). The console
        StreamHandler installs a Filter that drops these records, so they never
        clutter terminal output regardless of verbose level. File handlers
        (configured via ``setup_logging(log_to_file=True)``) still capture
        them as a complete audit trail.
        """
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, msg, args, update_status=True)


class SpinnerFromLogsHandler(logging.Handler):
    """Routes log records flagged ``update_status=True`` to the active spinner subtitle.

    Attached to the ``proto_tools`` namespace on the parent. Sees both the
    re-emitted records from worker subprocesses (under
    ``proto_tools.worker.*``) and direct records from parent-side wrapper
    code (under ``proto_tools.tools.*``, ``proto_tools.utils.*``, etc.) since
    both propagate up to ``proto_tools``. It updates the bar via
    ``update_active_substatus`` (not ``set_substatus``) so a flagged record never
    loops back through the no-bar logging fallback; it no-ops when no spinner is
    active, so the handler is safe to leave permanently attached.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Update the active spinner subtitle if the record is flagged."""
        if not getattr(record, "update_status", False):
            return
        try:
            from proto_tools.utils.progress import update_active_substatus

            update_active_substatus(record.getMessage())
        except Exception:
            self.handleError(record)


def install_logger_class() -> None:
    """Make new loggers ``ProtoLogger`` instances; idempotent.

    Called from ``proto_tools/__init__.py`` so any subsequent
    ``logging.getLogger("proto_tools.X")`` returns a ``ProtoLogger`` and the
    ``update_status=True`` kwarg works on every log call.
    """
    if logging.getLoggerClass() is not ProtoLogger:
        logging.setLoggerClass(ProtoLogger)


def install_spinner_handler() -> None:
    """Attach :class:`SpinnerFromLogsHandler` to the ``proto_tools`` namespace.

    Called on the parent side (when ``TOOL_VENV_PATH`` is unset) so that any
    ``logger.info(..., update_status=True)`` call from either parent-side
    wrapper code or a re-emitted worker record updates the active spinner
    subtitle. Idempotent.

    Also bumps the ``proto_tools`` logger level so INFO records flow through;
    without this, root's default WARNING would drop INFO before the handler
    runs. The handler itself filters on ``update_status`` so non-flagged
    records still don't touch the spinner.
    """
    pt_logger = logging.getLogger("proto_tools")
    if not any(isinstance(h, SpinnerFromLogsHandler) for h in pt_logger.handlers):
        pt_logger.addHandler(SpinnerFromLogsHandler())
    if pt_logger.level == logging.NOTSET or pt_logger.level > logging.INFO:
        pt_logger.setLevel(logging.INFO)


def verbose_level_from_env() -> int:
    """Read ``PROTO_WORKER_VERBOSE`` as an int on the 0-3 scale.

    Returns 0 when unset or unparsable. Clamped to ``[0, 3]`` so the consumer
    can rely on the value as a direct level comparator.
    """
    raw = os.environ.get("PROTO_WORKER_VERBOSE", "0")
    try:
        level = int(raw)
    except (TypeError, ValueError):
        return 0
    return max(0, min(level, 3))
