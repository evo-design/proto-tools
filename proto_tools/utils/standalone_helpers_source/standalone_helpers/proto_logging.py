r"""Subprocess-side logging bridge.

Lives inside ``standalone_helpers/`` so it reaches every tool's micromamba venv
via the ``PYTHONPATH`` published by ``_build_subprocess_env``. This is the
producer side of the proto_tools structured worker logging architecture.

How it fits together:

- The standalone subprocess emits log records via ``get_logger(name)``, which
  returns ``logging.getLogger(f"worker.{name}")``.
- ``install()`` installs ``ProtoLogger`` as the default logger class (so
  every ``logging.getLogger`` call returns one) and attaches a
  ``_BridgeHandler`` at the ``worker`` logger. The handler serializes records
  as ``\\x00LOG\\x00<json>\\n`` lines on ``sys.stderr``.
- The parent's ``PersistentWorker._drain_stderr`` (or the one-shot drain in
  ``ToolInstance._run_oneshot``) demultiplexes those lines, strips the
  ``worker.`` prefix, and re-emits each record under
  ``proto_tools.worker.{toolkit}.{name}`` on the parent side.

This module deliberately has zero dependencies on the rest of
``standalone_helpers``. ``standalone_helpers/__init__.py`` imports
``proto_logging`` first and calls ``install()`` (gated on ``TOOL_VENV_PATH``)
*before* loading the other submodules, so their module-level
``get_logger(__name__)`` calls go through the already-installed bridge.

The wire-format constant ``_TAG_PREFIX`` must match the parent-side constant
in ``proto_tools/utils/logging_config.py``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, cast

# Sentinel prefix on every JSON-tagged stderr line. Must match proto_tools/utils/logging_config.py:_TAG_PREFIX.
_TAG_PREFIX = "\x00LOG\x00"

# Logger namespace subprocess records emit under; bridge handler attaches here, parent strips this prefix on re-emit.
_WORKER_NAMESPACE = "worker"


class ProtoLogger(logging.Logger):
    """``logging.Logger`` subclass that accepts an ``update_status`` keyword.

    The stdlib ``logging`` API doesn't allow arbitrary kwargs on log methods.
    This subclass adds ``update_status`` by overriding ``_log`` and translating
    the flag into the standard ``extra`` dict so it lands as a ``LogRecord``
    attribute that handlers can read.

    Example:
        >>> from standalone_helpers import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Loading checkpoint", update_status=True)
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


# Seconds between the per-item statuses a StatusCounter reports, so a fast loop reads at human pace.
STATUS_INTERVAL_SECONDS = 5.0


class StatusCounter:
    """Report progress through a known number of items as status updates, throttled to human pace.

    The first item is reported at once, then at most one every ``interval`` seconds, and the last item always.
    A tight loop costs one clock read per item. Works with either logger that has ``update_status``: this
    module's (in a tool's environment) or proto-tools' own (a tool that runs in the caller's process).

    Example:
        >>> counter = StatusCounter(logger, "Folding structure %d/%d", total=len(sequences))
        >>> for index, sequence in enumerate(sequences, start=1):
        ...     counter(index)
    """

    def __init__(
        self, logger: logging.Logger, template: str, total: int, interval: float = STATUS_INTERVAL_SECONDS
    ) -> None:
        """Create a counter.

        Args:
            logger (logging.Logger): A logger with ``update_status`` (``get_logger(...)`` here, or proto-tools').
            template (str): The status, with ``%d`` for the item number and then the total, e.g.
                ``"Folding structure %d/%d"``.
            total (int): How many items there are.
            interval (float): Minimum seconds between reports, except the last.
        """
        self._logger = logger
        self._template = template
        self._total = total
        self._interval = interval
        self._last: float | None = None

    def __call__(self, number: int) -> None:
        """Report item ``number`` (1-based) of the total, when the throttle allows."""
        now = time.monotonic()
        if number < self._total and self._last is not None and now - self._last < self._interval:
            return
        self._last = now
        self._logger.update_status(self._template, number, self._total)  # type: ignore[attr-defined]


class _BridgeHandler(logging.Handler):
    r"""Serializes ``LogRecord``\s as tagged JSON lines on ``sys.stderr``.

    Attached at the ``worker`` logger by :func:`install`. The parent's drain
    thread recognizes the ``\x00LOG\x00`` sentinel and demuxes the JSON.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Write one tagged JSON line to ``sys.stderr``."""
        try:
            payload = {
                "level": record.levelname,
                "name": record.name,
                "msg": record.getMessage(),
                "update_status": getattr(record, "update_status", False),
            }
            sys.stderr.write(_TAG_PREFIX + json.dumps(payload) + "\n")
            sys.stderr.flush()
        except Exception:
            # Logging handlers must never raise; fall back to handleError.
            self.handleError(record)


def install() -> None:
    """Install the producer-side bridge in the current process.

    Called from ``standalone_helpers/__init__.py`` when ``TOOL_VENV_PATH`` is
    set (i.e. inside a tool subprocess spawned by the parent's worker
    machinery). Idempotent: replaces any prior :class:`_BridgeHandler` on the
    ``worker`` logger with a fresh instance.

    Effects:
        - Sets :class:`ProtoLogger` as the default logger class so every
          ``logging.getLogger`` call returns one (and thus accepts the
          ``update_status=True`` kwarg).
        - Attaches a fresh :class:`_BridgeHandler` to the ``worker`` logger.
        - Sets ``propagate=False`` on the ``worker`` logger so records don't
          double-emit through any host-configured root handler.
        - Sets the ``worker`` logger level to ``DEBUG`` so all records reach
          the bridge handler; level filtering is the parent's responsibility.
    """
    if logging.getLoggerClass() is not ProtoLogger:
        logging.setLoggerClass(ProtoLogger)

    worker_logger = logging.getLogger(_WORKER_NAMESPACE)
    # Idempotent re-install: drop any prior _BridgeHandler instance.
    worker_logger.handlers = [h for h in worker_logger.handlers if not isinstance(h, _BridgeHandler)]
    worker_logger.addHandler(_BridgeHandler())
    worker_logger.propagate = False
    worker_logger.setLevel(logging.DEBUG)


def get_logger(name: str) -> ProtoLogger:
    """Return a logger under the ``worker.*`` namespace bridged to the parent.

    Producer convention used by every standalone script and helper submodule:

        from standalone_helpers import get_logger
        logger = get_logger(__name__)
        logger.info("...")
        logger.info("Loading weights", update_status=True)  # spinner takeover

    Args:
        name (str): Typically ``__name__`` from the caller. Stored as the
            record's ``name`` attribute and visible to the parent.

    Returns:
        ProtoLogger: A logger under ``worker.{name}`` whose records are
            bridged to the parent process via :class:`_BridgeHandler`. It is a
            :class:`ProtoLogger` even before :func:`install` runs (e.g. from a
            setup script), so ``update_status`` always works; without the
            bridge it is an ordinary INFO record.
    """
    logger = logging.getLogger(f"{_WORKER_NAMESPACE}.{name}")
    if not hasattr(logger, "update_status"):
        # A plain Logger, made before install() or outside a worker. ProtoLogger adds methods but no
        # state, so switching this one instance's class is safe and leaves the global logger class alone.
        logger.__class__ = ProtoLogger
    return cast(ProtoLogger, logger)
