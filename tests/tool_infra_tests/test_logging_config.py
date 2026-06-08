"""Tests for the parent-side structured worker-logging plumbing in ``logging_config``.

Covers ``ProtoLogger`` (``update_status`` kwarg), the spinner handler
(``set_substatus`` from flagged records), and the install entry points. The
producer-side bridge handler lives in ``standalone_helpers/proto_logging.py``
and is exercised by ``test_proto_logging.py``.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from proto_tools.utils.logging_config import (
    ProtoLogger,
    SpinnerFromLogsHandler,
    install_logger_class,
    install_spinner_handler,
    verbose_level_from_env,
)

# ── ProtoLogger / update_status kwarg ──────────────────────────────────────


def test_proto_logger_kwarg_does_not_corrupt_extra():
    """``update_status=True`` must compose with an explicit ``extra`` dict."""
    install_logger_class()
    logger = logging.getLogger("proto_tools.test.compose_extra")
    captured: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    logger.addHandler(_Capture())
    logger.setLevel(logging.DEBUG)
    try:
        logger.info("msg", extra={"foo": 1}, update_status=True)
    finally:
        logger.handlers.clear()

    record = captured[0]
    assert getattr(record, "update_status", False) is True
    assert getattr(record, "foo", None) == 1


# ── SpinnerFromLogsHandler: only fires for flagged records ─────────────────


def test_spinner_handler_calls_set_substatus_on_flagged_record():
    """A record with ``update_status=True`` should reach ``set_substatus``."""
    handler = SpinnerFromLogsHandler()
    record = logging.LogRecord(
        name="proto_tools.x",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg="step A",
        args=(),
        exc_info=None,
    )
    record.update_status = True

    with patch("proto_tools.utils.progress.set_substatus") as mock_set:
        handler.emit(record)
        mock_set.assert_called_once_with("step A")


def test_spinner_handler_skips_unflagged_record():
    """Records without the flag should not call ``set_substatus``."""
    handler = SpinnerFromLogsHandler()
    record = logging.LogRecord(
        name="proto_tools.x",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg="plain",
        args=(),
        exc_info=None,
    )

    with patch("proto_tools.utils.progress.set_substatus") as mock_set:
        handler.emit(record)
        mock_set.assert_not_called()


# ── install_* idempotency + scoping ────────────────────────────────────────


def test_install_logger_class_sets_proto_logger_class():
    install_logger_class()
    install_logger_class()
    assert logging.getLoggerClass() is ProtoLogger


def test_auto_configure_runs_when_only_spinner_handler_attached(monkeypatch: pytest.MonkeyPatch):
    """``SpinnerFromLogsHandler`` should not block ``_auto_configure_logging``.

    Regression: in the parent-mode install path, the spinner handler is attached
    before any tool runs. ``_auto_configure_logging`` originally treated any
    non-NullHandler as "real" and exited early, leaving the proto_tools logger
    without a stderr StreamHandler — silently swallowing all WARNING/INFO output.
    """
    from proto_tools.utils.logging_config import _auto_configure_logging, _state

    # Simulate parent-mode startup: only NullHandler + SpinnerFromLogsHandler on
    # the proto_tools logger, and no handlers on root. The latter mirrors a
    # vanilla Python startup; without isolation, pytest's caplog plugin attaches
    # a root handler that would also short-circuit _auto_configure_logging.
    pt = logging.getLogger("proto_tools")
    saved_pt_handlers = list(pt.handlers)
    saved_root_handlers = list(logging.getLogger().handlers)
    pt.handlers = [logging.NullHandler(), SpinnerFromLogsHandler()]
    logging.getLogger().handlers = []
    _state["auto_configured"] = False

    try:
        _auto_configure_logging()
        # Auto-configure should add a stderr StreamHandler (incl. the bar-aware subclass) despite the spinner.
        stream_handlers = [h for h in pt.handlers if isinstance(h, logging.StreamHandler)]
        assert stream_handlers, (
            f"_auto_configure_logging skipped StreamHandler install; handlers={[type(h).__name__ for h in pt.handlers]}"
        )
    finally:
        pt.handlers = saved_pt_handlers
        logging.getLogger().handlers = saved_root_handlers
        _state["auto_configured"] = False


def test_install_spinner_handler_idempotent():
    install_spinner_handler()
    install_spinner_handler()
    pt = logging.getLogger("proto_tools")
    spinners = [h for h in pt.handlers if isinstance(h, SpinnerFromLogsHandler)]
    assert len(spinners) == 1
    pt.handlers = [h for h in pt.handlers if not isinstance(h, SpinnerFromLogsHandler)]


# ── verbose_level_from_env ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "env_value, expected",
    [
        ("0", 0),
        ("1", 1),
        ("2", 2),
        ("3", 3),
        ("4", 3),  # clamped
        ("-1", 0),  # clamped
        ("not_an_int", 0),  # malformed
        ("", 0),  # empty
    ],
)
def test_verbose_level_from_env_parses_and_clamps(monkeypatch: pytest.MonkeyPatch, env_value: str, expected: int):
    monkeypatch.setenv("PROTO_WORKER_VERBOSE", env_value)
    assert verbose_level_from_env() == expected


def test_verbose_level_from_env_unset_defaults_to_zero(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PROTO_WORKER_VERBOSE", raising=False)
    assert verbose_level_from_env() == 0
