r"""Tests for ``standalone_helpers.proto_logging``, the subprocess-side bridge.

The module under test ships inside ``proto_tools/utils/standalone_helpers_source/``
and is normally only loaded inside a tool subprocess (where it's been copied
into the standalone dir). The tests load it directly from the source tree by
prepending the source dir to ``sys.path``.

Coverage:
    - ``ProtoLogger`` accepts ``update_status=True`` kwarg.
    - ``_BridgeHandler`` serializes records as ``\\x00LOG\\x00<json>`` lines.
    - ``install()`` is idempotent and only installs at the ``worker`` namespace
      so third-party loggers are not bridged.
    - ``get_logger(name)`` returns a logger named ``worker.{name}``.
"""

from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add standalone_helpers_source/ to sys.path so `from standalone_helpers.proto_logging` resolves in the parent test process.
_SRC_DIR = Path(__file__).resolve().parent.parent.parent / "proto_tools" / "utils" / "standalone_helpers_source"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from standalone_helpers.proto_logging import (  # noqa: E402
    _TAG_PREFIX,
    ProtoLogger,
    _BridgeHandler,
    get_logger,
    install,
)


@pytest.fixture(autouse=True)
def _restore_logging_state():
    """Snapshot the worker logger and global class so each test runs clean."""
    original_class = logging.getLoggerClass()
    worker_logger = logging.getLogger("worker")
    original_handlers = list(worker_logger.handlers)
    original_propagate = worker_logger.propagate
    original_level = worker_logger.level
    try:
        yield
    finally:
        logging.setLoggerClass(original_class)
        worker_logger.handlers = original_handlers
        worker_logger.propagate = original_propagate
        worker_logger.level = original_level


# ── get_logger naming ───────────────────────────────────────────────────────


def test_get_logger_returns_under_worker_namespace():
    """``get_logger("foo.bar")`` must return ``worker.foo.bar``."""
    logger = get_logger("foo.bar")
    assert logger.name == "worker.foo.bar"


# ── install() idempotency ───────────────────────────────────────────────────


def test_install_attaches_single_bridge_handler():
    """``install()`` must add exactly one ``_BridgeHandler`` to ``worker``."""
    install()
    worker = logging.getLogger("worker")
    bridges = [h for h in worker.handlers if isinstance(h, _BridgeHandler)]
    assert len(bridges) == 1


def test_install_is_idempotent():
    """Multiple ``install()`` calls must not stack handlers."""
    install()
    install()
    install()
    worker = logging.getLogger("worker")
    bridges = [h for h in worker.handlers if isinstance(h, _BridgeHandler)]
    assert len(bridges) == 1


def test_install_sets_proto_logger_class():
    """After ``install()``, new loggers must be ``ProtoLogger`` instances."""
    install()
    new_logger = logging.getLogger("worker.test.proto_class")
    assert isinstance(new_logger, ProtoLogger)


def test_install_sets_propagate_false_on_worker():
    """``worker`` must not propagate to root after install (no double-emit)."""
    install()
    assert logging.getLogger("worker").propagate is False


# ── ProtoLogger accepts update_status kwarg ────────────────────────────────


def test_log_record_carries_update_status_attribute():
    """``update_status=True`` must land as a ``LogRecord`` attribute."""
    install()
    captured: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    logger = get_logger("test.attribute")
    spy = _Capture()
    logger.addHandler(spy)
    try:
        logger.info("flagged", update_status=True)
        logger.info("plain")
    finally:
        logger.removeHandler(spy)
    assert len(captured) == 2
    assert getattr(captured[0], "update_status", False) is True
    assert getattr(captured[1], "update_status", False) is False


# ── _BridgeHandler tagged JSON emission ────────────────────────────────────


def test_bridge_handler_writes_tagged_json_to_stderr():
    """Every record reaching the bridge must produce one tagged JSON line; ``update_status`` defaults to ``false`` when the kwarg is absent."""
    install()
    fake_stderr = io.StringIO()
    logger = get_logger("test.bridge.basic")
    with patch("sys.stderr", fake_stderr):
        logger.info("loaded checkpoint v3", update_status=True)
        logger.info("plain message")
    flagged_line, plain_line = fake_stderr.getvalue().splitlines()
    assert flagged_line.startswith(_TAG_PREFIX)
    flagged = json.loads(flagged_line[len(_TAG_PREFIX) :])
    assert flagged == {
        "level": "INFO",
        "name": "worker.test.bridge.basic",
        "msg": "loaded checkpoint v3",
        "update_status": True,
    }
    plain = json.loads(plain_line[len(_TAG_PREFIX) :])
    assert plain["update_status"] is False


# ── Namespace isolation: third-party loggers are NOT bridged ───────────────


def test_third_party_loggers_are_not_bridged():
    """``transformers``, ``urllib3``, etc. must bypass the bridge handler.

    They sit on root, not under ``worker``. Their records reach root's
    default handlers (or get dropped) but never become tagged JSON.
    """
    install()
    fake_stderr = io.StringIO()
    foreign_logger = logging.getLogger("transformers.foo")
    with patch("sys.stderr", fake_stderr):
        foreign_logger.warning("third-party warning, do not bridge")
    assert _TAG_PREFIX not in fake_stderr.getvalue()


def test_tag_prefix_matches_parent_constant():
    """The wire-format constant must equal the parent-side counterpart."""
    from proto_tools.utils.logging_config import _TAG_PREFIX as parent_tag

    assert parent_tag == _TAG_PREFIX


# ── Producer / parent ProtoLogger parity ───────────────────────────────────
# The two ProtoLogger classes are duplicated by design (subprocess side must be
# stdlib-only, parent side lives with the rest of logging_config). These tests
# lock in the contract so a future change to one side must update the other.


def test_proto_logger_log_signature_matches_parent_side():
    """``_log`` override must take identical params on both sides.

    Failure means the kwarg surface diverged - update both copies.
    """
    import inspect

    from proto_tools.utils.logging_config import ProtoLogger as ParentProtoLogger

    parent_sig = inspect.signature(ParentProtoLogger._log)
    producer_sig = inspect.signature(ProtoLogger._log)
    assert parent_sig.parameters == producer_sig.parameters, (
        f"ProtoLogger._log signature divergence:\n  parent:   {parent_sig}\n  producer: {producer_sig}"
    )


class _CaptureSpy(logging.Handler):
    """Test handler that appends every record into a caller-supplied list."""

    def __init__(self, sink: list[logging.LogRecord]) -> None:
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self._sink.append(record)


def test_proto_logger_update_status_behavior_matches_parent_side():
    """Both ProtoLogger classes must translate ``update_status=True`` into a record attribute identically."""
    from proto_tools.utils.logging_config import ProtoLogger as ParentProtoLogger

    for cls in (ParentProtoLogger, ProtoLogger):
        captured: list[logging.LogRecord] = []
        logger = cls(f"parity.{cls.__module__}")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(_CaptureSpy(captured))
        logger.info("flagged", update_status=True)
        logger.info("plain")
        assert len(captured) == 2, f"{cls!r} emitted {len(captured)} records, expected 2"
        assert getattr(captured[0], "update_status", False) is True, f"{cls!r} dropped update_status=True"
        assert getattr(captured[1], "update_status", False) is False, (
            f"{cls!r} leaked update_status onto an unflagged record"
        )


# ── update_status() method ─────────────────────────────────────────────────


def test_update_status_emits_record_with_flag_set():
    """``logger.update_status("foo")`` must produce one INFO record with update_status=True."""
    install()
    captured: list[logging.LogRecord] = []
    logger = get_logger("test.update_status_method")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(_CaptureSpy(captured))
    logger.update_status("loading checkpoint")
    assert len(captured) == 1
    assert captured[0].levelno == logging.INFO
    assert captured[0].getMessage() == "loading checkpoint"
    assert getattr(captured[0], "update_status", False) is True


def test_update_status_method_parity_between_proto_logger_copies():
    """Producer and parent ProtoLogger must implement update_status identically."""
    from proto_tools.utils.logging_config import ProtoLogger as ParentProtoLogger

    for cls in (ParentProtoLogger, ProtoLogger):
        captured: list[logging.LogRecord] = []
        logger = cls(f"parity_method.{cls.__module__}")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(_CaptureSpy(captured))
        logger.update_status("phase A")
        assert len(captured) == 1, f"{cls!r}.update_status emitted {len(captured)} records, expected 1"
        assert getattr(captured[0], "update_status", False) is True, f"{cls!r}.update_status did not flag the record"


def test_drop_update_status_filter_is_bar_aware():
    """The console filter drops update_status records only while a bar shows them; plain records always pass."""
    from proto_tools.utils.logging_config import _drop_update_status_records

    flagged = logging.LogRecord(
        name="proto_tools.x",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg="status",
        args=(),
        exc_info=None,
    )
    flagged.update_status = True
    plain = logging.LogRecord(
        name="proto_tools.x",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg="info",
        args=(),
        exc_info=None,
    )

    # Bar active: the bar renders the flagged record, so the console drops it.
    with patch("proto_tools.utils.progress.has_active_progress_bar", return_value=True):
        assert _drop_update_status_records(flagged) is False
        assert _drop_update_status_records(plain) is True

    # No bar: nothing else is showing the phase, so keep it visible (the no-bar set_substatus fallback).
    with patch("proto_tools.utils.progress.has_active_progress_bar", return_value=False):
        assert _drop_update_status_records(flagged) is True
        assert _drop_update_status_records(plain) is True
