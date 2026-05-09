"""Test fixtures shared by ``tests/tool_infra_tests/``.

Tests in this directory exercise the structured worker logging machinery,
which mutates global state: ``logging.setLoggerClass`` (process-wide), the
``proto_tools`` logger handlers, and the ``worker`` logger handlers. Without
isolation, a test that calls ``install()`` could poison later tests in the
same pytest session.

The autouse fixture below snapshots and restores the relevant logger state
around every test in this directory.
"""

from __future__ import annotations

import logging
from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def _restore_logger_state() -> Generator[None, None, None]:
    """Snapshot/restore ``getLoggerClass`` and the ``proto_tools``/``worker`` handlers."""
    original_class = logging.getLoggerClass()
    pt_logger = logging.getLogger("proto_tools")
    worker_logger = logging.getLogger("worker")
    pt_handlers = list(pt_logger.handlers)
    worker_handlers = list(worker_logger.handlers)
    pt_propagate = pt_logger.propagate
    worker_propagate = worker_logger.propagate
    pt_level = pt_logger.level
    worker_level = worker_logger.level
    try:
        yield
    finally:
        logging.setLoggerClass(original_class)
        pt_logger.handlers = pt_handlers
        worker_logger.handlers = worker_handlers
        pt_logger.propagate = pt_propagate
        worker_logger.propagate = worker_propagate
        pt_logger.level = pt_level
        worker_logger.level = worker_level


@pytest.fixture
def capture_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set PROTO_CAPTURE_ERRORS=1 for tests that exercise the capture-mode path."""
    monkeypatch.setenv("PROTO_CAPTURE_ERRORS", "1")
