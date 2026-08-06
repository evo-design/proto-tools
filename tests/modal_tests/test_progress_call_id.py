"""Replayed progress records name the call that produced them.

Both remote paths replay a worker's output through a module-level logger: ``device='modal'`` from
a Modal queue partition, ``device='proto'`` from the server's NDJSON log stream. A session with
one call in flight needs nothing more, and that is the case these were written for.

A process replaying for several callers at once is a different matter. Without a call id on the
record, a handler cannot tell two callers' output apart, and progress lines carry sequence names,
file names and tool parameters — so on a shared server that is a leak rather than a display bug.
"""

from __future__ import annotations

import logging

from proto_tools.modal.progress import CALL_ID_FIELD, replay_record


class _Capture(logging.Handler):
    """Collects records the way a consumer forwarding progress would."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _capture(logger_name: str) -> tuple[_Capture, logging.Logger]:
    handler = _Capture()
    logger = logging.getLogger(logger_name)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return handler, logger


def test_a_replayed_record_carries_its_call_id(monkeypatch):
    """The whole point: a consumer must be able to tell whose output a record is."""
    monkeypatch.setattr("proto_tools.modal.progress.has_active_progress_bar", lambda: False)
    handler, logger = _capture("proto_tools.modal.remote")
    try:
        replay_record({"m": "Running esmfold", "l": logging.INFO}, call_id="partition-abc")
    finally:
        logger.removeHandler(handler)

    assert len(handler.records) == 1
    assert getattr(handler.records[0], CALL_ID_FIELD) == "partition-abc"


def test_two_calls_are_distinguishable_on_one_logger(monkeypatch):
    """Records from concurrent callers share a logger, so the id is the only thing separating them."""
    monkeypatch.setattr("proto_tools.modal.progress.has_active_progress_bar", lambda: False)
    handler, logger = _capture("proto_tools.modal.remote")
    try:
        replay_record({"m": "alice's sequence", "l": logging.INFO}, call_id="alice")
        replay_record({"m": "bob's sequence", "l": logging.INFO}, call_id="bob")
    finally:
        logger.removeHandler(handler)

    by_call = {getattr(r, CALL_ID_FIELD): r.getMessage() for r in handler.records}
    assert by_call == {"alice": "alice's sequence", "bob": "bob's sequence"}


def test_a_local_session_still_works_without_one(monkeypatch):
    """A caller with one call in flight has nothing to attribute and must not be made to care."""
    monkeypatch.setattr("proto_tools.modal.progress.has_active_progress_bar", lambda: False)
    handler, logger = _capture("proto_tools.modal.remote")
    try:
        replay_record({"m": "Running esmfold", "l": logging.INFO})
    finally:
        logger.removeHandler(handler)

    assert len(handler.records) == 1
    assert getattr(handler.records[0], CALL_ID_FIELD) is None


def test_the_spinner_path_is_unchanged(monkeypatch):
    """With a bar on screen a record becomes its subtitle and never reaches the logger."""
    seen: list[str] = []
    monkeypatch.setattr("proto_tools.modal.progress.has_active_progress_bar", lambda: True)
    monkeypatch.setattr("proto_tools.modal.progress.update_active_substatus", seen.append)
    handler, logger = _capture("proto_tools.modal.remote")
    try:
        replay_record({"m": "Running esmfold", "l": logging.INFO}, call_id="partition-abc")
    finally:
        logger.removeHandler(handler)

    assert seen == ["Running esmfold"]
    assert not handler.records


def test_the_tailer_stamps_the_partition_it_is_reading(monkeypatch):
    """The id has to arrive without the caller passing one, since the tailer owns the partition."""
    import queue as queue_module
    import threading

    from proto_tools.modal import progress

    class _Queue:
        def __init__(self) -> None:
            self.served = False

        def get_many(self, *_args, **_kwargs):
            if self.served:
                raise queue_module.Empty
            self.served = True
            return [{"m": "Running esmfold", "l": logging.INFO}, {"t": "end"}]

    monkeypatch.setattr(progress, "open_progress_queue", lambda **_kwargs: _Queue())
    monkeypatch.setattr(progress, "has_active_progress_bar", lambda: False)
    handler, logger = _capture("proto_tools.modal.remote")
    try:
        progress.stream_modal_progress("partition-xyz", expected_ends=1, stop=threading.Event())
    finally:
        logger.removeHandler(handler)

    assert handler.records, "the tailer replayed nothing"
    assert getattr(handler.records[0], CALL_ID_FIELD) == "partition-xyz"


def test_both_remote_paths_agree_on_the_field_name():
    """One name, imported rather than duplicated, so the two paths cannot drift apart."""
    from proto_tools.proto import CALL_ID_FIELD as proto_field

    assert proto_field == CALL_ID_FIELD
