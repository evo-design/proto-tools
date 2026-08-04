"""tests/test_progress.py.

Live progress streaming from a Modal container, exercised against a fake queue.

The properties that matter are not "a line arrived" but the ones that protect the run: the tool
thread never formats or writes, a broken queue is absorbed, and the tailer always terminates.
"""

import logging
import threading
import time

import pytest

from proto_tools.modal.progress import (
    PARTITION_TTL_SECONDS,
    ProgressDrainer,
    QueueProgressHandler,
    container_progress,
    replay_record,
    stream_modal_progress,
)


class FakeQueue:
    """Records what a Modal Queue would have been asked to do."""

    def __init__(self, fail: bool = False):
        self.batches: list[tuple[list[dict], str]] = []
        self.singles: list[tuple[dict, str]] = []
        self.fail = fail

    def put_many(self, values, block=True, partition=None, partition_ttl=None):
        if self.fail:
            raise RuntimeError("queue unavailable")
        assert block is False, "a blocking write would stall the drainer behind the network"
        assert partition_ttl == PARTITION_TTL_SECONDS
        self.batches.append((values, partition))

    def put(self, value, block=True, partition=None, partition_ttl=None):
        if self.fail:
            raise RuntimeError("queue unavailable")
        self.singles.append((value, partition))

    def records(self) -> list[dict]:
        return [record for batch, _ in self.batches for record in batch]


def _record(message, args=(), level=logging.INFO, update_status=False) -> logging.LogRecord:
    record = logging.LogRecord("proto_tools.t", level, __file__, 1, message, args, None)
    record.update_status = update_status
    return record


# ============================================================================
# The performance contract
# ============================================================================
def test_emit_stores_the_record_unformatted():
    """Formatting on the tool thread is the cost this design exists to avoid."""
    handler = QueueProgressHandler()
    handler.emit(_record("folded %d of %d", (3, 10)))

    assert handler._buffer[0][1] == "folded %d of %d", "emit formatted the message on the tool thread"
    assert handler._buffer[0][2] == (3, 10)
    assert handler.drain()[0]["m"] == "folded 3 of 10", "drain must apply the formatting emit skipped"


def test_emit_never_raises_on_a_record_it_cannot_store():
    """A logging call that raises would surface inside the tool, which owns no error policy here."""
    handler = QueueProgressHandler()
    handler._buffer = None  # any failure inside emit, without depending on how one arises

    handler.emit(_record("anything"))  # must not raise


def test_a_bad_format_string_costs_one_line_not_the_batch():
    """A mismatched argument is the tool's bug; dropping the surrounding progress would be ours."""
    handler = QueueProgressHandler()
    handler.emit(_record("expects %d args", ("not-a-number", "extra")))
    handler.emit(_record("fine"))

    assert [record["m"] for record in handler.drain()] == ["expects %d args", "fine"]


def test_the_buffer_drops_the_oldest_rather_than_blocking():
    """Backpressure on a full buffer would push the queue's latency onto the tool."""
    handler = QueueProgressHandler(maxlen=3)
    for i in range(10):
        handler.emit(_record("line %d", (i,)))

    assert [record["m"] for record in handler.drain()] == ["line 7", "line 8", "line 9"]


def test_the_drainer_batches_many_records_into_few_writes():
    """One write per flush interval is what keeps log volume off the network."""
    handler = QueueProgressHandler()
    queue = FakeQueue()
    drainer = ProgressDrainer(handler, "p1", interval=10.0, open_queue=lambda: queue)
    for i in range(500):
        handler.emit(_record("line %d", (i,)))

    drainer.start()
    drainer.close()

    assert len(queue.batches) == 1, f"500 records took {len(queue.batches)} writes"
    assert drainer.records == 500


def test_the_tool_thread_does_no_network_work():
    """The queue is resolved on the drainer, so constructing progress cannot block the tool."""
    resolved = threading.Event()
    handler = QueueProgressHandler()

    def open_queue():
        resolved.set()
        return FakeQueue()

    drainer = ProgressDrainer(handler, "p1", interval=10.0, open_queue=open_queue)
    assert not resolved.is_set(), "the queue was resolved before the drainer thread started"

    drainer.start()
    drainer.close()
    assert resolved.is_set()


# ============================================================================
# Failure is absorbed, never propagated
# ============================================================================
def test_a_broken_queue_disables_progress_and_leaves_the_run_alone():
    """Progress is worth nothing if losing it can lose the result."""
    handler = QueueProgressHandler()
    queue = FakeQueue(fail=True)
    drainer = ProgressDrainer(handler, "p1", interval=0.01, open_queue=lambda: queue)
    handler.emit(_record("line"))

    drainer.start()
    drainer.close()  # must not raise

    assert drainer._disabled
    assert drainer.flushes == 0


def test_an_unreachable_queue_disables_progress():
    """A workspace where the queue cannot be opened still has to run tools."""
    drainer = ProgressDrainer(QueueProgressHandler(), "p1", interval=0.01, open_queue=lambda: 1 / 0)

    drainer.start()
    drainer.close()

    assert drainer._disabled


def test_container_progress_is_a_no_op_without_a_partition():
    """A caller who is not watching pays for nothing, which is the common case."""
    target = logging.getLogger("proto_tools")
    before = list(target.handlers)

    with container_progress(None):
        assert list(target.handlers) == before, "a handler was installed for a call nobody is tailing"

    assert list(target.handlers) == before


def test_container_progress_removes_its_handler_and_restores_the_level():
    """A handler left behind would stream a later call into a partition nobody reads."""
    target = logging.getLogger("proto_tools")
    before, prior_level = list(target.handlers), target.level

    with container_progress("p1", level=logging.DEBUG):
        assert len(target.handlers) == len(before) + 1
        assert target.level == logging.DEBUG

    assert list(target.handlers) == before
    assert target.level == prior_level


# ============================================================================
# The tailer terminates
# ============================================================================
def test_the_tailer_stops_when_the_caller_says_the_run_is_over():
    """A deployment too old to emit sends no sentinel, and must not leave a thread polling."""

    class SilentQueue:
        def get_many(self, n, block=True, timeout=None, partition=None):
            time.sleep(timeout or 0)
            return []

    stop = threading.Event()
    tailer = threading.Thread(
        target=stream_modal_progress,
        args=("p1", 1, stop),
        kwargs={"poll_timeout": 0.01},
        daemon=True,
    )
    import proto_tools.modal.progress as progress_module

    original = progress_module.open_progress_queue
    progress_module.open_progress_queue = lambda **_: SilentQueue()
    try:
        tailer.start()
        assert tailer.is_alive()
        stop.set()
        tailer.join(timeout=2.0)
    finally:
        progress_module.open_progress_queue = original

    assert not tailer.is_alive(), "the tailer ignored the stop signal and would poll forever"


def test_the_tailer_stops_on_one_sentinel_per_chunk(monkeypatch):
    """Fan-out shares a partition, so the stream closes only when every chunk has reported."""
    delivered = [
        [{"l": 20, "m": "chunk 0 running", "s": True}],
        [{"c": 0, "t": "end"}],
        [{"c": 1, "t": "end"}],
    ]

    class ScriptedQueue:
        def get_many(self, n, block=True, timeout=None, partition=None):
            return delivered.pop(0) if delivered else []

    monkeypatch.setattr("proto_tools.modal.progress.open_progress_queue", lambda **_: ScriptedQueue())
    seen: list[dict] = []
    stop = threading.Event()

    stream_modal_progress("p1", 2, stop, on_record=seen.append, poll_timeout=0.01)

    assert [record["m"] for record in seen] == ["chunk 0 running"]
    assert not delivered, "the tailer stopped before the second chunk's sentinel"


def test_a_transport_error_does_not_end_the_stream(monkeypatch):
    """A blip mid-run should cost a poll, not the rest of the run's progress."""
    calls = {"n": 0}

    class FlakyQueue:
        def get_many(self, n, block=True, timeout=None, partition=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient")
            return [{"l": 20, "m": "recovered", "s": False}, {"t": "end"}]

    monkeypatch.setattr("proto_tools.modal.progress.open_progress_queue", lambda **_: FlakyQueue())
    seen: list[dict] = []

    stream_modal_progress("p1", 1, threading.Event(), on_record=seen.append, poll_timeout=0.01)

    assert [record["m"] for record in seen] == ["recovered"]


# ============================================================================
# Replay drives the spinner the way a local run does
# ============================================================================
def test_replay_preserves_the_flag_the_spinner_keys_on(caplog):
    """Without update_status a streamed line logs but never moves the bar."""
    with caplog.at_level(logging.INFO, logger="proto_tools.modal.remote"):
        replay_record({"l": logging.INFO, "m": "folding 3/10", "s": True})
        replay_record({"l": logging.INFO, "m": "background detail", "s": False})

    assert [record.getMessage() for record in caplog.records] == ["folding 3/10", "background detail"]
    assert [record.update_status for record in caplog.records] == [True, False]


def test_replay_does_not_reformat_a_message_containing_percent_signs(caplog):
    """The container already formatted; a second pass would corrupt any literal % in the text."""
    with caplog.at_level(logging.INFO, logger="proto_tools.modal.remote"):
        replay_record({"l": logging.INFO, "m": "coverage 80% (2/2 chains)", "s": False})

    assert caplog.records[0].getMessage() == "coverage 80% (2/2 chains)"


@pytest.mark.parametrize("record", [{}, {"m": ""}, {"m": None}])
def test_replay_ignores_a_record_with_nothing_to_show(record, caplog):
    """A malformed record should be skipped, not turned into a blank spinner line."""
    with caplog.at_level(logging.INFO, logger="proto_tools.modal.remote"):
        replay_record(record)

    assert caplog.records == []


# ============================================================================
# Queue resolution
# ============================================================================
def test_the_client_hydrates_the_queue_it_asks_to_create(monkeypatch):
    """``from_name`` defers creation to first use, and the client never writes.

    Found on a real deployment: every worker wrote into a queue that did not exist yet, and the
    whole feature silently produced nothing. Only the client can create it, before dispatch.
    """
    import modal

    from proto_tools.modal.progress import open_progress_queue

    class Handle:
        def __init__(self):
            self.hydrated = False

        def hydrate(self):
            self.hydrated = True

    seen = {}

    def from_name(name, *, create_if_missing=False, **kwargs):
        seen["create_if_missing"] = create_if_missing
        return Handle()

    monkeypatch.setattr(modal.Queue, "from_name", staticmethod(from_name))

    assert open_progress_queue(create=True).hydrated, "the client left the queue uncreated"
    assert seen["create_if_missing"] is True
    assert not open_progress_queue().hydrated, "a container must not pay to hydrate a queue it may not use"


# ============================================================================
# The wait before a worker reports
# ============================================================================
def test_both_backends_name_themselves_while_connecting():
    """A cold start is seconds of silence, and the spinner should say which machine it waits on."""
    from proto_tools.proto import _PROTO_INITIAL_PHASE
    from proto_tools.utils.progress import remote_connecting_status

    assert remote_connecting_status("modal") == "Connecting to modal container"
    assert remote_connecting_status("proto") == _PROTO_INITIAL_PHASE


def test_the_modal_dispatch_says_so_before_it_dispatches(monkeypatch):
    """Set after the dispatch call would defeat the purpose, which is covering the wait."""
    import inspect

    from proto_tools.modal import client

    body = inspect.getsource(client._live_progress)
    said = body.index("remote_connecting_status")
    opened = body.index("open_progress_queue")

    assert said < opened, "the status is set after work that can itself block"
