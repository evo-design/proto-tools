"""Live progress streaming from a Modal container to the caller's spinner.

Holds the container-side logging handler and drainer, and the client-side queue tailer.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import queue as queue_module
import threading
from collections import deque
from collections.abc import Callable, Iterator
from typing import Any

from proto_tools.utils.progress import has_active_progress_bar, update_active_substatus

logger = logging.getLogger(__name__)

# One persistent queue per Modal environment, partitioned per tool call. A partition is an
# independent FIFO inside the queue, which lets every call stream through the same object without a
# per-call queue to create and tear down.
PROGRESS_QUEUE_NAME = "proto-tools-progress"

# Partitions expire this long after their last write, so a client that died mid-run leaves nothing
# behind. Well above any tool's runtime, and far below the 24-hour default.
PARTITION_TTL_SECONDS = 6 * 60 * 60

# Seconds between drainer flushes. The container makes at most one write per interval no matter how
# much the tool logs.
FLUSH_INTERVAL_SECONDS = 0.3

# Records buffered in the container before the oldest are dropped. Progress is lossy by design.
BUFFER_LIMIT = 2000

_END = "end"

# Records replay through the ``proto_tools`` namespace, where SpinnerFromLogsHandler is attached, so
# a streamed line drives the bar exactly as the same line does in a local run.
_remote_logger = logging.getLogger("proto_tools.modal.remote")

# Attribute naming the call a replayed record came from. One call at a time can ignore it; a
# process replaying for several callers cannot tell their output apart without it.
CALL_ID_FIELD = "proto_call_id"


# ============================================================================
# Container side
# ============================================================================
class QueueProgressHandler(logging.Handler):
    """Buffer log records for a :class:`ProgressDrainer` to batch to a Modal Queue.

    ``emit`` runs on the tool's own thread and therefore never blocks, formats, or raises. It
    performs one bounded-deque append of the raw record. Formatting is deferred to :meth:`drain`,
    which the drainer thread calls, and that deferral is what keeps the cost off the tool.

    A dropped record costs nothing beyond a missing progress line, whereas a slow ``emit`` would
    cost the run, which is why the buffer discards the oldest rather than applying backpressure.
    """

    def __init__(self, chunk_id: int | None = None, *, maxlen: int = BUFFER_LIMIT, level: int = logging.INFO) -> None:
        """Buffer up to ``maxlen`` records at ``level`` and above, tagged with ``chunk_id``."""
        super().__init__(level=level)
        self._buffer: deque[tuple[int, Any, Any, bool]] = deque(maxlen=maxlen)
        self._chunk_id = chunk_id

    def emit(self, record: logging.LogRecord) -> None:
        """Append the unformatted record to the buffer, swallowing any failure."""
        try:  # noqa: SIM105 - contextlib.suppress builds an object per call; this is the tool's hot path
            self._buffer.append((record.levelno, record.msg, record.args, getattr(record, "update_status", False)))
        except Exception:  # noqa: S110 - logging must never break the tool
            pass

    def drain(self) -> list[dict[str, Any]]:
        """Pop every buffered record oldest first, applying %-formatting off the tool thread.

        Returns:
            list[dict[str, Any]]: Wire records, each ``{"c", "l", "m", "s"}``.
        """
        drained: list[dict[str, Any]] = []
        buffer = self._buffer
        while True:
            try:
                levelno, message, args, update_status = buffer.popleft()
            except IndexError:
                return drained
            try:
                # Deferred from emit. An argument mutated in between renders its later value, which
                # is acceptable for progress and is the reason emit stays O(1).
                text = str(message) % args if args else str(message)
            except Exception:
                text = str(message)
            drained.append({"c": self._chunk_id, "l": levelno, "m": text, "s": update_status})


class ProgressDrainer(threading.Thread):
    """Daemon that flushes a :class:`QueueProgressHandler` to a Modal Queue partition, throttled.

    Every ``interval`` seconds it batch-writes whatever accumulated, which holds the container to
    roughly one network write per interval regardless of log volume. The queue itself is resolved on
    this thread rather than at construction, so the tool thread performs no network work at all.

    Best-effort throughout. The first queue error disables writing for the remainder of the run and
    the tool continues untouched.

    The stop Event is ``_stopping`` rather than ``_stop`` because ``threading.Thread`` defines its
    own ``_stop()`` and shadowing it crashes the thread at teardown.
    """

    def __init__(
        self,
        handler: QueueProgressHandler,
        partition: str,
        *,
        chunk_id: int | None = None,
        interval: float = FLUSH_INTERVAL_SECONDS,
        open_queue: Callable[[], Any] | None = None,
    ) -> None:
        """Drain ``handler`` into ``partition`` every ``interval`` seconds until stopped."""
        super().__init__(daemon=True, name=f"proto-tools-progress-{chunk_id if chunk_id is not None else 'single'}")
        self._handler = handler
        self._partition = partition
        self._chunk_id = chunk_id
        self._interval = interval
        self._open_queue = open_queue or open_progress_queue
        self._queue: Any = None
        self._stopping = threading.Event()
        self._disabled = False
        self.flushes = 0
        self.records = 0

    def run(self) -> None:
        """Resolve the queue, then flush on the interval until stopped, then flush the remainder."""
        try:
            self._queue = self._open_queue()
        except Exception:
            self._disabled = True
        if self._queue is None:
            self._disabled = True
        while not self._stopping.is_set():
            self._flush()
            self._stopping.wait(self._interval)
        self._flush()

    def _flush(self) -> None:
        """Write one batch, disabling the drainer permanently on the first failure."""
        if self._disabled:
            return
        batch = self._handler.drain()
        if not batch:
            return
        try:
            self._queue.put_many(batch, block=False, partition=self._partition, partition_ttl=PARTITION_TTL_SECONDS)
            self.flushes += 1
            self.records += len(batch)
        except Exception:
            self._disabled = True

    def close(self, *, send_end: bool = True) -> None:
        """Stop the drainer, flush what remains, and write the end-of-stream sentinel."""
        self._stopping.set()
        self.join(timeout=2.0)
        if send_end and not self._disabled:
            with contextlib.suppress(Exception):
                self._queue.put(
                    {"c": self._chunk_id, "t": _END},
                    block=False,
                    partition=self._partition,
                    partition_ttl=PARTITION_TTL_SECONDS,
                )


@contextlib.contextmanager
def container_progress(
    partition: str | None,
    chunk_id: int | None = None,
    *,
    level: int = logging.INFO,
    logger_name: str = "proto_tools",
    interval: float = FLUSH_INTERVAL_SECONDS,
) -> Iterator[None]:
    """Stream ``logging`` output to ``partition`` for the duration of the block.

    A no-op when ``partition`` is None, which lets a caller pass an optional value through
    unconditionally. Installs a non-blocking handler on ``logger_name`` at ``level`` and above plus
    a throttled drainer, and on exit removes the handler, flushes, and closes the stream.

    Args:
        partition (str | None): Queue partition identifying this call, or None to disable.
        chunk_id (int | None): Fan-out chunk index tagging these records, None for a single call.
        level (int): Minimum level captured.
        logger_name (str): Logger to attach to, scoped so unrelated framework output stays out.
        interval (float): Seconds between flushes.
    """
    if not partition:
        yield
        return
    handler = QueueProgressHandler(chunk_id=chunk_id, level=level)
    drainer = ProgressDrainer(handler, partition, chunk_id=chunk_id, interval=interval)
    target = logging.getLogger(logger_name)
    prior_level = target.level
    if target.level == logging.NOTSET or target.level > level:
        target.setLevel(level)  # otherwise records at `level` never reach the handler
    target.addHandler(handler)
    drainer.start()
    try:
        yield
    finally:
        target.removeHandler(handler)
        target.setLevel(prior_level)
        drainer.close()


# ============================================================================
# Queue resolution, used from both sides
# ============================================================================
def open_progress_queue(*, create: bool = False, environment: str | None = None, client: Any | None = None) -> Any:
    """Return the shared progress queue for one Modal environment.

    There is one queue per environment, so the writer and the reader must name the same one or
    the container fills a queue nobody is tailing. A container leaves both arguments unset,
    because ambient resolution inside the workspace is already correct.

    Args:
        create (bool): Create the queue when it does not exist. The client passes True; a container
            never creates one, because a partition nobody reads is wasted work.
        environment (str | None): Modal environment holding the queue, or ``None`` for the ambient one.
        client (Any | None): Modal client to open as, or ``None`` for the process's own.

    Returns:
        Any: A ``modal.Queue``.
    """
    import modal

    handle = modal.Queue.from_name(
        PROGRESS_QUEUE_NAME, environment_name=environment, client=client, create_if_missing=create
    )
    if create:
        # ``from_name`` is lazy and defers creation to first use, so the client hydrates here rather
        # than leaving the workers to race a queue that does not exist yet.
        handle.hydrate()
    return handle


# ============================================================================
# Client side
# ============================================================================
def stream_modal_progress(
    partition: str,
    expected_ends: int,
    stop: threading.Event,
    *,
    environment: str | None = None,
    client: Any | None = None,
    on_record: Callable[[dict[str, Any]], None] | None = None,
    batch: int = 64,
    poll_timeout: float = 0.25,
) -> None:
    """Tail ``partition`` and replay each record locally until the run finishes.

    Runs on a daemon thread while the main thread blocks on the tool result. ``get_many`` blocks
    until data arrives or ``poll_timeout`` elapses, so this waits rather than busy-polls.

    Exits on whichever comes first, ``expected_ends`` end sentinels (one per dispatched chunk) or
    ``stop`` being set by the main thread once the result is in hand. The second condition is what
    guarantees termination against a deployment too old to emit anything.

    Args:
        partition (str): Queue partition for this call.
        expected_ends (int): End sentinels to wait for.
        stop (threading.Event): Set by the caller when the result has returned.
        environment (str | None): Modal environment holding the queue. Must match the one the
            dispatch resolves in, or this tails an empty queue of the same name.
        client (Any | None): Modal client to tail as, or ``None`` for the process's own.
        on_record (Callable[[dict[str, Any]], None] | None): Record consumer, defaults to local replay.
        batch (int): Records to request per poll.
        poll_timeout (float): Seconds to block per poll.
    """
    # The partition identifies this call, so it rides along on every record.
    consume = on_record or functools.partial(replay_record, call_id=partition)
    try:
        progress_queue = open_progress_queue(environment=environment, client=client)
    except Exception:
        logger.debug("progress queue unavailable; live updates disabled", exc_info=True)
        return
    ends = 0
    while ends < expected_ends and not stop.is_set():
        try:
            items = progress_queue.get_many(batch, block=True, timeout=poll_timeout, partition=partition)
        except queue_module.Empty:
            continue
        except Exception:
            stop.wait(poll_timeout)
            continue
        for record in items or []:
            if isinstance(record, dict) and record.get("t") == _END:
                ends += 1
            else:
                with contextlib.suppress(Exception):
                    consume(record)


def replay_record(record: dict[str, Any], call_id: str | None = None) -> None:
    """Re-emit one streamed record locally, into the spinner when there is one.

    With a bar on screen every record becomes its subtitle, so a long call reads as one line
    that keeps changing rather than a bar with output scrolling past it. Without a bar, records
    go to the logger as they would on the ``device='proto'`` path, carrying the ``update_status``
    flag so the same line behaves the same way.

    Args:
        record (dict[str, Any]): One streamed record.
        call_id (str | None): Which call produced it, stamped on as :data:`CALL_ID_FIELD`.
    """
    message = record.get("m")
    if not message:
        return
    if has_active_progress_bar():
        update_active_substatus(message)
        return
    _remote_logger.log(
        int(record.get("l", logging.INFO)),
        "%s",
        message,
        extra={"update_status": bool(record.get("s", False)), CALL_ID_FIELD: call_id},
    )
