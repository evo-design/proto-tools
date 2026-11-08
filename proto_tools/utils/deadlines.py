"""Ambient wall-clock deadline for a tool call.

The registry opens a :func:`deadline` from ``config.effective_timeout()`` around every tool
call. Code below it never takes a timeout argument: the shared HTTP session
(:mod:`proto_tools.utils.http_session`), :func:`~proto_tools.utils.polling.poll_until_complete`
and :func:`sleep_within_deadline` read the deadline from a ``ContextVar`` and raise
:class:`DeadlineExceededError` once it passes. This is what bounds tools that run in the caller's
process, where no subprocess can be killed when the timeout elapses.

A ``ContextVar`` is not copied into ``ThreadPoolExecutor`` workers. Wrap work submitted to an
executor with :func:`propagate`.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")

_DEADLINE: ContextVar[float | None] = ContextVar("proto_tools_deadline", default=None)


class DeadlineExceededError(TimeoutError):
    """The tool call's ``config.timeout`` elapsed."""


@contextmanager
def deadline(seconds: float | None) -> Iterator[None]:
    """Bound everything inside the block to ``seconds`` from now.

    A nested deadline never extends an enclosing one: the earlier of the two applies.

    Args:
        seconds (float | None): Budget in seconds. ``None`` leaves any enclosing deadline as is.
    """
    if seconds is None:
        yield
        return
    with _at(time.monotonic() + seconds):
        yield


@contextmanager
def _at(when: float | None) -> Iterator[None]:
    """Apply the absolute monotonic deadline ``when``, keeping an earlier enclosing one."""
    current = _DEADLINE.get()
    if when is None or (current is not None and current <= when):
        yield
        return
    token = _DEADLINE.set(when)
    try:
        yield
    finally:
        _DEADLINE.reset(token)


def remaining() -> float | None:
    """Return the seconds left before the deadline (negative once passed), or ``None`` without one."""
    when = _DEADLINE.get()
    return None if when is None else when - time.monotonic()


def expired() -> bool:
    """Return True when a deadline is set and has passed."""
    left = remaining()
    return left is not None and left <= 0


def check_deadline(what: str) -> None:
    """Raise :class:`DeadlineExceededError` naming ``what`` when the deadline has passed."""
    if expired():
        raise DeadlineExceededError(f"Timed out {what}: the tool's timeout elapsed")


def sleep_within_deadline(seconds: float, what: str) -> None:
    """Sleep ``seconds``, or until the deadline and then raise when the wait would run past it.

    The call never ends before its deadline: a wait that does not fit sleeps out the time left
    first, since whatever follows the wait could not start in time anyway.

    Raises:
        DeadlineExceededError: When ``seconds`` does not fit in the time remaining.
    """
    left = remaining()
    if left is not None and seconds >= left:
        time.sleep(max(left, 0.0))
        raise DeadlineExceededError(f"Timed out {what}: the tool's timeout elapsed")
    time.sleep(max(seconds, 0.0))


def propagate(fn: Callable[P, T]) -> Callable[P, T]:
    """Return ``fn`` bound to the current deadline, for running in another thread.

    Args:
        fn (Callable[P, T]): Work to submit to an executor.

    Returns:
        Callable[P, T]: ``fn`` running under the deadline active when ``propagate`` was called.
    """
    when = _DEADLINE.get()

    @functools.wraps(fn)
    def bound(*args: P.args, **kwargs: P.kwargs) -> T:
        with _at(when):
            return fn(*args, **kwargs)

    return bound


__all__ = [
    "DeadlineExceededError",
    "check_deadline",
    "deadline",
    "expired",
    "propagate",
    "remaining",
    "sleep_within_deadline",
]
