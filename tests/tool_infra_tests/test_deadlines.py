"""Tests for proto_tools.utils.deadlines."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from proto_tools.utils.deadlines import (
    DeadlineExceededError,
    check_deadline,
    deadline,
    propagate,
    remaining,
    sleep_within_deadline,
)


def test_nested_deadline_never_extends_the_enclosing_one():
    with deadline(1.0):
        outer = remaining()
        with deadline(100.0):
            assert remaining() <= outer
        with deadline(0.1):
            assert remaining() <= 0.1
        assert remaining() == pytest.approx(outer, abs=0.05)
    assert remaining() is None


def test_none_budget_keeps_the_enclosing_deadline():
    with deadline(1.0), deadline(None):
        assert remaining() is not None


def test_check_deadline_raises_once_passed():
    with deadline(0.01):
        time.sleep(0.02)
        with pytest.raises(DeadlineExceededError, match="fetching x"):
            check_deadline("fetching x")


def test_sleep_that_cannot_fit_ends_at_the_deadline_not_before():
    started = time.monotonic()
    with deadline(0.3), pytest.raises(DeadlineExceededError):
        sleep_within_deadline(10.0, "waiting")
    assert 0.28 <= time.monotonic() - started < 0.5


def test_propagate_carries_the_deadline_into_executor_threads():
    with deadline(5.0), ThreadPoolExecutor(2) as pool:
        bare = pool.submit(remaining).result()
        bound = list(pool.map(propagate(lambda _: remaining()), range(3)))
    assert bare is None
    assert all(left is not None and 0 < left <= 5.0 for left in bound)
