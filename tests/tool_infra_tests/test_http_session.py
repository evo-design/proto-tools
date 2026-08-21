"""tests/tool_infra_tests/test_http_session.py.

Tests for proto_tools.utils.http_session.request_with_retry.
"""

import requests

from proto_tools.utils.http_session import request_with_retry


def test_retries_until_success_within_budget():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.exceptions.ConnectionError("Connection reset by peer")
        return "ok"

    assert request_with_retry(flaky, retries=2, backoff_seconds=0.0) == "ok"
    assert calls["n"] == 3


def test_gives_up_after_retries_exhausted():
    calls = {"n": 0}

    def always_fails():
        calls["n"] += 1
        raise requests.exceptions.ConnectionError("nope")

    try:
        request_with_retry(always_fails, retries=2, backoff_seconds=0.0)
    except requests.exceptions.ConnectionError:
        pass
    else:
        raise AssertionError("expected ConnectionError to propagate")
    assert calls["n"] == 3


def test_retryable_exceptions_is_configurable():
    """A caller can widen what counts as transient without re-implementing the backoff."""
    calls = {"n": 0}

    def flaky_timeout():
        calls["n"] += 1
        if calls["n"] < 2:
            raise requests.exceptions.Timeout("read timed out")
        return "ok"

    result = request_with_retry(
        flaky_timeout,
        retries=1,
        backoff_seconds=0.0,
        retryable_exceptions=(requests.exceptions.Timeout,),
    )
    assert result == "ok"
    assert calls["n"] == 2


def test_exception_outside_the_configured_set_is_not_retried():
    calls = {"n": 0}

    def raises_value_error():
        calls["n"] += 1
        raise ValueError("not a transient failure")

    try:
        request_with_retry(raises_value_error, retries=3, backoff_seconds=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError to propagate immediately")
    assert calls["n"] == 1
