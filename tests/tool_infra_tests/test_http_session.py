"""tests/tool_infra_tests/test_http_session.py.

Tests for proto_tools.utils.http_session: request_with_retry, raise_for_status_with_body and
deadline-bounded sessions.
"""

import time

import pytest
import requests

from proto_tools.utils import build_http_session, deadline, iter_content_within_deadline
from proto_tools.utils.http_session import raise_for_status_with_body, request_with_retry
from tests import _http_servers


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


# ── Deadline enforcement (#343) ──────────────────────────────────────────────
# A requests ``timeout=`` bounds each socket read, not the request. These run real requests
# against misbehaving localhost servers and check the session still ends at the deadline.


def _session() -> requests.Session:
    return build_http_session(http_retries=2, backoff_seconds=0.0, user_agent="test", mount_http=True)


@pytest.mark.parametrize(
    "handler",
    [_http_servers.TricklingHandler, _http_servers.ChunkedTricklingHandler, _http_servers.StallingHandler],
)
def test_slow_body_ends_at_deadline(handler):
    """A body that trickles or stalls cannot outlast the deadline, though no single read times out."""
    with _http_servers.serve(handler) as server, _session() as session, deadline(1.0):
        started = time.monotonic()
        with pytest.raises(TimeoutError) as excinfo:
            session.get(server.url, timeout=15)
        elapsed = time.monotonic() - started
    assert not isinstance(excinfo.value, requests.RequestException)
    assert 0.95 <= elapsed < 1.5


def test_retry_after_past_deadline_ends_at_deadline():
    """A Retry-After longer than the time left waits out the deadline, then fails without retrying."""
    with _http_servers.serve(_http_servers.RetryAfterHandler) as server, _session() as session, deadline(1.0):
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            session.get(server.url, timeout=15)
        elapsed = time.monotonic() - started
    assert 0.95 <= elapsed < 1.5
    assert server.requests_seen == 1


def test_backoff_after_retry_after_zero_ends_at_deadline():
    """``Retry-After: 0`` makes urllib3 fall back to its backoff, which must stop at the deadline too."""
    session = build_http_session(http_retries=2, backoff_seconds=2.0, user_agent="test", mount_http=True)
    with _http_servers.serve(_http_servers.RetryAfterZeroHandler) as server, session, deadline(1.0):
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            session.get(server.url, timeout=15)
        elapsed = time.monotonic() - started
    assert 0.95 <= elapsed < 1.5


def test_streamed_download_ends_at_deadline():
    """``iter_content_within_deadline`` bounds a ``stream=True`` body the same way."""
    with _http_servers.serve(_http_servers.TricklingHandler) as server, _session() as session, deadline(1.0):
        started = time.monotonic()
        with pytest.raises(TimeoutError), session.get(server.url, timeout=15, stream=True) as response:
            for _ in iter_content_within_deadline(response, 1024):
                pass
        elapsed = time.monotonic() - started
    assert 0.95 <= elapsed < 1.5


@pytest.mark.parametrize("budget", [None, 30.0])
def test_redirect_and_gzip_decode_with_and_without_deadline(budget):
    """Reading the body under a deadline still follows redirects, decodes gzip and frees the connection."""
    with _http_servers.serve(_http_servers.GzipRedirectHandler) as server, _session() as session, deadline(budget):
        first = session.get(f"{server.url}/start", timeout=15)
        second = session.get(f"{server.url}/start", timeout=15)
    assert first.json() == second.json() == {"ok": True}
    assert [r.status_code for r in first.history] == [302]


def test_streamed_body_without_deadline_is_read_normally():
    with _http_servers.serve(_http_servers.GzipRedirectHandler) as server, _session() as session:
        with session.get(f"{server.url}/data", timeout=15, stream=True) as response:
            body = b"".join(iter_content_within_deadline(response, 4))
    assert body == b'{"ok": true}'


def test_request_with_retry_does_not_retry_a_deadline_timeout():
    """The deadline error is not a requests.ConnectionError, so the retry helper lets it through."""
    calls = {"n": 0}
    with _http_servers.serve(_http_servers.StallingHandler) as server, _session() as session, deadline(0.5):

        def get():
            calls["n"] += 1
            return session.get(server.url, timeout=15)

        with pytest.raises(TimeoutError):
            request_with_retry(get, retries=3, backoff_seconds=0.0)
    assert calls["n"] == 1


def test_request_with_retry_backoff_ends_at_deadline():
    """A backoff longer than the time left waits out the deadline, then raises the timeout."""
    calls = {"n": 0}

    def always_fails():
        calls["n"] += 1
        raise requests.exceptions.ConnectionError("reset")

    started = time.monotonic()
    with deadline(0.3), pytest.raises(TimeoutError):
        request_with_retry(always_fails, retries=3, backoff_seconds=5.0)
    assert 0.28 <= time.monotonic() - started < 0.5
    assert calls["n"] == 1


def _response(status_code: int, text: str) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = text.encode()
    response.url = "https://api.example.org/search"
    return response


def test_raise_for_status_with_body_names_the_operation_and_quotes_the_body():
    response = _response(400, "query syntax error near 'AND'")
    with pytest.raises(requests.HTTPError, match=r"Example search failed: 400 .*query syntax error") as info:
        raise_for_status_with_body(response, "Example search")
    assert info.value.response is response


def test_raise_for_status_with_body_truncates_a_long_body():
    with pytest.raises(requests.HTTPError) as info:
        raise_for_status_with_body(_response(500, "x" * 5000), "Example search")
    assert "x" * 1000 in str(info.value) and "x" * 1001 not in str(info.value)


def test_raise_for_status_with_body_passes_a_success():
    raise_for_status_with_body(_response(200, "ok"), "Example search")


def test_empty_status_forcelist_hands_a_retryable_status_back_to_the_caller():
    """The default forcelist would retry this 503 and raise RetryError, hiding its Retry-After."""
    session = build_http_session(
        http_retries=0, backoff_seconds=0.0, user_agent="test", mount_http=True, status_forcelist=[]
    )
    with _http_servers.serve(_http_servers.RetryAfterHandler) as server, session:
        response = session.get(server.url, timeout=5)
    assert (response.status_code, response.headers["Retry-After"]) == (503, "20")
    assert server.requests_seen == 1
