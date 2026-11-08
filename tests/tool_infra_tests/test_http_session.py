"""tests/tool_infra_tests/test_http_session.py.

Tests for proto_tools.utils.http_session.request_with_retry and raise_for_status_with_body.
"""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests

from proto_tools.utils.http_session import build_http_session, raise_for_status_with_body, request_with_retry


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


def test_empty_status_forcelist_hands_a_429_back_to_the_caller():
    """The default forcelist would raise RetryError here, hiding the response and its Retry-After."""

    class _RateLimited(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(429)
            self.send_header("Retry-After", "7")
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), _RateLimited)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        session = build_http_session(
            http_retries=0, backoff_seconds=0.0, user_agent="test", mount_http=True, status_forcelist=[]
        )
        response = session.get(f"http://127.0.0.1:{server.server_port}/", timeout=5)
    finally:
        server.shutdown()
    assert (response.status_code, response.headers["Retry-After"]) == (429, "7")
