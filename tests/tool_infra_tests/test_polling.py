"""Tests for `proto_tools.utils.polling.poll_until_complete`."""

from unittest.mock import MagicMock

import pytest

from proto_tools.utils.polling import extract_text_status, poll_until_complete


def _mock_session(payloads):
    """Build a session that returns the given JSON payloads in order on successive .get() calls."""
    session = MagicMock()
    responses = []
    for payload in payloads:
        response = MagicMock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.json.return_value = payload
        responses.append(response)
    session.get.side_effect = responses
    return session


def _mock_text_session(bodies):
    """Build a session that returns the given plain-text bodies in order on successive .get() calls."""
    session = MagicMock()
    responses = []
    for body in bodies:
        response = MagicMock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.text = body
        responses.append(response)
    session.get.side_effect = responses
    return session


def test_poll_until_complete_returns_on_terminal_state(monkeypatch):
    """A single COMPLETE response returns the payload without sleeping."""
    monkeypatch.setattr("proto_tools.utils.polling.time.sleep", lambda _s: None)
    session = _mock_session([{"id": "abc", "status": "COMPLETE"}])

    result = poll_until_complete(session, "https://example/ticket/abc")

    assert result == {"id": "abc", "status": "COMPLETE"}
    assert session.get.call_count == 1


def test_poll_until_complete_polls_through_running_states(monkeypatch):
    """Intermediate PENDING/RUNNING states are tolerated until COMPLETE arrives."""
    monkeypatch.setattr("proto_tools.utils.polling.time.sleep", lambda _s: None)
    session = _mock_session(
        [
            {"id": "abc", "status": "PENDING"},
            {"id": "abc", "status": "RUNNING"},
            {"id": "abc", "status": "COMPLETE"},
        ]
    )

    result = poll_until_complete(session, "https://example/ticket/abc")

    assert result["status"] == "COMPLETE"
    assert session.get.call_count == 3


def test_poll_until_complete_raises_on_error_state(monkeypatch):
    """ERROR state raises ValueError carrying the full payload."""
    monkeypatch.setattr("proto_tools.utils.polling.time.sleep", lambda _s: None)
    session = _mock_session([{"id": "abc", "status": "ERROR", "reason": "bad input"}])

    with pytest.raises(ValueError, match="Job failed"):
        poll_until_complete(session, "https://example/ticket/abc")


def test_poll_until_complete_raises_on_timeout(monkeypatch):
    """When the deadline passes without a terminal state, TimeoutError is raised."""
    monkeypatch.setattr("proto_tools.utils.polling.time.sleep", lambda _s: None)
    # First call (deadline computation) at t=0; every subsequent call already past the 10s deadline.
    call_count = {"n": 0}

    def fake_monotonic() -> float:
        call_count["n"] += 1
        return 0.0 if call_count["n"] == 1 else 999.0

    monkeypatch.setattr("proto_tools.utils.polling.time.monotonic", fake_monotonic)
    # Many pending responses: the timeout check should fire before they're exhausted.
    session = _mock_session([{"id": "abc", "status": "PENDING"}] * 20)

    with pytest.raises(TimeoutError, match="Timeout after"):
        poll_until_complete(session, "https://example/ticket/abc", timeout_seconds=10.0)


def test_poll_until_complete_text_extractor_iprscan_vocabulary(monkeypatch):
    """Plain-text mode with iprscan5's vocabulary returns the final status string on FINISHED."""
    monkeypatch.setattr("proto_tools.utils.polling.time.sleep", lambda _s: None)
    session = _mock_text_session(["QUEUED", "RUNNING", "FINISHED"])

    result = poll_until_complete(
        session,
        "https://example/iprscan5/status/abc",
        success_states=frozenset({"FINISHED"}),
        failure_states=frozenset({"ERROR", "FAILURE", "NOT_FOUND"}),
        status_extractor=extract_text_status,
    )

    assert result == "FINISHED"
    assert session.get.call_count == 3


def test_poll_until_complete_custom_success_state(monkeypatch):
    """Caller-supplied success_states overrides the default 'COMPLETE' sentinel."""
    monkeypatch.setattr("proto_tools.utils.polling.time.sleep", lambda _s: None)
    session = _mock_session([{"id": "abc", "status": "DONE"}])

    result = poll_until_complete(
        session,
        "https://example/ticket/abc",
        success_states=frozenset({"DONE"}),
    )

    assert result == {"id": "abc", "status": "DONE"}


def test_poll_until_complete_recovers_from_transient_mid_poll(monkeypatch):
    """RUNNING → ConnectionError → COMPLETE; the long submit-path doesn't die on one TCP reset."""
    import requests

    monkeypatch.setattr("proto_tools.utils.polling.time.sleep", lambda _s: None)
    session = MagicMock()
    running = MagicMock()
    running.raise_for_status.return_value = None
    running.json.return_value = {"id": "abc", "status": "RUNNING"}
    success = MagicMock()
    success.raise_for_status.return_value = None
    success.json.return_value = {"id": "abc", "status": "COMPLETE"}
    session.get.side_effect = [running, requests.ConnectionError("reset"), success]

    result = poll_until_complete(session, "https://example/ticket/abc")

    assert result == {"id": "abc", "status": "COMPLETE"}
    assert session.get.call_count == 3


def test_poll_until_complete_transient_errors_respect_wall_clock(monkeypatch):
    """Persistent transient errors past the deadline raise TimeoutError that preserves the underlying error."""
    import requests

    monkeypatch.setattr("proto_tools.utils.polling.time.sleep", lambda _s: None)
    call_count = {"n": 0}

    def fake_monotonic() -> float:
        call_count["n"] += 1
        return 0.0 if call_count["n"] == 1 else 999.0

    monkeypatch.setattr("proto_tools.utils.polling.time.monotonic", fake_monotonic)
    session = MagicMock()
    session.get.side_effect = requests.ConnectionError("server unreachable")

    with pytest.raises(TimeoutError, match=r"Timeout after.*server unreachable"):
        poll_until_complete(session, "https://example/ticket/abc", timeout_seconds=10.0)


def test_poll_until_complete_does_not_retry_4xx(monkeypatch):
    """A 4xx is a caller error (bad job ID, auth) — re-raised immediately, no retry loop."""
    import requests

    monkeypatch.setattr("proto_tools.utils.polling.time.sleep", lambda _s: None)
    session = MagicMock()
    response = MagicMock()
    response.status_code = 404
    response.raise_for_status.side_effect = requests.HTTPError("404 Not Found", response=response)
    session.get.return_value = response

    with pytest.raises(requests.HTTPError, match="404"):
        poll_until_complete(session, "https://example/ticket/missing")
    assert session.get.call_count == 1


def test_poll_until_complete_retries_5xx_after_urllib3_exhaustion(monkeypatch):
    """A 5xx that bubbles past urllib3's retry adapter is treated as transient and retried until success."""
    import requests

    monkeypatch.setattr("proto_tools.utils.polling.time.sleep", lambda _s: None)
    session = MagicMock()
    bad = MagicMock()
    bad.status_code = 503
    bad.raise_for_status.side_effect = requests.HTTPError("503 Service Unavailable", response=bad)
    success = MagicMock()
    success.raise_for_status.return_value = None
    success.json.return_value = {"id": "abc", "status": "COMPLETE"}
    session.get.side_effect = [bad, success]

    result = poll_until_complete(session, "https://example/ticket/abc")

    assert result == {"id": "abc", "status": "COMPLETE"}
    assert session.get.call_count == 2


def test_poll_until_complete_fails_loud_on_missing_status_key(monkeypatch):
    """A 200 with valid JSON but missing 'status' key is a server-schema regression — fail loud, don't loop."""
    monkeypatch.setattr("proto_tools.utils.polling.time.sleep", lambda _s: None)
    session = _mock_session([{"id": "abc", "result": "??"}])  # no 'status' key

    with pytest.raises(ValueError, match="missing string 'status' key"):
        poll_until_complete(session, "https://example/ticket/abc")
    assert session.get.call_count == 1


def test_poll_until_complete_fails_loud_on_empty_text_status(monkeypatch):
    """An empty plain-text status body is a server-side regression — fail loud, don't loop until deadline."""
    monkeypatch.setattr("proto_tools.utils.polling.time.sleep", lambda _s: None)
    session = _mock_text_session([""])

    with pytest.raises(ValueError, match="Empty plain-text status body"):
        poll_until_complete(session, "https://example/iprscan5/status/abc", status_extractor=extract_text_status)
    assert session.get.call_count == 1
