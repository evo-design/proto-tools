"""Tests for `proto_tools.utils.polling.poll_until_complete`."""

from unittest.mock import MagicMock

import pytest

from proto_tools.utils.polling import poll_until_complete


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
