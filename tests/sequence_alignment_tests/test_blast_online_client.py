"""Tests for identified BLAST transport and unchanged search semantics."""

import io
from urllib.parse import parse_qs

import pytest
from Bio.Blast import NCBIWWW

from proto_tools.tools.sequence_alignment.blast import _online_client as client
from proto_tools.tools.sequence_alignment.blast import blast_search as search


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(client.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(client.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(client, "_NEXT_REQUEST", 0)
    monkeypatch.delenv("PROTO_BLAST_PACING_PATH", raising=False)
    return now


def test_identified_submit_and_poll_preserve_parameters(monkeypatch, clock):
    original_request = NCBIWWW.Request
    requests = []
    responses = iter([b"RID = TEST123\nRTOE = 30\n", b"Status=WAITING\n", b"<?xml version='1.0'?><BlastOutput/>"])

    def open_request(request, timeout):
        requests.append((parse_qs(request.data.decode()), request.get_header("User-agent"), clock[0], timeout))
        return io.BytesIO(next(responses))

    monkeypatch.setattr(client, "urlopen", open_request)
    xml = client.search_xml(
        {
            "PROGRAM": "blastn",
            "DATABASE": "nt",
            "QUERY": "ACGT",
            "EXPECT": 0.001,
            "PERC_IDENT": 70.0,
            "HITLIST_SIZE": 20,
        },
        300,
    )
    assert xml.startswith("<?xml")
    assert requests[0][0]["PERC_IDENT"] == ["70.0"]
    assert requests[0][0]["DATABASE"] == ["nt"]
    assert requests[0][0]["EXPECT"] == ["0.001"]
    assert requests[0][0]["QUERY"] == ["ACGT"]
    assert requests[1][0]["RID"] == ["TEST123"]
    assert requests[1][0]["FORMAT_TYPE"] == ["XML"]
    assert all("Proto-Tools" in item[1] and item[1] != "BiopythonClient" for item in requests)
    assert [item[2] for item in requests] == [100, 160, 220]
    assert [item[3] for item in requests] == [300, 240, 180]
    assert NCBIWWW.Request is original_request


@pytest.mark.parametrize(
    ("response", "expected_state", "expected_status"),
    [
        ("", client._SearchState.POLL, None),
        ("Status=WAITING", client._SearchState.POLL, "WAITING"),
        ("Status=FAILED", client._SearchState.FAILURE, "FAILED"),
        ("Status=UNKNOWN", client._SearchState.FAILURE, "UNKNOWN"),
        ("Status=READY", client._SearchState.FAILURE, "READY"),
        ("<?xml version='1.0'?><BlastOutput/>", client._SearchState.SUCCESS, None),
    ],
)
def test_poll_response_state_classification(response, expected_state, expected_status):
    """Every poll response maps to an explicit non-ambiguous protocol state."""
    assert client._classify_poll(response) == (expected_state, expected_status)


def test_submission_rejection_does_not_poll(monkeypatch, clock):
    calls = []

    def reject(*args, **kwargs):
        calls.append(1)
        return io.BytesIO(b'<p class="error">Cannot accept request, error code: -99</p>')

    monkeypatch.setattr(client, "urlopen", reject)
    with pytest.raises(client.BlastSubmissionRejected, match="error code: -99"):
        client.search_xml({"QUERY": "ACGT"}, 300)
    assert len(calls) == 3


def test_rejection_recovers_under_one_deadline(monkeypatch, clock):
    responses = iter(
        [
            b"Cannot accept request, error code: -99",
            b"Cannot accept request, error code: -99",
            b"RID = TEST123\nRTOE = 30\n",
            b"<?xml version='1.0'?><BlastOutput/>",
        ]
    )
    calls = []

    def respond(request, timeout):
        calls.append((clock[0], request.data, timeout))
        return io.BytesIO(next(responses))

    monkeypatch.setattr(client, "urlopen", respond)
    assert client.search_xml({"QUERY": "ACGT"}, 300).startswith("<?xml")
    assert [call[0] for call in calls] == [100, 160, 280, 340]
    assert calls[0][1] == calls[1][1] == calls[2][1]
    assert [call[2] for call in calls] == [300, 240, 120, 60]


def test_rejection_deadline_is_not_retryable_by_outer_layer(monkeypatch, clock):
    from proto_tools.utils.remote_errors import is_retryable_remote_error

    monkeypatch.setattr(client, "urlopen", lambda *a, **k: io.BytesIO(b"Cannot accept request, error code: -99"))
    with pytest.raises(client.BlastSubmissionRejected, match="deadline") as error:
        client.search_xml({"QUERY": "ACGT"}, 50)
    assert not is_retryable_remote_error(error.value)


def test_poll_rejection_reuses_rid(monkeypatch, clock):
    responses = iter([b"RID = TEST123\nRTOE = 30\n", b"Cannot accept request, error code: -99", b"<BlastOutput/>"])
    calls = []

    def respond(request, timeout):
        calls.append(parse_qs(request.data.decode()))
        return io.BytesIO(next(responses))

    monkeypatch.setattr(client, "urlopen", respond)
    assert client.search_xml({"QUERY": "ACGT"}, 300) == "<BlastOutput/>"
    assert [call["CMD"] for call in calls] == [["Put"], ["Get"], ["Get"]]
    assert calls[1]["RID"] == calls[2]["RID"]


def test_other_rejection_is_not_retried(monkeypatch, clock):
    calls = []

    def reject(*args, **kwargs):
        calls.append(1)
        return io.BytesIO(b'<p class="error">Cannot accept request, error code: -98</p>')

    monkeypatch.setattr(client, "urlopen", reject)
    with pytest.raises(ValueError, match="-98"):
        client.search_xml({"QUERY": "ACGT"}, 300)
    assert len(calls) == 1


def test_outer_attempts_do_not_multiply_rejection_budget(monkeypatch, clock):
    import inspect

    calls = []

    def reject(*args, **kwargs):
        calls.append(1)
        return io.BytesIO(b"Cannot accept request, error code: -99")

    monkeypatch.setattr(client, "urlopen", reject)
    with pytest.raises(client.BlastSubmissionRejected, match="exhausted"):
        inspect.unwrap(search.run_blast_search)(
            search.BlastSearchInput(query="ACGT"), search.BlastSearchConfig(online_attempts=5, timeout=600)
        )
    assert len(calls) == 3


def test_deadline_after_retried_submission_stays_terminal(monkeypatch, clock):
    responses = iter([b"Cannot accept request, error code: -99", b"RID = TEST123\nRTOE = 30\n"])
    monkeypatch.setattr(client, "urlopen", lambda *a, **k: io.BytesIO(next(responses)))
    with pytest.raises(client.BlastSubmissionRejected, match="deadline"):
        client.search_xml({"QUERY": "ACGT"}, 100)


def test_request_timeout_during_rejection_recovery_stays_terminal(monkeypatch, clock):
    calls = []

    def respond(*args, **kwargs):
        calls.append(1)
        if len(calls) > 1:
            raise TimeoutError("socket timeout")
        return io.BytesIO(b"Cannot accept request, error code: -99")

    monkeypatch.setattr(client, "urlopen", respond)
    with pytest.raises(client.BlastSubmissionRejected, match="deadline"):
        client.search_xml({"QUERY": "ACGT"}, 300)


def test_shared_pacing_survives_process_local_reset(monkeypatch, clock, tmp_path):
    path = tmp_path / "pacing"
    monkeypatch.setenv("PROTO_BLAST_PACING_PATH", str(path))
    monkeypatch.setattr(client.time, "time", lambda: clock[0] + 1000)
    calls = []

    def respond(*args, **kwargs):
        calls.append(clock[0])
        return io.BytesIO(b"Cannot accept request, error code: -99")

    monkeypatch.setattr(client, "urlopen", respond)
    client._request({"CMD": "Put"}, None)
    monkeypatch.setattr(client, "_NEXT_REQUEST", 0)
    client._request({"CMD": "Put"}, None)
    assert calls == [100, 160]


def test_corrupt_shared_pacing_fails_closed(monkeypatch, clock, tmp_path):
    path = tmp_path / "pacing"
    path.write_text("nan")
    monkeypatch.setenv("PROTO_BLAST_PACING_PATH", str(path))
    monkeypatch.setattr(client, "urlopen", lambda *a, **k: pytest.fail("No request permitted"))
    with pytest.raises(ValueError, match="pacing timestamp"):
        client._request({"CMD": "Put"}, None)


@pytest.mark.slow
def test_two_processes_share_request_spacing(monkeypatch, tmp_path):
    import subprocess
    import sys

    monkeypatch.setenv("PROTO_BLAST_PACING_PATH", str(tmp_path / "pacing"))
    code = """
import io
import time
from proto_tools.tools.sequence_alignment.blast import _online_client as client
def response(*args, **kwargs):
    print(time.time(), flush=True)
    return io.BytesIO(b'response')
client.urlopen = response
client._request({'CMD': 'Put'}, time.monotonic() + 30)
"""
    first = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    second = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        outputs = [process.communicate(timeout=45) for process in (first, second)]
        assert first.returncode == second.returncode == 0, outputs
        times = sorted(float(output[0].strip()) for output in outputs)
        assert times[1] - times[0] >= 10.0
    finally:
        for process in (first, second):
            if process.poll() is None:
                process.kill()
                process.wait()


@pytest.mark.parametrize("response", [b"Status=FAILED", b"Status=UNKNOWN", b"<html>failure</html>", b"Status=READY"])
def test_bad_poll_never_becomes_no_hits(monkeypatch, clock, response):
    responses = iter([b"RID = TEST123\nRTOE = 30\n", response])
    monkeypatch.setattr(client, "urlopen", lambda *args, **kwargs: io.BytesIO(next(responses)))
    with pytest.raises(ValueError, match=r"status|non-XML"):
        client.search_xml({"QUERY": "ACGT"}, 300)


def test_deadline_prevents_early_or_late_poll(monkeypatch, clock):
    calls = []

    def submit(*args, **kwargs):
        calls.append(1)
        return io.BytesIO(b"RID = TEST123\nRTOE = 30\n")

    monkeypatch.setattr(client, "urlopen", submit)
    with pytest.raises(TimeoutError, match="next permitted request"):
        client.search_xml({"QUERY": "ACGT"}, 30)
    assert len(calls) == 1


def test_request_pacing_spans_separate_calls(monkeypatch, clock):
    times = []

    def record(*args, **kwargs):
        times.append(clock[0])
        return io.BytesIO(b"response")

    monkeypatch.setattr(client, "urlopen", record)
    client._request({"CMD": "Put"}, None)
    client._request({"CMD": "Put"}, None)
    assert times == [100, 110]


def test_online_search_preserves_config_mapping(monkeypatch):
    captured = {}

    def capture(parameters, timeout):
        captured.update(parameters)
        assert timeout == 300
        raise RuntimeError("stop before parse")

    monkeypatch.setattr(client, "search_xml", capture)
    config = search.BlastSearchConfig(
        evalue=0.001, perc_identity=70, hitlist_size=20, gapopen=5, gapextend=2, timeout=300
    )
    with pytest.raises(RuntimeError, match="stop before parse"):
        search._online_search(search.BlastSearchInput(query="ACGT"), config)
    assert captured == {
        "PROGRAM": "blastn",
        "DATABASE": "nt",
        "QUERY": "ACGT",
        "EXPECT": 0.001,
        "PERC_IDENT": 70,
        "HITLIST_SIZE": 20,
        "GAPCOSTS": "5 2",
    }
