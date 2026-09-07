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


def test_submission_rejection_does_not_poll(monkeypatch, clock):
    calls = []

    def reject(*args, **kwargs):
        calls.append(1)
        return io.BytesIO(b'<p class="error">Cannot accept request, error code: -99</p>')

    monkeypatch.setattr(client, "urlopen", reject)
    with pytest.raises(ValueError, match="error code: -99"):
        client.search_xml({"QUERY": "ACGT"}, 300)
    assert len(calls) == 1


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
