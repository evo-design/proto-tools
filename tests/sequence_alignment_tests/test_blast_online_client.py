"""Tests for the NCBI QBLAST submit-and-poll client."""

from urllib.parse import parse_qs, urlsplit

import pytest

from proto_tools.tools.sequence_alignment.blast import _online_client as client
from proto_tools.tools.sequence_alignment.blast import blast_search as search

_RID_PAGE = "RID = TEST123\nRTOE = 30\n"
_REJECTION = '<p class="error">Cannot accept request, error code: -99</p>'
_XML = "<?xml version='1.0'?><BlastOutput/>"


class _Response:
    def __init__(self, text: str, url: str = "") -> None:
        self.text = text
        self.content = text.encode()
        self.url = url

    def raise_for_status(self) -> None:
        pass


class _Session:
    """Replays canned submit and poll pages, recording each request and the fake clock time."""

    def __init__(self, clock: list[float], posts: list[str], gets: list[str]) -> None:
        self.clock = clock
        self.posts = iter(posts)
        self.gets = iter(gets)
        self.calls: list[tuple[str, float, object]] = []

    def __enter__(self) -> "_Session":
        return self

    def __exit__(self, *_exc: object) -> None:
        pass

    def post(self, url: str, data: dict, timeout: float) -> _Response:
        self.calls.append(("POST", self.clock[0], data))
        return _Response(next(self.posts), url)

    def get(self, url: str, timeout: float) -> _Response:
        self.calls.append(("GET", self.clock[0], parse_qs(urlsplit(url).query)))
        return _Response(next(self.gets), url)


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(client.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(client.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(client, "_next_submit", 0.0)
    return now


def _use_session(monkeypatch, session: _Session) -> dict:
    built: dict = {}
    monkeypatch.setattr(client, "build_http_session", lambda **kwargs: built.update(kwargs) or session)
    return built


def test_submit_then_poll_identifies_caller_and_keeps_parameters(monkeypatch, clock):
    session = _Session(clock, posts=[_RID_PAGE], gets=["Status=WAITING", _XML])
    built = _use_session(monkeypatch, session)

    xml = client.search_xml(
        {"PROGRAM": "blastn", "QUERY": "ACGT", "EXPECT": 0.001}, timeout=600, client_identity="me@lab"
    )

    assert xml == _XML
    assert built["user_agent"].startswith("proto-tools/") and "me@lab" in built["user_agent"]
    (_, submitted_at, submitted), (_, first_poll_at, poll), (_, second_poll_at, _) = session.calls
    assert submitted == {
        "EXPECT": 0.001,
        "HITLIST_SIZE": 50,
        "ENTREZ_QUERY": "(none)",
        "PROGRAM": "blastn",
        "QUERY": "ACGT",
        "CMD": "Put",
        "tool": "proto-tools",
    }
    assert poll["RID"] == ["TEST123"] and poll["FORMAT_TYPE"] == ["XML"]
    assert [submitted_at, first_poll_at, second_poll_at] == [100, 160, 220]


@pytest.mark.parametrize("page", ["", "Status=WAITING", _REJECTION])
def test_pending_poll_pages_keep_polling(page):
    assert client._qblast_status(_Response(page))[0] == "WAITING"


@pytest.mark.parametrize("page", ["Status=FAILED", "Status=UNKNOWN", "Status=READY", "<html>failure</html>"])
def test_failed_poll_pages_never_read_as_no_hits(page):
    with pytest.raises(ValueError, match="no hits can be inferred"):
        client._qblast_status(_Response(page))


def test_submission_rejection_recovers_after_cooldown(monkeypatch, clock):
    session = _Session(clock, posts=[_REJECTION, _RID_PAGE], gets=[_XML])
    _use_session(monkeypatch, session)

    assert client.search_xml({"QUERY": "ACGT"}, timeout=600, client_identity=None) == _XML
    assert [(method, at) for method, at, _ in session.calls] == [("POST", 100), ("POST", 160), ("GET", 220)]


def test_submission_rejection_is_retried_twice_then_fails(monkeypatch, clock):
    session = _Session(clock, posts=[_REJECTION] * 3, gets=[])
    _use_session(monkeypatch, session)

    with pytest.raises(ValueError, match="error code -99"):
        client.search_xml({"QUERY": "ACGT"}, timeout=600, client_identity=None)
    assert [at for _, at, _ in session.calls] == [100, 160, 280]


def test_search_times_out_while_pending(monkeypatch, clock):
    """The search fails when its timeout elapses, not before and not a poll interval later."""
    _use_session(monkeypatch, _Session(clock, posts=[_RID_PAGE], gets=["Status=WAITING"] * 10))

    with pytest.raises(TimeoutError):
        client.search_xml({"QUERY": "ACGT"}, timeout=150, client_identity=None)
    assert clock[0] == pytest.approx(250)


def test_rejection_cooldown_stops_at_the_timeout(monkeypatch, clock):
    """A -99 cooldown that would outlast the timeout ends at it instead of sleeping past it (#343)."""
    session = _Session(clock, posts=[_REJECTION] * 3, gets=[])
    _use_session(monkeypatch, session)

    with pytest.raises(TimeoutError):
        client.search_xml({"QUERY": "ACGT"}, timeout=100, client_identity=None)
    assert clock[0] == pytest.approx(200)
    assert [at for _, at, _ in session.calls] == [100, 160]


def test_submissions_are_spaced_within_a_process(clock):
    session = _Session(clock, posts=[_RID_PAGE, _RID_PAGE], gets=[])
    client._submit(session, {"QUERY": "ACGT"})
    client._submit(session, {"QUERY": "TGCA"})
    assert [at for _, at, _ in session.calls] == [100, 110]


def test_online_search_preserves_config_mapping(monkeypatch):
    captured = {}

    def capture(parameters, *, timeout, client_identity):
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
