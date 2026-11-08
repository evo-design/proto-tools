"""Tests for arXiv search."""

import itertools
import threading
from collections.abc import Callable
from typing import Any

import pytest
import requests

from proto_tools.tools.database_retrieval import ArxivSearchConfig, ArxivSearchInput, run_arxiv_search
from proto_tools.tools.database_retrieval.arxiv import arxiv_search as module

_FEED = b"""<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns="http://www.w3.org/2005/Atom">
  <opensearch:totalResults>15</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2504.00146v2</id>
    <title>Why risk matters for
      protein binder design</title>
    <updated>2025-04-02T11:43:24Z</updated>
    <link href="https://arxiv.org/abs/2504.00146v2" rel="alternate" type="text/html"/>
    <link href="https://arxiv.org/pdf/2504.00146v2" rel="related" type="application/pdf" title="pdf"/>
    <summary>  Bayesian optimization
      for binders.  </summary>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <category term="q-bio.QM" scheme="http://arxiv.org/schemas/atom"/>
    <published>2025-03-31T18:54:38Z</published>
    <arxiv:doi>10.1000/example</arxiv:doi>
    <arxiv:journal_ref>GEM Workshop 2025</arxiv:journal_ref>
    <arxiv:primary_category term="cs.LG"/>
    <author><name>Tudor-Stefan Cotet</name></author>
    <author><name>Igor Krawczuk</name></author>
  </entry>
</feed>"""

# The error feed shape from arXiv's API user manual (section 3.4).
_ERROR_FEED = b"""<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>1</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/api/errors#max_results_must_be_an_integer</id>
    <title>Error</title>
    <summary>max_results must be an integer</summary>
    <link href="http://arxiv.org/api/errors#max_results_must_be_an_integer" rel="alternate" type="text/html"/>
    <author><name>arXiv api core</name></author>
  </entry>
</feed>"""


class _WatchedLock:
    """Stands in for the gate's lock, signalling once a second request is waiting to enter it."""

    def __init__(self) -> None:
        self.contended = threading.Event()
        self._lock = threading.Lock()
        self._entries = 0
        self._count_lock = threading.Lock()

    def __enter__(self) -> None:
        with self._count_lock:
            self._entries += 1
            if self._entries >= 2:
                self.contended.set()
        self._lock.acquire()

    def __exit__(self, *_exc: object) -> None:
        self._lock.release()


class _Clock:
    """Thread-safe fake monotonic clock; ``sleep`` advances it instead of waiting."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []
        self._lock = threading.Lock()

    def monotonic(self) -> float:
        with self._lock:
            return self.now

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.sleeps.append(seconds)
            self.now += seconds

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.now += seconds


class _Response:
    def __init__(self, status_code: int = 200, content: bytes = _FEED, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.content = content
        self.text = "synthetic body"
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error")


class _Session:
    """Fake arXiv session that records each request's parameters, timing and concurrency."""

    def __init__(
        self,
        clock: _Clock,
        respond: Callable[[int, str], _Response] | None = None,
        hold_first_until: threading.Event | None = None,
    ) -> None:
        self.clock = clock
        self.respond = respond or (lambda _index, _query: _Response())
        self.hold_first_until = hold_first_until
        self.calls: list[dict[str, Any]] = []
        self.spans: list[tuple[float, float]] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.closed = False
        self._lock = threading.Lock()

    def get(self, url: str, **kwargs: Any) -> _Response:
        assert url == module._QUERY_URL
        with self._lock:
            index = len(self.calls)
            self.calls.append(kwargs["params"])
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        start = self.clock.monotonic()
        if index == 0 and self.hold_first_until is not None:
            self.hold_first_until.wait(timeout=5)  # stay in flight until another caller is contending
        self.clock.advance(0.5)
        with self._lock:
            self.in_flight -= 1
            self.spans.append((start, self.clock.monotonic()))
        return self.respond(index, kwargs["params"]["search_query"])

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(module, "monotonic", fake.monotonic)
    monkeypatch.setattr(module, "sleep", fake.sleep)
    monkeypatch.setattr(module._gate, "next_request_at", float("-inf"))
    return fake


def _install(monkeypatch: pytest.MonkeyPatch, fake: _Session) -> _Session:
    monkeypatch.setattr(module, "build_http_session", lambda **_kwargs: fake)
    return fake


# ── Parsing ──────────────────────────────────────────────────────────────────


def test_entries_are_parsed(monkeypatch: pytest.MonkeyPatch, clock: _Clock) -> None:
    session = _install(monkeypatch, _Session(clock))
    output = run_arxiv_search(ArxivSearchInput(queries='ti:"protein binder"'), ArxivSearchConfig(max_results=3))
    assert session.closed is True and clock.sleeps == []
    assert session.calls == [{"search_query": 'ti:"protein binder"', "start": 0, "max_results": 3}]
    (result,) = output.results
    assert (result.total_count, result.error) == (15, None)
    (hit,) = result.hits
    assert (hit.arxiv_id, hit.title, hit.summary) == (
        "2504.00146v2",
        "Why risk matters for protein binder design",
        "Bayesian optimization for binders.",
    )
    assert hit.authors == ["Tudor-Stefan Cotet", "Igor Krawczuk"]
    assert (hit.primary_category, hit.categories) == ("cs.LG", ["cs.LG", "q-bio.QM"])
    assert (hit.doi, hit.journal_ref, hit.comment) == ("10.1000/example", "GEM Workshop 2025", None)
    assert (hit.abs_url, hit.pdf_url) == ("https://arxiv.org/abs/2504.00146v2", "https://arxiv.org/pdf/2504.00146v2")


# ── Process-wide rate limit ──────────────────────────────────────────────────


def test_limit_holds_across_concurrent_calls_and_retries(monkeypatch: pytest.MonkeyPatch, clock: _Clock) -> None:
    """Two concurrent calls never overlap, keep three seconds apart, and a 429 waits its Retry-After."""
    rate_limited = _Response(429, b"Rate exceeded", {"Retry-After": "10"})
    lock = _WatchedLock()
    monkeypatch.setattr(module._gate, "lock", lock)
    session = _install(
        monkeypatch,
        _Session(clock, lambda index, _query: rate_limited if index == 1 else _Response(), lock.contended),
    )
    batches = [["ti:a1", "ti:a2"], ["ti:b1", "ti:b2", "ti:b3"]]
    outputs: dict[int, Any] = {}
    barrier = threading.Barrier(len(batches))

    def call(slot: int) -> None:
        barrier.wait()
        outputs[slot] = run_arxiv_search(ArxivSearchInput(queries=batches[slot]))

    threads = [threading.Thread(target=call, args=(slot,)) for slot in range(len(batches))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads)

    assert session.max_in_flight == 1
    assert len(session.spans) == 6  # five queries plus one retry
    spans = sorted(session.spans)
    gaps = [start - previous_end for (_, previous_end), (start, _) in itertools.pairwise(spans)]
    assert all(gap >= module._REQUEST_INTERVAL_SECONDS for gap in gaps)
    assert gaps[1] >= 10.0  # the request after the 429 (request 1) waited out its Retry-After
    for slot, queries in enumerate(batches):
        assert [result.query for result in outputs[slot].results] == queries
        assert all(result.error is None and result.total_count == 15 for result in outputs[slot].results)


# ── Errors ───────────────────────────────────────────────────────────────────


def test_rejected_query_gets_an_error_and_the_batch_continues(monkeypatch: pytest.MonkeyPatch, clock: _Clock) -> None:
    def respond(_index: int, query: str) -> _Response:
        return _Response(400) if query == "ti:(((" else _Response()

    session = _install(monkeypatch, _Session(clock, respond))
    output = run_arxiv_search(ArxivSearchInput(queries=["ti:a", "ti:(((", "ti:b"]))
    assert len(session.calls) == 3
    good, bad, last = output.results
    assert (good.error, last.error) == (None, None) and good.hits and last.hits
    assert (bad.query, bad.total_count, bad.hits) == ("ti:(((", 0, [])
    assert bad.error is not None and "arXiv search failed" in bad.error and "synthetic body" in bad.error


def test_error_feed_entry_becomes_the_error(monkeypatch: pytest.MonkeyPatch, clock: _Clock) -> None:
    _install(monkeypatch, _Session(clock, lambda _index, _query: _Response(content=_ERROR_FEED)))
    (result,) = run_arxiv_search(ArxivSearchInput(queries="ti:a")).results
    assert (result.total_count, result.hits) == (0, [])
    assert result.error == "arXiv search failed: max_results must be an integer"


def test_server_error_raises_after_retries(monkeypatch: pytest.MonkeyPatch, clock: _Clock) -> None:
    session = _install(monkeypatch, _Session(clock, lambda _index, _query: _Response(503)))
    with pytest.raises(requests.HTTPError, match=r"arXiv search failed.*synthetic body"):
        run_arxiv_search(ArxivSearchInput(queries=["ti:a", "ti:b"]))
    assert len(session.calls) == module._HTTP_RETRIES + 1
    assert session.closed is True


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_arxiv_search_finds_a_known_preprint_by_title():
    """A title phrase query returns the matching arXiv record with its version."""
    output = run_arxiv_search(
        ArxivSearchInput(queries='ti:"Why risk matters for protein binder design"'), ArxivSearchConfig(max_results=3)
    )
    hits = output.results[0].hits
    assert any(hit.arxiv_id.startswith("2504.00146v") for hit in hits)
    assert all(hit.pdf_url.startswith("https://arxiv.org/pdf/") for hit in hits)


@pytest.mark.integration
def test_arxiv_search_category_filter():
    """cat:q-bio.BM keeps every hit in that category."""
    output = run_arxiv_search(ArxivSearchInput(queries="cat:q-bio.BM AND abs:binder"), ArxivSearchConfig(max_results=5))
    hits = output.results[0].hits
    assert hits and all("q-bio.BM" in hit.categories for hit in hits)
