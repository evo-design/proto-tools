"""Tests for Paperclip literature retrieval and its direct service client.

Fake replies reproduce the formats paperclip.gxl.ai returns from its MCP tools.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest
from pydantic import ValidationError

from proto_tools.tools.database_retrieval import (
    PaperclipDocumentConfig,
    PaperclipDocumentInput,
    PaperclipGrepConfig,
    PaperclipGrepInput,
    PaperclipLookupConfig,
    PaperclipLookupInput,
    PaperclipSearchConfig,
    PaperclipSearchInput,
    run_paperclip_document,
    run_paperclip_grep,
    run_paperclip_lookup,
    run_paperclip_search,
)
from proto_tools.tools.database_retrieval.paperclip import paperclip_client, paperclip_retrieval
from proto_tools.tools.database_retrieval.paperclip.paperclip_client import (
    PaperclipAuthError,
    PaperclipClient,
    PaperclipCommandError,
    PaperclipError,
    PaperclipToolResult,
)

_CONTENT = "/papers/PMC1/content.lines"

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: Any = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


def _mcp_reply(text: str, *, is_error: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    if is_error:
        result["isError"] = True
    return {"jsonrpc": "2.0", "id": 1, "result": result}


def _key(tool: str, **arguments: Any) -> str:
    return f"{tool} {json.dumps(arguments, sort_keys=True)}"


class _FakeSession:
    """Records requests and replays canned replies keyed by tool call or result id."""

    def __init__(self, calls: dict[str, Any] | None = None, results: dict[str, Any] | None = None):
        self.calls = calls or {}
        self.results = results or {}
        self.posts: list[dict[str, Any]] = []
        self.gets: list[dict[str, Any]] = []

    def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str], timeout: float) -> _FakeResponse:
        self.posts.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        params = json["params"]
        reply = self.calls[_key(params["name"], **params["arguments"])]
        return reply if isinstance(reply, _FakeResponse) else _FakeResponse(200, reply)

    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> _FakeResponse:
        self.gets.append({"url": url, "headers": headers, "timeout": timeout})
        reply = self.results[url.rsplit("/", 1)[-1]]
        return reply if isinstance(reply, _FakeResponse) else _FakeResponse(200, reply)


def _client(calls: dict[str, Any] | None = None, results: dict[str, Any] | None = None):
    session = _FakeSession(calls, results)
    return PaperclipClient(api_key="k", session=session), session


class _FakeClient:
    """Stand-in for PaperclipClient at the tool boundary."""

    def __init__(
        self,
        *,
        files: dict[str, str] | None = None,
        lines: list[str] | None = None,
        grep_output: str = "",
        grep_result_id: str | None = None,
        results: dict[str, list[dict[str, Any]]] | None = None,
    ):
        self.kwargs: dict[str, Any] = {}
        self.files = files or {}
        self.lines = lines or []
        self.grep_output = grep_output
        self.grep_result_id = grep_result_id
        self.results = results or {}
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.saved_result: tuple[str | None, list[dict[str, Any]]] = (None, [])

    def search(self, query: str, **kwargs: Any) -> tuple[str | None, list[dict[str, Any]]]:
        self.calls.append(("search", (query,), kwargs))
        return self.saved_result

    def lookup(self, field: str, value: str, **kwargs: Any) -> tuple[str | None, list[dict[str, Any]]]:
        self.calls.append(("lookup", (field, value), kwargs))
        return self.saved_result

    def grep(self, pattern: str, path: str, **kwargs: Any) -> PaperclipToolResult:
        self.calls.append(("grep", (pattern, path), kwargs))
        return PaperclipToolResult(tool="grep", output=self.grep_output, result_id=self.grep_result_id)

    def fetch_result_papers(self, result_id: str) -> list[dict[str, Any]]:
        self.calls.append(("fetch_result_papers", (result_id,), {}))
        return self.results[result_id]

    def read_file(self, path: str) -> str:
        self.calls.append(("read_file", (path,), {}))
        return self.files[path]

    def read_lines(self, path: str, start: int, end: int) -> list[str]:
        self.calls.append(("read_lines", (path, start, end), {}))
        return self.lines

    def operations(self) -> list[str]:
        return [name for name, _, _ in self.calls]


@pytest.fixture
def fake(monkeypatch) -> _FakeClient:
    client = _FakeClient(files={"/papers/PMC1/meta.json": '{"doi": "10.1/x"}'})

    def build(**kwargs: Any) -> _FakeClient:
        client.kwargs = kwargs
        return client

    monkeypatch.setattr(paperclip_retrieval, "PaperclipClient", build)
    return client


@pytest.fixture(autouse=True)
def _no_ambient_credentials(request, monkeypatch):
    """Keep a developer's own credentials out of the unit tests; the live tests need them."""
    if request.node.get_closest_marker("integration"):
        return
    for name in ("PAPERCLIP_API_KEY", "PAPERCLIP_TOKEN", "PAPERCLIP_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(paperclip_client, "_BACKOFF_SECONDS", 0.0)


# ---------------------------------------------------------------------------
# Client: credentials and the wire protocol
# ---------------------------------------------------------------------------


def test_client_requires_credentials():
    with pytest.raises(PaperclipAuthError, match="PAPERCLIP_API_KEY"):
        PaperclipClient(session=_FakeSession())


def test_client_prefers_the_api_key_and_sends_it_as_header():
    session = _FakeSession({_key("cat", path="/papers/PMC1/meta.json"): _mcp_reply("{}")})
    PaperclipClient(api_key="pk_test", token="tok", session=session).read_file("/papers/PMC1/meta.json")

    headers = session.posts[0]["headers"]
    assert headers["X-API-Key"] == "pk_test"
    assert "Authorization" not in headers


def test_config_token_from_environment_is_sent_as_bearer(monkeypatch):
    monkeypatch.setenv("PAPERCLIP_TOKEN", "env-token")
    session = _FakeSession({_key("cat", path="/papers/PMC1/meta.json"): _mcp_reply("{}")})
    config = PaperclipDocumentConfig()
    PaperclipClient(api_key=config.paperclip_api_key, token=config.paperclip_token, session=session).read_file(
        "/papers/PMC1/meta.json"
    )

    assert session.posts[0]["headers"]["Authorization"] == "Bearer env-token"


def test_call_names_the_tool_and_drops_unset_arguments(monkeypatch):
    monkeypatch.setenv("PAPERCLIP_BASE_URL", "https://paperclip.example/")
    session = _FakeSession({_key("grep", pattern="x", path="/papers/"): _mcp_reply("No matches")})
    client = PaperclipClient(api_key="k", timeout_seconds=42, session=session)

    client.grep("x", "/papers/", ignore_case=False, max_matches=None, context_lines=0)

    post = session.posts[0]
    assert post["url"] == "https://paperclip.example/mcp"
    assert post["timeout"] == 42
    assert post["json"]["method"] == "tools/call"
    assert post["json"]["params"] == {"name": "grep", "arguments": {"pattern": "x", "path": "/papers/"}}


def test_search_sends_typed_filters_and_reads_the_saved_result():
    reply = _mcp_reply("Found 2 papers  [s_52c029ac]\n\n  1. A paper\n     PMC1 · PMC · 2024")
    hits = [{"document_id": "PMC1"}, {"document_id": "PMC2"}, "not-a-paper"]
    client, session = _client(
        {
            _key(
                "search",
                query="BRCA1",
                source="pmc,abstracts",
                limit=5,
                exact=True,
                since="1y",
                year="2024",
                journal="Nature",
                author="Smith",
                article_type="review",
            ): reply
        },
        {"s_52c029ac": {"result_id": "s_52c029ac", "result_data": {"papers": hits}}},
    )

    result_id, papers = client.search(
        "BRCA1",
        sources=["pmc", "abstracts"],
        limit=5,
        exact=True,
        since="1y",
        year=2024,
        journal="Nature",
        author="Smith",
        article_type="review",
    )

    assert result_id == "s_52c029ac"
    assert papers == hits[:2]
    assert session.gets[0]["url"] == "https://paperclip.gxl.ai/api/cli/results/s_52c029ac"


def test_search_without_hits_saves_no_result():
    client, session = _client({_key("search", query="zzz", source="pmc", limit=3): _mcp_reply("Found 0 papers")})
    assert client.search("zzz", sources=["pmc"], limit=3) == (None, [])
    assert session.gets == []


def test_fetch_result_papers_accepts_the_results_alias_and_missing_data():
    client, _ = _client(
        results={
            "s_1": {"result_data": {"results": [{"document_id": "PMC1"}]}},
            "s_2": {"result_data": None},
            "s_3": {"output": "text only"},
        }
    )
    assert client.fetch_result_papers("s_1") == [{"document_id": "PMC1"}]
    assert client.fetch_result_papers("s_2") == []
    assert client.fetch_result_papers("s_3") == []


def test_lookup_with_a_saved_result_reads_it_back():
    client, _ = _client(
        {
            _key("lookup", field="author", value="Dauparas", limit=3): _mcp_reply(
                "Found 4 papers (showing 3)  [s_fccef39b]"
            )
        },
        {"s_fccef39b": {"result_data": {"papers": [{"document_id": "bio_29a07e4e799b"}]}}},
    )
    assert client.lookup("author", "Dauparas", limit=3) == ("s_fccef39b", [{"document_id": "bio_29a07e4e799b"}])


def test_lookup_without_a_saved_result_reads_each_listed_paper():
    listing = (
        "Found 1 papers\n\n  1. CAR-T cell therapy\n     Yaojie Kong; Jingyao Li\n"
        "     PMC11754230 · Frontiers in Immunology · 2025\n     https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11754230/\n"
    )
    meta = {"document_id": "PMC11754230", "journal": "Frontiers in Immunology", "pub_year": 2025}
    client, _ = _client(
        {
            _key("lookup", field="pmc", value="PMC11754230", limit=10): _mcp_reply(listing),
            _key("cat", path="/papers/PMC11754230/meta.json"): _mcp_reply(json.dumps(meta)),
        }
    )
    assert client.lookup("pmc", "PMC11754230", limit=10) == (None, [meta])


def test_read_lines_pages_past_the_character_limit_and_drops_the_cut_line():
    first = "L1: one\nL2: two\nL3: thr\n\n[output truncated at 50000 chars]\n"
    client, session = _client(
        {
            _key("cat", path=_CONTENT, lines="1-4"): _mcp_reply(first),
            _key("cat", path=_CONTENT, lines="3-4"): _mcp_reply("L3: three\nL4: four\n"),
        }
    )

    assert client.read_lines(_CONTENT, 1, 4) == ["L1: one", "L2: two", "L3: three", "L4: four"]
    assert len(session.posts) == 2


def test_read_lines_stops_at_the_end_of_the_file():
    client, _ = _client({_key("cat", path=_CONTENT, lines="470-500"): _mcp_reply("L470: a\nL471: b\nL472: c\n")})
    assert client.read_lines(_CONTENT, 470, 500) == ["L470: a", "L471: b", "L472: c"]


def test_read_lines_refuses_a_line_longer_than_one_read():
    client, _ = _client(
        {_key("cat", path=_CONTENT, lines="1-2"): _mcp_reply("L1: endless\n[output truncated at 50000 chars]")}
    )
    with pytest.raises(PaperclipError, match="line 1 is longer"):
        client.read_lines(_CONTENT, 1, 2)


@pytest.mark.parametrize(
    ("status", "exc_type", "fragment"),
    [
        (401, PaperclipAuthError, "rejected the credentials"),
        (403, PaperclipAuthError, "denied access"),
        (404, PaperclipCommandError, "no such resource"),
        (429, PaperclipError, "rate limit"),
        (500, PaperclipError, "HTTP 500"),
    ],
)
def test_http_errors_map_to_typed_exceptions(status, exc_type, fragment):
    client, _ = _client({_key("cat", path="/papers/PMC1/meta.json"): _FakeResponse(status, {"detail": "nope"}, "nope")})
    with pytest.raises(exc_type, match=fragment):
        client.read_file("/papers/PMC1/meta.json")


def test_jsonrpc_errors_and_tool_errors_raise_command_errors():
    client, _ = _client(
        {
            _key("cat", path="/papers/PMC1/meta.json"): {"jsonrpc": "2.0", "id": 1, "error": {"message": "boom"}},
            _key("cat", path="/papers/PMC2/meta.json"): _mcp_reply("No such file", is_error=True),
        }
    )
    with pytest.raises(PaperclipCommandError, match="boom"):
        client.read_file("/papers/PMC1/meta.json")
    with pytest.raises(PaperclipCommandError, match="No such file"):
        client.read_file("/papers/PMC2/meta.json")


def test_non_json_reply_raises():
    client, _ = _client({_key("cat", path="/papers/PMC1/meta.json"): _FakeResponse(200, None, text="<html>")})
    with pytest.raises(PaperclipError, match="non-JSON"):
        client.read_file("/papers/PMC1/meta.json")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def test_search_normalizes_citable_hits(fake):
    fake.saved_result = (
        "s_fixture",
        [
            {
                "document_id": "PMC11754230",
                "title": "CAR-T cell therapy",
                "source": "pmc",
                "abstract": "An abstract.",
                "chunk_text": "A matched passage.",
                "pmc_id": "PMC11754230",
                "doi": "10.1/example",
                "journal_title": "Frontiers in Immunology",
                "citation_canonical_work": {"pmid": "39850899"},
                "rerank_score": 0.9,
            },
            {"pmc_id": "PMC2", "source": "abstracts", "score": 0.4, "pub_date": "2024-09-25"},
        ],
    )

    output = run_paperclip_search(
        PaperclipSearchInput(query="CAR T"),
        PaperclipSearchConfig(paperclip_api_key="pk", limit=7, sources=["pmc", "abstracts", "pmc"]),
    )

    assert fake.kwargs["api_key"] == "pk"
    _, args, kwargs = fake.calls[0]
    assert args == ("CAR T",)
    assert kwargs["sources"] == ["pmc", "abstracts"]
    assert kwargs["limit"] == 7
    first, second = output.hits
    assert first.pmid == "39850899"
    assert first.journal == "Frontiers in Immunology"
    assert first.access_kind == "abstract"
    assert first.relevance_score == 0.9
    assert first.source_url == "https://paperclip.gxl.ai/citations/papers/PMC11754230"
    assert second.document_id == "PMC2"
    assert second.title == "PMC2"
    assert second.year == 2024  # from pub_date when the hit has no pub_year
    assert second.access_kind == "metadata"


def test_search_hits_split_authors_into_a_list(fake):
    fake.saved_result = (
        "s_1",
        [{"document_id": "PMC1", "source": "pmc", "authors": "Joseph L. Watson; David Juergens"}],
    )

    (hit,) = run_paperclip_search(PaperclipSearchInput(query="x"), PaperclipSearchConfig(paperclip_api_key="pk")).hits

    assert hit.authors == ["Joseph L. Watson", "David Juergens"]


def test_search_can_leave_out_abstracts(fake):
    fake.saved_result = ("s_1", [{"document_id": "PMC1", "source": "pmc", "abstract": "A long abstract."}])

    (hit,) = run_paperclip_search(
        PaperclipSearchInput(query="x"), PaperclipSearchConfig(paperclip_api_key="pk", include_abstracts=False)
    ).hits

    assert hit.abstract is None
    assert hit.access_kind == "metadata"


def test_search_rejects_hits_without_a_document_id(fake):
    fake.saved_result = ("s_1", [{"title": "anonymous"}])
    with pytest.raises(ValueError, match="stable document_id"):
        run_paperclip_search(PaperclipSearchInput(query="x"), PaperclipSearchConfig(paperclip_api_key="pk"))


def test_lookup_reads_the_journal_from_paper_metadata(fake):
    fake.saved_result = (None, [{"document_id": "PMC7", "title": "T", "journal": "Nature", "pub_year": 2025}])

    output = run_paperclip_lookup(
        PaperclipLookupInput(field="doi", value="10.1/x"), PaperclipLookupConfig(paperclip_api_key="pk", limit=3)
    )

    assert fake.calls == [("lookup", ("doi", "10.1/x"), {"limit": 3})]
    assert output.result_id is None
    assert (output.hits[0].journal, output.hits[0].year) == ("Nature", 2025)


def test_grep_returns_structured_matches_from_the_saved_result(fake):
    fake.grep_result_id = "s_8bd0dd27"
    fake.results = {
        "s_8bd0dd27": [
            {
                "document_id": "arx_2605.11347",
                "path": "/papers/arx_2605.11347/",
                "matches": [{"match": "ProteinMPNN", "source": "arxiv", "content": "uses ProteinMPNN"}],
            },
            {
                "document_id": "PMC5",
                "matches": [
                    {"match": "ProteinMPNN", "source": "pmc", "content": "first"},
                    {"match": "ProteinMPNN", "source": "pmc", "content": "second"},
                ],
            },
        ]
    }

    output = run_paperclip_grep(
        PaperclipGrepInput(pattern="ProteinMPNN"), PaperclipGrepConfig(paperclip_api_key="pk", max_matches=5)
    )

    _, args, kwargs = fake.calls[0]
    assert args == ("ProteinMPNN", "/papers/")
    assert kwargs == {"ignore_case": False, "max_matches": 5, "exhaustive": False}
    assert [(m.document_id, m.source, m.text) for m in output.matches] == [
        ("arx_2605.11347", "arxiv", "uses ProteinMPNN"),
        ("PMC5", "pmc", "first"),
        ("PMC5", "pmc", "second"),
    ]
    assert output.result_id == "s_8bd0dd27"


def test_grep_without_matches_reads_no_result(fake):
    output = run_paperclip_grep(PaperclipGrepInput(pattern="zzz"), PaperclipGrepConfig(paperclip_api_key="pk"))
    assert output.matches == []
    assert "fetch_result_papers" not in fake.operations()


def test_document_reads_a_line_range_by_default(fake):
    fake.lines = ["L1: Title", "L2: Abstract"]

    document = run_paperclip_document(
        PaperclipDocumentInput(document_id="PMC1"), PaperclipDocumentConfig(paperclip_api_key="pk", end_line=2)
    ).document

    assert ("read_lines", (_CONTENT, 1, 2), {}) in fake.calls
    assert document.access_kind == "excerpt"
    assert document.excerpt == "L1: Title\nL2: Abstract"
    assert (document.paper.document_id, document.paper.doi) == ("PMC1", "10.1/x")


def test_document_grep_parses_match_and_context_lines(fake):
    fake.grep_output = "L44-context before\nL45:the IC50 was 3 nM\nnot a line\n"

    document = run_paperclip_document(
        PaperclipDocumentInput(document_id="PMC1"),
        PaperclipDocumentConfig(paperclip_api_key="pk", read="grep", pattern="IC50", context_lines=1),
    ).document

    grep_call = next(call for call in fake.calls if call[0] == "grep")
    assert grep_call == ("grep", ("IC50", _CONTENT), {"ignore_case": False, "max_matches": 20, "context_lines": 1})
    assert [(m.line_number, m.text, m.is_match) for m in document.matches] == [
        (44, "context before", False),
        (45, "the IC50 was 3 nM", True),
    ]
    assert document.access_kind == "excerpt"


def test_document_grep_without_matches_returns_metadata_only(fake):
    fake.grep_output = "No matches"
    document = run_paperclip_document(
        PaperclipDocumentInput(document_id="PMC1"),
        PaperclipDocumentConfig(paperclip_api_key="pk", read="grep", pattern="absent"),
    ).document
    assert document.matches == []
    assert document.access_kind == "metadata"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [({"read": "grep"}, "pattern is required"), ({"start_line": 10, "end_line": 5}, "greater than or equal")],
)
def test_document_config_rejects_inconsistent_read_settings(overrides: dict[str, Any], message: str):
    with pytest.raises(ValidationError, match=message):
        PaperclipDocumentConfig(**overrides)


def test_document_full_text_reads_to_the_end_of_the_file(fake):
    fake.lines = ["L1: one", "L2: two", "L3: three"]

    document = run_paperclip_document(
        PaperclipDocumentInput(document_id="PMC1"), PaperclipDocumentConfig(paperclip_api_key="pk", read="full_text")
    ).document

    assert ("read_lines", (_CONTENT, 1, paperclip_client.LAST_LINE), {}) in fake.calls
    assert document.access_kind == "full_text"
    assert (document.content, document.line_count) == ("L1: one\nL2: two\nL3: three", 3)


def test_document_metadata_mode_reads_only_metadata(fake):
    document = run_paperclip_document(
        PaperclipDocumentInput(document_id="PMC1"), PaperclipDocumentConfig(paperclip_api_key="pk", read="metadata")
    ).document
    assert fake.operations() == ["read_file"]
    assert document.access_kind == "metadata"


def test_document_rejects_non_json_metadata(fake):
    fake.files["/papers/PMC1/meta.json"] = "cat: not found"
    with pytest.raises(PaperclipCommandError, match="non-JSON metadata"):
        run_paperclip_document(
            PaperclipDocumentInput(document_id="PMC1"), PaperclipDocumentConfig(paperclip_api_key="pk")
        )


@pytest.mark.parametrize(
    ("copied", "document_id"),
    [
        ("pmc11754230", "PMC11754230"),
        ("https://pmc.ncbi.nlm.nih.gov/articles/PMC11754230/", "PMC11754230"),
        ("https://paperclip.gxl.ai/citations/papers/bio_29a07e4e799b", "bio_29a07e4e799b"),
        ("arx_2605.11347", "arx_2605.11347"),
        ("arXiv:2605.11347v2", "arx_2605.11347v2"),
        ("https://arxiv.org/pdf/2605.11347.pdf", "arx_2605.11347"),
    ],
)
def test_document_id_accepts_ids_as_users_copy_them(copied, document_id):
    assert PaperclipDocumentInput(document_id=copied).document_id == document_id


def test_document_keeps_the_requested_id_over_the_one_in_meta_json(fake):
    """The arXiv meta.json records 2605.11347; the paper is only reachable as arx_2605.11347."""
    fake.files["/papers/arx_2605.11347/meta.json"] = json.dumps(
        {"document_id": "2605.11347", "title": "Noise optimization", "source": "arxiv", "pub_date": "2026-10-20"}
    )

    paper = run_paperclip_document(
        PaperclipDocumentInput(document_id="arx_2605.11347"),
        PaperclipDocumentConfig(paperclip_api_key="pk", read="metadata"),
    ).document.paper

    assert (paper.document_id, paper.year) == ("arx_2605.11347", 2026)
    assert paper.source_url == "https://paperclip.gxl.ai/citations/papers/arx_2605.11347"


def test_document_id_cannot_escape_virtual_papers_directory():
    with pytest.raises(ValidationError, match="without a path"):
        PaperclipDocumentInput(document_id="../../credentials.json")


# ---------------------------------------------------------------------------
# Live service
# ---------------------------------------------------------------------------

_LIVE = pytest.mark.skipif(not os.environ.get("PAPERCLIP_API_KEY"), reason="PAPERCLIP_API_KEY is not configured")


@pytest.mark.integration
@_LIVE
def test_paperclip_search_and_lookup_live():
    search = run_paperclip_search(
        PaperclipSearchInput(query="CAR T cell binder design"), PaperclipSearchConfig(limit=3)
    )
    assert search.result_id and len(search.hits) == 3
    lookup = run_paperclip_lookup(PaperclipLookupInput(field="pmc", value="PMC11754230"), PaperclipLookupConfig())
    assert [hit.document_id for hit in lookup.hits] == ["PMC11754230"]


@pytest.mark.integration
@_LIVE
def test_paperclip_grep_and_document_live():
    grep = run_paperclip_grep(PaperclipGrepInput(pattern="ProteinMPNN"), PaperclipGrepConfig(max_matches=3))
    assert grep.matches and all(match.text for match in grep.matches)
    document = run_paperclip_document(
        PaperclipDocumentInput(document_id="PMC11754230"), PaperclipDocumentConfig(read="full_text")
    ).document
    assert document.access_kind == "full_text"
    assert document.content is not None and len(document.content.splitlines()) == document.line_count
