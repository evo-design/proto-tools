"""Tests for Paperclip literature retrieval and its direct service client."""

from __future__ import annotations

import json
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
    PaperclipCommandResult,
    PaperclipError,
)
from proto_tools.tools.tool_registry import ToolRegistry

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


class _FakeSession:
    """Records requests and replays canned responses keyed by command or URL."""

    def __init__(self, commands: dict[str, Any] | None = None, results: dict[str, Any] | None = None):
        self.commands = commands or {}
        self.results = results or {}
        self.posts: list[dict[str, Any]] = []
        self.gets: list[dict[str, Any]] = []

    def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str], timeout: float) -> _FakeResponse:
        self.posts.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        command = json["params"]["arguments"]["command"]
        reply = self.commands[command]
        return reply if isinstance(reply, _FakeResponse) else _FakeResponse(200, reply)

    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> _FakeResponse:
        self.gets.append({"url": url, "headers": headers, "timeout": timeout})
        reply = self.results[url.rsplit("/", 1)[-1]]
        return reply if isinstance(reply, _FakeResponse) else _FakeResponse(200, reply)


class _FakeClient:
    """Stand-in for PaperclipClient at the tool boundary."""

    def __init__(
        self,
        *,
        files: dict[str, str] | None = None,
        sizes: dict[str, int] | None = None,
        grep_output: str = "",
        grep_result_id: str | None = None,
        **kwargs,
    ):
        self.kwargs = kwargs
        self.files = files or {}
        self.sizes = sizes or {}
        self.grep_output = grep_output
        self.grep_result_id = grep_result_id
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.saved_result: tuple[str | None, list[dict[str, Any]]] = (None, [])

    def search(self, query: str, **kwargs: Any) -> tuple[str | None, list[dict[str, Any]]]:
        self.calls.append(("search", (query,), kwargs))
        return self.saved_result

    def lookup(self, field: str, value: str, **kwargs: Any) -> tuple[str | None, list[dict[str, Any]]]:
        self.calls.append(("lookup", (field, value), kwargs))
        return self.saved_result

    def grep(self, pattern: str, path: str, **kwargs: Any) -> PaperclipCommandResult:
        self.calls.append(("grep", (pattern, path), kwargs))
        return PaperclipCommandResult(command="grep", output=self.grep_output, result_id=self.grep_result_id)

    def read_file(self, path: str) -> str:
        self.calls.append(("read_file", (path,), {}))
        return self.files[path]

    def read_lines(self, path: str, start: int, end: int) -> str:
        self.calls.append(("read_lines", (path, start, end), {}))
        return self.files.get(path, "")

    def byte_count(self, path: str) -> int | None:
        return self.sizes.get(path)

    def operations(self) -> list[str]:
        return [name for name, _, _ in self.calls]


def _install_fake_client(monkeypatch, fake: _FakeClient) -> None:
    def build(**kwargs: Any) -> _FakeClient:
        fake.kwargs = kwargs
        return fake

    monkeypatch.setattr(paperclip_retrieval, "PaperclipClient", build)


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch):
    for name in ("PAPERCLIP_API_KEY", "PAPERCLIP_BEARER_TOKEN", "PAPERCLIP_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(paperclip_client, "_BACKOFF_SECONDS", 0.0)


# ---------------------------------------------------------------------------
# Client: credentials and routing
# ---------------------------------------------------------------------------


def test_client_requires_credentials():
    with pytest.raises(PaperclipAuthError, match="PAPERCLIP_API_KEY"):
        PaperclipClient(session=_FakeSession())


def test_client_prefers_explicit_api_key_and_sends_it_as_header(monkeypatch):
    monkeypatch.setenv("PAPERCLIP_BEARER_TOKEN", "env-bearer")
    session = _FakeSession(commands={"wc /papers/PMC1/content.lines": _mcp_reply("1 2 8 /papers/PMC1/content.lines")})
    client = PaperclipClient(api_key="pk_test", session=session)

    assert client.byte_count("/papers/PMC1/content.lines") == 8
    headers = session.posts[0]["headers"]
    assert headers["X-API-Key"] == "pk_test"
    assert "Authorization" not in headers


def test_client_falls_back_to_bearer_token_from_environment(monkeypatch):
    monkeypatch.setenv("PAPERCLIP_BEARER_TOKEN", "env-bearer")
    session = _FakeSession(commands={"cat /papers/PMC1/meta.json": _mcp_reply("{}")})
    PaperclipClient(session=session).read_file("/papers/PMC1/meta.json")

    assert session.posts[0]["headers"]["Authorization"] == "Bearer env-bearer"


def test_client_routes_to_mcp_on_the_public_origin_and_papers_on_the_mcp_origin():
    assert PaperclipClient(api_key="k", session=_FakeSession()).command_url == "https://paperclip.gxl.ai/mcp"
    direct = PaperclipClient(api_key="k", base_url="https://mcp.paperclip.gxl.ai/", session=_FakeSession())
    assert direct.command_url == "https://mcp.paperclip.gxl.ai/papers"


def test_client_reads_base_url_from_environment(monkeypatch):
    monkeypatch.setenv("PAPERCLIP_BASE_URL", "https://paperclip.example/")
    client = PaperclipClient(api_key="k", session=_FakeSession())
    assert client.base_url == "https://paperclip.example"


# ---------------------------------------------------------------------------
# Client: wire protocol
# ---------------------------------------------------------------------------


def test_execute_sends_a_complete_plain_jsonrpc_tool_call():
    session = _FakeSession(commands={"cat /papers/PMC1/meta.json": _mcp_reply('{"a": 1}')})
    client = PaperclipClient(api_key="k", timeout_seconds=42, session=session)

    result = client.execute("cat", ["/papers/PMC1/meta.json"])

    assert result.output == '{"a": 1}'
    post = session.posts[0]
    assert post["url"] == "https://paperclip.gxl.ai/mcp"
    assert post["timeout"] == 42
    assert post["json"]["method"] == "tools/call"
    assert post["json"]["params"]["name"] == "paperclip"
    arguments = post["json"]["params"]["arguments"]
    assert arguments["skip_truncation"] is True
    assert arguments["plain"] is True
    assert arguments["no_color"] is True


def test_execute_shell_quotes_arguments():
    command = "search -e 'CAR T \"binder\" design' -n 5 -s pmc,abstracts"
    session = _FakeSession(commands={command: _mcp_reply("no hits")})
    client = PaperclipClient(api_key="k", session=session)

    result_id, papers = client.search('CAR T "binder" design', sources=["pmc", "abstracts"], limit=5, exact=True)

    assert (result_id, papers) == (None, [])
    assert session.posts[0]["json"]["params"]["arguments"]["command"] == command


def test_search_fetches_structured_hits_from_the_saved_result():
    hits = [{"document_id": "PMC1"}, {"document_id": "PMC2"}, "not-a-paper"]
    session = _FakeSession(
        commands={"search 'CAR T' -n 10 -s pmc --since 2024": _mcp_reply("2 hits saved as s_0abc12\n1. ...")},
        results={"s_0abc12": {"result_id": "s_0abc12", "result_data": {"papers": hits}}},
    )
    client = PaperclipClient(api_key="k", session=session)

    result_id, papers = client.search("CAR T", sources=["pmc"], limit=10, since="2024")

    assert result_id == "s_0abc12"
    assert papers == hits[:2]
    assert session.gets[0]["url"] == "https://paperclip.gxl.ai/api/cli/results/s_0abc12"
    assert session.gets[0]["headers"]["X-API-Key"] == "k"


def test_fetch_result_papers_accepts_the_results_alias_and_missing_data():
    session = _FakeSession(
        results={
            "s_1": {"result_data": {"results": [{"document_id": "PMC1"}]}},
            "s_2": {"result_data": None},
            "s_3": {"output": "text only"},
        }
    )
    client = PaperclipClient(api_key="k", session=session)

    assert client.fetch_result_papers("s_1") == [{"document_id": "PMC1"}]
    assert client.fetch_result_papers("s_2") == []
    assert client.fetch_result_papers("s_3") == []


def _command_client(*commands: str) -> tuple[PaperclipClient, _FakeSession]:
    session = _FakeSession(commands={command: _mcp_reply("") for command in commands})
    return PaperclipClient(api_key="k", session=session), session


def test_search_sends_ranking_and_metadata_filters_as_paperclip_flags():
    command = (
        "search BRCA1 -n 5 -s papers --ranking bm25 --since 2024-01-01 --year 2024 "
        "--journal Nature --author Smith --article-type review"
    )
    client, session = _command_client(command)
    client.search(
        "BRCA1",
        sources=["papers"],
        limit=5,
        ranking="bm25",
        since="2024-01-01",
        year=2024,
        journal="Nature",
        author="Smith",
        article_type="review",
    )
    assert session.posts[0]["json"]["params"]["arguments"]["command"] == command


def test_lookup_saves_hits_and_reads_them_back():
    session = _FakeSession(
        commands={"lookup doi 10.1/x -n 3": _mcp_reply("1 paper saved as s_00beef")},
        results={"s_00beef": {"result_data": {"papers": [{"document_id": "PMC1"}]}}},
    )
    result_id, papers = PaperclipClient(api_key="k", session=session).lookup("doi", "10.1/x", limit=3)
    assert (result_id, papers) == ("s_00beef", [{"document_id": "PMC1"}])


def test_grep_sends_match_limit_context_and_scan_flags():
    command = "grep -i -m 20 -B 2 -A 2 --exhaustive 'BRCA[12]' /papers/"
    client, session = _command_client(command)
    client.grep("BRCA[12]", "/papers/", ignore_case=True, max_matches=20, context_lines=2, exhaustive=True)
    assert session.posts[0]["json"]["params"]["arguments"]["command"] == command


def test_read_lines_uses_an_inclusive_sed_range():
    command = "sed -n 10,20p /papers/PMC1/content.lines"
    client, session = _command_client(command)
    client.read_lines("/papers/PMC1/content.lines", 10, 20)
    assert session.posts[0]["json"]["params"]["arguments"]["command"] == command


def test_byte_count_returns_none_for_unparseable_output():
    session = _FakeSession(commands={"wc /papers/PMC1/content.lines": _mcp_reply("wc: no such file")})
    assert PaperclipClient(api_key="k", session=session).byte_count("/papers/PMC1/content.lines") is None


# ---------------------------------------------------------------------------
# Client: errors
# ---------------------------------------------------------------------------


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
    session = _FakeSession(
        commands={"cat /papers/PMC1/meta.json": _FakeResponse(status, {"detail": "nope"}, text="nope")}
    )
    client = PaperclipClient(api_key="k", session=session)
    with pytest.raises(exc_type, match=fragment):
        client.read_file("/papers/PMC1/meta.json")


def test_jsonrpc_error_and_is_error_results_raise_command_errors():
    session = _FakeSession(
        commands={
            "cat /papers/PMC1/meta.json": {"jsonrpc": "2.0", "id": 1, "error": {"code": -1, "message": "boom"}},
            "cat /papers/PMC2/meta.json": _mcp_reply("cat: /papers/PMC2/meta.json: No such file", is_error=True),
        }
    )
    client = PaperclipClient(api_key="k", session=session)
    with pytest.raises(PaperclipCommandError, match="boom"):
        client.read_file("/papers/PMC1/meta.json")
    with pytest.raises(PaperclipCommandError, match="No such file"):
        client.read_file("/papers/PMC2/meta.json")


def test_non_json_reply_raises():
    session = _FakeSession(commands={"cat /papers/PMC1/meta.json": _FakeResponse(200, None, text="<html>")})
    with pytest.raises(PaperclipError, match="non-JSON"):
        PaperclipClient(api_key="k", session=session).read_file("/papers/PMC1/meta.json")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def test_search_normalizes_citable_hits(monkeypatch):
    fake = _FakeClient()
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
                "citation_canonical_work": {"pmid": "39850899"},
                "rerank_score": 0.9,
            },
            {"pmc_id": "PMC2", "source": "abstracts", "score": 0.4},
        ],
    )
    _install_fake_client(monkeypatch, fake)

    output = run_paperclip_search(
        PaperclipSearchInput(query="CAR T"),
        PaperclipSearchConfig(paperclip_api_key="pk", limit=7, sources=["pmc", "abstracts", "pmc"], ranking="hybrid"),
    )

    assert fake.kwargs["api_key"] == "pk"
    _, args, kwargs = fake.calls[0]
    assert args == ("CAR T",)
    assert kwargs["sources"] == ["pmc", "abstracts"]
    assert kwargs["limit"] == 7
    assert kwargs["ranking"] == "hybrid"
    assert output.result_id == "s_fixture"
    first, second = output.hits
    assert first.document_id == "PMC11754230"
    assert first.pmid == "39850899"
    assert first.access_kind == "abstract"
    assert first.relevance_score == 0.9
    assert first.provenance.provider == "Paperclip"
    assert first.provenance.details == {}
    assert second.document_id == "PMC2"
    assert second.title == "PMC2"
    assert second.access_kind == "metadata"
    assert second.relevance_score == 0.4


def test_search_rejects_hits_without_a_document_id(monkeypatch):
    fake = _FakeClient()
    fake.saved_result = ("s_1", [{"title": "anonymous"}])
    _install_fake_client(monkeypatch, fake)
    with pytest.raises(ValueError, match="stable document_id"):
        run_paperclip_search(PaperclipSearchInput(query="x"), PaperclipSearchConfig(paperclip_api_key="pk"))


def test_lookup_returns_normalized_hits(monkeypatch):
    fake = _FakeClient()
    fake.saved_result = ("s_2", [{"document_id": "PMC7", "title": "T", "source": "pmc", "doi": "10.1/x"}])
    _install_fake_client(monkeypatch, fake)

    output = run_paperclip_lookup(
        PaperclipLookupInput(field="doi", value="10.1/x"), PaperclipLookupConfig(paperclip_api_key="pk", limit=3)
    )

    assert fake.calls == [("lookup", ("doi", "10.1/x"), {"limit": 3})]
    assert output.result_id == "s_2"
    assert [hit.document_id for hit in output.hits] == ["PMC7"]
    assert output.hits[0].provenance.request_parameters == {"operation": "lookup", "source": "pmc"}


def test_lookup_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        PaperclipLookupInput(field="sequence", value="MKT")


def test_grep_collects_matching_papers_in_order_of_first_match(monkeypatch):
    output_text = (
        "/papers/PMC5/content.lines:L12: uses ProteinMPNN\n"
        "/papers/bio_ab12/content.lines:L3: ProteinMPNN redesign\n"
        "/papers/PMC5/content.lines:L40: ProteinMPNN again\n\n"
    )
    fake = _FakeClient(grep_output=output_text, grep_result_id="s_9")
    _install_fake_client(monkeypatch, fake)

    output = run_paperclip_grep(
        PaperclipGrepInput(pattern="ProteinMPNN"), PaperclipGrepConfig(paperclip_api_key="pk", max_matches=5)
    )

    _, args, kwargs = fake.calls[0]
    assert args == ("ProteinMPNN", "/papers/")
    assert kwargs == {"ignore_case": False, "max_matches": 5, "exhaustive": False}
    assert output.document_ids == ["PMC5", "bio_ab12"]
    assert len(output.lines) == 3
    assert output.result_id == "s_9"
    assert output.provenance.request_parameters["pattern"] == "ProteinMPNN"


def test_document_reads_a_line_range_by_default(monkeypatch):
    excerpt = "L1 Title\nL2 Abstract\n"
    fake = _FakeClient(files={"/papers/PMC1/meta.json": '{"license": "CC BY"}', "/papers/PMC1/content.lines": excerpt})
    _install_fake_client(monkeypatch, fake)

    document = run_paperclip_document(
        PaperclipDocumentInput(document_id="PMC1"), PaperclipDocumentConfig(paperclip_api_key="pk", end_line=2)
    ).document

    assert fake.operations() == ["read_file", "read_lines"]
    assert fake.calls[1][1] == ("/papers/PMC1/content.lines", 1, 2)
    assert document.access_kind == "excerpt"
    assert document.excerpt == excerpt
    assert document.content is None
    assert document.content_provenance is not None
    assert document.content_provenance.matches(excerpt.encode())
    assert document.content_provenance.request_parameters == {"operation": "lines", "start_line": 1, "end_line": 2}
    assert document.metadata_provenance.details == {"license": "CC BY"}


def test_document_grep_parses_match_and_context_lines(monkeypatch):
    fake = _FakeClient(
        files={"/papers/PMC1/meta.json": "{}"},
        grep_output="L44- context before\nL45: the IC50 was 3 nM\nnot a line\n",
    )
    _install_fake_client(monkeypatch, fake)

    document = run_paperclip_document(
        PaperclipDocumentInput(document_id="PMC1"),
        PaperclipDocumentConfig(paperclip_api_key="pk", read="grep", pattern="IC50", context_lines=1),
    ).document

    assert fake.calls[1] == (
        "grep",
        ("IC50", "/papers/PMC1/content.lines"),
        {"ignore_case": False, "context_lines": 1},
    )
    assert [(m.line_number, m.text, m.is_match) for m in document.matches] == [
        (44, "context before", False),
        (45, "the IC50 was 3 nM", True),
    ]
    assert document.access_kind == "excerpt"


def test_document_grep_without_matches_returns_metadata_only(monkeypatch):
    fake = _FakeClient(files={"/papers/PMC1/meta.json": "{}"}, grep_output="no matches")
    _install_fake_client(monkeypatch, fake)
    document = run_paperclip_document(
        PaperclipDocumentInput(document_id="PMC1"),
        PaperclipDocumentConfig(paperclip_api_key="pk", read="grep", pattern="absent"),
    ).document
    assert document.matches == []
    assert document.access_kind == "metadata"
    assert document.content_provenance is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [({"read": "grep"}, "pattern is required"), ({"start_line": 10, "end_line": 5}, "greater than or equal")],
)
def test_document_config_rejects_inconsistent_read_settings(overrides: dict[str, Any], message: str):
    with pytest.raises(ValidationError, match=message):
        PaperclipDocumentConfig(**overrides)


def test_document_full_text_preserves_exact_text_and_checksum(monkeypatch):
    metadata = json.dumps({"source": "pmc", "license": "CC BY"})
    content = "L1 First line\nL2 Second line\n"
    fake = _FakeClient(
        files={"/papers/PMC11754230/meta.json": metadata, "/papers/PMC11754230/content.lines": content},
        sizes={"/papers/PMC11754230/content.lines": len(content.encode())},
    )
    _install_fake_client(monkeypatch, fake)

    document = run_paperclip_document(
        PaperclipDocumentInput(document_id="PMC11754230"),
        PaperclipDocumentConfig(paperclip_api_key="pk", read="full_text"),
    ).document

    assert document.access_kind == "full_text"
    assert document.content == content
    assert document.expected_content_bytes == len(content.encode())
    assert document.metadata_provenance.matches(metadata.encode())
    assert document.content_provenance is not None
    assert document.content_provenance.matches(content.encode())
    assert document.content_provenance.details == {"license": "CC BY"}


def test_document_full_text_keeps_short_content_as_preview(monkeypatch):
    fake = _FakeClient(
        files={"/papers/PMC1/meta.json": "{}", "/papers/PMC1/content.lines": "L1 partial\n"},
        sizes={"/papers/PMC1/content.lines": 5000},
    )
    _install_fake_client(monkeypatch, fake)

    document = run_paperclip_document(
        PaperclipDocumentInput(document_id="PMC1"), PaperclipDocumentConfig(paperclip_api_key="pk", read="full_text")
    ).document

    assert document.access_kind == "inaccessible"
    assert document.content is None
    assert document.content_preview == "L1 partial\n"
    assert document.content_provenance is None


def test_document_metadata_mode_reads_only_metadata(monkeypatch):
    fake = _FakeClient(files={"/papers/PMC1/meta.json": '{"year": 2024}'})
    _install_fake_client(monkeypatch, fake)

    document = run_paperclip_document(
        PaperclipDocumentInput(document_id="PMC1"), PaperclipDocumentConfig(paperclip_api_key="pk", read="metadata")
    ).document

    assert fake.operations() == ["read_file"]
    assert document.access_kind == "metadata"
    assert document.metadata_provenance.version == "2024"


def test_document_rejects_non_json_metadata(monkeypatch):
    fake = _FakeClient(files={"/papers/PMC1/meta.json": "cat: not found"})
    _install_fake_client(monkeypatch, fake)
    with pytest.raises(PaperclipCommandError, match="non-JSON metadata"):
        run_paperclip_document(
            PaperclipDocumentInput(document_id="PMC1"), PaperclipDocumentConfig(paperclip_api_key="pk")
        )


def test_document_id_cannot_escape_virtual_papers_directory():
    with pytest.raises(ValidationError, match="without a path"):
        PaperclipDocumentInput(document_id="../../credentials.json")


def test_search_config_requires_a_source():
    with pytest.raises(ValidationError, match="at least one"):
        PaperclipSearchConfig(sources=[])


def test_api_key_is_excluded_from_the_cache_key():
    with_key = PaperclipSearchConfig(paperclip_api_key="pk_one")
    other_key = PaperclipSearchConfig(paperclip_api_key="pk_two")
    assert with_key.cache_key() == other_key.cache_key()


@pytest.mark.parametrize(
    "key",
    [
        "paperclip-literature-search",
        "paperclip-literature-lookup",
        "paperclip-literature-grep",
        "paperclip-literature-document",
    ],
)
def test_registry_runs_paperclip_in_process(key):
    spec = ToolRegistry.get(key)
    assert spec.has_standalone_env is False
    assert spec.local_cpu is True
    assert spec.local_only
