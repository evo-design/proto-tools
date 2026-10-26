"""Search, look up, grep, and read literature through Paperclip without a local corpus.

The four tools follow the workflow GXL documents for Paperclip: ``search`` to find papers by topic,
``lookup`` to resolve a DOI, PMID, or other metadata field, corpus-wide ``grep`` to find papers
that mention exact text, and targeted reads of one paper (a line range or the lines matching a
pattern) rather than its whole full text.
"""

from __future__ import annotations

import json
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from proto_tools.tools.database_retrieval.paperclip.paperclip_client import PaperclipClient, PaperclipCommandError
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseConfig, BaseToolInput, BaseToolOutput, ConfigField, InputField

PaperclipSource = Literal["pmc", "papers", "abstracts", "biorxiv", "medrxiv", "arxiv"]
PaperclipRanking = Literal["hybrid", "bm25", "vector", "analogical"]
PaperclipLookupField = Literal["doi", "pmid", "pmc", "arxiv", "title", "author", "journal", "year"]
_ID_PATTERN = r"PMC\d+|bio_[A-Za-z0-9]+|med_[A-Za-z0-9]+|arx_[A-Za-z0-9]+"
_DOCUMENT_ID = re.compile(rf"^(?:{_ID_PATTERN})$")
_DOCUMENT_ID_IN_TEXT = re.compile(rf"\b({_ID_PATTERN})\b")
_GREP_LINE = re.compile(r"^L(\d+)(:|-)\s?(.*)$")
_PAPER_URL = "https://paperclip.gxl.ai/citations/papers/{document_id}"
_SERVICE_URL = "https://paperclip.gxl.ai/"


class LiteratureAccessKind(str, Enum):
    """Granularity of literature content actually returned."""

    METADATA = "metadata"
    ABSTRACT = "abstract"
    EXCERPT = "excerpt"
    FULL_TEXT = "full_text"
    INACCESSIBLE = "inaccessible"


class _PaperclipConfig(BaseConfig):
    """Shared Paperclip client settings.

    Attributes:
        paperclip_base_url (str | None): Optional Paperclip service override.
        paperclip_api_key (str | None): Optional API key; falls back to the environment.
        timeout_seconds (float): Maximum duration of one service operation.
    """

    paperclip_base_url: str | None = ConfigField(
        default_factory=lambda: os.environ.get("PAPERCLIP_BASE_URL"),
        title="Paperclip Base URL",
        description="Optional Paperclip service override",
    )
    paperclip_api_key: str | None = ConfigField(
        default_factory=lambda: os.environ.get("PAPERCLIP_API_KEY"),
        title="Paperclip API Key",
        description="Optional API key; otherwise PAPERCLIP_API_KEY or PAPERCLIP_BEARER_TOKEN is used",
        include_in_key=False,
    )
    timeout_seconds: float = ConfigField(
        default=120.0,
        gt=0,
        le=600,
        title="Timeout Seconds",
        description="Maximum duration of one Paperclip service operation",
    )


class _JsonOutput(BaseToolOutput):
    @property
    def output_format_options(self) -> list[str]:
        return ["json"]

    @property
    def output_format_default(self) -> str:
        return "json"

    def _export_output(self, export_path: Any, file_format: str) -> None:
        if file_format != "json":
            raise ValueError(f"Unsupported format: {file_format}")
        Path(export_path).with_suffix(".json").write_text(
            json.dumps(self.model_dump(mode="json"), indent=2), encoding="utf-8"
        )


class PaperclipSearchHit(BaseModel):
    """One normalized, citable Paperclip paper returned by search or lookup."""

    document_id: str = Field(title="Document ID", description="Stable Paperclip document identifier")
    title: str = Field(title="Title", description="Publication title")
    authors: str | None = Field(default=None, title="Authors", description="Publication author text")
    journal: str | None = Field(default=None, title="Journal", description="Journal or source title")
    publication_date: str | None = Field(default=None, title="Publication Date", description="Publication date text")
    year: int | None = Field(default=None, title="Year", description="Publication year")
    doi: str | None = Field(default=None, title="DOI", description="Digital object identifier")
    pmid: str | None = Field(default=None, title="PMID", description="PubMed identifier")
    pmcid: str | None = Field(default=None, title="PMCID", description="PubMed Central identifier")
    source: str = Field(title="Source", description="Paperclip source corpus")
    abstract: str | None = Field(default=None, title="Abstract", description="Publication abstract when available")
    matched_text: str | None = Field(default=None, title="Matched Text", description="Text fragment matched by search")
    relevance_score: float | None = Field(
        default=None, title="Relevance Score", description="Paperclip relevance score"
    )
    access_kind: LiteratureAccessKind = Field(title="Access Kind", description="Content granularity returned")
    source_url: str = Field(title="Source URL", description="Paperclip citation page for the paper")
    raw_metadata: dict[str, Any] = Field(
        default_factory=dict, title="Raw Metadata", description="Unmodified Paperclip result", repr=False
    )


def _client(config: _PaperclipConfig) -> PaperclipClient:
    """Build the authenticated service client from the tool config."""
    return PaperclipClient(
        api_key=config.paperclip_api_key,
        base_url=config.paperclip_base_url,
        timeout_seconds=config.timeout_seconds,
    )


def _normalize_hit(item: dict[str, Any], operation: str) -> PaperclipSearchHit:
    document_id = str(item.get("document_id") or item.get("pmc_id") or "")
    if not document_id:
        raise ValueError(f"Paperclip {operation} result did not include a stable document_id")
    abstract = item.get("abstract")
    source = str(item.get("source") or item.get("corpus") or "unknown")
    pmid = item.get("pmid") or (item.get("citation_canonical_work") or {}).get("pmid")
    return PaperclipSearchHit(
        document_id=document_id,
        title=str(item.get("title") or item.get("citation_title") or document_id),
        authors=item.get("authors"),
        journal=item.get("journal_title"),
        publication_date=item.get("pub_date"),
        year=item.get("pub_year"),
        doi=item.get("doi") or item.get("citation_doi"),
        pmid=str(pmid) if pmid else None,
        pmcid=item.get("pmc_id"),
        source=source,
        abstract=abstract,
        matched_text=item.get("chunk_text"),
        relevance_score=item.get("rerank_score") if item.get("rerank_score") is not None else item.get("score"),
        access_kind=LiteratureAccessKind.ABSTRACT if abstract else LiteratureAccessKind.METADATA,
        source_url=_PAPER_URL.format(document_id=document_id),
        raw_metadata=item,
    )


# ============================================================================
# Search
# ============================================================================


class PaperclipSearchInput(BaseToolInput):
    """Biomedical literature query submitted to Paperclip.

    Attributes:
        query (str): Biomedical literature search query.
    """

    query: str = InputField(
        min_length=1,
        max_length=2000,
        title="Query",
        description="Biomedical literature search query",
    )


class PaperclipSearchConfig(_PaperclipConfig):
    """Paperclip corpus selection, ranking, and metadata filters.

    Attributes:
        paperclip_base_url (str | None): Optional Paperclip service override.
        paperclip_api_key (str | None): Optional API key; falls back to the environment.
        timeout_seconds (float): Maximum duration of one service operation.
        sources (list[PaperclipSource]): Literature corpora to search.
        limit (int): Maximum number of returned hits.
        ranking (PaperclipRanking | None): Ranking mode; None uses Paperclip's hybrid default.
        exact (bool): Whether to use exact-text search.
        since (str | None): Optional date lower bound for PMC, bioRxiv, and arXiv.
        year (int | None): Only return papers published in this year.
        journal (str | None): Only return papers from this journal.
        author (str | None): Only return papers by this author.
        article_type (str | None): Only return papers of this article type.
    """

    sources: list[PaperclipSource] = ConfigField(
        default_factory=lambda: ["pmc"],
        title="Sources",
        description="Paperclip corpora to search; 'papers' covers all four full-text corpora",
    )
    limit: int = ConfigField(default=10, ge=1, le=100, title="Result Limit", description="Maximum search hits")
    ranking: PaperclipRanking | None = ConfigField(
        default=None,
        title="Ranking",
        description="Ranking mode; analogical finds cross-domain method analogues from an abstract",
    )
    exact: bool = ConfigField(default=False, title="Exact Search", description="Use Paperclip exact-text search")
    since: str | None = ConfigField(
        default=None,
        title="Since",
        description="Optional date lower bound for PMC, bioRxiv, and arXiv",
    )
    year: int | None = ConfigField(default=None, title="Year", description="Only return papers from this year")
    journal: str | None = ConfigField(default=None, title="Journal", description="Only return papers from this journal")
    author: str | None = ConfigField(default=None, title="Author", description="Only return papers by this author")
    article_type: str | None = ConfigField(
        default=None, title="Article Type", description="Only return papers of this article type"
    )

    @field_validator("sources")
    @classmethod
    def _deduplicate_sources(cls, value: list[PaperclipSource]) -> list[PaperclipSource]:
        if not value:
            raise ValueError("sources must contain at least one Paperclip corpus")
        return list(dict.fromkeys(value))


class PaperclipSearchOutput(_JsonOutput):
    """Structured Paperclip search results.

    Attributes:
        query (str): Submitted literature query.
        sources (list[str]): Searched Paperclip corpora.
        result_id (str | None): Paperclip search-result identifier.
        hits (list[PaperclipSearchHit]): Normalized literature matches.
    """

    query: str = Field(title="Query", description="Submitted literature query")
    sources: list[str] = Field(title="Sources", description="Searched Paperclip corpora")
    result_id: str | None = Field(default=None, title="Result ID", description="Paperclip result identifier")
    hits: list[PaperclipSearchHit] = Field(
        default_factory=list, title="Hits", description="Normalized literature matches"
    )


def _search_example() -> Any:
    return PaperclipSearchInput(query="CAR T cell binder design")


@tool(
    key="paperclip-literature-search",
    local_only=(
        "paperclip-literature-search does not use a gpu and does not need an environment, so it can run in process"
    ),
    label="Paperclip Literature Search",
    category="database_retrieval",
    input_class=PaperclipSearchInput,
    config_class=PaperclipSearchConfig,
    output_class=PaperclipSearchOutput,
    description="Search Paperclip full-text and abstract literature corpora for citable papers",
    uses_gpu=False,
    example_input=_search_example,
    cacheable=True,
)
def run_paperclip_search(
    inputs: PaperclipSearchInput, config: PaperclipSearchConfig, instance: Any = None
) -> PaperclipSearchOutput:
    """Search configured Paperclip literature corpora and normalize the hits.

    Args:
        inputs (PaperclipSearchInput): The literature query.
        config (PaperclipSearchConfig): Corpus selection, ranking, filters, and client settings.
        instance (Any): Unused; the tool runs in process.

    Returns:
        PaperclipSearchOutput: Normalized, citable hits, each with its citation URL.
    """
    del instance
    result_id, papers = _client(config).search(
        inputs.query,
        sources=config.sources,
        limit=config.limit,
        ranking=config.ranking,
        exact=config.exact,
        since=config.since,
        year=config.year,
        journal=config.journal,
        author=config.author,
        article_type=config.article_type,
    )
    return PaperclipSearchOutput(
        query=inputs.query,
        sources=list(config.sources),
        result_id=result_id,
        hits=[_normalize_hit(item, "search") for item in papers],
    )


# ============================================================================
# Lookup
# ============================================================================


class PaperclipLookupInput(BaseToolInput):
    """Metadata field and value identifying papers in Paperclip.

    Attributes:
        field (PaperclipLookupField): Metadata field to match, such as doi or pmid.
        value (str): Value to look up in that field.
    """

    field: PaperclipLookupField = InputField(title="Field", description="Metadata field to match, such as doi or pmid")
    value: str = InputField(min_length=1, max_length=500, title="Value", description="Value to look up in that field")


class PaperclipLookupConfig(_PaperclipConfig):
    """Settings for Paperclip metadata lookup.

    Attributes:
        paperclip_base_url (str | None): Optional Paperclip service override.
        paperclip_api_key (str | None): Optional API key; falls back to the environment.
        timeout_seconds (float): Maximum duration of one service operation.
        limit (int): Maximum number of returned papers.
    """

    limit: int = ConfigField(default=10, ge=1, le=100, title="Result Limit", description="Maximum papers returned")


class PaperclipLookupOutput(_JsonOutput):
    """Papers matching one metadata field.

    Attributes:
        field (str): Metadata field that was matched.
        value (str): Value that was looked up.
        result_id (str | None): Paperclip result identifier.
        hits (list[PaperclipSearchHit]): Matching papers.
    """

    field: str = Field(title="Field", description="Metadata field that was matched")
    value: str = Field(title="Value", description="Value that was looked up")
    result_id: str | None = Field(default=None, title="Result ID", description="Paperclip result identifier")
    hits: list[PaperclipSearchHit] = Field(default_factory=list, title="Hits", description="Matching papers")


def _lookup_example() -> Any:
    return PaperclipLookupInput(field="pmc", value="PMC11754230")


@tool(
    key="paperclip-literature-lookup",
    local_only=(
        "paperclip-literature-lookup does not use a gpu and does not need an environment, so it can run in process"
    ),
    label="Paperclip Literature Lookup",
    category="database_retrieval",
    input_class=PaperclipLookupInput,
    config_class=PaperclipLookupConfig,
    output_class=PaperclipLookupOutput,
    description="Find Paperclip papers by DOI, PMID, PMC ID, arXiv ID, title, author, journal, or year",
    uses_gpu=False,
    example_input=_lookup_example,
    cacheable=True,
)
def run_paperclip_lookup(
    inputs: PaperclipLookupInput, config: PaperclipLookupConfig, instance: Any = None
) -> PaperclipLookupOutput:
    """Find papers whose metadata field matches a value, such as a DOI or PMID.

    Args:
        inputs (PaperclipLookupInput): The metadata field and value.
        config (PaperclipLookupConfig): Result limit and client settings.
        instance (Any): Unused; the tool runs in process.

    Returns:
        PaperclipLookupOutput: Matching papers, each with its citation URL.
    """
    del instance
    result_id, papers = _client(config).lookup(inputs.field, inputs.value, limit=config.limit)
    return PaperclipLookupOutput(
        field=inputs.field,
        value=inputs.value,
        result_id=result_id,
        hits=[_normalize_hit(item, "lookup") for item in papers],
    )


# ============================================================================
# Corpus grep
# ============================================================================


class PaperclipGrepInput(BaseToolInput):
    """Regular expression to find in the full text of every Paperclip paper.

    Attributes:
        pattern (str): Regular expression matched against paper full text.
    """

    pattern: str = InputField(
        min_length=1,
        max_length=500,
        title="Pattern",
        description="Regular expression matched against paper full text, such as a gene or accession",
    )


class PaperclipGrepConfig(_PaperclipConfig):
    """Settings for corpus-wide Paperclip grep.

    Attributes:
        paperclip_base_url (str | None): Optional Paperclip service override.
        paperclip_api_key (str | None): Optional API key; falls back to the environment.
        timeout_seconds (float): Maximum duration of one service operation.
        max_matches (int): Maximum matching lines returned.
        ignore_case (bool): Whether matching ignores case.
        exhaustive (bool): Whether to give the scan its full time budget.
    """

    max_matches: int = ConfigField(default=50, ge=1, le=1000, title="Max Matches", description="Maximum matching lines")
    ignore_case: bool = ConfigField(default=False, title="Ignore Case", description="Match regardless of case")
    exhaustive: bool = ConfigField(
        default=False,
        title="Exhaustive",
        description="Scan for the full time budget; use when a rare pattern returns nothing",
    )


class PaperclipGrepOutput(_JsonOutput):
    """Lines matching a pattern across the Paperclip paper corpus.

    Attributes:
        pattern (str): Submitted regular expression.
        result_id (str | None): Paperclip result identifier, when one was saved.
        document_ids (list[str]): Papers with at least one match, in order of first match.
        lines (list[str]): Matching lines exactly as Paperclip returned them.
        source_url (str): Paperclip service the corpus was searched on.
    """

    pattern: str = Field(title="Pattern", description="Submitted regular expression")
    result_id: str | None = Field(default=None, title="Result ID", description="Paperclip result identifier")
    document_ids: list[str] = Field(
        default_factory=list, title="Document IDs", description="Papers with a match, in order of first match"
    )
    lines: list[str] = Field(
        default_factory=list, title="Lines", description="Matching lines exactly as Paperclip returned them"
    )
    source_url: str = Field(title="Source URL", description="Paperclip service the corpus was searched on")


def _grep_example() -> Any:
    return PaperclipGrepInput(pattern="ProteinMPNN")


@tool(
    key="paperclip-literature-grep",
    local_only=(
        "paperclip-literature-grep does not use a gpu and does not need an environment, so it can run in process"
    ),
    label="Paperclip Literature Grep",
    category="database_retrieval",
    input_class=PaperclipGrepInput,
    config_class=PaperclipGrepConfig,
    output_class=PaperclipGrepOutput,
    description="Find papers whose full text matches a regular expression across the Paperclip corpus",
    uses_gpu=False,
    example_input=_grep_example,
    cacheable=True,
)
def run_paperclip_grep(
    inputs: PaperclipGrepInput, config: PaperclipGrepConfig, instance: Any = None
) -> PaperclipGrepOutput:
    """Find papers that mention exact text anywhere in their full text.

    Corpus-wide grep is time-bounded, so repeated calls can return a different set or order of
    matches; ``exhaustive`` gives the scan more time but does not make it ranked.

    Args:
        inputs (PaperclipGrepInput): The regular expression.
        config (PaperclipGrepConfig): Match limit, case handling, scan budget, and client settings.
        instance (Any): Unused; the tool runs in process.

    Returns:
        PaperclipGrepOutput: Matching lines and the papers they came from.
    """
    del instance
    result = _client(config).grep(
        inputs.pattern,
        "/papers/",
        ignore_case=config.ignore_case,
        max_matches=config.max_matches,
        exhaustive=config.exhaustive,
    )
    lines = [line for line in result.output.splitlines() if line.strip()]
    document_ids = list(
        dict.fromkeys(match.group(1) for line in lines for match in _DOCUMENT_ID_IN_TEXT.finditer(line))
    )
    return PaperclipGrepOutput(
        pattern=inputs.pattern,
        result_id=result.result_id,
        document_ids=document_ids,
        lines=lines,
        source_url=_SERVICE_URL,
    )


# ============================================================================
# Document
# ============================================================================


class PaperclipDocumentInput(BaseToolInput):
    """Stable Paperclip corpus document identifier.

    Attributes:
        document_id (str): Stable Paperclip document identifier.
    """

    document_id: str = InputField(
        title="Document ID",
        description="Paperclip corpus identifier, such as PMC11754230",
    )

    @field_validator("document_id")
    @classmethod
    def _safe_document_id(cls, value: str) -> str:
        if not _DOCUMENT_ID.fullmatch(value):
            raise ValueError("document_id must be a Paperclip paper identifier without a path")
        return value


class PaperclipDocumentConfig(_PaperclipConfig):
    """Which part of a Paperclip paper to read.

    Attributes:
        paperclip_base_url (str | None): Optional Paperclip service override.
        paperclip_api_key (str | None): Optional API key; falls back to the environment.
        timeout_seconds (float): Maximum duration of one service operation.
        read (Literal["metadata", "lines", "grep", "full_text"]): What to read beyond metadata.
        start_line (int): First line to read in ``lines`` mode (1-indexed, inclusive).
        end_line (int): Last line to read in ``lines`` mode (1-indexed, inclusive).
        pattern (str | None): Regular expression to find in ``grep`` mode.
        context_lines (int): Lines of context around each ``grep`` match.
        ignore_case (bool): Whether ``grep`` matching ignores case.
    """

    read: Literal["metadata", "lines", "grep", "full_text"] = ConfigField(
        default="lines",
        title="Read",
        description="What to read beyond metadata; GXL recommends lines or grep over full_text",
    )
    start_line: int = ConfigField(default=1, ge=1, title="Start Line", description="First line read in lines mode")
    end_line: int = ConfigField(default=100, ge=1, title="End Line", description="Last line read in lines mode")
    pattern: str | None = ConfigField(
        default=None, max_length=500, title="Pattern", description="Regular expression to find in grep mode"
    )
    context_lines: int = ConfigField(
        default=0, ge=0, le=20, title="Context Lines", description="Lines of context around each grep match"
    )
    ignore_case: bool = ConfigField(
        default=False, title="Ignore Case", description="Match grep patterns regardless of case"
    )

    @model_validator(mode="after")
    def _check_read_settings(self) -> PaperclipDocumentConfig:
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        if self.read == "grep" and not self.pattern:
            raise ValueError("pattern is required when read is 'grep'")
        return self


class PaperclipLineMatch(BaseModel):
    """One line returned by an in-paper grep."""

    line_number: int = Field(ge=1, title="Line Number", description="Line number in the paper, for citations")
    text: str = Field(title="Text", description="Text of the line")
    is_match: bool = Field(title="Is Match", description="True for a matching line, False for a context line")


class PaperclipDocument(BaseModel):
    """Metadata for one paper and whichever part of its text was requested."""

    document_id: str = Field(title="Document ID", description="Stable Paperclip document identifier")
    metadata: dict[str, Any] = Field(title="Metadata", description="Paperclip publication metadata")
    access_kind: LiteratureAccessKind = Field(title="Access Kind", description="Content granularity returned")
    excerpt: str | None = Field(
        default=None, title="Excerpt", description="Requested line range as Paperclip returned it", repr=False
    )
    matches: list[PaperclipLineMatch] = Field(
        default_factory=list, title="Matches", description="Lines matched in grep mode, with context lines"
    )
    content: str | None = Field(
        default=None, title="Content", description="Complete line-numbered full text", repr=False
    )
    content_preview: str | None = Field(
        default=None, title="Content Preview", description="Incomplete full text returned by Paperclip", repr=False
    )
    expected_content_bytes: int | None = Field(
        default=None, title="Expected Content Bytes", description="Full-text byte count Paperclip reports"
    )
    source_url: str = Field(title="Source URL", description="Paperclip citation page for the paper")


class PaperclipDocumentOutput(_JsonOutput):
    """One retrieved Paperclip literature document.

    Attributes:
        document (PaperclipDocument): Retrieved metadata and requested content.
    """

    document: PaperclipDocument = Field(title="Document", description="Retrieved literature document")


def _document_example() -> Any:
    return PaperclipDocumentInput(document_id="PMC11754230")


def _parse_grep_lines(output: str) -> list[PaperclipLineMatch]:
    """Parse ``L<n>: text`` match lines and ``L<n>- text`` context lines from an in-paper grep."""
    matches = []
    for raw_line in output.splitlines():
        parsed = _GREP_LINE.match(raw_line.strip())
        if parsed:
            matches.append(
                PaperclipLineMatch(
                    line_number=int(parsed.group(1)), text=parsed.group(3), is_match=parsed.group(2) == ":"
                )
            )
    return matches


@tool(
    key="paperclip-literature-document",
    local_only=(
        "paperclip-literature-document does not use a gpu and does not need an environment, so it can run in process"
    ),
    label="Paperclip Literature Document",
    category="database_retrieval",
    input_class=PaperclipDocumentInput,
    config_class=PaperclipDocumentConfig,
    output_class=PaperclipDocumentOutput,
    description="Read a Paperclip paper's metadata plus a line range, pattern matches, or full text",
    uses_gpu=False,
    example_input=_document_example,
    cacheable=True,
)
def run_paperclip_document(
    inputs: PaperclipDocumentInput, config: PaperclipDocumentConfig, instance: Any = None
) -> PaperclipDocumentOutput:
    """Read one paper's metadata and the requested part of its line-numbered text.

    ``lines`` mode reads a line range and ``grep`` mode returns the lines matching a pattern, which
    is how GXL recommends reading a paper. ``full_text`` reads the whole paper and labels it
    complete only when its length matches the byte count Paperclip reports; anything shorter is
    kept as a preview.

    Args:
        inputs (PaperclipDocumentInput): The Paperclip document identifier.
        config (PaperclipDocumentConfig): What to read and client settings.
        instance (Any): Unused; the tool runs in process.

    Returns:
        PaperclipDocumentOutput: Metadata and the requested content.
    """
    del instance
    client = _client(config)
    paper_dir = f"/papers/{inputs.document_id}"
    content_path = f"{paper_dir}/content.lines"
    metadata_output = client.read_file(f"{paper_dir}/meta.json")
    try:
        metadata = json.loads(metadata_output)
    except ValueError as exc:
        raise PaperclipCommandError(
            f"Paperclip returned non-JSON metadata for {inputs.document_id}: {metadata_output[:300]!r}"
        ) from exc
    if not isinstance(metadata, dict):
        raise PaperclipCommandError(f"Paperclip metadata for {inputs.document_id} is not a JSON object")
    document = PaperclipDocument(
        document_id=inputs.document_id,
        metadata=metadata,
        access_kind=LiteratureAccessKind.METADATA,
        source_url=_PAPER_URL.format(document_id=inputs.document_id),
    )
    if config.read == "lines":
        excerpt = client.read_lines(content_path, config.start_line, config.end_line)
        if excerpt.strip():
            document.excerpt = excerpt
            document.access_kind = LiteratureAccessKind.EXCERPT
        else:
            document.access_kind = LiteratureAccessKind.INACCESSIBLE
    elif config.read == "grep":
        pattern = config.pattern or ""
        output = client.grep(
            pattern, content_path, ignore_case=config.ignore_case, context_lines=config.context_lines
        ).output
        document.matches = _parse_grep_lines(output)
        if document.matches:
            document.access_kind = LiteratureAccessKind.EXCERPT
    elif config.read == "full_text":
        content_output = client.read_file(content_path)
        if content_output.strip():
            document.expected_content_bytes = client.byte_count(content_path)
            if document.expected_content_bytes == len(content_output.encode("utf-8")):
                document.content = content_output
                document.access_kind = LiteratureAccessKind.FULL_TEXT
            else:
                document.content_preview = content_output
                document.access_kind = LiteratureAccessKind.INACCESSIBLE
        else:
            document.access_kind = LiteratureAccessKind.INACCESSIBLE
    return PaperclipDocumentOutput(document=document)
