"""proto_tools/tools/database_retrieval/arxiv/arxiv_search.py.

arXiv search: arXiv API queries (``ti:``, ``au:``, ``abs:``, ``cat:``, ``all:`` fields with
``AND``/``OR``/``ANDNOT``) return preprints with their versioned IDs, abstracts, categories,
DOI and journal reference when present, and abstract/PDF links.

arXiv's API terms allow one request every three seconds over a single connection. Every request
this module makes, retries included, passes through one process-wide gate that enforces both, so
concurrent calls share the budget.
"""

import csv
import json
import logging
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from time import monotonic
from typing import Any, Literal

import requests
from pydantic import BaseModel, ConfigDict, Field, field_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    DeadlineExceededError,
    InputField,
    build_http_session,
    raise_for_status_with_body,
    remaining,
    sleep_within_deadline,
)

logger = logging.getLogger(__name__)

_QUERY_URL = "https://export.arxiv.org/api/query"
_REQUEST_TIMEOUT_SECONDS = 60
_HTTP_RETRIES = 2
_RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_USER_AGENT = "proto-tools/arxiv-search-v1"
# arXiv's API terms ask for no more than one request every three seconds.
_REQUEST_INTERVAL_SECONDS = 3.0
_ERROR_ID_PREFIX = "http://arxiv.org/api/errors"
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"
_OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"
_ABS_PREFIX = "arxiv.org/abs/"

ArxivSortBy = Literal["relevance", "submittedDate", "lastUpdatedDate"]


class _Gate:
    """Process-wide gate every arXiv request passes through.

    Attributes:
        lock (threading.Lock): Held for the whole request, so there is one connection at a time.
        next_request_at (float): Monotonic time before which the next request may not start.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.next_request_at = float("-inf")


_gate = _Gate()


# ============================================================================
# Data Models
# ============================================================================


class ArxivSearchInput(BaseToolInput):
    """Queries to run against arXiv.

    Attributes:
        queries (list[str]): arXiv ``search_query`` strings, e.g. ``'ti:"protein binder" AND
            cat:q-bio.BM'`` or ``"all:diffusion protein design"``. A bare string is normalized to a
            one-item list.
    """

    queries: list[str] = InputField(
        min_length=1,
        title="Queries",
        description="arXiv search_query strings, e.g. 'ti:\"protein binder\" AND cat:q-bio.BM'",
    )

    @field_validator("queries", mode="before")
    @classmethod
    def _normalize_queries(cls, value: Any) -> Any:
        """Accept a bare query string and strip whitespace; reject empty queries."""
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list):
            cleaned = [item.strip() if isinstance(item, str) else item for item in value]
            if any(item == "" for item in cleaned):
                raise ValueError("queries must not be empty strings")
            return cleaned
        return value


class ArxivSearchConfig(BaseConfig):
    """Result size and ordering for arXiv search.

    Attributes:
        max_results (int): Maximum hits returned per query (1-500).
        sort_by (ArxivSortBy): ``relevance``, or newest first by ``submittedDate`` or
            ``lastUpdatedDate``.
    """

    max_results: int = ConfigField(
        default=10, ge=1, le=500, title="Max Results", description="Maximum hits returned per query"
    )
    sort_by: ArxivSortBy = ConfigField(
        default="relevance",
        title="Sort By",
        description="relevance, or newest first by submittedDate or lastUpdatedDate",
    )


class ArxivSearchHit(BaseModel):
    """One arXiv preprint returned by a search.

    Attributes:
        arxiv_id (str): arXiv ID with version, e.g. ``2504.00146v2``.
        title (str): Title.
        authors (list[str]): Author names.
        summary (str): Abstract.
        published (str): Submission time of version 1 (ISO 8601).
        updated (str): Submission time of this version (ISO 8601).
        primary_category (str | None): Primary arXiv category, e.g. ``q-bio.BM``.
        categories (list[str]): All arXiv categories.
        doi (str | None): DOI of the published version, when the authors supplied one.
        journal_ref (str | None): Journal reference, when the authors supplied one.
        comment (str | None): Author comment (pages, figures, venue).
        abs_url (str): arXiv abstract page.
        pdf_url (str): arXiv PDF.
    """

    model_config = ConfigDict(extra="forbid")

    arxiv_id: str = Field(title="arXiv ID", description="arXiv ID with version, e.g. '2504.00146v2'")
    title: str = Field(title="Title", description="Title")
    authors: list[str] = Field(default_factory=list, title="Authors", description="Author names")
    summary: str = Field(title="Summary", description="Abstract")
    published: str = Field(title="Published", description="Submission time of version 1 (ISO 8601)")
    updated: str = Field(title="Updated", description="Submission time of this version (ISO 8601)")
    primary_category: str | None = Field(default=None, title="Primary Category", description="Primary category")
    categories: list[str] = Field(default_factory=list, title="Categories", description="All arXiv categories")
    doi: str | None = Field(default=None, title="DOI", description="DOI of the published version, if given")
    journal_ref: str | None = Field(default=None, title="Journal Ref", description="Journal reference, if given")
    comment: str | None = Field(default=None, title="Comment", description="Author comment (pages, venue)")
    abs_url: str = Field(title="Abstract URL", description="arXiv abstract page")
    pdf_url: str = Field(title="PDF URL", description="arXiv PDF")


class ArxivSearchResult(BaseModel):
    """Hits for one query.

    Attributes:
        query (str): The query as sent to arXiv.
        total_count (int): Total matching preprints reported by arXiv.
        hits (list[ArxivSearchHit]): Hits in the requested order.
        error (str | None): Why arXiv rejected this query; None when it was retrieved.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(title="Query", description="Query as sent")
    total_count: int = Field(title="Total Count", description="Total matching preprints")
    hits: list[ArxivSearchHit] = Field(default_factory=list, title="Hits", description="Hits in the requested order")
    error: str | None = Field(
        default=None, title="Error", description="Why this item has no data; None when it was retrieved"
    )


class ArxivSearchOutput(BaseToolOutput):
    """Search results, one per input query.

    Attributes:
        results (list[ArxivSearchResult]): Results in input order.
        source_url (str): arXiv API endpoint used.
    """

    results: list[ArxivSearchResult] = Field(
        default_factory=list, title="Results", description="Results in input order"
    )
    source_url: str = Field(title="Source URL", description="arXiv API endpoint used")

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json", "csv"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: Any, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")
        if file_format == "json":
            with path.open("w", encoding="utf-8") as f:
                json.dump(self.model_dump(mode="json"), f, indent=2)
            return
        if file_format == "csv":
            fieldnames = ["query_index", "query", "rank", *ArxivSearchHit.model_fields]
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for index, result in enumerate(self.results):
                    for rank, hit in enumerate(result.hits, start=1):
                        row = hit.model_dump()
                        row["authors"] = ";".join(row["authors"])
                        row["categories"] = ";".join(row["categories"])
                        writer.writerow({"query_index": index, "query": result.query, "rank": rank, **row})
            return
        raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return ArxivSearchInput(queries=['ti:"protein binder"'])


@tool(
    key="arxiv-search",
    local_only="arxiv-search does not use a gpu and does not need an environment, so it can run in process",
    label="arXiv Search",
    category="database_retrieval",
    input_class=ArxivSearchInput,
    config_class=ArxivSearchConfig,
    output_class=ArxivSearchOutput,
    description=(
        "Search arXiv preprints (q-bio, cs.LG and other categories) by title, author, abstract or "
        "category for IDs, abstracts, DOIs and PDF links"
    ),
    uses_gpu=False,
    example_input=example_input,
    iterable_input_fields=["queries"],
    iterable_output_field="results",
    cacheable=True,
)
def run_arxiv_search(
    inputs: ArxivSearchInput,
    config: ArxivSearchConfig,
    instance: Any = None,
) -> ArxivSearchOutput:
    """Search arXiv for each query, keeping to arXiv's limit of one request every three seconds.

    The limit holds across every call in the process: concurrent calls queue on one gate, and
    retries wait their turn too. A query arXiv rejects (an HTTP 4xx other than 429, or an error
    entry in the feed) yields a result with ``error`` set, and the other queries still run.

    Args:
        inputs (ArxivSearchInput): arXiv ``search_query`` strings.
        config (ArxivSearchConfig): Result size and sort order.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        ArxivSearchOutput: One result per query, in input order.

    Raises:
        requests.HTTPError: If arXiv still answers 429 or 5xx after retries.
        requests.RequestException: On a timeout, or a connection error after retries.
    """
    del instance

    # Retries go through the gate in _get_with_retry, so the adapter returns every response as is.
    session = build_http_session(http_retries=0, backoff_seconds=0.0, user_agent=_USER_AGENT, status_forcelist=[])
    try:
        results = [_search(query, config, session) for query in inputs.queries]
    finally:
        session.close()
    return ArxivSearchOutput(results=results, source_url=_QUERY_URL)


# ============================================================================
# Private Helpers
# ============================================================================


def _search(query: str, config: ArxivSearchConfig, session: requests.Session) -> ArxivSearchResult:
    """Run one query and parse the Atom feed."""
    params: dict[str, str | int] = {"search_query": query, "start": 0, "max_results": config.max_results}
    if config.sort_by != "relevance":
        params.update({"sortBy": config.sort_by, "sortOrder": "descending"})
    response = _get_with_retry(session, params)
    if 400 <= response.status_code < 500 and response.status_code != 429:
        try:
            raise_for_status_with_body(response, "arXiv search")
        except requests.HTTPError as error:
            return ArxivSearchResult(query=query, total_count=0, error=str(error))
    raise_for_status_with_body(response, "arXiv search")
    feed = ET.fromstring(response.content)  # noqa: S314 -- expat refuses external entities and caps entity expansion
    entries = feed.findall(f"{_ATOM}entry")
    # arXiv reports some query errors as a feed whose single entry has an id under its errors URL.
    if len(entries) == 1 and _clean(entries[0].findtext(f"{_ATOM}id")).startswith(_ERROR_ID_PREFIX):
        message = _clean(entries[0].findtext(f"{_ATOM}summary"))
        return ArxivSearchResult(query=query, total_count=0, error=f"arXiv search failed: {message}")
    total = feed.findtext(f"{_OPENSEARCH}totalResults")
    logger.debug("arXiv returned %d of %s hits for %r", len(entries), total, query)
    return ArxivSearchResult(
        query=query,
        total_count=int(total) if total else len(entries),
        hits=[_parse_entry(entry) for entry in entries],
    )


def _get_with_retry(session: requests.Session, params: dict[str, str | int]) -> requests.Response:
    """Send a query, retrying connection errors, 429 and 5xx through the gate."""
    for attempt in range(1, _HTTP_RETRIES + 1):
        try:
            response = _gated_get(session, params)
        except requests.ConnectionError as error:
            logger.warning("arXiv connection failed (attempt %d), retrying: %s", attempt, error)
            continue
        if response.status_code not in _RETRY_STATUS_CODES:
            return response
        logger.warning("arXiv returned HTTP %d (attempt %d), retrying", response.status_code, attempt)
    return _gated_get(session, params)


def _gated_get(session: requests.Session, params: dict[str, str | int]) -> requests.Response:
    """Send one request through the process-wide gate.

    The request starts no sooner than the interval (or a longer ``Retry-After``) after the previous
    one ended, and holds the gate until it ends, so only one connection is open at a time.
    """
    left = remaining()
    # Queued behind other arXiv calls, a call still gives up when its own timeout elapses.
    if not _gate.lock.acquire(timeout=-1 if left is None else max(left, 0.0)):
        raise DeadlineExceededError("Timed out waiting for an earlier arXiv request: the tool's timeout elapsed")
    try:
        delay = _gate.next_request_at - monotonic()
        if delay > 0:
            sleep_within_deadline(delay, "spacing arXiv requests")
        wait = _REQUEST_INTERVAL_SECONDS
        try:
            response = session.get(_QUERY_URL, params=params, timeout=_REQUEST_TIMEOUT_SECONDS)
            wait = max(wait, _retry_after_seconds(response))
            return response
        finally:
            _gate.next_request_at = monotonic() + wait
    finally:
        _gate.lock.release()


def _retry_after_seconds(response: requests.Response) -> float:
    """Return the delay in seconds a 429 or 503 asks for in ``Retry-After``, or 0."""
    if response.status_code not in (429, 503):
        return 0.0
    try:
        return max(float(response.headers.get("Retry-After", 0)), 0.0)
    except ValueError:
        return 0.0


def _parse_entry(entry: ET.Element) -> ArxivSearchHit:
    """Convert one Atom ``<entry>`` into a hit."""
    entry_id = _clean(entry.findtext(f"{_ATOM}id"))
    arxiv_id = entry_id.split(_ABS_PREFIX, 1)[-1]
    links = {link.get("title") or link.get("rel"): link.get("href") for link in entry.findall(f"{_ATOM}link")}
    primary = entry.find(f"{_ARXIV}primary_category")
    return ArxivSearchHit(
        arxiv_id=arxiv_id,
        title=_clean(entry.findtext(f"{_ATOM}title")),
        authors=[_clean(author.findtext(f"{_ATOM}name")) for author in entry.findall(f"{_ATOM}author")],
        summary=_clean(entry.findtext(f"{_ATOM}summary")),
        published=_clean(entry.findtext(f"{_ATOM}published")),
        updated=_clean(entry.findtext(f"{_ATOM}updated")),
        primary_category=primary.get("term") if primary is not None else None,
        categories=[str(cat.get("term")) for cat in entry.findall(f"{_ATOM}category") if cat.get("term")],
        doi=_clean(entry.findtext(f"{_ARXIV}doi")) or None,
        journal_ref=_clean(entry.findtext(f"{_ARXIV}journal_ref")) or None,
        comment=_clean(entry.findtext(f"{_ARXIV}comment")) or None,
        abs_url=links.get("alternate") or f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=links.get("pdf") or f"https://arxiv.org/pdf/{arxiv_id}",
    )


def _clean(text: str | None) -> str:
    """Collapse the line wrapping arXiv puts in titles and abstracts."""
    return " ".join((text or "").split())
