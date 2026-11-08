"""proto_tools/tools/database_retrieval/europepmc/europepmc_search.py.

Ranked Europe PMC search: free-text or Europe PMC query-syntax queries return candidate
publications (PubMed records, PMC full-text articles, and preprints) with identifiers,
citation details, open-access status and, optionally, abstracts.
"""

import csv
import html
import json
import logging
import re
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, ConfigDict, Field, field_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    build_http_session,
    raise_for_status_with_body,
    request_with_retry,
)

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_ARTICLE_URL = "https://europepmc.org/article/{source}/{id}"
_REQUEST_TIMEOUT_SECONDS = 30
_HTTP_RETRIES = 2
_BACKOFF_SECONDS = 1.0
_USER_AGENT = "proto-tools/europepmc-search-v1"
_BLOCK_TAG_RE = re.compile(r"</?(?:h\d|p|br|div)\b[^>]*>", re.IGNORECASE)
_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*\b[^<>]*>")


# ============================================================================
# Data Models
# ============================================================================


class EuropePMCSearchInput(BaseToolInput):
    """Queries to run against Europe PMC.

    Attributes:
        queries (list[str]): Free-text queries (e.g. ``"PD-L1 binder alanine scan"``) or Europe PMC
            query syntax (e.g. ``'TITLE:"protein binder" AND SRC:PPR'``). A bare string is
            normalized to a one-item list.
    """

    queries: list[str] = InputField(
        min_length=1,
        title="Queries",
        description="Free-text or Europe PMC query-syntax searches; a bare string is accepted",
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


class EuropePMCSearchConfig(BaseConfig):
    """Result size and detail for Europe PMC search.

    Attributes:
        max_results (int): Maximum hits returned per query (1-1000).
        include_abstracts (bool): Return abstracts and full citation details (Europe PMC ``core``
            results). Set False for a compact hit list (``lite`` results).
    """

    max_results: int = ConfigField(
        default=10, ge=1, le=1000, title="Max Results", description="Maximum hits returned per query"
    )
    include_abstracts: bool = ConfigField(
        default=True, title="Include Abstracts", description="Return abstracts (core results); False for lite results"
    )


class EuropePMCSearchHit(BaseModel):
    """One Europe PMC publication returned by a search.

    Attributes:
        id (str): Europe PMC record ID (a PMID for ``MED``, ``PPR...`` for preprints).
        source (str): Europe PMC source code: ``MED`` (PubMed), ``PMC``, ``PPR`` (preprint), etc.
        pmid (str | None): PubMed identifier.
        pmcid (str | None): PubMed Central identifier; pass to ``europepmc-fulltext``.
        doi (str | None): Digital object identifier.
        title (str): Publication title.
        authors (list[str]): Author names as Europe PMC gives them (``"Jin G"``), and group authors
            by their collective name.
        journal (str | None): Journal title, or the preprint server for preprints.
        year (int | None): Publication year.
        publication_date (str | None): First publication date (``core`` results only).
        is_open_access (bool): Whether the article is open access.
        has_full_text (bool): Whether Europe PMC holds the full text (``inEPMC``).
        cited_by_count (int | None): Number of citing records in Europe PMC.
        abstract (str | None): Abstract with markup removed (``core`` results only).
        published_pmid (str | None): For a preprint, the PMID of its published article (``core`` results only).
        source_url (str): Europe PMC page for the record.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(title="ID", description="Europe PMC record ID")
    source: str = Field(title="Source", description="Europe PMC source code (MED, PMC, PPR, ...)")
    pmid: str | None = Field(default=None, title="PMID", description="PubMed identifier")
    pmcid: str | None = Field(default=None, title="PMCID", description="PubMed Central identifier")
    doi: str | None = Field(default=None, title="DOI", description="Digital object identifier")
    title: str = Field(title="Title", description="Publication title")
    authors: list[str] = Field(
        default_factory=list, title="Authors", description="Author names as Europe PMC gives them"
    )
    journal: str | None = Field(default=None, title="Journal", description="Journal or preprint server")
    year: int | None = Field(default=None, title="Year", description="Publication year")
    publication_date: str | None = Field(
        default=None, title="Publication Date", description="First publication date (core results)"
    )
    is_open_access: bool = Field(default=False, title="Is Open Access", description="Whether the article is OA")
    has_full_text: bool = Field(
        default=False, title="Has Full Text", description="Whether Europe PMC holds the full text"
    )
    cited_by_count: int | None = Field(default=None, title="Cited By Count", description="Citing records")
    abstract: str | None = Field(default=None, title="Abstract", description="Abstract (core results)")
    published_pmid: str | None = Field(
        default=None, title="Published PMID", description="For a preprint, the PMID of its published article"
    )
    source_url: str = Field(title="Source URL", description="Europe PMC page for the record")


class EuropePMCSearchResult(BaseModel):
    """Hits for one query.

    Attributes:
        query (str): The query as sent to Europe PMC.
        total_count (int | None): Total matching records reported by Europe PMC; None on error.
        hits (list[EuropePMCSearchHit]): Hits in Europe PMC's relevance order.
        error (str | None): Why this query has no hits, such as a query syntax error; None when it
            was run.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(title="Query", description="Query as sent")
    total_count: int | None = Field(default=None, title="Total Count", description="Total matching records")
    hits: list[EuropePMCSearchHit] = Field(default_factory=list, title="Hits", description="Hits in relevance order")
    error: str | None = Field(
        default=None, title="Error", description="Why this item has no data; None when it was retrieved"
    )


class EuropePMCSearchOutput(BaseToolOutput):
    """Search results, one per input query.

    Attributes:
        results (list[EuropePMCSearchResult]): Results in input order.
        source_url (str): Europe PMC search endpoint used.
    """

    results: list[EuropePMCSearchResult] = Field(
        default_factory=list, title="Results", description="Results in input order"
    )
    source_url: str = Field(title="Source URL", description="Europe PMC search endpoint used")

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
            fieldnames = ["query_index", "query", "error", "rank", *EuropePMCSearchHit.model_fields]
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for index, result in enumerate(self.results):
                    if result.error:
                        writer.writerow({"query_index": index, "query": result.query, "error": result.error})
                    for rank, hit in enumerate(result.hits, start=1):
                        row = {"query_index": index, "query": result.query, "rank": rank, **hit.model_dump()}
                        row["authors"] = ";".join(hit.authors)
                        writer.writerow(row)
            return
        raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return EuropePMCSearchInput(queries=['"de novo" protein binder design OPEN_ACCESS:y'])


@tool(
    key="europepmc-search",
    local_only="europepmc-search does not use a gpu and does not need an environment, so it can run in process",
    label="Europe PMC Search",
    category="database_retrieval",
    input_class=EuropePMCSearchInput,
    config_class=EuropePMCSearchConfig,
    output_class=EuropePMCSearchOutput,
    description=(
        "Search Europe PMC literature (PubMed, PMC full text, bioRxiv/medRxiv preprints) for papers "
        "with PMIDs, PMCIDs, DOIs, abstracts and open-access status"
    ),
    uses_gpu=False,
    example_input=example_input,
    iterable_input_fields=["queries"],
    iterable_output_field="results",
    cacheable=True,
)
def run_europepmc_search(
    inputs: EuropePMCSearchInput,
    config: EuropePMCSearchConfig,
    instance: Any = None,
) -> EuropePMCSearchOutput:
    """Search Europe PMC and return ranked publications for each query.

    Args:
        inputs (EuropePMCSearchInput): Free-text or query-syntax searches.
        config (EuropePMCSearchConfig): Result size and whether to include abstracts.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        EuropePMCSearchOutput: One result per query, hits in Europe PMC's relevance order. A query
            Europe PMC rejects (HTTP 4xx other than 429) gets an ``error`` and no hits.

    Raises:
        requests.HTTPError: If Europe PMC fails with a server error or keeps rate limiting.
    """
    del instance

    session = build_http_session(
        http_retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
        user_agent=_USER_AGENT,
    )
    try:
        return EuropePMCSearchOutput(
            results=[_search(query, config, session) for query in inputs.queries], source_url=_SEARCH_URL
        )
    finally:
        session.close()


# ============================================================================
# Private Helpers
# ============================================================================


def _search(query: str, config: EuropePMCSearchConfig, session: requests.Session) -> EuropePMCSearchResult:
    """Run one search request and parse its hits."""
    params: dict[str, str | int] = {
        "query": query,
        "format": "json",
        "resultType": "core" if config.include_abstracts else "lite",
        "pageSize": config.max_results,
        "cursorMark": "*",
    }
    response = request_with_retry(
        lambda: session.get(_SEARCH_URL, params=params, timeout=_REQUEST_TIMEOUT_SECONDS),
        retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
    )
    error = _item_error(response, "Europe PMC search")
    if error:
        return EuropePMCSearchResult(query=query, error=error)
    payload = response.json()
    records = (payload.get("resultList") or {}).get("result", [])
    logger.debug("Europe PMC returned %d of %s hits for %r", len(records), payload.get("hitCount"), query)
    return EuropePMCSearchResult(
        query=query,
        total_count=int(payload.get("hitCount") or 0),
        hits=[_parse_hit(record) for record in records],
    )


def _parse_hit(record: dict[str, Any]) -> EuropePMCSearchHit:
    """Convert one Europe PMC JSON result (core or lite) into a hit."""
    journal_info = record.get("journalInfo") or {}
    journal = (
        (journal_info.get("journal") or {}).get("title")
        or record.get("journalTitle")
        or (record.get("bookOrReportDetails") or {}).get("publisher")
    )
    abstract = record.get("abstractText")
    year = record.get("pubYear")
    corrections = (record.get("commentCorrectionList") or {}).get("commentCorrection") or []
    published = next(
        (c.get("id") for c in corrections if c.get("type") == "Preprint of" and c.get("source") == "MED"), None
    )
    return EuropePMCSearchHit(
        id=str(record["id"]),
        source=str(record["source"]),
        pmid=record.get("pmid"),
        pmcid=record.get("pmcid"),
        doi=record.get("doi"),
        title=_strip_markup(str(record.get("title") or "")),
        authors=_authors(record),
        journal=journal,
        year=int(year) if year and str(year).isdigit() else None,
        publication_date=record.get("firstPublicationDate"),
        is_open_access=record.get("isOpenAccess") == "Y",
        has_full_text=record.get("inEPMC") == "Y",
        cited_by_count=record.get("citedByCount"),
        abstract=_strip_markup(abstract) if abstract else None,
        published_pmid=published,
        source_url=_ARTICLE_URL.format(source=record["source"], id=record["id"]),
    )


def _authors(record: dict[str, Any]) -> list[str]:
    """Return author names from ``core`` results' ``authorList``, else split ``lite`` results' ``authorString``."""
    listed = (record.get("authorList") or {}).get("author") or []
    if listed:
        names = (author.get("fullName") or author.get("collectiveName") for author in listed)
        return [name for name in names if name]
    author_string = str(record.get("authorString") or "").strip().removesuffix(".")
    return [name.strip() for name in author_string.split(", ") if name.strip()]


def _item_error(response: requests.Response, operation: str) -> str | None:
    """Return why a rejected item (HTTP 4xx other than 429) has no data; raise for any other failure.

    Args:
        response (requests.Response): Response to check.
        operation (str): What the request was doing, such as ``"Europe PMC search"``.

    Returns:
        str | None: The error, quoting the start of the response body, or None on success.

    Raises:
        requests.HTTPError: If the response is a server error or a rate limit.
    """
    try:
        raise_for_status_with_body(response, operation)
    except requests.HTTPError as error:
        if response.status_code >= 500 or response.status_code == 429:
            raise
        return str(error)
    return None


def _strip_markup(text: str) -> str:
    """Drop inline HTML (``<sub>``, ``<i>``) in place and turn block tags (``<h4>``, ``<p>``) into spaces."""
    return " ".join(html.unescape(_TAG_RE.sub("", _BLOCK_TAG_RE.sub(" ", text))).split())
