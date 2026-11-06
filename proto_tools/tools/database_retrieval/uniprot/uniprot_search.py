"""proto_tools/tools/database_retrieval/uniprot/uniprot_search.py.

Ranked UniProtKB search: free-text or UniProt query-syntax queries return a list of candidate
entries (accession, names, organism, length, review status), in UniProt's relevance order.
Complements ``uniprot-fetch``, which resolves one accession or one gene-name-plus-organism pair.
"""

import csv
import json
import logging
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
    request_with_retry,
)

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
_REQUEST_TIMEOUT_SECONDS = 30
_HTTP_RETRIES = 2
_BACKOFF_SECONDS = 1.0
_USER_AGENT = "proto-tools/uniprot-search-v1"
_RETURN_FIELDS = "accession,id,protein_name,gene_names,organism_name,organism_id,length,reviewed,annotation_score"


# ============================================================================
# Data Models
# ============================================================================


class UniProtSearchInput(BaseToolInput):
    """Queries to run against UniProtKB.

    Attributes:
        queries (list[str]): Free-text queries (e.g. ``"DNA polymerase Thermus aquaticus"``) or
            UniProt query syntax (e.g. ``'protein_name:"programmed cell death protein 1"'``). A
            bare string is normalized to a one-item list.
    """

    queries: list[str] = InputField(
        min_length=1,
        title="Queries",
        description="Free-text or UniProt query-syntax searches; a bare string is accepted",
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


class UniProtSearchConfig(BaseConfig):
    """Filters and result size for UniProt search.

    Attributes:
        organism (str | None): Restrict to an organism by name (``organism_name:"..."``), e.g.
            ``"Homo sapiens"``. Combined with the query using AND.
        taxon_id (int | None): Restrict to an NCBI taxonomy ID (``taxonomy_id:...``), including
            all descendant taxa. Combined with the query using AND.
        reviewed_only (bool): Restrict to reviewed Swiss-Prot entries. Defaults to True, which
            keeps results to curated entries and excludes large unreviewed TrEMBL sets.
        max_results (int): Maximum hits returned per query (1-500).
        include_sequence (bool): Also return each hit's sequence.
    """

    organism: str | None = ConfigField(
        default=None, title="Organism", description="Restrict to an organism name, e.g. 'Homo sapiens'"
    )
    taxon_id: int | None = ConfigField(
        default=None, ge=1, title="Taxon ID", description="Restrict to an NCBI taxonomy ID and its descendants"
    )
    reviewed_only: bool = ConfigField(
        default=True, title="Reviewed Only", description="Restrict to reviewed Swiss-Prot entries"
    )
    max_results: int = ConfigField(
        default=10, ge=1, le=500, title="Max Results", description="Maximum hits returned per query"
    )
    include_sequence: bool = ConfigField(
        default=False, title="Include Sequence", description="Also return each hit's sequence"
    )


class UniProtSearchHit(BaseModel):
    """One UniProtKB entry returned by a search.

    Attributes:
        accession (str): Primary UniProt accession.
        entry_name (str | None): UniProtKB entry name (e.g. ``DPO1_THEAQ``).
        protein_name (str | None): Recommended (or submitted) full protein name.
        gene_names (list[str]): Primary gene names.
        organism (str | None): Source organism scientific name.
        taxon_id (int | None): NCBI taxonomy ID of the source organism.
        length (int | None): Sequence length.
        reviewed (bool): True for Swiss-Prot entries.
        annotation_score (float | None): UniProt annotation score (1-5).
        sequence (str | None): Sequence, when ``include_sequence`` is set.
    """

    model_config = ConfigDict(extra="forbid")

    accession: str = Field(title="Accession", description="Primary UniProt accession")
    entry_name: str | None = Field(default=None, title="Entry Name", description="UniProtKB entry name")
    protein_name: str | None = Field(default=None, title="Protein Name", description="Recommended full name")
    gene_names: list[str] = Field(default_factory=list, title="Gene Names", description="Primary gene names")
    organism: str | None = Field(default=None, title="Organism", description="Source organism scientific name")
    taxon_id: int | None = Field(default=None, title="Taxon ID", description="NCBI taxonomy ID")
    length: int | None = Field(default=None, title="Length", description="Sequence length")
    reviewed: bool = Field(default=False, title="Reviewed", description="True for Swiss-Prot entries")
    annotation_score: float | None = Field(
        default=None, title="Annotation Score", description="UniProt annotation score (1-5)"
    )
    sequence: str | None = Field(default=None, title="Sequence", description="Sequence, if requested")


class UniProtSearchResult(BaseModel):
    """Hits for one query.

    Attributes:
        query (str): The query as sent to UniProt, including organism and review filters.
        total_count (int): Total matching entries reported by UniProt.
        hits (list[UniProtSearchHit]): Hits in UniProt's relevance order.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(title="Query", description="Query as sent, including filters")
    total_count: int = Field(title="Total Count", description="Total matching entries")
    hits: list[UniProtSearchHit] = Field(default_factory=list, title="Hits", description="Hits in relevance order")


class UniProtSearchOutput(BaseToolOutput):
    """Search results, one per input query.

    Attributes:
        results (list[UniProtSearchResult]): Results in input order.
        source_url (str): UniProt search endpoint used.
    """

    results: list[UniProtSearchResult] = Field(
        default_factory=list, title="Results", description="Results in input order"
    )
    source_url: str = Field(title="Source URL", description="UniProt search endpoint used")

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json", "csv", "fasta"]

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
            fieldnames = ["query_index", "query", "rank", *UniProtSearchHit.model_fields]
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for index, result in enumerate(self.results):
                    for rank, hit in enumerate(result.hits, start=1):
                        row = hit.model_dump()
                        row["gene_names"] = ";".join(row["gene_names"])
                        writer.writerow({"query_index": index, "query": result.query, "rank": rank, **row})
            return
        if file_format == "fasta":
            with path.open("w", encoding="utf-8") as f:
                for index, result in enumerate(self.results):
                    for hit in result.hits:
                        if hit.sequence:
                            f.write(f">{hit.accession} query={index} {hit.protein_name or ''}\n{hit.sequence}\n")
            return
        raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return UniProtSearchInput(queries=["DNA polymerase Thermus aquaticus"])


@tool(
    key="uniprot-search",
    local_only="uniprot-search does not use a gpu and does not need an environment, so it can run in process",
    label="UniProt Search",
    category="database_retrieval",
    input_class=UniProtSearchInput,
    config_class=UniProtSearchConfig,
    output_class=UniProtSearchOutput,
    description="Search UniProtKB with free text or query syntax and return ranked candidate entries",
    uses_gpu=False,
    example_input=example_input,
    iterable_input_fields=["queries"],
    iterable_output_field="results",
    cacheable=True,
)
def run_uniprot_search(
    inputs: UniProtSearchInput,
    config: UniProtSearchConfig,
    instance: Any = None,
) -> UniProtSearchOutput:
    """Search UniProtKB and return ranked candidate entries for each query.

    Args:
        inputs (UniProtSearchInput): Free-text or query-syntax searches.
        config (UniProtSearchConfig): Organism, taxon and review filters, result size and
            whether to include sequences.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        UniProtSearchOutput: One result per query, hits in UniProt's relevance order.
    """
    del instance

    session = build_http_session(
        http_retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
        user_agent=_USER_AGENT,
    )
    try:
        return UniProtSearchOutput(
            results=[_search(query, config, session) for query in inputs.queries], source_url=_SEARCH_URL
        )
    finally:
        session.close()


# ============================================================================
# Private Helpers
# ============================================================================


def _full_query(query: str, config: UniProtSearchConfig) -> str:
    """Combine the user query with the configured filters."""
    clauses = [f"({query})"]
    if config.organism:
        clauses.append(f'(organism_name:"{config.organism}")')
    if config.taxon_id:
        clauses.append(f"(taxonomy_id:{config.taxon_id})")
    if config.reviewed_only:
        clauses.append("(reviewed:true)")
    return " AND ".join(clauses) if len(clauses) > 1 else query


def _raise_for_status(response: Any) -> None:
    """Raise an HTTP error that includes a bounded UniProt response body (e.g. query syntax errors)."""
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        body = str(getattr(response, "text", ""))[:1000]
        raise requests.HTTPError(f"UniProt search failed: {error}; body={body!r}", response=response) from error


def _search(query: str, config: UniProtSearchConfig, session: requests.Session) -> UniProtSearchResult:
    """Run one search request and parse its hits."""
    full_query = _full_query(query, config)
    fields = f"{_RETURN_FIELDS},sequence" if config.include_sequence else _RETURN_FIELDS
    params: dict[str, str | int] = {"query": full_query, "fields": fields, "format": "json", "size": config.max_results}
    response = request_with_retry(
        lambda: session.get(_SEARCH_URL, params=params, timeout=_REQUEST_TIMEOUT_SECONDS),
        retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
    )
    _raise_for_status(response)
    entries = response.json().get("results", [])
    total = response.headers.get("X-Total-Results")
    return UniProtSearchResult(
        query=full_query,
        total_count=int(total) if total is not None else len(entries),
        hits=[_parse_hit(entry, config.include_sequence) for entry in entries],
    )


def _parse_hit(entry: dict[str, Any], include_sequence: bool) -> UniProtSearchHit:
    """Convert one UniProt JSON result into a hit."""
    organism = entry.get("organism") or {}
    sequence = entry.get("sequence") or {}
    return UniProtSearchHit(
        accession=entry["primaryAccession"],
        entry_name=entry.get("uniProtkbId"),
        protein_name=_protein_name(entry.get("proteinDescription") or {}),
        gene_names=[
            gene["geneName"]["value"] for gene in entry.get("genes", []) if gene.get("geneName", {}).get("value")
        ],
        organism=organism.get("scientificName"),
        taxon_id=organism.get("taxonId"),
        length=sequence.get("length"),
        reviewed="swiss-prot" in str(entry.get("entryType") or "").lower(),
        annotation_score=entry.get("annotationScore"),
        sequence=sequence.get("value") if include_sequence else None,
    )


def _protein_name(description: dict[str, Any]) -> str | None:
    """Return the recommended full name, falling back to the first submitted name."""
    recommended = description.get("recommendedName") or {}
    if recommended.get("fullName", {}).get("value"):
        return str(recommended["fullName"]["value"])
    for submitted in description.get("submissionNames") or []:
        if submitted.get("fullName", {}).get("value"):
            return str(submitted["fullName"]["value"])
    return None
