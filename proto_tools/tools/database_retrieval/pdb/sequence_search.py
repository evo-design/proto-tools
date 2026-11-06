"""Search experimental PDB polymer entities by sequence with RCSB's MMseqs2 service.

The RCSB Search API's sequence service aligns a query against every polymer entity in
the archive and returns per-hit alignment statistics in under a second. Hits are
optionally annotated with one batched RCSB Data API GraphQL request, which adds the
entity description, source organism, UniProt accessions, author chain IDs, and
canonical sequence.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal

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

_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
_GRAPHQL_URL = "https://data.rcsb.org/graphql"
_USER_AGENT = "proto-tools/pdb-sequence-search-v1"
_REQUEST_TIMEOUT_SECONDS = 120
_HTTP_RETRIES = 3
_BACKOFF_SECONDS = 1.0
_MIN_QUERY_LENGTH = 10  # RCSB rejects shorter queries with HTTP 400
_NO_CONTENT = 204  # RCSB's response when nothing matches

PdbSequenceType = Literal["protein", "dna", "rna"]

_ANNOTATION_QUERY = """
query ($ids: [String!]!) {
  polymer_entities(entity_ids: $ids) {
    rcsb_id
    rcsb_polymer_entity { pdbx_description }
    rcsb_entity_source_organism { ncbi_scientific_name }
    rcsb_polymer_entity_container_identifiers { uniprot_ids auth_asym_ids }
    entity_poly { pdbx_seq_one_letter_code_can }
  }
}
"""

_ALIGNMENT_FIELDS = (
    "sequence_identity",
    "evalue",
    "bitscore",
    "alignment_length",
    "mismatches",
    "gaps_opened",
    "query_beg",
    "query_end",
    "subject_beg",
    "subject_end",
    "query_aligned_seq",
    "subject_aligned_seq",
)


class PdbSequenceSearchQuery(BaseModel):
    """One query sequence.

    Attributes:
        sequence (str): Query sequence in one-letter code, at least 10 residues; whitespace is removed.
        sequence_type (PdbSequenceType): Polymer type to search: ``protein``, ``dna``, or ``rna``.
    """

    model_config = ConfigDict(extra="forbid")

    sequence: str = Field(
        title="Sequence",
        description="Query sequence in one-letter code, at least 10 residues; whitespace is removed",
    )
    sequence_type: PdbSequenceType = Field(
        default="protein",
        title="Sequence Type",
        description="Polymer type to search: protein, dna, or rna",
    )

    @field_validator("sequence")
    @classmethod
    def _clean_sequence(cls, value: str) -> str:
        """Uppercase, strip whitespace, and enforce RCSB's minimum length."""
        cleaned = "".join(value.split()).upper()
        if len(cleaned) < _MIN_QUERY_LENGTH:
            raise ValueError(f"RCSB sequence search needs at least {_MIN_QUERY_LENGTH} residues, got {len(cleaned)}")
        return cleaned


class PdbSequenceSearchInput(BaseToolInput):
    """Sequences to search against experimental PDB polymer entities.

    Attributes:
        queries (list[PdbSequenceSearchQuery]): Query sequences. A single query, or a bare
            sequence string, is normalized to a list.
    """

    queries: list[PdbSequenceSearchQuery] = InputField(
        min_length=1,
        title="Sequence Queries",
        description="Query sequences to search; a single query or a bare sequence string is accepted",
    )

    @field_validator("queries", mode="before")
    @classmethod
    def _normalize_queries(cls, value: Any) -> Any:
        """Accept a single query, a bare sequence string, or a list of either."""
        if isinstance(value, str | dict | PdbSequenceSearchQuery):
            value = [value]
        if isinstance(value, list):
            return [{"sequence": item} if isinstance(item, str) else item for item in value]
        return value


class PdbSequenceSearchConfig(BaseConfig):
    """RCSB sequence-search settings.

    Attributes:
        identity_cutoff (float): Minimum sequence identity over the aligned region.
        evalue_cutoff (float): Maximum E-value of a reported hit.
        max_hits (int): Maximum hits returned per query (at most 10,000).
        annotate (bool): Add description, organism, UniProt IDs, author chain IDs, and sequence to each hit.
        include_computed_models (bool): Also search RCSB's computed structure models.
    """

    identity_cutoff: float = ConfigField(
        default=0.0,
        ge=0.0,
        le=1.0,
        title="Identity Cutoff",
        description="Minimum sequence identity over the aligned region, from 0 to 1",
    )
    evalue_cutoff: float = ConfigField(
        default=0.1,
        gt=0.0,
        title="E-value Cutoff",
        description="Maximum E-value of a reported hit",
    )
    max_hits: int = ConfigField(
        default=25,
        ge=1,
        le=10000,
        title="Maximum Hits",
        description="Maximum hits returned per query; RCSB caps one request at 10,000",
    )
    annotate: bool = ConfigField(
        default=True,
        title="Annotate Hits",
        description="Add description, organism, UniProt IDs, author chain IDs, and sequence with one extra request",
    )
    include_computed_models: bool = ConfigField(
        default=False,
        title="Include Computed Models",
        description="Also search RCSB's computed structure models; RCSB's sequence server often times out on these",
    )


class PdbSequenceSearchHit(BaseModel):
    """One ranked RCSB sequence-search hit, with its best alignment to the query."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(title="Entity ID", description="RCSB polymer entity ID, e.g. 1BM7_1")
    pdb_id: str = Field(title="PDB ID", description="PDB entry ID of the entity")
    score: float | None = Field(default=None, title="Score", description="RCSB normalized ranking score")
    sequence_identity: float | None = Field(
        default=None, ge=0.0, le=1.0, title="Sequence Identity", description="Identity over the aligned region"
    )
    evalue: float | None = Field(default=None, title="E-value", description="Alignment E-value")
    bitscore: float | None = Field(default=None, title="Bit Score", description="Alignment bit score")
    alignment_length: int | None = Field(default=None, title="Alignment Length", description="Aligned columns")
    mismatches: int | None = Field(default=None, title="Mismatches", description="Mismatched aligned positions")
    gaps_opened: int | None = Field(default=None, title="Gaps Opened", description="Gap openings in the alignment")
    query_beg: int | None = Field(
        default=None, title="Query Start", description="First aligned query residue (1-indexed)"
    )
    query_end: int | None = Field(default=None, title="Query End", description="Last aligned query residue (inclusive)")
    subject_beg: int | None = Field(
        default=None, title="Subject Start", description="First aligned entity residue (1-indexed)"
    )
    subject_end: int | None = Field(
        default=None, title="Subject End", description="Last aligned entity residue (inclusive)"
    )
    query_aligned_seq: str | None = Field(
        default=None, title="Aligned Query", description="Query side of the alignment"
    )
    subject_aligned_seq: str | None = Field(
        default=None, title="Aligned Subject", description="Entity side of the alignment"
    )
    description: str | None = Field(default=None, title="Description", description="Entity description (annotated)")
    organism: str | None = Field(default=None, title="Organism", description="Source organism (annotated)")
    uniprot_ids: list[str] | None = Field(
        default=None, title="UniProt IDs", description="UniProt accessions (annotated)"
    )
    chain_ids: list[str] | None = Field(
        default=None,
        title="Chain IDs",
        description="Author chain IDs of the entity, as pdb-fetch-fasta reports them (annotated)",
    )
    sequence: str | None = Field(default=None, title="Sequence", description="Canonical entity sequence (annotated)")


class PdbSequenceSearchResult(BaseModel):
    """Ranked PDB matches for one query."""

    model_config = ConfigDict(extra="forbid")

    hits: list[PdbSequenceSearchHit] = Field(
        default_factory=list,
        title="Hits",
        description="Ranked polymer-entity matches",
    )
    total_count: int = Field(
        default=0,
        ge=0,
        title="Total Count",
        description="Number of matches reported by RCSB, including any beyond the returned hits",
    )


class PdbSequenceSearchOutput(BaseToolOutput):
    """Sequence-search results, one per query in input order.

    Attributes:
        results (list[PdbSequenceSearchResult]): One result per query, in input order.
        source_url (str): RCSB Search API endpoint used for every query.
    """

    results: list[PdbSequenceSearchResult] = Field(
        default_factory=list,
        title="Results",
        description="One result per query, in input order",
    )
    source_url: str = Field(
        title="Source URL",
        description="RCSB Search API endpoint used",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return supported export formats."""
        return ["json", "csv", "fasta"]

    @property
    def output_format_default(self) -> str:
        """Return the default export format."""
        return "json"

    def _export_output(self, export_path: Any, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")
        if file_format == "json":
            with path.open("w", encoding="utf-8") as handle:
                json.dump(self.model_dump(mode="json"), handle, indent=2)
            return
        if file_format == "csv":
            fields = ["query_index", *PdbSequenceSearchHit.model_fields]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for index, result in enumerate(self.results):
                    for hit in result.hits:
                        row = hit.model_dump()
                        for key in ("uniprot_ids", "chain_ids"):
                            row[key] = ";".join(row[key]) if row[key] else ""
                        writer.writerow({"query_index": index, **row})
            return
        if file_format == "fasta":
            with path.open("w", encoding="utf-8") as handle:
                for index, result in enumerate(self.results):
                    for hit in result.hits:
                        if hit.sequence:
                            label = f" {hit.description}" if hit.description else ""
                            handle.write(f">{hit.entity_id} query={index}{label}\n{hit.sequence}\n")
            return
        raise ValueError(f"Unsupported format: {file_format}")


def _raise_for_status(response: Any, operation: str) -> None:
    """Raise an HTTP error that includes a bounded RCSB response body."""
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        body = str(getattr(response, "text", ""))[:1000]
        raise requests.HTTPError(f"RCSB {operation} failed: {error}; body={body!r}", response=response) from error


def _search_payload(query: PdbSequenceSearchQuery, config: PdbSequenceSearchConfig) -> dict[str, Any]:
    """Build the RCSB Search API request for one query."""
    content_types = ["experimental", "computational"] if config.include_computed_models else ["experimental"]
    return {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "value": query.sequence,
                "sequence_type": query.sequence_type,
                "identity_cutoff": config.identity_cutoff,
                "evalue_cutoff": config.evalue_cutoff,
            },
        },
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": 0, "rows": config.max_hits},
            "results_content_type": content_types,
            "results_verbosity": "verbose",
        },
    }


def _best_alignment(item: dict[str, Any]) -> dict[str, Any]:
    """Return the first alignment record of the sequence service node, or an empty dict."""
    for service in item.get("services", []):
        if not isinstance(service, dict) or service.get("service_type") != "sequence":
            continue
        for node in service.get("nodes", []):
            contexts = node.get("match_context") if isinstance(node, dict) else None
            if contexts and isinstance(contexts[0], dict):
                return {key: contexts[0].get(key) for key in _ALIGNMENT_FIELDS}
    return {}


def _parse_hits(payload: dict[str, Any]) -> list[PdbSequenceSearchHit]:
    """Turn a verbose RCSB result set into hits, keeping RCSB's order."""
    hits = []
    for item in payload.get("result_set", []):
        if not isinstance(item, dict) or item.get("identifier") is None:
            continue
        entity_id = str(item["identifier"])
        hits.append(
            PdbSequenceSearchHit(
                entity_id=entity_id,
                pdb_id=entity_id.split("_", 1)[0],
                score=float(item["score"]) if isinstance(item.get("score"), int | float) else None,
                **_best_alignment(item),
            )
        )
    return hits


def _annotate(session: requests.Session, hits: list[PdbSequenceSearchHit]) -> list[PdbSequenceSearchHit]:
    """Fill annotation fields from one GraphQL request, joined by entity ID (RCSB does not keep order)."""
    if not hits:
        return hits
    request_payload = {"query": _ANNOTATION_QUERY, "variables": {"ids": [hit.entity_id for hit in hits]}}
    response = request_with_retry(
        lambda: session.post(_GRAPHQL_URL, json=request_payload, timeout=_REQUEST_TIMEOUT_SECONDS),
        retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
    )
    _raise_for_status(response, "entity annotation")
    entities = (response.json().get("data") or {}).get("polymer_entities") or []
    by_id = {entity["rcsb_id"]: entity for entity in entities if isinstance(entity, dict) and entity.get("rcsb_id")}
    annotated = []
    for hit in hits:
        entity = by_id.get(hit.entity_id)
        if entity is None:
            annotated.append(hit)
            continue
        identifiers = entity.get("rcsb_polymer_entity_container_identifiers") or {}
        organisms = entity.get("rcsb_entity_source_organism") or []
        annotated.append(
            hit.model_copy(
                update={
                    "description": (entity.get("rcsb_polymer_entity") or {}).get("pdbx_description"),
                    "organism": organisms[0].get("ncbi_scientific_name") if organisms else None,
                    "uniprot_ids": identifiers.get("uniprot_ids") or [],
                    "chain_ids": identifiers.get("auth_asym_ids") or [],
                    "sequence": (entity.get("entity_poly") or {}).get("pdbx_seq_one_letter_code_can"),
                }
            )
        )
    return annotated


def _search(
    session: requests.Session,
    query: PdbSequenceSearchQuery,
    config: PdbSequenceSearchConfig,
) -> PdbSequenceSearchResult:
    """Run one sequence search and optionally annotate its hits."""
    request_payload = _search_payload(query, config)
    response = request_with_retry(
        lambda: session.post(_SEARCH_URL, json=request_payload, timeout=_REQUEST_TIMEOUT_SECONDS),
        retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
    )
    if response.status_code == _NO_CONTENT:
        return PdbSequenceSearchResult()
    _raise_for_status(response, "sequence search")
    payload = response.json()
    hits = _parse_hits(payload)
    if config.annotate:
        hits = _annotate(session, hits)
    return PdbSequenceSearchResult(hits=hits, total_count=int(payload.get("total_count", len(hits))))


def example_input() -> Any:
    """A 30-residue transthyretin fragment; its top hits are transthyretin structures."""
    return PdbSequenceSearchInput(queries=[PdbSequenceSearchQuery(sequence="VRGSPAINVAVHVFRKAADDTWEPFASGKT")])


@tool(
    key="pdb-sequence-search",
    local_only="pdb-sequence-search does not use a gpu and does not need an environment, so it can run in process",
    label="PDB Sequence Search",
    category="database_retrieval",
    input_class=PdbSequenceSearchInput,
    config_class=PdbSequenceSearchConfig,
    output_class=PdbSequenceSearchOutput,
    description="Search experimental PDB polymer entities for sequences similar to a query with RCSB's MMseqs2 sequence service",
    uses_gpu=False,
    example_input=example_input,
    iterable_input_fields=["queries"],
    iterable_output_field="results",
    cacheable=True,
)
def run_pdb_sequence_search(
    inputs: PdbSequenceSearchInput,
    config: PdbSequenceSearchConfig,
    instance: Any = None,
) -> PdbSequenceSearchOutput:
    """Search RCSB's polymer-entity index by sequence, one search per query.

    Args:
        inputs (PdbSequenceSearchInput): Query sequences.
        config (PdbSequenceSearchConfig): Cutoffs, result limit, annotation, and content type.
        instance (Any): Unused; kept for tool API consistency.

    Returns:
        PdbSequenceSearchOutput: One result per query, in input order.
    """
    del instance
    session = build_http_session(
        http_retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
        user_agent=_USER_AGENT,
        # Search and annotation are both POSTs; the shared session retries GET only by default.
        allowed_methods=["POST"],
    )
    try:
        results = [_search(session, query, config) for query in inputs.queries]
    finally:
        session.close()
    return PdbSequenceSearchOutput(results=results, source_url=_SEARCH_URL)
