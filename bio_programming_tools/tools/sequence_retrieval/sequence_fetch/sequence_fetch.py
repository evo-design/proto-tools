"""
Sequence retrieval tool for NCBI Entrez, UniProt, and PDB.

This module provides a standardized interface for fetching DNA, RNA, protein,
and structure data from public databases with validation and batching support.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import requests
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig, ConfigField, tool_cache_iterable
from bio_programming_tools.utils.tool_io import BaseToolInput, BaseToolOutput

logger = logging.getLogger(__name__)


_NCBI_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_UNIPROT_BASE = "https://rest.uniprot.org"
_PDB_ENTRY_BASE = "https://data.rcsb.org/rest/v1/core/entry"
_PDB_FASTA_BASE = "https://www.rcsb.org/fasta/entry"
_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
_NON_CODING_PATTERNS = (
    r"\bncrna\b",
    r"\blncrna\b",
    r"\bsrna\b",
    r"\bsmall\s*rna\b",
    r"\bmicro\s*rna\b",
    r"\bmir[-\d]",
    r"\btrna\b",
    r"\brrna\b",
    r"\bsnrna\b",
    r"\bsnorna\b",
)


# ============================================================================
# Internal Exceptions
# ============================================================================


class _TypeMismatchError(ValueError):
    """Raised when requested molecular type conflicts with target annotations."""


class _NotFoundError(ValueError):
    """Raised when no matching record is found in a source database."""


class _AmbiguousMatchError(ValueError):
    """Raised when multiple candidates are plausible and unresolvable."""


class _UpstreamError(RuntimeError):
    """Raised when an upstream API fails after retries."""


class _RetryableBatchError(RuntimeError):
    """Raised when a whole chunk should be retried or split."""


# ============================================================================
# Data Models
# ============================================================================


class SequenceFetchRequest(BaseModel):
    """Single fetch request.

    Attributes:
        request_id (Optional[str]): Optional caller-provided request identifier.
        target_name (str): Gene, protein, or RNA name to resolve.
        organism (str): Organism name used for disambiguation.
        sequence_types (List[str]): Requested outputs: protein, dna, rna, or structure.
        uniprot_id (Optional[str]): UniProt accession override.
        genbank_accession (Optional[str]): GenBank accession override.
        refseq_accession (Optional[str]): RefSeq accession override.
        pdb_id (Optional[str]): PDB accession override.
        gene_id (Optional[str]): NCBI Gene ID override.
        protein_id (Optional[str]): NCBI protein accession override.
        transcript_id (Optional[str]): Transcript accession override.
        genomic_coordinates (Optional[str]): Genomic interval like NC_000913.3:1-100:+.
        assembly (Optional[str]): Genome assembly name for coordinate context.
        additional_ids (Dict[str, str]): Extra IDs used for custom routing.
    """

    request_id: Optional[str] = Field(default=None, description="Optional request identifier")
    target_name: str = Field(min_length=1, description="Gene, RNA, or protein name")
    organism: str = Field(min_length=1, description="Organism for disambiguation")
    sequence_types: List[
        Literal[
            "protein",
            "dna_genomic",
            "dna_cds",
            "rna_transcript",
            "rna_premrna",
            "structure",
        ]
    ] = Field(description="Requested output molecule types")
    uniprot_id: Optional[str] = Field(default=None, description="UniProt accession override")
    genbank_accession: Optional[str] = Field(default=None, description="GenBank accession override")
    refseq_accession: Optional[str] = Field(default=None, description="RefSeq accession override")
    pdb_id: Optional[str] = Field(default=None, description="PDB accession override")
    gene_id: Optional[str] = Field(default=None, description="NCBI Gene ID override")
    protein_id: Optional[str] = Field(default=None, description="NCBI protein accession override")
    transcript_id: Optional[str] = Field(default=None, description="Transcript accession override")
    genomic_coordinates: Optional[str] = Field(
        default=None,
        description="Genomic coordinates as accession:start-end:strand",
    )
    assembly: Optional[str] = Field(default=None, description="Genome assembly for coordinate mapping")
    additional_ids: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional IDs for custom routing",
    )

    @field_validator("sequence_types", mode="before")
    @classmethod
    def normalize_sequence_types(cls, value):
        """Normalize a single string to a list and remove duplicates."""
        if isinstance(value, str):
            value = [value]
        if not value:
            raise ValueError("sequence_types must include at least one type")

        normalized = []
        seen = set()
        for item in value:
            if item not in seen:
                normalized.append(item)
                seen.add(item)
        return normalized


class SequenceFetchInput(BaseToolInput):
    """Input for sequence retrieval.

    Attributes:
        requests (List[SequenceFetchRequest]): One or more retrieval requests.
    """

    requests: List[SequenceFetchRequest] = Field(description="One or more retrieval requests")

    @field_validator("requests", mode="before")
    @classmethod
    def normalize_requests(cls, value):
        """Accept a single request object or a list."""
        if isinstance(value, dict):
            return [value]
        if isinstance(value, SequenceFetchRequest):
            return [value]
        return value

    @field_validator("requests")
    @classmethod
    def validate_requests(cls, value: List[SequenceFetchRequest]) -> List[SequenceFetchRequest]:
        """Require at least one request."""
        if not value:
            raise ValueError("requests must not be empty")
        return value


class FetchedSequence(BaseModel):
    """Sequence payload returned by one source.

    Attributes:
        sequence_type (str): Requested type of this sequence.
        source_database (str): Upstream source database label.
        accession (Optional[str]): Source accession identifier.
        sequence (str): Retrieved sequence string.
        length (int): Sequence length.
        checksum_sha256 (Optional[str]): SHA256 checksum of sequence.
        source_url (Optional[str]): Source URL for provenance.
        inferred (bool): True if sequence is inferred, not directly curated.
    """

    sequence_type: Literal[
        "protein",
        "dna_genomic",
        "dna_cds",
        "rna_transcript",
        "rna_premrna",
    ] = Field(description="Requested type for this sequence")
    source_database: Literal["ncbi", "uniprot", "pdb"] = Field(
        description="Source database for this sequence"
    )
    accession: Optional[str] = Field(default=None, description="Source accession identifier")
    sequence: str = Field(description="Retrieved sequence")
    length: int = Field(ge=0, description="Sequence length")
    checksum_sha256: Optional[str] = Field(default=None, description="SHA256 checksum")
    source_url: Optional[str] = Field(default=None, description="Source URL for provenance")
    inferred: bool = Field(default=False, description="Whether sequence is inferred")


class FetchedStructure(BaseModel):
    """Structure payload returned by PDB.

    Attributes:
        pdb_id (str): PDB accession.
        source_database (str): Upstream source database label.
        title (Optional[str]): Structure title.
        method (Optional[str]): Experimental method.
        resolution (Optional[float]): Resolution in angstroms.
        source_url (str): Canonical URL for this structure.
    """

    pdb_id: str = Field(description="PDB accession")
    source_database: Literal["pdb"] = Field(default="pdb", description="Source database")
    title: Optional[str] = Field(default=None, description="Structure title")
    method: Optional[str] = Field(default=None, description="Experimental method")
    resolution: Optional[float] = Field(default=None, description="Resolution in angstroms")
    source_url: str = Field(description="Canonical structure URL")


class SequenceFetchResult(BaseModel):
    """Per-request fetch result.

    Attributes:
        request_id (str): Request identifier used in this result.
        target_name (str): Original target name.
        organism (str): Original organism name.
        requested_types (List[str]): Requested output molecule types.
        status (str): One of success, warning, or failed.
        fetched_sequences (List[FetchedSequence]): Retrieved sequence records.
        fetched_structures (List[FetchedStructure]): Retrieved structure records.
        resolved_ids (Dict[str, str]): IDs resolved or used during retrieval.
        warnings (List[str]): Non-fatal warnings.
        errors (List[str]): Fatal or partial failure messages.
        metadata (Dict[str, Any]): Extra structured provenance metadata.
    """

    request_id: str = Field(description="Request identifier")
    target_name: str = Field(description="Original target name")
    organism: str = Field(description="Original organism name")
    requested_types: List[str] = Field(description="Requested molecule types")
    status: Literal["success", "warning", "failed"] = Field(description="Result status")
    fetched_sequences: List[FetchedSequence] = Field(
        default_factory=list,
        description="Retrieved sequence records",
    )
    fetched_structures: List[FetchedStructure] = Field(
        default_factory=list,
        description="Retrieved structure records",
    )
    resolved_ids: Dict[str, str] = Field(
        default_factory=dict,
        description="Resolved identifiers used in retrieval",
    )
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")
    errors: List[str] = Field(default_factory=list, description="Fatal or partial failure messages")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra provenance metadata")


class SequenceFetchOutput(BaseToolOutput):
    """Output from sequence retrieval.

    Attributes:
        results (List[SequenceFetchResult]): Per-request retrieval outcomes.
        split_events (int): Number of adaptive chunk split events.
        retry_events (int): Number of chunk retry attempts.
        chunk_plan (List[int]): Final chunk sizes successfully processed.
        num_requests (int): Number of requests in this run.
        num_success (int): Number of successful request results.
        num_warning (int): Number of warning request results.
        num_completed (int): Number of successful or warning request results.
        num_failed (int): Number of failed request results.
    """

    results: List[SequenceFetchResult] = Field(
        default_factory=list,
        description="Per-request retrieval outcomes",
    )
    split_events: int = Field(default=0, ge=0, description="Adaptive chunk split count")
    retry_events: int = Field(default=0, ge=0, description="Chunk retry attempt count")
    chunk_plan: List[int] = Field(
        default_factory=list,
        description="Final processed chunk sizes",
    )

    @computed_field
    @property
    def num_requests(self) -> int:
        """Total number of requests."""
        return len(self.results)

    @computed_field
    @property
    def num_success(self) -> int:
        """Number of successful results."""
        return sum(1 for r in self.results if r.status == "success")

    @computed_field
    @property
    def num_warning(self) -> int:
        """Number of warning results."""
        return sum(1 for r in self.results if r.status == "warning")

    @computed_field
    @property
    def num_completed(self) -> int:
        """Number of completed results including warnings."""
        return sum(1 for r in self.results if r.status in {"success", "warning"})

    @computed_field
    @property
    def num_failed(self) -> int:
        """Number of failed results."""
        return sum(1 for r in self.results if r.status == "failed")

    @property
    def output_format_options(self) -> List[str]:
        return ["json", "fasta"]

    @property
    def output_format_default(self) -> str:
        return "json"

    def _export_output(self, export_path: str | Path, file_format: str):
        if file_format == "json":
            path = Path(export_path).with_suffix(".json")
            with path.open("w", encoding="utf-8") as handle:
                json.dump(self.model_dump(mode="json"), handle, indent=2)
            return

        if file_format == "fasta":
            path = Path(export_path).with_suffix(".fasta")
            with path.open("w", encoding="utf-8") as handle:
                for result in self.results:
                    for record in result.fetched_sequences:
                        acc = record.accession or "unknown"
                        header = (
                            f">{result.request_id}|{result.target_name}|"
                            f"{record.sequence_type}|{record.source_database}|{acc}"
                        )
                        handle.write(f"{header}\n{record.sequence}\n")
            return

        raise ValueError(f"Unsupported format: {file_format}")


class SequenceFetchConfig(BaseConfig):
    """Configuration for sequence retrieval.

    Attributes:
        chunk_size (int): Initial number of requests per chunk.
        max_chunk_size (int): Upper bound for chunk size.
        min_chunk_size (int): Lower bound for adaptive splitting.
        chunk_timeout_seconds (int): Timeout per chunk before splitting.
        request_timeout_seconds (int): Timeout per HTTP request.
        max_chunk_retries (int): Retries before splitting a chunk.
        http_retries (int): Retries per upstream HTTP call.
        backoff_seconds (float): Base exponential backoff in seconds.
        max_candidates_per_source (int): Max source candidates for name lookups.
        strict_type_checks (bool): Enforce ncRNA/protein mismatch checks.
        fail_on_type_mismatch (bool): Convert type mismatches to hard failures.
        include_sequence_checksums (bool): Include SHA256 checksums in outputs.
        ncbi_api_key (Optional[str]): Optional API key for NCBI Entrez.
        ncbi_email (Optional[str]): Optional contact email for NCBI requests.
        user_agent (str): HTTP user-agent sent to upstream APIs.
    """

    chunk_size: int = ConfigField(
        title="Chunk Size",
        default=25,
        ge=1,
        description="Initial requests processed per chunk",
    )
    max_chunk_size: int = ConfigField(
        title="Max Chunk Size",
        default=100,
        ge=1,
        description="Maximum requests allowed in one chunk",
        advanced=True,
    )
    min_chunk_size: int = ConfigField(
        title="Min Chunk Size",
        default=1,
        ge=1,
        description="Minimum requests allowed after splitting",
        advanced=True,
    )
    chunk_timeout_seconds: int = ConfigField(
        title="Chunk Timeout",
        default=60,
        ge=1,
        description="Chunk timeout in seconds",
        advanced=True,
    )
    request_timeout_seconds: int = ConfigField(
        title="Request Timeout",
        default=15,
        ge=1,
        description="HTTP timeout in seconds",
        advanced=True,
    )
    max_chunk_retries: int = ConfigField(
        title="Chunk Retries",
        default=2,
        ge=0,
        description="Retries before splitting a chunk",
        advanced=True,
    )
    http_retries: int = ConfigField(
        title="HTTP Retries",
        default=2,
        ge=0,
        description="Retries for upstream HTTP requests",
        advanced=True,
    )
    backoff_seconds: float = ConfigField(
        title="Backoff Seconds",
        default=1.0,
        ge=0.0,
        description="Base exponential backoff time",
        advanced=True,
    )
    max_candidates_per_source: int = ConfigField(
        title="Max Candidates",
        default=5,
        ge=1,
        le=25,
        description="Max candidates per source lookup",
        advanced=True,
    )
    strict_type_checks: bool = ConfigField(
        title="Strict Type Checks",
        default=True,
        description="Enforce ncRNA versus protein checks",
    )
    fail_on_type_mismatch: bool = ConfigField(
        title="Fail On Mismatch",
        default=True,
        description="Treat type mismatch as failure",
    )
    include_sequence_checksums: bool = ConfigField(
        title="Include Checksums",
        default=True,
        description="Include SHA256 checksums per sequence",
        advanced=True,
    )
    ncbi_api_key: Optional[str] = ConfigField(
        title="NCBI API Key",
        default=None,
        description="Optional NCBI API key",
        advanced=True,
    )
    ncbi_email: Optional[str] = ConfigField(
        title="NCBI Email",
        default=None,
        description="Optional NCBI contact email",
        advanced=True,
    )
    user_agent: str = ConfigField(
        title="User Agent",
        default="bio-programming-tools/sequence-fetch-v1",
        description="HTTP user-agent string",
        advanced=True,
    )

    @model_validator(mode="after")
    def validate_chunk_bounds(self):
        """Validate chunk size bounds."""
        if self.min_chunk_size > self.max_chunk_size:
            raise ValueError("min_chunk_size must be <= max_chunk_size")
        if self.chunk_size < self.min_chunk_size:
            raise ValueError("chunk_size must be >= min_chunk_size")
        if self.chunk_size > self.max_chunk_size:
            raise ValueError("chunk_size must be <= max_chunk_size")
        return self


# ============================================================================
# Tool Implementation
# ============================================================================


@tool(
    key="sequence-fetch",
    label="Multi-source Sequence Fetch",
    category="sequence_retrieval",
    input=SequenceFetchInput,
    config=SequenceFetchConfig,
    output=SequenceFetchOutput,
    description="Fetch DNA, RNA, protein, and structure records from NCBI, UniProt, and PDB",
    uses_gpu=False,
)
@tool_cache_iterable(
    input_iterable_field="requests",
    output_iterable_field="results",
    tool_name="sequence-fetch",
)
def run_sequence_fetch(
    inputs: SequenceFetchInput,
    config: SequenceFetchConfig,
    instance=None,
) -> SequenceFetchOutput:
    """Fetch DNA, RNA, protein, and structure records from NCBI, UniProt, and PDB.

    This tool resolves IDs and names across NCBI Entrez, UniProt, and PDB for
    sequence and structure retrieval. It supports batch requests with adaptive
    chunk splitting when chunks exceed timeout or retry limits.

    Args:
        inputs (SequenceFetchInput): One or more sequence retrieval requests.
        config (SequenceFetchConfig): Chunking, timeout, and validation settings.

    Returns:
        SequenceFetchOutput: Per-request retrieval status, sequences, and metadata.

    Examples:
        >>> inputs = SequenceFetchInput(
        ...     requests=[
        ...         {
        ...             "target_name": "lacI",
        ...             "organism": "Escherichia coli",
        ...             "sequence_types": ["protein", "dna_genomic"],
        ...         }
        ...     ]
        ... )
        >>> config = SequenceFetchConfig()
        >>> result = run_sequence_fetch(inputs, config)
        >>> print(result.num_requests, result.num_success)
    """
    del instance  # unused; kept for tool API consistency

    indexed_requests = list(enumerate(inputs.requests))
    result_map: Dict[int, SequenceFetchResult] = {}

    with requests.Session() as session:
        session.headers.update({"User-Agent": config.user_agent})

        split_events, retry_events, chunk_plan = _run_batched_fetch(
            indexed_requests=indexed_requests,
            result_map=result_map,
            config=config,
            session=session,
        )

    ordered_results = [result_map[idx] for idx in range(len(inputs.requests))]

    return SequenceFetchOutput(
        results=ordered_results,
        split_events=split_events,
        retry_events=retry_events,
        chunk_plan=chunk_plan,
        metadata={
            "num_requests": len(inputs.requests),
            "chunk_size": config.chunk_size,
            "max_chunk_size": config.max_chunk_size,
            "min_chunk_size": config.min_chunk_size,
            "sources": ["ncbi", "uniprot", "pdb"],
        },
    )


def _run_batched_fetch(
    indexed_requests: List[Tuple[int, SequenceFetchRequest]],
    result_map: Dict[int, SequenceFetchResult],
    config: SequenceFetchConfig,
    session: requests.Session,
) -> Tuple[int, int, List[int]]:
    """Run batched fetch with adaptive splitting."""
    split_events = 0
    retry_events = 0
    final_chunk_sizes: List[int] = []

    initial_chunk_size = min(config.chunk_size, config.max_chunk_size)
    chunks: List[List[Tuple[int, SequenceFetchRequest]]] = [
        indexed_requests[i: i + initial_chunk_size]
        for i in range(0, len(indexed_requests), initial_chunk_size)
    ]

    while chunks:
        chunk = chunks.pop(0)

        if len(chunk) > config.max_chunk_size:
            split_events += 1
            mid = len(chunk) // 2
            chunks = [chunk[:mid], chunk[mid:]] + chunks
            continue

        try:
            chunk_results, chunk_retries = _process_chunk(chunk, config, session)
            retry_events += chunk_retries
            final_chunk_sizes.append(len(chunk))
            for index, output in chunk_results:
                result_map[index] = output
        except _RetryableBatchError as exc:
            if len(chunk) <= config.min_chunk_size:
                retry_events += config.max_chunk_retries
                for index, request in chunk:
                    result_map[index] = _failed_result(
                        request=request,
                        request_index=index,
                        error=f"UPSTREAM_UNAVAILABLE: {exc}",
                    )
                continue

            split_events += 1
            mid = max(config.min_chunk_size, len(chunk) // 2)
            left = chunk[:mid]
            right = chunk[mid:]
            chunks = [left, right] + chunks

    return split_events, retry_events, final_chunk_sizes


def _process_chunk(
    chunk: List[Tuple[int, SequenceFetchRequest]],
    config: SequenceFetchConfig,
    session: requests.Session,
) -> Tuple[List[Tuple[int, SequenceFetchResult]], int]:
    """Process one chunk with retries and timeout handling."""
    retry_events = 0

    for attempt in range(config.max_chunk_retries + 1):
        started = time.monotonic()
        results: List[Tuple[int, SequenceFetchResult]] = []
        try:
            for request_index, request in chunk:
                elapsed = time.monotonic() - started
                if elapsed > config.chunk_timeout_seconds:
                    raise TimeoutError(
                        f"Chunk timed out after {elapsed:.2f}s "
                        f"(limit {config.chunk_timeout_seconds}s)"
                    )

                output = _process_single_request(
                    request=request,
                    request_index=request_index,
                    config=config,
                    session=session,
                )
                results.append((request_index, output))

            return results, retry_events
        except (_UpstreamError, TimeoutError) as exc:
            if attempt >= config.max_chunk_retries:
                raise _RetryableBatchError(str(exc)) from exc

            retry_events += 1
            sleep_seconds = config.backoff_seconds * (2**attempt)
            jitter = random.uniform(0.0, max(config.backoff_seconds, 0.001))
            time.sleep(sleep_seconds + jitter)


def _process_single_request(
    request: SequenceFetchRequest,
    request_index: int,
    config: SequenceFetchConfig,
    session: requests.Session,
) -> SequenceFetchResult:
    """Process one request and return a normalized result object."""
    request_id = request.request_id or f"request_{request_index}"
    warnings: List[str] = []
    errors: List[str] = []
    sequences: List[FetchedSequence] = []
    structures: List[FetchedStructure] = []
    resolved_ids: Dict[str, str] = {}

    try:
        _validate_request_type_compatibility(request, config)
    except _TypeMismatchError as exc:
        if config.fail_on_type_mismatch:
            return _failed_result(
                request=request,
                request_index=request_index,
                error=f"TYPE_MISMATCH: {exc}",
            )
        warnings.append(f"TYPE_MISMATCH: {exc}")

    if "protein" in request.sequence_types and request.genomic_coordinates:
        warnings.append(
            "Protein from genomic coordinates is inferred and may be ambiguous due to introns."
        )

    if (
        "protein" in request.sequence_types
        and "dna_genomic" in request.sequence_types
        and _organism_likely_has_introns(request.organism)
    ):
        warnings.append(
            "Genomic DNA includes introns in many organisms; DNA to protein mapping may be indirect."
        )

    for sequence_type in request.sequence_types:
        try:
            if sequence_type == "protein":
                fetched, ids, local_warnings = _fetch_protein(request, config, session)
                sequences.append(fetched)
                resolved_ids.update(ids)
                warnings.extend(local_warnings)

            elif sequence_type == "dna_genomic":
                fetched, ids, local_warnings = _fetch_dna_genomic(request, config, session)
                sequences.append(fetched)
                resolved_ids.update(ids)
                warnings.extend(local_warnings)

            elif sequence_type == "dna_cds":
                fetched, ids, local_warnings = _fetch_dna_cds(request, config, session)
                sequences.append(fetched)
                resolved_ids.update(ids)
                warnings.extend(local_warnings)

            elif sequence_type == "rna_transcript":
                fetched, ids, local_warnings = _fetch_rna_transcript(request, config, session)
                sequences.append(fetched)
                resolved_ids.update(ids)
                warnings.extend(local_warnings)

            elif sequence_type == "rna_premrna":
                fetched, ids, local_warnings = _fetch_rna_premrna(request, config, session)
                sequences.append(fetched)
                resolved_ids.update(ids)
                warnings.extend(local_warnings)

            elif sequence_type == "structure":
                fetched_structure, ids, local_warnings = _fetch_structure(
                    request, config, session, resolved_ids
                )
                structures.append(fetched_structure)
                resolved_ids.update(ids)
                warnings.extend(local_warnings)

        except _NotFoundError as exc:
            errors.append(f"NOT_FOUND[{sequence_type}]: {exc}")
        except _AmbiguousMatchError as exc:
            warnings.append(f"AMBIGUOUS_MATCH[{sequence_type}]: {exc}")
        except _TypeMismatchError as exc:
            errors.append(f"TYPE_MISMATCH[{sequence_type}]: {exc}")
        except _UpstreamError:
            raise
        except Exception as exc:  # pragma: no cover - guardrail
            errors.append(f"UNEXPECTED_ERROR[{sequence_type}]: {exc}")

    status: Literal["success", "warning", "failed"]
    if errors and not (sequences or structures):
        status = "failed"
    elif errors or warnings:
        status = "warning"
    else:
        status = "success"

    dedup_warnings = _dedupe_preserve_order(warnings)
    dedup_errors = _dedupe_preserve_order(errors)

    source_databases = {seq.source_database for seq in sequences}
    source_databases.update(struct.source_database for struct in structures)

    sequence_accessions = [seq.accession for seq in sequences if seq.accession]
    structure_ids = [struct.pdb_id for struct in structures if struct.pdb_id]

    return SequenceFetchResult(
        request_id=request_id,
        target_name=request.target_name,
        organism=request.organism,
        requested_types=list(request.sequence_types),
        status=status,
        fetched_sequences=sequences,
        fetched_structures=structures,
        resolved_ids=resolved_ids,
        warnings=dedup_warnings,
        errors=dedup_errors,
        metadata={
            "assembly": request.assembly,
            "num_requested_types": len(request.sequence_types),
            "num_sequences": len(sequences),
            "num_structures": len(structures),
            "num_warnings": len(dedup_warnings),
            "num_errors": len(dedup_errors),
            "source_databases": sorted(source_databases),
            "sequence_accessions": sequence_accessions,
            "structure_ids": structure_ids,
        },
    )


def _validate_request_type_compatibility(
    request: SequenceFetchRequest,
    config: SequenceFetchConfig,
) -> None:
    """Validate obvious ncRNA/protein mismatches."""
    if not config.strict_type_checks:
        return

    if "protein" not in request.sequence_types:
        return

    name = request.target_name.lower()
    if any(re.search(pattern, name) for pattern in _NON_CODING_PATTERNS):
        raise _TypeMismatchError(
            f"Target '{request.target_name}' appears non-coding but protein was requested"
        )

    refseq_like = (request.refseq_accession or request.transcript_id or "").upper()
    if refseq_like.startswith(("NR_", "XR_")):
        raise _TypeMismatchError(
            f"RefSeq transcript '{refseq_like}' is non-coding (NR_/XR_ prefix)"
        )


def _fetch_protein(
    request: SequenceFetchRequest,
    config: SequenceFetchConfig,
    session: requests.Session,
) -> Tuple[FetchedSequence, Dict[str, str], List[str]]:
    """Fetch protein sequence using ID-priority resolution."""
    warnings: List[str] = []

    if request.uniprot_id:
        entry = _fetch_uniprot_entry(request.uniprot_id, config, session)
        sequence = entry.get("sequence", {}).get("value")
        if not sequence:
            raise _NotFoundError(f"No sequence found for UniProt ID '{request.uniprot_id}'")

        accession = entry.get("primaryAccession", request.uniprot_id)
        ids = {"uniprot_id": accession}

        pdb_xrefs = _extract_uniprot_pdb_crossrefs(entry)
        if pdb_xrefs and not request.pdb_id and "structure" not in request.sequence_types:
            warnings.append(
                f"UniProt entry maps to PDB IDs {', '.join(pdb_xrefs[:3])}; use structure type to fetch"
            )

        return (
            _sequence_record(
                sequence_type="protein",
                source_database="uniprot",
                accession=accession,
                sequence=sequence,
                source_url=f"{_UNIPROT_BASE}/uniprotkb/{accession}",
                config=config,
                inferred=False,
            ),
            ids,
            warnings,
        )

    protein_accession = request.protein_id or _preferred_accession(request)
    if protein_accession:
        header, sequence, url = _ncbi_fetch_fasta(
            db="protein",
            identifier=protein_accession,
            config=config,
            session=session,
            rettype="fasta",
        )
        accession = _accession_from_header(header) or protein_accession
        return (
            _sequence_record(
                sequence_type="protein",
                source_database="ncbi",
                accession=accession,
                sequence=sequence,
                source_url=url,
                config=config,
                inferred=False,
            ),
            {"protein_id": accession},
            warnings,
        )

    if request.pdb_id:
        fasta_text = _request_text(
            session=session,
            method="GET",
            url=f"{_PDB_FASTA_BASE}/{request.pdb_id.upper()}",
            config=config,
            source_label="pdb-fasta",
        )
        records = _parse_fasta_records(fasta_text)
        if not records:
            raise _NotFoundError(f"No protein sequence found for PDB ID '{request.pdb_id}'")
        protein_records = [
            (h, s) for h, s in records if _is_protein_sequence(s)
        ]
        if not protein_records:
            raise _NotFoundError(
                f"PDB ID '{request.pdb_id}' has no protein chains "
                f"(found {len(records)} non-protein chain(s))"
            )
        header, sequence = protein_records[0]
        accession = request.pdb_id.upper()
        if len(protein_records) > 1:
            warnings.append(
                f"Using first protein chain from PDB FASTA; "
                f"{len(protein_records)} protein chains available."
            )
        return (
            _sequence_record(
                sequence_type="protein",
                source_database="pdb",
                accession=accession,
                sequence=sequence,
                source_url=f"https://www.rcsb.org/structure/{accession}",
                config=config,
                inferred=False,
            ),
            {"pdb_id": accession, "protein_id": _accession_from_header(header) or accession},
            warnings,
        )

    # Name-based fallback: UniProt first, then NCBI protein search.
    entry = _search_uniprot_entry(
        target_name=request.target_name,
        organism=request.organism,
        prefer_pdb_crossref="structure" in request.sequence_types,
        config=config,
        session=session,
    )
    if entry is not None:
        sequence = entry.get("sequence", {}).get("value")
        accession = entry.get("primaryAccession")
        if sequence and accession:
            return (
                _sequence_record(
                    sequence_type="protein",
                    source_database="uniprot",
                    accession=accession,
                    sequence=sequence,
                    source_url=f"{_UNIPROT_BASE}/uniprotkb/{accession}",
                    config=config,
                    inferred=False,
                ),
                {"uniprot_id": accession},
                [],
            )

    ids = _ncbi_esearch(
        db="protein",
        term=_ncbi_term_for_request(request),
        config=config,
        session=session,
    )
    if not ids:
        raise _NotFoundError(
            f"No protein candidates found for '{request.target_name}' in '{request.organism}'"
        )

    if len(ids) > 1:
        warnings.append(
            f"Multiple NCBI protein candidates found ({len(ids)}); using top hit {ids[0]}"
        )

    header, sequence, url = _ncbi_fetch_fasta(
        db="protein",
        identifier=ids[0],
        config=config,
        session=session,
        rettype="fasta",
    )
    accession = _accession_from_header(header) or ids[0]

    return (
        _sequence_record(
            sequence_type="protein",
            source_database="ncbi",
            accession=accession,
            sequence=sequence,
            source_url=url,
            config=config,
            inferred=False,
        ),
        {"protein_id": accession},
        warnings,
    )


def _fetch_dna_genomic(
    request: SequenceFetchRequest,
    config: SequenceFetchConfig,
    session: requests.Session,
) -> Tuple[FetchedSequence, Dict[str, str], List[str]]:
    """Fetch genomic DNA sequence."""
    warnings: List[str] = []

    coords = _parse_coordinates(request.genomic_coordinates)
    accession_hint = request.genbank_accession or request.refseq_accession

    if coords is not None:
        coord_accession = coords[0] or accession_hint
        if coord_accession is None:
            raise _NotFoundError(
                "Genomic coordinates need an accession in coordinates or accession fields"
            )

        header, sequence, url = _ncbi_fetch_fasta(
            db="nuccore",
            identifier=coord_accession,
            config=config,
            session=session,
            rettype="fasta",
            seq_start=coords[1],
            seq_stop=coords[2],
            strand=coords[3],
        )
        accession = _accession_from_header(header) or coord_accession
        return (
            _sequence_record(
                sequence_type="dna_genomic",
                source_database="ncbi",
                accession=accession,
                sequence=sequence,
                source_url=url,
                config=config,
                inferred=False,
            ),
            {"genbank_accession": accession},
            warnings,
        )

    direct_accession = _preferred_accession(request)
    if direct_accession:
        header, sequence, url = _ncbi_fetch_fasta(
            db="nuccore",
            identifier=direct_accession,
            config=config,
            session=session,
            rettype="fasta",
        )
        accession = _accession_from_header(header) or direct_accession
        return (
            _sequence_record(
                sequence_type="dna_genomic",
                source_database="ncbi",
                accession=accession,
                sequence=sequence,
                source_url=url,
                config=config,
                inferred=False,
            ),
            {"genbank_accession": accession},
            warnings,
        )

    term = (
        f"({_ncbi_term_for_request(request)}) "
        "AND biomol_genomic[PROP]"
    )
    ids = _ncbi_esearch(db="nuccore", term=term, config=config, session=session)
    if not ids:
        ids = _ncbi_esearch(
            db="nuccore",
            term=_ncbi_term_for_request(request),
            config=config,
            session=session,
        )
    if not ids:
        raise _NotFoundError(
            f"No genomic DNA candidates found for '{request.target_name}' in '{request.organism}'"
        )

    attempted_not_found: List[str] = []
    selected_id: Optional[str] = None
    header = ""
    sequence = ""
    url = ""

    for candidate_id in ids:
        try:
            header, sequence, url = _ncbi_fetch_fasta(
                db="nuccore",
                identifier=candidate_id,
                config=config,
                session=session,
                rettype="fasta",
            )
            selected_id = candidate_id
            break
        except _NotFoundError:
            attempted_not_found.append(candidate_id)

    if selected_id is None:
        attempted_display = ", ".join(ids[:5])
        try:
            gene_record, gene_ids, gene_warnings = _fetch_dna_genomic_from_gene_locus(
                request=request,
                config=config,
                session=session,
            )
            warnings.append(
                "Nuccore genomic candidates lacked FASTA; used gene-locus genomic fallback"
            )
            warnings.extend(gene_warnings)
            return gene_record, gene_ids, warnings
        except _NotFoundError as gene_exc:
            raise _NotFoundError(
                "No FASTA records for any genomic candidates in nuccore "
                f"({attempted_display}); gene-locus fallback failed: {gene_exc}"
            ) from gene_exc

    if attempted_not_found:
        warnings.append(
            "Primary genomic candidate(s) had no FASTA records; "
            f"used fallback candidate {selected_id}"
        )

    accession = _accession_from_header(header) or selected_id

    return (
        _sequence_record(
            sequence_type="dna_genomic",
            source_database="ncbi",
            accession=accession,
            sequence=sequence,
            source_url=url,
            config=config,
            inferred=False,
        ),
        {"genbank_accession": accession},
        warnings,
    )


def _fetch_dna_genomic_from_gene_locus(
    request: SequenceFetchRequest,
    config: SequenceFetchConfig,
    session: requests.Session,
) -> Tuple[FetchedSequence, Dict[str, str], List[str]]:
    """Fetch genomic DNA by resolving gene locus coordinates from NCBI gene."""
    warnings: List[str] = []

    if request.gene_id:
        gene_ids = [request.gene_id]
    else:
        gene_ids = _ncbi_esearch(
            db="gene",
            term=_ncbi_gene_term(request.target_name, request.organism),
            config=config,
            session=session,
        )
    if not gene_ids:
        raise _NotFoundError(
            f"No gene records found for '{request.target_name}' in '{request.organism}'"
        )

    selected_gene_id = gene_ids[0]
    if len(gene_ids) > 1:
        warnings.append(
            f"Multiple gene candidates found ({len(gene_ids)}); using top hit {selected_gene_id}"
        )

    summary = _ncbi_esummary(
        db="gene",
        identifier=selected_gene_id,
        config=config,
        session=session,
    )
    gene_payload = summary.get(selected_gene_id, {})
    genomic_info = gene_payload.get("genomicinfo", [])
    if not genomic_info:
        raise _NotFoundError(
            f"Gene record '{selected_gene_id}' lacks genomic coordinates in esummary"
        )

    selected_region = None
    for region in genomic_info:
        if region.get("chraccver"):
            selected_region = region
            break
    if selected_region is None:
        raise _NotFoundError(
            f"Gene record '{selected_gene_id}' has no chraccver coordinates"
        )

    chr_accession = selected_region["chraccver"]
    chr_start = int(selected_region["chrstart"])
    chr_stop = int(selected_region["chrstop"])
    seq_start = min(chr_start, chr_stop) + 1
    seq_stop = max(chr_start, chr_stop) + 1
    strand = "-" if chr_start > chr_stop else "+"

    header, sequence, url = _ncbi_fetch_fasta(
        db="nuccore",
        identifier=chr_accession,
        config=config,
        session=session,
        rettype="fasta",
        seq_start=seq_start,
        seq_stop=seq_stop,
        strand=strand,
    )

    accession = _accession_from_header(header) or chr_accession
    warnings.append(
        f"Fetched gene-locus genomic interval {chr_accession}:{seq_start}-{seq_stop}:{strand}"
    )

    return (
        _sequence_record(
            sequence_type="dna_genomic",
            source_database="ncbi",
            accession=accession,
            sequence=sequence,
            source_url=url,
            config=config,
            inferred=False,
        ),
        {"gene_id": selected_gene_id, "genbank_accession": accession},
        warnings,
    )


def _fetch_dna_cds(
    request: SequenceFetchRequest,
    config: SequenceFetchConfig,
    session: requests.Session,
) -> Tuple[FetchedSequence, Dict[str, str], List[str]]:
    """Fetch coding DNA sequence (CDS)."""
    warnings: List[str] = []

    accession = _preferred_accession(request)
    if accession:
        fasta_text, url = _ncbi_fetch_text(
            db="nuccore",
            identifier=accession,
            config=config,
            session=session,
            rettype="fasta_cds_na",
        )
        records = _parse_fasta_records(fasta_text)
        selected = _select_best_record(records, request.target_name)
        if selected is None:
            raise _NotFoundError(f"No CDS sequence found for accession '{accession}'")

        header, sequence = selected
        cds_acc = _accession_from_header(header) or accession
        return (
            _sequence_record(
                sequence_type="dna_cds",
                source_database="ncbi",
                accession=cds_acc,
                sequence=sequence,
                source_url=url,
                config=config,
                inferred=False,
            ),
            {"cds_accession": cds_acc},
            warnings,
        )

    # Fallback via protein accession: map protein -> nuccore using CDS FASTA output.
    if request.protein_id:
        fasta_text, url = _ncbi_fetch_text(
            db="protein",
            identifier=request.protein_id,
            config=config,
            session=session,
            rettype="fasta_cds_na",
        )
        records = _parse_fasta_records(fasta_text)
        selected = _select_best_record(records, request.target_name)
        if selected is not None:
            header, sequence = selected
            cds_acc = _accession_from_header(header) or request.protein_id
            warnings.append("CDS inferred from protein record using NCBI fasta_cds_na")
            return (
                _sequence_record(
                    sequence_type="dna_cds",
                    source_database="ncbi",
                    accession=cds_acc,
                    sequence=sequence,
                    source_url=url,
                    config=config,
                    inferred=True,
                ),
                {"cds_accession": cds_acc},
                warnings,
            )

    # Name-based fallback through nuccore.
    term = (
        f"({_ncbi_term_for_request(request)}) "
        "AND (mRNA[Title] OR CDS[Title] OR coding[Title])"
    )
    ids = _ncbi_esearch(db="nuccore", term=term, config=config, session=session)
    if not ids:
        raise _NotFoundError(
            f"No CDS candidates found for '{request.target_name}' in '{request.organism}'"
        )

    if len(ids) > 1:
        warnings.append(f"Multiple CDS candidates found ({len(ids)}); using top hit {ids[0]}")

    fasta_text, url = _ncbi_fetch_text(
        db="nuccore",
        identifier=ids[0],
        config=config,
        session=session,
        rettype="fasta_cds_na",
    )
    records = _parse_fasta_records(fasta_text)
    selected = _select_best_record(records, request.target_name)
    if selected is None:
        raise _NotFoundError(f"No CDS records could be parsed for nuccore id '{ids[0]}'")

    header, sequence = selected
    cds_acc = _accession_from_header(header) or ids[0]

    return (
        _sequence_record(
            sequence_type="dna_cds",
            source_database="ncbi",
            accession=cds_acc,
            sequence=sequence,
            source_url=url,
            config=config,
            inferred=False,
        ),
        {"cds_accession": cds_acc},
        warnings,
    )


def _fetch_rna_transcript(
    request: SequenceFetchRequest,
    config: SequenceFetchConfig,
    session: requests.Session,
) -> Tuple[FetchedSequence, Dict[str, str], List[str]]:
    """Fetch transcript RNA sequence."""
    warnings: List[str] = []

    transcript_accession = request.transcript_id or request.refseq_accession
    if transcript_accession:
        header, sequence, url = _ncbi_fetch_fasta(
            db="nuccore",
            identifier=transcript_accession,
            config=config,
            session=session,
            rettype="fasta",
        )
        accession = _accession_from_header(header) or transcript_accession
        return (
            _sequence_record(
                sequence_type="rna_transcript",
                source_database="ncbi",
                accession=accession,
                sequence=_dna_to_rna(sequence),
                source_url=url,
                config=config,
                inferred=True,
            ),
            {"transcript_id": accession},
            warnings,
        )

    term = (
        f"({_ncbi_term_for_request(request)}) "
        "AND (biomol_mrna[PROP] OR biomol_rna[PROP])"
    )
    ids = _ncbi_esearch(db="nuccore", term=term, config=config, session=session)
    if not ids:
        raise _NotFoundError(
            f"No RNA transcript candidates found for '{request.target_name}' in '{request.organism}'"
        )

    if len(ids) > 1:
        warnings.append(
            f"Multiple transcript candidates found ({len(ids)}); using top hit {ids[0]}"
        )

    header, sequence, url = _ncbi_fetch_fasta(
        db="nuccore",
        identifier=ids[0],
        config=config,
        session=session,
        rettype="fasta",
    )
    accession = _accession_from_header(header) or ids[0]

    return (
        _sequence_record(
            sequence_type="rna_transcript",
            source_database="ncbi",
            accession=accession,
            sequence=_dna_to_rna(sequence),
            source_url=url,
            config=config,
            inferred=True,
        ),
        {"transcript_id": accession},
        warnings,
    )


def _fetch_rna_premrna(
    request: SequenceFetchRequest,
    config: SequenceFetchConfig,
    session: requests.Session,
) -> Tuple[FetchedSequence, Dict[str, str], List[str]]:
    """Fetch or infer pre-mRNA sequence from genomic sequence."""
    genomic_record, ids, warnings = _fetch_dna_genomic(request, config, session)

    premrna = genomic_record.sequence

    warnings.append(
        "pre-mRNA sequence is inferred from genomic DNA and includes introns where present."
    )

    return (
        _sequence_record(
            sequence_type="rna_premrna",
            source_database=genomic_record.source_database,
            accession=genomic_record.accession,
            sequence=_dna_to_rna(premrna),
            source_url=genomic_record.source_url,
            config=config,
            inferred=True,
        ),
        ids,
        warnings,
    )


def _fetch_structure(
    request: SequenceFetchRequest,
    config: SequenceFetchConfig,
    session: requests.Session,
    resolved_ids: Optional[Dict[str, str]] = None,
) -> Tuple[FetchedStructure, Dict[str, str], List[str]]:
    """Fetch structure metadata from PDB."""
    warnings: List[str] = []

    pdb_id = request.pdb_id
    uniprot_id = request.uniprot_id
    entry: Optional[Dict[str, Any]] = None

    if not pdb_id and not uniprot_id and resolved_ids:
        uniprot_id = resolved_ids.get("uniprot_id")
        if uniprot_id:
            warnings.append(
                f"Using resolved UniProt ID '{uniprot_id}' from protein fetch for structure lookup"
            )

    if not pdb_id and not uniprot_id:
        entry = _search_uniprot_entry(
            target_name=request.target_name,
            organism=request.organism,
            prefer_pdb_crossref=True,
            config=config,
            session=session,
        )
        if entry is not None:
            uniprot_id = entry.get("primaryAccession")
            if uniprot_id:
                warnings.append(
                    f"Resolved UniProt ID '{uniprot_id}' from name+organism for structure lookup"
                )

    if not pdb_id and uniprot_id:
        if entry is None:
            entry = _fetch_uniprot_entry(uniprot_id, config, session)
        pdb_ids = _extract_uniprot_pdb_crossrefs(entry)
        if pdb_ids:
            pdb_id = pdb_ids[0]
            warnings.append(
                f"Using first UniProt-linked PDB ID '{pdb_id}' from cross references"
            )
        else:
            raise _NotFoundError(
                f"UniProt ID '{uniprot_id}' has no linked PDB cross references."
            )

    if not pdb_id:
        raise _NotFoundError(
            "No PDB ID provided or resolved. Provide pdb_id/uniprot_id, or ensure name+organism maps to a UniProt entry with PDB cross references."
        )

    pdb_id = pdb_id.upper()
    data = _request_json(
        session=session,
        method="GET",
        url=f"{_PDB_ENTRY_BASE}/{pdb_id}",
        config=config,
        source_label="pdb-entry",
    )

    struct = data.get("struct", {})
    entry_info = data.get("rcsb_entry_info", {})
    exptl = data.get("exptl", [])

    title = struct.get("title")
    method = exptl[0].get("method") if exptl else None

    resolution = None
    for key in (
        "resolution_combined",
        "d_resolution_high",
        "em_resolution",
    ):
        value = entry_info.get(key)
        if isinstance(value, list) and value:
            resolution = float(value[0])
            break
        if isinstance(value, (float, int)):
            resolution = float(value)
            break

    return (
        FetchedStructure(
            pdb_id=pdb_id,
            title=title,
            method=method,
            resolution=resolution,
            source_url=f"https://www.rcsb.org/structure/{pdb_id}",
        ),
        {"pdb_id": pdb_id, **({"uniprot_id": uniprot_id} if uniprot_id else {})},
        warnings,
    )


def _fetch_uniprot_entry(
    uniprot_id: str,
    config: SequenceFetchConfig,
    session: requests.Session,
) -> Dict[str, Any]:
    """Fetch a UniProtKB entry by accession."""
    return _request_json(
        session=session,
        method="GET",
        url=f"{_UNIPROT_BASE}/uniprotkb/{uniprot_id}.json",
        config=config,
        source_label="uniprot-entry",
    )


def _search_uniprot_entry(
    target_name: str,
    organism: str,
    prefer_pdb_crossref: bool,
    config: SequenceFetchConfig,
    session: requests.Session,
) -> Optional[Dict[str, Any]]:
    """Search UniProt by name and organism and return best ranked entry."""
    all_results: List[Dict[str, Any]] = []
    seen_accessions: set[str] = set()

    queries = [
        f'gene_exact:{target_name} AND organism_name:"{organism}"',
        f'(gene_exact:{target_name} OR protein_name:{target_name}) AND organism_name:"{organism}"',
    ]

    for query in queries:
        params = {
            "query": query,
            "format": "json",
            "size": config.max_candidates_per_source,
        }
        data = _request_json(
            session=session,
            method="GET",
            url=f"{_UNIPROT_BASE}/uniprotkb/search",
            config=config,
            source_label="uniprot-search",
            params=params,
        )

        for entry in data.get("results", []):
            accession = entry.get("primaryAccession")
            if accession and accession not in seen_accessions:
                seen_accessions.add(accession)
                all_results.append(entry)

        if all_results:
            break

    if not all_results:
        return None

    return max(
        all_results,
        key=lambda entry: _uniprot_entry_priority(entry, target_name, prefer_pdb_crossref),
    )


def _uniprot_entry_priority(
    entry: Dict[str, Any],
    target_name: str,
    prefer_pdb_crossref: bool,
) -> Tuple[int, int, int, int]:
    """Rank UniProt candidates for deterministic, biologically sensible selection."""
    target = target_name.strip().lower()
    gene_names = _extract_uniprot_gene_names(entry)
    has_exact_gene = int(target in gene_names)
    has_pdb = int(bool(_extract_uniprot_pdb_crossrefs(entry)))
    reviewed = int("reviewed" in str(entry.get("entryType", "")).lower())
    accession = str(entry.get("primaryAccession", ""))
    return (
        has_exact_gene,
        has_pdb if prefer_pdb_crossref else 0,
        reviewed,
        len(accession),
    )


def _extract_uniprot_gene_names(entry: Dict[str, Any]) -> set[str]:
    """Extract normalized gene symbol candidates from a UniProt entry."""
    names: set[str] = set()
    genes = entry.get("genes", [])
    if not isinstance(genes, list):
        return names

    for gene_obj in genes:
        if not isinstance(gene_obj, dict):
            continue
        primary = gene_obj.get("geneName", {})
        if isinstance(primary, dict):
            value = primary.get("value")
            if isinstance(value, str) and value.strip():
                names.add(value.strip().lower())

    return names


def _extract_uniprot_pdb_crossrefs(entry: Dict[str, Any]) -> List[str]:
    """Extract PDB cross references from UniProt entry JSON."""
    xrefs = entry.get("uniProtKBCrossReferences", [])
    pdb_ids = []
    for ref in xrefs:
        if ref.get("database") == "PDB" and ref.get("id"):
            pdb_ids.append(ref["id"])
    return pdb_ids


def _preferred_accession(request: SequenceFetchRequest) -> Optional[str]:
    """Return the best available accession override."""
    return (
        request.genbank_accession
        or request.refseq_accession
        or request.additional_ids.get("accession")
    )


def _request_json(
    session: requests.Session,
    method: str,
    url: str,
    config: SequenceFetchConfig,
    source_label: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """HTTP request helper returning JSON with retries."""
    response = _request(session, method, url, config, source_label, params=params)
    try:
        return response.json()
    except ValueError as exc:
        raise _UpstreamError(f"Invalid JSON response from {source_label}: {url}") from exc


def _request_text(
    session: requests.Session,
    method: str,
    url: str,
    config: SequenceFetchConfig,
    source_label: str,
    params: Optional[Dict[str, Any]] = None,
) -> str:
    """HTTP request helper returning text with retries."""
    response = _request(session, method, url, config, source_label, params=params)
    return response.text


def _request(
    session: requests.Session,
    method: str,
    url: str,
    config: SequenceFetchConfig,
    source_label: str,
    params: Optional[Dict[str, Any]] = None,
) -> requests.Response:
    """Execute HTTP request with exponential backoff and retry."""
    last_error: Optional[Exception] = None

    for attempt in range(config.http_retries + 1):
        try:
            response = session.request(
                method=method,
                url=url,
                params=params,
                timeout=config.request_timeout_seconds,
            )

            if response.status_code == 404:
                raise _NotFoundError(f"No record found at {source_label}: {response.url}")

            if response.status_code in _RETRY_STATUS_CODES:
                raise _UpstreamError(
                    f"Retryable upstream error {response.status_code} from {source_label}"
                )

            if response.status_code >= 400:
                raise _UpstreamError(
                    f"Upstream error {response.status_code} from {source_label}: {response.text[:200]}"
                )

            return response
        except (_NotFoundError, _AmbiguousMatchError, _TypeMismatchError):
            raise
        except Exception as exc:  # requests errors and custom upstream errors
            last_error = exc
            if attempt >= config.http_retries:
                break
            sleep_seconds = config.backoff_seconds * (2**attempt)
            jitter = random.uniform(0.0, max(config.backoff_seconds, 0.001))
            time.sleep(sleep_seconds + jitter)

    raise _UpstreamError(
        f"Upstream request failed for {source_label}: {url} ({last_error})"
    ) from last_error


def _ncbi_common_params(config: SequenceFetchConfig) -> Dict[str, Any]:
    """Build common NCBI eutils parameters."""
    params: Dict[str, Any] = {"tool": "bio_programming_tools_sequence_fetch"}
    if config.ncbi_email:
        params["email"] = config.ncbi_email
    if config.ncbi_api_key:
        params["api_key"] = config.ncbi_api_key
    return params


def _ncbi_esearch(
    db: str,
    term: str,
    config: SequenceFetchConfig,
    session: requests.Session,
) -> List[str]:
    """Run NCBI esearch and return ID list."""
    params = {
        "db": db,
        "term": term,
        "retmode": "json",
        "retmax": config.max_candidates_per_source,
    }
    params.update(_ncbi_common_params(config))

    data = _request_json(
        session=session,
        method="GET",
        url=f"{_NCBI_EUTILS_BASE}/esearch.fcgi",
        config=config,
        source_label=f"ncbi-esearch-{db}",
        params=params,
    )

    return data.get("esearchresult", {}).get("idlist", [])


def _ncbi_esummary(
    db: str,
    identifier: str,
    config: SequenceFetchConfig,
    session: requests.Session,
) -> Dict[str, Any]:
    """Run NCBI esummary and return result map."""
    params: Dict[str, Any] = {
        "db": db,
        "id": identifier,
        "retmode": "json",
    }
    params.update(_ncbi_common_params(config))

    data = _request_json(
        session=session,
        method="GET",
        url=f"{_NCBI_EUTILS_BASE}/esummary.fcgi",
        config=config,
        source_label=f"ncbi-esummary-{db}",
        params=params,
    )

    return data.get("result", {})


def _ncbi_fetch_fasta(
    db: str,
    identifier: str,
    config: SequenceFetchConfig,
    session: requests.Session,
    rettype: str = "fasta",
    seq_start: Optional[int] = None,
    seq_stop: Optional[int] = None,
    strand: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Fetch one FASTA record from NCBI efetch."""
    text, url = _ncbi_fetch_text(
        db=db,
        identifier=identifier,
        config=config,
        session=session,
        rettype=rettype,
        seq_start=seq_start,
        seq_stop=seq_stop,
        strand=strand,
    )
    records = _parse_fasta_records(text)
    if not records:
        raise _NotFoundError(f"No FASTA records for {db}:{identifier}")
    header, sequence = records[0]
    return header, sequence, url


def _ncbi_fetch_text(
    db: str,
    identifier: str,
    config: SequenceFetchConfig,
    session: requests.Session,
    rettype: str,
    seq_start: Optional[int] = None,
    seq_stop: Optional[int] = None,
    strand: Optional[str] = None,
) -> Tuple[str, str]:
    """Fetch raw text from NCBI efetch."""
    params: Dict[str, Any] = {
        "db": db,
        "id": identifier,
        "rettype": rettype,
        "retmode": "text",
    }
    if seq_start is not None:
        params["seq_start"] = seq_start
    if seq_stop is not None:
        params["seq_stop"] = seq_stop
    if strand is not None:
        params["strand"] = "2" if strand == "-" else "1"

    params.update(_ncbi_common_params(config))

    response = _request(
        session=session,
        method="GET",
        url=f"{_NCBI_EUTILS_BASE}/efetch.fcgi",
        config=config,
        source_label=f"ncbi-efetch-{db}",
        params=params,
    )

    return response.text, _sanitize_url(str(response.url))


def _sanitize_url(url: str) -> str:
    """Strip sensitive query parameters (api_key, email) from a URL."""
    from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

    parts = urlsplit(url)
    params = parse_qs(parts.query, keep_blank_values=True)
    for key in ("api_key", "email"):
        params.pop(key, None)
    clean_query = urlencode(params, doseq=True)
    return urlunsplit(parts._replace(query=clean_query))


def _ncbi_term(target_name: str, organism: str) -> str:
    """Compose a conservative Entrez term from target and organism."""
    escaped_target = target_name.replace('"', "")
    escaped_organism = organism.replace('"', "")
    return f'("{escaped_target}"[Gene] OR "{escaped_target}"[Title]) AND "{escaped_organism}"[Organism]'


def _ncbi_gene_term(target_name: str, organism: str) -> str:
    """Compose a gene-database-specific Entrez term."""
    escaped_target = target_name.replace('"', "")
    escaped_organism = organism.replace('"', "")
    return (
        f'("{escaped_target}"[Gene Name] OR "{escaped_target}"[Title]) '
        f'AND "{escaped_organism}"[Organism]'
    )


def _ncbi_term_for_request(request: SequenceFetchRequest) -> str:
    """Compose an Entrez term with optional gene_id preference."""
    base_term = _ncbi_term(request.target_name, request.organism)
    if request.gene_id:
        return f'({request.gene_id}[Gene ID] OR {base_term})'
    return base_term


def _parse_fasta_records(text: str) -> List[Tuple[str, str]]:
    """Parse FASTA text into (header, sequence) tuples."""
    records: List[Tuple[str, str]] = []
    header: Optional[str] = None
    lines: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(lines).strip()))
            header = line[1:].strip()
            lines = []
        else:
            lines.append(line)

    if header is not None:
        records.append((header, "".join(lines).strip()))

    return [(h, s) for h, s in records if s]


def _accession_from_header(header: str) -> Optional[str]:
    """Best-effort accession extraction from FASTA header."""
    tokens = header.split()
    if not tokens:
        return None

    first = tokens[0]
    if "|" in first:
        pieces = [p for p in first.split("|") if p]
        if len(pieces) >= 2:
            return pieces[1]
    return first


def _select_best_record(
    records: List[Tuple[str, str]],
    target_name: str,
) -> Optional[Tuple[str, str]]:
    """Pick the first matching record by target name, then first overall."""
    if not records:
        return None

    lowered_target = target_name.lower()
    for header, sequence in records:
        if lowered_target in header.lower():
            return header, sequence

    return records[0]


def _parse_coordinates(value: Optional[str]) -> Optional[Tuple[Optional[str], int, int, Optional[str]]]:
    """Parse accession:start-end:strand coordinates."""
    if not value:
        return None

    match = re.match(r"^(?P<acc>[^:]+):(?P<start>\d+)-(?P<end>\d+)(:(?P<strand>[+-]))?$", value)
    if not match:
        return None

    accession = match.group("acc")
    start = int(match.group("start"))
    end = int(match.group("end"))
    strand = match.group("strand")

    if start > end:
        start, end = end, start

    return accession, start, end, strand


_AMINO_ACID_CHARS = set("ACDEFGHIKLMNPQRSTVWY")
_NUCLEOTIDE_ONLY_CHARS = set("ATGCNU")


def _is_protein_sequence(seq: str) -> bool:
    """Return True if sequence looks like protein rather than DNA/RNA.

    A sequence is classified as protein if it contains at least one standard
    amino acid letter that is NOT also a nucleotide character. Pure
    ATGCNU-only sequences are classified as nucleotide.
    """
    upper = set(seq.upper()) - {"-", "*", "X", " "}
    if not upper:
        return False
    protein_only = upper & (_AMINO_ACID_CHARS - _NUCLEOTIDE_ONLY_CHARS)
    return bool(protein_only)


def _dna_to_rna(seq: str) -> str:
    """Convert DNA sequence to RNA sequence by T->U."""
    return seq.upper().replace("T", "U")


def _organism_likely_has_introns(organism: str) -> bool:
    """Heuristic to decide whether intron warning should be emitted."""
    text = organism.lower()
    prokaryote_markers = (
        "bacter",
        "archaea",
        "escherichia",
        "bacillus",
        "staphylococcus",
        "salmonella",
        "pseudomonas",
        "vibrio",
        "clostridium",
        "mycoplasma",
    )
    return not any(marker in text for marker in prokaryote_markers)


def _sequence_record(
    sequence_type: Literal[
        "protein",
        "dna_genomic",
        "dna_cds",
        "rna_transcript",
        "rna_premrna",
    ],
    source_database: Literal["ncbi", "uniprot", "pdb"],
    accession: Optional[str],
    sequence: str,
    source_url: Optional[str],
    config: SequenceFetchConfig,
    inferred: bool,
) -> FetchedSequence:
    """Build a normalized FetchedSequence record."""
    clean_sequence = re.sub(r"\s+", "", sequence).upper()
    checksum = hashlib.sha256(clean_sequence.encode("utf-8")).hexdigest() if config.include_sequence_checksums else None

    return FetchedSequence(
        sequence_type=sequence_type,
        source_database=source_database,
        accession=accession,
        sequence=clean_sequence,
        length=len(clean_sequence),
        checksum_sha256=checksum,
        source_url=source_url,
        inferred=inferred,
    )


def _failed_result(
    request: SequenceFetchRequest,
    request_index: int,
    error: str,
) -> SequenceFetchResult:
    """Create a normalized failed result object."""
    request_id = request.request_id or f"request_{request_index}"
    return SequenceFetchResult(
        request_id=request_id,
        target_name=request.target_name,
        organism=request.organism,
        requested_types=list(request.sequence_types),
        status="failed",
        errors=[error],
    )


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    """Deduplicate a string list while preserving first-seen order."""
    deduped: List[str] = []
    seen = set()
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped
