"""proto_tools/tools/database_retrieval/sequence_fetch/sequence_fetch.py

Thin orchestrator that chains database-specific tools (ncbi, uniprot,
pdb) with molecule-type routing and cross-fetcher ID resolution."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import requests
from Bio.Seq import transcribe
from pydantic import BaseModel, Field, computed_field, field_validator

# Database tool imports (orchestrator calls these directly)
from proto_tools.tools.database_retrieval.ncbi.shared_data_models import (
    NCBIFastaRecord,
    NCBIFetchConfig,
    _accession_from_header,
    _ncbi_efetch,
    _ncbi_esearch,
    _ncbi_esummary,
    _parse_fasta_records,
)
from proto_tools.tools.database_retrieval.pdb.shared_data_models import (
    PdbFetchConfig,
    _fetch_pdb_entry,
    _fetch_pdb_fasta,
    _is_protein_sequence,
)
from proto_tools.tools.database_retrieval.uniprot.uniprot_fetch import (
    _UNIPROT_BASE,
    UniProtFetchConfig,
    _extract_pdb_crossrefs,
)
from proto_tools.tools.database_retrieval.uniprot.uniprot_fetch import (
    _fetch_entry as _fetch_uniprot_entry,
)
from proto_tools.tools.database_retrieval.uniprot.uniprot_fetch import (
    _search_entry as _search_uniprot_entry,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    build_http_session,
)

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
# Data Models
# ============================================================================


class SequenceFetchRequest(BaseModel):
    """Single fetch request.

    Attributes:
        request_id (str | None): Optional caller-provided request identifier.
        target_name (str): Gene, protein, or RNA name to resolve.
        organism (str): Organism name used for disambiguation.
        sequence_types (list[Literal['protein', 'dna_genomic', 'dna_cds', 'rna_transcript', 'rna_premrna', 'structure']]): Requested outputs: protein, dna, rna, or structure.
        uniprot_id (str | None): UniProt accession override.
        genbank_accession (str | None): GenBank accession override.
        refseq_accession (str | None): RefSeq accession override.
        pdb_id (str | None): PDB accession override.
        gene_id (str | None): NCBI Gene ID override.
        protein_id (str | None): NCBI protein accession override.
        transcript_id (str | None): Transcript accession override.
        genomic_coordinates (str | None): Genomic interval like NC_000913.3:1-100:+.
        additional_ids (dict[str, str]): Extra IDs used for custom routing.
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
        requests (list[SequenceFetchRequest]): One or more retrieval requests.
    """

    requests: List[SequenceFetchRequest] = InputField(description="One or more retrieval requests")

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
        sequence_type (Literal['protein', 'dna_genomic', 'dna_cds', 'rna_transcript', 'rna_premrna']): Requested type of this sequence.
        source_database (Literal['ncbi', 'uniprot', 'pdb']): Upstream source database label.
        accession (str | None): Source accession identifier.
        sequence (str): Retrieved sequence string.
        length (int): Sequence length.
        checksum_sha256 (str | None): SHA256 checksum of sequence.
        source_url (str | None): Source URL for provenance.
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
        source_database (Literal['pdb']): Upstream source database label.
        title (str | None): Structure title.
        method (str | None): Experimental method.
        resolution (float | None): Resolution in angstroms.
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
        requested_types (list[str]): Requested output molecule types.
        status (Literal['success', 'warning', 'failed']): One of success, warning, or failed.
        fetched_sequences (list[FetchedSequence]): Retrieved sequence records.
        fetched_structures (list[FetchedStructure]): Retrieved structure records.
        resolved_ids (dict[str, str]): IDs resolved or used during retrieval.
        warnings (list[str]): Non-fatal warnings.
        errors (list[str]): Fatal or partial failure messages.
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


class SequenceFetchOutput(BaseToolOutput):
    """Output from sequence retrieval.

    Attributes:
        results (list[SequenceFetchResult]): Per-request retrieval outcomes.
        num_requests: Number of requests in this run.
        num_success: Number of successful request results.
        num_warning: Number of warning request results.
        num_completed: Number of successful or warning request results.
        num_failed: Number of failed request results.
    """

    results: List[SequenceFetchResult] = Field(
        default_factory=list,
        description="Per-request retrieval outcomes",
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
        request_timeout_seconds (int): Timeout per HTTP request.
        http_retries (int): Retries per upstream HTTP call.
        backoff_seconds (float): Seconds to wait between retries (doubles
            after each attempt).
        max_candidates_per_source (int): Maximum database candidates to
            evaluate per name-based search.
        strict_type_checks (bool): Reject requests where molecule type
            conflicts with target (e.g. protein for an ncRNA gene).
        fail_on_type_mismatch (bool): Treat molecule type mismatches as errors
            instead of warnings.
        include_sequence_checksums (bool): Include SHA256 checksums in outputs.
        ncbi_api_key (str | None): Optional API key for NCBI Entrez.
        ncbi_email (str | None): Optional contact email for NCBI requests.
        user_agent (str): Identifier string sent to database APIs with each
            request.
    """

    request_timeout_seconds: int = ConfigField(
        title="Request Timeout",
        default=15,
        ge=1,
        description="HTTP timeout in seconds",
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
        description="Seconds to wait between retries (doubles after each attempt)",
        advanced=True,
    )
    max_candidates_per_source: int = ConfigField(
        title="Max Candidates",
        default=5,
        ge=1,
        le=25,
        description="Maximum database candidates to evaluate per name-based search",
        advanced=True,
    )
    strict_type_checks: bool = ConfigField(
        title="Strict Type Checks",
        default=True,
        description="Reject requests where molecule type conflicts with target (e.g. protein for an ncRNA gene)",
    )
    fail_on_type_mismatch: bool = ConfigField(
        title="Fail On Mismatch",
        default=True,
        description="Treat molecule type mismatches as errors instead of warnings",
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
        default="proto-tools/sequence-fetch-v1",
        description="Identifier string sent to database APIs with each request",
        advanced=True,
    )


# ============================================================================
# Config Adapters: map SequenceFetchConfig to per-database configs
# ============================================================================


def _ncbi_config(config: SequenceFetchConfig) -> NCBIFetchConfig:
    """Create an NCBIFetchConfig from the orchestrator config."""
    return NCBIFetchConfig(
        request_timeout_seconds=config.request_timeout_seconds,
        http_retries=config.http_retries,
        backoff_seconds=config.backoff_seconds,
        ncbi_api_key=config.ncbi_api_key,
        ncbi_email=config.ncbi_email,
        user_agent=config.user_agent,
    )


def _uniprot_config(config: SequenceFetchConfig) -> UniProtFetchConfig:
    """Create a UniProtFetchConfig from the orchestrator config."""
    return UniProtFetchConfig(
        request_timeout_seconds=config.request_timeout_seconds,
        http_retries=config.http_retries,
        backoff_seconds=config.backoff_seconds,
        user_agent=config.user_agent,
    )


def _pdb_config(config: SequenceFetchConfig) -> PdbFetchConfig:
    """Create a PdbFetchConfig from the orchestrator config."""
    return PdbFetchConfig(
        request_timeout_seconds=config.request_timeout_seconds,
        http_retries=config.http_retries,
        backoff_seconds=config.backoff_seconds,
        user_agent=config.user_agent,
    )


# ============================================================================
# NCBI convenience helper
# ============================================================================


def _ncbi_fetch_first_fasta(
    db: str,
    identifier: str,
    config: NCBIFetchConfig,
    session: requests.Session,
    rettype: str = "fasta",
    seq_start: Optional[int] = None,
    seq_stop: Optional[int] = None,
    strand: Optional[str] = None,
) -> Optional[Tuple[str, str, str]]:
    """Fetch the first FASTA record from NCBI efetch, or None if not found."""
    result = _ncbi_efetch(
        db=db,
        identifier=identifier,
        rettype=rettype,
        config=config,
        session=session,
        seq_start=seq_start,
        seq_stop=seq_stop,
        strand=strand,
    )
    if result is None:
        return None
    text, url = result
    records = _parse_fasta_records(text)
    if not records:
        return None
    return records[0].header, records[0].sequence, url


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input():
    """Minimal valid input for testing and examples."""
    return SequenceFetchInput(requests=[
        SequenceFetchRequest(target_name="TP53", organism="Homo sapiens", sequence_types=["protein"])
    ])


@tool(
    key="sequence-fetch",
    label="Multi-source Sequence Fetch",
    category="database_retrieval",
    input_class=SequenceFetchInput,
    config_class=SequenceFetchConfig,
    output_class=SequenceFetchOutput,
    description="Fetch DNA, RNA, protein, and structure records from NCBI, UniProt, and PDB",
    uses_gpu=False,
    example_input=example_input,
    iterable_input_field="requests",
    iterable_output_field="results",
    cacheable=True,
)
def run_sequence_fetch(
    inputs: SequenceFetchInput,
    config: SequenceFetchConfig | None = None,
    instance=None,
) -> SequenceFetchOutput:
    """Fetch DNA, RNA, protein, and structure records from NCBI, UniProt, and PDB.

    This tool resolves IDs and names across NCBI Entrez, UniProt, and PDB for
    sequence and structure retrieval.

    Args:
        inputs (SequenceFetchInput): One or more sequence retrieval requests.
        config (SequenceFetchConfig | None): Timeout and validation settings.

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

    ncfg = _ncbi_config(config)
    session = build_http_session(
        http_retries=ncfg.http_retries,
        backoff_seconds=ncfg.backoff_seconds,
        user_agent=config.user_agent,
        allowed_methods=["GET", "POST"],
    )

    try:
        results = [
            _process_single_request(
                request=request,
                request_index=i,
                config=config,
                session=session,
            )
            for i, request in enumerate(inputs.requests)
        ]
    finally:
        session.close()

    return SequenceFetchOutput(results=results)


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

    type_error = _validate_request_type_compatibility(request, config)
    if type_error:
        if config.fail_on_type_mismatch:
            return _failed_result(
                request=request,
                request_index=request_index,
                error=f"TYPE_MISMATCH: {type_error}",
            )
        warnings.append(f"TYPE_MISMATCH: {type_error}")

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

    sequence_fetchers = {
        "protein": _fetch_protein,
        "dna_genomic": _fetch_dna_genomic,
        "dna_cds": _fetch_dna_cds,
        "rna_transcript": _fetch_rna_transcript,
        "rna_premrna": _fetch_rna_premrna,
    }

    for sequence_type in request.sequence_types:
        try:
            if sequence_type == "structure":
                result = _fetch_structure(
                    request, config, session, resolved_ids
                )
            else:
                fetcher = sequence_fetchers[sequence_type]
                result = fetcher(request, config, session)

            if isinstance(result, str):
                errors.append(f"NOT_FOUND[{sequence_type}]: {result}")
                continue

            fetched, ids, local_warnings = result
            if sequence_type == "structure":
                structures.append(fetched)
            else:
                sequences.append(fetched)
            resolved_ids.update(ids)
            warnings.extend(local_warnings)

        except requests.exceptions.HTTPError as exc:
            errors.append(f"HTTP_ERROR[{sequence_type}]: {exc}")
        except Exception as exc:  # pragma: no cover
            errors.append(f"UNEXPECTED_ERROR[{sequence_type}]: {exc}")

    status: Literal["success", "warning", "failed"]
    if errors and not (sequences or structures):
        status = "failed"
    elif errors or warnings:
        status = "warning"
    else:
        status = "success"

    return SequenceFetchResult(
        request_id=request_id,
        target_name=request.target_name,
        organism=request.organism,
        requested_types=list(request.sequence_types),
        status=status,
        fetched_sequences=sequences,
        fetched_structures=structures,
        resolved_ids=resolved_ids,
        warnings=list(dict.fromkeys(warnings)),
        errors=list(dict.fromkeys(errors)),
    )


def _validate_request_type_compatibility(
    request: SequenceFetchRequest,
    config: SequenceFetchConfig,
) -> Optional[str]:
    """Validate obvious ncRNA/protein mismatches. Returns error message or None."""
    if not config.strict_type_checks:
        return None

    if "protein" not in request.sequence_types:
        return None

    name = request.target_name.lower()
    if any(re.search(pattern, name) for pattern in _NON_CODING_PATTERNS):
        return f"Target '{request.target_name}' appears non-coding but protein was requested"

    refseq_like = (request.refseq_accession or request.transcript_id or "").upper()
    if refseq_like.startswith(("NR_", "XR_")):
        return f"RefSeq transcript '{refseq_like}' is non-coding (NR_/XR_ prefix)"

    return None


# ============================================================================
# Molecule-type Fetchers: route to database tools
# ============================================================================


def _fetch_protein(
    request: SequenceFetchRequest,
    config: SequenceFetchConfig,
    session: requests.Session,
) -> Optional[Tuple[FetchedSequence, Dict[str, str], List[str]]]:
    """Fetch protein sequence using ID-priority resolution."""
    warnings: List[str] = []
    ncfg = _ncbi_config(config)
    ucfg = _uniprot_config(config)
    pcfg = _pdb_config(config)

    if request.uniprot_id:
        entry = _fetch_uniprot_entry(request.uniprot_id, ucfg, session)
        if entry is None:
            return f"UniProt ID '{request.uniprot_id}' not found"
        sequence = entry.get("sequence", {}).get("value")
        if not sequence:
            return f"No sequence found for UniProt ID '{request.uniprot_id}'"

        accession = entry.get("primaryAccession", request.uniprot_id)
        ids = {"uniprot_id": accession}

        pdb_xrefs = _extract_pdb_crossrefs(entry)
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
        result = _ncbi_fetch_first_fasta(
            db="protein",
            identifier=protein_accession,
            config=ncfg,
            session=session,
            rettype="fasta",
        )
        if result is None:
            return f"Protein accession '{protein_accession}' not found in NCBI"
        header, sequence, url = result
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
        fasta_records = _fetch_pdb_fasta(request.pdb_id.upper(), pcfg, session)
        if not fasta_records:
            return f"PDB ID '{request.pdb_id}' returned no FASTA records"
        protein_records = [
            (h, s) for h, s in fasta_records if _is_protein_sequence(s)
        ]
        if not protein_records:
            return (
                f"PDB ID '{request.pdb_id}' has no protein chains "
                f"(found {len(fasta_records)} non-protein chain(s))"
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
        max_candidates=config.max_candidates_per_source,
        config=ucfg,
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

    term = _ncbi_term_for_request(request)
    ids = _ncbi_esearch(db="protein", term=term, config=ncfg, session=session,
                        max_results=config.max_candidates_per_source)
    if not ids:
        return f"No protein found for '{request.target_name}' in '{request.organism}'"

    if len(ids) > 1:
        warnings.append(
            f"Multiple NCBI protein candidates found ({len(ids)}); using top hit {ids[0]}"
        )

    result = _ncbi_fetch_first_fasta(
        db="protein",
        identifier=ids[0],
        config=ncfg,
        session=session,
        rettype="fasta",
    )
    if result is None:
        return f"NCBI protein efetch returned no data for ID '{ids[0]}'"
    header, sequence, url = result
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
) -> Optional[Tuple[FetchedSequence, Dict[str, str], List[str]]]:
    """Fetch genomic DNA sequence."""
    warnings: List[str] = []
    ncfg = _ncbi_config(config)

    coords = _parse_coordinates(request.genomic_coordinates)
    accession_hint = request.genbank_accession or request.refseq_accession

    if coords is not None:
        coord_accession = coords[0] or accession_hint
        if coord_accession is None:
            return "Genomic coordinates provided but no accession to anchor them"

        result = _ncbi_fetch_first_fasta(
            db="nuccore",
            identifier=coord_accession,
            config=ncfg,
            session=session,
            rettype="fasta",
            seq_start=coords[1],
            seq_stop=coords[2],
            strand=coords[3],
        )
        if result is None:
            return f"Coordinate-based genomic fetch failed for '{coord_accession}'"
        header, sequence, url = result
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
        result = _ncbi_fetch_first_fasta(
            db="nuccore",
            identifier=direct_accession,
            config=ncfg,
            session=session,
            rettype="fasta",
        )
        if result is None:
            return f"Genomic accession '{direct_accession}' not found in NCBI nuccore"
        header, sequence, url = result
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
    ids = _ncbi_esearch(db="nuccore", term=term, config=ncfg, session=session,
                        max_results=config.max_candidates_per_source)
    if not ids:
        ids = _ncbi_esearch(
            db="nuccore",
            term=_ncbi_term_for_request(request),
            config=ncfg,
            session=session,
            max_results=config.max_candidates_per_source,
        )
    if not ids:
        return f"No genomic DNA found for '{request.target_name}' in '{request.organism}'"

    attempted_not_found: List[str] = []
    selected_id: Optional[str] = None
    header = ""
    sequence = ""
    url = ""

    for candidate_id in ids:
        result = _ncbi_fetch_first_fasta(
            db="nuccore",
            identifier=candidate_id,
            config=ncfg,
            session=session,
            rettype="fasta",
        )
        if result is not None:
            header, sequence, url = result
            selected_id = candidate_id
            break
        attempted_not_found.append(candidate_id)

    if selected_id is None:
        gene_result = _fetch_dna_genomic_from_gene_locus(
            request=request,
            config=config,
            session=session,
        )
        if isinstance(gene_result, str):
            return gene_result
        gene_record, gene_ids, gene_warnings = gene_result
        warnings.append(
            "Nuccore genomic candidates lacked FASTA; used gene-locus genomic fallback"
        )
        warnings.extend(gene_warnings)
        return gene_record, gene_ids, warnings

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
) -> Optional[Tuple[FetchedSequence, Dict[str, str], List[str]]]:
    """Fetch genomic DNA by resolving gene locus coordinates from NCBI gene."""
    warnings: List[str] = []
    ncfg = _ncbi_config(config)

    if request.gene_id:
        gene_ids = [request.gene_id]
    else:
        gene_ids = _ncbi_esearch(
            db="gene",
            term=_ncbi_gene_term(request.target_name, request.organism),
            config=ncfg,
            session=session,
            max_results=config.max_candidates_per_source,
        )
    if not gene_ids:
        return f"No gene IDs found for '{request.target_name}' in '{request.organism}'"

    selected_gene_id = gene_ids[0]
    if len(gene_ids) > 1:
        warnings.append(
            f"Multiple gene candidates found ({len(gene_ids)}); using top hit {selected_gene_id}"
        )

    summary_result = _ncbi_esummary(
        db="gene",
        identifier=selected_gene_id,
        config=ncfg,
        session=session,
    )
    if summary_result is None:
        return f"Gene summary not found for gene ID '{selected_gene_id}'"
    summary, _url = summary_result

    gene_payload = summary.get(selected_gene_id, {})
    genomic_info = gene_payload.get("genomicinfo", [])
    if not genomic_info:
        return f"Gene record '{selected_gene_id}' lacks genomic coordinates in esummary"

    selected_region = None
    for region in genomic_info:
        if region.get("chraccver"):
            selected_region = region
            break
    if selected_region is None:
        return f"Gene record '{selected_gene_id}' has genomic info but no chromosome accession"

    chr_accession = selected_region["chraccver"]
    chr_start = int(selected_region["chrstart"])
    chr_stop = int(selected_region["chrstop"])
    seq_start = min(chr_start, chr_stop) + 1
    seq_stop = max(chr_start, chr_stop) + 1
    strand = "-" if chr_start > chr_stop else "+"

    result = _ncbi_fetch_first_fasta(
        db="nuccore",
        identifier=chr_accession,
        config=ncfg,
        session=session,
        rettype="fasta",
        seq_start=seq_start,
        seq_stop=seq_stop,
        strand=strand,
    )
    if result is None:
        return f"Genomic fetch failed for locus '{chr_accession}:{seq_start}-{seq_stop}'"
    header, sequence, url = result

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
) -> Optional[Tuple[FetchedSequence, Dict[str, str], List[str]]]:
    """Fetch coding DNA sequence (CDS)."""
    warnings: List[str] = []
    ncfg = _ncbi_config(config)

    accession = _preferred_accession(request)
    if accession:
        efetch_result = _ncbi_efetch(
            db="nuccore",
            identifier=accession,
            config=ncfg,
            session=session,
            rettype="fasta_cds_na",
        )
        if efetch_result is None:
            return f"CDS efetch failed for accession '{accession}'"
        text, url = efetch_result
        records = _parse_fasta_records(text)
        selected = _select_best_record(records, request.target_name)
        if selected is None:
            return f"No CDS sequence found for accession '{accession}'"

        cds_acc = selected.accession or accession
        return (
            _sequence_record(
                sequence_type="dna_cds",
                source_database="ncbi",
                accession=cds_acc,
                sequence=selected.sequence,
                source_url=url,
                config=config,
                inferred=False,
            ),
            {"cds_accession": cds_acc},
            warnings,
        )

    # Fallback via protein accession
    if request.protein_id:
        efetch_result = _ncbi_efetch(
            db="protein",
            identifier=request.protein_id,
            config=ncfg,
            session=session,
            rettype="fasta_cds_na",
        )
        if efetch_result is not None:
            text, url = efetch_result
            records = _parse_fasta_records(text)
            selected = _select_best_record(records, request.target_name)
            if selected is not None:
                cds_acc = selected.accession or request.protein_id
                warnings.append("CDS inferred from protein record using NCBI fasta_cds_na")
                return (
                    _sequence_record(
                        sequence_type="dna_cds",
                        source_database="ncbi",
                        accession=cds_acc,
                        sequence=selected.sequence,
                        source_url=url,
                        config=config,
                        inferred=True,
                    ),
                    {"cds_accession": cds_acc},
                    warnings,
                )

    # Name-based fallback through nuccore
    term = (
        f"({_ncbi_term_for_request(request)}) "
        "AND (mRNA[Title] OR CDS[Title] OR coding[Title])"
    )
    ids = _ncbi_esearch(db="nuccore", term=term, config=ncfg, session=session,
                        max_results=config.max_candidates_per_source)
    if not ids:
        return f"No CDS found for '{request.target_name}' in '{request.organism}'"

    if len(ids) > 1:
        warnings.append(f"Multiple CDS candidates found ({len(ids)}); using top hit {ids[0]}")

    efetch_result = _ncbi_efetch(
        db="nuccore",
        identifier=ids[0],
        config=ncfg,
        session=session,
        rettype="fasta_cds_na",
    )
    if efetch_result is None:
        return f"CDS efetch failed for NCBI ID '{ids[0]}'"
    text, url = efetch_result
    records = _parse_fasta_records(text)
    selected = _select_best_record(records, request.target_name)
    if selected is None:
        return f"No CDS sequence found in fasta_cds_na response for NCBI ID '{ids[0]}'"

    cds_acc = selected.accession or ids[0]

    return (
        _sequence_record(
            sequence_type="dna_cds",
            source_database="ncbi",
            accession=cds_acc,
            sequence=selected.sequence,
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
) -> Optional[Tuple[FetchedSequence, Dict[str, str], List[str]]]:
    """Fetch transcript RNA sequence."""
    warnings: List[str] = []
    ncfg = _ncbi_config(config)

    transcript_accession = request.transcript_id or request.refseq_accession
    if transcript_accession:
        result = _ncbi_fetch_first_fasta(
            db="nuccore",
            identifier=transcript_accession,
            config=ncfg,
            session=session,
            rettype="fasta",
        )
        if result is None:
            return f"Transcript '{transcript_accession}' not found in NCBI nuccore"
        header, sequence, url = result
        accession = _accession_from_header(header) or transcript_accession
        return (
            _sequence_record(
                sequence_type="rna_transcript",
                source_database="ncbi",
                accession=accession,
                sequence=transcribe(sequence.upper()),
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
    ids = _ncbi_esearch(db="nuccore", term=term, config=ncfg, session=session,
                        max_results=config.max_candidates_per_source)
    if not ids:
        return f"No transcript found for '{request.target_name}' in '{request.organism}'"

    if len(ids) > 1:
        warnings.append(
            f"Multiple transcript candidates found ({len(ids)}); using top hit {ids[0]}"
        )

    result = _ncbi_fetch_first_fasta(
        db="nuccore",
        identifier=ids[0],
        config=ncfg,
        session=session,
        rettype="fasta",
    )
    if result is None:
        return f"Transcript efetch failed for NCBI ID '{ids[0]}'"
    header, sequence, url = result
    accession = _accession_from_header(header) or ids[0]

    return (
        _sequence_record(
            sequence_type="rna_transcript",
            source_database="ncbi",
            accession=accession,
            sequence=transcribe(sequence.upper()),
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
) -> Optional[Tuple[FetchedSequence, Dict[str, str], List[str]]]:
    """Fetch or infer pre-mRNA sequence from genomic sequence."""
    genomic_result = _fetch_dna_genomic(request, config, session)
    if isinstance(genomic_result, str):
        return f"pre-mRNA inference failed: {genomic_result}"

    genomic_record, ids, warnings = genomic_result
    premrna = genomic_record.sequence

    warnings.append(
        "pre-mRNA sequence is inferred from genomic DNA and includes introns where present."
    )

    return (
        _sequence_record(
            sequence_type="rna_premrna",
            source_database=genomic_record.source_database,
            accession=genomic_record.accession,
            sequence=transcribe(premrna.upper()),
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
) -> Optional[Tuple[FetchedStructure, Dict[str, str], List[str]]]:
    """Fetch structure metadata from PDB."""
    warnings: List[str] = []
    ucfg = _uniprot_config(config)
    pcfg = _pdb_config(config)

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
            max_candidates=config.max_candidates_per_source,
            config=ucfg,
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
            entry = _fetch_uniprot_entry(uniprot_id, ucfg, session)
        if entry is None:
            return f"UniProt entry '{uniprot_id}' not found"
        pdb_ids = _extract_pdb_crossrefs(entry)
        if pdb_ids:
            pdb_id = pdb_ids[0]
            warnings.append(
                f"Using first UniProt-linked PDB ID '{pdb_id}' from cross references"
            )
        else:
            return f"UniProt ID '{uniprot_id}' has no linked PDB cross-references"

    if not pdb_id:
        return f"No PDB ID resolved for '{request.target_name}' in '{request.organism}'"

    pdb_id = pdb_id.upper()
    meta = _fetch_pdb_entry(pdb_id, pcfg, session)
    if meta is None:
        return f"PDB ID '{pdb_id}' not found"

    return (
        FetchedStructure(
            pdb_id=pdb_id,
            title=meta.get("title"),
            method=meta.get("method"),
            resolution=meta.get("resolution"),
            source_url=f"https://www.rcsb.org/structure/{pdb_id}",
        ),
        {"pdb_id": pdb_id, **({"uniprot_id": uniprot_id} if uniprot_id else {})},
        warnings,
    )


# ============================================================================
# Helpers
# ============================================================================


def _preferred_accession(request: SequenceFetchRequest) -> Optional[str]:
    """Return the best available accession override."""
    return (
        request.genbank_accession
        or request.refseq_accession
        or request.additional_ids.get("accession")
    )


def _ncbi_term_for_request(request: SequenceFetchRequest) -> str:
    """Compose an Entrez term with optional gene_id preference."""
    base_term = _ncbi_term(request.target_name, request.organism)
    if request.gene_id:
        return f'({request.gene_id}[Gene ID] OR {base_term})'
    return base_term


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


def _ncbi_term(target_name: str, organism: str) -> str:
    """Build an Entrez search term for protein/nucleotide databases."""
    escaped_target = target_name.replace('"', "")
    escaped_organism = organism.replace('"', "")
    return (
        f'("{escaped_target}"[Gene] OR "{escaped_target}"[Title]) '
        f'AND "{escaped_organism}"[Organism]'
    )


def _ncbi_gene_term(target_name: str, organism: str) -> str:
    """Build an Entrez search term for the gene database."""
    escaped_target = target_name.replace('"', "")
    escaped_organism = organism.replace('"', "")
    return (
        f'("{escaped_target}"[Gene Name] OR "{escaped_target}"[Title]) '
        f'AND "{escaped_organism}"[Organism]'
    )


def _select_best_record(
    records: List[NCBIFastaRecord],
    target_name: str,
) -> Optional[NCBIFastaRecord]:
    """Select the FASTA record whose header best matches target_name."""
    if not records:
        return None
    lowered = target_name.lower()
    for record in records:
        if lowered in record.header.lower():
            return record
    return records[0]
