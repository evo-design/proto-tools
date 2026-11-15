"""proto_tools/tools/database_retrieval/sequence_fetch/sequence_fetch.py.

Thin orchestrator that chains database-specific tools (ncbi, uniprot,
pdb) with molecule-type routing and cross-fetcher ID resolution.
"""

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal, get_args

import requests
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

# Database tool imports (orchestrator calls these directly)
from proto_tools.tools.database_retrieval.ncbi.shared_data_models import (
    _BACKOFF_SECONDS,
    _HTTP_RETRIES,
    _USER_AGENT,
    NCBIFastaRecord,
    NCBIFetchConfig,
    _accession_from_header,
    _ncbi_efetch,
    _ncbi_esearch,
    _ncbi_esummary,
    _parse_fasta_records,
)
from proto_tools.tools.database_retrieval.pdb.shared_data_models import (
    _fetch_pdb_entry,
    _fetch_pdb_fasta,
    _is_protein_sequence,
)
from proto_tools.tools.database_retrieval.uniprot.uniprot_fetch import (
    _UNIPROT_BASE,
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
    DeadlineExceededError,
    InputField,
    build_http_session,
)


class SequenceFetchError(ValueError):
    """Raised when a sequence fetch fails for a single molecule type.

    Used internally by fetcher functions to signal not-found or other
    retrieval failures.  The orchestrator catches these and converts
    them into ``errors`` entries on :class:`SequenceFetchResult`.
    """


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


AccessionKind = Literal["uniprot", "pdb", "protein", "nucleotide", "gene"]

# Checked in order; the first match names the kind. Two formats overlap, and the more common reading wins: bare
# digits are an NCBI Gene ID (write ``pdb:1234`` for an all-digit PDB ID), and UniProt's [OPQ] + 5 digits beats a
# one-letter GenBank nucleotide accession (write ``nucleotide:P12345`` for that).
_ACCESSION_PATTERNS: tuple[tuple[AccessionKind, re.Pattern[str]], ...] = (
    ("gene", re.compile(r"^\d+$")),
    ("nucleotide", re.compile(r"^(AC|NC|NG|NT|NW|NZ|NM|NR|XM|XR)_[A-Z]*\d+(\.\d+)?$")),
    ("protein", re.compile(r"^(AP|NP|YP|XP|WP)_\d+(\.\d+)?$")),
    ("pdb", re.compile(r"^\d[A-Z0-9]{3}$")),
    ("uniprot", re.compile(r"^([OPQ]\d[A-Z0-9]{3}\d|[A-NR-Z]\d([A-Z][A-Z0-9]{2}\d){1,2})(-\d+)?$")),
    ("protein", re.compile(r"^[A-Z]{3}\d{5,7}(\.\d+)?$")),
    ("nucleotide", re.compile(r"^([A-Z]\d{5}|[A-Z]{2}\d{6,8}|[A-Z]{4,6}\d{8,10})(\.\d+)?$")),
)

_ACCESSION_FORMATS = (
    "UniProt (P04637), PDB (2LYZ), RefSeq (NM_000546.6, NP_000537.3), GenBank (U49845, AAA12345) "
    "or an NCBI Gene ID (7157); prefix one with uniprot:, pdb:, protein:, nucleotide: or gene: to name its kind"
)


def parse_accession(value: str) -> tuple[AccessionKind, str]:
    """Work out which database an accession belongs to from its format.

    Args:
        value (str): An accession, optionally prefixed with its kind (e.g. ``"gene:7157"``).

    Returns:
        tuple[AccessionKind, str]: The kind and the accession, upper-cased when the kind was detected.

    Raises:
        ValueError: If the accession matches no known format.
    """
    text = value.strip()
    prefix, separator, rest = text.partition(":")
    if separator and prefix.lower() in get_args(AccessionKind):
        return prefix.lower(), rest.strip()  # type: ignore[return-value]
    upper = text.upper()
    for kind, pattern in _ACCESSION_PATTERNS:
        if pattern.match(upper):
            return kind, upper
    raise ValueError(f"Unrecognised accession {value!r}: expected {_ACCESSION_FORMATS}")


# ============================================================================
# Data Models
# ============================================================================


class SequenceFetchRequest(BaseModel):
    """Single fetch request.

    Attributes:
        request_id (str | None): Optional caller-provided request identifier.
        target_name (str | None): Gene, protein, or RNA name to resolve; needed with organism unless an identifier is given.
        organism (str | None): Organism name used for disambiguation; needed with target_name unless an identifier is given.
        sequence_types (list[Literal['protein', 'dna_genomic', 'dna_cds', 'rna_transcript', 'rna_premrna', 'structure']]): Requested outputs: protein, dna, rna, or structure.
        accessions (list[str]): Accessions to route on, at most one of each kind; see :func:`parse_accession`.
        genomic_coordinates (str | None): Genomic interval like NC_000913.3:1-100:+.
    """

    request_id: str | None = Field(default=None, title="Request ID", description="Optional request identifier")
    target_name: str | None = Field(
        default=None,
        min_length=1,
        title="Target Name",
        description="Gene, RNA, or protein name; needed with organism unless an identifier is given",
    )
    organism: str | None = Field(
        default=None,
        min_length=1,
        title="Organism",
        description="Organism for disambiguation; needed with target_name unless an identifier is given",
    )
    sequence_types: list[
        Literal[
            "protein",
            "dna_genomic",
            "dna_cds",
            "rna_transcript",
            "rna_premrna",
            "structure",
        ]
    ] = Field(title="Sequence Types", description="Requested output molecule types")
    accessions: list[str] = Field(
        default_factory=list,
        title="Accessions",
        description="UniProt, PDB, RefSeq or GenBank accessions or NCBI Gene IDs; each one's kind is detected",
    )
    genomic_coordinates: str | None = Field(
        default=None,
        title="Genomic Coordinates",
        description="Genomic coordinates as accession:start-end:strand",
    )

    @field_validator("sequence_types", mode="before")
    @classmethod
    def normalize_sequence_types(cls, value: Any) -> Any:
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

    @field_validator("accessions", mode="before")
    @classmethod
    def normalize_accessions(cls, value: Any) -> Any:
        """Accept a single accession or a list."""
        return [value] if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_lookup_params(self) -> "SequenceFetchRequest":
        """Require target_name and organism or an accession, and at most one accession of each kind."""
        by_kind: dict[str, str] = {}
        for accession in self.accessions:
            kind, parsed = parse_accession(accession)
            if kind in by_kind and by_kind[kind] != parsed:
                raise ValueError(f"Give at most one {kind} accession per request, got {by_kind[kind]} and {parsed}")
            by_kind[kind] = parsed
        if not (self.target_name and self.organism) and not by_kind and not self.genomic_coordinates:
            raise ValueError("Provide both target_name and organism, or an accession or genomic_coordinates")
        return self

    def _accession(self, kind: AccessionKind) -> str | None:
        """Return this request's accession of one kind, if it has one."""
        for accession in self.accessions:
            accession_kind, parsed = parse_accession(accession)
            if accession_kind == kind:
                return parsed
        return None

    @property
    def uniprot_id(self) -> str | None:
        """The UniProt accession, if given."""
        return self._accession("uniprot")

    @property
    def pdb_id(self) -> str | None:
        """The PDB ID, if given."""
        return self._accession("pdb")

    @property
    def protein_accession(self) -> str | None:
        """The NCBI protein (RefSeq or GenBank) accession, if given."""
        return self._accession("protein")

    @property
    def nucleotide_accession(self) -> str | None:
        """The NCBI nucleotide (RefSeq or GenBank) accession, if given."""
        return self._accession("nucleotide")

    @property
    def gene_id(self) -> str | None:
        """The NCBI Gene ID, if given."""
        return self._accession("gene")


class SequenceFetchInput(BaseToolInput):
    """Input for sequence retrieval.

    Attributes:
        requests (list[SequenceFetchRequest]): One or more retrieval requests.
    """

    requests: list[SequenceFetchRequest] = InputField(title="Requests", description="One or more retrieval requests")

    @field_validator("requests", mode="before")
    @classmethod
    def normalize_requests(cls, value: Any) -> Any:
        """Accept a single request object or a list."""
        if isinstance(value, dict):
            return [value]
        if isinstance(value, SequenceFetchRequest):
            return [value]
        return value

    @field_validator("requests")
    @classmethod
    def validate_requests(cls, value: list[SequenceFetchRequest]) -> list[SequenceFetchRequest]:
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
        source_url (str): Source URL for provenance.
        inferred (bool): True if sequence is inferred, not directly curated.
    """

    sequence_type: Literal[
        "protein",
        "dna_genomic",
        "dna_cds",
        "rna_transcript",
        "rna_premrna",
    ] = Field(title="Sequence Type", description="Requested type for this sequence")
    source_database: Literal["ncbi", "uniprot", "pdb"] = Field(
        title="Source Database", description="Source database for this sequence"
    )
    accession: str | None = Field(default=None, title="Accession", description="Source accession identifier")
    sequence: str = Field(title="Sequence", description="Retrieved sequence")
    length: int = Field(ge=0, title="Length", description="Sequence length")
    checksum_sha256: str | None = Field(default=None, title="SHA256 Checksum", description="SHA256 checksum")
    source_url: str = Field(title="Source URL", description="Source URL for provenance")
    inferred: bool = Field(
        default=False,
        title="Inferred",
        description="True if sequence was reconstructed (e.g. translated CDS) rather than fetched directly",
    )


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

    pdb_id: str = Field(title="PDB ID", description="PDB accession")
    source_database: Literal["pdb"] = Field(default="pdb", title="Source Database", description="Source database")
    title: str | None = Field(default=None, title="Title", description="Structure title")
    method: str | None = Field(
        default=None,
        title="Experimental Method",
        description="PDB experimental method (e.g. X-RAY DIFFRACTION, SOLUTION NMR, ELECTRON MICROSCOPY)",
    )
    resolution: float | None = Field(
        default=None, title="Resolution", description="Resolution in Å (None for NMR or fiber diffraction)"
    )
    source_url: str = Field(title="Source URL", description="Canonical structure URL")


class SequenceFetchResult(BaseModel):
    """Per-request fetch result.

    Attributes:
        request_id (str): Request identifier used in this result.
        target_name (str | None): Original target name, if given.
        organism (str | None): Original organism name, if given.
        requested_types (list[str]): Requested output molecule types.
        status (Literal['success', 'warning', 'failed']): One of success, warning, or failed.
        fetched_sequences (list[FetchedSequence]): Retrieved sequence records.
        fetched_structures (list[FetchedStructure]): Retrieved structure records.
        resolved_ids (dict[str, str]): IDs resolved or used during retrieval.
        warnings (list[str]): Non-fatal warnings.
        errors (list[str]): Fatal or partial failure messages.
    """

    request_id: str = Field(title="Request ID", description="Request identifier")
    target_name: str | None = Field(default=None, title="Target Name", description="Original target name, if given")
    organism: str | None = Field(default=None, title="Organism", description="Original organism name, if given")
    requested_types: list[str] = Field(title="Requested Types", description="Requested molecule types")
    status: Literal["success", "warning", "failed"] = Field(title="Status", description="Result status")
    fetched_sequences: list[FetchedSequence] = Field(
        default_factory=list,
        title="Fetched Sequences",
        description="Retrieved sequence records",
    )
    fetched_structures: list[FetchedStructure] = Field(
        default_factory=list,
        title="Fetched Structures",
        description="Retrieved structure records",
    )
    resolved_ids: dict[str, str] = Field(
        default_factory=dict,
        title="Resolved IDs",
        description="Resolved identifiers used in retrieval",
    )
    warnings: list[str] = Field(default_factory=list, title="Warnings", description="Non-fatal warnings")
    errors: list[str] = Field(default_factory=list, title="Errors", description="Fatal or partial failure messages")


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

    results: list[SequenceFetchResult] = Field(
        default_factory=list,
        title="Retrieval Results",
        description="Per-request retrieval outcomes",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_requests(self) -> int:
        """Total number of requests."""
        return len(self.results)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_success(self) -> int:
        """Number of successful results."""
        return sum(1 for r in self.results if r.status == "success")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_warning(self) -> int:
        """Number of warning results."""
        return sum(1 for r in self.results if r.status == "warning")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_completed(self) -> int:
        """Number of completed results including warnings."""
        return sum(1 for r in self.results if r.status in {"success", "warning"})

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_failed(self) -> int:
        """Number of failed results."""
        return sum(1 for r in self.results if r.status == "failed")

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json", "csv", "fasta"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")
        if file_format == "json":
            with path.open("w", encoding="utf-8") as handle:
                json.dump(self.model_dump(mode="json"), handle, indent=2)
            return

        if file_format == "csv":
            # One row per fetched sequence record (skips structure-only results
            # and request-level metadata; use JSON for those).
            rows: list[dict[str, Any]] = [
                {
                    "request_id": result.request_id,
                    "target_name": result.target_name,
                    "organism": result.organism,
                    "status": result.status,
                    **record.model_dump(),
                }
                for result in self.results
                for record in result.fetched_sequences
            ]
            with path.open("w", encoding="utf-8", newline="") as handle:
                if not rows:
                    return
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            return

        if file_format == "fasta":
            with path.open("w", encoding="utf-8") as handle:
                for result in self.results:
                    for record in result.fetched_sequences:
                        acc = record.accession or "unknown"
                        header = (
                            f">{result.request_id}|{result.target_name or ''}|"
                            f"{record.sequence_type}|{record.source_database}|{acc}"
                        )
                        handle.write(f"{header}\n{record.sequence}\n")
            return

        raise ValueError(f"Unsupported format: {file_format}")


class SequenceFetchConfig(BaseConfig):
    """Configuration for sequence retrieval.

    Attributes:
        max_candidates_per_source (int): Maximum database candidates to
            evaluate per name-based search.
        type_check_mode (Literal["off", "warn", "error"]): Controls how
            molecule-type mismatches are handled (e.g. requesting "protein"
            for a name that looks like an ncRNA gene).
            ``"off"`` skips validation entirely; ``"warn"`` records a warning
            but continues; ``"error"`` (default) fails the request.
        ncbi_api_key (str | None): Optional NCBI API key (lifts rate limit
            from 3 to 10 requests/second). Defaults to the ``NCBI_API_KEY``
            environment variable; an explicit value passed to the config
            overrides the env var.
        ncbi_email (str | None): Optional contact email. Defaults to the
            ``NCBI_EMAIL`` environment variable; an explicit value passed to
            the config overrides the env var.
    """

    max_candidates_per_source: int = ConfigField(
        title="Max Candidates",
        default=5,
        ge=1,
        le=25,
        description="Maximum database candidates to evaluate per name-based search",
    )
    type_check_mode: Literal["off", "warn", "error"] = ConfigField(
        title="Type Check Mode",
        default="error",
        description="Molecule-type mismatch handling: 'off' (skip), 'warn' (log + continue), 'error' (fail)",
    )
    # Mirrors NCBIFetchConfig.ncbi_api_key / ncbi_email — keep behavior and
    # env-var fallbacks in sync with ncbi/shared_data_models.py.
    ncbi_api_key: str | None = ConfigField(
        title="NCBI API Key",
        description="Optional NCBI API key (3 to 10 req/s).",
        credential="NCBI_API_KEY",
    )
    ncbi_email: str | None = ConfigField(
        title="NCBI Email",
        description="Optional contact email for NCBI.",
        credential="NCBI_EMAIL",
    )


# ============================================================================
# Config Adapters: map SequenceFetchConfig to per-database configs
# ============================================================================


def _ncbi_config(config: SequenceFetchConfig) -> NCBIFetchConfig:
    """Create an NCBIFetchConfig from the orchestrator config."""
    return NCBIFetchConfig(
        ncbi_api_key=config.ncbi_api_key,
        ncbi_email=config.ncbi_email,
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
    seq_start: int | None = None,
    seq_stop: int | None = None,
    strand: str | None = None,
) -> tuple[str, str, str] | None:
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


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return SequenceFetchInput(
        requests=[SequenceFetchRequest(target_name="TP53", organism="Homo sapiens", sequence_types=["protein"])]
    )


@tool(
    key="sequence-fetch",
    local_only="sequence-fetch does not use a gpu and does not need an environment, so it can run in process",
    label="Multi-source Sequence Fetch",
    category="database_retrieval",
    input_class=SequenceFetchInput,
    config_class=SequenceFetchConfig,
    output_class=SequenceFetchOutput,
    description="Fetch DNA, RNA, protein, and structure records from NCBI, UniProt, and PDB",
    uses_gpu=False,
    example_input=example_input,
    iterable_input_fields=["requests"],
    iterable_output_field="results",
    cacheable=True,
)
def run_sequence_fetch(
    inputs: SequenceFetchInput,
    config: SequenceFetchConfig,
    instance: Any = None,
) -> SequenceFetchOutput:
    """Fetch DNA, RNA, protein, and structure records from NCBI, UniProt, and PDB.

    This tool resolves IDs and names across NCBI Entrez, UniProt, and PDB for
    sequence and structure retrieval.

    Each request's ``accessions`` are sorted by kind from their format (see
    :func:`parse_accession`). Routing priority (per request):
        Protein: UniProt → NCBI protein → PDB → name search.
        Genomic: ``genomic_coordinates`` → NCBI nucleotide →
            name search → gene-locus fallback.
        Coding DNA: NCBI nucleotide → NCBI protein → name search.
        Transcript: NCBI nucleotide → name search.
        Structure: PDB → UniProt cross-reference → name search.

    Args:
        inputs (SequenceFetchInput): One or more sequence retrieval requests.
        config (SequenceFetchConfig): Candidate-limit, type-check, and NCBI credential settings.

        instance (Any): Optional ToolInstance for subprocess execution.

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

    session = build_http_session(
        http_retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
        user_agent=_USER_AGENT,
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
    warnings: list[str] = []
    errors: list[str] = []
    sequences: list[FetchedSequence] = []
    structures: list[FetchedStructure] = []
    resolved_ids: dict[str, str] = {}

    type_error = _validate_request_type_compatibility(request, config)
    if type_error:
        if config.type_check_mode == "error":
            return _failed_result(
                request=request,
                request_index=request_index,
                error=f"TYPE_MISMATCH: {type_error}",
            )
        if config.type_check_mode == "warn":
            warnings.append(f"TYPE_MISMATCH: {type_error}")

    if "protein" in request.sequence_types and request.genomic_coordinates:
        warnings.append("Protein from genomic coordinates is inferred and may be ambiguous due to introns.")

    if (
        "protein" in request.sequence_types
        and "dna_genomic" in request.sequence_types
        and _organism_likely_has_introns(request.organism or "")
    ):
        warnings.append("Genomic DNA includes introns in many organisms; DNA to protein mapping may be indirect.")

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
                fetched_struct, fetched_ids, local_warnings = _fetch_structure(request, config, session, resolved_ids)
                structures.append(fetched_struct)
            else:
                fetcher = sequence_fetchers[sequence_type]
                fetched_seq, fetched_ids, local_warnings = fetcher(request, config, session)
                sequences.append(fetched_seq)
            resolved_ids.update(fetched_ids)
            warnings.extend(local_warnings)

        except SequenceFetchError as exc:  # noqa: PERF203 -- each fetch type is independent
            errors.append(f"NOT_FOUND[{sequence_type}]: {exc}")
        except requests.exceptions.HTTPError as exc:
            errors.append(f"HTTP_ERROR[{sequence_type}]: {exc}")
        except DeadlineExceededError:
            # The call's timeout elapsed: fail the call rather than record every remaining item as missing.
            raise
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
) -> str | None:
    """Validate obvious ncRNA/protein mismatches. Returns error message or None."""
    if config.type_check_mode == "off":
        return None

    if "protein" not in request.sequence_types:
        return None

    name = (request.target_name or "").lower()
    if any(re.search(pattern, name) for pattern in _NON_CODING_PATTERNS):
        return f"Target '{request.target_name}' appears non-coding but protein was requested"

    refseq_like = request.nucleotide_accession or ""
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
) -> tuple[FetchedSequence, dict[str, str], list[str]]:
    """Fetch protein sequence using ID-priority resolution."""
    warnings: list[str] = []
    ncfg = _ncbi_config(config)

    if request.uniprot_id:
        entry = _fetch_uniprot_entry(request.uniprot_id, session)
        if entry is None:
            raise SequenceFetchError(f"UniProt ID '{request.uniprot_id}' not found")
        sequence = entry.get("sequence", {}).get("value")
        if not sequence:
            raise SequenceFetchError(f"No sequence found for UniProt ID '{request.uniprot_id}'")

        accession = entry.get("primaryAccession", request.uniprot_id)
        ids = {"uniprot_id": accession}

        pdb_xrefs = _extract_pdb_crossrefs(entry)
        if pdb_xrefs and not request.pdb_id and "structure" not in request.sequence_types:
            warnings.append(f"UniProt entry maps to PDB IDs {', '.join(pdb_xrefs[:3])}; use structure type to fetch")

        return (
            _sequence_record(
                sequence_type="protein",
                source_database="uniprot",
                accession=accession,
                sequence=sequence,
                source_url=f"{_UNIPROT_BASE}/uniprotkb/{accession}",
                inferred=False,
            ),
            ids,
            warnings,
        )

    protein_accession = request.protein_accession
    if protein_accession:
        result = _ncbi_fetch_first_fasta(
            db="protein",
            identifier=protein_accession,
            config=ncfg,
            session=session,
            rettype="fasta",
        )
        if result is None:
            raise SequenceFetchError(f"Protein accession '{protein_accession}' not found in NCBI")
        header, sequence, url = result
        accession = _accession_from_header(header) or protein_accession
        return (
            _sequence_record(
                sequence_type="protein",
                source_database="ncbi",
                accession=accession,
                sequence=sequence,
                source_url=url,
                inferred=False,
            ),
            {"protein_id": accession},
            warnings,
        )

    if request.pdb_id:
        fasta_records = _fetch_pdb_fasta(request.pdb_id.upper(), session)
        if not fasta_records:
            raise SequenceFetchError(f"PDB ID '{request.pdb_id}' returned no FASTA records")
        protein_records = [(h, s) for h, s in fasta_records if _is_protein_sequence(s)]
        if not protein_records:
            raise SequenceFetchError(
                f"PDB ID '{request.pdb_id}' has no protein chains (found {len(fasta_records)} non-protein chain(s))"
            )
        header, sequence = protein_records[0]
        accession = request.pdb_id.upper()
        if len(protein_records) > 1:
            warnings.append(
                f"Using first protein chain from PDB FASTA; {len(protein_records)} protein chains available."
            )
        return (
            _sequence_record(
                sequence_type="protein",
                source_database="pdb",
                accession=accession,
                sequence=sequence,
                source_url=f"https://www.rcsb.org/structure/{accession}",
                inferred=False,
            ),
            {"pdb_id": accession, "protein_id": _accession_from_header(header) or accession},
            warnings,
        )

    # Name-based fallback: UniProt first, then NCBI protein search.
    entry = None
    if request.target_name and request.organism:
        entry = _search_uniprot_entry(
            target_name=request.target_name,
            organism=request.organism,
            prefer_pdb_crossref="structure" in request.sequence_types,
            max_candidates=config.max_candidates_per_source,
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
                    inferred=False,
                ),
                {"uniprot_id": accession},
                [],
            )

    term = _ncbi_term_for_request(request)
    search_ids, _ = _ncbi_esearch(
        db="protein",
        term=term,
        max_results=config.max_candidates_per_source,
        config=ncfg,
        session=session,
    )
    if not search_ids:
        raise SequenceFetchError(f"No protein found for {_target_label(request)}")

    if len(search_ids) > 1:
        warnings.append(f"Multiple NCBI protein candidates found ({len(search_ids)}); using top hit {search_ids[0]}")

    result = _ncbi_fetch_first_fasta(
        db="protein",
        identifier=search_ids[0],
        config=ncfg,
        session=session,
        rettype="fasta",
    )
    if result is None:
        raise SequenceFetchError(f"NCBI protein efetch returned no data for ID '{search_ids[0]}'")
    header, sequence, url = result
    accession = _accession_from_header(header) or search_ids[0]

    return (
        _sequence_record(
            sequence_type="protein",
            source_database="ncbi",
            accession=accession,
            sequence=sequence,
            source_url=url,
            inferred=False,
        ),
        {"protein_id": accession},
        warnings,
    )


def _fetch_dna_genomic(
    request: SequenceFetchRequest,
    config: SequenceFetchConfig,
    session: requests.Session,
) -> tuple[FetchedSequence, dict[str, str], list[str]]:
    """Fetch genomic DNA sequence."""
    warnings: list[str] = []
    ncfg = _ncbi_config(config)

    coords = _parse_coordinates(request.genomic_coordinates)
    accession_hint = request.nucleotide_accession

    if coords is not None:
        coord_accession = coords[0] or accession_hint
        if coord_accession is None:
            raise SequenceFetchError("Genomic coordinates provided but no accession to anchor them")

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
            raise SequenceFetchError(f"Coordinate-based genomic fetch failed for '{coord_accession}'")
        header, sequence, url = result
        accession = _accession_from_header(header) or coord_accession
        return (
            _sequence_record(
                sequence_type="dna_genomic",
                source_database="ncbi",
                accession=accession,
                sequence=sequence,
                source_url=url,
                inferred=False,
            ),
            {"genbank_accession": accession},
            warnings,
        )

    direct_accession = request.nucleotide_accession
    if direct_accession:
        result = _ncbi_fetch_first_fasta(
            db="nuccore",
            identifier=direct_accession,
            config=ncfg,
            session=session,
            rettype="fasta",
        )
        if result is None:
            raise SequenceFetchError(f"Genomic accession '{direct_accession}' not found in NCBI nuccore")
        header, sequence, url = result
        accession = _accession_from_header(header) or direct_accession
        return (
            _sequence_record(
                sequence_type="dna_genomic",
                source_database="ncbi",
                accession=accession,
                sequence=sequence,
                source_url=url,
                inferred=False,
            ),
            {"genbank_accession": accession},
            warnings,
        )

    term = f"({_ncbi_term_for_request(request)}) AND biomol_genomic[PROP]"
    ids, _ = _ncbi_esearch(
        db="nuccore", term=term, config=ncfg, session=session, max_results=config.max_candidates_per_source
    )
    if not ids:
        ids, _ = _ncbi_esearch(
            db="nuccore",
            term=_ncbi_term_for_request(request),
            config=ncfg,
            session=session,
            max_results=config.max_candidates_per_source,
        )
    if not ids:
        raise SequenceFetchError(f"No genomic DNA found for {_target_label(request)}")

    attempted_not_found: list[str] = []
    selected_id: str | None = None
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
        gene_record, gene_ids, gene_warnings = _fetch_dna_genomic_from_gene_locus(
            request=request,
            config=config,
            session=session,
        )
        warnings.append("Nuccore genomic candidates lacked FASTA; used gene-locus genomic fallback")
        warnings.extend(gene_warnings)
        return gene_record, gene_ids, warnings

    if attempted_not_found:
        warnings.append(f"Primary genomic candidate(s) had no FASTA records; used fallback candidate {selected_id}")

    accession = _accession_from_header(header) or selected_id

    return (
        _sequence_record(
            sequence_type="dna_genomic",
            source_database="ncbi",
            accession=accession,
            sequence=sequence,
            source_url=url,
            inferred=False,
        ),
        {"genbank_accession": accession},
        warnings,
    )


def _fetch_dna_genomic_from_gene_locus(
    request: SequenceFetchRequest,
    config: SequenceFetchConfig,
    session: requests.Session,
) -> tuple[FetchedSequence, dict[str, str], list[str]]:
    """Fetch genomic DNA by resolving gene locus coordinates from NCBI gene."""
    warnings: list[str] = []
    ncfg = _ncbi_config(config)

    if request.gene_id:
        gene_ids = [request.gene_id]
    else:
        gene_ids, _ = _ncbi_esearch(
            db="gene",
            term=_ncbi_gene_term(*_name_and_organism(request)),
            config=ncfg,
            session=session,
            max_results=config.max_candidates_per_source,
        )
    if not gene_ids:
        raise SequenceFetchError(f"No gene IDs found for {_target_label(request)}")

    selected_gene_id = gene_ids[0]
    if len(gene_ids) > 1:
        warnings.append(f"Multiple gene candidates found ({len(gene_ids)}); using top hit {selected_gene_id}")

    summary_result = _ncbi_esummary(
        db="gene",
        identifier=selected_gene_id,
        config=ncfg,
        session=session,
    )
    if summary_result is None:
        raise SequenceFetchError(f"Gene summary not found for gene ID '{selected_gene_id}'")
    summary, _url = summary_result

    gene_payload = summary.get(selected_gene_id, {})
    genomic_info = gene_payload.get("genomicinfo", [])
    if not genomic_info:
        raise SequenceFetchError(f"Gene record '{selected_gene_id}' lacks genomic coordinates in esummary")

    selected_region = None
    for region in genomic_info:
        if region.get("chraccver"):
            selected_region = region
            break
    if selected_region is None:
        raise SequenceFetchError(f"Gene record '{selected_gene_id}' has genomic info but no chromosome accession")

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
        raise SequenceFetchError(f"Genomic fetch failed for locus '{chr_accession}:{seq_start}-{seq_stop}'")
    header, sequence, url = result

    accession = _accession_from_header(header) or chr_accession
    warnings.append(f"Fetched gene-locus genomic interval {chr_accession}:{seq_start}-{seq_stop}:{strand}")

    return (
        _sequence_record(
            sequence_type="dna_genomic",
            source_database="ncbi",
            accession=accession,
            sequence=sequence,
            source_url=url,
            inferred=False,
        ),
        {"gene_id": selected_gene_id, "genbank_accession": accession},
        warnings,
    )


def _fetch_dna_cds(
    request: SequenceFetchRequest,
    config: SequenceFetchConfig,
    session: requests.Session,
) -> tuple[FetchedSequence, dict[str, str], list[str]]:
    """Fetch coding DNA sequence (CDS)."""
    warnings: list[str] = []
    ncfg = _ncbi_config(config)

    accession = request.nucleotide_accession
    if accession:
        efetch_result = _ncbi_efetch(
            db="nuccore",
            identifier=accession,
            config=ncfg,
            session=session,
            rettype="fasta_cds_na",
        )
        if efetch_result is None:
            raise SequenceFetchError(f"CDS efetch failed for accession '{accession}'")
        text, url = efetch_result
        records = _parse_fasta_records(text)
        selected = _select_best_record(records, request.target_name)
        if selected is None:
            raise SequenceFetchError(f"No CDS sequence found for accession '{accession}'")

        cds_acc = selected.accession or accession
        return (
            _sequence_record(
                sequence_type="dna_cds",
                source_database="ncbi",
                accession=cds_acc,
                sequence=selected.sequence,
                source_url=url,
                inferred=False,
            ),
            {"cds_accession": cds_acc},
            warnings,
        )

    # Fallback via protein accession
    if request.protein_accession:
        efetch_result = _ncbi_efetch(
            db="protein",
            identifier=request.protein_accession,
            config=ncfg,
            session=session,
            rettype="fasta_cds_na",
        )
        if efetch_result is not None:
            text, url = efetch_result
            records = _parse_fasta_records(text)
            selected = _select_best_record(records, request.target_name)
            if selected is not None:
                cds_acc = selected.accession or request.protein_accession
                warnings.append("CDS inferred from protein record using NCBI fasta_cds_na")
                return (
                    _sequence_record(
                        sequence_type="dna_cds",
                        source_database="ncbi",
                        accession=cds_acc,
                        sequence=selected.sequence,
                        source_url=url,
                        inferred=True,
                    ),
                    {"cds_accession": cds_acc},
                    warnings,
                )

    # Name-based fallback through nuccore
    term = f"({_ncbi_term_for_request(request)}) AND (mRNA[Title] OR CDS[Title] OR coding[Title])"
    ids, _ = _ncbi_esearch(
        db="nuccore", term=term, config=ncfg, session=session, max_results=config.max_candidates_per_source
    )
    if not ids:
        raise SequenceFetchError(f"No CDS found for {_target_label(request)}")

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
        raise SequenceFetchError(f"CDS efetch failed for NCBI ID '{ids[0]}'")
    text, url = efetch_result
    records = _parse_fasta_records(text)
    selected = _select_best_record(records, request.target_name)
    if selected is None:
        raise SequenceFetchError(f"No CDS sequence found in fasta_cds_na response for NCBI ID '{ids[0]}'")

    cds_acc = selected.accession or ids[0]

    return (
        _sequence_record(
            sequence_type="dna_cds",
            source_database="ncbi",
            accession=cds_acc,
            sequence=selected.sequence,
            source_url=url,
            inferred=False,
        ),
        {"cds_accession": cds_acc},
        warnings,
    )


def _fetch_rna_transcript(
    request: SequenceFetchRequest,
    config: SequenceFetchConfig,
    session: requests.Session,
) -> tuple[FetchedSequence, dict[str, str], list[str]]:
    """Fetch transcript RNA sequence."""
    from Bio.Seq import transcribe

    warnings: list[str] = []
    ncfg = _ncbi_config(config)

    transcript_accession = request.nucleotide_accession
    if transcript_accession:
        result = _ncbi_fetch_first_fasta(
            db="nuccore",
            identifier=transcript_accession,
            config=ncfg,
            session=session,
            rettype="fasta",
        )
        if result is None:
            raise SequenceFetchError(f"Transcript '{transcript_accession}' not found in NCBI nuccore")
        header, sequence, url = result
        accession = _accession_from_header(header) or transcript_accession
        return (
            _sequence_record(
                sequence_type="rna_transcript",
                source_database="ncbi",
                accession=accession,
                sequence=transcribe(sequence.upper()),  # type: ignore[no-untyped-call]
                source_url=url,
                inferred=True,
            ),
            {"transcript_id": accession},
            warnings,
        )

    term = f"({_ncbi_term_for_request(request)}) AND (biomol_mrna[PROP] OR biomol_rna[PROP])"
    ids, _ = _ncbi_esearch(
        db="nuccore", term=term, config=ncfg, session=session, max_results=config.max_candidates_per_source
    )
    if not ids:
        raise SequenceFetchError(f"No transcript found for {_target_label(request)}")

    if len(ids) > 1:
        warnings.append(f"Multiple transcript candidates found ({len(ids)}); using top hit {ids[0]}")

    result = _ncbi_fetch_first_fasta(
        db="nuccore",
        identifier=ids[0],
        config=ncfg,
        session=session,
        rettype="fasta",
    )
    if result is None:
        raise SequenceFetchError(f"Transcript efetch failed for NCBI ID '{ids[0]}'")
    header, sequence, url = result
    accession = _accession_from_header(header) or ids[0]

    return (
        _sequence_record(
            sequence_type="rna_transcript",
            source_database="ncbi",
            accession=accession,
            sequence=transcribe(sequence.upper()),  # type: ignore[no-untyped-call]
            source_url=url,
            inferred=True,
        ),
        {"transcript_id": accession},
        warnings,
    )


def _fetch_rna_premrna(
    request: SequenceFetchRequest,
    config: SequenceFetchConfig,
    session: requests.Session,
) -> tuple[FetchedSequence, dict[str, str], list[str]]:
    """Fetch or infer pre-mRNA sequence from genomic sequence."""
    from Bio.Seq import transcribe

    try:
        genomic_result = _fetch_dna_genomic(request, config, session)
    except SequenceFetchError as exc:
        raise SequenceFetchError(f"pre-mRNA inference failed: {exc}") from exc

    genomic_record, ids, warnings = genomic_result
    premrna = genomic_record.sequence

    warnings.append("pre-mRNA sequence is inferred from genomic DNA and includes introns where present.")

    return (
        _sequence_record(
            sequence_type="rna_premrna",
            source_database=genomic_record.source_database,
            accession=genomic_record.accession,
            sequence=transcribe(premrna.upper()),  # type: ignore[no-untyped-call]
            source_url=genomic_record.source_url,
            inferred=True,
        ),
        ids,
        warnings,
    )


def _fetch_structure(
    request: SequenceFetchRequest,
    config: SequenceFetchConfig,
    session: requests.Session,
    resolved_ids: dict[str, str] | None = None,
) -> tuple[FetchedStructure, dict[str, str], list[str]]:
    """Fetch structure metadata from PDB."""
    warnings: list[str] = []

    pdb_id = request.pdb_id
    uniprot_id = request.uniprot_id
    entry: dict[str, Any] | None = None

    if not pdb_id and not uniprot_id and resolved_ids:
        uniprot_id = resolved_ids.get("uniprot_id")
        if uniprot_id:
            warnings.append(f"Using resolved UniProt ID '{uniprot_id}' from protein fetch for structure lookup")

    if not pdb_id and not uniprot_id:
        target_name, organism = _name_and_organism(request)
        entry = _search_uniprot_entry(
            target_name=target_name,
            organism=organism,
            prefer_pdb_crossref=True,
            max_candidates=config.max_candidates_per_source,
            session=session,
        )
        if entry is not None:
            uniprot_id = entry.get("primaryAccession")
            if uniprot_id:
                warnings.append(f"Resolved UniProt ID '{uniprot_id}' from name+organism for structure lookup")

    if not pdb_id and uniprot_id:
        if entry is None:
            entry = _fetch_uniprot_entry(uniprot_id, session)
        if entry is None:
            raise SequenceFetchError(f"UniProt entry '{uniprot_id}' not found")
        pdb_ids = _extract_pdb_crossrefs(entry)
        if pdb_ids:
            pdb_id = pdb_ids[0]
            warnings.append(f"Using first UniProt-linked PDB ID '{pdb_id}' from cross references")
        else:
            raise SequenceFetchError(f"UniProt ID '{uniprot_id}' has no linked PDB cross-references")

    if not pdb_id:
        raise SequenceFetchError(f"No PDB ID resolved for {_target_label(request)}")

    pdb_id = pdb_id.upper()
    meta = _fetch_pdb_entry(pdb_id, session)
    if meta is None:
        raise SequenceFetchError(f"PDB ID '{pdb_id}' not found")

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


def _name_and_organism(request: SequenceFetchRequest) -> tuple[str, str]:
    """Return the request's target name and organism for a search by name, or fail if it has none."""
    if not (request.target_name and request.organism):
        raise SequenceFetchError(
            "this sequence type needs a search by name, so give target_name and organism, or an accession for it"
        )
    return request.target_name, request.organism


def _target_label(request: SequenceFetchRequest) -> str:
    """Describe what a by-name or by-gene search looked for, for error messages."""
    if request.target_name and request.organism:
        return f"'{request.target_name}' in '{request.organism}'"
    return f"gene ID '{request.gene_id}'"


def _ncbi_term_for_request(request: SequenceFetchRequest) -> str:
    """Compose an Entrez term with optional gene_id preference."""
    if request.gene_id and not (request.target_name and request.organism):
        return f"{request.gene_id}[Gene ID]"
    base_term = _ncbi_term(*_name_and_organism(request))
    if request.gene_id:
        return f"({request.gene_id}[Gene ID] OR {base_term})"
    return base_term


def _parse_coordinates(value: str | None) -> tuple[str | None, int, int, str | None] | None:
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
    accession: str | None,
    sequence: str,
    source_url: str,
    inferred: bool,
) -> FetchedSequence:
    """Build a normalized FetchedSequence record."""
    clean_sequence = re.sub(r"\s+", "", sequence).upper()
    checksum = hashlib.sha256(clean_sequence.encode("utf-8")).hexdigest()

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
    return f'("{escaped_target}"[Gene] OR "{escaped_target}"[Title]) AND "{escaped_organism}"[Organism]'


def _ncbi_gene_term(target_name: str, organism: str) -> str:
    """Build an Entrez search term for the gene database."""
    escaped_target = target_name.replace('"', "")
    escaped_organism = organism.replace('"', "")
    return f'("{escaped_target}"[Gene Name] OR "{escaped_target}"[Title]) AND "{escaped_organism}"[Organism]'


def _select_best_record(
    records: list[NCBIFastaRecord],
    target_name: str | None,
) -> NCBIFastaRecord | None:
    """Select the FASTA record whose header best matches target_name, or the first without one."""
    if not records:
        return None
    if not target_name:
        return records[0]
    lowered = target_name.lower()
    for record in records:
        if lowered in record.header.lower():
            return record
    return records[0]
