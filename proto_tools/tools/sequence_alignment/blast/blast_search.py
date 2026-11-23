"""Unified BLAST search tool supporting both online (NCBI) and local modes."""

import csv
import io
import logging
import re
import tempfile
import warnings
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)
from proto_tools.utils.device import RemoteDevice
from proto_tools.utils.file_inputs import FASTA_FILES, FileInput

logger = logging.getLogger(__name__)

BLAST_PROGRAMS = Literal["blastn", "blastp", "blastx", "tblastn", "tblastx"]
BLAST_DATABASES = Literal[
    "nt",  # Nucleotide collection
    "nr",  # Non-redundant protein sequences
    "refseq_rna",  # RefSeq RNA sequences
    "refseq_protein",  # RefSeq protein sequences
    "swissprot",  # Swiss-Prot protein sequences
    "pdb",  # Protein Data Bank
    "pataa",  # Patent protein sequences
    "patnt",  # Patent nucleotide sequences
]
SCORING_MATRICES = Literal[
    "BLOSUM45",
    "BLOSUM50",
    "BLOSUM62",
    "BLOSUM80",
    "BLOSUM90",
    "PAM30",
    "PAM70",
    "PAM250",
]
BLAST_TASKS = Literal[
    "megablast",
    "dc-megablast",
    "blastn",
    "blastn-short",
    "blastp",
    "blastp-short",
    "blastp-fast",
    "blastx",
    "blastx-fast",
    "tblastn",
    "tblastn-fast",
]

# Simple heuristic for raw sequence detection: only contains valid
# nucleotide/protein characters plus common whitespace.
_SEQUENCE_CHARS = re.compile(r"^[A-Za-z*\-\s]+$")
# A query is a FASTA file unless it is a valid sequence
_QUERY_FILES = FileInput("query sequence", FASTA_FILES.extensions, lambda text: bool(_SEQUENCE_CHARS.match(text)))
_STANDARD_HIT_FIELDS = (
    "qseqid",
    "sseqid",
    "pident",
    "length",
    "mismatch",
    "gapopen",
    "qstart",
    "qend",
    "sstart",
    "send",
    "evalue",
    "bitscore",
)
_ALIGNMENT_DETAIL_FIELDS = ("qseq", "sseq", "btop")


# ============================================================================
# Data Models
# ============================================================================
# Output hit model:
class BlastHit(BaseModel):
    """A single BLAST alignment hit.

    Represents one query-subject alignment from BLAST tabular output format.

    Attributes:
        qseqid (str): Query sequence ID.
        sseqid (str): Subject sequence ID.
        pident (float): Percentage of identical matches.
        length (int): Alignment length.
        mismatch (int): Number of mismatches.
        gapopen (int): Number of gap openings.
        qstart (int): Start of alignment in query.
        qend (int): End of alignment in query.
        sstart (int): Start of alignment in subject.
        send (int): End of alignment in subject.
        evalue (float): Expect value.
        bitscore (float): Bit score.
        qseq (str | None): Aligned query sequence, including gaps.
        sseq (str | None): Aligned subject sequence, including gaps.
        btop (str | None): BLAST traceback operations encoding mismatches and gaps.
    """

    qseqid: str = Field(title="Query ID", description="Query sequence ID")
    sseqid: str = Field(title="Subject ID", description="Subject sequence ID")
    pident: float = Field(title="Percent Identity", description="Percentage of identical matches")
    length: int = Field(title="Alignment Length", description="Alignment length")
    mismatch: int = Field(title="Mismatch Count", description="Number of mismatches")
    gapopen: int = Field(title="Gap Open Count", description="Number of gap openings")
    qstart: int = Field(title="Query Start", description="1-indexed inclusive start in query")
    qend: int = Field(title="Query End", description="1-indexed inclusive end in query")
    sstart: int = Field(
        title="Subject Start",
        description="1-indexed inclusive subject start (sstart > send means minus strand for nucleotide subjects)",
    )
    send: int = Field(title="Subject End", description="1-indexed inclusive subject end")
    evalue: float = Field(title="E-value", description="Expect value; smaller values are more significant")
    bitscore: float = Field(title="Bit Score", description="Alignment bit score; higher is better")
    qseq: str | None = Field(
        default=None,
        title="Aligned Query Sequence",
        description="Aligned query sequence including gaps, when returned by the backend",
    )
    sseq: str | None = Field(
        default=None,
        title="Aligned Subject Sequence",
        description="Aligned subject sequence including gaps, when returned by the backend",
    )
    btop: str | None = Field(
        default=None,
        title="BLAST Traceback Operations",
        description="BTOP mismatch and gap encoding, when returned by the backend",
    )


# Input:
class BlastSearchInput(BaseToolInput):
    """Input for BLAST search.

    The ``query`` field accepts either a raw sequence string or a path to a
    FASTA file. A ``Path`` is always a file. A string of sequence characters is
    a sequence, even if a file by that name exists. Any other string is read as
    a FASTA path, and fails validation if no such file exists.

    Attributes:
        query (str | Path): A raw nucleotide/protein sequence (e.g. ``"ATGCGTAAA"``)
            or a path to a FASTA file; a valid sequence is never read as a path.
        query_type (Literal['sequence', 'fasta_path']): Automatically set to ``"sequence"`` or ``"fasta_path"``
            during validation. Read-only; do not set manually.
    """

    query: str | Path = InputField(
        title="Query",
        description=("Query sequence string or path to a FASTA file containing query sequence(s)"),
    )
    query_type: Literal["sequence", "fasta_path"] = InputField(
        default="sequence",
        title="Query Type",
        description=("Auto-inferred query type: 'sequence' for raw string, 'fasta_path' for a file path."),
    )

    @field_validator("query")
    @classmethod
    def _resolve_query(cls, query: str | Path) -> str:
        """Return a FASTA file query as its absolute path, or check a sequence query's characters."""
        path = _QUERY_FILES.resolve(query)
        if path is not None:
            return str(path.resolve())
        cleaned = str(query).strip()
        if not cleaned:
            raise ValueError("Query sequence cannot be empty")
        if not _SEQUENCE_CHARS.match(cleaned):
            raise ValueError(f"Query is neither a valid sequence nor a path to a FASTA file: {query!r}")
        return str(query)

    @model_validator(mode="after")
    def infer_query_type(self) -> "BlastSearchInput":
        """Set ``query_type`` from the resolved query. A FASTA path is absolute, so it never matches the sequence characters."""
        query_type = "sequence" if _SEQUENCE_CHARS.match(str(self.query).strip()) else "fasta_path"
        object.__setattr__(self, "query_type", query_type)
        return self


# Output:
class BlastSearchOutput(BaseToolOutput):
    """Output from BLAST search.

    Attributes:
        hits (list[BlastHit]): BLAST alignment hits with standard tabular
            columns: qseqid, sseqid, pident, length, mismatch, gapopen,
            qstart, qend, sstart, send, evalue, bitscore.
    """

    hits: list[BlastHit] = Field(
        default_factory=list,
        title="Hits",
        description="BLAST alignment hits",
    )

    @property
    def num_hits(self) -> int:
        """Total number of alignment hits found."""
        return len(self.hits)

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["csv", "json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "csv"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        import pandas as pd

        if not self.hits:
            warnings.warn(
                "No BLAST results to export. The search returned no hits.",
                UserWarning,
                stacklevel=2,
            )
            return

        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "csv":
            # Preserve the established twelve-column CSV contract. Alignment
            # detail remains available on BlastHit and in the JSON export.
            df = pd.DataFrame([hit.model_dump(include=set(_STANDARD_HIT_FIELDS)) for hit in self.hits])
            df.to_csv(path, index=False)
        elif file_format == "json":
            df = pd.DataFrame([hit.model_dump() for hit in self.hits])
            df.to_json(path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# Config:
class BlastSearchConfig(BaseConfig):
    """Configuration for BLAST search.

    Controls search mode (online vs local), BLAST program selection, scoring,
    filtering, and output options. The wrapper's online path goes through
    NCBI's QBLAST web API which only exposes a subset of BLAST+ flags.
    Fields outside that subset are local-only and gated via ``depends_on`` /
    listed in ``_local_only_fields``.

    Attributes:
        search_mode (Literal['online', 'local']): ``"online"`` routes to NCBI
            QBLAST; ``"local"`` runs BLAST+ CLI against a local database.
        program (BLAST_PROGRAMS): BLAST algorithm (blastn, blastp, blastx,
            tblastn, tblastx).
        database (BLAST_DATABASES): NCBI database to search (online only).
        entrez_query (str | None): Restrict online search with an Entrez query.
        hitlist_size (int | None): Number of hits to return (online only).
        megablast (bool | None): Use MegaBLAST (online, blastn only).
        local_db (str | None): Path to a local BLAST database (local only, required).
        num_threads (int): CPU threads for local search.
        evalue (float | None): E-value threshold (both modes).
        word_size (int | None): Word size for initial matches (both modes).
        gapopen (int | None): Cost to open a gap (both modes).
        gapextend (int | None): Cost to extend a gap (both modes).
        matrix (SCORING_MATRICES | None): Scoring matrix for protein searches
            (both modes).
        reward (int | None): Nucleotide match reward (blastn only, both modes).
        penalty (int | None): Nucleotide mismatch penalty (blastn only, both modes).
        threshold (int | None): Min word score for lookup table (protein only,
            both modes).
        comp_based_stats (Literal[0, 1, 2, 3] | None): Composition-based stats
            mode (protein only, both modes).
        max_target_seqs (int | None): Max aligned sequences to keep (local only).
        perc_identity (float | None): Min percent identity filter (both modes).
        qcov_hsp_perc (float | None): Min query coverage per HSP (local only).
        soft_masking (bool | None): Soft masking for initial matches (local only).
        lcase_masking (bool | None): Treat lowercase in FASTA as masked (both modes).
        dust (str | None): Low-complexity filter for nucleotide queries (local only).
        seg (str | None): Low-complexity filter for protein queries (local only).
        task (BLAST_TASKS | None): Task preset (local only).
        ungapped (bool | None): Ungapped alignment only (both modes).
        strand (Literal['both', 'plus', 'minus'] | None): Query strand
            (local only; for blastn/blastx/tblastx).
        query_gencode (int | None): Genetic code for translating query
            (blastx/tblastx, both modes).
        db_gencode (int | None): Genetic code for translating DB
            (tblastn/tblastx, both modes).
        extra_args (list[str]): Verbatim BLAST+ CLI tokens for niche flags
            not exposed above (e.g. ``["-max_hsps", "1"]``). Local mode only;
            online mode goes through the NCBI web API which doesn't accept
            arbitrary CLI tokens.
    """

    # --- Mode selection ---
    search_mode: Literal["online", "local"] = ConfigField(
        default="online",
        title="Search Mode",
        description="`online` queries NCBI's QBLAST servers; `local` runs BLAST+ CLI against a local database.",
    )
    program: BLAST_PROGRAMS = ConfigField(
        default="blastn",
        title="BLAST Program",
        description="BLAST algorithm (blastn/blastp/blastx/tblastn/tblastx).",
    )

    # --- Online-only ---
    database: BLAST_DATABASES = ConfigField(
        default="nt",
        title="NCBI Database",
        description="NCBI database to search against (e.g. `nt`, `nr`, `swissprot`, `pdb`).",
    )
    entrez_query: str | None = ConfigField(
        default=None,
        title="Entrez Query",
        description="Restrict the online search with an Entrez query (e.g. `Homo sapiens[Organism]`).",
    )
    hitlist_size: int | None = ConfigField(
        default=None,
        title="Hit List Size",
        description="Number of database sequences to return; NCBI defaults to 50 when unset.",
        ge=1,
    )
    megablast: bool | None = ConfigField(
        default=None,
        title="Use MegaBLAST",
        description="Use the MegaBLAST algorithm for `blastn` (faster, optimized for highly similar sequences).",
    )

    # --- Local-only ---
    local_db: str | None = ConfigField(
        default=None,
        title="Local BLAST Database",
        description="Path to a local BLAST database stem (no file extensions). Required for local mode.",
    )
    num_threads: int = ConfigField(
        default=4,
        ge=1,
        title="Number of Threads",
        description="CPU threads for local BLAST search (upstream BLAST+ default is 1).",
        include_in_key=False,
    )

    # --- Scoring parameters ---
    evalue: float | None = ConfigField(
        default=None,
        title="E-value Threshold",
        description="Expectation value threshold for reporting hits; BLAST+ default is 10.0 when unset.",
        gt=0,
    )
    word_size: int | None = ConfigField(
        default=None,
        title="Word Size",
        description="Length of the initial exact match; BLAST+ defaults: 28 (megablast), 11 (blastn), 3 (blastp).",
        ge=2,
    )
    gapopen: int | None = ConfigField(
        default=None,
        title="Gap Open Cost",
        description="Cost to open a gap; BLAST+ defaults: 5 (blastn), 11 (protein programs).",
        ge=0,
    )
    gapextend: int | None = ConfigField(
        default=None,
        title="Gap Extend Cost",
        description="Cost to extend a gap; BLAST+ defaults 2 (blastn), 1 (protein). Not for tblastx.",
        ge=0,
    )
    matrix: SCORING_MATRICES | None = ConfigField(
        default=None,
        title="Scoring Matrix",
        description="Substitution matrix for protein alignments; BLAST+ default is BLOSUM62. Not applicable to blastn.",
    )
    reward: int | None = ConfigField(
        default=None,
        title="Nucleotide Match Reward",
        description="Match reward (blastn only); BLAST+ defaults 1 (megablast), 2 (blastn).",
        ge=0,
    )
    penalty: int | None = ConfigField(
        default=None,
        title="Nucleotide Mismatch Penalty",
        description="Mismatch penalty (blastn, ≤0); BLAST+ defaults -2 (megablast), -3 (others).",
        le=0,
    )
    threshold: int | None = ConfigField(
        default=None,
        title="Word Score Threshold",
        description="Minimum word score for the BLAST lookup table (protein only); BLAST+ default is 11 for blastp.",
        ge=1,
    )
    comp_based_stats: Literal[0, 1, 2, 3] | None = ConfigField(
        default=None,
        title="Composition-Based Statistics",
        description="Composition-based scoring (protein): 0=off, 1=stats, 2=adjust (default), 3=unconditional.",
    )

    # --- Filtering parameters ---
    # Many of these are local-only because the wrapper's online path goes
    # through the NCBI web API, which doesn't expose them.
    max_target_seqs: int | None = ConfigField(
        default=None,
        title="Max Target Sequences",
        description="Maximum number of aligned sequences to keep; BLAST+ default is 500.",
        ge=1,
    )
    perc_identity: float | None = ConfigField(
        default=None,
        title="Percent Identity Cutoff",
        description="Minimum percent identity for reported alignments (0-100); most useful with blastn.",
        ge=0,
        le=100,
    )
    qcov_hsp_perc: float | None = ConfigField(
        default=None,
        title="Query Coverage Per HSP",
        description="Minimum query coverage percentage per HSP (0-100).",
        ge=0,
        le=100,
    )

    # --- Masking parameters ---
    soft_masking: bool | None = ConfigField(
        default=None,
        title="Soft Masking",
        description="Apply filter as soft masks (seeding only); BLAST+ defaults True (blastn), False (protein).",
    )
    lcase_masking: bool | None = ConfigField(
        default=None,
        title="Lowercase Masking",
        description="Treat lowercase letters in the FASTA input as masked.",
    )
    dust: str | None = ConfigField(
        default=None,
        title="DUST Filter",
        description="Nucleotide low-complexity filter: `yes`, `no`, or `level window linker` (default `20 64 1`).",
    )
    seg: str | None = ConfigField(
        default=None,
        title="SEG Filter",
        description="Protein low-complexity filter: `yes`, `no`, or `window locut hicut` (blastp default `no`).",
    )

    # --- Search parameters ---
    task: BLAST_TASKS | None = ConfigField(
        default=None,
        title="BLAST Task",
        description="Task preset (e.g. `megablast`, `blastp-fast`, `blastx-fast`); flips multiple defaults at once.",
    )
    ungapped: bool | None = ConfigField(
        default=None,
        title="Ungapped Alignment",
        description="Perform ungapped alignment only.",
    )
    strand: Literal["both", "plus", "minus"] | None = ConfigField(
        default=None,
        title="Query Strand",
        description="Query strand(s) to search; BLAST+ default is `both`. Only for blastn, blastx, tblastx.",
    )
    query_gencode: int | None = ConfigField(
        default=None,
        title="Query Genetic Code",
        description="Genetic code for translating the query (blastx/tblastx only); BLAST+ default 1 (Standard).",
        ge=1,
    )
    db_gencode: int | None = ConfigField(
        default=None,
        title="Database Genetic Code",
        description="Genetic code for translating database sequences (tblastn/tblastx only); BLAST+ default 1.",
        ge=1,
    )

    # --- Escape hatch ---
    extra_args: list[str] = ConfigField(
        default=[],
        title="Extra CLI Arguments",
        description="Verbatim BLAST+ CLI tokens for niche flags (e.g. `['-max_hsps', '1']`). Local mode only.",
    )

    # Fields that only apply to online mode
    _online_only_fields: tuple[Any, ...] = (
        "database",
        "entrez_query",
        "hitlist_size",
        "megablast",
    )
    # Fields that only apply to local mode. Includes everything in the
    # local CLI surface that the wrapper's online qblast path doesn't map.
    _local_only_fields: tuple[Any, ...] = (
        "local_db",
        "num_threads",
        "max_target_seqs",
        "qcov_hsp_perc",
        "soft_masking",
        "dust",
        "seg",
        "task",
        "strand",
        "extra_args",
    )

    @model_validator(mode="after")
    def validate_mode_requirements(self) -> "BlastSearchConfig":
        """Validate mode-specific required fields and warn about mode-inappropriate parameters.

        Raises ``ValueError`` for hard errors (e.g. missing ``local_db``
        in local mode) and logs warnings for fields that will be ignored.
        """
        if self.search_mode == "local" and not self.local_db:
            raise ValueError("local_db is required when search_mode is 'local'")

        # Warn when online-only params are set in local mode
        if self.search_mode == "local":
            for field_name in self._online_only_fields:
                value = getattr(self, field_name)
                default = BlastSearchConfig.model_fields[field_name].default
                if value != default:
                    logger.warning(
                        "Config field '%s' is online-only and will be ignored in local search mode.",
                        field_name,
                    )

        # Warn when local-only params are set in online mode
        if self.search_mode == "online":
            for field_name in self._local_only_fields:
                value = getattr(self, field_name)
                default = BlastSearchConfig.model_fields[field_name].default
                if value != default:
                    logger.warning(
                        "Config field '%s' is local-only and will be ignored in online search mode.",
                        field_name,
                    )

        return self

    def remote_unsupported_reason(self, device: RemoteDevice) -> str | None:
        """Local-DB search can't run on the hosted cloud — the database lives on the caller's machine."""
        if self.search_mode == "local":
            return (
                "search_mode='local' needs a local BLAST database, which can't be staged to "
                f"device='{device}'. Use search_mode='online', or run locally with device='cpu'."
            )
        return None

    def local_execution_reason(self, device: RemoteDevice) -> str | None:
        """Online search is an HTTP call to NCBI, so a remote worker would only add a hop."""
        if self.search_mode == "online":
            return (
                f"search_mode='online' submits to NCBI QBLAST over HTTP, so device='{device}' would "
                f"only add a network hop and NCBI's queue would dominate either way."
            )
        return None


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return BlastSearchInput(query="MKTLLILAVVAAALA")


@tool(
    key="blast-search",
    label="BLAST Search",
    category="sequence_alignment",
    input_class=BlastSearchInput,
    config_class=BlastSearchConfig,
    output_class=BlastSearchOutput,
    description="Search sequences against BLAST databases (online or local)",
    example_input=example_input,
    cacheable=True,
)
def run_blast_search(
    inputs: BlastSearchInput,
    config: BlastSearchConfig,
    instance: Any = None,
) -> BlastSearchOutput:
    """Search sequences against BLAST databases.

    Dispatches to online (NCBI QBLAST) or local (BLAST+ CLI) search
    based on ``config.search_mode``.

    Args:
        inputs (BlastSearchInput): Validated BLAST search input.
        config (BlastSearchConfig): Validated BLAST search configuration.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        BlastSearchOutput: Structured output with BLAST alignment hits.

    Raises:
        RuntimeError: If the BLAST search fails.

    Examples:
        >>> from proto_tools.tools.sequence_alignment import run_blast_search, BlastSearchInput, BlastSearchConfig
        >>> inputs = BlastSearchInput(query="ATGCGTAAA")
        >>> config = BlastSearchConfig(program="blastn", database="nt")
        >>> result = run_blast_search(inputs, config)
        >>> print(f"Found {result.num_hits} hits")
    """
    if config.search_mode == "local":
        return _local_search(inputs, config, instance=instance)
    return _online_search(inputs, config)


# ============================================================================
# Helper Functions
# ============================================================================
def _online_search(inputs: BlastSearchInput, config: BlastSearchConfig) -> BlastSearchOutput:
    """Submit query to NCBI QBLAST and return results."""
    from Bio import SeqIO
    from Bio.Blast import NCBIXML

    from proto_tools.tools.sequence_alignment.blast._online_client import search_xml

    # Preserve every FASTA record and its identifier, including no-hit queries.
    query_ids: list[str] | None = None
    if inputs.query_type == "fasta_path":
        sequences = list(SeqIO.parse(inputs.query, "fasta"))  # type: ignore[no-untyped-call]
        query_ids = [record.id for record in sequences]
        if not sequences or any(not record.seq for record in sequences):
            raise ValueError("Online BLAST requires nonempty FASTA records")
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("Online BLAST requires unique FASTA identifiers")
        query_seq = "".join(f">{record.id}\n{record.seq}\n" for record in sequences)
        query_length = sum(len(record.seq) for record in sequences)
    else:
        query_seq = str(inputs.query)
        query_length = len(query_seq)

    # Build qblast keyword arguments from typed config fields.
    # Mapping: config field name → qblast kwarg name.
    _QBLAST_PARAM_MAP = {
        "evalue": "expect",
        "hitlist_size": "hitlist_size",
        "word_size": "word_size",
        "matrix": "matrix_name",
        "reward": "nucl_reward",
        "penalty": "nucl_penalty",
        "threshold": "threshold",
        "comp_based_stats": "composition_based_statistics",
        "perc_identity": "perc_ident",
        "entrez_query": "entrez_query",
        "megablast": "megablast",
        "ungapped": "ungapped_alignment",
        "lcase_masking": "lcase_mask",
        "query_gencode": "genetic_code",
        "db_gencode": "db_genetic_code",
    }
    qblast_kwargs: dict[str, Any] = {
        qblast_key: getattr(config, config_field)
        for config_field, qblast_key in _QBLAST_PARAM_MAP.items()
        if getattr(config, config_field) is not None
    }

    # gapcosts requires combining two fields into a single string
    if config.gapopen is not None or config.gapextend is not None:
        go = config.gapopen if config.gapopen is not None else 11
        ge = config.gapextend if config.gapextend is not None else 1
        qblast_kwargs["gapcosts"] = f"{go} {ge}"

    raw_xml = search_xml(
        {
            "PROGRAM": config.program,
            "DATABASE": config.database,
            "QUERY": query_seq,
            **{key.upper(): value for key, value in qblast_kwargs.items()},
        },
        timeout=config.timeout,
        client_identity=config._client_identity,
    )

    blast_records = list(NCBIXML.parse(io.StringIO(raw_xml)))  # type: ignore[no-untyped-call]
    returned_ids = [record.query.split()[0] for record in blast_records]
    if query_ids is not None:
        if len(returned_ids) != len(query_ids) or set(returned_ids) != set(query_ids):
            raise ValueError("Online BLAST response query identifiers do not match submitted FASTA")
    elif len(blast_records) != 1:
        raise ValueError("Online BLAST single-sequence response must contain one query")
    hits = _blast_results_to_hits(blast_records)

    return BlastSearchOutput(
        metadata={
            "search_mode": "online",
            "program": config.program,
            "database": config.database,
            "query_length": query_length,
            "query_ids": returned_ids,
        },
        hits=hits,
    )


def _local_search(
    inputs: BlastSearchInput,
    config: BlastSearchConfig,
    instance: Any = None,
) -> BlastSearchOutput:
    """Run BLAST+ locally against a local database."""
    # If query is a raw sequence, write it to a temp FASTA file
    if inputs.query_type == "sequence":
        tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 -- delete=False requires manual cleanup
            mode="w", suffix=".fasta", delete=False
        )
        tmp.write(f">query\n{inputs.query}\n")
        tmp.close()
        query_path = tmp.name
    else:
        query_path = str(inputs.query)

    # Typed config fields that map 1:1 to BLAST+ CLI flags (same name).
    _CLI_PARAMS = (
        "evalue",
        "word_size",
        "gapopen",
        "gapextend",
        "matrix",
        "reward",
        "penalty",
        "threshold",
        "comp_based_stats",
        "max_target_seqs",
        "perc_identity",
        "qcov_hsp_perc",
        "soft_masking",
        "lcase_masking",
        "dust",
        "seg",
        "task",
        "ungapped",
        "strand",
        "query_gencode",
        "db_gencode",
    )
    cli_args: list[str] = []
    for name in _CLI_PARAMS:
        val = getattr(config, name)
        if val is None:
            continue
        flag = f"-{name}"
        if isinstance(val, bool):
            if val:
                cli_args.append(flag)
        else:
            cli_args.extend([flag, str(val)])
    cli_args.extend(str(arg) for arg in config.extra_args)

    # Alignment strings and BTOP let downstream consumers locate mismatches,
    # including biologically important mismatches near an oligo's 3-prime end.
    if "-outfmt" not in cli_args:
        cli_args.extend(["-outfmt", "6 std qseq sseq btop"])

    input_data = {
        "operation": "local_blast",
        "program": config.program,
        "query_path": query_path,
        "db": config.local_db,
        "num_threads": config.num_threads,
        "cli_args": cli_args,
    }

    input_data["device"] = "cpu"
    try:
        output_data = ToolInstance.dispatch(
            "blast",
            input_data,
            instance=instance,
            config=config,
        )
    finally:
        # Clean up temp file if we created one
        if inputs.query_type == "sequence":
            Path(query_path).unlink(missing_ok=True)

    # Parse raw tabular output into BlastHit objects

    raw_output = output_data["stdout"]
    hits: list[BlastHit] = []
    if raw_output.strip():
        reader = csv.DictReader(
            io.StringIO(raw_output),
            delimiter="\t",
            fieldnames=[*_STANDARD_HIT_FIELDS, *_ALIGNMENT_DETAIL_FIELDS],
        )
        hits.extend(BlastHit(**row) for row in reader)  # type: ignore[arg-type]

    return BlastSearchOutput(
        metadata={
            "search_mode": "local",
            "program": config.program,
            "database": config.local_db,
            "num_threads": config.num_threads,
        },
        hits=hits,
    )


def _blast_results_to_hits(blast_records: Any) -> list[BlastHit]:
    """Convert Biopython BLAST records to BlastHit objects.

    Produces the standard BLAST fields plus aligned query and subject strings.
    BTOP is unavailable from Biopython's XML representation and remains
    ``None`` for online results.

    Args:
        blast_records (Any): List of Bio.Blast.Record objects from NCBIXML.parse.

    Returns:
        list[BlastHit]: List of BLAST alignment hits.
    """
    hits = []
    for record in blast_records:
        query_id = record.query.split()[0]
        for alignment in record.alignments:
            for hsp in alignment.hsps:
                align_len = getattr(hsp, "align_length", len(hsp.match))
                ident = hsp.identities if isinstance(hsp.identities, int) else 0
                gaps = getattr(hsp, "gaps", 0)
                gaps = gaps if isinstance(gaps, int) else 0
                gap_opens = sum(len(re.findall(r"-+", seq)) for seq in (hsp.query, hsp.sbjct))
                hits.append(
                    BlastHit(
                        qseqid=query_id,
                        sseqid=getattr(alignment, "accession", alignment.hit_id),
                        pident=(ident / align_len * 100) if align_len > 0 else 0,
                        length=align_len,
                        mismatch=align_len - ident - gaps,
                        gapopen=gap_opens,
                        qstart=hsp.query_start,
                        qend=hsp.query_end,
                        sstart=hsp.sbjct_start,
                        send=hsp.sbjct_end,
                        evalue=hsp.expect,
                        bitscore=hsp.bits,
                        qseq=hsp.query,
                        sseq=hsp.sbjct,
                    )
                )

    return hits
