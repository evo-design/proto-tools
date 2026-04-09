"""proto_tools/tools/gene_annotation/blast/blast_search.py.

Unified BLAST search tool supporting both online (NCBI) and local modes.
"""

import csv
import io
import logging
import re
import tempfile
import warnings
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)

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
    """

    qseqid: str = Field(description="Query sequence ID")
    sseqid: str = Field(description="Subject sequence ID")
    pident: float = Field(description="Percentage of identical matches")
    length: int = Field(description="Alignment length")
    mismatch: int = Field(description="Number of mismatches")
    gapopen: int = Field(description="Number of gap openings")
    qstart: int = Field(description="Start of alignment in query")
    qend: int = Field(description="End of alignment in query")
    sstart: int = Field(description="Start of alignment in subject")
    send: int = Field(description="End of alignment in subject")
    evalue: float = Field(description="Expect value")
    bitscore: float = Field(description="Bit score")


# Input:
class BlastSearchInput(BaseToolInput):
    """Input for BLAST search.

    The ``query`` field accepts either a raw sequence string or a path to a
    FASTA file. Validation distinguishes between the two: if the value points
    to an existing file it is treated as a FASTA path; otherwise it is treated
    as a raw sequence string.

    Attributes:
        query (str): A raw nucleotide/protein sequence (e.g. ``"ATGCGTAAA"``)
            or a path to a FASTA file.
        query_type (Literal['sequence', 'fasta_path']): Automatically set to ``"sequence"`` or ``"fasta_path"``
            during validation. Read-only; do not set manually.
    """

    query: str = InputField(
        description=("Query sequence string or path to a FASTA file containing query sequence(s)"),
    )
    query_type: Literal["sequence", "fasta_path"] = InputField(
        default="sequence",
        description=("Auto-inferred query type: 'sequence' for raw string, 'fasta_path' for a file path."),
    )

    @model_validator(mode="after")
    def infer_query_type(self) -> "BlastSearchInput":
        """Classify query as a raw sequence or a FASTA file path."""
        path = Path(self.query)
        if path.exists() and path.is_file():
            object.__setattr__(self, "query_type", "fasta_path")
        else:
            # Not a file; treat as raw sequence. Validate characters.
            cleaned = self.query.strip()
            if not cleaned:
                raise ValueError("Query sequence cannot be empty")
            if not _SEQUENCE_CHARS.match(cleaned):
                raise ValueError(
                    f"Query does not look like a valid sequence (contains "
                    f"unexpected characters) and is not an existing file path: "
                    f"{self.query!r}"
                )
            object.__setattr__(self, "query_type", "sequence")
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

        df = pd.DataFrame([hit.model_dump() for hit in self.hits])

        if file_format == "csv":
            df.to_csv(path, index=False)
        elif file_format == "json":
            df.to_json(path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# Config:
class BlastSearchConfig(BaseConfig):
    """Configuration for BLAST search.

    Controls search mode (online vs local), BLAST program selection, scoring
    parameters, filtering thresholds, and output options. Parameters that only
    apply to one search mode are documented as such and are ignored when using
    the other mode.

    Attributes:
        search_mode (Literal['online', 'local']): ``"online"`` routes to NCBI QBLAST; ``"local"`` runs
            BLAST+ CLI against a local database.
        program (BLAST_PROGRAMS): BLAST algorithm (blastn, blastp, blastx, tblastn, tblastx).
        database (BLAST_DATABASES): NCBI database to search (online mode only).
        local_db (str | None): Path to a local BLAST database (local mode only, required).
        num_threads (int): CPU threads for local search.
        evalue (float | None): E-value threshold.
        max_target_seqs (int | None): Max aligned sequences to keep.
        max_hsps (int | None): Max HSPs per query-subject pair.
        word_size (int | None): Word size for initial matches.
        gapopen (int | None): Cost to open a gap.
        gapextend (int | None): Cost to extend a gap.
        matrix (SCORING_MATRICES | None): Scoring matrix for protein searches.
        reward (int | None): Nucleotide match reward (blastn only).
        penalty (int | None): Nucleotide mismatch penalty (blastn only).
        task (BLAST_TASKS | None): BLAST task for optimized defaults.
        perc_identity (float | None): Minimum percent identity filter.
        qcov_hsp_perc (float | None): Minimum query coverage per HSP.
        threshold (int | None): Min word score for lookup table (protein only).
        comp_based_stats (Literal[0, 1, 2, 3] | None): Composition-based statistics mode (protein only).
        soft_masking (bool | None): Use soft masking for initial matches.
        lcase_masking (bool | None): Treat lowercase in FASTA as masked.
        dust (str | None): Low-complexity filter for nucleotide queries (blastn only).
        seg (str | None): Low-complexity filter for protein queries.
        ungapped (bool | None): Perform ungapped alignment only.
        strand (Literal['both', 'plus', 'minus'] | None): Query strand(s) to search (nucleotide queries only).
        query_gencode (int | None): Genetic code for translating query (blastx/tblastx).
        db_gencode (int | None): Genetic code for translating DB (tblastn/tblastx).
        window_size (int | None): Multiple-hits window size.
        xdrop_ungap (float | None): X-dropoff for ungapped extensions.
        xdrop_gap (float | None): X-dropoff for preliminary gapped extensions.
        xdrop_gap_final (float | None): X-dropoff for final gapped alignment.
        use_sw_tback (bool | None): Compute Smith-Waterman alignments (protein only).
        culling_limit (int | None): Delete hits enveloped by better hits.
        best_hit_overhang (float | None): Best-hit algorithm overhang value.
        best_hit_score_edge (float | None): Best-hit algorithm score edge value.
        subject_besthit (bool | None): Only report best hit per subject.
        entrez_query (str | None): Restrict online search with an Entrez query.
        hitlist_size (int | None): Number of hits to return (online mode only).
        megablast (bool | None): Use MegaBLAST algorithm (online mode, blastn only).
    """

    # --- Mode selection ---
    search_mode: Literal["online", "local"] = ConfigField(
        default="online",
        title="Search Mode",
        description=('Search mode: "online" queries NCBI servers, "local" runs BLAST+ against a local database'),
    )
    program: BLAST_PROGRAMS = ConfigField(
        default="blastn",
        title="BLAST Program",
        description="BLAST algorithm to use for the search",
    )

    # --- Online-only ---
    database: BLAST_DATABASES = ConfigField(
        default="nt",
        title="NCBI Database",
        description=("NCBI database to search against (online mode only). Ignored when search_mode is 'local'."),
        advanced=True,
        depends_on={"search_mode": ["online"]},
    )
    entrez_query: str | None = ConfigField(
        default=None,
        title="Entrez Query",
        description=("Restrict online search with an Entrez query (e.g. 'Homo sapiens[Organism]'). Online only."),
        advanced=True,
        depends_on={"search_mode": ["online"]},
    )
    hitlist_size: int | None = ConfigField(
        default=None,
        title="Hit List Size",
        description=("Number of database sequences to return. Defaults to 50 on NCBI. Online mode only."),
        ge=1,
        advanced=True,
        depends_on={"search_mode": ["online"]},
    )
    megablast: bool | None = ConfigField(
        default=None,
        title="Use MegaBLAST",
        description=(
            "Use MegaBLAST algorithm for blastn searches (online mode only). Ignored when search_mode is 'local'."
        ),
        advanced=True,
        depends_on={"search_mode": ["online"]},
    )

    # --- Local-only ---
    local_db: str | None = ConfigField(
        default=None,
        title="Local BLAST Database",
        description=("Path to a local BLAST database (no file extensions). Required for local mode."),
        depends_on={"search_mode": ["local"]},
    )
    num_threads: int = ConfigField(
        default=4,
        ge=1,
        title="Number of Threads",
        description="Number of CPU threads for local BLAST search",
        advanced=True,
        depends_on={"search_mode": ["local"]},
    )

    # --- Scoring parameters ---
    evalue: float | None = ConfigField(
        default=None,
        title="E-value Threshold",
        description=("Expectation value threshold for reporting hits. Default is 10.0 in both online and local BLAST."),
        gt=0,
    )
    word_size: int | None = ConfigField(
        default=None,
        title="Word Size",
        description=("Length of initial exact match. Defaults: 28 (megablast), 11 (blastn), 3 (protein)."),
        ge=2,
        advanced=True,
    )
    gapopen: int | None = ConfigField(
        default=None,
        title="Gap Open Cost",
        description=("Cost to open a gap. Defaults: 5 (blastn), 11 (protein programs)."),
        ge=0,
        advanced=True,
    )
    gapextend: int | None = ConfigField(
        default=None,
        title="Gap Extend Cost",
        description=(
            "Cost to extend a gap. Not supported for tblastx. Defaults: 2 (blastn), 1 (blastp/blastx/tblastn)."
        ),
        ge=0,
        advanced=True,
    )
    matrix: SCORING_MATRICES | None = ConfigField(
        default=None,
        title="Scoring Matrix",
        description=("Substitution matrix for protein alignments. Not applicable to blastn. Default: BLOSUM62."),
        advanced=True,
    )
    reward: int | None = ConfigField(
        default=None,
        title="Nucleotide Match Reward",
        description=("Reward for a nucleotide match (blastn only). Default: 1 (megablast), 2 (blastn/dc-megablast)."),
        ge=0,
        advanced=True,
    )
    penalty: int | None = ConfigField(
        default=None,
        title="Nucleotide Mismatch Penalty",
        description=("Penalty for nucleotide mismatch (blastn only). Must be negative. Default: -2 or -3."),
        le=0,
        advanced=True,
    )
    threshold: int | None = ConfigField(
        default=None,
        title="Word Score Threshold",
        description=("Minimum word score for BLAST lookup table (protein only). Default: 11-13 depending on program."),
        ge=1,
        advanced=True,
    )
    comp_based_stats: Literal[0, 1, 2, 3] | None = ConfigField(
        default=None,
        title="Composition-Based Statistics",
        description=(
            "Composition-based score adjustment (protein). 0=off, 1=stats, 2=adjust (default), 3=unconditional."
        ),
        advanced=True,
    )

    # --- Filtering parameters ---
    max_target_seqs: int | None = ConfigField(
        default=None,
        title="Max Target Sequences",
        description=("Maximum number of aligned sequences to keep. Default: 500 (local mode)."),
        ge=1,
        advanced=True,
    )
    max_hsps: int | None = ConfigField(
        default=None,
        title="Max HSPs",
        description=("Maximum number of HSPs (high-scoring segment pairs) per query-subject pair. Default: no limit."),
        ge=1,
        advanced=True,
    )
    perc_identity: float | None = ConfigField(
        default=None,
        title="Percent Identity Cutoff",
        description=("Minimum percent identity for reported alignments (0-100). Most useful for blastn."),
        ge=0,
        le=100,
        advanced=True,
    )
    qcov_hsp_perc: float | None = ConfigField(
        default=None,
        title="Query Coverage Per HSP",
        description="Minimum query coverage percentage per HSP (0-100).",
        ge=0,
        le=100,
        advanced=True,
    )
    culling_limit: int | None = ConfigField(
        default=None,
        title="Culling Limit",
        description=("Delete hits enveloped by at least this many higher-scoring hits."),
        ge=0,
        advanced=True,
    )
    best_hit_overhang: float | None = ConfigField(
        default=None,
        title="Best Hit Overhang",
        description=("Best Hit algorithm overhang value (>0 and <0.5). Incompatible with culling_limit."),
        gt=0,
        lt=0.5,
        advanced=True,
    )
    best_hit_score_edge: float | None = ConfigField(
        default=None,
        title="Best Hit Score Edge",
        description=("Best Hit algorithm score edge value (>0 and <0.5). Incompatible with culling_limit."),
        gt=0,
        lt=0.5,
        advanced=True,
    )
    subject_besthit: bool | None = ConfigField(
        default=None,
        title="Subject Best Hit",
        description="Only report the best hit per subject sequence.",
        advanced=True,
    )

    # --- Masking parameters ---
    soft_masking: bool | None = ConfigField(
        default=None,
        title="Soft Masking",
        description=(
            "Apply filtering as soft masks (for seeding only). Default: true (blastn), false (protein programs)."
        ),
        advanced=True,
    )
    lcase_masking: bool | None = ConfigField(
        default=None,
        title="Lowercase Masking",
        description="Treat lowercase letters in FASTA input as masked.",
        advanced=True,
    )
    dust: str | None = ConfigField(
        default=None,
        title="DUST Filter",
        description=("Low-complexity filter for nucleotide queries. Values: 'yes', 'no', or 'level window linker'."),
        advanced=True,
    )
    seg: str | None = ConfigField(
        default=None,
        title="SEG Filter",
        description=("Low-complexity filter for protein queries. Values: 'yes', 'no', or 'window locut hicut'."),
        advanced=True,
    )

    # --- Search parameters ---
    task: BLAST_TASKS | None = ConfigField(
        default=None,
        title="BLAST Task",
        description=("Task preset that sets optimized default parameters (e.g. megablast, blastp-fast, blastx-fast)."),
        advanced=True,
    )
    ungapped: bool | None = ConfigField(
        default=None,
        title="Ungapped Alignment",
        description="Perform ungapped alignment only.",
        advanced=True,
    )
    strand: Literal["both", "plus", "minus"] | None = ConfigField(
        default=None,
        title="Query Strand",
        description=("Query strand(s) to search: 'both', 'plus', or 'minus'. Only for blastn, blastx, tblastx."),
        advanced=True,
    )
    query_gencode: int | None = ConfigField(
        default=None,
        title="Query Genetic Code",
        description=(
            "Genetic code for translating the query sequence. Only for blastx and tblastx. Default: 1 (Standard)."
        ),
        ge=1,
        advanced=True,
    )
    db_gencode: int | None = ConfigField(
        default=None,
        title="Database Genetic Code",
        description=("Genetic code for translating database sequences. Only for tblastn/tblastx. Default: 1."),
        ge=1,
        advanced=True,
    )
    window_size: int | None = ConfigField(
        default=None,
        title="Window Size",
        description=("Multiple-hits window size for combining initial word hits."),
        ge=0,
        advanced=True,
    )
    xdrop_ungap: float | None = ConfigField(
        default=None,
        title="X-dropoff Ungapped",
        description="X-dropoff value (in bits) for ungapped extensions.",
        advanced=True,
    )
    xdrop_gap: float | None = ConfigField(
        default=None,
        title="X-dropoff Gapped",
        description=("X-dropoff value (in bits) for preliminary gapped extensions."),
        advanced=True,
    )
    xdrop_gap_final: float | None = ConfigField(
        default=None,
        title="X-dropoff Final",
        description="X-dropoff value (in bits) for final gapped alignment.",
        advanced=True,
    )
    use_sw_tback: bool | None = ConfigField(
        default=None,
        title="Smith-Waterman Traceback",
        description=("Compute locally optimal Smith-Waterman alignments. Only for blastp, blastx, tblastn."),
        advanced=True,
    )

    # Fields that only apply to online mode
    _online_only_fields: tuple[Any, ...] = (
        "database",
        "entrez_query",
        "hitlist_size",
        "megablast",
    )
    # Fields that only apply to local mode
    _local_only_fields: tuple[Any, ...] = (
        "local_db",
        "num_threads",
    )

    @model_validator(mode="after")
    def validate_mode_requirements(self) -> "BlastSearchConfig":
        """Validate that mode-specific required fields are provided and.

        warn about mode-inappropriate parameters.

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


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return BlastSearchInput(query="MKTLLILAVVAAALA")


@tool(
    key="blast-search",
    label="BLAST Search",
    category="gene_annotation",
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
        >>> from proto_tools.tools.gene_annotation import run_blast_search, BlastSearchInput, BlastSearchConfig
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
    from Bio.Blast import NCBIWWW, NCBIXML

    # Resolve query sequence
    if inputs.query_type == "fasta_path":
        seq_record = next(SeqIO.parse(inputs.query, "fasta"))  # type: ignore[no-untyped-call]
        query_seq = str(seq_record.seq)
    else:
        query_seq = inputs.query

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

    handle = NCBIWWW.qblast(
        program=config.program,
        database=config.database,
        sequence=query_seq,
        format_type="XML",
        **qblast_kwargs,
    )

    raw_xml = handle.read()
    handle.close()

    blast_records = list(NCBIXML.parse(io.StringIO(raw_xml)))  # type: ignore[no-untyped-call]
    hits = _blast_results_to_hits(blast_records)

    return BlastSearchOutput(
        metadata={
            "search_mode": "online",
            "program": config.program,
            "database": config.database,
            "query_length": len(query_seq),
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
        query_path = inputs.query

    # Build CLI params from typed config fields.
    # These config fields map 1:1 to BLAST+ CLI flags (same name).
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
        "max_hsps",
        "perc_identity",
        "qcov_hsp_perc",
        "culling_limit",
        "best_hit_overhang",
        "best_hit_score_edge",
        "subject_besthit",
        "soft_masking",
        "lcase_masking",
        "dust",
        "seg",
        "task",
        "ungapped",
        "strand",
        "query_gencode",
        "db_gencode",
        "window_size",
        "xdrop_ungap",
        "xdrop_gap",
        "xdrop_gap_final",
        "use_sw_tback",
    )
    cli_params: dict[str, Any] = {
        name: getattr(config, name) for name in _CLI_PARAMS if getattr(config, name) is not None
    }

    input_data = {
        "operation": "local_blast",
        "program": config.program,
        "query_path": query_path,
        "db": config.local_db,
        "num_threads": config.num_threads,
        "additional_params": cli_params,
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
        reader = csv.DictReader(io.StringIO(raw_output), delimiter="\t", fieldnames=list(BlastHit.model_fields))
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

    Produces the same 12-column layout as BLAST+ ``-outfmt 6`` so that
    online and local results are directly comparable.

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
                    )
                )

    return hits
