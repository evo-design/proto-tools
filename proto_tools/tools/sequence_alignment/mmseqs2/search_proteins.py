"""MMseqs2 protein-vs-database search via ``mmseqs easy-search``.

Also defines shared data models (Mmseqs2Hit, Mmseqs2SequenceSearchResult),
constants, and helper functions used by all MMseqs2 search tools.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

if TYPE_CHECKING:
    import pandas as pd

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
    resolve_sequence_ids,
)

# ============================================================================
# Constants
# ============================================================================

# 0 = let mmseqs auto-detect all available cores.
DEFAULT_THREADS = 0
DEFAULT_SPLIT = 0
DEFAULT_SEARCH_SENSITIVITY = 5.7
# Genome-search wrapper opinion: bias toward higher sensitivity by default.
DEFAULT_GENOME_SENSITIVITY = 7.5
DEFAULT_CLUSTER_SENSITIVITY = 4.0
DEFAULT_MIN_SEQ_ID = 0.0
DEFAULT_CLUSTER_EVALUE = 0.001
DEFAULT_SEARCH_EVALUE = 0.001
DEFAULT_COVERAGE = 0.0
DEFAULT_CLUSTER_COVERAGE = 0.8
DEFAULT_COV_MODE = 0
DEFAULT_CLUSTER_MODE = 0
DEFAULT_SEARCH_MAX_SEQS = 300
DEFAULT_CLUSTER_MAX_SEQS = 20
DEFAULT_NUCL_STRAND = 2
SEARCH_TYPE_NUCLEOTIDE = 3

# Standard column formats for MMseqs2 output
M8_COLUMNS = ["query", "target", "pident", "evalue"]

# Reusable type aliases used across the mmseqs2 toolkit configs.
CovMode = Literal[0, 1, 2, 3, 4, 5]
ClusterMode = Literal[0, 1, 2, 3]
NuclStrand = Literal[0, 1, 2]


# ============================================================================
# Data Models
# ============================================================================
# Shared:
class Mmseqs2Hit(BaseModel):
    """A single MMseqs2 search hit.

    Represents a single alignment between a query sequence and a target sequence
    in the database.

    Attributes:
        target_id (str): Identifier of the target (database) sequence.
        pident (float): Percentage of identical matches (0-100).
        evalue (float): Expected value (E-value) - statistical significance.
    """

    target_id: str = Field(title="Target ID", description="Target sequence identifier from the database")
    pident: float = Field(title="Percent Identity", description="Percentage identity (0-100)")
    evalue: float = Field(title="E-value", description="E-value (expected number of chance matches)")


class Mmseqs2SequenceSearchResult(BaseModel):
    """Results for a single query sequence from an MMseqs2 search.

    Contains all hits found for one query sequence, along with the query
    sequence itself for reference.

    Attributes:
        query_id (str): Identifier of the query sequence.
        query_sequence (str): The input query sequence.
        hits (list[Mmseqs2Hit]): All hits found for this query, sorted by pident descending.
    """

    query_id: str = Field(title="Query ID", description="Query sequence identifier")
    query_sequence: str = Field(title="Query Sequence", description="The input query sequence")
    hits: list[Mmseqs2Hit] = Field(default_factory=list, title="Hits", description="List of hits for this query")

    @property
    def top_hit(self) -> Mmseqs2Hit | None:
        """Get the best hit by percent identity, or None if no hits."""
        return max(self.hits, key=lambda h: h.pident) if self.hits else None

    @property
    def num_hits(self) -> int:
        """Number of hits found for this query."""
        return len(self.hits)

    @property
    def has_hits(self) -> bool:
        """Whether any hits were found for this query."""
        return len(self.hits) > 0


# Input:
class Mmseqs2SearchProteinsInput(BaseToolInput):
    """Input object for MMseqs2 protein search.

    This class defines the input parameters for running MMseqs2 protein sequence
    searches.

    Attributes:
        query_sequences (list[str]): List of protein sequence strings (amino acid
            sequences) to search.
        sequence_ids (list[str] | None): Optional list of sequence identifiers.
            If not provided, sequences are assigned sequential IDs (seq_0, seq_1, ...).
        mmseqs_db (str | None): Target DB (path/slug/AssetRef). Mutually
            exclusive with ``target_sequences``.
        target_sequences (list[str] | None): Inline target sequences.
            Mutually exclusive with ``mmseqs_db``.
    """

    query_sequences: list[str] = InputField(
        title="Query Sequences",
        description="List of protein sequences to search",
    )
    sequence_ids: list[str] | None = InputField(
        default=None,
        title="Sequence IDs",
        description="Optional sequence identifiers (defaults to seq_0, seq_1, ...)",
    )
    mmseqs_db: str | None = InputField(
        default=None,
        title="MMseqs2 Database",
        description="Target DB (path/slug/AssetRef). Mutually exclusive with `target_sequences`.",
    )
    target_sequences: list[str] | None = InputField(
        default=None,
        title="Target Sequences",
        description="Inline target protein sequences. Mutually exclusive with `mmseqs_db`.",
    )

    @field_validator("query_sequences", mode="before")
    @classmethod
    def validate_query_sequences(cls, v: Any) -> Any:
        """Validate query sequences input."""
        if not isinstance(v, list):
            raise ValueError(f"query_sequences must be a list, got {type(v)}")
        if not v:
            raise ValueError("query_sequences list cannot be empty")
        if not all(isinstance(item, str) for item in v):
            raise ValueError("All items in query_sequences list must be strings")
        return v

    @field_validator("target_sequences", mode="before")
    @classmethod
    def validate_target_sequences(cls, v: Any) -> Any:
        """Validate target sequences when provided inline."""
        if v is None:
            return v
        if not isinstance(v, list):
            raise ValueError(f"target_sequences must be a list, got {type(v)}")
        if not v:
            raise ValueError("target_sequences list cannot be empty")
        if not all(isinstance(item, str) for item in v):
            raise ValueError("All items in target_sequences list must be strings")
        return v

    @model_validator(mode="after")
    def exactly_one_target(self) -> Mmseqs2SearchProteinsInput:
        """Enforce the 'target' XOR constraint at runtime."""
        if (self.mmseqs_db is None) == (self.target_sequences is None):
            raise ValueError("mmseqs2-search-proteins: provide exactly one of `mmseqs_db` or `target_sequences`.")
        return self


# Output:
class Mmseqs2SearchProteinsOutput(BaseToolOutput):
    """Output from MMseqs2 protein search.

    Contains per-sequence search results matching the input order.

    Attributes:
        results (list[Mmseqs2SequenceSearchResult]): List of search results, one per
            input sequence. The order matches the input sequences order.
    """

    results: list[Mmseqs2SequenceSearchResult] = Field(
        title="Search Results", description="List of search results, one per input sequence"
    )

    def __len__(self) -> int:
        """Get the number of results."""
        return len(self.results)

    def __getitem__(self, idx: int) -> Mmseqs2SequenceSearchResult:
        """Get a result by index."""
        return self.results[idx]

    def __iter__(self) -> Iterator[Mmseqs2SequenceSearchResult]:  # type: ignore[override]
        """Iterate over the results."""
        return iter(self.results)

    @property
    def total_hits(self) -> int:
        """Total number of hits across all sequences."""
        return sum(r.num_hits for r in self.results)

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["m8", "csv", "json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "m8"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        import pandas as pd

        path = Path(export_path).with_suffix(f".{file_format}")

        # Flatten results for tabular formats
        data = [
            {
                "query": result.query_id,
                "target": hit.target_id,
                "pident": hit.pident,
                "evalue": hit.evalue,
            }
            for result in self.results
            for hit in result.hits
        ]

        df = (
            pd.DataFrame(data, columns=["query", "target", "pident", "evalue"])
            if data
            else pd.DataFrame(columns=["query", "target", "pident", "evalue"])
        )

        if file_format in ["m8", "csv"]:
            sep = "\t" if file_format == "m8" else ","

            # m8 usually doesn't have header, but csv does.
            # Standard m8 is tab-separated content.
            header = file_format != "m8"
            df.to_csv(path, sep=sep, index=False, header=header)

        elif file_format == "json":
            # Export structured data — use pydantic's model_dump/dict
            json_data = [r.model_dump() for r in self.results]
            with open(path, "w") as f:
                json.dump(json_data, f, indent=2)

        else:
            raise ValueError(f"Unsupported format: {file_format}")


# Config:
class Mmseqs2SearchProteinsConfig(BaseConfig):
    """Configuration object for MMseqs2 protein search.

    Attributes:
        threads (int): CPU threads; ``0`` auto-detects all cores (the wrapper
            omits ``--threads`` since ``mmseqs`` rejects ``--threads 0``).
        split (int): Split into N chunks to bound memory; ``0`` = auto.
        sensitivity (float): Prefilter sensitivity (1.0-7.5); higher = slower
            but finds more remote homologs.
        evalue (float): E-value threshold for reported hits.
        min_seq_id (float): Minimum sequence identity (0.0-1.0) for reported hits.
        coverage (float): Minimum aligned-residue fraction (0.0-1.0); semantics
            depend on ``cov_mode``.
        cov_mode (CovMode): 0=query AND target, 1=target, 2=query,
            3-5=length-ratio variants.
        max_seqs (int): Max prefilter results per query.
        only_top_hits (bool): Wrapper filter; keep only the best hit
            (highest pident) per query sequence.
        use_gpu (bool): Run MMseqs2-GPU (``--gpu 1``); requires a ``.idx_pad``
            sibling on the target DB (built via ``mmseqs makepaddedseqdb``).
        extra_args (list[str]): Verbatim ``mmseqs easy-search`` CLI tokens
            for niche flags (e.g. ``["--alignment-mode", "3"]``).
    """

    threads: int = ConfigField(
        title="Number of Threads",
        default=DEFAULT_THREADS,
        ge=0,
        description="CPU threads; `0` lets MMseqs2 auto-detect all available cores.",
        include_in_key=False,
    )
    split: int = ConfigField(
        title="Split",
        default=DEFAULT_SPLIT,
        ge=0,
        description="Split input into N chunks for memory-bounded prefiltering; `0` picks the best split automatically.",
    )
    sensitivity: float = ConfigField(
        title="Search Sensitivity",
        default=DEFAULT_SEARCH_SENSITIVITY,
        ge=1.0,
        le=7.5,
        description="Prefilter sensitivity (1.0-7.5); higher = slower but more remote homologs found.",
    )
    evalue: float = ConfigField(
        title="E-value Threshold",
        default=DEFAULT_SEARCH_EVALUE,
        gt=0.0,
        description="E-value threshold for reported hits; raise to keep weaker matches.",
    )
    min_seq_id: float = ConfigField(
        title="Minimum Sequence Identity",
        default=DEFAULT_MIN_SEQ_ID,
        ge=0.0,
        le=1.0,
        description="Minimum sequence identity (0-1) for reported hits; raise to filter to closer homologs.",
    )
    coverage: float = ConfigField(
        title="Coverage Threshold",
        default=DEFAULT_COVERAGE,
        ge=0.0,
        le=1.0,
        description="Minimum aligned-residue fraction (0-1); semantics depend on `cov_mode`.",
    )
    cov_mode: CovMode = ConfigField(
        title="Coverage Mode",
        default=DEFAULT_COV_MODE,
        description=("How `coverage` is measured: 0=query AND target, 1=target, 2=query, 3-5=length-ratio variants."),
    )
    max_seqs: int = ConfigField(
        title="Max Prefilter Hits",
        default=DEFAULT_SEARCH_MAX_SEQS,
        ge=1,
        description="Max prefilter results per query; raise for deeper searches at the cost of runtime/memory.",
    )
    only_top_hits: bool = ConfigField(
        title="Only Top Hits",
        default=True,
        description="Wrapper-side filter: when True, keep only the best hit (highest pident) per query sequence.",
    )
    use_gpu: bool = ConfigField(
        title="Use GPU",
        default=False,
        description="Run MMseqs2-GPU (`--gpu 1`); requires a `*.idx_pad` GPU index alongside the target DB.",
    )
    extra_args: list[str] = ConfigField(
        title="Extra CLI Arguments",
        default=[],
        description="Verbatim `mmseqs easy-search` tokens for niche flags (e.g. `['--alignment-mode', '3']`).",
    )

    @property
    def gpus_per_instance(self) -> int:
        """Per-call GPU need: 1 when ``use_gpu`` is set, 0 otherwise."""
        return 1 if self.use_gpu else 0


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return Mmseqs2SearchProteinsInput(
        query_sequences=["MKTL"],
        target_sequences=["MKTL", "ARND"],
    )


@tool(
    key="mmseqs2-search-proteins",
    label="MMseqs2 Protein Search",
    category="sequence_alignment",
    input_class=Mmseqs2SearchProteinsInput,
    config_class=Mmseqs2SearchProteinsConfig,
    output_class=Mmseqs2SearchProteinsOutput,
    description="Search protein sequences using MMseqs2 with per-sequence results",
    example_input=example_input,
    iterable_input_field="query_sequences",
    iterable_output_field="results",
    cacheable=True,
)
def run_mmseqs2_search_proteins(
    inputs: Mmseqs2SearchProteinsInput,
    config: Mmseqs2SearchProteinsConfig,
    instance: Any = None,
) -> Mmseqs2SearchProteinsOutput:
    """Perform protein sequence search using MMseqs2.

    Searches query protein sequences against a target database and returns
    per-sequence results with all hits.

    Args:
        inputs (Mmseqs2SearchProteinsInput): Validated input containing query
            sequences and target database path.
        config (Mmseqs2SearchProteinsConfig): Configuration with search parameters.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        Mmseqs2SearchProteinsOutput: Per-sequence search results in input order.

    Raises:
        FileNotFoundError: If target database cannot be found.
        RuntimeError: If MMseqs2 command execution fails.

    Examples:
        >>> inputs = Mmseqs2SearchProteinsInput(
        ...     query_sequences=["MSKGEELFT", "MVLSPADKTN"], mmseqs_db="/path/to/protein_db"
        ... )
        >>> config = Mmseqs2SearchProteinsConfig()
        >>> result = run_mmseqs2_search_proteins(inputs, config)
        >>> print(f"Found {result[0].num_hits} hits for first sequence")
        >>> if result[0].top_hit:
        ...     print(f"Top hit: {result[0].top_hit.pident}% identity")
    """
    sequences = inputs.query_sequences
    sequence_ids = resolve_sequence_ids(sequences, inputs.sequence_ids)
    num_sequences = len(sequences)

    # GPU mode needs a pre-built padded DB; inline targets are CPU-only.
    mmseqs_db_for_dispatch: str | None = None
    if inputs.target_sequences is not None:
        if config.use_gpu:
            raise ValueError(
                "mmseqs2-search-proteins: use_gpu=True requires a pre-built GPU-padded "
                "MMseqs2 DB; inline `target_sequences` aren't supported in GPU mode."
            )
    else:
        # XOR validator guarantees mmseqs_db is non-None here; explicit check is for mypy narrowing.
        mmseqs_db_for_dispatch = inputs.mmseqs_db
        if config.use_gpu and inputs.mmseqs_db is not None:
            padded_stem = _resolve_gpu_db_stem(inputs.mmseqs_db)
            if padded_stem is None:
                raise ValueError(
                    f"mmseqs2-search-proteins: use_gpu=True requires a GPU-padded MMseqs2 DB "
                    f"alongside {inputs.mmseqs_db!r}. Build one with:\n"
                    f"  mmseqs createdb <fasta> <db>     # only if your input is a FASTA file\n"
                    f"  mmseqs makepaddedseqdb <db> <db>.idx_pad\n"
                    f"or set use_gpu=False."
                )
            mmseqs_db_for_dispatch = padded_stem

    output_data = ToolInstance.dispatch(
        "mmseqs2",
        {
            "device": "cuda" if config.use_gpu else "cpu",
            "operation": "protein_search",
            "sequences": sequences,
            "sequence_ids": sequence_ids,
            "mmseqs_db": mmseqs_db_for_dispatch,
            "target_sequences": inputs.target_sequences,
            "threads": config.threads,
            "split": config.split,
            "sensitivity": config.sensitivity,
            "evalue": config.evalue,
            "min_seq_id": config.min_seq_id,
            "coverage": config.coverage,
            "cov_mode": config.cov_mode,
            "max_seqs": config.max_seqs,
            "use_gpu": config.use_gpu,
            "extra_args": list(config.extra_args),
            "m8_columns": M8_COLUMNS,
        },
        instance=instance,
        config=config,
    )

    # Parse results
    raw_output = output_data["stdout"]
    df = _parse_m8_output(raw_output)

    # Filter to top hits if requested
    if config.only_top_hits and not df.empty:
        df = _filter_top_hits(df)

    # Build per-sequence results
    results = _build_sequence_search_results(sequences, sequence_ids, df)

    return Mmseqs2SearchProteinsOutput(
        metadata={
            "mmseqs_db": inputs.mmseqs_db,
            "threads": config.threads,
            "sensitivity": config.sensitivity,
            "evalue": config.evalue,
            "min_seq_id": config.min_seq_id,
            "coverage": config.coverage,
            "cov_mode": config.cov_mode,
            "max_seqs": config.max_seqs,
            "only_top_hits": config.only_top_hits,
            "num_sequences": num_sequences,
            "use_gpu": config.use_gpu,
        },
        results=results,
    )


# ============================================================================
# Helper Functions
# ============================================================================
def _resolve_gpu_db_stem(mmseqs_db: str) -> str | None:
    """Resolve the GPU-padded MMseqs2 DB stem to pass to ``easy-search --gpu 1``.

    Per the proto-tools convention (matching ``databases/entries/uniref30_2302.py``
    and the existing ``mmseqs2-homology-search`` tool), the GPU-padded DB built
    via ``mmseqs makepaddedseqdb <stem> <stem>.idx_pad`` lives at
    ``<stem>.idx_pad``. A valid padded DB also has the standard MMseqs2
    metadata file ``<padded_stem>.dbtype`` alongside.

    Args:
        mmseqs_db (str): Path to an MMseqs2 DB stem (regular or padded), or a
            directory containing one. FASTA inputs always return ``None`` —
            ``easy-search --gpu 1`` requires a padded DB, not a raw FASTA.

    Returns:
        str | None: The padded DB stem path if a valid padded DB is found,
            otherwise ``None``.
    """
    db = Path(mmseqs_db)

    def _is_valid_padded_db(stem: Path) -> bool:
        return stem.exists() and Path(f"{stem}.dbtype").is_file()

    # Case 1: user passed the padded DB stem directly.
    if db.name.endswith(".idx_pad") and _is_valid_padded_db(db):
        return str(db)

    # Case 2: padded sibling at <db>.idx_pad (proto-tools convention).
    sibling = db.parent / f"{db.name}.idx_pad"
    if _is_valid_padded_db(sibling):
        return str(sibling)

    # Case 3: passed path is a directory containing a padded DB.
    if db.is_dir():
        for cand in sorted(db.glob("*.idx_pad")):
            if _is_valid_padded_db(cand):
                return str(cand)

    return None


def _parse_m8_output(raw_output: str) -> pd.DataFrame:
    """Parse raw m8 tabular output string into a pandas DataFrame.

    Args:
        raw_output (str): Raw tab-separated m8 output text from MMseqs2.

    Returns:
        pd.DataFrame: pandas.DataFrame with columns: query, target, pident, evalue.
    """
    import pandas as pd

    col_names = ["query", "target", "pident", "evalue"]

    if not raw_output.strip():
        return pd.DataFrame(columns=col_names)

    try:
        df = pd.read_csv(io.StringIO(raw_output), sep="\t", header=None, names=col_names)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=col_names)
    except Exception as e:
        raise ValueError(f"mmseqs2-search-proteins: failed to parse m8 output ({len(raw_output)} bytes): {e}") from e

    return df


def _filter_top_hits(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to keep only the best hit per query sequence.

    Args:
        df (pd.DataFrame): DataFrame containing search results with 'query' and 'pident' columns.

    Returns:
        pd.DataFrame: DataFrame with one row per unique query sequence (best pident).
    """
    if df.empty:
        return df
    return df.loc[df.groupby("query")["pident"].idxmax()].reset_index(drop=True)


def _build_sequence_search_results(
    sequences: list[str], sequence_ids: list[str], df: pd.DataFrame
) -> list[Mmseqs2SequenceSearchResult]:
    """Build per-sequence search results from DataFrame.

    Args:
        sequences (list[str]): List of input sequences.
        sequence_ids (list[str]): List of sequence identifiers corresponding to sequences.
        df (pd.DataFrame): DataFrame with MMseqs2 search results.

    Returns:
        list[Mmseqs2SequenceSearchResult]: List of Mmseqs2SequenceSearchResult objects, one per input sequence.
    """
    results = []
    for seq, seq_id in zip(sequences, sequence_ids, strict=False):
        # Get hits for this sequence
        if not df.empty and "query" in df.columns:
            seq_df = df[df["query"] == seq_id]
            hits = [
                Mmseqs2Hit(
                    target_id=row["target"],
                    pident=row["pident"],
                    evalue=row["evalue"],
                )
                for _, row in seq_df.iterrows()
            ]
            # Sort by pident descending
            hits.sort(key=lambda h: h.pident, reverse=True)
        else:
            hits = []

        results.append(
            Mmseqs2SequenceSearchResult(
                query_id=seq_id,
                query_sequence=seq,
                hits=hits,
            )
        )

    return results
