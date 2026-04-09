"""proto_tools/tools/gene_annotation/mmseqs/search_proteins.py.

Also defines shared data models (MmseqsHit, MmseqsSequenceSearchResult),
constants, and helper functions used by all MMseqs2 search tools.
"""

import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator

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

DEFAULT_THREADS = 96
DEFAULT_SPLIT = 0
DEFAULT_SENSITIVITY = 4.0
DEFAULT_GENOME_SENSITIVITY = 7.5
DEFAULT_MIN_SEQ_ID = 0.60
SEARCH_TYPE_NUCLEOTIDE = 3

# Standard column formats for MMseqs2 output
M8_COLUMNS = ["query", "target", "pident", "evalue"]


# ============================================================================
# Data Models
# ============================================================================
# Shared:
class MmseqsHit(BaseModel):
    """A single MMseqs2 search hit.

    Represents a single alignment between a query sequence and a target sequence
    in the database.

    Attributes:
        target_id (str): Identifier of the target (database) sequence.
        pident (float): Percentage of identical matches (0-100).
        evalue (float): Expected value (E-value) - statistical significance.
    """

    target_id: str = Field(description="Target sequence identifier from the database")
    pident: float = Field(description="Percentage identity (0-100)")
    evalue: float = Field(description="E-value (expected number of chance matches)")


class MmseqsSequenceSearchResult(BaseModel):
    """Results for a single query sequence from an MMseqs2 search.

    Contains all hits found for one query sequence, along with the query
    sequence itself for reference.

    Attributes:
        query_id (str): Identifier of the query sequence.
        query_sequence (str): The input query sequence.
        hits (list[MmseqsHit]): All hits found for this query, sorted by pident descending.
    """

    query_id: str = Field(description="Query sequence identifier")
    query_sequence: str = Field(description="The input query sequence")
    hits: list[MmseqsHit] = Field(default_factory=list, description="List of hits for this query")

    @property
    def top_hit(self) -> MmseqsHit | None:
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
class MmseqsSearchProteinsInput(BaseToolInput):
    """Input object for MMseqs2 protein search.

    This class defines the input parameters for running MMseqs2 protein sequence
    searches.

    Attributes:
        query_sequences (list[str]): List of protein sequence strings (amino acid
            sequences) to search.
        sequence_ids (list[str] | None): Optional list of sequence identifiers.
            If not provided, sequences are assigned sequential IDs (seq_0, seq_1, ...).
        mmseqs_db (str): Path to the target database for searching. Can be:
            - Path to a FASTA file (MMseqs2 will create a temporary database)
            - Path to a pre-built MMseqs2 database (created with ``mmseqs createdb``)
    """

    query_sequences: list[str] = InputField(
        description="List of protein sequences to search",
    )
    sequence_ids: list[str] | None = InputField(
        default=None,
        description="Optional sequence identifiers (defaults to seq_0, seq_1, ...)",
    )
    mmseqs_db: str = InputField(description="Path to target database (FASTA file or MMseqs2 database)")

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

    @field_validator("mmseqs_db")
    @classmethod
    def validate_mmseqs_db(cls, v: str) -> str:
        """Validate that MMseqs2 database exists."""
        if not Path(v).exists():
            raise ValueError(f"MMseqs2 database not found: {v}")
        return v


# Output:
class MmseqsSearchProteinsOutput(BaseToolOutput):
    """Output from MMseqs2 protein search.

    Contains per-sequence search results matching the input order.

    Attributes:
        results (list[MmseqsSequenceSearchResult]): List of search results, one per
            input sequence. The order matches the input sequences order.
    """

    results: list[MmseqsSequenceSearchResult] = Field(description="List of search results, one per input sequence")

    def __len__(self) -> int:
        """Get the number of results."""
        return len(self.results)

    def __getitem__(self, idx: int) -> MmseqsSequenceSearchResult:
        """Get a result by index."""
        return self.results[idx]

    def __iter__(self) -> Iterator[MmseqsSequenceSearchResult]:  # type: ignore[override]
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
class MmseqsSearchProteinsConfig(BaseConfig):
    """Configuration object for MMseqs2 protein search.

    Attributes:
        threads (int): Number of CPU threads for parallel processing.
        split (int): Memory management mode (0=auto).
        sensitivity (float): Search sensitivity (1.0=fast, 7.5=very sensitive).
        only_top_hits (bool): If True, keep only the best hit per query sequence.
    """

    threads: int = ConfigField(
        title="Number of Threads",
        default=DEFAULT_THREADS,
        ge=1,
        description="Number of CPU threads for parallel processing",
        hidden=True,
    )
    split: int = ConfigField(
        title="Split",
        default=DEFAULT_SPLIT,
        ge=0,
        description="Memory management mode (0=auto, higher uses less memory)",
    )
    sensitivity: float = ConfigField(
        title="Search Sensitivity",
        default=DEFAULT_SENSITIVITY,
        ge=1.0,
        le=7.5,
        description="Search sensitivity (1.0=fast, 7.5=very sensitive)",
        advanced=True,
    )
    only_top_hits: bool = ConfigField(
        title="Only Top Hits",
        default=True,
        description="If True, keep only the best hit per query sequence",
        advanced=True,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return MmseqsSearchProteinsInput(
        query_sequences=["MKTL"],
        mmseqs_db=str(Path(__file__).parent / "examples" / "example.fasta"),
    )


@tool(
    key="mmseqs-search-proteins",
    label="MMseqs2 Protein Search",
    category="gene_annotation",
    input_class=MmseqsSearchProteinsInput,
    config_class=MmseqsSearchProteinsConfig,
    output_class=MmseqsSearchProteinsOutput,
    description="Search protein sequences using MMseqs2 with per-sequence results",
    example_input=example_input,
    iterable_input_field="query_sequences",
    iterable_output_field="results",
    cacheable=True,
)
def run_mmseqs_search_proteins(
    inputs: MmseqsSearchProteinsInput,
    config: MmseqsSearchProteinsConfig,
    instance: Any = None,
) -> MmseqsSearchProteinsOutput:
    """Perform protein sequence search using MMseqs2.

    Searches query protein sequences against a target database and returns
    per-sequence results with all hits.

    Args:
        inputs (MmseqsSearchProteinsInput): Validated input containing query
            sequences and target database path.
        config (MmseqsSearchProteinsConfig): Configuration with search parameters.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        MmseqsSearchProteinsOutput: Per-sequence search results in input order.

    Raises:
        FileNotFoundError: If target database cannot be found.
        RuntimeError: If MMseqs2 command execution fails.

    Examples:
        >>> inputs = MmseqsSearchProteinsInput(
        ...     query_sequences=["MSKGEELFT", "MVLSPADKTN"], mmseqs_db="/path/to/protein_db"
        ... )
        >>> config = MmseqsSearchProteinsConfig()
        >>> result = run_mmseqs_search_proteins(inputs, config)
        >>> print(f"Found {result[0].num_hits} hits for first sequence")
        >>> if result[0].top_hit:
        ...     print(f"Top hit: {result[0].top_hit.pident}% identity")
    """
    sequences = inputs.query_sequences
    sequence_ids = resolve_sequence_ids(sequences, inputs.sequence_ids)
    num_sequences = len(sequences)

    output_data = ToolInstance.dispatch(
        "mmseqs",
        {
            "device": "cpu",
            "operation": "protein_search",
            "sequences": sequences,
            "sequence_ids": sequence_ids,
            "mmseqs_db": inputs.mmseqs_db,
            "threads": config.threads,
            "split": config.split,
            "sensitivity": config.sensitivity,
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

    return MmseqsSearchProteinsOutput(
        metadata={
            "mmseqs_db": inputs.mmseqs_db,
            "threads": config.threads,
            "sensitivity": config.sensitivity,
            "only_top_hits": config.only_top_hits,
            "num_sequences": num_sequences,
        },
        results=results,
    )


# ============================================================================
# Helper Functions
# ============================================================================
def _parse_m8_output(raw_output: str) -> pd.DataFrame:
    """Parse raw m8 tabular output string into a pandas DataFrame.

    Args:
        raw_output (str): Raw tab-separated m8 output text from MMseqs2.

    Returns:
        pd.DataFrame: pandas.DataFrame with columns: query, target, pident, evalue.
    """
    col_names = ["query", "target", "pident", "evalue"]

    if not raw_output.strip():
        return pd.DataFrame(columns=col_names)

    try:
        df = pd.read_csv(io.StringIO(raw_output), sep="\t", header=None, names=col_names)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=col_names)
    except Exception as e:
        raise ValueError(f"Failed to parse m8 output: {e}") from e

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
) -> list[MmseqsSequenceSearchResult]:
    """Build per-sequence search results from DataFrame.

    Args:
        sequences (list[str]): List of input sequences.
        sequence_ids (list[str]): List of sequence identifiers corresponding to sequences.
        df (pd.DataFrame): DataFrame with MMseqs2 search results.

    Returns:
        list[MmseqsSequenceSearchResult]: List of MmseqsSequenceSearchResult objects, one per input sequence.
    """
    results = []
    for seq, seq_id in zip(sequences, sequence_ids, strict=False):
        # Get hits for this sequence
        if not df.empty and "query" in df.columns:
            seq_df = df[df["query"] == seq_id]
            hits = [
                MmseqsHit(
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
            MmseqsSequenceSearchResult(
                query_id=seq_id,
                query_sequence=seq,
                hits=hits,
            )
        )

    return results
