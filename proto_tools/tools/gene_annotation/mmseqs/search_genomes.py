"""proto_tools/tools/gene_annotation/mmseqs/search_genomes.py.

MMseqs2 genome-to-genome nucleotide search tool.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import Field, field_validator

from proto_tools.tools.gene_annotation.mmseqs.search_proteins import (
    DEFAULT_GENOME_SENSITIVITY,
    DEFAULT_THREADS,
    M8_COLUMNS,
    SEARCH_TYPE_NUCLEOTIDE,
    MmseqsSequenceSearchResult,
    _build_sequence_search_results,
    _parse_m8_output,
)
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
# Data Models
# ============================================================================
# Input:
class MmseqsSearchGenomesInput(BaseToolInput):
    """Input object for MMseqs2 genome search.

    Attributes:
        query_genomes (list[str]): List of nucleotide sequence strings (DNA/RNA)
            to use as queries.
        query_ids (list[str] | None): Optional list of query identifiers.
            If not provided, sequences are assigned sequential IDs (seq_0, seq_1, ...).
        target_genomes (list[str]): List of nucleotide sequence strings to search
            against.
        target_ids (list[str] | None): Optional list of target identifiers.
            If not provided, sequences are assigned sequential IDs (target_0, target_1, ...).
    """

    query_genomes: list[str] = InputField(description="List of query genome sequences (nucleotide)")
    query_ids: list[str] | None = InputField(
        default=None,
        description="Optional query identifiers (defaults to seq_0, seq_1, ...)",
    )
    target_genomes: list[str] = InputField(description="List of target genome sequences to search against")
    target_ids: list[str] | None = InputField(
        default=None,
        description="Optional target identifiers (defaults to target_0, target_1, ...)",
    )

    @field_validator("query_genomes", mode="before")
    @classmethod
    def validate_query_genomes(cls, v: Any) -> Any:
        """Validate query genomes input."""
        if not isinstance(v, list):
            raise ValueError(f"query_genomes must be a list, got {type(v)}")
        if not v:
            raise ValueError("query_genomes list cannot be empty")
        if not all(isinstance(item, str) for item in v):
            raise ValueError("All items in query_genomes list must be strings")
        return v

    @field_validator("target_genomes", mode="before")
    @classmethod
    def validate_target_genomes(cls, v: Any) -> Any:
        """Validate target genomes input."""
        if not isinstance(v, list):
            raise ValueError(f"target_genomes must be a list, got {type(v)}")
        if not v:
            raise ValueError("target_genomes list cannot be empty")
        if not all(isinstance(item, str) for item in v):
            raise ValueError("All items in target_genomes list must be strings")
        return v


# Output:
class MmseqsSearchGenomesOutput(BaseToolOutput):
    """Output from MMseqs2 genome search.

    Contains per-sequence search results matching the input query order.

    Attributes:
        results (list[MmseqsSequenceSearchResult]): List of search results, one per
            input query genome. The order matches the input query genomes order.
    """

    results: list[MmseqsSequenceSearchResult] = Field(description="List of search results, one per input query genome")

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
        """Total number of hits across all query genomes."""
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

            header = file_format != "m8"
            df.to_csv(path, sep=sep, index=False, header=header)

        elif file_format == "json":
            json_data = [r.model_dump() for r in self.results]
            with open(path, "w") as f:
                json.dump(json_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# Config:
class MmseqsSearchGenomesConfig(BaseConfig):
    """Configuration object for MMseqs2 genome-to-genome search.

    Attributes:
        search_type (int): MMseqs2 search type (3=nucleotide vs nucleotide).
        threads (int): Number of CPU threads for parallel processing.
        sensitivity (float): Search sensitivity (1.0=fast, 7.5=very sensitive).
    """

    search_type: int = ConfigField(
        title="Search Type",
        default=SEARCH_TYPE_NUCLEOTIDE,
        description="MMseqs2 search type (3=nucleotide vs nucleotide)",
        advanced=True,
    )
    threads: int = ConfigField(
        title="Number of Threads",
        default=DEFAULT_THREADS,
        ge=1,
        description="Number of CPU threads for parallel processing",
        hidden=True,
    )
    sensitivity: float = ConfigField(
        title="Search Sensitivity",
        default=DEFAULT_GENOME_SENSITIVITY,
        ge=1.0,
        le=7.5,
        description="Search sensitivity (7.5=very sensitive, default for genomes)",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return MmseqsSearchGenomesInput(
        query_genomes=["ATCGATCG"],
        target_genomes=["ATCGATCG", "GCTAGCTA"],
    )


@tool(
    key="mmseqs-search-genomes",
    label="MMseqs Genome Search",
    category="gene_annotation",
    input_class=MmseqsSearchGenomesInput,
    config_class=MmseqsSearchGenomesConfig,
    output_class=MmseqsSearchGenomesOutput,
    description="Execute nucleotide genome-to-genome search workflow",
    example_input=example_input,
    iterable_input_field="query_genomes",
    iterable_output_field="results",
    cacheable=True,
)
def run_mmseqs_search_genomes(
    inputs: MmseqsSearchGenomesInput,
    config: MmseqsSearchGenomesConfig,
    instance: Any = None,
) -> MmseqsSearchGenomesOutput:
    """Execute nucleotide genome-to-genome search workflow.

    Implements the full MMseqs2 nucleotide search pipeline including database
    creation, indexing, searching, and result conversion.

    Args:
        inputs (MmseqsSearchGenomesInput): Validated input containing query
            and target genome sequences.
        config (MmseqsSearchGenomesConfig): Configuration with search parameters.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        MmseqsSearchGenomesOutput: Per-sequence search results in query order.

    Raises:
        RuntimeError: If any MMseqs2 command fails during execution.

    Examples:
        >>> inputs = MmseqsSearchGenomesInput(
        ...     query_genomes=["ATGGTGCTGTCTCCT...", "ATGAAGCTGCTGGTG..."],
        ...     target_genomes=["ATGGTGCTGTCTCCT...", "ATGAAGCTGCTGGTG..."],
        ... )
        >>> config = MmseqsSearchGenomesConfig()
        >>> result = run_mmseqs_search_genomes(inputs, config)
        >>> for r in result:
        ...     print(f"Found {r.num_hits} hits")
    """
    query_sequences = inputs.query_genomes
    target_sequences = inputs.target_genomes
    query_ids = resolve_sequence_ids(query_sequences, inputs.query_ids)
    target_ids = resolve_sequence_ids(target_sequences, inputs.target_ids)
    num_queries = len(query_sequences)

    output_data = ToolInstance.dispatch(
        "mmseqs",
        {
            "device": "cpu",
            "operation": "genome_search",
            "query_sequences": query_sequences,
            "query_ids": query_ids,
            "target_sequences": target_sequences,
            "target_ids": target_ids,
            "search_type": config.search_type,
            "threads": config.threads,
            "sensitivity": config.sensitivity,
            "m8_columns": M8_COLUMNS,
        },
        instance=instance,
        config=config,
    )

    # Parse results
    raw_output = output_data["stdout"]
    df = _parse_m8_output(raw_output)

    # Build per-sequence results
    results = _build_sequence_search_results(query_sequences, query_ids, df)

    return MmseqsSearchGenomesOutput(
        metadata={
            "search_type": config.search_type,
            "threads": config.threads,
            "sensitivity": config.sensitivity,
            "num_queries": num_queries,
            "num_targets": len(target_sequences),
        },
        results=results,
    )
