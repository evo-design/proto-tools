"""MMseqs2 sequence clustering tool."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from bio_programming_tools.tools.infra.tool_cache import tool_cache
from bio_programming_tools.tools.infra.tool_io import BaseToolInput, BaseToolOutput
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.tools.utils import (
    BaseConfig,
    ConfigField,
    resolve_sequence_ids,
)

from .search_proteins import DEFAULT_MIN_SEQ_ID


# ============================================================================
# Data Models
# ============================================================================
# Shared:
class MmseqsClusterMember(BaseModel):
    """A member of an MMseqs2 cluster.

    Represents a sequence that belongs to a cluster, either as the representative
    or as a member.

    Attributes:
        sequence_id (str): Internal identifier (e.g., 'seq_0').
        sequence (str): The actual sequence string.
        is_representative (bool): Whether this sequence is the cluster representative.
    """
    sequence_id: str = Field(description="Sequence identifier")
    sequence: str = Field(description="The sequence string")
    is_representative: bool = Field(default=False, description="Whether this is the cluster representative")


class MmseqsClusterResult(BaseModel):
    """Clustering result for a single input sequence.

    Contains information about which cluster the sequence belongs to and
    whether it is the representative of that cluster.

    Attributes:
        sequence_id (str): Identifier of the input sequence.
        input_sequence (str): The original input sequence.
        cluster_id (str): Identifier of the cluster (usually the representative's ID).
        is_representative (bool): Whether this sequence is the cluster representative.
    """
    sequence_id: str = Field(description="Input sequence identifier")
    input_sequence: str = Field(description="The original input sequence")
    cluster_id: str = Field(description="Cluster identifier (representative sequence ID)")
    is_representative: bool = Field(default=False, description="Whether this sequence is the cluster representative")


# Input:
class MmseqsClusteringInput(BaseToolInput):
    """Input object for MMseqs2 sequence clustering.

    Attributes:
        input_sequences (List[str]): List of sequence strings (protein or nucleotide)
            for clustering.
        sequence_ids (Optional[List[str]]): Optional list of sequence identifiers.
            If not provided, sequences are assigned sequential IDs (seq_0, seq_1, ...).
    """

    input_sequences: List[str] = Field(
        description="List of sequences to cluster",
    )
    sequence_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional sequence identifiers (defaults to seq_0, seq_1, ...)",
    )

    @field_validator("input_sequences", mode="before")
    @classmethod
    def validate_input_sequences(cls, v):
        """Validate input sequences."""
        if not isinstance(v, list):
            raise ValueError(f"input_sequences must be a list, got {type(v)}")
        if not v:
            raise ValueError("input_sequences list cannot be empty")
        if not all(isinstance(item, str) for item in v):
            raise ValueError("All items in input_sequences list must be strings")
        return v


# Output:
class MmseqsClusteringOutput(BaseToolOutput):
    """Output from MMseqs2 sequence clustering.

    Contains per-sequence clustering results matching the input order.

    Attributes:
        results (List[MmseqsClusterResult]): List of clustering results, one per
            input sequence. The order matches the input sequences order.
    """
    results: List[MmseqsClusterResult] = Field(
        description="List of clustering results, one per input sequence"
    )

    def __len__(self) -> int:
        """Get the number of results."""
        return len(self.results)

    def __getitem__(self, idx: int) -> MmseqsClusterResult:
        """Get a result by index."""
        return self.results[idx]

    def __iter__(self) -> Iterator[MmseqsClusterResult]:
        """Iterate over the results."""
        return iter(self.results)

    @property
    def num_clusters(self) -> int:
        """Total number of unique clusters found."""
        return len(set(r.cluster_id for r in self.results))

    @property
    def representative_indices(self) -> List[int]:
        """Get indices of sequences that are cluster representatives."""
        return [i for i, r in enumerate(self.results) if r.is_representative]

    @property
    def output_format_options(self) -> List[str]:
        return ["csv", "json"]

    @property
    def output_format_default(self) -> str:
        return "csv"

    def _export_output(self, export_path: str | Path, file_format: str):
        path = Path(export_path).with_suffix(f".{file_format}")

        data = [r.model_dump() for r in self.results]
        df = pd.DataFrame(data) if data else pd.DataFrame(columns=["sequence_id", "input_sequence", "cluster_id", "is_representative"])

        if file_format == "csv":
            df.to_csv(path, index=False)

        elif file_format == "json":
            import json
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# Config:
class MmseqsClusteringConfig(BaseConfig):
    """Configuration object for MMseqs2 sequence clustering.

    Attributes:
        min_seq_id (float): Minimum sequence identity threshold (0.0-1.0) for
            grouping sequences into the same cluster.
    """
    min_seq_id: float = ConfigField(
        title="Minimum Sequence Identity",
        default=DEFAULT_MIN_SEQ_ID,
        ge=0.0,
        le=1.0,
        description="Minimum sequence identity threshold for clustering (0.0-1.0)",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
@tool_cache(tool_name="mmseqs-clustering")
@tool(
    key="mmseqs-clustering",
    label="MMseqs Clustering",
    input=MmseqsClusteringInput,
    config=MmseqsClusteringConfig,
    output=MmseqsClusteringOutput,
    description="Perform sequence clustering using MMseqs2 to reduce redundancy",
)
def run_mmseqs_clustering(
    inputs: MmseqsClusteringInput, config: MmseqsClusteringConfig
) -> MmseqsClusteringOutput:
    """Perform sequence clustering using MMseqs2.

    Groups similar sequences based on sequence identity threshold and returns
    per-sequence cluster assignments.

    Args:
        inputs (MmseqsClusteringInput): Validated input containing sequences
            to cluster.
        config (MmseqsClusteringConfig): Configuration with clustering threshold.

    Returns:
        MmseqsClusteringOutput: Per-sequence cluster assignments in input order.

    Raises:
        RuntimeError: If any MMseqs2 command fails during execution.

    Examples:
        >>> inputs = MmseqsClusteringInput(
        ...     input_sequences=["MVLSPADKTN...", "MVLSPADKTN...", "MKLLVVAAAA..."]
        ... )
        >>> config = MmseqsClusteringConfig(min_seq_id=0.95)
        >>> result = run_mmseqs_clustering(inputs, config)
        >>> print(f"Found {result.num_clusters} clusters")
        >>> for i, r in enumerate(result):
        ...     print(f"Seq {i}: cluster={r.cluster_id}, rep={r.is_representative}")
    """
    from bio_programming_tools.tools.infra.env_manager import EnvManager

    sequences = inputs.input_sequences
    sequence_ids = resolve_sequence_ids(sequences, inputs.sequence_ids)
    num_sequences = len(sequences)

    venv_manager = EnvManager(model_name="mmseqs")

    input_data = {
        "operation": "clustering",
        "sequences": sequences,
        "sequence_ids": sequence_ids,
        "min_seq_id": config.min_seq_id,
    }

    output_data = venv_manager.call_standalone_script_in_venv(
        script_path=Path(__file__).parent / "standalone" / "run.py",
        input_dict=input_data,
        device="cpu",
    )

    # Parse cluster assignments
    cluster_assignments = output_data["cluster_assignments"]

    # Build per-sequence results
    results = []
    for seq, seq_id in zip(sequences, sequence_ids):
        cluster_id = cluster_assignments.get(seq_id, seq_id)
        is_rep = (seq_id == cluster_id)
        results.append(MmseqsClusterResult(
            sequence_id=seq_id,
            input_sequence=seq,
            cluster_id=cluster_id,
            is_representative=is_rep,
        ))

    return MmseqsClusteringOutput(
        metadata={
            "min_seq_id": config.min_seq_id,
            "num_sequences": num_sequences,
        },
        results=results,
    )
