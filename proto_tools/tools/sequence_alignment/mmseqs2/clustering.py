"""MMseqs2 sequence clustering tool."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from proto_tools.tools.sequence_alignment.mmseqs2.search_proteins import (
    DEFAULT_CLUSTER_COVERAGE,
    DEFAULT_CLUSTER_EVALUE,
    DEFAULT_CLUSTER_MAX_SEQS,
    DEFAULT_CLUSTER_MODE,
    DEFAULT_CLUSTER_SENSITIVITY,
    DEFAULT_COV_MODE,
    ClusterMode,
    CovMode,
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

DEFAULT_CLUSTER_MIN_SEQ_ID = 0.60


# ============================================================================
# Data Models
# ============================================================================
# Shared:
class Mmseqs2ClusterMember(BaseModel):
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


class Mmseqs2ClusterResult(BaseModel):
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
class Mmseqs2ClusteringInput(BaseToolInput):
    """Input object for MMseqs2 sequence clustering.

    Attributes:
        input_sequences (list[str]): List of sequence strings (protein or nucleotide)
            for clustering.
        sequence_ids (list[str] | None): Optional list of sequence identifiers.
            If not provided, sequences are assigned sequential IDs (seq_0, seq_1, ...).
    """

    input_sequences: list[str] = InputField(
        description="List of sequences to cluster",
    )
    sequence_ids: list[str] | None = InputField(
        default=None,
        description="Optional sequence identifiers (defaults to seq_0, seq_1, ...)",
    )

    @field_validator("input_sequences", mode="before")
    @classmethod
    def validate_input_sequences(cls, v: Any) -> Any:
        """Validate input sequences."""
        if not isinstance(v, list):
            raise ValueError(f"input_sequences must be a list, got {type(v)}")
        if not v:
            raise ValueError("input_sequences list cannot be empty")
        if not all(isinstance(item, str) for item in v):
            raise ValueError("All items in input_sequences list must be strings")
        return v


# Output:
class Mmseqs2ClusteringOutput(BaseToolOutput):
    """Output from MMseqs2 sequence clustering.

    Contains per-sequence clustering results matching the input order.

    Attributes:
        results (list[Mmseqs2ClusterResult]): List of clustering results, one per
            input sequence. The order matches the input sequences order.
    """

    results: list[Mmseqs2ClusterResult] = Field(description="List of clustering results, one per input sequence")

    def __len__(self) -> int:
        """Get the number of results."""
        return len(self.results)

    def __getitem__(self, idx: int) -> Mmseqs2ClusterResult:
        """Get a result by index."""
        return self.results[idx]

    def __iter__(self) -> Iterator[Mmseqs2ClusterResult]:  # type: ignore[override]
        """Iterate over the results."""
        return iter(self.results)

    @property
    def num_clusters(self) -> int:
        """Total number of unique clusters found."""
        return len({r.cluster_id for r in self.results})

    @property
    def representative_indices(self) -> list[int]:
        """Get indices of sequences that are cluster representatives."""
        return [i for i, r in enumerate(self.results) if r.is_representative]

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

        path = Path(export_path).with_suffix(f".{file_format}")

        data = [r.model_dump() for r in self.results]
        df = (
            pd.DataFrame(data)
            if data
            else pd.DataFrame(columns=["sequence_id", "input_sequence", "cluster_id", "is_representative"])
        )

        if file_format == "csv":
            df.to_csv(path, index=False)

        elif file_format == "json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# Config:
class Mmseqs2ClusteringConfig(BaseConfig):
    """Configuration object for MMseqs2 sequence clustering.

    Attributes:
        min_seq_id (float): Min identity (0.0-1.0) to share a cluster.
            Wrapper default 0.6 (upstream MMseqs2 = 0.0).
        coverage (float): Minimum aligned-residue fraction (0.0-1.0); semantics
            depend on ``cov_mode``.
        cov_mode (CovMode): 0=query AND target, 1=target, 2=query,
            3-5=length-ratio variants.
        evalue (float): E-value threshold for the prefilter step.
        cluster_mode (ClusterMode): 0=Set-Cover (greedy), 1=Connected component
            (BLASTclust), 2-3=Greedy by length (CD-HIT).
        max_seqs (int): Max prefilter results per query.
        sensitivity (float): Prefilter sensitivity (1.0-7.5).
        extra_args (list[str]): Verbatim ``mmseqs cluster`` CLI tokens for
            niche flags (e.g. ``["--similarity-type", "2"]``).
    """

    min_seq_id: float = ConfigField(
        title="Minimum Sequence Identity",
        default=DEFAULT_CLUSTER_MIN_SEQ_ID,
        ge=0.0,
        le=1.0,
        description="Min identity (0-1) to share a cluster. Wrapper defaults 0.6; upstream = 0.0.",
    )
    coverage: float = ConfigField(
        title="Coverage Threshold",
        default=DEFAULT_CLUSTER_COVERAGE,
        ge=0.0,
        le=1.0,
        description="Minimum aligned-residue fraction (0-1); semantics depend on `cov_mode`.",
        advanced=True,
    )
    cov_mode: CovMode = ConfigField(
        title="Coverage Mode",
        default=DEFAULT_COV_MODE,
        description=("How `coverage` is measured: 0=query AND target, 1=target, 2=query, 3-5=length-ratio variants."),
        advanced=True,
    )
    evalue: float = ConfigField(
        title="E-value Threshold",
        default=DEFAULT_CLUSTER_EVALUE,
        gt=0.0,
        description="E-value threshold for the prefilter step; raise to keep weaker matches.",
        advanced=True,
    )
    cluster_mode: ClusterMode = ConfigField(
        title="Cluster Mode",
        default=DEFAULT_CLUSTER_MODE,
        description="0=Set-Cover greedy, 1=Connected component (BLASTclust), 2-3=Greedy by length (CD-HIT).",
        advanced=True,
    )
    max_seqs: int = ConfigField(
        title="Max Prefilter Hits",
        default=DEFAULT_CLUSTER_MAX_SEQS,
        ge=1,
        description="Max prefilter results per query; raise for deeper clustering at the cost of runtime/memory.",
        advanced=True,
    )
    sensitivity: float = ConfigField(
        title="Search Sensitivity",
        default=DEFAULT_CLUSTER_SENSITIVITY,
        ge=1.0,
        le=7.5,
        description="Prefilter sensitivity (1.0-7.5); higher is slower but finds more remote homologs.",
        advanced=True,
    )
    extra_args: list[str] = ConfigField(
        title="Extra CLI Arguments",
        default=[],
        description="Verbatim `mmseqs cluster` CLI tokens for niche flags (e.g. `['--similarity-type', '2']`).",
        advanced=True,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return Mmseqs2ClusteringInput(input_sequences=["MKTL", "MKTL", "ARND"])


@tool(
    key="mmseqs2-clustering",
    label="MMseqs2 Clustering",
    category="sequence_alignment",
    input_class=Mmseqs2ClusteringInput,
    config_class=Mmseqs2ClusteringConfig,
    output_class=Mmseqs2ClusteringOutput,
    description="Perform sequence clustering using MMseqs2 to reduce redundancy",
    example_input=example_input,
    cacheable=True,
)
def run_mmseqs2_clustering(
    inputs: Mmseqs2ClusteringInput,
    config: Mmseqs2ClusteringConfig,
    instance: Any = None,
) -> Mmseqs2ClusteringOutput:
    """Perform sequence clustering using MMseqs2.

    Groups similar sequences based on sequence identity threshold and returns
    per-sequence cluster assignments.

    Args:
        inputs (Mmseqs2ClusteringInput): Validated input containing sequences
            to cluster.
        config (Mmseqs2ClusteringConfig): Configuration with clustering threshold.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        Mmseqs2ClusteringOutput: Per-sequence cluster assignments in input order.

    Raises:
        RuntimeError: If any MMseqs2 command fails during execution.

    Examples:
        >>> inputs = Mmseqs2ClusteringInput(input_sequences=["MVLSPADKTN...", "MVLSPADKTN...", "MKLLVVAAAA..."])
        >>> config = Mmseqs2ClusteringConfig(min_seq_id=0.95)
        >>> result = run_mmseqs2_clustering(inputs, config)
        >>> print(f"Found {result.num_clusters} clusters")
        >>> for i, r in enumerate(result):
        ...     print(f"Seq {i}: cluster={r.cluster_id}, rep={r.is_representative}")
    """
    sequences = inputs.input_sequences
    sequence_ids = resolve_sequence_ids(sequences, inputs.sequence_ids)
    num_sequences = len(sequences)

    output_data = ToolInstance.dispatch(
        "mmseqs2",
        {
            "device": "cpu",
            "operation": "clustering",
            "sequences": sequences,
            "sequence_ids": sequence_ids,
            "min_seq_id": config.min_seq_id,
            "coverage": config.coverage,
            "cov_mode": config.cov_mode,
            "evalue": config.evalue,
            "cluster_mode": config.cluster_mode,
            "max_seqs": config.max_seqs,
            "sensitivity": config.sensitivity,
            "extra_args": list(config.extra_args),
        },
        instance=instance,
        config=config,
    )

    # Parse cluster assignments
    cluster_assignments = output_data["cluster_assignments"]

    # Build per-sequence results
    results = []
    for seq, seq_id in zip(sequences, sequence_ids, strict=False):
        cluster_id = cluster_assignments.get(seq_id, seq_id)
        is_rep = seq_id == cluster_id
        results.append(
            Mmseqs2ClusterResult(
                sequence_id=seq_id,
                input_sequence=seq,
                cluster_id=cluster_id,
                is_representative=is_rep,
            )
        )

    return Mmseqs2ClusteringOutput(
        metadata={
            "min_seq_id": config.min_seq_id,
            "coverage": config.coverage,
            "cov_mode": config.cov_mode,
            "evalue": config.evalue,
            "cluster_mode": config.cluster_mode,
            "max_seqs": config.max_seqs,
            "sensitivity": config.sensitivity,
            "num_sequences": num_sequences,
        },
        results=results,
    )
