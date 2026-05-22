"""proto_tools/tools/sequence_alignment/mafft/mafft.py.

This module provides a standardized interface for MAFFT multiple sequence alignment.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from proto_tools.tools.sequence_alignment.msas import MSA
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
class MafftInput(BaseToolInput):
    """Input object for MAFFT multiple sequence alignment.

    This class defines the input parameters for running MAFFT alignment.

    Attributes:
        sequences (list[str]): List of sequence strings (protein or nucleotide)
            to align. At least 2 sequences are required for alignment.
        sequence_ids (list[str] | None): Optional list of sequence identifiers.
            If not provided, sequences are assigned sequential IDs (seq_0, seq_1, ...).
    """

    sequences: list[str] = InputField(
        title="Sequences",
        description="List of sequences to align (minimum 2 required)",
    )
    sequence_ids: list[str] | None = InputField(
        default=None,
        title="Sequence IDs",
        description="Optional sequence identifiers (defaults to seq_0, seq_1, ...)",
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def validate_sequences(cls, v: Any) -> Any:
        """Validate sequences input."""
        if not isinstance(v, list):
            raise ValueError(f"sequences must be a list, got {type(v)}")
        if len(v) < 2:
            raise ValueError(f"At least 2 sequences are required for alignment, got {len(v)}")
        if not all(isinstance(item, str) for item in v):
            raise ValueError("All items in sequences list must be strings")
        if not all(len(item) > 0 for item in v):
            raise ValueError("All sequences must be non-empty strings")
        return v


# Output:
class MafftOutput(BaseToolOutput):
    """Output from MAFFT multiple sequence alignment.

    Attributes:
        msa (MSA): The multiple sequence alignment result containing aligned sequences,
            sequence IDs, and original unaligned sequences.
    """

    msa: MSA = Field(title="MSA Result", description="The multiple sequence alignment")

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["fasta", "a3m"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "fasta"

    def _export_output(self, export_path: Path | str, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")

        path.parent.mkdir(parents=True, exist_ok=True)

        if file_format == "fasta":
            self.msa.to_fasta_file(str(path))
        elif file_format == "a3m":
            self.msa.to_a3m_file(str(path))
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# Config:
class MafftConfig(BaseConfig):
    """Configuration object for MAFFT alignment.

    Attributes:
        align_method (Literal['auto', 'localpair', 'globalpair', 'genafpair']):
            ``"auto"`` (MAFFT picks by input size), ``"localpair"`` (L-INS-i),
            ``"globalpair"`` (G-INS-i), or ``"genafpair"`` (E-INS-i).
        max_iterations (int): Iterative-refinement cycles. ``0`` = no refinement;
            ~1000 enables the full *-INS-i pipelines with ``*pair`` methods.
        threads (int): Number of CPU threads for parallel processing.
        extra_args (list[str]): Verbatim ``mafft`` CLI tokens for niche flags
            (e.g. ``["--retree", "3", "--reorder"]``).
    """

    align_method: Literal["auto", "localpair", "globalpair", "genafpair"] = ConfigField(
        title="Alignment Method",
        default="auto",
        description="auto, localpair (L-INS-i), globalpair (G-INS-i), or genafpair (E-INS-i).",
    )
    max_iterations: int = ConfigField(
        title="Maximum Iterations",
        default=0,
        ge=0,
        description="Iterative-refinement cycles; 0 = no refinement; ~1000 enables full *-INS-i.",
    )
    threads: int = ConfigField(
        title="Number of Threads",
        default=1,
        ge=1,
        description="CPU threads for parallel processing.",
        include_in_key=False,
    )
    extra_args: list[str] = ConfigField(
        title="Extra CLI Arguments",
        default=[],
        description="Verbatim `mafft` CLI tokens for niche flags (e.g. `['--retree', '3', '--reorder']`).",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return MafftInput(sequences=["MKTL", "MKTA"])


@tool(
    key="mafft-align",
    label="MAFFT Alignment",
    category="sequence_alignment",
    input_class=MafftInput,
    config_class=MafftConfig,
    output_class=MafftOutput,
    description="Multiple sequence alignment (MSA) using MAFFT (Multiple Alignment using Fast Fourier Transform)",
    example_input=example_input,
    cacheable=True,
)
def run_mafft_align(inputs: MafftInput, config: MafftConfig, instance: Any = None) -> MafftOutput:
    """Perform multiple sequence alignment using MAFFT.

    Aligns input sequences using MAFFT with the specified method and
    parameters. Returns aligned sequences with gap characters inserted
    to maximize alignment quality.

    Args:
        inputs (MafftInput): Validated input containing sequences to align.
        config (MafftConfig): Configuration with alignment parameters.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        MafftOutput: MSA result with alignment metadata.

    Raises:
        RuntimeError: If MAFFT command execution fails.
        FileNotFoundError: If MAFFT is not installed.

    Examples:
        >>> inputs = MafftInput(sequences=["MVLSPADKTN", "MVLSAADKTN", "MVLTPADKTN"])
        >>> config = MafftConfig(align_method="auto")
        >>> result = run_mafft_align(inputs, config)
        >>> print(f"Alignment length: {result.msa.alignment_length}")
        >>> for i, seq in enumerate(result.msa):
        ...     print(f"{result.sequence_ids[i]}: {seq}")
    """
    sequences = inputs.sequences
    sequence_ids = resolve_sequence_ids(sequences, inputs.sequence_ids)
    num_sequences = len(sequences)

    # Prepare input data for standalone script
    input_data = {
        "sequences": sequences,
        "sequence_ids": sequence_ids,
        "align_method": config.align_method,
        "max_iterations": config.max_iterations,
        "threads": config.threads,
        "extra_args": list(config.extra_args),
    }

    input_data["device"] = "cpu"  # type: ignore[assignment]  # Literal mismatch: "cpu" not in declared Literal union
    output_data = ToolInstance.dispatch(
        "mafft",
        input_data,
        instance=instance,
        config=config,
    )

    # Extract results from output
    aligned_sequences = output_data["aligned_sequences"]

    return MafftOutput(
        metadata={
            "align_method": config.align_method,
            "max_iterations": config.max_iterations,
            "threads": config.threads,
            "num_sequences": num_sequences,
        },
        msa=MSA(
            aligned_sequences=aligned_sequences,
            sequence_ids=sequence_ids,
        ),
    )
