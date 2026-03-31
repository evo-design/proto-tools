"""
proto_tools/tools/sequence_alignment/mafft/mafft.py

This module provides a standardized interface for MAFFT multiple sequence alignment.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional, Union

from pydantic import ConfigDict, Field, field_validator

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

    sequences: List[str] = InputField(
        description="List of sequences to align (minimum 2 required)",
    )
    sequence_ids: Optional[List[str]] = InputField(
        default=None,
        description="Optional sequence identifiers (defaults to seq_0, seq_1, ...)",
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def validate_sequences(cls, v):
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

    msa: MSA = Field(description="The multiple sequence alignment")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def output_format_options(self) -> List[str]:
        return ["fasta", "a3m"]

    @property
    def output_format_default(self) -> str:
        return "fasta"

    def _export_output(self, export_path: Union[Path, str], file_format: str):
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
        align_method (Literal['auto', 'localpair', 'globalpair', 'genafpair']): Alignment method to use:
            - "auto": Automatically select based on input (default)
            - "localpair": L-INS-i method (accurate, for <200 sequences)
            - "globalpair": G-INS-i method (for similar-length sequences)
            - "genafpair": E-INS-i method (for sequences with large unalignable regions)
        max_iterations (int): Maximum number of iterative refinement cycles.
            Higher values improve accuracy but increase runtime.
            Default is 0 (use align_method default).
        threads (int): Number of CPU threads for parallel processing.
    """

    align_method: Literal["auto", "localpair", "globalpair", "genafpair"] = ConfigField(
        title="Alignment Method",
        default="auto",
        description="Alignment method: auto, localpair (L-INS-i), globalpair (G-INS-i), or genafpair (E-INS-i)",
    )
    max_iterations: int = ConfigField(
        title="Maximum Iterations",
        default=0,
        ge=0,
        description="Maximum iterative refinement cycles (0 is default, higher values are more accurate)",
        advanced=True,
    )
    threads: int = ConfigField(
        title="Number of Threads",
        default=1,
        ge=1,
        description="Number of CPU threads for parallel processing",
        hidden=True,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input():
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
def run_mafft_align(inputs: MafftInput, config: MafftConfig | None = None, instance=None) -> MafftOutput:
    """Perform multiple sequence alignment using MAFFT.

    Aligns input sequences using MAFFT with the specified method and
    parameters. Returns aligned sequences with gap characters inserted
    to maximize alignment quality.

    Args:
        inputs (MafftInput): Validated input containing sequences to align.
        config (MafftConfig | None): Configuration with alignment parameters.

    Returns:
        MafftOutput: MSA result with alignment metadata.

    Raises:
        RuntimeError: If MAFFT command execution fails.
        FileNotFoundError: If MAFFT is not installed.

    Examples:
        >>> inputs = MafftInput(
        ...     sequences=["MVLSPADKTN", "MVLSAADKTN", "MVLTPADKTN"]
        ... )
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
    }

    input_data["device"] = "cpu"
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
            aligned_sequences_or_filepath=aligned_sequences,
            sequence_ids=sequence_ids,
        ),
    )
