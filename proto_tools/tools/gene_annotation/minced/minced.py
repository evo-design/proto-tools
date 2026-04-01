"""proto_tools/tools/gene_annotation/minced/minced.py.

This module provides a standardized interface for detecting CRISPR arrays
in nucleotide sequences using MinCED (Mining CRISPRs in Environmental Datasets),
a tool for finding CRISPR repeats and spacers in genomic sequences.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
# Data Models
# ============================================================================
class CrisprRepeatSpacer(BaseModel):
    """A single repeat-spacer unit within a CRISPR array."""

    position: int = Field(description="Position of the repeat in the sequence")
    repeat: str = Field(description="Repeat sequence")
    spacer: str | None = Field(default=None, description="Spacer sequence (None for last repeat)")
    repeat_length: int | None = Field(default=None, description="Length of the repeat")
    spacer_length: int | None = Field(default=None, description="Length of the spacer")


class CrisprArray(BaseModel):
    """A single CRISPR array detected in a sequence."""

    repeats_and_spacers: list[CrisprRepeatSpacer] = Field(
        default_factory=list,
        description="List of repeat-spacer units in this CRISPR array",
    )

    @property
    def num_repeats(self) -> int:
        """Number of repeats in this CRISPR array."""
        return len(self.repeats_and_spacers)

    @property
    def spacers(self) -> list[str]:
        """Extract spacer sequences from this array."""
        return [rs.spacer for rs in self.repeats_and_spacers if rs.spacer is not None and rs.spacer.strip()]


class MincedSequenceResult(BaseModel):
    """MinCED results for a single input sequence."""

    sequence_id: str = Field(description="ID of the input sequence")
    crispr_arrays: list[CrisprArray] = Field(
        default_factory=list,
        description="CRISPR arrays detected in this sequence",
    )

    @property
    def has_crispr(self) -> bool:
        """Whether any CRISPR arrays were detected."""
        return len(self.crispr_arrays) > 0

    @property
    def num_arrays(self) -> int:
        """Number of CRISPR arrays detected."""
        return len(self.crispr_arrays)


# Input:
class MincedInput(BaseToolInput):
    """Input for MinCED CRISPR array detection.

    Attributes:
        sequences (list[str]): Nucleotide sequence(s) to search for CRISPR arrays.
        sequence_ids (list[str] | None): Optional sequence identifiers.
    """

    sequences: list[str] = InputField(description="Nucleotide sequence(s) to search for CRISPR arrays")
    sequence_ids: list[str] | None = InputField(
        default=None,
        description="Optional sequence identifiers (defaults to seq_0, seq_1, ...)",
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, value: Any) -> list[str]:
        """Normalize a single sequence to a list."""
        if isinstance(value, str):
            return [value]
        return value  # type: ignore[no-any-return]


# Output:
class MincedOutput(BaseToolOutput):
    """Output from MinCED CRISPR array detection.

    Attributes:
        results (list[MincedSequenceResult]): Per-sequence CRISPR detection results.
    """

    results: list[MincedSequenceResult] = Field(
        default_factory=list,
        description="Per-sequence CRISPR array detection results",
    )

    @property
    def num_sequences_with_crispr(self) -> int:
        """Number of sequences with at least one CRISPR array."""
        return sum(1 for r in self.results if r.has_crispr)

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["csv", "json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        import pandas as pd

        path = Path(export_path).with_suffix(f".{file_format}")
        rows = [
            {
                "sequence_id": result.sequence_id,
                "array_index": arr_idx,
                "position": rs.position,
                "repeat": rs.repeat,
                "spacer": rs.spacer,
                "repeat_length": rs.repeat_length,
                "spacer_length": rs.spacer_length,
            }
            for result in self.results
            for arr_idx, array in enumerate(result.crispr_arrays)
            for rs in array.repeats_and_spacers
        ]
        df = pd.DataFrame(rows)
        if file_format == "csv":
            df.to_csv(path, index=False)
        elif file_format == "json":
            df.to_json(path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# Config:
class MincedConfig(BaseConfig):
    """Configuration for MinCED CRISPR array detection.

    Attributes:
        min_num_repeats (int): Minimum number of repeats in a CRISPR array.
            Default: 3.
        min_repeat_length (int): Minimum length of a repeat sequence.
            Default: 27.
    """

    min_num_repeats: int = ConfigField(
        title="Minimum Number of Repeats",
        default=3,
        ge=2,
        description="Minimum number of repeats required for a CRISPR array",
    )
    min_repeat_length: int = ConfigField(
        title="Minimum Repeat Length",
        default=27,
        ge=10,
        description="Minimum length of a repeat sequence in nucleotides",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return MincedInput(sequences=["ATCGATCG"])


@tool(
    key="minced-crispr",
    label="MinCED CRISPR Array Detection",
    category="gene_annotation",
    input_class=MincedInput,
    config_class=MincedConfig,
    output_class=MincedOutput,
    description="Detect CRISPR arrays in nucleotide sequences using MinCED",
    example_input=example_input,
    iterable_input_field="sequences",
    iterable_output_field="results",
    cacheable=True,
)
def run_minced(inputs: MincedInput, config: MincedConfig | None = None, instance: Any = None) -> MincedOutput:
    """Detect CRISPR arrays in nucleotide sequences using MinCED.

    Uses MinCED (Mining CRISPRs in Environmental Datasets) to identify
    CRISPR repeats and spacers in input nucleotide sequences. This is
    used as a Stage 1 filter in the Cas9 filtering pipeline to confirm
    that candidate sequences contain functional CRISPR loci.

    Args:
        inputs (MincedInput): Validated input containing nucleotide sequences.
        config (MincedConfig | None): MinCED configuration with minimum repeat count
            and length thresholds.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        MincedOutput: Per-sequence CRISPR array detection results.

    Examples:
        >>> inputs = MincedInput(sequences=["ATCG..." * 1000])
        >>> config = MincedConfig(min_num_repeats=3, min_repeat_length=27)
        >>> result = run_minced(inputs, config)
        >>> print(f"{result.num_sequences_with_crispr} sequences have CRISPR arrays")
    """
    sequence_ids = resolve_sequence_ids(inputs.sequences, inputs.sequence_ids)

    input_data = {
        "sequences": inputs.sequences,
        "sequence_ids": sequence_ids,
        "config": {
            "min_num_repeats": config.min_num_repeats,  # type: ignore[union-attr]
            "min_repeat_length": config.min_repeat_length,  # type: ignore[union-attr]
        },
    }

    input_data["device"] = "cpu"
    output_data = ToolInstance.dispatch(
        "minced",
        input_data,
        instance=instance,
        config=config,
    )

    results = []
    for result_dict in output_data["results"]:
        arrays = []
        for array_dict in result_dict["crispr_arrays"]:
            rs_list = [CrisprRepeatSpacer(**rs) for rs in array_dict["repeats_and_spacers"]]
            arrays.append(CrisprArray(repeats_and_spacers=rs_list))
        results.append(
            MincedSequenceResult(
                sequence_id=result_dict["sequence_id"],
                crispr_arrays=arrays,
            )
        )

    return MincedOutput(
        metadata={
            "min_num_repeats": config.min_num_repeats,  # type: ignore[union-attr]
            "min_repeat_length": config.min_repeat_length,  # type: ignore[union-attr]
            "num_sequences": len(inputs.sequences),
        },
        results=results,
    )
