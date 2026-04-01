"""proto_tools/tools/sequence_scoring/enformer/enformer_prediction.py.

Enformer sequence scoring tool.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
    return_invalid_nucleotide_chars,
)

logger = logging.getLogger(__name__)

# Enformer model constants
ENFORMER_CONTEXT = 196_608
ENFORMER_OUTPUT = 896


# ============================================================================
# Data Models
# ============================================================================
# Input:
class EnformerInput(BaseToolInput):
    """Input for Enformer regulatory activity prediction.

    Attributes:
        sequence (str): DNA sequence for Enformer inference. Must be exactly
            196,608 bp and only contain valid nucleotide characters.
    """

    sequence: str = InputField(description="DNA sequence to score")

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, sequence: str) -> str:
        """Validate and normalize nucleotide sequence; require length 196,608 bp."""
        if not sequence or not sequence.strip():
            raise ValueError("Sequence cannot be empty")
        sequence = sequence.upper()
        invalid_chars = return_invalid_nucleotide_chars(sequence, additional_valid_chars="N")
        if invalid_chars:
            raise ValueError(f"Invalid nucleotide characters in sequence: {', '.join(sorted(invalid_chars))}")
        if len(sequence) != ENFORMER_CONTEXT:
            raise ValueError(f"Input sequence must have length {ENFORMER_CONTEXT}, got {len(sequence)}")
        return sequence


# Output:
class EnformerOutput(BaseToolOutput):
    """Output from Enformer prediction.

    Attributes:
        sequence (str): Input DNA sequence that was scored.
        sequence_length (int): Length of the input sequence (always 196,608).
        prediction (list[list[float]]): Predicted signal matrix with shape
            ``[896, num_tracks]``.
        output_tracks (list[int]): Track indices that were extracted.
        species (str): Species used for prediction (``"human"`` or ``"mouse"``).
    """

    sequence: str = Field(description="Input DNA/RNA sequence")
    sequence_length: int = Field(description="Length of input sequence")
    prediction: list[list[float]] = Field(description="Predicted activity matrix with shape [896, num_tracks]")
    output_tracks: list[int] = Field(description="Track indices extracted from Enformer")
    species: str = Field(description="Species used for prediction ('human' or 'mouse')")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json", "csv"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: Path | str, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")
        _metadata_fields = {
            "tool_id",
            "execution_time",
            "timestamp",
            "success",
            "warnings",
            "errors",
            "metadata",
        }
        data = {k: v for k, v in self.model_dump().items() if k not in _metadata_fields}
        if file_format == "json":
            import json

            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        elif file_format == "csv":
            import csv

            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(data.keys())
                writer.writerow(data.values())
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# Config:
class EnformerConfig(BaseConfig):
    """Configuration for Enformer inference.

    Attributes:
        output_tracks (list[int]): Track indices to extract from the Enformer output.
        species (Literal['human', 'mouse']): Species track head to use.
        device (str): Device used for inference.
    """

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on (e.g., 'cuda', 'cpu')",
        hidden=True,
        include_in_key=False,
    )
    output_tracks: list[int] = ConfigField(
        title="Output Tracks",
        default=[0],
        description="Track indices to extract from model output",
    )
    species: Literal["human", "mouse"] = ConfigField(
        title="Species",
        default="human",
        description="Species track head to use",
        advanced=True,
        reload_on_change=True,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return EnformerInput(sequence="A" * 196608)


@tool(
    key="enformer-prediction",
    label="Enformer Prediction",
    category="sequence_scoring",
    input_class=EnformerInput,
    config_class=EnformerConfig,
    output_class=EnformerOutput,
    description="Gene expression and regulatory activity prediction using Enformer",
    uses_gpu=True,
    example_input=example_input,
)
def run_enformer(inputs: EnformerInput, config: EnformerConfig | None = None, instance: Any = None) -> EnformerOutput:
    """Predict regulatory activity with Enformer.

    Args:
        inputs (EnformerInput): Validated sequence input.
        config (EnformerConfig | None): Validated runtime and model configuration.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        EnformerOutput: Prediction object with sequence, tracks, and metadata.
    """
    logger.debug("Using local venv for Enformer prediction")

    result = ToolInstance.dispatch(
        "enformer",
        {
            "sequence": inputs.sequence,
            "output_tracks": config.output_tracks,  # type: ignore[union-attr]
            "species": config.species,  # type: ignore[union-attr]
            "device": config.device,  # type: ignore[union-attr]
            "verbose": config.verbose,  # type: ignore[union-attr]
        },
        instance=instance,
        config=config,
    )

    return EnformerOutput(
        sequence=inputs.sequence,
        sequence_length=len(inputs.sequence),
        prediction=result["prediction"],
        output_tracks=config.output_tracks,  # type: ignore[union-attr]
        species=config.species,  # type: ignore[union-attr]
    )
