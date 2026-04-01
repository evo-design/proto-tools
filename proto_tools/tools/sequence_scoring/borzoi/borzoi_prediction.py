"""proto_tools/tools/sequence_scoring/borzoi/borzoi_prediction.py.

Borzoi single-replicate sequence scoring tool.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

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

BORZOI_CONTEXT = 524_288
BORZOI_OUTPUT = 6_144


# ============================================================================
# Data Models
# ============================================================================


class BorzoiInput(BaseToolInput):
    """Input for Borzoi prediction.

    Attributes:
        sequence (str): DNA sequence for Borzoi inference. Must be exactly
            524,288 bp and only contain valid nucleotide characters.
    """

    sequence: str = InputField(description="DNA sequence to score")

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, sequence: str) -> str:
        """Validate and normalize nucleotide sequence; require length 524,288 bp."""
        if not sequence or not sequence.strip():
            raise ValueError("Sequence cannot be empty")
        sequence = sequence.upper()
        invalid_chars = return_invalid_nucleotide_chars(sequence, additional_valid_chars="N")
        if invalid_chars:
            raise ValueError(f"Invalid nucleotide characters in sequence: {', '.join(sorted(invalid_chars))}")
        if len(sequence) != BORZOI_CONTEXT:
            raise ValueError(f"Input sequence must have length {BORZOI_CONTEXT}, got {len(sequence)}")
        return sequence


# Output:
class BorzoiOutput(BaseToolOutput):
    """Output from Borzoi single-replicate prediction.

    Attributes:
        sequence (str): Input DNA sequence that was scored.
        sequence_length (int): Length of the input sequence (always 524,288).
        prediction (list[list[float]]): Prediction matrix with shape
            ``[num_tracks, 6144]``.
        output_tracks (list[int]): Track indices used for prediction.
        species (str): Species used for prediction (``"human"`` or ``"mouse"``).
        replicate (str): Borzoi replicate used (``"0"`` through ``"3"``).
        avg_output_tracks (bool): Whether requested tracks were averaged.
    """

    sequence: str = Field(description="Input DNA/RNA sequence")
    sequence_length: int = Field(description="Length of input sequence")
    prediction: list[list[float]] = Field(description="Prediction matrix with shape [num_tracks, 6144]")
    output_tracks: list[int] = Field(description="Track indices used for prediction")
    species: str = Field(description="Species used for prediction ('human' or 'mouse')")
    replicate: str = Field(description="Replicate used for prediction ('0' to '3')")
    avg_output_tracks: bool = Field(description="Whether track outputs were averaged")

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
class BorzoiConfig(BaseConfig):
    """Configuration for Borzoi single-replicate prediction.

    Attributes:
        output_tracks (list[int]): Track indices to extract from model output.
        species (Literal['human', 'mouse']): Species model to use.
        replicate (Literal['0', '1', '2', '3']): Replicate ID to run.
        avg_output_tracks (bool): Whether to average selected tracks.
        use_flash_attn (bool): Whether to run FlashAttention-backed models.
        device (str): Device used for inference (inherited).
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
        description="Species model to use",
        reload_on_change=True,
    )
    replicate: Literal["0", "1", "2", "3"] = ConfigField(
        title="Replicate",
        default="0",
        description="Replicate ID to run",
        advanced=True,
        reload_on_change=True,
    )
    avg_output_tracks: bool = ConfigField(
        title="Average Tracks",
        default=True,
        description="Whether to average selected tracks into one output",
        advanced=True,
    )
    use_flash_attn: bool = ConfigField(
        title="Use FlashAttention",
        default=True,
        description="Whether to use FlashAttention models",
        hidden=True,
        reload_on_change=True,
    )

    @model_validator(mode="after")
    def validate_mouse_flash_attn(self) -> BorzoiConfig:
        """Mouse Borzoi checkpoints do not support FlashAttention."""
        if self.species == "mouse" and self.use_flash_attn:
            raise ValueError("FlashAttention (use_flash_attn=True) is not available for mouse models.")
        return self


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return BorzoiInput(sequence="A" * 524288)


@tool(
    key="borzoi-prediction",
    label="Borzoi Prediction",
    category="sequence_scoring",
    input_class=BorzoiInput,
    config_class=BorzoiConfig,
    output_class=BorzoiOutput,
    description="Regulatory activity prediction using a single Borzoi replicate",
    uses_gpu=True,
    example_input=example_input,
)
def run_borzoi(inputs: BorzoiInput, config: BorzoiConfig | None = None, instance: Any = None) -> BorzoiOutput:
    """Predict regulatory activity using a single Borzoi replicate.

    Args:
        inputs (BorzoiInput): Validated sequence input.
        config (BorzoiConfig | None): Validated runtime and model configuration.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        BorzoiOutput: Prediction object for one Borzoi replicate.
    """
    if config.use_flash_attn and not config.device.startswith("cuda"):  # type: ignore[union-attr]
        raise ValueError("Must run on GPU to use FlashAttention with Borzoi")

    logger.debug("Using local venv for Borzoi prediction")

    result = ToolInstance.dispatch(
        "borzoi",
        {
            "sequence": inputs.sequence,
            "output_tracks": config.output_tracks,  # type: ignore[union-attr]
            "species": config.species,  # type: ignore[union-attr]
            "replicate": config.replicate,  # type: ignore[union-attr]
            "use_flash_attn": config.use_flash_attn,  # type: ignore[union-attr]
            "avg_output_tracks": config.avg_output_tracks,  # type: ignore[union-attr]
            "device": config.device,  # type: ignore[union-attr]
            "verbose": config.verbose,  # type: ignore[union-attr]
        },
        instance=instance,
        config=config,
    )

    return BorzoiOutput(
        sequence=inputs.sequence,
        sequence_length=len(inputs.sequence),
        prediction=result["prediction"],
        output_tracks=config.output_tracks,  # type: ignore[union-attr]
        species=config.species,  # type: ignore[union-attr]
        replicate=config.replicate,  # type: ignore[union-attr]
        avg_output_tracks=config.avg_output_tracks,  # type: ignore[union-attr]
    )
