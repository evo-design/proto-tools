"""proto_tools/tools/sequence_scoring/borzoi/borzoi_prediction.py.

Borzoi single-replicate sequence scoring tool.
"""

import csv
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

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


def _validate_borzoi_sequence(sequence: str) -> str:
    """Validate and normalize one Borzoi input sequence."""
    if not sequence or not sequence.strip():
        raise ValueError("Sequence cannot be empty")
    sequence = sequence.upper()
    invalid_chars = return_invalid_nucleotide_chars(sequence, additional_valid_chars="N")
    if invalid_chars:
        raise ValueError(f"Invalid nucleotide characters in sequence: {', '.join(sorted(invalid_chars))}")
    if len(sequence) != BORZOI_CONTEXT:
        raise ValueError(f"Input sequence must have length {BORZOI_CONTEXT}, got {len(sequence)}")
    return sequence


class BorzoiInput(BaseToolInput):
    """Input for Borzoi prediction.

    Attributes:
        sequences (list[str]): DNA sequence(s) for Borzoi inference. A string
            passed to this plural field is normalized to a one-item list. Each
            sequence must be exactly 524,288 bp and only contain valid nucleotide
            characters.
    """

    sequences: list[str] = InputField(description="DNA sequence(s) to score", min_length=1)

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, value: Any) -> list[Any]:
        """Normalize the plural sequence field to a list."""
        if value is None:
            raise ValueError("sequences cannot be None")
        if isinstance(value, str):
            return [value]
        if not value:
            raise ValueError("sequences cannot be empty")
        return value  # type: ignore[no-any-return]

    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, sequences: list[str]) -> list[str]:
        """Validate and normalize nucleotide sequences; require length 524,288 bp."""
        return [_validate_borzoi_sequence(sequence) for sequence in sequences]

    def __len__(self) -> int:
        """Return the number of input sequences."""
        return len(self.sequences)


# Output:
class BorzoiPredictionResult(BaseModel):
    """Per-sequence Borzoi prediction result.

    Attributes:
        sequence (str): Input DNA sequence that was scored.
        sequence_length (int): Length of the input sequence.
        prediction (list[list[float]]): Prediction matrix with shape
            ``[num_tracks, 6144]``.
    """

    sequence: str = Field(description="Input DNA sequence")
    sequence_length: int = Field(description="Length of input sequence")
    prediction: list[list[float]] = Field(description="Prediction matrix with shape [num_tracks, 6144]")


class BorzoiOutput(BaseToolOutput):
    """Output from Borzoi single-replicate prediction.

    Attributes:
        results (list[BorzoiPredictionResult]): Per-sequence prediction results.
        output_tracks (list[int]): Track indices used for prediction.
        species (str): Species used for prediction (``"human"`` or ``"mouse"``).
        replicate (str): Borzoi replicate used (``"0"`` through ``"3"``).
        avg_output_tracks (bool): Whether requested tracks were averaged.
    """

    results: list[BorzoiPredictionResult] = Field(description="Per-sequence Borzoi prediction results")
    output_tracks: list[int] = Field(description="Track indices used for prediction")
    species: str = Field(description="Species used for prediction ('human' or 'mouse')")
    replicate: str = Field(description="Replicate used for prediction ('0' to '3')")
    avg_output_tracks: bool = Field(description="Whether track outputs were averaged")

    def __len__(self) -> int:
        """Return the number of per-sequence prediction results."""
        return len(self.results)

    def __getitem__(self, index: int) -> BorzoiPredictionResult:
        """Return a per-sequence prediction result."""
        return self.results[index]

    def __iter__(self) -> Iterator[BorzoiPredictionResult]:  # type: ignore[override]
        """Iterate over per-sequence prediction results."""
        return iter(self.results)

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
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        elif file_format == "csv":
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "sequence_index",
                        "sequence_length",
                        "output_tracks",
                        "species",
                        "replicate",
                        "avg_output_tracks",
                        "prediction",
                    ]
                )
                for idx, result in enumerate(self.results):
                    writer.writerow(
                        [
                            idx,
                            result.sequence_length,
                            json.dumps(self.output_tracks),
                            self.species,
                            self.replicate,
                            self.avg_output_tracks,
                            json.dumps(result.prediction),
                        ]
                    )
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
        batch_size (int): Number of sequences to process in each GPU batch.
        device (str): Device used for inference (inherited).
    """

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
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
    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Number of sequences to process simultaneously on GPU",
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
    def validate_mouse_flash_attn(self) -> "BorzoiConfig":
        """Mouse Borzoi checkpoints do not support FlashAttention."""
        if self.species == "mouse" and self.use_flash_attn:
            raise ValueError("FlashAttention (use_flash_attn=True) is not available for mouse models.")
        return self


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return BorzoiInput(sequences=["A" * 524288])


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
    iterable_input_field="sequences",
    iterable_output_field="results",
)
def run_borzoi(inputs: BorzoiInput, config: BorzoiConfig, instance: Any = None) -> BorzoiOutput:
    """Predict regulatory activity using a single Borzoi replicate.

    Args:
        inputs (BorzoiInput): Validated sequence input containing one or more
            sequences.
        config (BorzoiConfig): Validated runtime and model configuration.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        BorzoiOutput: Prediction object for one Borzoi replicate. Contains one
            result per input sequence.
    """
    if config.use_flash_attn and not config.device.startswith("cuda"):
        raise ValueError("Must run on GPU to use FlashAttention with Borzoi")

    logger.debug("Using local venv for Borzoi prediction")

    result = ToolInstance.dispatch(
        "borzoi",
        {
            "operation": "predict",
            "sequences": inputs.sequences,
            "output_tracks": config.output_tracks,
            "species": config.species,
            "replicate": config.replicate,
            "use_flash_attn": config.use_flash_attn,
            "avg_output_tracks": config.avg_output_tracks,
            "batch_size": config.batch_size,
            "device": config.device,
            "verbose": config.verbose,
            "seed": config.seed,
        },
        instance=instance,
        config=config,
    )

    predictions = result["predictions"]
    if len(predictions) != len(inputs.sequences):
        raise ValueError(f"Expected {len(inputs.sequences)} Borzoi predictions, got {len(predictions)}")

    return BorzoiOutput(
        results=[
            BorzoiPredictionResult(
                sequence=sequence,
                sequence_length=len(sequence),
                prediction=prediction,
            )
            for sequence, prediction in zip(inputs.sequences, predictions, strict=True)
        ],
        output_tracks=config.output_tracks,
        species=config.species,
        replicate=config.replicate,
        avg_output_tracks=config.avg_output_tracks,
    )
