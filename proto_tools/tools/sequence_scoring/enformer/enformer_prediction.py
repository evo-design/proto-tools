"""proto_tools/tools/sequence_scoring/enformer/enformer_prediction.py.

Enformer sequence scoring tool.
"""

import csv
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

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
def _validate_enformer_sequence(sequence: str) -> str:
    """Validate and normalize one Enformer input sequence."""
    if not sequence or not sequence.strip():
        raise ValueError("Sequence cannot be empty")
    sequence = sequence.upper()
    invalid_chars = return_invalid_nucleotide_chars(sequence, additional_valid_chars="N")
    if invalid_chars:
        raise ValueError(f"Invalid nucleotide characters in sequence: {', '.join(sorted(invalid_chars))}")
    if len(sequence) != ENFORMER_CONTEXT:
        raise ValueError(f"Input sequence must have length {ENFORMER_CONTEXT}, got {len(sequence)}")
    return sequence


class EnformerInput(BaseToolInput):
    """Input for Enformer regulatory activity prediction.

    Attributes:
        sequences (list[str]): DNA sequence(s) for Enformer inference. A string
            passed to this plural field is normalized to a one-item list. Each
            sequence must be exactly 196,608 bp and only contain valid nucleotide
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
        """Validate and normalize nucleotide sequences; require length 196,608 bp."""
        return [_validate_enformer_sequence(sequence) for sequence in sequences]

    def __len__(self) -> int:
        """Return the number of input sequences."""
        return len(self.sequences)


# Output:
class EnformerPredictionResult(BaseModel):
    """Per-sequence Enformer prediction result.

    Attributes:
        sequence (str): Input DNA sequence that was scored.
        sequence_length (int): Length of the input sequence.
        prediction (list[list[float]]): Predicted signal matrix with shape
            ``[896, num_tracks]``.
    """

    sequence: str = Field(description="Input DNA sequence")
    sequence_length: int = Field(description="Length of input sequence")
    prediction: list[list[float]] = Field(description="Predicted activity matrix with shape [896, num_tracks]")


class EnformerOutput(BaseToolOutput):
    """Output from Enformer prediction.

    Attributes:
        results (list[EnformerPredictionResult]): Per-sequence prediction results.
        output_tracks (list[int]): Track indices that were extracted.
        species (str): Species used for prediction (``"human"`` or ``"mouse"``).
    """

    results: list[EnformerPredictionResult] = Field(description="Per-sequence Enformer prediction results")
    output_tracks: list[int] = Field(description="Track indices extracted from Enformer")
    species: str = Field(description="Species used for prediction ('human' or 'mouse')")

    def __len__(self) -> int:
        """Return the number of per-sequence prediction results."""
        return len(self.results)

    def __getitem__(self, index: int) -> EnformerPredictionResult:
        """Return a per-sequence prediction result."""
        return self.results[index]

    def __iter__(self) -> Iterator[EnformerPredictionResult]:  # type: ignore[override]
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
                writer.writerow(["sequence_index", "sequence_length", "output_tracks", "species", "prediction"])
                for idx, result in enumerate(self.results):
                    writer.writerow(
                        [
                            idx,
                            result.sequence_length,
                            json.dumps(self.output_tracks),
                            self.species,
                            json.dumps(result.prediction),
                        ]
                    )
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# Config:
class EnformerConfig(BaseConfig):
    """Configuration for Enformer inference.

    Attributes:
        output_tracks (list[int]): Track indices to extract from the Enformer output.
        species (Literal['human', 'mouse']): Species track head to use.
        batch_size (int): Number of sequences to process in each GPU batch.
        device (str): Device used for inference.
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
        description="Species track head to use",
        advanced=True,
        reload_on_change=True,
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Number of sequences to process simultaneously on GPU",
        advanced=True,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return EnformerInput(sequences=["A" * 196608])


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
    iterable_input_field="sequences",
    iterable_output_field="results",
)
def run_enformer(inputs: EnformerInput, config: EnformerConfig, instance: Any = None) -> EnformerOutput:
    """Predict regulatory activity with Enformer.

    Args:
        inputs (EnformerInput): Validated sequence input containing one or more
            sequences.
        config (EnformerConfig): Validated runtime and model configuration.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        EnformerOutput: Prediction object with one result per input sequence.
    """
    logger.debug("Using local venv for Enformer prediction")

    result = ToolInstance.dispatch(
        "enformer",
        {
            "operation": "predict",
            "sequences": inputs.sequences,
            "output_tracks": config.output_tracks,
            "species": config.species,
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
        raise ValueError(f"Expected {len(inputs.sequences)} Enformer predictions, got {len(predictions)}")

    return EnformerOutput(
        results=[
            EnformerPredictionResult(
                sequence=sequence,
                sequence_length=len(sequence),
                prediction=prediction,
            )
            for sequence, prediction in zip(inputs.sequences, predictions, strict=True)
        ],
        output_tracks=config.output_tracks,
        species=config.species,
    )
