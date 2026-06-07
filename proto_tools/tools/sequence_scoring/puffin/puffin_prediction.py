"""Puffin fast transcription initiation prediction (Puffin.predict, no interpretation)."""

import csv
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from proto_tools.tools.sequence_scoring.shared_data_models import validate_dna_sequence
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)

logger = logging.getLogger(__name__)

PUFFIN_PADDING = 325
PUFFIN_MIN_INPUT_LENGTH = 2 * PUFFIN_PADDING + 1
PUFFIN_OUTPUT_CHANNELS = 10
TRACK_NAMES = (
    "FANTOM_CAGE+",
    "ENCODE_CAGE+",
    "ENCODE_RAMPAGE+",
    "GRO_CAP+",
    "PRO_CAP+",
    "PRO_CAP-",
    "GRO_CAP-",
    "ENCODE_RAMPAGE-",
    "ENCODE_CAGE-",
    "FANTOM_CAGE-",
)


# ============================================================================
# Data Models
# ============================================================================
class PuffinPredictionInput(BaseToolInput):
    """Input for Puffin transcription initiation prediction.

    Each sequence must be at least 651 bp. The model uses 325 bp padding on each
    side of the region of interest, so the per-base output length equals
    ``len(sequence) - 650``.

    Attributes:
        sequences (list[str]): DNA sequence(s) at least 651 bp long. A single
            string is normalized to a one-item list. Only ``A``, ``C``, ``G``,
            ``T``, ``N`` are accepted.
    """

    sequences: list[str] = InputField(
        title="Sequences",
        description="DNA sequence(s) >= 651 bp. Per-base output length = len(seq) - 650",
        min_length=1,
    )

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
        """Validate nucleotide composition and minimum length."""
        validated = [validate_dna_sequence(sequence) for sequence in sequences]
        for index, sequence in enumerate(validated):
            if len(sequence) < PUFFIN_MIN_INPUT_LENGTH:
                raise ValueError(
                    f"sequences[{index}]: must be at least {PUFFIN_MIN_INPUT_LENGTH} bp "
                    f"(2 * {PUFFIN_PADDING} bp padding + 1 bp output); got {len(sequence)}"
                )
        return validated

    def __len__(self) -> int:
        """Number of input sequences."""
        return len(self.sequences)


class PuffinPredictionResult(BaseModel):
    """Per-sequence Puffin prediction result.

    The 10 output channels are ordered as ``TRACK_NAMES``: five forward-strand
    targets (FANTOM_CAGE+, ENCODE_CAGE+, ENCODE_RAMPAGE+, GRO_CAP+, PRO_CAP+),
    then five reverse-strand targets (PRO_CAP-, GRO_CAP-, ENCODE_RAMPAGE-,
    ENCODE_CAGE-, FANTOM_CAGE-). Per-base values are in log-scale and can be
    interpreted as ``ln(count_scale_signal + 1)``.

    Attributes:
        sequence (str): Input DNA sequence that was scored.
        sequence_length (int): Length of the input sequence.
        output_length (int): Number of per-base output positions
            (= ``sequence_length - 650``).
        output_start (int): 0-based sequence coordinate of the first per-base
            output position (always ``325``).
        output_end (int): 0-based exclusive end of the per-base output span in
            the input sequence (= ``sequence_length - 325``).
        predictions (list[list[float]]): Per-base predictions with shape
            ``[output_length, 10]``. Channel order matches ``TRACK_NAMES``.
    """

    sequence: str = Field(title="Sequence", description="DNA sequence originally provided to the tool")
    sequence_length: int = Field(title="Sequence Length", description="Length of the provided DNA sequence")
    output_length: int = Field(
        title="Output Length",
        description="Number of per-base output positions (= sequence_length - 650)",
    )
    output_start: int = Field(
        title="Output Start",
        description="0-based sequence coordinate of the first output position",
    )
    output_end: int = Field(
        title="Output End",
        description="0-based exclusive end of the output span in the input sequence",
    )
    predictions: list[list[float]] = Field(
        title="Predictions",
        description="Per-base log-scale predictions in TRACK_NAMES order (shape [output_length, 10])",
    )


class PuffinPredictionConfig(BaseConfig):
    """Configuration for Puffin prediction.

    Attributes:
        device (str): Device used for inference.
    """

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        include_in_key=False,
    )


class PuffinPredictionOutput(BaseToolOutput):
    """Output from Puffin prediction.

    Attributes:
        results (list[PuffinPredictionResult]): Per-sequence prediction results.
        track_names (list[str]): Names of the 10 output channels in order.
    """

    results: list[PuffinPredictionResult] = Field(title="Results", description="Per-sequence Puffin prediction results")
    track_names: list[str] = Field(title="Track Names", description="Names of the 10 output channels in order")

    def __len__(self) -> int:
        """Number of per-sequence results."""
        return len(self.results)

    def __getitem__(self, index: int) -> PuffinPredictionResult:
        """Per-sequence result by index."""
        return self.results[index]

    def __iter__(self) -> Iterator[PuffinPredictionResult]:  # type: ignore[override]
        """Iterate per-sequence results."""
        return iter(self.results)

    @property
    def output_format_options(self) -> list[str]:
        """Supported export formats."""
        return ["json", "csv"]

    @property
    def output_format_default(self) -> str:
        """Default export format."""
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
                        "output_length",
                        "output_start",
                        "output_end",
                        "track_names",
                        "predictions",
                    ]
                )
                for idx, result in enumerate(self.results):
                    writer.writerow(
                        [
                            idx,
                            result.sequence_length,
                            result.output_length,
                            result.output_start,
                            result.output_end,
                            json.dumps(self.track_names),
                            json.dumps(result.predictions),
                        ]
                    )
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> PuffinPredictionInput:
    """Minimal valid input for testing and examples."""
    return PuffinPredictionInput(sequences=["A" * 1650])


@tool(
    key="puffin-prediction",
    label="Puffin Prediction",
    category="sequence_scoring",
    input_class=PuffinPredictionInput,
    config_class=PuffinPredictionConfig,
    output_class=PuffinPredictionOutput,
    description="Basepair-resolution transcription initiation prediction with Puffin",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    cacheable=True,
)
def run_puffin_prediction(
    inputs: PuffinPredictionInput,
    config: PuffinPredictionConfig,
    instance: Any = None,
) -> PuffinPredictionOutput:
    """Predict per-base transcription initiation signals with Puffin.

    Args:
        inputs (PuffinPredictionInput): Validated DNA sequence input.
        config (PuffinPredictionConfig): Validated runtime configuration.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        PuffinPredictionOutput: Prediction object with one result per input
            sequence and the 10-channel track ordering.
    """
    logger.debug("Using local venv for Puffin prediction")

    result = ToolInstance.dispatch(
        "puffin",
        {
            "operation": "predict",
            "sequences": inputs.sequences,
            "device": config.device,
            "verbose": config.verbose,
            "seed": config.seed,
        },
        instance=instance,
        config=config,
    )

    predictions = result["predictions"]
    if len(predictions) != len(inputs.sequences):
        raise ValueError(f"Expected {len(inputs.sequences)} Puffin predictions, got {len(predictions)}")

    results = [
        PuffinPredictionResult(
            sequence=sequence,
            sequence_length=len(sequence),
            output_length=len(sequence) - 2 * PUFFIN_PADDING,
            output_start=PUFFIN_PADDING,
            output_end=len(sequence) - PUFFIN_PADDING,
            predictions=prediction,
        )
        for sequence, prediction in zip(inputs.sequences, predictions, strict=True)
    ]
    return PuffinPredictionOutput(results=results, track_names=list(TRACK_NAMES))
