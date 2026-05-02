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

from proto_tools.tools.sequence_scoring.shared_data_models import (
    SequenceTargetRange,
    prepare_model_windows,
    validate_dna_sequence,
)
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

BORZOI_CONTEXT = 524_288
BORZOI_OUTPUT = 6_144
BORZOI_OUTPUT_RESOLUTION = 32
BORZOI_OUTPUT_LENGTH = BORZOI_OUTPUT * BORZOI_OUTPUT_RESOLUTION
BORZOI_OUTPUT_FLANK = (BORZOI_CONTEXT - BORZOI_OUTPUT_LENGTH) // 2


# ============================================================================
# Data Models
# ============================================================================


class BorzoiInput(BaseToolInput):
    """Input for Borzoi prediction.

    There are two supported modes:

    * Exact-window mode: pass sequence(s) that are exactly 524,288 bp, the
      Borzoi model context length. The full sequence is sent to the model.
    * Target-range mode: pass longer source sequence(s) plus one
      ``SequenceTargetRange`` per sequence. Each range identifies the
      sequence-relative span the caller wants covered by Borzoi's output bins.
      The tool extracts the fixed 524,288 bp model context window and records
      where that context and output window came from in the original sequence.

    Attributes:
        sequences (list[str]): DNA sequence(s) for Borzoi inference. A string
            passed to this plural field is normalized to a one-item list.
        target_ranges (list[SequenceTargetRange] | None): Optional target
            ranges within each provided sequence. Coordinates are relative to
            the corresponding sequence, 0-based, and use exclusive ends. When
            provided, each sequence must contain enough real context to extract
            a full Borzoi input window; the tool does not pad missing context.
    """

    sequences: list[str] = InputField(
        description="DNA sequence(s): exact model windows, or sources paired with target_ranges",
        min_length=1,
    )
    target_ranges: list[SequenceTargetRange] | None = InputField(
        default=None,
        description="Sequence-relative range(s) that must remain inside Borzoi output bins",
        advanced=True,
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
        """Validate and normalize nucleotide sequences."""
        return [validate_dna_sequence(sequence) for sequence in sequences]

    @field_validator("target_ranges", mode="before")
    @classmethod
    def normalize_target_ranges(cls, value: Any) -> list[Any] | None:
        """Normalize a single target range to a list."""
        if value is None:
            return None
        if isinstance(value, list):
            if not value:
                raise ValueError("target_ranges cannot be empty")
            return value
        return [value]

    @model_validator(mode="after")
    def validate_window_inputs(self) -> "BorzoiInput":
        """Validate exact-context or target-aligned window inputs."""
        prepare_model_windows(
            self.sequences,
            context_length=BORZOI_CONTEXT,
            output_length=BORZOI_OUTPUT_LENGTH,
            target_ranges=self.target_ranges,
        )
        return self

    def __len__(self) -> int:
        """Return the number of input sequences."""
        return len(self.sequences)


# Output:
class BorzoiPredictionResult(BaseModel):
    """Per-sequence Borzoi prediction result.

    Coordinates in this result are relative to the input sequence returned in
    ``sequence``. ``context_start``/``context_end`` describe the 524,288 bp
    model input window that was dispatched. ``output_start``/``output_end``
    describe the span of that source sequence covered by Borzoi's 6,144 output
    bins. If ``target_ranges`` were supplied, ``target_start``/``target_end``
    echo the requested target range that was validated to fit inside the output
    window.

    Attributes:
        sequence (str): Input DNA sequence that was scored.
        sequence_length (int): Length of the input sequence.
        prediction (list[list[float]]): Prediction matrix with shape
            ``[num_tracks, 6144]``.
        context_start (int): Start coordinate of the Borzoi input window in the
            source sequence.
        context_end (int): End coordinate of the Borzoi input window in the
            source sequence.
        output_start (int): Source-sequence coordinate of the first Borzoi
            output bin.
        output_end (int): Source-sequence coordinate immediately after the last
            Borzoi output bin.
        output_resolution (int): Base pairs represented by each output bin.
        target_start (int | None): Target start coordinate supplied for this
            sequence.
        target_end (int | None): Target end coordinate supplied for this
            sequence.
    """

    sequence: str = Field(description="DNA sequence originally provided to the tool")
    sequence_length: int = Field(description="Length of the provided DNA sequence")
    prediction: list[list[float]] = Field(description="Prediction matrix with shape [num_tracks, 6144]")
    context_start: int = Field(description="0-based start of the Borzoi input window in sequence")
    context_end: int = Field(description="0-based exclusive end of the Borzoi input window")
    output_start: int = Field(description="0-based start of the span covered by Borzoi output bins")
    output_end: int = Field(description="0-based exclusive end of the Borzoi output-bin span")
    output_resolution: int = Field(
        default=BORZOI_OUTPUT_RESOLUTION, description="Base pairs represented by each output bin"
    )
    target_start: int | None = Field(default=None, description="Requested target start, if target_ranges was provided")
    target_end: int | None = Field(default=None, description="Requested target end, if target_ranges was provided")


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
                        "context_start",
                        "context_end",
                        "output_start",
                        "output_end",
                        "output_resolution",
                        "target_start",
                        "target_end",
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
                            result.context_start,
                            result.context_end,
                            result.output_start,
                            result.output_end,
                            result.output_resolution,
                            result.target_start,
                            result.target_end,
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
    use_flash_attn = config.species == "human"
    if use_flash_attn and not config.device.startswith("cuda"):
        raise ValueError("Must run on GPU to use FlashAttention with Borzoi")

    logger.debug("Using local venv for Borzoi prediction")

    prepared_windows = prepare_model_windows(
        inputs.sequences,
        context_length=BORZOI_CONTEXT,
        output_length=BORZOI_OUTPUT_LENGTH,
        target_ranges=inputs.target_ranges,
    )
    model_sequences = [window.model_sequence for window in prepared_windows]

    result = ToolInstance.dispatch(
        "borzoi",
        {
            "operation": "predict",
            "sequences": model_sequences,
            "output_tracks": config.output_tracks,
            "species": config.species,
            "replicate": config.replicate,
            "use_flash_attn": use_flash_attn,
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
    if len(predictions) != len(model_sequences):
        raise ValueError(f"Expected {len(model_sequences)} Borzoi predictions, got {len(predictions)}")

    return BorzoiOutput(
        results=[
            BorzoiPredictionResult(
                sequence=sequence,
                sequence_length=len(sequence),
                prediction=prediction,
                context_start=window.context_start,
                context_end=window.context_end,
                output_start=window.output_start,
                output_end=window.output_end,
                output_resolution=BORZOI_OUTPUT_RESOLUTION,
                target_start=window.target_start,
                target_end=window.target_end,
            )
            for sequence, prediction, window in zip(inputs.sequences, predictions, prepared_windows, strict=True)
        ],
        output_tracks=config.output_tracks,
        species=config.species,
        replicate=config.replicate,
        avg_output_tracks=config.avg_output_tracks,
    )
