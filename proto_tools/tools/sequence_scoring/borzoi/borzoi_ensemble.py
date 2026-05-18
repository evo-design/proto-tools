"""proto_tools/tools/sequence_scoring/borzoi/borzoi_ensemble.py.

Borzoi ensemble sequence scoring tool.
"""

import csv
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from proto_tools.tools.sequence_scoring.borzoi.borzoi_prediction import (
    BORZOI_OUTPUT_RESOLUTION,
    BorzoiConfig,
    BorzoiInput,
    run_borzoi,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseConfig, BaseToolOutput, ConfigField
from proto_tools.utils.progress import progress_bar

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
class BorzoiEnsemblePredictionResult(BaseModel):
    """Per-sequence Borzoi ensemble prediction result.

    Coordinates in this result are relative to the input sequence returned in
    ``sequence``. ``context_start``/``context_end`` describe the model input
    window used for every replicate. ``output_start``/``output_end`` describe
    the source-sequence span covered by Borzoi's output bins. If target ranges
    were supplied, ``target_start``/``target_end`` echo the requested range that
    was validated to fit inside the output window.

    Attributes:
        sequence (str): Input DNA sequence that was scored.
        sequence_length (int): Length of the input sequence.
        predictions (list[list[list[float]]]): Stacked predictions with shape
            ``[4, num_tracks, 6144]`` for replicates 0-3.
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
    predictions: list[list[list[float]]] = Field(description="Stacked predictions with shape [4, num_tracks, 6144]")
    context_start: int = Field(description="0-based start of the Borzoi input window in sequence")
    context_end: int = Field(description="0-based exclusive end of the Borzoi input window")
    output_start: int = Field(description="0-based start of the span covered by Borzoi output bins")
    output_end: int = Field(description="0-based exclusive end of the Borzoi output-bin span")
    output_resolution: int = Field(
        default=BORZOI_OUTPUT_RESOLUTION, description="Base pairs represented by each output bin"
    )
    target_start: int | None = Field(default=None, description="Requested target start, if target_ranges was provided")
    target_end: int | None = Field(default=None, description="Requested target end, if target_ranges was provided")


class BorzoiEnsembleOutput(BaseToolOutput):
    """Output from Borzoi ensemble prediction.

    Attributes:
        results (list[BorzoiEnsemblePredictionResult]): Per-sequence ensemble
            prediction results.
        output_tracks (list[int]): Track indices used for prediction.
        species (str): Species used for prediction (``"human"`` or ``"mouse"``).
        avg_output_tracks (bool): Whether requested tracks were averaged.
        num_replicates (int): Number of replicates returned (always 4).
    """

    results: list[BorzoiEnsemblePredictionResult] = Field(description="Per-sequence ensemble prediction results")
    output_tracks: list[int] = Field(description="Track indices used for prediction")
    species: str = Field(description="Species used for prediction ('human' or 'mouse')")
    avg_output_tracks: bool = Field(description="Whether track outputs were averaged")
    num_replicates: int = Field(default=4, description="Number of Borzoi replicates returned")

    def __len__(self) -> int:
        """Return the number of per-sequence ensemble results."""
        return len(self.results)

    def __getitem__(self, index: int) -> BorzoiEnsemblePredictionResult:
        """Return a per-sequence ensemble prediction result."""
        return self.results[index]

    def __iter__(self) -> Iterator[BorzoiEnsemblePredictionResult]:  # type: ignore[override]
        """Iterate over per-sequence ensemble results."""
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
                        "avg_output_tracks",
                        "num_replicates",
                        "context_start",
                        "context_end",
                        "output_start",
                        "output_end",
                        "output_resolution",
                        "target_start",
                        "target_end",
                        "predictions",
                    ]
                )
                for idx, result in enumerate(self.results):
                    writer.writerow(
                        [
                            idx,
                            result.sequence_length,
                            json.dumps(self.output_tracks),
                            self.species,
                            self.avg_output_tracks,
                            self.num_replicates,
                            result.context_start,
                            result.context_end,
                            result.output_start,
                            result.output_end,
                            result.output_resolution,
                            result.target_start,
                            result.target_end,
                            json.dumps(result.predictions),
                        ]
                    )
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# Config:
class BorzoiEnsembleConfig(BaseConfig):
    """Configuration for Borzoi ensemble prediction.

    Attributes:
        output_tracks (list[int]): Track indices to extract from model output.
        species (Literal['human', 'mouse']): Species model to use.
        avg_output_tracks (bool): Whether to average selected tracks.
        batch_size (int): Number of sequences to process in each GPU batch.
        device (str): Device used for inference.
    """

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        include_in_key=False,
    )
    output_tracks: list[int] = ConfigField(
        title="Output Tracks",
        default=[0],
        description="Indices of Borzoi output tracks to extract (7611 human / 2608 mouse)",
    )
    species: Literal["human", "mouse"] = ConfigField(
        title="Species",
        default="human",
        description="Species model to use",
    )
    avg_output_tracks: bool = ConfigField(
        title="Average Tracks",
        default=True,
        description="Whether to average selected tracks into one output",
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Number of sequences to process simultaneously on GPU",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return BorzoiInput(sequences=["A" * 524288])


@tool(
    key="borzoi-ensemble",
    label="Borzoi Ensemble",
    category="sequence_scoring",
    input_class=BorzoiInput,
    config_class=BorzoiEnsembleConfig,
    output_class=BorzoiEnsembleOutput,
    description="Regulatory activity prediction using all 4 Borzoi replicates",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="sequences",
    iterable_output_field="results",
)
def run_borzoi_ensemble(
    inputs: BorzoiInput,
    config: BorzoiEnsembleConfig,
    instance: Any = None,
) -> BorzoiEnsembleOutput:
    """Predict regulatory activity using all Borzoi replicates.

    Args:
        inputs (BorzoiInput): Validated sequence input.
        config (BorzoiEnsembleConfig): Validated runtime and model configuration.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        BorzoiEnsembleOutput: Stacked predictions from Borzoi replicates 0-3.
    """
    logger.debug("Using local execution for Borzoi ensemble prediction")

    replicate_outputs = []
    iterator = progress_bar(
        range(4),
        desc="Borzoi replicates",
        unit="replicate",
        total=4,
        disable=not config.verbose,
    )

    for replicate in iterator:
        replicate_config = BorzoiConfig(
            output_tracks=config.output_tracks,
            species=config.species,
            replicate=str(replicate),  # type: ignore[arg-type]
            avg_output_tracks=config.avg_output_tracks,
            batch_size=config.batch_size,
            device=config.device,
            verbose=config.verbose,
            timeout=config.timeout,
            seed=config.seed,
        )
        replicate_output = run_borzoi(inputs, replicate_config, instance=instance)
        replicate_outputs.append(replicate_output)

    reference_output = replicate_outputs[0]

    return BorzoiEnsembleOutput(
        results=[
            BorzoiEnsemblePredictionResult(
                sequence=sequence,
                sequence_length=len(sequence),
                predictions=[replicate_output.results[idx].prediction for replicate_output in replicate_outputs],
                context_start=reference_output.results[idx].context_start,
                context_end=reference_output.results[idx].context_end,
                output_start=reference_output.results[idx].output_start,
                output_end=reference_output.results[idx].output_end,
                output_resolution=reference_output.results[idx].output_resolution,
                target_start=reference_output.results[idx].target_start,
                target_end=reference_output.results[idx].target_end,
            )
            for idx, sequence in enumerate(inputs.sequences)
        ],
        output_tracks=config.output_tracks,
        species=config.species,
        avg_output_tracks=config.avg_output_tracks,
        num_replicates=4,
    )
