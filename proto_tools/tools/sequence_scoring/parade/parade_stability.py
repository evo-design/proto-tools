"""PARADE 3' UTR mRNA stability prediction."""

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from proto_tools.tools.sequence_scoring.parade.shared_data_models import (
    ParadeCheckpointConfig,
    ParadeSequenceInput,
    resolve_checkpoint_source,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseToolOutput, ToolInstance
from proto_tools.utils.tool_io import Metrics, MetricSpec

# Input:
ParadeStabilityInput = ParadeSequenceInput
# Config:
ParadeStabilityConfig = ParadeCheckpointConfig


class ParadeStabilityMetrics(Metrics):
    """PARADE mRNA stability prediction for one 3' UTR sequence.

    Metrics documented in ``metric_spec``:
        log_ratio (float): Predicted RNA/gDNA log-ratio; higher means more stable.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "log_ratio": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": None,
            "description": "Predicted PARADE RNA/gDNA log-ratio; higher is more stable.",
            "better_values_are": "higher",
        },
    }


class ParadeStabilityResult(BaseModel):
    """Per-sequence PARADE stability result.

    Attributes:
        sequence (str): 3' UTR sequence that was scored (DNA alphabet).
        sequence_length (int): Length of the scored sequence.
        log_ratio (float): Predicted RNA/gDNA log-ratio; higher means more stable.
    """

    sequence: str = Field(title="Sequence", description="3' UTR sequence scored by PARADE")
    sequence_length: int = Field(title="Sequence Length", description="Length of the scored UTR sequence")
    log_ratio: float = Field(
        title="Log Ratio", description="Predicted RNA/gDNA log-ratio; higher is more stable"
    )


class ParadeStabilityOutput(BaseToolOutput):
    """Output from PARADE 3' UTR stability scoring.

    Attributes:
        results (list[ParadeStabilityResult]): Per-sequence PARADE stability predictions.
    """

    results: list[ParadeStabilityResult] = Field(
        title="Results", description="Per-sequence PARADE stability scoring results"
    )

    def __len__(self) -> int:
        """Return the number of per-sequence results."""
        return len(self.results)

    def __getitem__(self, index: int) -> ParadeStabilityResult:
        """Return a per-sequence result."""
        return self.results[index]

    def __iter__(self) -> Iterator[ParadeStabilityResult]:  # type: ignore[override]
        """Iterate over per-sequence results."""
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
        if file_format == "json":
            data = {
                "results": [
                    {
                        "sequence": result.sequence,
                        "sequence_length": result.sequence_length,
                        "log_ratio": result.log_ratio,
                    }
                    for result in self.results
                ]
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            return
        if file_format == "csv":
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["sequence_index", "sequence", "sequence_length", "log_ratio"])
                for idx, result in enumerate(self.results):
                    writer.writerow([idx, result.sequence, result.sequence_length, result.log_ratio])
            return
        raise ValueError(f"Unsupported format: {file_format}")


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return ParadeStabilityInput(sequences=["ACGT" * 12 + "AC"])


@tool(
    key="parade-stability",
    label="PARADE mRNA Stability",
    category="sequence_scoring",
    input_class=ParadeStabilityInput,
    config_class=ParadeStabilityConfig,
    output_class=ParadeStabilityOutput,
    metrics_class=ParadeStabilityMetrics,
    description="Predict 3' UTR mRNA stability (RNA/gDNA log-ratio) with the PARADE LegNet model",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    cacheable=True,
)
def run_parade_stability(
    inputs: ParadeStabilityInput,
    config: ParadeStabilityConfig,
    instance: Any = None,
) -> ParadeStabilityOutput:
    """Predict 3' UTR mRNA stability with PARADE.

    Args:
        inputs (ParadeStabilityInput): 3' UTR sequences to score.
        config (ParadeStabilityConfig): PARADE runtime and model configuration.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        ParadeStabilityOutput: Per-sequence predicted RNA/gDNA log-ratios.
    """
    # Sequences in one call are stacked into a single batched tensor, so they must share
    # a length; callers with mixed lengths should group by length across separate calls.
    lengths = {len(sequence) for sequence in inputs.sequences}
    if len(lengths) != 1:
        raise ValueError(f"All PARADE sequences in one call must share a length; got {sorted(lengths)}")

    url, md5, filename = resolve_checkpoint_source("stability", config.checkpoint_url, config.checkpoint_md5)

    output_data = ToolInstance.dispatch(
        "parade",
        {
            "operation": "stability",
            "sequences": inputs.sequences,
            "checkpoint_path": config.checkpoint_path,
            "checkpoint_url": url,
            "checkpoint_md5": md5,
            "checkpoint_filename": filename,
            "batch_size": config.batch_size,
            "device": config.device,
            "verbose": config.verbose,
            "seed": config.seed,
        },
        instance=instance,
        config=config,
    )

    log_ratios = output_data["log_ratios"]
    if len(log_ratios) != len(inputs.sequences):
        raise ValueError(f"Expected {len(inputs.sequences)} PARADE stability rows, got {len(log_ratios)}")

    return ParadeStabilityOutput(
        results=[
            ParadeStabilityResult(
                sequence=sequence,
                sequence_length=len(sequence),
                log_ratio=float(log_ratio),
            )
            for sequence, log_ratio in zip(inputs.sequences, log_ratios, strict=True)
        ]
    )
