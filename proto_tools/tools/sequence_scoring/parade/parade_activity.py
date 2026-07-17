"""PARADE cell-type-specific UTR activity prediction."""

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from proto_tools.tools.sequence_scoring.parade.shared_data_models import (
    PARADE_CELL_TYPES,
    ParadeActivityMetrics,
    ParadeCellType,
    ParadeCheckpointConfig,
    ParadeConstructType,
    ParadeSequenceInput,
    resolve_checkpoint_source,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseToolOutput, ConfigField, ToolInstance

# Input:
ParadeActivityInput = ParadeSequenceInput


class ParadeActivityConfig(ParadeCheckpointConfig):
    """Configuration for PARADE UTR activity scoring.

    Attributes:
        construct_type (ParadeConstructType): Which UTR model to use — ``"utr5"``
            (5' UTR) or ``"utr3"`` (3' UTR). Selects the checkpoint, the cell-code
            panel, and the fixed reporter flanks the model was trained with.
        cell_types (list[ParadeCellType]): PARADE cell codes to return. Empty means
            the full panel for ``construct_type``. Requested codes must belong to
            that panel.
        checkpoint_path (str): Optional local override path to a PARADE ``.ckpt``.
            Leave empty to download the pinned upstream checkpoint.
        checkpoint_url (str): Optional HTTPS override for the checkpoint download.
            Leave empty to use the pinned per-target URL.
        checkpoint_md5 (str): Optional MD5 override for the downloaded checkpoint.
            Leave empty to use the pinned per-target checksum.
        batch_size (int): Number of sequences to run per GPU batch.
    """

    construct_type: ParadeConstructType = ConfigField(
        title="Construct Type",
        default="utr5",
        description="UTR model to use: 'utr5' (5' UTR) or 'utr3' (3' UTR).",
        reload_on_change=True,
    )
    cell_types: list[ParadeCellType] = ConfigField(
        title="Cell Types",
        default_factory=list,
        description="PARADE cell codes to return; empty returns the full panel for the construct type.",
    )

    @field_validator("cell_types", mode="before")
    @classmethod
    def normalize_cell_types(cls, value: Any) -> list[Any]:
        """Normalize a single cell code to a list."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value  # type: ignore[no-any-return]

    @model_validator(mode="after")
    def validate_cell_types(self) -> "ParadeActivityConfig":
        """Resolve the default panel and validate requested codes against it."""
        panel = PARADE_CELL_TYPES[self.construct_type]
        if not self.cell_types:
            self.cell_types = list(panel)
            return self
        if len(set(self.cell_types)) != len(self.cell_types):
            raise ValueError("cell_types must be unique")
        unsupported = [code for code in self.cell_types if code not in panel]
        if unsupported:
            raise ValueError(
                f"cell_types {unsupported} are not in the {self.construct_type} panel {list(panel)}"
            )
        return self


class ParadeActivityResult(BaseModel):
    """Per-sequence PARADE activity result.

    Attributes:
        sequence (str): UTR sequence that was scored (DNA alphabet).
        sequence_length (int): Length of the scored sequence.
        scores (ParadeActivityMetrics): Predicted activity keyed by cell code.
    """

    sequence: str = Field(title="Sequence", description="UTR sequence scored by PARADE")
    sequence_length: int = Field(title="Sequence Length", description="Length of the scored UTR sequence")
    scores: ParadeActivityMetrics = Field(
        title="Activity Scores", description="PARADE activity predictions keyed by cell code"
    )


class ParadeActivityOutput(BaseToolOutput):
    """Output from PARADE UTR activity scoring.

    Attributes:
        results (list[ParadeActivityResult]): Per-sequence PARADE predictions.
        construct_type (str): UTR model used (``"utr5"`` or ``"utr3"``).
        cell_types (list[str]): Cell codes included in each result's ``scores``.
    """

    results: list[ParadeActivityResult] = Field(
        title="Results", description="Per-sequence PARADE activity scoring results"
    )
    construct_type: str = Field(title="Construct Type", description="UTR model used for scoring")
    cell_types: list[str] = Field(title="Cell Types", description="Cell codes included in each score dictionary")

    def __len__(self) -> int:
        """Return the number of per-sequence results."""
        return len(self.results)

    def __getitem__(self, index: int) -> ParadeActivityResult:
        """Return a per-sequence result."""
        return self.results[index]

    def __iter__(self) -> Iterator[ParadeActivityResult]:  # type: ignore[override]
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
                        "scores": dict(result.scores.items()),
                    }
                    for result in self.results
                ],
                "construct_type": self.construct_type,
                "cell_types": self.cell_types,
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            return
        if file_format == "csv":
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["sequence_index", "sequence", "sequence_length", *self.cell_types])
                for idx, result in enumerate(self.results):
                    writer.writerow(
                        [
                            idx,
                            result.sequence,
                            result.sequence_length,
                            *[result.scores[cell_type] for cell_type in self.cell_types],
                        ]
                    )
            return
        raise ValueError(f"Unsupported format: {file_format}")


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return ParadeActivityInput(sequences=["ACGT" * 12 + "AC"])


@tool(
    key="parade-activity",
    label="PARADE UTR Activity",
    category="sequence_scoring",
    input_class=ParadeActivityInput,
    config_class=ParadeActivityConfig,
    output_class=ParadeActivityOutput,
    metrics_class=ParadeActivityMetrics,
    description="Predict cell-type-specific 5'/3' UTR activity with the PARADE LegNet model",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    cacheable=True,
)
def run_parade_activity(
    inputs: ParadeActivityInput,
    config: ParadeActivityConfig,
    instance: Any = None,
) -> ParadeActivityOutput:
    """Predict cell-type-specific UTR activity with PARADE.

    Args:
        inputs (ParadeActivityInput): UTR sequences to score.
        config (ParadeActivityConfig): PARADE runtime and model configuration.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        ParadeActivityOutput: Per-sequence PARADE activity predictions keyed by cell code.
    """
    # Sequences in one call are stacked into a single batched tensor, so they must share
    # a length; callers with mixed lengths should group by length across separate calls.
    lengths = {len(sequence) for sequence in inputs.sequences}
    if len(lengths) != 1:
        raise ValueError(f"All PARADE sequences in one call must share a length; got {sorted(lengths)}")

    url, md5, filename = resolve_checkpoint_source(config.construct_type, config.checkpoint_url, config.checkpoint_md5)

    output_data = ToolInstance.dispatch(
        "parade",
        {
            "operation": "activity",
            "sequences": inputs.sequences,
            "construct_type": config.construct_type,
            "cell_types": list(config.cell_types),
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

    score_rows = output_data["scores"]
    if len(score_rows) != len(inputs.sequences):
        raise ValueError(f"Expected {len(inputs.sequences)} PARADE score rows, got {len(score_rows)}")

    return ParadeActivityOutput(
        results=[
            ParadeActivityResult(
                sequence=sequence,
                sequence_length=len(sequence),
                scores=ParadeActivityMetrics.model_validate(
                    {cell_type: float(scores[cell_type]) for cell_type in config.cell_types}
                ),
            )
            for sequence, scores in zip(inputs.sequences, score_rows, strict=True)
        ],
        construct_type=config.construct_type,
        cell_types=list(config.cell_types),
    )
