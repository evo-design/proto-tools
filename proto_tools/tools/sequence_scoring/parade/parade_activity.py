"""PARADE cell-type-specific UTR activity prediction."""

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, computed_field, field_validator

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
            (5' UTR) or ``"utr3"`` (3' UTR). Selects the checkpoint and the cell-code
            panel. Matching the upstream predictor, the model scores the bare insert
            (no reporter flanks are added).
        cell_types (list[ParadeCellType]): PARADE cell codes to return. Empty means
            the full panel for ``construct_type``. Requested codes must belong to
            that panel.
        checkpoint (str): Optional override for the pinned checkpoint — a local ``.ckpt`` path or an
            ``https`` link (a schemeless ``host.tld/path`` is accepted). Caller overrides run on local
            devices only (rejected on ``device="proto"``). Empty uses the pinned per-target checkpoint.
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

    @field_validator("construct_type")
    @classmethod
    def validate_construct_type(cls, value: ParadeConstructType, info: ValidationInfo) -> ParadeConstructType:
        """Reject a construct update that would invalidate explicitly selected cells."""
        cell_types = info.data.get("cell_types", [])
        unsupported = [code for code in cell_types if code not in PARADE_CELL_TYPES[value]]
        if unsupported:
            raise ValueError(f"cell_types {unsupported} are not in the {value} panel {list(PARADE_CELL_TYPES[value])}")
        return value

    @field_validator("cell_types")
    @classmethod
    def validate_cell_types(cls, value: list[ParadeCellType], info: ValidationInfo) -> list[ParadeCellType]:
        """Validate explicitly requested codes while preserving empty as the full-panel sentinel."""
        if len(set(value)) != len(value):
            raise ValueError("cell_types must be unique")
        construct_type = info.data.get("construct_type", "utr5")
        panel = PARADE_CELL_TYPES[construct_type]
        unsupported = [code for code in value if code not in panel]
        if unsupported:
            raise ValueError(f"cell_types {unsupported} are not in the {construct_type} panel {list(panel)}")
        return value

    @property
    def resolved_cell_types(self) -> list[ParadeCellType]:
        """Requested cell codes, or the full panel when ``cell_types`` is empty."""
        return list(self.cell_types or PARADE_CELL_TYPES[self.construct_type])


class ParadeActivityResult(BaseModel):
    """Per-sequence PARADE activity result.

    Attributes:
        sequence (str): UTR sequence that was scored (DNA alphabet).
        sequence_length (int): Length of the scored sequence.
        construct_type (ParadeConstructType): UTR model that produced these scores.
            Carried per-result so the provenance survives iterable-cache reconstruction.
        scores (ParadeActivityMetrics): Predicted activity keyed by cell code.
    """

    sequence: str = Field(title="Sequence", description="UTR sequence scored by PARADE")
    sequence_length: int = Field(title="Sequence Length", description="Length of the scored UTR sequence")
    construct_type: ParadeConstructType = Field(
        title="Construct Type", description="UTR model that produced these scores"
    )
    scores: ParadeActivityMetrics = Field(
        title="Activity Scores", description="PARADE activity predictions keyed by cell code"
    )


class ParadeActivityOutput(BaseToolOutput):
    """Output from PARADE UTR activity scoring.

    Attributes:
        results (list[ParadeActivityResult]): Per-sequence PARADE predictions. The only
            stored field, so the output survives the framework's iterable-cache
            reconstruction (which preserves only the iterable field).
        construct_type (str): UTR model used, derived from ``results`` (computed field).
        cell_types (list[str]): Cell codes in each result, derived from ``results``
            (computed field).
    """

    results: list[ParadeActivityResult] = Field(
        default_factory=list, title="Results", description="Per-sequence PARADE activity scoring results"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def construct_type(self) -> str:
        """UTR model that produced the scores, derived from ``results``.

        A computed field (not stored) so it persists through ``model_dump``/serialization
        and stays correct after a full iterable-cache hit, where only ``results`` is kept.
        """
        return self.results[0].construct_type if self.results else ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cell_types(self) -> list[str]:
        """Cell codes present in each result, derived from ``results`` (computed field)."""
        return [code for code, _ in self.results[0].scores.items()] if self.results else []

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
    max_chunk_size=32,
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
    inputs = ParadeActivityInput.model_validate(inputs.model_dump())
    # Revalidate a dumped copy so in-place mutation of the list field cannot bypass
    # Pydantic's assignment hooks between config construction and execution.
    config = ParadeActivityConfig.model_validate(config.model_dump())
    # Sequences may have mixed lengths; the standalone batches them per length group, so this
    # is safe under the framework's per-item iterable cache (partial cache hits pass any subset).
    checkpoint_path, url, md5, filename = resolve_checkpoint_source(config.construct_type, config.checkpoint)

    output_data = ToolInstance.dispatch(
        "parade",
        {
            "operation": "activity",
            "sequences": inputs.sequences,
            "construct_type": config.construct_type,
            "cell_types": config.resolved_cell_types,
            "checkpoint_path": checkpoint_path,
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
                construct_type=config.construct_type,
                scores=ParadeActivityMetrics.model_validate(
                    {cell_type: float(scores[cell_type]) for cell_type in config.resolved_cell_types}
                ),
            )
            for sequence, scores in zip(inputs.sequences, score_rows, strict=True)
        ],
    )
