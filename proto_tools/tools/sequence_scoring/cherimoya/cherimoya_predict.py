"""Cherimoya base-resolution chromatin accessibility and ChIP-seq profile prediction."""

import csv
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from proto_tools.tools.sequence_scoring.cherimoya.catalog import CHERIMOYA_CONTEXT
from proto_tools.tools.sequence_scoring.cherimoya.shared_data_models import (
    CherimoyaModelInfo,
    CherimoyaModelsConfig,
    CherimoyaSequenceInput,
    CherimoyaWindowResult,
    check_row_models,
    checkpoint_specs,
    model_infos,
    window_fields,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseToolOutput, ConfigField, ToolInstance

logger = logging.getLogger(__name__)

CherimoyaProfileValues = Literal["expected_counts", "probabilities"]

# Input:
CherimoyaPredictInput = CherimoyaSequenceInput


class CherimoyaPredictConfig(CherimoyaModelsConfig):
    """Configuration for Cherimoya profile prediction.

    Attributes:
        collection (CherimoyaCollectionKey): Collection the accessions in ``models`` come from.
        models (list[str]): ENCODE experiment accessions from the collection, or paths to
            Cherimoya ``.torch`` checkpoints with the same 2,114 bp in, 1,000 bp out geometry.
        folds (list[CherimoyaFold]): Folds to load for each accession; several are averaged.
            A checkpoint path is one model and ignores ``folds``.
        reverse_complement_average (bool): Average forward and reverse-complement predictions.
        batch_size (int): Sequences per forward pass.
        device (str): Device used for inference.
        values (CherimoyaProfileValues): ``"expected_counts"`` returns predicted reads per
            base; ``"probabilities"`` returns the profile shape, which sums to 1.
    """

    values: CherimoyaProfileValues = ConfigField(
        title="Profile Values",
        default="expected_counts",
        description="Per-base expected reads, or the profile as probabilities that sum to 1.",
    )


class CherimoyaProfile(BaseModel):
    """Predicted profile of one model on one sequence.

    Attributes:
        model (str): Model this profile comes from.
        log_counts (float): Predicted ln(reads + 1) over the output window, mean over folds.
        fold_log_counts (list[float]): ``log_counts`` of each fold before averaging.
        values (list[float]): 1,000 per-base values over ``output_start``..``output_end``;
            a ChIP-seq model's two strands are summed.
    """

    model: str = Field(title="Model", description="Model this profile comes from")
    log_counts: float = Field(title="Log Counts", description="Predicted ln(reads + 1) over the output window")
    fold_log_counts: list[float] = Field(title="Fold Log Counts", description="log_counts of each fold")
    values: list[float] = Field(title="Values", description="1,000 per-base values over the output window")


class CherimoyaPredictionResult(CherimoyaWindowResult):
    """Cherimoya profiles for one input sequence.

    Attributes:
        profiles (list[CherimoyaProfile]): One profile per requested model, in config order.
    """

    profiles: list[CherimoyaProfile] = Field(title="Profiles", description="Per-model profiles, in config order")


class CherimoyaPredictOutput(BaseToolOutput):
    """Output from Cherimoya profile prediction.

    Attributes:
        results (list[CherimoyaPredictionResult]): Per-sequence profiles.
        models (list[CherimoyaModelInfo]): Provenance of each model, in config order.
        values (str): What the profile values are (``"expected_counts"`` or ``"probabilities"``).
        reverse_complement_average (bool): Whether strands were averaged.
    """

    results: list[CherimoyaPredictionResult] = Field(title="Results", description="Per-sequence Cherimoya profiles")
    models: list[CherimoyaModelInfo] = Field(title="Models", description="Provenance of each model, in config order")
    values: str = Field(title="Profile Values", description="expected_counts or probabilities")
    reverse_complement_average: bool = Field(
        title="Reverse-Complement Average", description="Whether forward and reverse-complement were averaged"
    )

    def __len__(self) -> int:
        """Return the number of per-sequence results."""
        return len(self.results)

    def __getitem__(self, index: int) -> CherimoyaPredictionResult:
        """Return a per-sequence result."""
        return self.results[index]

    def __iter__(self) -> Iterator[CherimoyaPredictionResult]:  # type: ignore[override]
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
            data = self.model_dump(include={"results", "models", "values", "reverse_complement_average"})
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            return
        if file_format == "csv":
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["sequence_index", "model", "output_start", "log_counts", "values"])
                for idx, result in enumerate(self.results):
                    for profile in result.profiles:
                        writer.writerow(
                            [idx, profile.model, result.output_start, profile.log_counts, json.dumps(profile.values)]
                        )
            return
        raise ValueError(f"Unsupported format: {file_format}")


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return CherimoyaPredictInput(sequences=["ACGT" * (CHERIMOYA_CONTEXT // 4) + "AC"])  # type: ignore[list-item]


@tool(
    key="cherimoya-predict",
    label="Cherimoya Predict",
    category="sequence_scoring",
    input_class=CherimoyaPredictInput,
    config_class=CherimoyaPredictConfig,
    output_class=CherimoyaPredictOutput,
    description="Predict base-resolution accessibility (CATv1) or ChIP-seq (BATv1) profiles with Cherimoya models",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    max_chunk_size=256,
    cacheable=True,
)
def run_cherimoya_predict(
    inputs: CherimoyaPredictInput,
    config: CherimoyaPredictConfig,
    instance: Any = None,
) -> CherimoyaPredictOutput:
    """Predict base-resolution accessibility profiles with one or more Cherimoya models.

    Args:
        inputs (CherimoyaPredictInput): DNA sequences, each a 2,114 bp window or a longer
            sequence with a ``target_range``.
        config (CherimoyaPredictConfig): Models, folds, profile values, and runtime settings.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        CherimoyaPredictOutput: A 1,000 bp profile per sequence and model.
    """
    windows = inputs.prepared_windows()
    specs = checkpoint_specs(config.models, config.folds, config.collection)

    logger.debug("Using local venv for Cherimoya profile prediction")
    result = ToolInstance.dispatch(
        "cherimoya",
        {
            "operation": "predict",
            "sequences": [window.model_sequence for window in windows],
            "checkpoints": specs,
            "models": config.models,
            "values": config.values,
            "reverse_complement_average": config.reverse_complement_average,
            "batch_size": config.batch_size,
            "device": config.device,
            "verbose": config.verbose,
        },
        instance=instance,
        config=config,
    )
    infos = model_infos(config.models, config.collection, specs, result["applied"])
    rows = result["profiles"]
    if len(rows) != len(windows):
        raise ValueError(f"Expected {len(windows)} Cherimoya profile rows, got {len(rows)}")
    check_row_models(rows, config.models)

    return CherimoyaPredictOutput(
        results=[
            CherimoyaPredictionResult(
                **window_fields(raw.sequence, window), profiles=[CherimoyaProfile(**entry) for entry in row]
            )
            for raw, window, row in zip(inputs.sequences, windows, rows, strict=True)
        ],
        models=infos,
        values=config.values,
        reverse_complement_average=config.reverse_complement_average,
    )
