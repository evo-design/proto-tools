"""Cherimoya chromatin accessibility scoring of DNA sequences."""

import csv
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import Field

from proto_tools.tools.sequence_scoring.cherimoya.catalog import CHERIMOYA_CONTEXT
from proto_tools.tools.sequence_scoring.cherimoya.shared_data_models import (
    CherimoyaModelInfo,
    CherimoyaModelsConfig,
    CherimoyaScoreMetrics,
    CherimoyaSequenceInput,
    CherimoyaWindowResult,
    check_row_models,
    checkpoint_specs,
    model_infos,
    window_fields,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseToolOutput, ToolInstance

logger = logging.getLogger(__name__)

# Input:
CherimoyaScoreInput = CherimoyaSequenceInput


class CherimoyaScoreConfig(CherimoyaModelsConfig):
    """Configuration for Cherimoya scoring.

    Attributes:
        collection (CherimoyaCollectionKey): Collection the accessions in ``models`` come from.
        models (list[str]): ENCODE experiment accessions from the collection, or paths to
            Cherimoya ``.torch`` checkpoints with the same 2,114 bp in, 1,000 bp out geometry.
        folds (list[CherimoyaFold]): Folds to load for each accession; several are averaged.
            A checkpoint path is one model and ignores ``folds``.
        reverse_complement_average (bool): Average forward and reverse-complement predictions.
        batch_size (int): Sequences per forward pass.
        device (str): Device used for inference.
    """


class CherimoyaScoreResult(CherimoyaWindowResult):
    """Cherimoya scores for one input sequence.

    Attributes:
        scores (list[CherimoyaScoreMetrics]): One entry per requested model, in config order.
    """

    scores: list[CherimoyaScoreMetrics] = Field(title="Scores", description="Per-model scores, in config order")


class CherimoyaScoreOutput(BaseToolOutput):
    """Output from Cherimoya scoring.

    Attributes:
        results (list[CherimoyaScoreResult]): Per-sequence scores.
        models (list[CherimoyaModelInfo]): Provenance of each model, in config order.
        reverse_complement_average (bool): Whether strands were averaged.
    """

    results: list[CherimoyaScoreResult] = Field(title="Results", description="Per-sequence Cherimoya scores")
    models: list[CherimoyaModelInfo] = Field(title="Models", description="Provenance of each model, in config order")
    reverse_complement_average: bool = Field(
        title="Reverse-Complement Average", description="Whether forward and reverse-complement were averaged"
    )

    def __len__(self) -> int:
        """Return the number of per-sequence results."""
        return len(self.results)

    def __getitem__(self, index: int) -> CherimoyaScoreResult:
        """Return a per-sequence result."""
        return self.results[index]

    def __iter__(self) -> Iterator[CherimoyaScoreResult]:  # type: ignore[override]
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
            data = self.model_dump(include={"results", "models", "reverse_complement_average"})
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            return
        if file_format == "csv":
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["sequence_index", "model", "log_counts", "target_counts", "fold_log_counts", "output_start"]
                )
                for idx, result in enumerate(self.results):
                    for score in result.scores:
                        writer.writerow(
                            [
                                idx,
                                score.model,
                                score["log_counts"],
                                score.get("target_counts"),
                                json.dumps(score.fold_log_counts),
                                result.output_start,
                            ]
                        )
            return
        raise ValueError(f"Unsupported format: {file_format}")


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return CherimoyaScoreInput(sequences=["ACGT" * (CHERIMOYA_CONTEXT // 4) + "AC"])  # type: ignore[list-item]


@tool(
    key="cherimoya-score",
    label="Cherimoya Score",
    category="sequence_scoring",
    input_class=CherimoyaScoreInput,
    config_class=CherimoyaScoreConfig,
    output_class=CherimoyaScoreOutput,
    metrics_class=CherimoyaScoreMetrics,
    description="Predict DNase-seq or ATAC-seq read counts for DNA sequences with Cherimoya (CATv1) models",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    max_chunk_size=32,
    cacheable=True,
)
def run_cherimoya_score(
    inputs: CherimoyaScoreInput,
    config: CherimoyaScoreConfig,
    instance: Any = None,
) -> CherimoyaScoreOutput:
    """Score DNA sequences with one or more Cherimoya models.

    Args:
        inputs (CherimoyaScoreInput): DNA sequences, each a 2,114 bp window or a longer
            sequence with a ``target_range``.
        config (CherimoyaScoreConfig): Models, folds, and runtime settings.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        CherimoyaScoreOutput: Per-sequence log counts (and target counts when a
            ``target_range`` is set) for every requested model.
    """
    windows = inputs.prepared_windows()
    target_spans = None
    if windows[0].target_start is not None:
        target_spans = []
        for window in windows:
            assert window.target_start is not None and window.target_end is not None  # noqa: S101 -- all-or-none
            target_spans.append((window.target_start - window.output_start, window.target_end - window.output_start))
    specs = checkpoint_specs(config.models, config.folds, config.collection)

    logger.debug("Using local venv for Cherimoya scoring")
    result = ToolInstance.dispatch(
        "cherimoya",
        {
            "operation": "score",
            "sequences": [window.model_sequence for window in windows],
            "checkpoints": specs,
            "models": config.models,
            "target_spans": target_spans,
            "reverse_complement_average": config.reverse_complement_average,
            "batch_size": config.batch_size,
            "device": config.device,
            "verbose": config.verbose,
        },
        instance=instance,
        config=config,
    )
    infos = model_infos(config.models, config.collection, specs, result["applied"])
    rows = result["scores"]
    if len(rows) != len(windows):
        raise ValueError(f"Expected {len(windows)} Cherimoya score rows, got {len(rows)}")
    check_row_models(rows, config.models)

    results = []
    for raw, window, row in zip(inputs.sequences, windows, rows, strict=True):
        scores = [
            CherimoyaScoreMetrics.model_validate(
                {
                    "model": entry["model"],
                    "fold_log_counts": entry["fold_log_counts"],
                    "log_counts": float(entry["log_counts"]),
                    "target_counts": entry.get("target_counts"),
                }
            )
            for entry in row
        ]
        results.append(CherimoyaScoreResult(**window_fields(raw.sequence, window), scores=scores))

    return CherimoyaScoreOutput(
        results=results,
        models=infos,
        reverse_complement_average=config.reverse_complement_average,
    )
