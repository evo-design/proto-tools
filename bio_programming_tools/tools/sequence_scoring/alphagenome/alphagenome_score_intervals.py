"""AlphaGenome batched interval scoring tool."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Iterator, List, Literal, Optional, Union

from pydantic import Field, field_validator

from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig, ConfigField
from bio_programming_tools.utils.tool_instance import ToolInstance
from bio_programming_tools.utils.tool_io import BaseToolInput, BaseToolOutput

from .shared_data_models import (
    DEFAULT_ALPHAGENOME_MODEL_VERSION,
    AlphaGenomeInterval,
    AlphaGenomeScoreOutput,
)

logger = logging.getLogger(__name__)

IntervalScorerName = Literal["RNA_SEQ"]


# ============================================================================
# Data Models
# ============================================================================


class AlphaGenomeScoreIntervalsInput(BaseToolInput):
    """Input for batched AlphaGenome interval scoring.

    Attributes:
        intervals (List[AlphaGenomeInterval]): Genomic intervals to score.
            A single interval is auto-wrapped into a list.
    """

    intervals: List[AlphaGenomeInterval] = Field(
        description="Genomic intervals for scoring",
    )

    @field_validator("intervals", mode="before")
    @classmethod
    def normalize_intervals(cls, value: Any) -> list:
        if value is None:
            raise ValueError("intervals cannot be None")
        if not isinstance(value, list):
            value = [value]
        if not value:
            raise ValueError("intervals cannot be empty")
        return value


class AlphaGenomeScoreIntervalsOutput(BaseToolOutput):
    """Output from batched AlphaGenome interval scoring.

    Attributes:
        results (List[AlphaGenomeScoreOutput]): Per-interval score outputs.
    """

    results: List[AlphaGenomeScoreOutput] = Field(
        description="Per-interval AlphaGenome score outputs",
    )

    @property
    def output_format_options(self) -> List[str]:
        return ["json", "csv"]

    @property
    def output_format_default(self) -> str:
        return "json"

    def _export_output(self, export_path: Union[Path, str], file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "json":
            payload = self.model_dump(mode="json")
            with open(path, "w") as handle:
                json.dump(payload, handle, indent=2)
            return

        if file_format == "csv":
            all_scores = [s for result in self.results for s in result.scores]
            if not all_scores:
                path.write_text("")
                return
            fieldnames = list(all_scores[0].keys())
            with open(path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_scores)
            return

        raise ValueError(f"Unsupported format: {file_format}")

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index: int) -> AlphaGenomeScoreOutput:
        return self.results[index]

    def __iter__(self) -> Iterator[AlphaGenomeScoreOutput]:
        return iter(self.results)


class AlphaGenomeScoreIntervalsConfig(BaseConfig):
    """Configuration for batched AlphaGenome interval scoring.

    Attributes:
        model_version (str): AlphaGenome Hugging Face model version.
        interval_scorers (Optional[List[str]]): Scorer names from the library's
            ``RECOMMENDED_INTERVAL_SCORERS``. ``None`` uses all recommended.
        organism (Literal["human", "mouse"]): Organism for predictions.
        device (str): Device to run inference on.
    """

    model_version: str = ConfigField(
        title="Model Version",
        default=DEFAULT_ALPHAGENOME_MODEL_VERSION,
        description="AlphaGenome Hugging Face model version",
        advanced=True,
        reload_on_change=True,
    )
    interval_scorers: Optional[List[IntervalScorerName]] = ConfigField(
        title="Interval Scorers",
        default=None,
        description="Scorer names to use. None uses all recommended scorers.",
    )
    organism: Literal["human", "mouse"] = ConfigField(
        title="Organism",
        default="human",
        description="Organism for AlphaGenome predictions",
        advanced=True,
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run AlphaGenome inference on",
        hidden=True,
        include_in_key=False,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input():
    """Minimal valid input for testing and examples."""
    return AlphaGenomeScoreIntervalsInput(
        intervals=[AlphaGenomeInterval(chromosome="chr1", interval_start=0, interval_end=196608)]
    )


@tool(
    key="alphagenome-score-intervals",
    label="AlphaGenome Score Intervals",
    category="sequence_scoring",
    input_class=AlphaGenomeScoreIntervalsInput,
    config_class=AlphaGenomeScoreIntervalsConfig,
    output_class=AlphaGenomeScoreIntervalsOutput,
    description="Score genomic intervals in batch with AlphaGenome interval scorers",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="intervals",
    iterable_output_field="results",
    cacheable=True,
)
def run_alphagenome_score_intervals(
    inputs: AlphaGenomeScoreIntervalsInput,
    config: AlphaGenomeScoreIntervalsConfig | None = None,
    instance=None,
) -> AlphaGenomeScoreIntervalsOutput:
    """Score genomic intervals in batch using AlphaGenome interval scorers."""
    dispatch_result = ToolInstance.dispatch(
        "alphagenome",
        {
            "operation": "score_intervals",
            "intervals": [
                {
                    "chromosome": item.chromosome,
                    "interval_start": item.interval_start,
                    "interval_end": item.interval_end,
                }
                for item in inputs.intervals
            ],
            "interval_scorers": config.interval_scorers,
            "organism": config.organism,
            "model_version": config.model_version,
            "device": config.device,
        },
        instance=instance,
        config=config,
    )

    scores = dispatch_result["scores"]
    outputs = [
        AlphaGenomeScoreOutput(scores=score_list)
        for score_list in scores
    ]
    return AlphaGenomeScoreIntervalsOutput(results=outputs)
