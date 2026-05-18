"""proto_tools/tools/sequence_scoring/alphagenome/alphagenome_score_intervals.py.

AlphaGenome batched interval scoring tool.
"""

import csv
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from proto_tools.tools.sequence_scoring.alphagenome.shared_data_models import (
    DEFAULT_ALPHAGENOME_MODEL_VERSION,
    AlphaGenomeInterval,
    AlphaGenomeScoreOutput,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
    require_hf_token,
)

logger = logging.getLogger(__name__)

IntervalScorerName = Literal["RNA_SEQ"]


# ============================================================================
# Data Models
# ============================================================================


class AlphaGenomeScoreIntervalsInput(BaseToolInput):
    """Input for batched AlphaGenome interval scoring.

    Attributes:
        intervals (list[AlphaGenomeInterval]): Genomic intervals to score.
            A single interval is auto-wrapped into a list.
    """

    intervals: list[AlphaGenomeInterval] = InputField(
        description="Genomic intervals for scoring",
    )

    @field_validator("intervals", mode="before")
    @classmethod
    def normalize_intervals(cls, value: Any) -> list[Any]:
        """Validate and normalize interval specifications from raw input."""
        if value is None:
            raise ValueError("intervals cannot be None")
        if not isinstance(value, list):
            value = [value]
        if not value:
            raise ValueError("intervals cannot be empty")
        return value  # type: ignore[no-any-return]


class AlphaGenomeScoreIntervalsOutput(BaseToolOutput):
    """Output from batched AlphaGenome interval scoring.

    Attributes:
        results (list[AlphaGenomeScoreOutput]): Per-interval score outputs.
    """

    results: list[AlphaGenomeScoreOutput] = Field(
        description="Per-interval AlphaGenome score outputs",
    )

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

    def __iter__(self) -> Iterator[AlphaGenomeScoreOutput]:  # type: ignore[override]
        return iter(self.results)


class AlphaGenomeScoreIntervalsConfig(BaseConfig):
    """Configuration for batched AlphaGenome interval scoring.

    Attributes:
        model_version (str): AlphaGenome Hugging Face model version.
        interval_scorers (list[IntervalScorerName] | None): Scorer names from the library's
            ``RECOMMENDED_INTERVAL_SCORERS``. ``None`` uses all recommended.
        organism (Literal['human', 'mouse']): Organism for predictions.
        device (str): Device to run inference on.
    """

    model_version: str = ConfigField(
        title="Model Version",
        default=DEFAULT_ALPHAGENOME_MODEL_VERSION,
        description="AlphaGenome Hugging Face model version",
        reload_on_change=True,
    )
    interval_scorers: list[IntervalScorerName] | None = ConfigField(
        title="Interval Scorers",
        default=None,
        description="Scorer names to use. None uses all recommended scorers.",
    )
    organism: Literal["human", "mouse"] = ConfigField(
        title="Organism",
        default="human",
        description="Organism for AlphaGenome predictions",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run AlphaGenome inference on",
        include_in_key=False,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
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
    config: AlphaGenomeScoreIntervalsConfig,
    instance: Any = None,
) -> AlphaGenomeScoreIntervalsOutput:
    """Score genomic intervals in batch using AlphaGenome interval scorers."""
    require_hf_token("AlphaGenome", "https://huggingface.co/google/alphagenome-all-folds")

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
    outputs = [AlphaGenomeScoreOutput(scores=score_list) for score_list in scores]
    return AlphaGenomeScoreIntervalsOutput(results=outputs)
