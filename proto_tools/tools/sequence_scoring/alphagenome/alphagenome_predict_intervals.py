"""proto_tools/tools/sequence_scoring/alphagenome/alphagenome_predict_intervals.py.

AlphaGenome batched interval prediction tool.
"""

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from proto_tools.tools.sequence_scoring.alphagenome.shared_data_models import (
    AlphaGenomeInterval,
    AlphaGenomePredictConfig,
    AlphaGenomePredictOutput,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseToolInput,
    BaseToolOutput,
    InputField,
    ToolInstance,
    require_hf_token,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================


class AlphaGenomePredictIntervalsInput(BaseToolInput):
    """Input for batched AlphaGenome interval prediction.

    Attributes:
        intervals (list[AlphaGenomeInterval]): Genomic intervals to predict.
            A single interval is auto-wrapped into a list.
    """

    intervals: list[AlphaGenomeInterval] = InputField(
        title="Intervals",
        description="Genomic intervals for prediction",
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


class AlphaGenomePredictIntervalsOutput(BaseToolOutput):
    """Output from batched AlphaGenome interval prediction.

    Attributes:
        results (list[AlphaGenomePredictOutput]): Per-interval prediction outputs.
    """

    results: list[AlphaGenomePredictOutput] = Field(
        title="Results",
        description="Per-interval AlphaGenome prediction outputs",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json", "npy"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: Path | str, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")
        payload = self.model_dump(mode="json")

        if file_format == "json":
            with open(path, "w") as handle:
                json.dump(payload, handle, indent=2)
            return

        if file_format == "npy":
            import numpy as np

            np.save(path, payload)  # type: ignore[arg-type]
            return

        raise ValueError(f"Unsupported format: {file_format}")

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index: int) -> AlphaGenomePredictOutput:
        return self.results[index]

    def __iter__(self) -> Iterator[AlphaGenomePredictOutput]:  # type: ignore[override]
        return iter(self.results)


AlphaGenomePredictIntervalsConfig = AlphaGenomePredictConfig


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples.

    Uses the smallest supported context length (16,384 bp) to keep test runs
    fast. See ``alphagenome_predict_variants.example_input`` for the rationale.
    """
    return AlphaGenomePredictIntervalsInput(
        intervals=[AlphaGenomeInterval(chromosome="chr1", interval_start=0, interval_end=16_384)]
    )


@tool(
    key="alphagenome-predict-intervals",
    label="AlphaGenome Predict Intervals",
    category="sequence_scoring",
    input_class=AlphaGenomePredictIntervalsInput,
    config_class=AlphaGenomePredictIntervalsConfig,
    output_class=AlphaGenomePredictIntervalsOutput,
    description="Predict genomic signals for batched intervals using AlphaGenome",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="intervals",
    iterable_output_field="results",
    cacheable=True,
)
def run_alphagenome_predict_intervals(
    inputs: AlphaGenomePredictIntervalsInput,
    config: AlphaGenomePredictIntervalsConfig,
    instance: Any = None,
) -> AlphaGenomePredictIntervalsOutput:
    """Predict genomic features for batched intervals using AlphaGenome."""
    require_hf_token("AlphaGenome", "https://huggingface.co/google/alphagenome-all-folds")

    dispatch_result = ToolInstance.dispatch(
        "alphagenome",
        {
            "operation": "predict_intervals",
            "intervals": [
                {
                    "chromosome": item.chromosome,
                    "interval_start": item.interval_start,
                    "interval_end": item.interval_end,
                }
                for item in inputs.intervals
            ],
            "requested_outputs": config.requested_outputs,
            "ontology_terms": config.ontology_terms,
            "organism": config.organism,
            "model_version": config.model_version,
            "device": config.device,
        },
        instance=instance,
        config=config,
    )

    predictions = dispatch_result["predictions"]
    outputs = [
        AlphaGenomePredictOutput(
            chromosome=item.chromosome,
            interval_start=item.interval_start,
            interval_end=item.interval_end,
            requested_outputs=config.requested_outputs,
            result={"predictions": prediction},
        )
        for item, prediction in zip(inputs.intervals, predictions, strict=True)
    ]
    return AlphaGenomePredictIntervalsOutput(results=outputs)
