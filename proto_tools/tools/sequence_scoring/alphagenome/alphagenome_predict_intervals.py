"""proto_tools/tools/sequence_scoring/alphagenome/alphagenome_predict_intervals.py

AlphaGenome batched interval prediction tool."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator, List, Union

from pydantic import Field, field_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseToolInput,
    BaseToolOutput,
    InputField,
    ToolInstance,
    require_hf_token,
)

from .shared_data_models import (
    AlphaGenomeInterval,
    AlphaGenomePredictConfig,
    AlphaGenomePredictOutput,
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

    intervals: List[AlphaGenomeInterval] = InputField(
        description="Genomic intervals for prediction",
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


class AlphaGenomePredictIntervalsOutput(BaseToolOutput):
    """Output from batched AlphaGenome interval prediction.

    Attributes:
        results (list[AlphaGenomePredictOutput]): Per-interval prediction outputs.
    """

    results: List[AlphaGenomePredictOutput] = Field(
        description="Per-interval AlphaGenome prediction outputs",
    )

    @property
    def output_format_options(self) -> List[str]:
        return ["json", "npy"]

    @property
    def output_format_default(self) -> str:
        return "json"

    def _export_output(self, export_path: Union[Path, str], file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")
        payload = self.model_dump(mode="json")

        if file_format == "json":
            with open(path, "w") as handle:
                json.dump(payload, handle, indent=2)
            return

        if file_format == "npy":
            import numpy as np

            np.save(path, payload)
            return

        raise ValueError(f"Unsupported format: {file_format}")

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index: int) -> AlphaGenomePredictOutput:
        return self.results[index]

    def __iter__(self) -> Iterator[AlphaGenomePredictOutput]:
        return iter(self.results)


AlphaGenomePredictIntervalsConfig = AlphaGenomePredictConfig


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input():
    """Minimal valid input for testing and examples."""
    return AlphaGenomePredictIntervalsInput(
        intervals=[AlphaGenomeInterval(chromosome="chr1", interval_start=0, interval_end=196608)]
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
    config: AlphaGenomePredictIntervalsConfig | None = None,
    instance=None,
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
