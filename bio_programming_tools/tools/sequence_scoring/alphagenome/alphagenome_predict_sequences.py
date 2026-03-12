"""AlphaGenome batched raw-sequence prediction tool."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator, List, Union

from pydantic import Field, field_validator

from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils.tool_instance import ToolInstance
from bio_programming_tools.utils.tool_io import BaseToolInput, BaseToolOutput, InputField

from .shared_data_models import (
    AlphaGenomePredictConfig,
    AlphaGenomePredictOutput,
    SUPPORTED_CONTEXT_LENGTHS,
)

logger = logging.getLogger(__name__)


def validate_raw_sequence(sequence: str) -> str:
    sequence = sequence.strip().upper()
    if not sequence:
        raise ValueError("sequence cannot be empty")
    if not set(sequence) <= set("ACGTN"):
        raise ValueError("sequence must only contain DNA bases A/C/G/T/N")
    if len(sequence) not in SUPPORTED_CONTEXT_LENGTHS:
        supported = ", ".join(str(length) for length in sorted(SUPPORTED_CONTEXT_LENGTHS))
        raise ValueError(
            "sequence length must match a supported AlphaGenome context length "
            f"({supported} bp)"
        )
    return sequence


# ============================================================================
# Data Models
# ============================================================================

class AlphaGenomePredictSequencesInput(BaseToolInput):
    """Input for batched AlphaGenome raw-sequence prediction.

    Attributes:
        sequences (List[str]): Raw DNA sequences (A/C/G/T/N characters).
            A single string is auto-wrapped into a list. Each sequence
            must match a supported context length.
    """

    sequences: List[str] = InputField(description="Raw DNA sequences for prediction")

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, value: Any) -> list:
        if value is None:
            raise ValueError("sequences cannot be None")
        if isinstance(value, str):
            value = [value]
        if not value:
            raise ValueError("sequences cannot be empty")
        return value

    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, sequences: List[str]) -> List[str]:
        return [validate_raw_sequence(sequence) for sequence in sequences]


class AlphaGenomePredictSequencesOutput(BaseToolOutput):
    """Output from batched AlphaGenome sequence prediction.

    Attributes:
        results (List[AlphaGenomePredictOutput]): Per-sequence prediction outputs.
    """

    results: List[AlphaGenomePredictOutput] = Field(
        description="Per-sequence AlphaGenome prediction outputs",
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


AlphaGenomePredictSequencesConfig = AlphaGenomePredictConfig


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input():
    """Minimal valid input for testing and examples."""
    return AlphaGenomePredictSequencesInput(sequences=["A" * 16384])


@tool(
    key="alphagenome-predict-sequences",
    label="AlphaGenome Predict Sequences",
    category="sequence_scoring",
    input_class=AlphaGenomePredictSequencesInput,
    config_class=AlphaGenomePredictSequencesConfig,
    output_class=AlphaGenomePredictSequencesOutput,
    description="Predict genomic signals from batched raw DNA sequences using AlphaGenome",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="sequences",
    iterable_output_field="results",
    cacheable=True,
)
def run_alphagenome_predict_sequences(
    inputs: AlphaGenomePredictSequencesInput,
    config: AlphaGenomePredictSequencesConfig | None = None,
    instance=None,
) -> AlphaGenomePredictSequencesOutput:
    """Predict genomic features from batched raw DNA sequences using AlphaGenome."""
    dispatch_result = ToolInstance.dispatch(
        "alphagenome",
        {
            "operation": "predict_sequences",
            "sequences": inputs.sequences,
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
            chromosome="sequence",
            interval_start=0,
            interval_end=len(sequence),
            requested_outputs=config.requested_outputs,
            result={"predictions": prediction},
        )
        for sequence, prediction in zip(inputs.sequences, predictions, strict=True)
    ]
    return AlphaGenomePredictSequencesOutput(results=outputs)
