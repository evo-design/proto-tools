"""proto_tools/tools/sequence_scoring/alphagenome/alphagenome_predict_sequences.py.

AlphaGenome batched raw-sequence prediction tool.
"""

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from proto_tools.tools.sequence_scoring.alphagenome.shared_data_models import (
    SUPPORTED_CONTEXT_LENGTHS,
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


def validate_raw_sequence(sequence: str) -> str:
    """Validate and normalize a raw DNA sequence for AlphaGenome prediction."""
    sequence = sequence.strip().upper()
    if not sequence:
        raise ValueError("sequence cannot be empty")
    if not set(sequence) <= set("ACGTN"):
        raise ValueError("sequence must only contain DNA bases A/C/G/T/N")
    if len(sequence) not in SUPPORTED_CONTEXT_LENGTHS:
        supported = ", ".join(str(length) for length in sorted(SUPPORTED_CONTEXT_LENGTHS))
        raise ValueError(f"sequence length must match a supported AlphaGenome context length ({supported} bp)")
    return sequence


# ============================================================================
# Data Models
# ============================================================================


class AlphaGenomePredictSequencesInput(BaseToolInput):
    """Input for batched AlphaGenome raw-sequence prediction.

    Attributes:
        sequences (list[str]): Raw DNA sequences (A/C/G/T/N characters).
            A single string is auto-wrapped into a list. Each sequence
            must match a supported context length.
    """

    sequences: list[str] = InputField(title="Sequences", description="Raw DNA sequences for prediction")

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, value: Any) -> list[Any]:
        """Validate and normalize sequence specifications from raw input."""
        if value is None:
            raise ValueError("sequences cannot be None")
        if isinstance(value, str):
            value = [value]
        if not value:
            raise ValueError("sequences cannot be empty")
        return value  # type: ignore[no-any-return]

    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, sequences: list[str]) -> list[str]:
        """Validate that all sequences meet AlphaGenome length requirements."""
        return [validate_raw_sequence(sequence) for sequence in sequences]


class AlphaGenomePredictSequencesOutput(BaseToolOutput):
    """Output from batched AlphaGenome sequence prediction.

    Attributes:
        results (list[AlphaGenomePredictOutput]): Per-sequence prediction outputs.
    """

    results: list[AlphaGenomePredictOutput] = Field(
        title="Results",
        description="Per-sequence AlphaGenome prediction outputs",
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


AlphaGenomePredictSequencesConfig = AlphaGenomePredictConfig


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
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
    pin_visible_devices=True,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    cacheable=True,
)
def run_alphagenome_predict_sequences(
    inputs: AlphaGenomePredictSequencesInput,
    config: AlphaGenomePredictSequencesConfig,
    instance: Any = None,
) -> AlphaGenomePredictSequencesOutput:
    """Predict genomic features from batched raw DNA sequences using AlphaGenome."""
    require_hf_token("AlphaGenome", "https://huggingface.co/google/alphagenome-all-folds")

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
