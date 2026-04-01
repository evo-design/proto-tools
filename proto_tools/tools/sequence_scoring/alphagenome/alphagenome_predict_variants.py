"""proto_tools/tools/sequence_scoring/alphagenome/alphagenome_predict_variants.py.

AlphaGenome batched variant-effect prediction tool.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from proto_tools.tools.sequence_scoring.alphagenome.shared_data_models import (
    AlphaGenomePredictConfig,
    AlphaGenomePredictOutput,
    AlphaGenomeVariant,
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


class AlphaGenomePredictVariantsInput(BaseToolInput):
    """Input for batched AlphaGenome variant-effect prediction.

    Attributes:
        variants (list[AlphaGenomeVariant]): Variants to predict.
            A single variant is auto-wrapped into a list.
    """

    variants: list[AlphaGenomeVariant] = InputField(
        description="Variants (with intervals) for prediction",
    )

    @field_validator("variants", mode="before")
    @classmethod
    def normalize_variants(cls, value: Any) -> list[Any]:
        """Validate and normalize variant specifications from raw input."""
        if value is None:
            raise ValueError("variants cannot be None")
        if not isinstance(value, list):
            value = [value]
        if not value:
            raise ValueError("variants cannot be empty")
        return value  # type: ignore[no-any-return]


class AlphaGenomePredictVariantsOutput(BaseToolOutput):
    """Output from batched AlphaGenome variant-effect prediction.

    Attributes:
        results (list[AlphaGenomePredictOutput]): Per-variant prediction outputs.
    """

    results: list[AlphaGenomePredictOutput] = Field(
        description="Per-variant AlphaGenome prediction outputs",
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


AlphaGenomePredictVariantsConfig = AlphaGenomePredictConfig


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return AlphaGenomePredictVariantsInput(
        variants=[
            AlphaGenomeVariant(
                chromosome="chr1",
                interval_start=0,
                interval_end=196608,
                variant_position=100000,
                reference_bases="A",
                alternate_bases="G",
            )
        ]
    )


@tool(
    key="alphagenome-predict-variants",
    label="AlphaGenome Predict Variants",
    category="sequence_scoring",
    input_class=AlphaGenomePredictVariantsInput,
    config_class=AlphaGenomePredictVariantsConfig,
    output_class=AlphaGenomePredictVariantsOutput,
    description="Predict variant effects in batch using AlphaGenome",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="variants",
    iterable_output_field="results",
    cacheable=True,
)
def run_alphagenome_predict_variants(
    inputs: AlphaGenomePredictVariantsInput,
    config: AlphaGenomePredictVariantsConfig | None = None,
    instance: Any = None,
) -> AlphaGenomePredictVariantsOutput:
    """Predict variant effects in batch using AlphaGenome."""
    require_hf_token("AlphaGenome", "https://huggingface.co/google/alphagenome-all-folds")

    dispatch_result = ToolInstance.dispatch(
        "alphagenome",
        {
            "operation": "predict_variants",
            "intervals": [
                {
                    "chromosome": v.chromosome,
                    "interval_start": v.interval_start,
                    "interval_end": v.interval_end,
                }
                for v in inputs.variants
            ],
            "variants": [
                {
                    "chromosome": v.chromosome,
                    "variant_position": v.variant_position,
                    "reference_bases": v.reference_bases,
                    "alternate_bases": v.alternate_bases,
                }
                for v in inputs.variants
            ],
            "requested_outputs": config.requested_outputs,  # type: ignore[union-attr]
            "ontology_terms": config.ontology_terms,  # type: ignore[union-attr]
            "organism": config.organism,  # type: ignore[union-attr]
            "model_version": config.model_version,  # type: ignore[union-attr]
            "device": config.device,  # type: ignore[union-attr]
        },
        instance=instance,
        config=config,
    )

    predictions = dispatch_result["predictions"]
    outputs = [
        AlphaGenomePredictOutput(
            chromosome=v.chromosome,
            interval_start=v.interval_start,
            interval_end=v.interval_end,
            requested_outputs=config.requested_outputs,  # type: ignore[union-attr]
            result={"predictions": prediction},
            variant={
                "position": v.variant_position,
                "reference_bases": v.reference_bases,
                "alternate_bases": v.alternate_bases,
            },
        )
        for v, prediction in zip(inputs.variants, predictions, strict=True)
    ]
    return AlphaGenomePredictVariantsOutput(results=outputs)
