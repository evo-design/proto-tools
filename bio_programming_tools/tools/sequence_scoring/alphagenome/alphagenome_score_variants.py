"""AlphaGenome batched variant scoring tool."""
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
    AlphaGenomeScoreOutput,
    AlphaGenomeVariant,
    VariantScorerName,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================


class AlphaGenomeScoreVariantsInput(BaseToolInput):
    """Input for batched AlphaGenome variant scoring.

    Attributes:
        variants (List[AlphaGenomeVariant]): Variants to score.
            A single variant is auto-wrapped into a list.
    """

    variants: List[AlphaGenomeVariant] = Field(
        description="Variants (with intervals) for scoring",
    )

    @field_validator("variants", mode="before")
    @classmethod
    def normalize_variants(cls, value: Any) -> list:
        if value is None:
            raise ValueError("variants cannot be None")
        if not isinstance(value, list):
            value = [value]
        if not value:
            raise ValueError("variants cannot be empty")
        return value


class AlphaGenomeScoreVariantsOutput(BaseToolOutput):
    """Output from batched AlphaGenome variant scoring.

    Attributes:
        results (List[AlphaGenomeScoreOutput]): Per-variant score outputs.
    """

    results: List[AlphaGenomeScoreOutput] = Field(
        description="Per-variant AlphaGenome score outputs",
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


class AlphaGenomeScoreVariantsConfig(BaseConfig):
    """Configuration for batched AlphaGenome variant scoring.

    Attributes:
        model_version (str): AlphaGenome Hugging Face model version.
        variant_scorers (Optional[List[str]]): Scorer names from the library's
            ``RECOMMENDED_VARIANT_SCORERS``. ``None`` uses all recommended.
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
    variant_scorers: Optional[List[VariantScorerName]] = ConfigField(
        title="Variant Scorers",
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
    return AlphaGenomeScoreVariantsInput(
        variants=[AlphaGenomeVariant(
            chromosome="chr1", interval_start=0, interval_end=196608,
            variant_position=100000, reference_bases="A", alternate_bases="G",
        )]
    )


@tool(
    key="alphagenome-score-variants",
    label="AlphaGenome Score Variants",
    category="sequence_scoring",
    input_class=AlphaGenomeScoreVariantsInput,
    config_class=AlphaGenomeScoreVariantsConfig,
    output_class=AlphaGenomeScoreVariantsOutput,
    description="Score variant effects in batch with AlphaGenome variant scorers",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="variants",
    iterable_output_field="results",
    cacheable=True,
)
def run_alphagenome_score_variants(
    inputs: AlphaGenomeScoreVariantsInput,
    config: AlphaGenomeScoreVariantsConfig | None = None,
    instance=None,
) -> AlphaGenomeScoreVariantsOutput:
    """Score variant effects in batch using AlphaGenome variant scorers."""
    dispatch_result = ToolInstance.dispatch(
        "alphagenome",
        {
            "operation": "score_variants",
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
            "variant_scorers": config.variant_scorers,
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
    return AlphaGenomeScoreVariantsOutput(results=outputs)
