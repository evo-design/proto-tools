"""proto_tools/tools/sequence_scoring/alphagenome/alphagenome_score_ism_variants_batch.py.

AlphaGenome batched in-silico mutagenesis (ISM) tool.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator

from proto_tools.tools.sequence_scoring.alphagenome.alphagenome_score_variants import AlphaGenomeScoreVariantsConfig
from proto_tools.tools.sequence_scoring.alphagenome.shared_data_models import (
    AlphaGenomeInterval,
    AlphaGenomeScoreOutput,
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


class AlphaGenomeISM(AlphaGenomeInterval):
    """Input object for a single AlphaGenome in-silico mutagenesis request.

    Attributes:
        chromosome (str): Chromosome identifier, e.g. ``'chr1'``.
        interval_start (int): Interval start (0-based, inclusive).
        interval_end (int): Interval end (0-based, exclusive).
        ism_interval_start (int): ISM sub-interval start (0-based, inclusive).
        ism_interval_end (int): ISM sub-interval end (0-based, exclusive).
        variant_position (int | None): Optional existing variant position
            to apply before ISM (0-based).
        reference_bases (str | None): Optional existing variant ref allele.
        alternate_bases (str | None): Optional existing variant alt allele.
    """

    ism_interval_start: int = InputField(
        ge=0,
        description="ISM sub-interval start (0-based, inclusive)",
    )
    ism_interval_end: int = InputField(
        ge=1,
        description="ISM sub-interval end (0-based, exclusive)",
    )
    variant_position: int | None = InputField(
        default=None,
        ge=0,
        description="Optional existing variant position for ISM context (0-based)",
    )
    reference_bases: str | None = InputField(
        default=None,
        description="Optional existing variant reference allele",
    )
    alternate_bases: str | None = InputField(
        default=None,
        description="Optional existing variant alternate allele",
    )

    @field_validator("reference_bases", "alternate_bases")
    @classmethod
    def validate_allele_bases(cls, bases: str | None) -> str | None:
        """Validate allele sequence characters if provided."""
        if bases is None:
            return None
        normalized = bases.strip().upper()
        if not normalized:
            raise ValueError("Allele values cannot be empty")
        if not set(normalized) <= set("ACGTN"):
            raise ValueError("Allele values must only contain DNA bases A/C/G/T/N")
        return normalized

    @model_validator(mode="after")
    def validate_ism_interval(self) -> AlphaGenomeISM:
        """Validate ISM interval relationships."""
        if self.ism_interval_end <= self.ism_interval_start:
            raise ValueError("ism_interval_end must be greater than ism_interval_start")
        if self.ism_interval_start < self.interval_start or self.ism_interval_end > self.interval_end:
            raise ValueError("ISM interval must be fully contained in the interval")
        variant_fields = [
            self.variant_position,
            self.reference_bases,
            self.alternate_bases,
        ]
        if any(f is not None for f in variant_fields) and not all(f is not None for f in variant_fields):
            raise ValueError(
                "variant_position, reference_bases, and alternate_bases must all be provided together or all omitted"
            )
        if self.variant_position is not None and not (self.interval_start <= self.variant_position < self.interval_end):
            raise ValueError("variant_position must be within [interval_start, interval_end)")
        return self


class AlphaGenomeScoreISMInput(BaseToolInput):
    """Input for batched AlphaGenome in-silico mutagenesis.

    Attributes:
        requests (list[AlphaGenomeISM]): ISM requests to process.
            A single request is auto-wrapped into a list.
    """

    requests: list[AlphaGenomeISM] = InputField(
        description="ISM requests to process",
    )

    @field_validator("requests", mode="before")
    @classmethod
    def normalize_requests(cls, value: Any) -> list[Any]:
        """Validate and normalize ISM batch request specifications from raw input."""
        if value is None:
            raise ValueError("requests cannot be None")
        if not isinstance(value, list):
            value = [value]
        if not value:
            raise ValueError("requests cannot be empty")
        return value  # type: ignore[no-any-return]


class AlphaGenomeScoreISMOutput(BaseToolOutput):
    """Output from batched AlphaGenome in-silico mutagenesis.

    Attributes:
        results (list[AlphaGenomeScoreOutput]): Per-request score outputs.
    """

    results: list[AlphaGenomeScoreOutput] = Field(
        description="Per-request AlphaGenome ISM score outputs",
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


AlphaGenomeScoreISMConfig = AlphaGenomeScoreVariantsConfig


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return AlphaGenomeScoreISMInput(
        requests=[
            AlphaGenomeISM(
                chromosome="chr1",
                interval_start=0,
                interval_end=196608,
                ism_interval_start=100000,
                ism_interval_end=100010,
            )
        ]
    )


@tool(
    key="alphagenome-score-ism-variants-batch",
    label="AlphaGenome Score ISM Variants Batch",
    category="sequence_scoring",
    input_class=AlphaGenomeScoreISMInput,
    config_class=AlphaGenomeScoreISMConfig,
    output_class=AlphaGenomeScoreISMOutput,
    description="Run batched in-silico mutagenesis with AlphaGenome variant scorers",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="requests",
    iterable_output_field="results",
    cacheable=True,
)
def run_alphagenome_score_ism_variants_batch(
    inputs: AlphaGenomeScoreISMInput,
    config: AlphaGenomeScoreISMConfig | None = None,
    instance: Any = None,
) -> AlphaGenomeScoreISMOutput:
    """Run batched in-silico mutagenesis using AlphaGenome variant scorers."""
    require_hf_token("AlphaGenome", "https://huggingface.co/google/alphagenome-all-folds")

    serialized_requests = []
    for req in inputs.requests:
        request: dict[str, Any] = {
            "chromosome": req.chromosome,
            "interval_start": req.interval_start,
            "interval_end": req.interval_end,
            "ism_interval_start": req.ism_interval_start,
            "ism_interval_end": req.ism_interval_end,
        }
        if req.variant_position is not None:
            request["variant_position"] = req.variant_position
            request["reference_bases"] = req.reference_bases
            request["alternate_bases"] = req.alternate_bases
        serialized_requests.append(request)

    dispatch_result = ToolInstance.dispatch(
        "alphagenome",
        {
            "operation": "score_ism_variants_batch",
            "requests": serialized_requests,
            "variant_scorers": config.variant_scorers,  # type: ignore[union-attr]
            "organism": config.organism,  # type: ignore[union-attr]
            "model_version": config.model_version,  # type: ignore[union-attr]
            "device": config.device,  # type: ignore[union-attr]
        },
        instance=instance,
        config=config,
    )

    scores = dispatch_result["scores"]
    outputs = [AlphaGenomeScoreOutput(scores=score_list) for score_list in scores]
    return AlphaGenomeScoreISMOutput(results=outputs)
