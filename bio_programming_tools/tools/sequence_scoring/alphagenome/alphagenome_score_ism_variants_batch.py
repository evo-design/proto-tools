"""AlphaGenome batched in-silico mutagenesis (ISM) tool."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Iterator, List, Optional, Union

from pydantic import Field, field_validator, model_validator

from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils.tool_cache import tool_cache_iterable
from bio_programming_tools.utils.tool_instance import ToolInstance
from bio_programming_tools.utils.tool_io import BaseToolInput, BaseToolOutput

from .alphagenome_score_variants import AlphaGenomeScoreVariantsConfig
from .shared_data_models import AlphaGenomeInterval, AlphaGenomeScoreOutput

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
        variant_position (Optional[int]): Optional existing variant position
            to apply before ISM (0-based).
        reference_bases (Optional[str]): Optional existing variant ref allele.
        alternate_bases (Optional[str]): Optional existing variant alt allele.
    """

    ism_interval_start: int = Field(
        ge=0,
        description="ISM sub-interval start (0-based, inclusive)",
    )
    ism_interval_end: int = Field(
        ge=1,
        description="ISM sub-interval end (0-based, exclusive)",
    )
    variant_position: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional existing variant position for ISM context (0-based)",
    )
    reference_bases: Optional[str] = Field(
        default=None,
        description="Optional existing variant reference allele",
    )
    alternate_bases: Optional[str] = Field(
        default=None,
        description="Optional existing variant alternate allele",
    )

    @field_validator("reference_bases", "alternate_bases")
    @classmethod
    def validate_allele_bases(cls, bases: Optional[str]) -> Optional[str]:
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
                "variant_position, reference_bases, and alternate_bases must all be "
                "provided together or all omitted"
            )
        if self.variant_position is not None:
            if not (self.interval_start <= self.variant_position < self.interval_end):
                raise ValueError("variant_position must be within [interval_start, interval_end)")
        return self



class AlphaGenomeScoreISMInput(BaseToolInput):
    """Input for batched AlphaGenome in-silico mutagenesis.

    Attributes:
        requests (List[AlphaGenomeISM]): ISM requests to process.
            A single request is auto-wrapped into a list.
    """

    requests: List[AlphaGenomeISM] = Field(
        description="ISM requests to process",
    )

    @field_validator("requests", mode="before")
    @classmethod
    def normalize_requests(cls, value: Any) -> list:
        if value is None:
            raise ValueError("requests cannot be None")
        if not isinstance(value, list):
            value = [value]
        if not value:
            raise ValueError("requests cannot be empty")
        return value


class AlphaGenomeScoreISMOutput(BaseToolOutput):
    """Output from batched AlphaGenome in-silico mutagenesis.

    Attributes:
        results (List[AlphaGenomeScoreOutput]): Per-request score outputs.
    """

    results: List[AlphaGenomeScoreOutput] = Field(
        description="Per-request AlphaGenome ISM score outputs",
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


AlphaGenomeScoreISMConfig = AlphaGenomeScoreVariantsConfig


# ============================================================================
# Tool Implementation
# ============================================================================
@tool(
    key="alphagenome-score-ism-variants-batch",
    label="AlphaGenome Score ISM Variants Batch",
    category="sequence_scoring",
    input=AlphaGenomeScoreISMInput,
    config=AlphaGenomeScoreISMConfig,
    output=AlphaGenomeScoreISMOutput,
    description="Run batched in-silico mutagenesis with AlphaGenome variant scorers",
    uses_gpu=True,
)
@tool_cache_iterable(
    input_iterable_field="requests",
    output_iterable_field="results",
    tool_name="alphagenome-score-ism-variants-batch",
)
def run_alphagenome_score_ism_variants_batch(
    inputs: AlphaGenomeScoreISMInput,
    config: AlphaGenomeScoreISMConfig,
    instance=None,
) -> AlphaGenomeScoreISMOutput:
    """Run batched in-silico mutagenesis using AlphaGenome variant scorers."""
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
            "variant_scorers": config.variant_scorers,
            "organism": config.organism,
            "model_version": config.model_version,
            "device": config.device,
            "timeout": config.timeout,
        },
        instance=instance,
        verbose=config.verbose,
        timeout=config.timeout,
        reload_on=type(config).reload_fields(),
    )

    scores = dispatch_result["scores"]
    outputs = [
        AlphaGenomeScoreOutput(scores=score_list)
        for score_list in scores
    ]
    return AlphaGenomeScoreISMOutput(results=outputs)
