"""AlphaGenome in-silico mutagenesis (ISM) tool."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator

from bio_programming.bio_tools.tools.infra.env_manager import EnvManager
from bio_programming.bio_tools.tools.tool_registry import tool

from .alphagenome_score_variant import AlphaGenomeScoreVariantConfig
from .shared_data_models import AlphaGenomeInput, AlphaGenomeScoreOutput

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

# Input:
class AlphaGenomeScoreISMInput(AlphaGenomeInput):
    """Input object for AlphaGenome in-silico mutagenesis.

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
    def validate_ism_interval(self) -> AlphaGenomeScoreISMInput:
        """Validate ISM interval relationships."""
        if self.ism_interval_end <= self.ism_interval_start:
            raise ValueError("ism_interval_end must be greater than ism_interval_start")
        if self.ism_interval_start < self.interval_start or self.ism_interval_end > self.interval_end:
            raise ValueError("ISM interval must be fully contained in the interval")
        # If variant fields are provided, all three must be present
        variant_fields = [
            self.variant_position,
            self.reference_bases,
            self.alternate_bases,
        ]
        if any(f is not None for f in variant_fields) and not all(f is not None for f in variant_fields):
            raise ValueError("variant_position, reference_bases, and alternate_bases must all be provided together or all omitted")
        if self.variant_position is not None:
            if not (self.interval_start <= self.variant_position < self.interval_end):
                raise ValueError("variant_position must be within [interval_start, interval_end)")
        return self


# Output:
AlphaGenomeScoreISMOutput = AlphaGenomeScoreOutput

# Config:
AlphaGenomeScoreISMConfig = AlphaGenomeScoreVariantConfig


# ============================================================================
# Tool Implementation
# ============================================================================
@tool(
    key="alphagenome-score-ism-variants",
    label="AlphaGenome Score ISM Variants",
    input=AlphaGenomeScoreISMInput,
    config=AlphaGenomeScoreISMConfig,
    output=AlphaGenomeScoreISMOutput,
    description="Run in-silico mutagenesis with AlphaGenome variant scorers",
)
def run_alphagenome_score_ism_variants(
    inputs: AlphaGenomeScoreISMInput,
    config: AlphaGenomeScoreISMConfig,
) -> AlphaGenomeScoreISMOutput:
    """Run in-silico mutagenesis using AlphaGenome variant scorers.

    Mutates every position in the ISM sub-interval and scores each mutation.
    """
    input_dict = {
        "operation": "score_ism_variants",
        "chromosome": inputs.chromosome,
        "interval_start": inputs.interval_start,
        "interval_end": inputs.interval_end,
        "ism_interval_start": inputs.ism_interval_start,
        "ism_interval_end": inputs.ism_interval_end,
        "variant_scorers": config.variant_scorers,
        "organism": config.organism,
        "model_version": config.model_version,
    }

    if inputs.variant_position is not None:
        input_dict["variant_position"] = inputs.variant_position
        input_dict["reference_bases"] = inputs.reference_bases
        input_dict["alternate_bases"] = inputs.alternate_bases

    venv_manager = EnvManager("alphagenome")
    script_path = Path(__file__).parent / "standalone" / "inference.py"
    result = venv_manager.call_standalone_script_in_venv(
        script_path=script_path,
        input_dict=input_dict,
        device=config.device,
    )

    return AlphaGenomeScoreISMOutput(scores=result)
