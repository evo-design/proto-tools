"""AlphaGenome variant scoring tool."""
from __future__ import annotations

import logging
from typing import List, Literal, Optional

from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig, ConfigField
from bio_programming_tools.utils.tool_instance import ToolInstance

from .shared_data_models import (
    DEFAULT_ALPHAGENOME_MODEL_VERSION,
    AlphaGenomeScoreOutput,
    AlphaGenomeVariantInput,
    VariantScorerName,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

# Input:
AlphaGenomeScoreVariantInput = AlphaGenomeVariantInput

# Output:
AlphaGenomeScoreVariantOutput = AlphaGenomeScoreOutput

# Config:
class AlphaGenomeScoreVariantConfig(BaseConfig):
    """Configuration for AlphaGenome variant scoring.

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
    )


# ============================================================================
# Tool Implementation
# ============================================================================
@tool(
    key="alphagenome-score-variant",
    label="AlphaGenome Score Variant",
    category="sequence_scoring",
    input=AlphaGenomeScoreVariantInput,
    config=AlphaGenomeScoreVariantConfig,
    output=AlphaGenomeScoreVariantOutput,
    description="Score variant effects with AlphaGenome using recommended variant scorers",
    uses_gpu=True,
)
def run_alphagenome_score_variant(
    inputs: AlphaGenomeScoreVariantInput,
    config: AlphaGenomeScoreVariantConfig,
    instance=None,
) -> AlphaGenomeScoreVariantOutput:
    """Score variant effects using AlphaGenome variant scorers."""
    result = ToolInstance.dispatch(
        "alphagenome",
        {
            "operation": "score_variant",
            "chromosome": inputs.chromosome,
            "interval_start": inputs.interval_start,
            "interval_end": inputs.interval_end,
            "variant_position": inputs.variant_position,
            "reference_bases": inputs.reference_bases,
            "alternate_bases": inputs.alternate_bases,
            "variant_scorers": config.variant_scorers,
            "organism": config.organism,
            "model_version": config.model_version,
            "device": config.device,
        },
        instance=instance,
        verbose=config.verbose,
        timeout=config.timeout,
        reload_on=type(config).reload_fields(),
    )

    return AlphaGenomeScoreVariantOutput(scores=result)
