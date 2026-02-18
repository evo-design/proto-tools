"""AlphaGenome interval scoring tool."""
from __future__ import annotations

import logging
from typing import List, Literal, Optional

from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig, ConfigField
from bio_programming_tools.utils.tool_instance import ToolInstance

from .shared_data_models import (
    DEFAULT_ALPHAGENOME_MODEL_VERSION,
    AlphaGenomeInput,
    AlphaGenomeScoreOutput,
)

logger = logging.getLogger(__name__)

IntervalScorerName = Literal["RNA_SEQ"]


# ============================================================================
# Data Models
# ============================================================================

# Input:
AlphaGenomeScoreIntervalInput = AlphaGenomeInput

# Output:
AlphaGenomeScoreIntervalOutput = AlphaGenomeScoreOutput

# Config:
class AlphaGenomeScoreIntervalConfig(BaseConfig):
    """Configuration for AlphaGenome interval scoring.

    Attributes:
        model_version (str): AlphaGenome Hugging Face model version.
        interval_scorers (Optional[List[str]]): Scorer names from the library's
            ``RECOMMENDED_INTERVAL_SCORERS``. ``None`` uses all recommended.
        organism (Literal["human", "mouse"]): Organism for predictions.
        device (str): Device to run inference on.
    """

    model_version: str = ConfigField(
        title="Model Version",
        default=DEFAULT_ALPHAGENOME_MODEL_VERSION,
        description="AlphaGenome Hugging Face model version",
        advanced=True,
    )
    interval_scorers: Optional[List[IntervalScorerName]] = ConfigField(
        title="Interval Scorers",
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
    key="alphagenome-score-interval",
    label="AlphaGenome Score Interval",
    category="sequence_scoring",
    input=AlphaGenomeScoreIntervalInput,
    config=AlphaGenomeScoreIntervalConfig,
    output=AlphaGenomeScoreIntervalOutput,
    description="Score genomic intervals with AlphaGenome using recommended interval scorers",
    uses_gpu=True,
)
def run_alphagenome_score_interval(
    inputs: AlphaGenomeScoreIntervalInput,
    config: AlphaGenomeScoreIntervalConfig,
    instance=None,
) -> AlphaGenomeScoreIntervalOutput:
    """Score genomic intervals using AlphaGenome interval scorers."""
    result = ToolInstance.dispatch(
        "alphagenome",
        {
            "operation": "score_interval",
            "chromosome": inputs.chromosome,
            "interval_start": inputs.interval_start,
            "interval_end": inputs.interval_end,
            "interval_scorers": config.interval_scorers,
            "organism": config.organism,
            "model_version": config.model_version,
            "device": config.device,
        },
        instance=instance,
        reload_on=type(config).reload_fields(),
    )

    return AlphaGenomeScoreIntervalOutput(scores=result)
