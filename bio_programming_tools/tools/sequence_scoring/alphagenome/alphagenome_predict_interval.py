"""AlphaGenome interval prediction tool."""
from __future__ import annotations

import logging

from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils.tool_instance import ToolInstance

from .shared_data_models import (
    AlphaGenomeInput,
    AlphaGenomePredictConfig,
    AlphaGenomePredictOutput,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

# Input:
AlphaGenomePredictIntervalInput = AlphaGenomeInput

# Output:
AlphaGenomePredictIntervalOutput = AlphaGenomePredictOutput

# Config:
AlphaGenomePredictIntervalConfig = AlphaGenomePredictConfig


# ============================================================================
# Tool Implementation
# ============================================================================
@tool(
    key="alphagenome-predict-interval",
    label="AlphaGenome Predict Interval",
    category="sequence_scoring",
    input=AlphaGenomePredictIntervalInput,
    config=AlphaGenomePredictIntervalConfig,
    output=AlphaGenomePredictIntervalOutput,
    description="Predict genomic signals for a region using AlphaGenome open weights",
    uses_gpu=True,
)
def run_alphagenome_predict_interval(
    inputs: AlphaGenomePredictIntervalInput,
    config: AlphaGenomePredictIntervalConfig,
    instance=None,
) -> AlphaGenomePredictIntervalOutput:
    """Predict genomic features for an interval using AlphaGenome open weights."""
    result = ToolInstance.dispatch(
        "alphagenome",
        {
            "operation": "predict_interval",
            "chromosome": inputs.chromosome,
            "interval_start": inputs.interval_start,
            "interval_end": inputs.interval_end,
            "requested_outputs": config.requested_outputs,
            "ontology_terms": config.ontology_terms,
            "organism": config.organism,
            "model_version": config.model_version,
            "device": config.device,
        },
        instance=instance,
        reload_on=type(config).reload_fields(),
    )

    return AlphaGenomePredictIntervalOutput(
        chromosome=inputs.chromosome,
        interval_start=inputs.interval_start,
        interval_end=inputs.interval_end,
        requested_outputs=config.requested_outputs,
        result=result,
    )
