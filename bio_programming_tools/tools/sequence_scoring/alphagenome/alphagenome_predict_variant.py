"""AlphaGenome variant-effect prediction tool."""
from __future__ import annotations

import logging

from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils.tool_instance import ToolInstance

from .shared_data_models import (
    AlphaGenomePredictConfig,
    AlphaGenomePredictOutput,
    AlphaGenomeVariantInput,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

# Input:
AlphaGenomePredictVariantInput = AlphaGenomeVariantInput

# Output:
AlphaGenomePredictVariantOutput = AlphaGenomePredictOutput

# Config:
AlphaGenomePredictVariantConfig = AlphaGenomePredictConfig


# ============================================================================
# Tool Implementation
# ============================================================================
@tool(
    key="alphagenome-predict-variant",
    label="AlphaGenome Predict Variant",
    category="sequence_scoring",
    input=AlphaGenomePredictVariantInput,
    config=AlphaGenomePredictVariantConfig,
    output=AlphaGenomePredictVariantOutput,
    description="Predict variant effects with AlphaGenome open weights",
    uses_gpu=True,
)
def run_alphagenome_predict_variant(
    inputs: AlphaGenomePredictVariantInput,
    config: AlphaGenomePredictVariantConfig,
    instance=None,
) -> AlphaGenomePredictVariantOutput:
    """Predict variant effects using AlphaGenome open weights."""
    result = ToolInstance.dispatch(
        "alphagenome",
        {
            "operation": "predict_variant",
            "chromosome": inputs.chromosome,
            "interval_start": inputs.interval_start,
            "interval_end": inputs.interval_end,
            "variant_position": inputs.variant_position,
            "reference_bases": inputs.reference_bases,
            "alternate_bases": inputs.alternate_bases,
            "requested_outputs": config.requested_outputs,
            "ontology_terms": config.ontology_terms,
            "organism": config.organism,
            "model_version": config.model_version,
            "device": config.device,
        },
        instance=instance,
        reload_on=type(config).reload_fields(),
    )

    return AlphaGenomePredictVariantOutput(
        chromosome=inputs.chromosome,
        interval_start=inputs.interval_start,
        interval_end=inputs.interval_end,
        requested_outputs=config.requested_outputs,
        result=result,
        variant={
            "position": inputs.variant_position,
            "reference_bases": inputs.reference_bases,
            "alternate_bases": inputs.alternate_bases,
        },
    )
