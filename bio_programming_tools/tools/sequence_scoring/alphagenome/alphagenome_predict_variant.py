"""AlphaGenome variant-effect prediction tool."""
from __future__ import annotations

import logging
from pathlib import Path

from bio_programming_tools.tools.infra.env_manager import EnvManager
from bio_programming_tools.tools.tool_registry import tool

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
    input=AlphaGenomePredictVariantInput,
    config=AlphaGenomePredictVariantConfig,
    output=AlphaGenomePredictVariantOutput,
    description="Predict variant effects with AlphaGenome open weights",
)
def run_alphagenome_predict_variant(
    inputs: AlphaGenomePredictVariantInput,
    config: AlphaGenomePredictVariantConfig,
) -> AlphaGenomePredictVariantOutput:
    """Predict variant effects using AlphaGenome open weights."""
    venv_manager = EnvManager("alphagenome")
    script_path = Path(__file__).parent / "standalone" / "inference.py"
    result = venv_manager.call_standalone_script_in_venv(
        script_path=script_path,
        input_dict={
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
        },
        device=config.device,
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
