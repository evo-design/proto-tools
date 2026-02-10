"""AlphaGenome interval prediction tool."""
from __future__ import annotations

import logging
from pathlib import Path

from bio_programming_tools.tools.infra.env_manager import EnvManager
from bio_programming_tools.tools.tool_registry import tool

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
    input=AlphaGenomePredictIntervalInput,
    config=AlphaGenomePredictIntervalConfig,
    output=AlphaGenomePredictIntervalOutput,
    description="Predict genomic signals for a region using AlphaGenome open weights",
)
def run_alphagenome_predict_interval(
    inputs: AlphaGenomePredictIntervalInput,
    config: AlphaGenomePredictIntervalConfig,
) -> AlphaGenomePredictIntervalOutput:
    """Predict genomic features for an interval using AlphaGenome open weights."""
    venv_manager = EnvManager("alphagenome")
    script_path = Path(__file__).parent / "standalone" / "inference.py"
    result = venv_manager.call_standalone_script_in_venv(
        script_path=script_path,
        input_dict={
            "operation": "predict_interval",
            "chromosome": inputs.chromosome,
            "interval_start": inputs.interval_start,
            "interval_end": inputs.interval_end,
            "requested_outputs": config.requested_outputs,
            "ontology_terms": config.ontology_terms,
            "organism": config.organism,
            "model_version": config.model_version,
        },
        device=config.device,
    )

    return AlphaGenomePredictIntervalOutput(
        chromosome=inputs.chromosome,
        interval_start=inputs.interval_start,
        interval_end=inputs.interval_end,
        requested_outputs=config.requested_outputs,
        result=result,
    )
