"""AlphaGenome sequence prediction tool."""
from __future__ import annotations

import logging
from pathlib import Path

from pydantic import Field, field_validator

from bio_programming.bio_tools.tools.infra.env_manager import EnvManager
from bio_programming.bio_tools.tools.infra.tool_io import BaseToolInput
from bio_programming.bio_tools.tools.tool_registry import tool

from .shared_data_models import (
    AlphaGenomePredictConfig,
    AlphaGenomePredictOutput,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

# Input:
class AlphaGenomePredictSequenceInput(BaseToolInput):
    """Input object for AlphaGenome raw-sequence prediction.

    AlphaGenome's architecture requires input sequences whose length matches
    one of the supported context lengths:

    * **1,048,576 bp** (1 MB, recommended)
    * **524,288 bp** (500 KB)
    * **131,072 bp** (100 KB)
    * **16,384 bp** (16 KB)
    * **2,048 bp** (2 KB)

    If the supplied sequence does not match a supported length it is
    automatically resized at inference time (see ``standalone/inference.py``).

    Attributes:
        sequence (str): Raw DNA sequence string (A/C/G/T/N characters).
    """

    sequence: str = Field(description="Raw DNA sequence for prediction")

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, sequence: str) -> str:
        """Validate sequence contains only valid nucleotide characters."""
        sequence = sequence.strip().upper()
        if not sequence:
            raise ValueError("sequence cannot be empty")
        if not set(sequence) <= set("ACGTN"):
            raise ValueError("sequence must only contain DNA bases A/C/G/T/N")
        return sequence


# Output:
AlphaGenomePredictSequenceOutput = AlphaGenomePredictOutput

# Config:
AlphaGenomePredictSequenceConfig = AlphaGenomePredictConfig


# ============================================================================
# Tool Implementation
# ============================================================================
@tool(
    key="alphagenome-predict-sequence",
    label="AlphaGenome Predict Sequence",
    input=AlphaGenomePredictSequenceInput,
    config=AlphaGenomePredictSequenceConfig,
    output=AlphaGenomePredictSequenceOutput,
    description="Predict genomic signals from a raw DNA sequence using AlphaGenome open weights",
)
def run_alphagenome_predict_sequence(
    inputs: AlphaGenomePredictSequenceInput,
    config: AlphaGenomePredictSequenceConfig,
) -> AlphaGenomePredictSequenceOutput:
    """Predict genomic features from a raw DNA sequence using AlphaGenome."""
    venv_manager = EnvManager("alphagenome")
    script_path = Path(__file__).parent / "standalone" / "inference.py"
    result = venv_manager.call_standalone_script_in_venv(
        script_path=script_path,
        input_dict={
            "operation": "predict_sequence",
            "sequence": inputs.sequence,
            "requested_outputs": config.requested_outputs,
            "ontology_terms": config.ontology_terms,
            "organism": config.organism,
            "model_version": config.model_version,
        },
        device=config.device,
    )

    seq_len = len(inputs.sequence)
    return AlphaGenomePredictSequenceOutput(
        chromosome="sequence",
        interval_start=0,
        interval_end=seq_len,
        requested_outputs=config.requested_outputs,
        result=result,
    )
