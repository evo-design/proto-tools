"""AlphaGenome sequence prediction tool."""
from __future__ import annotations

import logging

from pydantic import Field, field_validator

from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils.tool_instance import ToolInstance
from bio_programming_tools.utils.tool_io import BaseToolInput

from .shared_data_models import (
    AlphaGenomePredictConfig,
    AlphaGenomePredictOutput,
    SUPPORTED_CONTEXT_LENGTHS,
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

    Raw sequence inputs must already match one of these supported lengths.

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
        if len(sequence) not in SUPPORTED_CONTEXT_LENGTHS:
            supported = ", ".join(str(length) for length in sorted(SUPPORTED_CONTEXT_LENGTHS))
            raise ValueError(
                "sequence length must match a supported AlphaGenome context length "
                f"({supported} bp)"
            )
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
    category="sequence_scoring",
    input=AlphaGenomePredictSequenceInput,
    config=AlphaGenomePredictSequenceConfig,
    output=AlphaGenomePredictSequenceOutput,
    description="Predict genomic signals from a raw DNA sequence using AlphaGenome open weights",
    uses_gpu=True,
)
def run_alphagenome_predict_sequence(
    inputs: AlphaGenomePredictSequenceInput,
    config: AlphaGenomePredictSequenceConfig,
    instance=None,
) -> AlphaGenomePredictSequenceOutput:
    """Predict genomic features from a raw DNA sequence using AlphaGenome."""
    result = ToolInstance.dispatch(
        "alphagenome",
        {
            "operation": "predict_sequence",
            "sequence": inputs.sequence,
            "requested_outputs": config.requested_outputs,
            "ontology_terms": config.ontology_terms,
            "organism": config.organism,
            "model_version": config.model_version,
            "device": config.device,
            "timeout": config.timeout,
        },
        instance=instance,
        verbose=config.verbose,
        reload_on=type(config).reload_fields(),
    )

    seq_len = len(inputs.sequence)
    return AlphaGenomePredictSequenceOutput(
        chromosome="sequence",
        interval_start=0,
        interval_end=seq_len,
        requested_outputs=config.requested_outputs,
        result=result,
    )
