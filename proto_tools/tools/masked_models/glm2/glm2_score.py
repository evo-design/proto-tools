"""gLM2 score for prepared mixed protein/DNA sequences."""

import logging
from typing import Any

from proto_tools.tools.masked_models.execution import dispatch_masked_model
from proto_tools.tools.masked_models.glm2.config import GLM2Config
from proto_tools.tools.masked_models.mixed_configs import MixedScoringConfig
from proto_tools.tools.masked_models.mixed_data_models import (
    MixedScoringMetrics,
    MixedScoringOutput,
    MixedSequenceInput,
)
from proto_tools.tools.tool_registry import tool

logger = logging.getLogger(__name__)

# Input:
GLM2ScoringInput = MixedSequenceInput

# Output:
GLM2ScoringOutput = MixedScoringOutput


class GLM2ScoringConfig(GLM2Config, MixedScoringConfig):
    """Configuration for gLM2 score.

    Attributes:
        model_checkpoint (Literal['tattabio/gLM2_150M', 'tattabio/gLM2_650M']): Public model checkpoint.
        batch_size (int): Individually masked variants per forward pass.
        return_logits (bool): Include masked biological logits; unscored context rows are zero.
    """


def example_input() -> GLM2ScoringInput:
    """Return a short prepared locus for examples and infrastructure checks."""
    return GLM2ScoringInput(sequences=["<+>MKTL<+>acgt<->ACDE"])


@tool(
    key="glm2-score",
    label="gLM2 Scoring",
    category="masked_models",
    input_class=GLM2ScoringInput,
    config_class=GLM2ScoringConfig,
    output_class=GLM2ScoringOutput,
    description="Score mixed protein/DNA loci by masked pseudo-log-likelihood with gLM2",
    uses_gpu=True,
    example_input=example_input,
    cacheable=True,
    iterable_input_fields=["sequences"],
    iterable_output_field="scores",
    max_chunk_size=32,
    metrics_class=MixedScoringMetrics,
)
def run_glm2_score(
    inputs: GLM2ScoringInput,
    config: GLM2ScoringConfig,
    instance: Any = None,
) -> GLM2ScoringOutput:
    """Run gLM2 score on prepared mixed-modality input.

    Args:
        inputs (GLM2ScoringInput): Validated mixed-sequence input.
        config (GLM2ScoringConfig): Checkpoint and operation settings.
        instance (Any): Optional persistent tool instance.

    Returns:
        GLM2ScoringOutput: Token-aligned score output.
    """
    logger.debug("Using local worker for gLM2 score: %s", config.model_checkpoint)
    return GLM2ScoringOutput(**dispatch_masked_model("glm2", "score", inputs, config, instance))
