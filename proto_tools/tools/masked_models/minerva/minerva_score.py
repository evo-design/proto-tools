"""Minerva score for prepared mixed protein/DNA sequences."""

import logging
from typing import Any, Literal

from proto_tools.tools.masked_models.mixed_data_models import (
    MixedScoringConfig,
    MixedScoringMetrics,
    MixedScoringOutput,
    MixedSequenceInput,
    dispatch_mixed_model,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ConfigField

logger = logging.getLogger(__name__)

# Input:
MinervaScoringInput = MixedSequenceInput

# Output:
MinervaScoringOutput = MixedScoringOutput


class MinervaScoringConfig(MixedScoringConfig):
    """Configuration for Minerva score.

    Attributes:
        model_checkpoint (Literal['gbrixi/minerva-mlm', 'gbrixi/minerva-mlm-8k']): Public model checkpoint.
        batch_size (int): Individually masked variants per forward pass.
        return_logits (bool): Include masked biological logits; unscored context rows are zero.
    """

    model_checkpoint: Literal["gbrixi/minerva-mlm", "gbrixi/minerva-mlm-8k"] = ConfigField(
        default="gbrixi/minerva-mlm",
        title="Model Checkpoint",
        description="Minerva checkpoint; downloaded automatically from Hugging Face",
        reload_on_change=True,
    )


def example_input() -> MinervaScoringInput:
    """Return a short prepared locus for examples and infrastructure checks."""
    return MinervaScoringInput(sequences=["<+>MKTL<+>acgt<->ACDE"])


@tool(
    key="minerva-score",
    label="Minerva Scoring",
    category="masked_models",
    input_class=MinervaScoringInput,
    config_class=MinervaScoringConfig,
    output_class=MinervaScoringOutput,
    description="Score mixed protein/DNA loci by masked pseudo-log-likelihood with Minerva",
    uses_gpu=True,
    example_input=example_input,
    cacheable=True,
    iterable_input_fields=["sequences"],
    iterable_output_field="scores",
    max_chunk_size=32,
    metrics_class=MixedScoringMetrics,
)
def run_minerva_score(
    inputs: MinervaScoringInput,
    config: MinervaScoringConfig,
    instance: Any = None,
) -> MinervaScoringOutput:
    """Run Minerva score on prepared mixed-modality input.

    Args:
        inputs (MinervaScoringInput): Validated mixed-sequence input.
        config (MinervaScoringConfig): Checkpoint and operation settings.
        instance (Any): Optional persistent tool instance.

    Returns:
        MinervaScoringOutput: Token-aligned score output.
    """
    logger.debug("Using local worker for Minerva score: %s", config.model_checkpoint)
    return MinervaScoringOutput(**dispatch_mixed_model("minerva", "score", inputs, config, instance))
