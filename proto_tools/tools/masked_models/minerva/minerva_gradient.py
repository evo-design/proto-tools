"""Minerva gradient for prepared mixed protein/DNA sequences."""

import logging
from typing import Any

from proto_tools.tools.masked_models.execution import dispatch_masked_model
from proto_tools.tools.masked_models.minerva.config import MinervaConfig
from proto_tools.tools.masked_models.mixed_configs import MixedGradientConfig
from proto_tools.tools.masked_models.mixed_data_models import (
    MixedGradientOutput,
    MixedSequenceGradientInput,
    one_hot_mixed_logits,
)
from proto_tools.tools.tool_registry import tool

logger = logging.getLogger(__name__)

# Input:
MinervaGradientInput = MixedSequenceGradientInput

# Output:
MinervaGradientOutput = MixedGradientOutput


class MinervaGradientConfig(MinervaConfig, MixedGradientConfig):
    """Configuration for Minerva gradient.

    Attributes:
        model_checkpoint (Literal['gbrixi/minerva-mlm', 'gbrixi/minerva-mlm-8k']): Public model checkpoint.
        batch_size (int): Individually masked variants per forward and backward pass.
        use_ste (bool): Hard forward tokens with soft-probability gradients.
        compute_gradient (bool): Compute the gradient, or evaluate only the masked PLL loss.
    """


def example_input() -> MinervaGradientInput:
    """Return a short prepared locus for examples and infrastructure checks."""
    return MinervaGradientInput(sequence="+MKTL+acgt", logits=one_hot_mixed_logits("+MKTL+acgt", sharpness=2.0))


@tool(
    key="minerva-gradient",
    label="Minerva Gradient",
    category="masked_models",
    input_class=MinervaGradientInput,
    config_class=MinervaGradientConfig,
    output_class=MinervaGradientOutput,
    description="Differentiate masked pseudo-log-likelihood for relaxed mixed protein/DNA tokens with Minerva",
    uses_gpu=True,
    example_input=example_input,
    cacheable=False,
)
def run_minerva_gradient(
    inputs: MinervaGradientInput,
    config: MinervaGradientConfig,
    instance: Any = None,
) -> MinervaGradientOutput:
    """Run Minerva gradient on prepared mixed-modality input.

    Args:
        inputs (MinervaGradientInput): Validated mixed-sequence input.
        config (MinervaGradientConfig): Checkpoint and operation settings.
        instance (Any): Optional persistent tool instance.

    Returns:
        MinervaGradientOutput: Token-aligned gradient output.
    """
    logger.debug("Using local worker for Minerva gradient: %s", config.model_checkpoint)
    return MinervaGradientOutput(**dispatch_masked_model("minerva", "gradient", inputs, config, instance))
