"""gLM2 gradient for prepared mixed protein/DNA sequences."""

import logging
from typing import Any

from proto_tools.tools.masked_models.execution import dispatch_masked_model
from proto_tools.tools.masked_models.glm2.config import GLM2Config
from proto_tools.tools.masked_models.mixed_configs import MixedGradientConfig
from proto_tools.tools.masked_models.mixed_data_models import (
    MixedGradientOutput,
    MixedSequenceGradientInput,
    one_hot_mixed_logits,
)
from proto_tools.tools.tool_registry import tool

logger = logging.getLogger(__name__)

# Input:
GLM2GradientInput = MixedSequenceGradientInput

# Output:
GLM2GradientOutput = MixedGradientOutput


class GLM2GradientConfig(GLM2Config, MixedGradientConfig):
    """Configuration for gLM2 gradient.

    Attributes:
        model_checkpoint (Literal['tattabio/gLM2_150M', 'tattabio/gLM2_650M']): Public model checkpoint.
        batch_size (int): Individually masked variants per forward and backward pass.
        use_ste (bool): Hard forward tokens with soft-probability gradients.
        compute_gradient (bool): Compute the gradient, or evaluate only the masked PLL loss.
    """


def example_input() -> GLM2GradientInput:
    """Return a short prepared locus for examples and infrastructure checks."""
    return GLM2GradientInput(sequence="<+>MKTL<+>acgt", logits=one_hot_mixed_logits("<+>MKTL<+>acgt", sharpness=2.0))


@tool(
    key="glm2-gradient",
    label="gLM2 Gradient",
    category="masked_models",
    input_class=GLM2GradientInput,
    config_class=GLM2GradientConfig,
    output_class=GLM2GradientOutput,
    description="Differentiate masked pseudo-log-likelihood for relaxed mixed protein/DNA tokens with gLM2",
    uses_gpu=True,
    example_input=example_input,
    cacheable=False,
)
def run_glm2_gradient(
    inputs: GLM2GradientInput,
    config: GLM2GradientConfig,
    instance: Any = None,
) -> GLM2GradientOutput:
    """Run gLM2 gradient on prepared mixed-modality input.

    Args:
        inputs (GLM2GradientInput): Validated mixed-sequence input.
        config (GLM2GradientConfig): Checkpoint and operation settings.
        instance (Any): Optional persistent tool instance.

    Returns:
        GLM2GradientOutput: Token-aligned gradient output.
    """
    logger.debug("Using local worker for gLM2 gradient: %s", config.model_checkpoint)
    return GLM2GradientOutput(**dispatch_masked_model("glm2", "gradient", inputs, config, instance))
