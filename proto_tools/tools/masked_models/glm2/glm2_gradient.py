"""gLM2 gradient for prepared mixed protein/DNA sequences."""

import logging
from typing import Any, ClassVar, Literal

from pydantic import Field

from proto_tools.tools.masked_models.execution import dispatch_masked_model
from proto_tools.tools.masked_models.glm2.config import GLM2Config
from proto_tools.tools.masked_models.mixed_configs import MixedGradientConfig
from proto_tools.tools.masked_models.mixed_data_models import (
    MixedGradientOutput,
    MixedSequenceGradientCandidate,
    MixedSequenceGradientInput,
    one_hot_mixed_logits,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import DifferentiableTerm, attach_intrinsic_term
from proto_tools.utils.tool_io import Directionality

logger = logging.getLogger(__name__)

# Input:
GLM2GradientInput = MixedSequenceGradientInput

# Output:
GLM2GradientOutput = MixedGradientOutput


class GLM2MaskedPLLTerm(DifferentiableTerm):
    """gLM2's one differentiable quantity: the masked pseudo-log-likelihood.

    The tool exposes a single scalar, so the term carries no selectors beyond the identifier
    naming what it is.

    Attributes:
        id (Literal["masked_pll"]): Fixed identifier for this kind of quantity.
    """

    #: A mean negative log-likelihood is better the smaller it is.
    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "lower"

    #: Nats, averaged over the positions the mean is taken across.
    UNITS: ClassVar[str] = "nats per position"

    #: What this quantity measures, copied into each gradient's metadata.
    DESCRIPTION: ClassVar[str] = "Mean masked negative log-likelihood over canonical protein and DNA token positions"

    id: Literal["masked_pll"] = Field(
        default="masked_pll",
        title="Term ID",
        description="Identifies this quantity as gLM2's masked pseudo-log-likelihood.",
    )


class GLM2GradientConfig(GLM2Config, MixedGradientConfig):
    """Configuration for gLM2 gradient.

    Attributes:
        model_checkpoint (Literal['tattabio/gLM2_150M', 'tattabio/gLM2_650M']): Public model checkpoint.
        batch_size (int): Individually masked variants per forward and backward pass.
        soft (float): Blend from the raw input matrix (0) to softmax probabilities (1).
            Recommended to leave at 1.0 in almost all use cases; exposed for research and
            ColabDesign parity. See
            :class:`~proto_tools.utils.gradient.models.RelaxationConfig`.
            Recommended to leave at 1.0 in almost all use cases; exposed for research and
            ColabDesign parity. See
            :class:`~proto_tools.utils.gradient.models.RelaxationConfig`.
        hard (float): Straight-through coefficient; 1 forwards a hard one-hot.
        compute_gradient (bool): Compute the gradient, or evaluate only the masked PLL loss.
    """


def example_input() -> GLM2GradientInput:
    """Return a short prepared locus for examples and infrastructure checks."""
    return GLM2GradientInput(
        inputs=[
            MixedSequenceGradientCandidate(
                sequence="+MKTL+acgt", logits=one_hot_mixed_logits("+MKTL+acgt", sharpness=2.0)
            )
        ]
    )


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
    iterable_input_fields=["inputs"],
    iterable_output_field="results",
    max_chunk_size=4,
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
    result = dispatch_masked_model("glm2", "gradient", inputs, config, instance)
    attach_intrinsic_term(
        result,
        GLM2MaskedPLLTerm(),
        matrices=[candidate.logits.root for candidate in inputs.inputs],
        temperature=inputs.temperature,
        soft=config.soft,
        hard=config.hard,
    )
    return GLM2GradientOutput(**result)
