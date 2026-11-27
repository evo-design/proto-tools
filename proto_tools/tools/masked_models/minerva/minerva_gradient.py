"""Minerva gradient for prepared mixed protein/DNA sequences."""

import logging
from typing import Any, ClassVar, Literal

from pydantic import Field

from proto_tools.tools.masked_models.execution import dispatch_masked_model
from proto_tools.tools.masked_models.minerva.config import MinervaConfig
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
MinervaGradientInput = MixedSequenceGradientInput

# Output:
MinervaGradientOutput = MixedGradientOutput


class MinervaMaskedPLLTerm(DifferentiableTerm):
    """Minerva's one differentiable quantity: the masked pseudo-log-likelihood.

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
        description="Identifies this quantity as Minerva's masked pseudo-log-likelihood.",
    )


class MinervaGradientConfig(MinervaConfig, MixedGradientConfig):
    """Configuration for Minerva gradient.

    Attributes:
        model_checkpoint (Literal['gbrixi/minerva-mlm', 'gbrixi/minerva-mlm-8k']): Public model checkpoint.
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


def example_input() -> MinervaGradientInput:
    """Return a short prepared locus for examples and infrastructure checks."""
    return MinervaGradientInput(
        inputs=[
            MixedSequenceGradientCandidate(
                sequence="+MKTL+acgt", logits=one_hot_mixed_logits("+MKTL+acgt", sharpness=2.0)
            )
        ]
    )


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
    iterable_input_fields=["inputs"],
    iterable_output_field="results",
    max_chunk_size=4,
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
    result = dispatch_masked_model("minerva", "gradient", inputs, config, instance)
    attach_intrinsic_term(
        result,
        MinervaMaskedPLLTerm(),
        matrices=[candidate.logits.root for candidate in inputs.inputs],
        temperature=inputs.temperature,
        soft=config.soft,
        hard=config.hard,
    )
    return MinervaGradientOutput(**result)
