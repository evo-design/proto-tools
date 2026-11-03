"""ESM C masked pseudo-log-likelihood gradient tool."""

import logging
from typing import Any, ClassVar, Literal

from pydantic import Field, field_validator

from proto_tools.tools.masked_models.esmc.esmc_embeddings import ESMC_MODEL_CHECKPOINTS
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    ConfigField,
    DifferentiableTerm,
    GradientInput,
    GradientOutput,
    InputField,
    ProteinSequenceLogits,
    RelaxationConfig,
    ToolInstance,
    attach_intrinsic_term,
    coerce_to_item_list,
    one_hot_protein_logits,
)
from proto_tools.utils.tool_io import Directionality

logger = logging.getLogger(__name__)


class ESMCGradientInput(GradientInput):
    """Input for the ESM C gradient tool: one or many relaxed protein states.

    Attributes:
        inputs (list[ProteinSequenceLogits]): One relaxed state per candidate, each ``(L, 20)``
            in canonical amino-acid order ``ACDEFGHIKLMNPQRSTVWY``.
        temperature (float | None): Softmax temperature, shared by every item. The default
            ``1.0`` reads each state as unnormalized and applies ``softmax(logits / T)``,
            returning logit-space gradients; ``None`` reads them as probabilities and returns
            probability-space gradients.
    """

    inputs: list[ProteinSequenceLogits] = InputField(
        title="Relaxed Sequences",
        description="One relaxed (L, 20) protein state per candidate",
    )

    @field_validator("inputs", mode="before")
    @classmethod
    def normalize_items(cls, value: Any) -> Any:
        """Accept one relaxed state as well as a list of them."""
        return coerce_to_item_list(value)


class ESMCMaskedPLLTerm(DifferentiableTerm):
    """ESM C's one differentiable quantity: the masked pseudo-log-likelihood.

    The tool exposes a single scalar, so the term carries no selectors beyond the identifier
    naming what it is.

    Attributes:
        id (Literal["masked_pll"]): Fixed identifier for this kind of quantity.
    """

    #: A mean negative log-likelihood is better the smaller it is.
    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "lower"

    #: Nats, averaged over the amino-acid positions the mean is taken across.
    UNITS: ClassVar[str] = "nats per position"

    id: Literal["masked_pll"] = Field(
        default="masked_pll",
        title="Term ID",
        description="Identifies this quantity as ESM C's masked pseudo-log-likelihood.",
    )


class ESMCGradientOutput(GradientOutput):
    """ESM C masked PLL output; gradient is optional in forward-only mode.

    Attributes:
        results (list[GradientResult]): One per input item, in input order, each holding that
            item's gradient w.r.t. its logits (``None`` in forward-only mode), its mean negative
            log-likelihood over AA positions, and its log-likelihood, perplexity, sequence length
            and objective details.
        vocab (list[str]): Amino-acid column ordering for the input logits.
    """


class ESMCGradientConfig(RelaxationConfig):
    """Configuration for the ESM C masked PLL gradient tool.

    Attributes:
        model_checkpoint (ESMC_MODEL_CHECKPOINTS): ESM C weights variant.
        soft (float): Inherited blend from the raw input matrix (0) to softmax probabilities
            (1). Recommended to leave at 1.0 in almost all use cases; exposed for research
            and ColabDesign parity. See
            :class:`~proto_tools.utils.gradient.models.RelaxationConfig`.
        hard (float): Inherited straight-through coefficient; 1 forwards a hard one-hot.
        compute_gradient (bool): Run backward pass and return gradient. Set ``False`` for
            forward-only log-likelihood scoring.
        batch_size (int | None): AA positions per forward pass for batched PLL. ``None``
            selects the backend default.
        device (str): Device to run the model on.
    """

    model_checkpoint: ESMC_MODEL_CHECKPOINTS = ConfigField(
        title="ESM C Model Checkpoint",
        default="esmc_300m",
        description="ESM C weights variant; trade off speed vs scoring fidelity",
        reload_on_change=True,
    )
    compute_gradient: bool = ConfigField(
        title="Compute Gradient",
        default=True,
        description="Run backward pass and return gradient; set False for forward-only log-likelihood",
    )
    batch_size: int | None = ConfigField(
        title="PLL Batch Size",
        default=None,
        gt=0,
        description="AA positions per forward pass. Lower if OOM, higher for throughput",
        include_in_key=False,
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        include_in_key=False,
    )


def example_input() -> ESMCGradientInput:
    """Minimal valid input for testing and examples."""
    return ESMCGradientInput(inputs=[ProteinSequenceLogits(one_hot_protein_logits("MKTL", sharpness=2.0))])


@tool(
    key="esmc-gradient",
    label="ESM C Gradient",
    category="masked_models",
    input_class=ESMCGradientInput,
    config_class=ESMCGradientConfig,
    output_class=ESMCGradientOutput,
    description="Compute ESM C masked pseudo-log-likelihood gradient for relaxed protein sequences",
    uses_gpu=True,
    example_input=example_input,
    cacheable=False,
    iterable_input_fields=["inputs"],
    iterable_output_field="results",
    max_chunk_size=8,
)
def run_esmc_gradient(
    inputs: ESMCGradientInput,
    config: ESMCGradientConfig,
    instance: Any = None,
) -> ESMCGradientOutput:
    """Compute ESM C masked PLL gradient with respect to relaxed protein logits."""
    logger.debug("Using local worker for ESM C gradient: %s (%d item(s))", config.model_checkpoint, len(inputs.inputs))
    result = ToolInstance.dispatch(
        "esmc",
        {
            "operation": "compute_gradient",
            "items": [matrix.root for matrix in inputs.inputs],
            "temperature": inputs.temperature,
            "soft": config.soft,
            "hard": config.hard,
            "compute_gradient": config.compute_gradient,
            "batch_size": config.batch_size,
            "model_checkpoint": config.model_checkpoint,
            "seed": config.seed,
            "device": config.device,
            "verbose": config.verbose,
        },
        instance=instance,
        config=config,
    )
    attach_intrinsic_term(
        result,
        ESMCMaskedPLLTerm(),
        matrices=[matrix.root for matrix in inputs.inputs],
        temperature=inputs.temperature,
        soft=config.soft,
        hard=config.hard,
        description="Mean masked negative log-likelihood over amino-acid positions",
    )
    return ESMCGradientOutput(**result)
