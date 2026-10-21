"""AbLang masked pseudo-log-likelihood gradient tool."""

import logging
from collections.abc import Sequence
from typing import Any

from pydantic import Field

from proto_tools.entities.antibody import AntibodyLogits
from proto_tools.tools.masked_models.ablang.ablang_embeddings import _resolve_model_choice
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    ConfigField,
    GradientInput,
    GradientOutput,
    ProteinSequenceLogits,
    RelaxationConfig,
    ToolInstance,
    one_hot_protein_logits,
)
from proto_tools.utils.tool_io import InputField

logger = logging.getLogger(__name__)


class AbLangGradientInput(GradientInput):
    """Input for the AbLang gradient tool.

    Attributes:
        antibody (AntibodyLogits): Antibody with relaxed sequence distributions, one matrix
            per chain. The model variant is selected automatically from which chains are
            provided, and the returned gradient concatenates them heavy-then-light.
        temperature (float | None): Softmax temperature. The default ``1.0`` reads each chain
            as unnormalized and applies ``softmax(logits / T)``, returning a logit-space
            gradient; ``None`` reads them as probabilities and returns a probability-space
            gradient.
    """

    antibody: AntibodyLogits = InputField(
        title="Antibody Logits",
        description="Antibody with relaxed sequence distributions over amino acids.",
    )

    @classmethod
    def from_probabilities(  # type: ignore[override]  # chains are named, not a single matrix
        cls,
        *,
        heavy_chain: Sequence[Sequence[float]] | None = None,
        light_chain: Sequence[Sequence[float]] | None = None,
        temperature: float = 1.0,
        **fields: Any,
    ) -> "AbLangGradientInput":
        """Build an input whose chain logits reproduce the given distributions.

        At least one chain must be supplied, and both are converted at the same temperature the
        input stores, so the two cannot disagree.

        Args:
            heavy_chain (Sequence[Sequence[float]] | None): ``(Lh, 20)`` strictly positive rows.
            light_chain (Sequence[Sequence[float]] | None): ``(Ll, 20)`` strictly positive rows.
            temperature (float): Positive, finite temperature stored on the input.
            fields (Any): Any remaining input fields.

        Returns:
            AbLangGradientInput: Input satisfying ``softmax(chain / temperature) == chain probs``.
        """
        convert = ProteinSequenceLogits.from_probabilities
        return cls(
            antibody=AntibodyLogits(
                heavy_chain=None if heavy_chain is None else convert(heavy_chain, temperature=temperature),
                light_chain=None if light_chain is None else convert(light_chain, temperature=temperature),
            ),
            temperature=temperature,
            **fields,
        )


class AbLangGradientOutput(GradientOutput):
    """AbLang masked PLL output; gradient is optional in forward-only mode.

    Attributes:
        gradient (list[list[float]] | None): Gradient w.r.t. input logits, or ``None``
            when ``compute_gradient=False`` (forward-only scoring).
        loss (float): Mean negative log-likelihood over AA positions.
        metrics (dict[str, Any]): ``log_likelihood``, ``avg_log_likelihood``, ``perplexity``,
            ``sequence_length``, ``model_choice``, ``objective``.
        vocab (list[str]): Amino-acid column ordering for the input logits.
    """

    gradient: list[list[float]] | None = Field(
        default=None,
        title="Gradient",
        description="Gradient w.r.t. input logits. None when compute_gradient=False.",
    )


class AbLangGradientConfig(RelaxationConfig):
    """Configuration for the AbLang masked PLL gradient tool.

    Attributes:
        soft (float): Inherited blend from the raw input matrix (0) to softmax probabilities (1).
        hard (float): Inherited straight-through coefficient; 1 forwards a hard one-hot.
        compute_gradient (bool): Run backward pass and return gradient. Set ``False`` for
            forward-only log-likelihood scoring (e.g. MCMC proposal ranking).
        batch_size (int | None): AA positions per forward pass for batched PLL. ``None`` auto-
            selects a per-model default (lower if OOM, higher for throughput).
        device (str): Device to run the model on.
    """

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


def example_input() -> AbLangGradientInput:
    """Minimal valid input for testing and examples."""
    heavy = ProteinSequenceLogits(one_hot_protein_logits("EVQLVESG", sharpness=2.0))
    return AbLangGradientInput(antibody=AntibodyLogits(heavy_chain=heavy))


@tool(
    key="ablang-gradient",
    label="AbLang Gradient",
    category="masked_models",
    input_class=AbLangGradientInput,
    config_class=AbLangGradientConfig,
    output_class=AbLangGradientOutput,
    description="Compute AbLang masked pseudo-log-likelihood gradient for relaxed antibody sequences",
    uses_gpu=True,
    example_input=example_input,
    cacheable=False,
    stochastic=True,
)
def run_ablang_gradient(
    inputs: AbLangGradientInput,
    config: AbLangGradientConfig,
    instance: Any = None,
) -> AbLangGradientOutput:
    """Compute AbLang masked PLL gradient with respect to relaxed antibody logits."""
    ab = inputs.antibody
    model_choice = _resolve_model_choice([ab])

    chain_break_position: int | None = None
    if ab.heavy_chain is not None and ab.light_chain is not None:
        logits = ab.heavy_chain.root + ab.light_chain.root
        chain_break_position = len(ab.heavy_chain)
    elif ab.heavy_chain is not None:
        logits = ab.heavy_chain.root
    else:
        logits = ab.light_chain.root  # type: ignore[union-attr]

    logger.debug(
        "Using local worker for AbLang gradient (model_choice=%s)",
        model_choice,
    )
    payload = {
        "operation": "compute_gradient",
        "logits": logits,
        "temperature": inputs.temperature,
        "soft": config.soft,
        "hard": config.hard,
        "compute_gradient": config.compute_gradient,
        "batch_size": config.batch_size,
        "model_choice": model_choice,
        "chain_break_position": chain_break_position,
        "seed": config.seed,
        "device": config.device,
        "verbose": config.verbose,
    }
    result = ToolInstance.dispatch(
        "ablang",
        payload,
        instance=instance,
        config=config,
    )
    return AbLangGradientOutput.from_single(result)
