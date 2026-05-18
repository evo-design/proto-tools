"""AbLang masked pseudo-log-likelihood gradient tool."""

import logging
from typing import Any

from pydantic import Field

from proto_tools.entities.antibody import AntibodyLogits
from proto_tools.tools.masked_models.ablang.ablang_embeddings import _resolve_model_choice
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    ConfigField,
    GradientOutput,
    ToolInstance,
    one_hot_protein_logits,
)
from proto_tools.utils.tool_io import BaseToolInput, InputField

logger = logging.getLogger(__name__)


class AbLangGradientInput(BaseToolInput):
    """Input for the AbLang gradient tool.

    Attributes:
        antibody (AntibodyLogits): Antibody with relaxed sequence distributions.
            The model variant is selected automatically based on which chains
            are provided.
        temperature (float | None): Optional softmax temperature. When set, applies
            ``softmax(input / temperature)`` before computing the gradient. When
            ``None`` (default), the input is used as-is.
    """

    antibody: AntibodyLogits = InputField(
        description="Antibody with relaxed sequence distributions over amino acids.",
    )
    temperature: float | None = InputField(
        default=None,
        description="Softmax temperature. Applies softmax(input / T) when set.",
        gt=0.0,
    )


class AbLangGradientOutput(GradientOutput):
    """AbLang masked PLL output; gradient is optional in forward-only mode.

    Attributes:
        gradient (list[list[float]] | None): Gradient w.r.t. input logits, or ``None``
            when ``compute_gradient=False`` (forward-only scoring).
        loss (float): Mean negative log-likelihood over AA positions.
        metrics (dict[str, Any]): ``log_likelihood``, ``sequence_length``, ``model_choice``, ``objective``.
        vocab (list[str]): Amino-acid column ordering for the input logits.
    """

    gradient: list[list[float]] | None = Field(
        default=None,
        description="Gradient w.r.t. input logits. None when compute_gradient=False.",
    )


class AbLangGradientConfig(BaseConfig):
    """Configuration for the AbLang masked PLL gradient tool.

    Attributes:
        use_ste (bool): Straight-Through Estimator: hard one-hot in the forward pass with
            gradients flowing through soft probabilities. When ``False``, uses soft blended
            embeddings directly.
        compute_gradient (bool): Run backward pass and return gradient. Set ``False`` for
            forward-only log-likelihood scoring (e.g. MCMC proposal ranking).
        batch_size (int | None): AA positions per forward pass for batched PLL. ``None`` auto-
            selects a per-model default (lower if OOM, higher for throughput).
        device (str): Device to run the model on.
    """

    use_ste: bool = ConfigField(
        title="Straight-Through Estimator",
        default=False,
        description="Hard one-hot forward pass with soft-probability gradients",
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


def example_input() -> AbLangGradientInput:
    """Minimal valid input for testing and examples."""
    return AbLangGradientInput(antibody=AntibodyLogits(heavy_chain=one_hot_protein_logits("EVQLVESG", sharpness=2.0)))


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
        logits = ab.heavy_chain + ab.light_chain
        chain_break_position = len(ab.heavy_chain)
    elif ab.heavy_chain is not None:
        logits = ab.heavy_chain
    else:
        logits = ab.light_chain  # type: ignore[assignment]

    logger.debug(
        "Using local worker for AbLang gradient (model_choice=%s)",
        model_choice,
    )
    payload = {
        "operation": "compute_gradient",
        "logits": logits,
        "temperature": inputs.temperature,
        "use_ste": config.use_ste,
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
    return AbLangGradientOutput(**result)
