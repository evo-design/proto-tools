"""AbLang relaxed-sequence gradient tool."""

import logging
from typing import Any

from proto_tools.entities.antibody import AntibodyLogits
from proto_tools.tools.masked_models.ablang.ablang_embeddings import _resolve_model_choice
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    PROTEIN_AMINO_ACIDS,
    BaseConfig,
    ConfigField,
    GradientOutput,
    ToolInstance,
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


AbLangGradientOutput = GradientOutput


class AbLangGradientConfig(BaseConfig):
    """Configuration for the AbLang shifted cross-entropy gradient tool.

    Attributes:
        use_ste (bool): When ``True``, uses a Straight-Through Estimator: hard one-hot
            tokens in the forward pass with gradients flowing through soft probabilities.
            When ``False`` (default), uses soft blended embeddings directly.
        device (str): Execution device for the model, for example 'cuda' or 'cpu'.
    """

    use_ste: bool = ConfigField(
        title="Straight-Through Estimator",
        default=False,
        description="Hard one-hot forward pass with soft-probability gradients.",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Execution device for the model, for example 'cuda' or 'cpu'.",
        hidden=True,
        include_in_key=False,
    )


def example_input() -> AbLangGradientInput:
    """Minimal valid input for testing and examples."""
    aa_index = {aa: i for i, aa in enumerate(PROTEIN_AMINO_ACIDS)}
    n_aas = len(PROTEIN_AMINO_ACIDS)
    logits = []
    for residue in "EVQLVESG":
        row = [0.0] * n_aas
        row[aa_index[residue]] = 2.0
        logits.append(row)
    return AbLangGradientInput(antibody=AntibodyLogits(heavy_chain=logits))


@tool(
    key="ablang-gradient",
    label="AbLang Gradient",
    category="masked_models",
    input_class=AbLangGradientInput,
    config_class=AbLangGradientConfig,
    output_class=AbLangGradientOutput,
    description="Compute AbLang shifted cross-entropy gradient for relaxed antibody sequences",
    uses_gpu=True,
    example_input=example_input,
    cacheable=False,
)
def run_ablang_gradient(
    inputs: AbLangGradientInput,
    config: AbLangGradientConfig,
    instance: Any = None,
) -> AbLangGradientOutput:
    """Compute AbLang gradient with respect to relaxed antibody logits."""
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
