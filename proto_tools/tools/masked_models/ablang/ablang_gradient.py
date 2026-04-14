"""AbLang relaxed-sequence gradient tool."""

import logging
from typing import Any, Literal

from pydantic import model_validator
from typing_extensions import Self

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    PROTEIN_AMINO_ACIDS,
    BaseConfig,
    ConfigField,
    GradientInput,
    GradientOutput,
    ToolInstance,
)

ABLANG_GRADIENT_MODEL_CHOICES = Literal["ablang1-heavy", "ablang1-light", "ablang2-paired"]

logger = logging.getLogger(__name__)

AbLangGradientInput = GradientInput
AbLangGradientOutput = GradientOutput


class AbLangGradientConfig(BaseConfig):
    """Configuration for the AbLang shifted cross-entropy gradient tool.

    Attributes:
        model_choice (ABLANG_GRADIENT_MODEL_CHOICES): AbLang model variant to use for
            computing the gradient.
        chain_break_position (int | None): Number of residues in the first chain, i.e.
            the position at which to insert the chain separator. Required for paired
            sequences with ablang2-paired.
        device (str): Execution device for the model, for example 'cuda' or 'cpu'.
    """

    model_choice: ABLANG_GRADIENT_MODEL_CHOICES = ConfigField(
        title="Model Variant",
        default="ablang1-heavy",
        description="AbLang model variant: 'ablang1-heavy', 'ablang1-light', or 'ablang2-paired'.",
        reload_on_change=True,
    )
    chain_break_position: int | None = ConfigField(
        title="Chain Break Position",
        default=None,
        ge=1,
        description="Number of residues in the first chain. Required for ablang2-paired.",
        depends_on={"model_choice": ["ablang2-paired"]},
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Execution device for the model, for example 'cuda' or 'cpu'.",
        hidden=True,
        include_in_key=False,
    )

    @model_validator(mode="after")
    def validate_paired_config(self) -> Self:
        """Require chain_break_position only for the paired model variant."""
        if self.model_choice == "ablang2-paired":
            if self.chain_break_position is None:
                raise ValueError("chain_break_position is required when model_choice='ablang2-paired'")
        elif self.chain_break_position is not None:
            raise ValueError("chain_break_position is only supported when model_choice='ablang2-paired'")
        return self


def example_input() -> AbLangGradientInput:
    """Minimal valid input for testing and examples."""
    aa_index = {aa: i for i, aa in enumerate(PROTEIN_AMINO_ACIDS)}
    n_aas = len(PROTEIN_AMINO_ACIDS)
    logits = []
    for residue in "EVQLVESG":
        row = [0.0] * n_aas
        row[aa_index[residue]] = 2.0
        logits.append(row)
    return AbLangGradientInput(logits=logits)


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
    logger.debug(
        "Using local worker for AbLang gradient (model_choice=%s)",
        config.model_choice,
    )
    payload = {
        "operation": "compute_gradient",
        "logits": inputs.logits,
        "temperature": inputs.temperature,
        "model_choice": config.model_choice,
        "chain_break_position": config.chain_break_position,
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
