"""Germinal-specific AbLang relaxed-sequence gradient tool."""

from __future__ import annotations

import logging
from typing import Any

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

logger = logging.getLogger(__name__)

AbLangGerminalGradientInput = GradientInput
AbLangGerminalGradientOutput = GradientOutput


class AbLangGerminalGradientConfig(BaseConfig):
    """Configuration for Germinal's existing AbLang objective.

    Attributes:
        use_single_chain_variable_fragment (bool): Use the paired scFv AbLang
            path instead of the single-domain VHH path.
        heavy_chain_first (bool): Whether the scFv layout is VH-linker-VL.
        heavy_chain_length (int | None): Number of heavy-chain residues in the
            relaxed sequence.
        light_chain_length (int | None): Number of light-chain residues in the
            relaxed sequence.
    """

    use_single_chain_variable_fragment: bool = ConfigField(
        title="scFv Mode",
        default=False,
        description="Use the paired scFv AbLang path instead of the single-domain VHH path.",
    )
    heavy_chain_first: bool = ConfigField(
        title="Heavy Chain First",
        default=True,
        description="Whether the scFv layout is VH-linker-VL. Ignored when scFv mode is off.",
        advanced=True,
    )
    heavy_chain_length: int | None = ConfigField(
        title="Heavy Chain Length",
        default=None,
        ge=1,
        description="Number of heavy-chain residues. Required when scFv mode is enabled.",
        advanced=True,
    )
    light_chain_length: int | None = ConfigField(
        title="Light Chain Length",
        default=None,
        ge=1,
        description="Number of light-chain residues. Required when scFv mode is enabled.",
        advanced=True,
    )
    seed: int | None = ConfigField(
        title="Seed",
        default=0,
        description="PyTorch random seed for Germinal adapter initialization.",
        advanced=True,
        include_in_key=False,
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Execution device for the model, for example 'cuda' or 'cpu'.",
        hidden=True,
        include_in_key=False,
    )

    @model_validator(mode="after")
    def validate_chain_layout(self) -> Self:
        """Require chain lengths only for the paired single-chain variable fragment path."""
        chain_lengths = (self.heavy_chain_length, self.light_chain_length)
        if self.use_single_chain_variable_fragment:
            if any(length is None for length in chain_lengths):
                raise ValueError(
                    "heavy_chain_length and light_chain_length are required when "
                    "use_single_chain_variable_fragment=True"
                )
        elif any(length is not None for length in chain_lengths):
            raise ValueError(
                "heavy_chain_length and light_chain_length are only supported when "
                "use_single_chain_variable_fragment=True"
            )
        return self


def example_input() -> AbLangGerminalGradientInput:
    """Minimal valid input for testing and examples."""
    return AbLangGerminalGradientInput(logits=[[0.0] * len(PROTEIN_AMINO_ACIDS)] * 4)


@tool(
    key="ablang-germinal-gradient",
    label="AbLang Germinal Gradient",
    category="masked_models",
    input_class=AbLangGerminalGradientInput,
    config_class=AbLangGerminalGradientConfig,
    output_class=AbLangGerminalGradientOutput,
    description="Compute Germinal's existing AbLang relaxed-sequence gradient objective",
    uses_gpu=True,
    example_input=example_input,
    cacheable=False,
)
def run_ablang_germinal_gradient(
    inputs: AbLangGerminalGradientInput,
    config: AbLangGerminalGradientConfig,
    instance: Any = None,
) -> AbLangGerminalGradientOutput:
    """Compute Germinal's AbLang gradient with respect to relaxed antibody logits."""
    logger.debug(
        "Using local worker for Germinal AbLang gradient (single_chain_variable_fragment=%s)",
        config.use_single_chain_variable_fragment,
    )
    payload = {
        "operation": "compute_germinal_gradient",
        **inputs.model_dump(),
        "use_single_chain_variable_fragment": config.use_single_chain_variable_fragment,
        "heavy_chain_first": config.heavy_chain_first,
        "heavy_chain_length": config.heavy_chain_length,
        "light_chain_length": config.light_chain_length,
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
    return AbLangGerminalGradientOutput(**result)
