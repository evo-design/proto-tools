"""Minerva interactions for prepared mixed protein/DNA sequences."""

import logging
from typing import Any, Literal

from pydantic import field_validator

from proto_tools.tools.masked_models.mixed_data_models import (
    InteractionHead,
    MixedInteractionsOutput,
    MixedModelConfig,
    MixedSequenceInput,
    dispatch_mixed_model,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ConfigField

logger = logging.getLogger(__name__)

# Input:
MinervaInteractionsInput = MixedSequenceInput

# Output:
MinervaInteractionsOutput = MixedInteractionsOutput


class MinervaInteractionsConfig(MixedModelConfig):
    """Configuration for Minerva interactions.

    Attributes:
        model_checkpoint (Literal['gbrixi/minerva-mlm', 'gbrixi/minerva-mlm-8k']): Public model checkpoint.
        heads (list[InteractionHead]): Selected named interaction heads.
        interaction_layers (Literal[2, 6]): Number of final transformer layers used by the heads.
        batch_size (int): Equal-token-length sequences per forward pass; one limits matrix memory.
    """

    model_checkpoint: Literal["gbrixi/minerva-mlm", "gbrixi/minerva-mlm-8k"] = ConfigField(
        default="gbrixi/minerva-mlm",
        title="Model Checkpoint",
        description="Minerva checkpoint; downloaded automatically from Hugging Face",
        reload_on_change=True,
    )

    heads: list[InteractionHead] = ConfigField(
        default_factory=lambda: ["base_pairing", "protein", "repeat"],
        min_length=1,
        title="Interaction Heads",
        description="Named interaction heads to return; select fewer to reduce output size",
    )
    interaction_layers: Literal[2, 6] = ConfigField(
        default=2,
        title="Interaction Layers",
        description="Use interaction heads trained on the last two or six transformer layers",
    )

    @field_validator("heads")
    @classmethod
    def unique_heads(cls, heads: list[InteractionHead]) -> list[InteractionHead]:
        """Reject duplicate head names."""
        if len(set(heads)) != len(heads):
            raise ValueError("heads must not contain duplicates")
        return heads


def example_input() -> MinervaInteractionsInput:
    """Return a short prepared locus for examples and infrastructure checks."""
    return MinervaInteractionsInput(sequences=["<+>MKTL<+>acgt<->ACDE"])


@tool(
    key="minerva-interactions",
    label="Minerva Interactions",
    category="masked_models",
    input_class=MinervaInteractionsInput,
    config_class=MinervaInteractionsConfig,
    output_class=MinervaInteractionsOutput,
    description="Predict protein, RNA base-pairing, and repeat interaction maps with Minerva",
    uses_gpu=True,
    example_input=example_input,
    cacheable=True,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    max_chunk_size=1,
)
def run_minerva_interactions(
    inputs: MinervaInteractionsInput,
    config: MinervaInteractionsConfig,
    instance: Any = None,
) -> MinervaInteractionsOutput:
    """Run Minerva interactions on prepared mixed-modality input.

    Args:
        inputs (MinervaInteractionsInput): Validated mixed-sequence input.
        config (MinervaInteractionsConfig): Checkpoint and operation settings.
        instance (Any): Optional persistent tool instance.

    Returns:
        MinervaInteractionsOutput: Token-aligned interactions output.
    """
    logger.debug("Using local worker for Minerva interactions: %s", config.model_checkpoint)
    return MinervaInteractionsOutput(**dispatch_mixed_model("minerva", "interactions", inputs, config, instance))
