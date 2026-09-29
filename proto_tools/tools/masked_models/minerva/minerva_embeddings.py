"""Minerva embeddings for prepared mixed protein/DNA sequences."""

import logging
from typing import Any

from proto_tools.tools.masked_models.execution import dispatch_masked_model
from proto_tools.tools.masked_models.minerva.config import MinervaConfig
from proto_tools.tools.masked_models.mixed_configs import MixedEmbeddingsConfig
from proto_tools.tools.masked_models.mixed_data_models import (
    MixedEmbeddingsOutput,
    MixedSequenceInput,
)
from proto_tools.tools.masked_models.projection import attach_projections
from proto_tools.tools.tool_registry import tool

logger = logging.getLogger(__name__)

# Input:
MinervaEmbeddingsInput = MixedSequenceInput

# Output:
MinervaEmbeddingsOutput = MixedEmbeddingsOutput


class MinervaEmbeddingsConfig(MinervaConfig, MixedEmbeddingsConfig):
    """Configuration for Minerva embeddings.

    Attributes:
        model_checkpoint (Literal['gbrixi/minerva-mlm', 'gbrixi/minerva-mlm-8k']): Public model checkpoint.
        batch_size (int): Sequences per forward pass.
        return_logits (bool): Include token-aligned biological logits.
        repr_layer (int): 0=embedding table, 1..N=transformer output, -1=last layer.
    """


def example_input() -> MinervaEmbeddingsInput:
    """Return a short prepared locus for examples and infrastructure checks."""
    return MinervaEmbeddingsInput(sequences=["<+>MKTL<+>acgt<->ACDE"])


@tool(
    key="minerva-embedding",
    label="Minerva Embeddings",
    category="masked_models",
    input_class=MinervaEmbeddingsInput,
    config_class=MinervaEmbeddingsConfig,
    output_class=MinervaEmbeddingsOutput,
    description="Extract mixed protein/DNA embeddings and biological logits with Minerva",
    uses_gpu=True,
    example_input=example_input,
    cacheable=True,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    max_chunk_size=32,
    post_process_iterable=attach_projections,
)
def run_minerva_embeddings(
    inputs: MinervaEmbeddingsInput,
    config: MinervaEmbeddingsConfig,
    instance: Any = None,
) -> MinervaEmbeddingsOutput:
    """Run Minerva embeddings on prepared mixed-modality input.

    Args:
        inputs (MinervaEmbeddingsInput): Validated mixed-sequence input.
        config (MinervaEmbeddingsConfig): Checkpoint and operation settings.
        instance (Any): Optional persistent tool instance.

    Returns:
        MinervaEmbeddingsOutput: Token-aligned embeddings output.
    """
    logger.debug("Using local worker for Minerva embeddings: %s", config.model_checkpoint)
    return MinervaEmbeddingsOutput(**dispatch_masked_model("minerva", "embeddings", inputs, config, instance))
