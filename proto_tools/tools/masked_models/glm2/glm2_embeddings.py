"""gLM2 embeddings for prepared mixed protein/DNA sequences."""

import logging
from typing import Any

from proto_tools.tools.masked_models.execution import dispatch_masked_model
from proto_tools.tools.masked_models.glm2.config import GLM2Config
from proto_tools.tools.masked_models.mixed_configs import MixedEmbeddingsConfig
from proto_tools.tools.masked_models.mixed_data_models import (
    MixedEmbeddingsOutput,
    MixedSequenceInput,
)
from proto_tools.tools.masked_models.projection import attach_projections
from proto_tools.tools.tool_registry import tool

logger = logging.getLogger(__name__)

# Input:
GLM2EmbeddingsInput = MixedSequenceInput

# Output:
GLM2EmbeddingsOutput = MixedEmbeddingsOutput


class GLM2EmbeddingsConfig(GLM2Config, MixedEmbeddingsConfig):
    """Configuration for gLM2 embeddings.

    Attributes:
        model_checkpoint (Literal['tattabio/gLM2_150M', 'tattabio/gLM2_650M']): Public model checkpoint.
        batch_size (int): Sequences per forward pass.
        return_logits (bool): Include token-aligned biological logits.
        repr_layer (int): 0=embedding table, 1..N=transformer output, -1=last layer.
    """


def example_input() -> GLM2EmbeddingsInput:
    """Return a short prepared locus for examples and infrastructure checks."""
    return GLM2EmbeddingsInput(sequences=["+MKTL+acgt-ACDE"])


@tool(
    key="glm2-embedding",
    label="gLM2 Embeddings",
    category="masked_models",
    input_class=GLM2EmbeddingsInput,
    config_class=GLM2EmbeddingsConfig,
    output_class=GLM2EmbeddingsOutput,
    description="Extract mixed protein/DNA embeddings and biological logits with gLM2",
    uses_gpu=True,
    example_input=example_input,
    cacheable=True,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    max_chunk_size=32,
    post_process_iterable=attach_projections,
)
def run_glm2_embeddings(
    inputs: GLM2EmbeddingsInput,
    config: GLM2EmbeddingsConfig,
    instance: Any = None,
) -> GLM2EmbeddingsOutput:
    """Run gLM2 embeddings on prepared mixed-modality input.

    Args:
        inputs (GLM2EmbeddingsInput): Validated mixed-sequence input.
        config (GLM2EmbeddingsConfig): Checkpoint and operation settings.
        instance (Any): Optional persistent tool instance.

    Returns:
        GLM2EmbeddingsOutput: Token-aligned embeddings output.
    """
    logger.debug("Using local worker for gLM2 embeddings: %s", config.model_checkpoint)
    return GLM2EmbeddingsOutput(**dispatch_masked_model("glm2", "embeddings", inputs, config, instance))
