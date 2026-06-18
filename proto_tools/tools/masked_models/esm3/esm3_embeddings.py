"""proto_tools/tools/masked_models/esm3/esm3_embeddings.py.

ESM3 embeddings tool.
"""

import logging
from typing import Any, Literal

from proto_tools.tools.masked_models.projection import attach_projections
from proto_tools.tools.masked_models.shared_data_models import (
    MaskedModelEmbeddingsConfig,
    MaskedModelEmbeddingsOutput,
    MaskedModelInput,
    SequenceEmbedding,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ConfigField, ToolInstance, require_hf_token

logger = logging.getLogger(__name__)

ESM3_MODEL_CHECKPOINTS = Literal["esm3_sm_open_v1",]

# ============================================================================
# Data Models
# ============================================================================
# Input:
ESM3EmbeddingsInput = MaskedModelInput


# Output:
class ESM3EmbeddingsOutput(MaskedModelEmbeddingsOutput):
    """Output from ESM3 protein language model embeddings/logits extraction.

    This class encapsulates the results of ESM3 inference, providing sequence
    embeddings, per-position logits, and attention masks for downstream analysis
    and visualization.

    Inherits from ``MaskedModelEmbeddingsOutput``.

    Attributes:
        results (list[SequenceEmbedding]): Per-sequence embedding results. Each
            ``SequenceEmbedding`` contains:

            - ``mean_embedding``: Mean-pooled embedding vector
            - ``attention_mask``: Binary mask (1 = valid, 0 = padding)
            - ``logits``: Optional per-position amino acid logits (seq_len, vocab_size)

    Note:
        All outputs are returned as nested Python lists (moved to CPU) for easy
        serialization and downstream processing.
    """


# Config:
class ESM3EmbeddingsConfig(MaskedModelEmbeddingsConfig):
    """Configuration for ESM3 protein language model embedding extraction.

    ESM3 is a generative protein language model from EvolutionaryScale; this tool uses
    it to extract mean-pooled embeddings and optional per-position logits.

    Inherits from ``MaskedModelEmbeddingsConfig``.

    Attributes:
        model_checkpoint (ESM3_MODEL_CHECKPOINTS): ESM3 weights variant. Currently
            ``"esm3_sm_open_v1"`` is the only public open-weights checkpoint.
        batch_size (int): Number of sequences to process in parallel. Larger batches
            improve throughput but require more GPU memory.
        device (str): Device to run the model on.
        verbose (bool): Print status messages during model execution.
        return_logits (bool): Include per-position logits in the output (large; disable
            to save memory).
        repr_layer (int): Transformer layer index for embeddings. ``-1`` returns the
            post-norm last-block output (matches ESM2/ESMC ``-1`` semantics); other
            indices select pre-norm per-block hiddens. Both are captured via a forward
            hook on ``model.transformer`` since ``ESM3.forward`` discards them.
    """

    model_checkpoint: ESM3_MODEL_CHECKPOINTS = ConfigField(
        title="ESM3 Model Checkpoint",
        default="esm3_sm_open_v1",
        description="ESM3 weights variant",
        reload_on_change=True,
    )
    return_logits: bool = ConfigField(
        title="Return Logits",
        default=False,
        description="Include per-position logits in the output (large; disable to save memory)",
    )
    repr_layer: int = ConfigField(
        title="Representation Layer",
        default=-1,
        ge=-1,
        description="Transformer layer index for embeddings; -1 returns post-norm output, others select pre-norm",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return ESM3EmbeddingsInput(sequences=["MKTL"])


@tool(
    key="esm3-embedding",
    label="ESM3 Embeddings",
    category="masked_models",
    input_class=ESM3EmbeddingsInput,
    config_class=ESM3EmbeddingsConfig,
    output_class=ESM3EmbeddingsOutput,
    description="Extract protein sequence embeddings and logits using ESM3",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    cacheable=True,
    post_process_iterable=attach_projections,
)
def run_esm3_embeddings(
    inputs: ESM3EmbeddingsInput, config: ESM3EmbeddingsConfig, instance: Any = None
) -> ESM3EmbeddingsOutput:
    """Extract protein sequence embeddings and logits using ESM3.

    Uses ESM3 open model from EvolutionaryScale to extract contextualized embeddings
    and per-position logits for protein sequences. The model is automatically
    loaded on-demand. Supports local GPU execution via isolated Python
    environments.

    Args:
        inputs (ESM3EmbeddingsInput): Validated input containing one or more protein
            sequences (amino acid sequences).
        config (ESM3EmbeddingsConfig): Validated ESM3 configuration specifying model variant,
            batch size, and device settings.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        ESM3EmbeddingsOutput: ``results`` is a list of ``SequenceEmbedding``
            (one per input), each with ``mean_embedding``, ``attention_mask``,
            and optional per-position ``logits``; run metadata is in ``metadata``.

    See Also:
        - ESM3 GitHub Repository: https://github.com/evolutionaryscale/esm

    Examples:
        >>> # Basic embedding extraction
        >>> inputs = MaskedModelInput(sequences=["MVLSPADKTNVKAAW", "GSSGSSGSS"])
        >>> config = ESM3EmbeddingsConfig(verbose=True)
        >>> result = run_esm3_embeddings(inputs, config)
        >>> print(f"Processed {len(result.results)} sequences")
        >>> print(f"Embedding dimension: {len(result.results[0].mean_embedding)}")
        >>>
        >>> # Analyze position-specific predictions
        >>> import numpy as np
        >>> logits_array = np.array(result.results[0].logits)  # First sequence
        >>> predicted_tokens = np.argmax(logits_array, axis=-1)
        >>>
        >>> # Process large batch efficiently
        >>> config = ESM3EmbeddingsConfig(batch_size=32)
        >>> result = run_esm3_embeddings(inputs, config)

    """
    require_hf_token("ESM3", "https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1")

    # Local execution
    logger.debug(f"Using local for ESM3 inference: {config.model_checkpoint}")
    outputs = ToolInstance.dispatch(
        "esm3",
        {
            "operation": "embeddings",
            "sequences": inputs.sequences,
            "batch_size": config.batch_size,
            "model_checkpoint": config.model_checkpoint,
            "device": config.device,
            "verbose": config.verbose,
            "return_logits": config.return_logits,
            "repr_layer": config.repr_layer,
        },
        instance=instance,
        config=config,
    )

    results = [
        SequenceEmbedding(
            mean_embedding=outputs["mean_embeddings"][i],
            attention_mask=outputs["attention_masks"][i],
            logits=outputs["logits"][i] if outputs["logits"] else None,
        )
        for i in range(len(inputs.sequences))
    ]

    return ESM3EmbeddingsOutput(
        metadata={
            "model_checkpoint": config.model_checkpoint,
            "num_sequences": len(inputs.sequences),
            "batch_size": config.batch_size,
            "device": config.device,
        },
        results=results,
    )
