"""proto_tools/tools/masked_models/esm2/esm2_embeddings.py.

ESM2 embeddings tool.
"""

import logging
from typing import Any, Literal

from pydantic import field_validator

from proto_tools.tools.masked_models.projection import attach_projections
from proto_tools.tools.masked_models.shared_data_models import (
    MaskedModelEmbeddingsConfig,
    MaskedModelEmbeddingsOutput,
    MaskedModelInput,
    SequenceEmbedding,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ConfigField, ToolInstance

logger = logging.getLogger(__name__)

ESM2_MODEL_CHECKPOINTS = Literal[
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
    "esm2_t48_15B_UR50D",
]

ESM2_MAX_SEQ_LENGTH = 1022


# ============================================================================
# Data Models
# ============================================================================
# Input:
class ESM2EmbeddingsInput(MaskedModelInput):
    """ESM-2 embedding input.

    Attributes:
        sequences (list[str]): Protein sequence(s) to process. Each must be ≤ 1022
            residues (ESM-2's positional-encoding cap); over-length inputs raise
            ``ValueError``.
    """

    @field_validator("sequences")
    @classmethod
    def _validate_max_length(cls, sequences: list[str]) -> list[str]:
        for idx, seq in enumerate(sequences):
            if len(seq) > ESM2_MAX_SEQ_LENGTH:
                raise ValueError(
                    f"esm2: supports sequences up to {ESM2_MAX_SEQ_LENGTH} residues; input {idx} has length {len(seq)}."
                )
        return sequences


# Output:
class ESM2EmbeddingsOutput(MaskedModelEmbeddingsOutput):
    """Output from ESM2 protein language model inference.

    This class encapsulates the results of ESM2 inference, providing sequence
    embeddings, per-position logits, and attention masks for downstream analysis
    and visualization.

    Inherits from ``MaskedModelEmbeddingsOutput``.

    Attributes:
        results (list[SequenceEmbedding]): Per-sequence embedding results. Each
            ``SequenceEmbedding`` contains:

            - ``mean_embedding``: Mean-pooled embedding vector (e.g., 1280-dim for
              ``esm2_t33_650M_UR50D``)
            - ``attention_mask``: Binary mask (1 = valid, 0 = padding)
            - ``logits``: Optional per-position amino acid logits (seq_len, 20)

    Note:
        All outputs are returned as nested Python lists (moved to CPU) for
        serialization and downstream processing.
    """


# Config:
class ESM2EmbeddingsConfig(MaskedModelEmbeddingsConfig):
    """Configuration for ESM2 protein language model embedding extraction.

    ESM2 (Evolutionary Scale Modeling 2) is a transformer-based protein language model
    trained on millions of protein sequences.

    Inherits from ``MaskedModelEmbeddingsConfig``.

    Attributes:
        model_checkpoint (ESM2_MODEL_CHECKPOINTS): ESM2 weights variant. Sizes range from
            8M (320-dim, fastest) to 15B (5120-dim, highest quality). The 650M variant
            offers a good speed/quality trade-off.
        batch_size (int): Number of sequences to process in parallel. Larger batches improve
            throughput but require more GPU memory.
        device (str): Device to run the model on.
        verbose (bool): Print status messages during model execution.
        return_logits (bool): Include per-position logits in the output (large; disable to
            save memory).
        repr_layer (int): Transformer layer index for embeddings. ``-1`` selects the last
            (top) layer; uses HuggingFace ``hidden_states`` indexing where ``0`` is the
            embedding-layer output and ``N`` is transformer layer N.
    """

    model_checkpoint: ESM2_MODEL_CHECKPOINTS = ConfigField(
        title="ESM2 Model Checkpoint",
        default="esm2_t33_650M_UR50D",
        description="ESM2 weights variant; trade off speed vs embedding quality",
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
        description="Transformer layer index for embeddings (0=embedding output, N=layer N, -1=last)",
    )


# Output:
# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return ESM2EmbeddingsInput(sequences=["MKTL"])


@tool(
    key="esm2-embedding",
    label="ESM2 Embeddings",
    category="masked_models",
    input_class=ESM2EmbeddingsInput,
    config_class=ESM2EmbeddingsConfig,
    output_class=ESM2EmbeddingsOutput,
    description="Extract protein sequence embeddings and logits using ESM2",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    cacheable=True,
    post_process_iterable=attach_projections,
)
def run_esm2_embeddings(
    inputs: ESM2EmbeddingsInput, config: ESM2EmbeddingsConfig, instance: Any = None
) -> ESM2EmbeddingsOutput:
    """Extract protein sequence embeddings and logits using ESM2.

    Uses ESM2 from Meta AI to extract contextualized embeddings and per-position
    logits for protein sequences. The model is automatically loaded on-demand.
    Supports local GPU execution via isolated Python environments.

    Args:
        inputs (ESM2EmbeddingsInput): Validated input containing one or more protein
            sequences (amino acid sequences).
        config (ESM2EmbeddingsConfig): Validated ESM2 configuration specifying model variant,
            batch size, and device settings.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        ESM2EmbeddingsOutput: Structured output containing:
            - ``mean_embeddings``: Mean-pooled embeddings for each sequence
            - ``logits``: Per-position amino acid logits for each sequence
            - ``attention_masks``: Binary masks for valid positions
            - ``num_sequences``: Number of sequences processed

    Examples:
        >>> # Basic embedding extraction
        >>> inputs = MaskedModelInput(sequences=["MVLSPADKTNVKAAW", "GSSGSSGSS"])
        >>> config = ESM2EmbeddingsConfig(model_checkpoint="esm2_t33_650M_UR50D", batch_size=2, verbose=True)
        >>> result = run_esm2_embeddings(inputs, config)
        >>> print(f"Processed {len(result.results)} sequences")
        >>> print(f"Embedding dimension: {len(result.results[0].mean_embedding)}")
        >>>
        >>> # Analyze position-specific predictions
        >>> import numpy as np
        >>> logits_array = np.array(result.results[0].logits)  # First sequence
        >>> predicted_tokens = np.argmax(logits_array, axis=-1)
        >>>
        >>> # Extract embeddings using smaller model on CPU
        >>> config = ESM2EmbeddingsConfig(model_checkpoint="esm2_t6_8M_UR50D", device="cpu")
        >>> result = run_esm2_embeddings(inputs, config)
        >>>
        >>> # Process large batch efficiently
        >>> config = ESM2EmbeddingsConfig(batch_size=32)
        >>> result = run_esm2_embeddings(inputs, config)

    See Also:
        - ESM2 GitHub Repository: https://github.com/facebookresearch/esm

    Note:
        - Larger models require more GPU memory but provide better representations
    """
    logger.debug(f"Using local for ESM2 inference: {config.model_checkpoint}")
    outputs = ToolInstance.dispatch(
        "esm2",
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

    return ESM2EmbeddingsOutput(
        metadata={
            "model_checkpoint": config.model_checkpoint,
            "num_sequences": len(inputs.sequences),
            "batch_size": config.batch_size,
            "device": config.device,
        },
        results=results,
    )
