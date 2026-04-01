"""proto_tools/tools/masked_models/esm2/esm2_embeddings.py.

ESM2 embeddings tool.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from proto_tools.tools.masked_models.shared_data_models import (
    MaskedModelConfig,
    MaskedModelInput,
    MaskedModelOutput,
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

# ============================================================================
# Data Models
# ============================================================================
# Input:
ESM2EmbeddingsInput = MaskedModelInput


# Output:
class ESM2EmbeddingsOutput(MaskedModelOutput):
    """Output from ESM2 protein language model inference.

    This class encapsulates the results of ESM2 inference, providing sequence
    embeddings, per-position logits, and attention masks for downstream analysis
    and visualization.

    Inherits from ``MaskedModelOutput``.

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
class ESM2EmbeddingsConfig(MaskedModelConfig):
    """Configuration object for ESM2 protein language model.

    This class defines configuration parameters for running ESM2 inference to
    extract protein sequence embeddings and logits. ESM2 (Evolutionary Scale
    Modeling 2) is a transformer-based protein language model trained on millions
    of protein sequences.

    Inherits from ``MaskedModelConfig``.

    Attributes:
        model_checkpoint (Literal[ESM2_MODEL_CHECKPOINTS]): Name of the ESM2 model variant to use. Options:

            - ``"esm2_t6_8M_UR50D"``: 8M parameters, 6 layers (fastest, 320-dim embeddings)
            - ``"esm2_t12_35M_UR50D"``: 35M parameters, 12 layers (480-dim embeddings)
            - ``"esm2_t30_150M_UR50D"``: 150M parameters, 30 layers (640-dim embeddings)
            - ``"esm2_t33_650M_UR50D"``: 650M parameters, 33 layers (1280-dim embeddings, default)
            - ``"esm2_t36_3B_UR50D"``: 3B parameters, 36 layers (2560-dim embeddings)
            - ``"esm2_t48_15B_UR50D"``: 15B parameters, 48 layers (5120-dim embeddings, best quality)

            Larger models provide higher quality embeddings and better predictions
            but require more GPU memory and inference time. The 650M model offers
            a good balance of quality and speed. Default: ``"esm2_t33_650M_UR50D"``.

        batch_size (int): Number of sequences to process in parallel. Larger batches
            improve throughput but require more GPU memory. Optimal values depend on
            GPU memory, model size, and sequence lengths. Typical values range from
            1 (safest) to 128 (faster, more memory). Default: 1.

        device (str): Device to run the model on. Options include ``"cuda"`` (NVIDIA GPU),
            ``"cpu"`` (CPU execution), ``"mps"`` (Apple Metal), or specific GPU devices
            like ``"cuda:0"``. Default: ``"cuda"``.

        verbose: Whether to print status messages during model execution,
            including loading progress and timing information. Default: ``False``.

        return_logits (bool): Whether to include per-position logits in the output.
            When ``True``, returns logits for each sequence. When ``False``, only
            returns metrics (saves memory and serialization time). Default: ``False``.

    Note:
        The model is loaded on-demand for each call.
    """

    model_checkpoint: Literal[ESM2_MODEL_CHECKPOINTS] = ConfigField(
        title="ESM2 Model Checkpoint",
        default="esm2_t33_650M_UR50D",
        description="Name of the ESM2 model variant to use",
        reload_on_change=True,
    )
    return_logits: bool = ConfigField(
        title="Return Logits",
        default=False,
        description="Whether to include per-position logits in the output. Disable to save memory.",
        advanced=True,
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
    iterable_input_field="sequences",
    iterable_output_field="results",
    cacheable=True,
)
def run_esm2_embeddings(
    inputs: ESM2EmbeddingsInput, config: ESM2EmbeddingsConfig | None = None, instance: Any = None
) -> ESM2EmbeddingsOutput:
    """Extract protein sequence embeddings and logits using ESM2.

    Uses ESM2 from Meta AI to extract contextualized embeddings and per-position
    logits for protein sequences. The model is automatically loaded on-demand.
    Supports local GPU execution via isolated Python environments.

    Args:
        inputs (ESM2EmbeddingsInput): Validated input containing one or more protein
            sequences (amino acid sequences).
        config (ESM2EmbeddingsConfig | None): Validated ESM2 configuration specifying model variant,
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
    logger.debug(f"Using local for ESM2 inference: {config.model_checkpoint}")  # type: ignore[union-attr]
    outputs = ToolInstance.dispatch(
        "esm2",
        {
            "operation": "embeddings",
            "sequences": inputs.sequences,
            "batch_size": config.batch_size,  # type: ignore[union-attr]
            "model_checkpoint": config.model_checkpoint,  # type: ignore[union-attr]
            "device": config.device,  # type: ignore[union-attr]
            "verbose": config.verbose,  # type: ignore[union-attr]
            "return_logits": config.return_logits,  # type: ignore[union-attr]
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
            "model_checkpoint": config.model_checkpoint,  # type: ignore[union-attr]
            "num_sequences": len(inputs.sequences),
            "batch_size": config.batch_size,  # type: ignore[union-attr]
            "device": config.device,  # type: ignore[union-attr]
        },
        results=results,
    )
