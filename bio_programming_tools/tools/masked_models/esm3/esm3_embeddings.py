"""ESM3 embeddings tool."""
from __future__ import annotations

import logging
from typing import Literal

from bio_programming_tools.tools.masked_models.shared_data_models import (
    MaskedModelConfig,
    MaskedModelInput,
    MaskedModelOutput,
    SequenceEmbedding,
)
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import ConfigField
from bio_programming_tools.utils.tool_instance import ToolInstance

logger = logging.getLogger(__name__)

ESM3_MODEL_CHECKPOINTS = Literal[
    "esm3_sm_open_v1",
]

# ============================================================================
# Data Models
# ============================================================================
# Input:
ESM3EmbeddingsInput = MaskedModelInput

# Output:
class ESM3EmbeddingsOutput(MaskedModelOutput):
    """Output from ESM3 protein language model embeddings/logits extraction.

    This class encapsulates the results of ESM3 inference, providing sequence
    embeddings, per-position logits, and attention masks for downstream analysis
    and visualization.

    Inherits from ``MaskedModelOutput``.

    Attributes:
        results (List[SequenceEmbedding]): Per-sequence embedding results. Each
            ``SequenceEmbedding`` contains:

            - ``mean_embedding``: Mean-pooled embedding vector
            - ``attention_mask``: Binary mask (1 = valid, 0 = padding)
            - ``logits``: Optional per-position amino acid logits (seq_len, vocab_size)

    Note:
        All outputs are returned as nested Python lists (moved to CPU) for easy
        serialization and downstream processing.
    """

# Config:
class ESM3EmbeddingsConfig(MaskedModelConfig):
    """Configuration object for ESM3 protein language model embeddings extraction.

    This class defines configuration parameters for running ESM3 inference to
    extract protein sequence embeddings and logits. ESM3 is a generative protein
    language model from EvolutionaryScale that can perform both embedding extraction
    and structure prediction.

    Inherits from ``MaskedModelConfig``.

    Attributes:
        model_checkpoint (str): ESM3 model checkpoint to use. Currently available:

            - ``"esm3_sm_open_v1"``: Small open-source ESM3 model (default)

            Future versions may include additional model variants.
            Default: ``"esm3_sm_open_v1"``.

        batch_size (int): Number of sequences to process in parallel. Larger batches
            improve throughput but require more GPU memory. Optimal values depend on
            GPU memory, model size, and sequence lengths. Typical values range from
            1 (safest) to 128 (faster, more memory). Default: 128.

        device (str): Device to run the model on. Options include ``"cuda"`` (NVIDIA GPU),
            ``"cpu"`` (CPU execution), ``"mps"`` (Apple Metal), or specific GPU devices
            like ``"cuda:0"``. Default: ``"cuda"``.

        verbose (bool): Whether to print status messages during model execution,
            including loading progress and timing information. Default: ``False``.

        return_logits (bool): Whether to include per-position logits in the output.
            When ``True``, returns logits for each sequence. When ``False``, only
            returns metrics (saves memory and serialization time). Default: ``False``.

    Note:
        The model is loaded on-demand for each call.
    """
    model_checkpoint: Literal[ESM3_MODEL_CHECKPOINTS] = ConfigField(
        title="ESM3 Model Checkpoint",
        default="esm3_sm_open_v1",
        description="ESM3 model checkpoint to use",
        reload_on_change=True,
    )
    return_logits: bool = ConfigField(
        title="Return Logits",
        default=False,
        description="Whether to include per-position logits in the output. Disable to save memory.",
        advanced=True,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input():
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
    iterable_input_field="sequences",
    iterable_output_field="results",
    cacheable=True,
)
def run_esm3_embeddings(inputs: ESM3EmbeddingsInput, config: ESM3EmbeddingsConfig | None = None, instance=None) -> ESM3EmbeddingsOutput:
    """Extract protein sequence embeddings and logits using ESM3.

    Uses ESM3 open model from EvolutionaryScale to extract contextualized embeddings
    and per-position logits for protein sequences. The model is automatically
    loaded on-demand. Supports local GPU execution via isolated Python
    environments.

    Args:
        inputs (MaskedModelInput): Validated input containing one or more protein
            sequences (amino acid sequences).
        config (ESM3EmbeddingsConfig): Validated ESM3 configuration specifying model variant,
            batch size, and device settings.

    Returns:
        ESM3EmbeddingsOutput: Structured output containing:
            - ``mean_embeddings``: Mean-pooled embeddings for each sequence
            - ``logits``: Per-position amino acid logits for each sequence
            - ``attention_masks``: Binary masks for valid positions
            - ``num_sequences``: Number of sequences processed

    See Also:
        - ESM3 GitHub Repository: https://github.com/evolutionaryscale/esm

    Examples:
        >>> # Basic embedding extraction
        >>> inputs = MaskedModelInput(
        ...     sequences=["MVLSPADKTNVKAAW", "GSSGSSGSS"]
        ... )
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
