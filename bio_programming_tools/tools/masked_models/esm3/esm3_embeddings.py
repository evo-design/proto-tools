"""ESM3 embeddings tool."""
from __future__ import annotations

import logging
from typing import List, Literal, Optional

from pydantic import Field

from bio_programming_tools.tools.masked_models.shared_data_models import (
    MaskedModelConfig,
    MaskedModelInput,
    MaskedModelOutput,
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
        mean_embeddings (List[List[float]]): Mean-pooled sequence embeddings for
            each input sequence. Shape: ``[num_sequences, embedding_dim]`` where
            ``embedding_dim`` depends on the model variant. These embeddings capture
            the overall semantic representation of each protein sequence and are
            useful for:

            - Sequence similarity comparisons
            - Clustering and classification tasks
            - Downstream machine learning models
            - Visualization via dimensionality reduction

        num_sequences (int): Total number of sequences that were processed in the
            inference run. This count matches the length of the input sequences list
            and all output lists.

        logits (List[List[List[float]]]): Per-position amino acid logits.
            Shape: ``[num_sequences, seq_len, vocab_size]``. Useful for:

            - Identifying uncertain positions in sequences
            - Guided protein design and mutagenesis
            - Computing sequence likelihoods and perplexities
            - Zero-shot variant effect prediction

        attention_masks (List[List[int]]): Binary masks indicating valid positions
            in each sequence. Shape: ``[num_sequences, seq_len]``. Values are:

            - ``1``: Valid amino acid position
            - ``0``: Padding position (for batched sequences of different lengths)

            Use these masks to filter out padding when analyzing embeddings or logits.

    Note:
        All outputs are returned as nested Python lists (moved to CPU) for easy
        serialization and downstream processing.
    """
    logits: Optional[List[List[List[float]]]] = Field(
        default=None,
        description="Per-position amino acid logits. Shape: [num_sequences, seq_len, vocab_size]."
    )
    attention_masks: List[List[int]] = Field(
        description="Attention masks for each sequence (shape: [num_sequences, seq_len])"
    )

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
@tool(
    key="esm3-embedding",
    label="ESM3 Embeddings",
    category="masked_models",
    input=ESM3EmbeddingsInput,
    config=ESM3EmbeddingsConfig,
    output=ESM3EmbeddingsOutput,
    description="Extract protein sequence embeddings and logits using ESM3",
    uses_gpu=True,
)
def run_esm3_embeddings(inputs: ESM3EmbeddingsInput, config: ESM3EmbeddingsConfig, instance=None) -> ESM3EmbeddingsOutput:
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
        >>> print(f"Processed {result.num_sequences} sequences")
        >>> print(f"Embedding dimension: {len(result.mean_embeddings[0])}")
        >>>
        >>> # Analyze position-specific predictions
        >>> import numpy as np
        >>> logits_array = np.array(result.logits[0])  # First sequence
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
        verbose=config.verbose,
        reload_on=type(config).reload_fields(),
    )

    return ESM3EmbeddingsOutput(
        metadata={
            "model_checkpoint": config.model_checkpoint,
            "num_sequences": len(inputs.sequences),
            "batch_size": config.batch_size,
            "device": config.device,
        },
        mean_embeddings=outputs["mean_embeddings"],
        num_sequences=len(inputs.sequences),
        logits=outputs["logits"],
        attention_masks=outputs["attention_masks"],
    )
