"""proto_tools/tools/masked_models/esmc/esmc_embeddings.py.

ESM C (Cambrian) embeddings tool.
"""

import logging
from typing import Any, Literal

from proto_tools.tools.masked_models.projection import attach_projections
from proto_tools.tools.masked_models.shared_data_models import (
    MaskedModelConfig,
    MaskedModelInput,
    MaskedModelOutput,
    SequenceEmbedding,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ConfigField, ToolInstance

logger = logging.getLogger(__name__)

ESMC_MODEL_CHECKPOINTS = Literal["esmc_300m", "esmc_600m"]

# ============================================================================
# Data Models
# ============================================================================
# Input:
ESMCEmbeddingsInput = MaskedModelInput


# Output:
class ESMCEmbeddingsOutput(MaskedModelOutput):
    """Output from ESM C protein language model embeddings/logits extraction.

    This class encapsulates the results of ESM C inference, providing sequence
    embeddings, per-position logits, and attention masks for downstream analysis.

    Inherits from ``MaskedModelOutput``.

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
class ESMCEmbeddingsConfig(MaskedModelConfig):
    """Configuration object for ESM C protein language model embeddings extraction.

    ESM C (Cambrian) is an embedding-focused masked language model from
    EvolutionaryScale. Two open-weights variants are self-hostable; the 6B variant
    is API-only via Forge and not exposed here.

    Inherits from ``MaskedModelConfig``.

    Attributes:
        model_checkpoint (Literal[ESMC_MODEL_CHECKPOINTS]): ESM C checkpoint to use:

            - ``"esmc_300m"`` (default): 300M-parameter model. Released under the
              `Cambrian Open License Agreement
              <https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement>`_
              — commercial use permitted.
            - ``"esmc_600m"``: 600M-parameter model. Released under the
              `Cambrian Non-Commercial License Agreement
              <https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement>`_
              — research/internal use only; commercial use is **not** permitted.

            Default: ``"esmc_300m"``.

        batch_size (int): Number of sequences to process in parallel. Larger batches
            improve throughput but require more GPU memory. Default: 1.

        device (str): Device to run the model on. Options include ``"cuda"`` (NVIDIA GPU),
            ``"cpu"`` (CPU execution), or specific GPU devices like ``"cuda:0"``.
            Default: ``"cuda"``.

        verbose (bool): Whether to print status messages during model execution.
            Default: ``False``.

        return_logits (bool): Whether to include per-position logits in the output.
            When ``True``, returns logits for each sequence. When ``False``, only
            returns embeddings (saves memory and serialization time). Default: ``False``.

    Note:
        The model is loaded on-demand for each call.
    """

    model_checkpoint: Literal[ESMC_MODEL_CHECKPOINTS] = ConfigField(
        title="ESM C Model Checkpoint",
        default="esmc_300m",
        description="ESM C model checkpoint to use ('esmc_300m' open license, 'esmc_600m' non-commercial)",
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
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return ESMCEmbeddingsInput(sequences=["MKTL"])


@tool(
    key="esmc-embedding",
    label="ESM C Embeddings",
    category="masked_models",
    input_class=ESMCEmbeddingsInput,
    config_class=ESMCEmbeddingsConfig,
    output_class=ESMCEmbeddingsOutput,
    description="Extract protein sequence embeddings and logits using ESM C (Cambrian)",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="sequences",
    iterable_output_field="results",
    cacheable=True,
    post_process_iterable=attach_projections,
)
def run_esmc_embeddings(
    inputs: ESMCEmbeddingsInput, config: ESMCEmbeddingsConfig, instance: Any = None
) -> ESMCEmbeddingsOutput:
    """Extract protein sequence embeddings and logits using ESM C.

    Uses ESM C (Cambrian) from EvolutionaryScale to produce contextualized
    per-sequence embeddings and (optionally) per-position logits. Runs locally
    on GPU in an isolated Python environment shared with the ESM3 wrapper
    (both ship in the same ``esm`` package).

    Args:
        inputs (ESMCEmbeddingsInput): Validated input containing one or more protein
            sequences (amino acid sequences).
        config (ESMCEmbeddingsConfig): Validated ESM C configuration specifying model
            variant, batch size, and device settings.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        ESMCEmbeddingsOutput: Structured output containing one
            ``SequenceEmbedding`` per input sequence. Each entry holds the
            mean-pooled embedding, attention mask, and (when requested)
            per-position amino acid logits.

    See Also:
        - ESM C blog post: https://www.evolutionaryscale.ai/blog/esm-cambrian
        - ESM GitHub: https://github.com/evolutionaryscale/esm

    Examples:
        >>> from proto_tools.tools.masked_models.esmc import (
        ...     ESMCEmbeddingsConfig,
        ...     ESMCEmbeddingsInput,
        ...     run_esmc_embeddings,
        ... )
        >>> inputs = ESMCEmbeddingsInput(sequences=["MVLSPADKTNVKAAW", "GSSGSSGSS"])
        >>> result = run_esmc_embeddings(inputs, ESMCEmbeddingsConfig())
        >>> print(f"Embedding dimension: {len(result.results[0].mean_embedding)}")
    """
    logger.debug(f"Using local for ESM C inference: {config.model_checkpoint}")
    outputs = ToolInstance.dispatch(
        "esmc",
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

    return ESMCEmbeddingsOutput(
        metadata={
            "model_checkpoint": config.model_checkpoint,
            "num_sequences": len(inputs.sequences),
            "batch_size": config.batch_size,
            "device": config.device,
        },
        results=results,
    )
