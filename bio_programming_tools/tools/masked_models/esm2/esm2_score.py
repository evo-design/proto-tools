"""ESM2 scoring tool."""
from __future__ import annotations

import logging
from typing import Literal

from bio_programming_tools.tools.masked_models.shared_data_models import (
    MaskedModelInput,
    MaskedModelScoringOutput,
    SequenceScores,
)
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig, ConfigField
from bio_programming_tools.utils.tool_instance import ToolInstance

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
ESM2ScoringInput = MaskedModelInput
# Output:
ESM2ScoringOutput = MaskedModelScoringOutput

# Config:
class ESM2ScoringConfig(BaseConfig):
    """Configuration for ESM2 sequence scoring.

    Computes true MLM pseudo-perplexity by masking each position individually and
    computing P(x_i | x_{-i}). Uses batched processing for efficiency.

    Attributes:
        model_checkpoint (str): ESM2 model checkpoint to use. Options include
            ``"esm2_t6_8M_UR50D"`` through ``"esm2_t48_15B_UR50D"``.
            Default: ``"esm2_t33_650M_UR50D"``.

        batch_size (int): Number of masked sequence variants to process per forward
            pass. For a sequence of length L, scoring requires L forward passes
            (one per position). This parameter controls how many of those masked
            variants are batched together. Larger batches improve throughput but
            use more GPU memory; reduce if encountering out-of-memory errors.
            Default: ``1``.

        device (str): Device to run the model on. Options include ``"cuda"``,
            ``"cpu"``, ``"mps"``, or specific GPU devices like ``"cuda:0"``.
            Default: ``"cuda"``.

        verbose (bool): Whether to print status messages during scoring.
            Default: ``False``.

        return_logits (bool): Whether to include per-position logits in the output.
            When ``True``, returns logits for each sequence. When ``False``, only
            returns metrics (saves memory and serialization time). Default: ``False``.

    Note:
        - Logits represent P(aa | context with position i masked) for each position
        - The 20 amino acids in vocab are: ACDEFGHIKLMNPQRSTVWY
        - Ambiguous amino acids (X, B, Z) are excluded from perplexity calculation
    """

    model_checkpoint: Literal[ESM2_MODEL_CHECKPOINTS] = ConfigField(
        title="ESM2 Model Checkpoint",
        default="esm2_t33_650M_UR50D",
        description="ESM2 model checkpoint to use",
        reload_on_change=True,
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Number of sequences to process simultaneously on GPU",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        hidden=True,
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
    key="esm2-score",
    label="ESM2 Scoring",
    category="masked_models",
    input=ESM2ScoringInput,
    config=ESM2ScoringConfig,
    output=ESM2ScoringOutput,
    description="Score protein sequences using ESM2 language model",
    uses_gpu=True,
)
def run_esm2_score(
    inputs: ESM2ScoringInput, config: ESM2ScoringConfig,
    instance=None,
) -> ESM2ScoringOutput:
    """Score protein sequences using ESM2 language model.

    Computes MLM pseudo-perplexity by masking each position individually and
    computing P(x_i | x_{-i}). Uses batched processing for efficiency.

    Ambiguous amino acids (X, B, Z, etc.) are excluded from the perplexity
    calculation using the industry-standard exclusion strategy. Only positions
    with standard amino acids (20 canonical AAs) contribute to log-likelihood
    and perplexity metrics.

    Args:
        inputs (MaskedModelInput): Validated input containing protein sequences
            to score.
        config (ESM2ScoringConfig): Scoring configuration specifying model,
            batch size, and whether to return logits.

    Returns:
        MaskedModelScoringOutput: Contains SequenceScores for each input sequence with:

            - ``metrics``: Dict with ``log_likelihood``, ``avg_log_likelihood``,
              ``perplexity``
            - ``logits``: Per-position logits tensor (seq_len, 20) if
              ``return_logits=True``, otherwise ``None``
            - ``vocab``: List of 20 standard amino acid characters if
              ``return_logits=True``, otherwise ``None``

    Examples:
        >>> # Basic scoring (metrics only, no logits)
        >>> inputs = MaskedModelInput(sequences=["MVLSPADKTNVKAAW", "GSSGSSGSS"])
        >>> config = ESM2ScoringConfig(batch_size=32)
        >>> result = run_esm2_score(inputs, config)
        >>> print(f"Perplexity: {result.scores[0].metrics['perplexity']}")
        >>>
        >>> # Scoring with logits for downstream analysis
        >>> config = ESM2ScoringConfig(batch_size=32, return_logits=True)
        >>> result = run_esm2_score(inputs, config)
        >>> print(f"Logits shape: {len(result.scores[0].logits)}")
        >>> print(f"Vocab: {result.scores[0].vocab}")

    Note:
        - Lower perplexity indicates higher model confidence in the sequence
        - ``batch_size`` controls how many masked variants are processed per
          forward pass (not the number of sequences)
        - Logits are from masked forward passes: logits[i] contains
          P(aa | context with position i masked)
        - Sequences with ambiguous AAs are scored only on standard AA positions
        - Set ``return_logits=False`` (default) to save memory when only metrics
          are needed
    """
    logger.debug(f"Using local for ESM2 scoring: {config.model_checkpoint}")
    result = ToolInstance.dispatch(
        "esm2",
        {
            "operation": "score",
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

    sequence_scores = [
        SequenceScores(
            metrics=metrics,
            logits=result["logits"][i] if result["logits"] is not None else None,
            vocab=result["vocab"],
        )
        for i, metrics in enumerate(result["metrics"])
    ]

    return ESM2ScoringOutput(scores=sequence_scores)
