"""proto_tools/tools/masked_models/esm2/esm2_score.py.

ESM2 scoring tool.
"""

import logging
from typing import Any, Literal

from pydantic import field_validator

from proto_tools.tools.masked_models.shared_data_models import (
    MaskedModelInput,
    MaskedModelScoringConfig,
    MaskedModelScoringMetrics,
    MaskedModelScoringOutput,
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
class ESM2ScoringInput(MaskedModelInput):
    """ESM-2 scoring input.

    Attributes:
        sequences (list[str]): Protein sequence(s) to score. Each must be ≤ 1022
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
ESM2ScoringOutput = MaskedModelScoringOutput


# Config:
class ESM2ScoringConfig(MaskedModelScoringConfig):
    """Configuration for ESM2 sequence scoring.

    Computes true MLM pseudo-perplexity by masking each position individually and
    computing P(x_i | x_{-i}). Uses batched processing for efficiency.

    Attributes:
        model_checkpoint (ESM2_MODEL_CHECKPOINTS): ESM2 weights variant.
        batch_size (int): Masked variants per forward pass, pooled across all input sequences.
            Larger batches improve throughput but use more memory.
        device (str): Device to run the model on.
        verbose (bool): Print status messages during scoring.
        return_logits (bool): Include per-position logits in the output (large; disable to
            save memory).

    Note:
        - Logits represent P(aa | context with position i masked) for each position.
        - The 20 amino acids in vocab are: ACDEFGHIKLMNPQRSTVWY.
        - Ambiguous amino acids (X, B, Z) are excluded from perplexity calculation.
    """

    model_checkpoint: ESM2_MODEL_CHECKPOINTS = ConfigField(
        title="ESM2 Model Checkpoint",
        default="esm2_t33_650M_UR50D",
        description="ESM2 weights variant; trade off speed vs scoring fidelity",
        reload_on_change=True,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return ESM2ScoringInput(sequences=["MKTL"])


@tool(
    key="esm2-score",
    label="ESM2 Scoring",
    category="masked_models",
    input_class=ESM2ScoringInput,
    config_class=ESM2ScoringConfig,
    output_class=ESM2ScoringOutput,
    description="Score protein sequences using ESM2 language model",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="sequences",
    iterable_output_field="scores",
    cacheable=True,
)
def run_esm2_score(
    inputs: ESM2ScoringInput,
    config: ESM2ScoringConfig,
    instance: Any = None,
) -> ESM2ScoringOutput:
    """Score protein sequences using ESM2 language model.

    Computes MLM pseudo-perplexity by masking each position individually and
    computing P(x_i | x_{-i}). Uses batched processing for efficiency.

    Ambiguous amino acids (X, B, Z, etc.) are excluded from the perplexity
    calculation using the industry-standard exclusion strategy. Only positions
    with standard amino acids (20 canonical AAs) contribute to log-likelihood
    and perplexity metrics.

    Args:
        inputs (ESM2ScoringInput): Validated input containing protein sequences
            to score.
        config (ESM2ScoringConfig): Scoring configuration specifying model,
            batch size, and whether to return logits.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        ESM2ScoringOutput: Contains a ``MaskedModelScoringMetrics`` for each input
            sequence with:

            - ``log_likelihood``, ``avg_log_likelihood``, ``perplexity`` (access via
              attribute ``score.perplexity`` or mapping ``score["perplexity"]``)
            - ``logits``: Per-position logits tensor (seq_len, 20) if
              ``return_logits=True``, otherwise ``None``
            - ``vocab``: List of 20 standard amino acid characters if
              ``return_logits=True``, otherwise ``None``

    Examples:
        >>> # Basic scoring (metrics only, no logits)
        >>> inputs = MaskedModelInput(sequences=["MVLSPADKTNVKAAW", "GSSGSSGSS"])
        >>> config = ESM2ScoringConfig(batch_size=32)
        >>> result = run_esm2_score(inputs, config)
        >>> print(f"Perplexity: {result.scores[0]['perplexity']}")
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
            "seed": config.seed,
        },
        instance=instance,
        config=config,
    )

    sequence_scores = [
        MaskedModelScoringMetrics(
            **metrics,
            logits=result["logits"][i] if result["logits"] is not None else None,
            vocab=result["vocab"],
        )
        for i, metrics in enumerate(result["metrics"])
    ]

    return ESM2ScoringOutput(scores=sequence_scores)
