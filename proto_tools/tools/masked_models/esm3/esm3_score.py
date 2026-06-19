"""proto_tools/tools/masked_models/esm3/esm3_score.py.

ESM3 scoring tool.
"""

import logging
from typing import Any, Literal

from proto_tools.tools.masked_models.shared_data_models import (
    MaskedModelInput,
    MaskedModelScoringConfig,
    MaskedModelScoringMetrics,
    MaskedModelScoringOutput,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    ConfigField,
    ToolInstance,
    require_hf_token,
)

logger = logging.getLogger(__name__)

ESM3_MODEL_CHECKPOINTS = Literal["esm3_sm_open_v1",]

# ============================================================================
# Data Models
# ============================================================================
# Input:
ESM3ScoringInput = MaskedModelInput
# Output:
ESM3ScoringOutput = MaskedModelScoringOutput


# Config:
class ESM3ScoringConfig(MaskedModelScoringConfig):
    """Configuration for ESM3 sequence scoring.

    Computes true MLM pseudo-perplexity by masking each position individually and
    computing P(x_i | x_{-i}). Uses batched processing for efficiency.

    Attributes:
        model_checkpoint (ESM3_MODEL_CHECKPOINTS): ESM3 weights variant.
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

    model_checkpoint: ESM3_MODEL_CHECKPOINTS = ConfigField(
        title="ESM3 Model Checkpoint",
        default="esm3_sm_open_v1",
        description="ESM3 weights variant",
        reload_on_change=True,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return ESM3ScoringInput(sequences=["MKTL"])


@tool(
    key="esm3-score",
    label="ESM3 Scoring",
    category="masked_models",
    input_class=ESM3ScoringInput,
    config_class=ESM3ScoringConfig,
    output_class=ESM3ScoringOutput,
    metrics_class=MaskedModelScoringMetrics,
    description="Score protein sequences using ESM3 language model",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="scores",
    cacheable=True,
)
def run_esm3_score(
    inputs: ESM3ScoringInput,
    config: ESM3ScoringConfig,
    instance: Any = None,
) -> ESM3ScoringOutput:
    """Score protein sequences using ESM3 language model.

    Computes MLM pseudo-perplexity by masking each position individually and
    computing P(x_i | x_{-i}). Uses batched processing for efficiency.

    Ambiguous amino acids (X, B, Z, etc.) are excluded from the perplexity
    calculation using the industry-standard exclusion strategy. Only positions
    with standard amino acids (20 canonical AAs) contribute to log-likelihood
    and perplexity metrics.

    Args:
        inputs (ESM3ScoringInput): Validated input containing protein sequences
            to score.
        config (ESM3ScoringConfig): Scoring configuration specifying model,
            batch size, and whether to return logits.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        ESM3ScoringOutput: Contains a ``MaskedModelScoringMetrics`` for each input
            sequence with:

            - ``log_likelihood``, ``avg_log_likelihood``, ``perplexity`` (access via
              attribute ``score.perplexity`` or mapping ``score["perplexity"]``)
            - ``logits``: Per-position logits tensor (seq_len, 20) if
              ``return_logits=True``, otherwise ``None``
            - ``vocab``: List of 20 standard amino acid characters giving the
              column order of ``logits`` (always populated)

    Examples:
        >>> # Basic scoring (metrics only, no logits)
        >>> inputs = MaskedModelInput(sequences=["MVLSPADKTNVKAAW", "GSSGSSGSS"])
        >>> config = ESM3ScoringConfig(batch_size=32)
        >>> result = run_esm3_score(inputs, config)
        >>> print(f"Perplexity: {result.scores[0]['perplexity']}")
        >>>
        >>> # Scoring with logits for downstream analysis
        >>> config = ESM3ScoringConfig(batch_size=32, return_logits=True)
        >>> result = run_esm3_score(inputs, config)
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
    require_hf_token("ESM3", "https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1")

    logger.debug(f"Using local for ESM3 scoring: {config.model_checkpoint}")
    result = ToolInstance.dispatch(
        "esm3",
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

    return ESM3ScoringOutput(scores=sequence_scores)
