"""proto_tools/tools/masked_models/esm3/esm3_score.py.

ESM3 scoring tool.
"""

import logging
from typing import Any, Literal

from proto_tools.tools.masked_models.shared_data_models import (
    MaskedModelInput,
    MaskedModelScoringOutput,
    SequenceScores,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
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
class ESM3ScoringConfig(BaseConfig):
    """Configuration for ESM3 sequence scoring.

    Computes true MLM pseudo-perplexity by masking each position individually and
    computing P(x_i | x_{-i}). Uses batched processing for efficiency.

    Attributes:
        model_checkpoint (Literal[ESM3_MODEL_CHECKPOINTS]): ESM3 model checkpoint to use. Currently available:
            ``"esm3_sm_open_v1"`` (small open-source model).
            Default: ``"esm3_sm_open_v1"``.

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

    model_checkpoint: Literal[ESM3_MODEL_CHECKPOINTS] = ConfigField(
        title="ESM3 Model Checkpoint",
        default="esm3_sm_open_v1",
        description="ESM3 model checkpoint to use",
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
        include_in_key=False,
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
    return ESM3ScoringInput(sequences=["MKTL"])


@tool(
    key="esm3-score",
    label="ESM3 Scoring",
    category="masked_models",
    input_class=ESM3ScoringInput,
    config_class=ESM3ScoringConfig,
    output_class=ESM3ScoringOutput,
    description="Score protein sequences using ESM3 language model",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="sequences",
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
        ESM3ScoringOutput: Contains SequenceScores for each input sequence with:

            - ``metrics``: Dict with ``log_likelihood``, ``avg_log_likelihood``,
              ``perplexity``
            - ``logits``: Per-position logits tensor (seq_len, 20) if
              ``return_logits=True``, otherwise ``None``
            - ``vocab``: List of 20 standard amino acid characters if
              ``return_logits=True``, otherwise ``None``

    Examples:
        >>> # Basic scoring (metrics only, no logits)
        >>> inputs = MaskedModelInput(sequences=["MVLSPADKTNVKAAW", "GSSGSSGSS"])
        >>> config = ESM3ScoringConfig(batch_size=32)
        >>> result = run_esm3_score(inputs, config)
        >>> print(f"Perplexity: {result.scores[0].metrics['perplexity']}")
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
            "seed": config.resolved_seed,
        },
        instance=instance,
        config=config,
    )

    sequence_scores = [
        SequenceScores(
            metrics=metrics,
            logits=result["logits"][i] if result["logits"] is not None else None,
            vocab=result["vocab"],
        )
        for i, metrics in enumerate(result["metrics"])
    ]

    return ESM3ScoringOutput(scores=sequence_scores)
