"""ProGen3 scoring tool."""

import logging
from typing import Any, Literal

from proto_tools.tools.causal_models.shared_data_models import (
    CausalModelScoringConfig,
    CausalModelScoringInput,
    CausalModelScoringMetrics,
    CausalModelScoringOutput,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    ConfigField,
    ToolInstance,
)

logger = logging.getLogger(__name__)

PROGEN3_MODEL_CHECKPOINTS = Literal[
    "progen3-112m",
    "progen3-219m",
    "progen3-339m",
    "progen3-762m",
    "progen3-1b",
    "progen3-3b",
]


# ============================================================================
# Data Models
# ============================================================================
ProGen3ScoringInput = CausalModelScoringInput


# Output:
ProGen3ScoringOutput = CausalModelScoringOutput


# Config:
class ProGen3ScoringConfig(CausalModelScoringConfig):
    """Configuration for ProGen3 protein sequence scoring.

    ProGen3 computes bidirectional autoregressive likelihood by averaging
    forward (N→C) and reverse (C→N) log-likelihoods.

    Attributes:
        model_checkpoint (PROGEN3_MODEL_CHECKPOINTS): ProGen3 weights variant. Sizes range
            from 112M to 3B parameters.
        local_path (str | None): Override HuggingFace download with a local weights directory.
        batch_size (int): Number of sequences to process simultaneously on GPU.
        return_logits (bool): Whether to include forward-pass per-position logits in the
            output. Reverse-pass info is already exposed via ``per_position_metrics``
            (forward/reverse/bidirectional log-likelihoods).

    Note:
        - Lower perplexity indicates higher model confidence.
        - Requires GPU with bfloat16 support (A100/H100 recommended).
    """

    model_checkpoint: PROGEN3_MODEL_CHECKPOINTS = ConfigField(
        title="Model Checkpoint",
        default="progen3-762m",
        description="ProGen3 weights variant",
        reload_on_change=True,
    )
    local_path: str | None = ConfigField(
        title="Local Model Path",
        default=None,
        description="Override the default download with a local weights directory",
        reload_on_change=True,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> ProGen3ScoringInput:
    """Minimal valid input for testing and examples."""
    return ProGen3ScoringInput(sequences=["MKTL"])


@tool(
    key="progen3-score",
    label="ProGen3 Scoring",
    category="causal_models",
    input_class=ProGen3ScoringInput,
    config_class=ProGen3ScoringConfig,
    output_class=ProGen3ScoringOutput,
    metrics_class=CausalModelScoringMetrics,
    description="Score protein sequences using ProGen3 language model",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="scores",
    cacheable=True,
)
def run_progen3_score(
    inputs: ProGen3ScoringInput,
    config: ProGen3ScoringConfig,
    instance: Any = None,
) -> ProGen3ScoringOutput:
    """Score protein sequences using ProGen3 bidirectional language model.

    For each sequence, runs forward (N→C) and reverse (C→N) autoregressive
    passes and averages their log-likelihoods.

    Args:
        inputs (ProGen3ScoringInput): Validated input containing protein sequences to score.
        config (ProGen3ScoringConfig): Scoring configuration specifying model and batch size.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        ProGen3ScoringOutput: A ``CausalModelScoringMetrics`` per sequence with:

            - ``log_likelihood``, ``avg_log_likelihood``, ``perplexity``
            - ``forward_log_likelihood_pp``, ``reverse_log_likelihood_pp``,
              ``log_likelihood_pp``: per-position lists (forward / reverse / bidirectional).
            - ``logits``: forward-pass logits ``(tokenized_len, vocab_size=34)``
              when ``config.return_logits=True``, else ``None``.
            - ``vocab``: 34-token vocab (specials + AA letters); always populated.

    Examples:
        >>> inputs = ProGen3ScoringInput(sequences=["MVLSPADKTN", "MKTLLILAVVAA"])
        >>> config = ProGen3ScoringConfig(model_checkpoint="progen3-762m")
        >>> result = run_progen3_score(inputs, config)
        >>> print(f"Perplexity: {result.scores[0]['perplexity']}")
    """
    logger.debug(f"Using local venv for ProGen3 scoring: {config.model_checkpoint}")

    result = ToolInstance.dispatch(
        "progen3",
        {
            "operation": "score",
            "sequences": inputs.sequences,
            "model_checkpoint": config.model_checkpoint,
            "local_path": config.local_path,
            "device": config.device,
            "verbose": config.verbose,
            "batch_size": config.batch_size,
            "return_logits": config.return_logits,
            "seed": config.seed,
        },
        instance=instance,
        config=config,
    )

    logits_list = result["logits"]
    scores = [
        CausalModelScoringMetrics(
            **metrics,
            **{f"{k}_pp": v for k, v in per_pos.items()},
            logits=logits_list[i] if logits_list is not None else None,
            vocab=result["vocab"],
        )
        for i, (metrics, per_pos) in enumerate(zip(result["metrics"], result["per_position_metrics"], strict=True))
    ]

    return ProGen3ScoringOutput(scores=scores)
