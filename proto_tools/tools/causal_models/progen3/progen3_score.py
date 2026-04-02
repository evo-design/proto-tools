"""ProGen3 scoring tool."""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import field_validator

from proto_tools.tools.causal_models.shared_data_models import (
    CausalModelScoringOutput,
    SequenceScores,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    ConfigField,
    InputField,
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
class ProGen3ScoringInput(BaseToolInput):
    """Input for ProGen3 protein sequence scoring.

    Sequences should be provided in N-to-C direction using standard amino
    acid characters. ProGen3 automatically scores in both forward and
    reverse directions and averages the results.

    Attributes:
        sequences (list[str]): Protein sequences to score.
    """

    sequences: list[str] = InputField(description="Protein sequences to score")

    @field_validator("sequences", mode="before")
    @classmethod
    def validate_sequences(cls, v: Any) -> Any:
        """Coerce a single string to a list and validate non-empty."""
        if isinstance(v, str):
            v = [v]
        if not v:
            raise ValueError("sequences must not be empty")
        return v


# Output:
ProGen3ScoringOutput = CausalModelScoringOutput


# Config:
class ProGen3ScoringConfig(BaseConfig):
    """Configuration for ProGen3 protein sequence scoring.

    ProGen3 computes bidirectional autoregressive likelihood by averaging
    forward (N→C) and reverse (C→N) log-likelihoods. This gives a more
    robust score than unidirectional models.

    Attributes:
        model_checkpoint (PROGEN3_MODEL_CHECKPOINTS): ProGen3 model checkpoint to use. Options include
            ``"progen3-112m"`` (112M), ``"progen3-219m"`` (219M),
            ``"progen3-339m"`` (339M), ``"progen3-762m"`` (762M),
            ``"progen3-1b"`` (1B), ``"progen3-3b"`` (3B).
            Default: ``"progen3-762m"``.

        local_path (str | None): Optional path to local model weights directory.
            If provided, loads model from local filesystem instead of
            downloading from HuggingFace. Default: ``None``.

        device (str): Device to run the model on. Default: ``"cuda"``.

        batch_size (int): Number of sequences to process simultaneously on GPU.
            Default: 1.

        reduction (Literal["mean", "sum"]): How to aggregate per-token log-likelihoods.
            ``"mean"`` averages over tokens, ``"sum"`` sums them.
            Default: ``"mean"``.

    Note:
        - ProGen3 uses bidirectional scoring: averages forward + reverse passes
        - Lower perplexity indicates higher model confidence
        - Requires GPU with bfloat16 support (A100/H100 recommended)
    """

    model_checkpoint: PROGEN3_MODEL_CHECKPOINTS = ConfigField(
        title="Model Checkpoint",
        default="progen3-762m",
        description="ProGen3 model checkpoint to use",
        reload_on_change=True,
    )
    local_path: str | None = ConfigField(
        title="Local Model Path",
        default=None,
        description="Path to local model weights",
        hidden=True,
        reload_on_change=True,
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        hidden=True,
        include_in_key=False,
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Number of sequences to process simultaneously on GPU",
        advanced=True,
    )
    reduction: Literal["mean", "sum"] = ConfigField(
        title="Reduction",
        default="mean",
        description="How to aggregate per-token log-likelihoods: 'mean' or 'sum'",
        advanced=True,
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
    description="Score protein sequences using ProGen3 language model",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="sequences",
    iterable_output_field="scores",
    cacheable=True,
)
def run_progen3_score(
    inputs: ProGen3ScoringInput,
    config: ProGen3ScoringConfig | None = None,
    instance: Any = None,
) -> ProGen3ScoringOutput:
    """Score protein sequences using ProGen3 bidirectional language model.

    Computes the likelihood of protein sequences using ProGen3's
    bidirectional scoring. For each sequence, computes the forward (N→C)
    and reverse (C→N) autoregressive log-likelihoods and averages them.

    Args:
        inputs (ProGen3ScoringInput): Validated input containing protein sequences to score.
        config (ProGen3ScoringConfig | None): Scoring configuration specifying model and batch size.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        ProGen3ScoringOutput: Contains SequenceScores for each input sequence with:

            - ``metrics``: Dict with ``log_likelihood``, ``avg_log_likelihood``,
              ``perplexity``

    Examples:
        >>> inputs = ProGen3ScoringInput(sequences=["MVLSPADKTN", "MKTLLILAVVAA"])
        >>> config = ProGen3ScoringConfig(model_checkpoint="progen3-762m")
        >>> result = run_progen3_score(inputs, config)
        >>> print(f"Perplexity: {result.scores[0].metrics['perplexity']}")

    Note:
        - Lower perplexity indicates higher model confidence in the sequence
        - Bidirectional scoring is more robust than unidirectional
    """
    logger.debug(f"Using local venv for ProGen3 scoring: {config.model_checkpoint}")  # type: ignore[union-attr]

    result = ToolInstance.dispatch(
        "progen3",
        {
            "operation": "score",
            "sequences": inputs.sequences,
            "model_checkpoint": config.model_checkpoint,  # type: ignore[union-attr]
            "local_path": config.local_path,  # type: ignore[union-attr]
            "device": config.device,  # type: ignore[union-attr]
            "verbose": config.verbose,  # type: ignore[union-attr]
            "batch_size": config.batch_size,  # type: ignore[union-attr]
            "reduction": config.reduction,  # type: ignore[union-attr]
        },
        instance=instance,
        config=config,
    )

    per_position_list = result.get("per_position_metrics", [None] * len(result["metrics"]))
    scores = [
        SequenceScores(
            metrics=metrics,
            per_position_metrics=per_pos,
        )
        for metrics, per_pos in zip(result["metrics"], per_position_list, strict=True)
    ]

    return ProGen3ScoringOutput(scores=scores)
