"""Evo2 scoring tool."""
from __future__ import annotations

import logging
from typing import List, Literal, Optional

from pydantic import field_validator

from bio_programming_tools.tools.causal_models.shared_data_models import (
    CausalModelScoringOutput,
    SequenceScores,
)
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig, ConfigField
from bio_programming_tools.utils.tool_io import BaseToolInput, InputField

from .evo2_cache import get_cached_evo2_model

logger = logging.getLogger(__name__)

EVO2_MODEL_CHECKPOINTS = Literal[
    "evo2_7b",
    "evo2_20b",
    "evo2_40b",
    "evo2_7b_base",
    "evo2_40b_base",
    "evo2_1b_base",
    "evo2_7b_262k",
    "evo2_7b_microviridae",
]

# ============================================================================
# Data Models
# ============================================================================
# Input: Evo2ScoringInput
class Evo2ScoringInput(BaseToolInput):
    """Input for Evo2 DNA sequence scoring.

    Attributes:
        sequences: DNA sequences to score.
    """

    sequences: List[str] = InputField(description="DNA sequences to score")

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, v):
        """Convert single string to list of strings."""
        if isinstance(v, str):
            return [v]
        if not v:
            raise ValueError("sequences must not be empty")
        return v
# Output:
Evo2ScoringOutput = CausalModelScoringOutput

# Config:
class Evo2ScoringConfig(BaseConfig):
    """Configuration for Evo2 DNA sequence scoring.

    Computes autoregressive likelihood by computing P(x_t | x_{<t}) for each
    position and summing the log probabilities. Uses batched processing.

    Attributes:
        model_checkpoint (str): Evo2 model checkpoint to use. Currently available:
            ``"evo2_7b"`` (7B parameters), ``"evo2_40b"`` (40B parameters),
            and base/specialized variants. Default: ``"evo2_7b"``.

        local_path (Optional[str]): Optional path to local model weights directory.
            If provided, loads model from local filesystem instead of downloading.
            Default: ``None``.

        device (str): Device to run the model on. Options include ``"cuda"``,
            ``"cpu"``, or specific GPU devices like ``"cuda:0"``.
            Default: ``"cuda"``.

        batch_size (int): Number of sequences to process simultaneously on GPU.
            Larger batches improve throughput but use more GPU memory; reduce
            if encountering out-of-memory errors. Default: ``1``.

        keep_on_gpu (bool): Whether to keep the model loaded on device after
            inference completes. Set to ``True`` for multiple scoring runs.
            Default: ``False``.

        return_logits (bool): Whether to include per-position logits in the output.
            When ``True``, returns logits for each sequence. When ``False``, only
            returns metrics (saves memory and serialization time). Default: ``False``.

    Note:
        - Evo2 uses byte-level tokenization with vocab_size=512
        - DNA nucleotides: 'A'=65, 'C'=67, 'G'=71, 'T'=84, 'N'=78 (ASCII values)
        - Evo2 is a large model; batch_size tuning may be needed for memory
    """

    model_checkpoint: EVO2_MODEL_CHECKPOINTS = ConfigField(
        title="Model Checkpoint",
        default="evo2_7b",
        description="Evo2 model checkpoint to use",
    )
    local_path: Optional[str] = ConfigField(
        title="Local Checkpoint Path",
        default=None,
        description="Optional path to local model weights",
        hidden=True,
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
    keep_on_gpu: bool = ConfigField(
        title="Keep on GPU",
        default=False,
        description="Whether to keep the model on device after scoring",
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
    key="evo2-score-in-process",
    label="Evo2 Scoring (In-Process)",
    category="causal_models",
    input_class=Evo2ScoringInput,
    config_class=Evo2ScoringConfig,
    output_class=Evo2ScoringOutput,
    description="Score DNA sequences using Evo2 language model (in-process mode)",
    uses_gpu=True,
)
def run_evo2_score(
    inputs: Evo2ScoringInput, config: Evo2ScoringConfig | None = None,
    instance=None,
) -> Evo2ScoringOutput:
    """Score DNA sequences using Evo2 autoregressive language model.

    Computes the likelihood of DNA sequences using Evo2's autoregressive
    modeling. For each position t, computes log P(x_t | x_{<t}) and sums
    these to get the total log-likelihood.

    Args:
        inputs (Evo2ScoringInput): Validated input containing DNA sequences
            to score.
        config (Evo2ScoringConfig): Scoring configuration specifying model,
            batch size, and whether to return logits.

    Returns:
        Evo2ScoringOutput: Contains SequenceScores for each input sequence with:

            - ``metrics``: Dict with ``log_likelihood``, ``avg_log_likelihood``,
              ``perplexity``
            - ``logits``: Per-position logits tensor (seq_len, vocab_size=512) if
              ``return_logits=True``, otherwise ``None``
            - ``vocab``: List of 512 byte-level tokens if ``return_logits=True``,
              otherwise ``None``

    Examples:
        >>> # Basic scoring (metrics only, no logits)
        >>> inputs = Evo2ScoringInput(sequences=["ATCGATCG", "GCTAGCTA"])
        >>> config = Evo2ScoringConfig(model_checkpoint="evo2_7b")
        >>> result = run_evo2_score(inputs, config)
        >>> print(f"Perplexity: {result.scores[0].metrics['perplexity']}")
        >>>
        >>> # Scoring with logits for downstream analysis
        >>> config = Evo2ScoringConfig(return_logits=True)
        >>> result = run_evo2_score(inputs, config)
        >>> print(f"Vocab size: {len(result.scores[0].vocab)}")

    Note:
        - Lower perplexity indicates higher model confidence in the sequence
        - Set ``return_logits=False`` (default) to save memory when only metrics
          are needed
        - Evo2 uses byte-level tokenization; DNA bases map to their ASCII values
    """
    logger.debug(f"Using local GPU for Evo2 scoring: {config.model_checkpoint}")

    model = get_cached_evo2_model(
        model_checkpoint=config.model_checkpoint,
        local_path=config.local_path,
    )

    result = model.score(
        sequences=inputs.sequences,
        device=config.device,
        verbose=config.verbose,
        batch_size=config.batch_size,
        return_logits=config.return_logits,
    )

    if not config.keep_on_gpu:
        model.unload()

    # Serialize tensors to nested lists at tool boundary
    logits = result["logits"]
    if isinstance(logits, list) and logits and hasattr(logits[0], "tolist"):
        logits = [t.cpu().tolist() for t in logits]
    elif hasattr(logits, "tolist"):
        logits = logits.cpu().tolist()

    scores = [
        SequenceScores(
            metrics=metrics,
            logits=logits[i] if logits is not None else None,
            vocab=result["vocab"],
        )
        for i, metrics in enumerate(result["metrics"])
    ]

    return Evo2ScoringOutput(scores=scores)
