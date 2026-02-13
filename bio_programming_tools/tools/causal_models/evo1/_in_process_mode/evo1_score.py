"""
Evo1 DNA sequence scoring (in-process mode).

Computes autoregressive log-likelihood for DNA sequences using the Evo1
model loaded directly into the current Python process. Shares the cached
model instance with ``run_evo1_sample`` via ``get_cached_evo1_model()``.
"""

from __future__ import annotations

import logging
from typing import List, Literal, Optional

from pydantic import Field, field_validator

from bio_programming_tools.tools.causal_models.shared_data_models import (
    CausalModelScoringOutput,
    SequenceScores,
)
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig, ConfigField
from bio_programming_tools.utils.tool_io import BaseToolInput

from .evo1_cache import get_cached_evo1_model

logger = logging.getLogger(__name__)

EVO1_MODEL_NAMES = Literal[
    "evo-1-8k-base",
    "evo-1-131k-base",
    "evo-1-8k-crispr",
    "evo-1-8k-transposon",
]


# ============================================================================
# Data Models
# ============================================================================
# Input:
class Evo1ScoringInput(BaseToolInput):
    """Input for Evo1 DNA sequence scoring.

    Attributes:
        sequences: DNA sequences to score.
    """

    sequences: List[str] = Field(description="DNA sequences to score")

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
Evo1ScoringOutput = CausalModelScoringOutput


# Config:
class Evo1ScoringConfig(BaseConfig):
    """Configuration for Evo1 DNA sequence scoring (in-process mode).

    Computes autoregressive likelihood by computing P(x_t | x_{<t}) for each
    position and summing the log probabilities.

    Attributes:
        model_name (str): Evo1 model checkpoint to use. Default: ``"evo-1-8k-base"``.
        batch_size (Optional[int]): Number of sequences to process per batch.
            If None, processes all sequences at once. Default: ``None``.
        device (str): Device to run the model on. Default: ``"cuda"``.
        keep_on_gpu (bool): Whether to keep the model loaded on device after
            scoring. Default: ``True``.
        return_logits (bool): Whether to include per-position logits in the
            output. Default: ``False``.
        verbose (bool): Whether to print status messages. Default: ``False``.
    """

    model_name: EVO1_MODEL_NAMES = ConfigField(
        title="Model Name",
        default="evo-1-8k-base",
        description="Evo1 model checkpoint to use",
    )
    batch_size: Optional[int] = ConfigField(
        title="Batch Size",
        default=None,
        ge=1,
        description="Max number of sequences on the GPU at once",
        advanced=True,
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        hidden=True,
    )
    keep_on_gpu: bool = ConfigField(
        title="Keep on GPU",
        default=True,
        description="Whether to keep the model on device after scoring",
        hidden=True,
    )
    return_logits: bool = ConfigField(
        title="Return Logits",
        default=False,
        description="Whether to include per-position logits in the output. Disable to save memory.",
        advanced=True,
    )
    verbose: bool = ConfigField(
        title="Verbose",
        default=False,
        description="Whether to print status messages",
        hidden=True,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
@tool(
    key="evo1-score-in-process",
    label="Evo1 Scoring (In-Process)",
    input=Evo1ScoringInput,
    config=Evo1ScoringConfig,
    output=Evo1ScoringOutput,
    description="Score DNA sequences using Evo1 language model (in-process mode)",
)
def run_evo1_score(
    inputs: Evo1ScoringInput, config: Evo1ScoringConfig
) -> Evo1ScoringOutput:
    """Score DNA sequences using Evo1 autoregressive language model.

    Computes the likelihood of DNA sequences using Evo1's autoregressive
    modeling. For each position t, computes log P(x_t | x_{<t}) and sums
    these to get the total log-likelihood.

    Args:
        inputs (Evo1ScoringInput): Validated input containing DNA sequences
            to score.
        config (Evo1ScoringConfig): Scoring configuration specifying model,
            batch size, and whether to return logits.

    Returns:
        Evo1ScoringOutput: Contains SequenceScores for each input sequence with:

            - ``metrics``: Dict with ``log_likelihood``, ``avg_log_likelihood``,
              ``perplexity``
            - ``logits``: Per-position logits tensor (seq_len, vocab_size=512) if
              ``return_logits=True``, otherwise ``None``
            - ``vocab``: List of 512 byte-level tokens if ``return_logits=True``,
              otherwise ``None``

    Examples:
        >>> inputs = Evo1ScoringInput(sequences=["ATCGATCG", "GCTAGCTA"])
        >>> config = Evo1ScoringConfig(model_name="evo-1-8k-base")
        >>> result = run_evo1_score(inputs, config)
        >>> print(f"Perplexity: {result.scores[0].metrics['perplexity']}")
    """
    logger.debug(f"Using local GPU for Evo1 scoring: {config.model_name}")

    model = get_cached_evo1_model(
        model_name=config.model_name,
        device=config.device,
    )

    result = model.score(
        sequences=inputs.sequences,
        batch_size=config.batch_size,
        return_logits=config.return_logits,
        verbose=config.verbose,
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

    return Evo1ScoringOutput(scores=scores)
