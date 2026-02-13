"""Evo1 scoring tool."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, field_validator

from bio_programming_tools.tools.causal_models.shared_data_models import (
    CausalModelScoringOutput,
    SequenceScores,
)
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig, ConfigField
from bio_programming_tools.utils.env_manager import EnvManager
from bio_programming_tools.utils.tool_io import BaseToolInput

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
    """Configuration for Evo1 DNA sequence scoring.

    Computes autoregressive likelihood by computing P(x_t | x_{<t}) for each
    position and summing the log probabilities.

    Attributes:
        model_name (str): Evo1 model checkpoint to use. Default: ``"evo-1-8k-base"``.
        batch_size (Optional[int]): Number of sequences to process per batch.
            If None, processes all sequences at once. Default: ``None``.
        device (str): Device to run the model on. Default: ``"cuda"``.
        verbose (bool): Whether to print status messages. Default: ``False``.
        return_logits (bool): Whether to include per-position logits in the
            output. Default: ``False``.

    Note:
        - Evo1 uses byte-level tokenization with vocab_size=512
        - DNA nucleotides: 'A'=65, 'C'=67, 'G'=71, 'T'=84, 'N'=78 (ASCII values)
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
    verbose: bool = ConfigField(
        title="Verbose",
        default=False,
        description="Whether to print status messages",
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
    key="evo1-score",
    label="Evo1 Scoring",
    input=Evo1ScoringInput,
    config=Evo1ScoringConfig,
    output=Evo1ScoringOutput,
    description="Score DNA sequences using Evo1 language model",
    uses_gpu=True,
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

    Note:
        - Lower perplexity indicates higher model confidence in the sequence
        - Set ``return_logits=False`` (default) to save memory when only metrics
          are needed
        - Evo1 uses byte-level tokenization; DNA bases map to their ASCII values
    """
    logger.debug(f"Using local venv for Evo1 scoring: {config.model_name}")

    venv_manager = EnvManager("evo1")
    script_path = Path(__file__).parent / "standalone" / "inference.py"
    result = venv_manager.call_standalone_script_in_venv(
        script_path=script_path,
        input_dict={
            "operation": "score",
            "sequences": inputs.sequences,
            "model_name": config.model_name,
            "device": config.device,
            "verbose": config.verbose,
            "batch_size": config.batch_size,
            "return_logits": config.return_logits,
        },
        device=config.device,
        verbose=config.verbose,
    )

    # Serialize tensors to nested lists at tool boundary if needed
    # EnvManager returns pre-serialized lists; this handles edge cases
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
