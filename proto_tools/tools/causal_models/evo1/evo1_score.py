"""proto_tools/tools/causal_models/evo1/evo1_score.py.

Evo1 scoring tool.
"""

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

EVO1_MODEL_CHECKPOINTS = Literal[
    "evo-1.5-8k-base",
    "evo-1-8k-base",
    "evo-1-131k-base",
    "evo-1-8k-crispr",
    "evo-1-8k-transposon",
]


# ============================================================================
# Data Models
# ============================================================================
# Input:
Evo1ScoringInput = CausalModelScoringInput

# Output:
Evo1ScoringOutput = CausalModelScoringOutput


# Config:
class Evo1ScoringConfig(CausalModelScoringConfig):
    """Configuration for Evo1 DNA sequence scoring.

    Computes autoregressive likelihood by computing P(x_t | x_{<t}) for each
    position and summing the log probabilities.

    Attributes:
        model_name (EVO1_MODEL_CHECKPOINTS): Evo1 weights variant.
        batch_size (int): Number of sequences to process simultaneously on GPU.
            Larger batches improve throughput but use more GPU memory.
        return_logits (bool): Include per-position logits in the output.

    Note:
        - Evo1 uses byte-level tokenization with vocab_size=512.
        - DNA nucleotides: 'A'=65, 'C'=67, 'G'=71, 'T'=84, 'N'=78 (ASCII values).
    """

    model_name: EVO1_MODEL_CHECKPOINTS = ConfigField(
        title="Model Name",
        default="evo-1-8k-base",
        description="Evo1 weights variant",
        reload_on_change=True,
    )
    timeout: int | None = ConfigField(
        title="Timeout",
        default=1800,
        ge=1,
        description="Maximum execution time in seconds",
        hidden=True,
        include_in_key=False,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return Evo1ScoringInput(sequences=["ATCGATCG"])


@tool(
    key="evo1-score",
    label="Evo1 Scoring",
    category="causal_models",
    input_class=Evo1ScoringInput,
    config_class=Evo1ScoringConfig,
    output_class=Evo1ScoringOutput,
    description="Score DNA sequences using Evo1 language model",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="sequences",
    iterable_output_field="scores",
    cacheable=True,
)
def run_evo1_score(
    inputs: Evo1ScoringInput,
    config: Evo1ScoringConfig,
    instance: Any = None,
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

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        Evo1ScoringOutput: Contains a ``CausalModelScoringMetrics`` for each input
            sequence with:

            - ``log_likelihood``, ``avg_log_likelihood``, ``perplexity`` (access via
              attribute ``score.perplexity`` or mapping ``score["perplexity"]``)
            - ``logits``: Per-position logits tensor (seq_len, vocab_size=512) if
              ``return_logits=True``, otherwise ``None``
            - ``vocab``: List of 512 byte-level tokens if ``return_logits=True``,
              otherwise ``None``

    Examples:
        >>> inputs = Evo1ScoringInput(sequences=["ATCGATCG", "GCTAGCTA"])
        >>> config = Evo1ScoringConfig(model_name="evo-1-8k-base")
        >>> result = run_evo1_score(inputs, config)
        >>> print(f"Perplexity: {result.scores[0]['perplexity']}")

    Note:
        - Lower perplexity indicates higher model confidence in the sequence
        - Set ``return_logits=False`` (default) to save memory when only metrics
          are needed
        - Evo1 uses byte-level tokenization; DNA bases map to their ASCII values
    """
    logger.debug(f"Using local venv for Evo1 scoring: {config.model_name}")

    result = ToolInstance.dispatch(
        "evo1",
        {
            "operation": "score",
            "sequences": inputs.sequences,
            "model_name": config.model_name,
            "device": config.device,
            "verbose": config.verbose,
            "batch_size": config.batch_size,
            "return_logits": config.return_logits,
            "seed": config.seed,
        },
        instance=instance,
        config=config,
    )

    # Serialize tensors to nested lists at tool boundary if needed
    logits = result["logits"]
    if isinstance(logits, list) and logits and hasattr(logits[0], "tolist"):
        logits = [t.cpu().tolist() for t in logits]
    elif hasattr(logits, "tolist"):
        logits = logits.cpu().tolist()

    scores = [
        CausalModelScoringMetrics(
            **metrics,
            logits=logits[i] if logits is not None else None,
            vocab=result["vocab"],
        )
        for i, metrics in enumerate(result["metrics"])
    ]

    return Evo1ScoringOutput(scores=scores)
