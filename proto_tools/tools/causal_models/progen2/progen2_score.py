"""proto_tools/tools/causal_models/progen2/progen2_score.py.

ProGen2 scoring tool.
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

PROGEN2_MODEL_CHECKPOINTS = Literal[
    "progen2-small",
    "progen2-medium",
    "progen2-base",
    "progen2-oas",
    "progen2-large",
    "progen2-BFD90",
    "progen2-xlarge",
]


# ============================================================================
# Data Models
# ============================================================================
ProGen2ScoringInput = CausalModelScoringInput


# Output:
ProGen2ScoringOutput = CausalModelScoringOutput


# Config:
class ProGen2ScoringConfig(CausalModelScoringConfig):
    """Configuration for ProGen2 protein sequence scoring.

    Computes autoregressive likelihood ``P(sequence) = prod_t P(x_t | x_{<t})``.

    Attributes:
        model_checkpoint (PROGEN2_MODEL_CHECKPOINTS): ProGen2 weights variant.
        local_path (str | None): Override the default download with a local weights directory.
        batch_size (int): Number of sequences to process simultaneously on GPU.
        return_logits (bool): Include per-position logits in the output.

    Note:
        - Metrics only count amino acid tokens, not special tokens (start, end, pad).
        - The vocab includes 30 tokens (special tokens + amino acids).
    """

    model_checkpoint: PROGEN2_MODEL_CHECKPOINTS = ConfigField(
        title="Model Checkpoint",
        default="progen2-large",
        description="ProGen2 weights variant",
        reload_on_change=True,
    )
    local_path: str | None = ConfigField(
        title="Local Model Path",
        default=None,
        description="Override the default download with a local weights directory",
        reload_on_change=True,
    )

    def cloud_unsupported_reason(self) -> str | None:
        """A local weights directory (``local_path``) isn't present on a hosted worker."""
        if self.local_path:
            return "local_path points to a local weights directory not available on device='cloud'. Unset it, or run locally with device='cpu'."
        return None


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return ProGen2ScoringInput(sequences=["MKTL"])


@tool(
    key="progen2-score",
    label="ProGen2 Scoring",
    category="causal_models",
    input_class=ProGen2ScoringInput,
    config_class=ProGen2ScoringConfig,
    output_class=ProGen2ScoringOutput,
    metrics_class=CausalModelScoringMetrics,
    description="Score protein sequences using ProGen2 language model",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="scores",
    cacheable=True,
)
def run_progen2_score(
    inputs: ProGen2ScoringInput,
    config: ProGen2ScoringConfig,
    instance: Any = None,
) -> ProGen2ScoringOutput:
    """Score protein sequences using ProGen2 autoregressive language model.

    Computes the likelihood of protein sequences using ProGen2's autoregressive
    modeling. For each position t, computes log P(x_t | x_{<t}) and sums
    these to get the total log-likelihood.

    Args:
        inputs (ProGen2ScoringInput): Validated input containing protein sequences
            to score.
        config (ProGen2ScoringConfig): Scoring configuration specifying model,
            batch size, and whether to return logits.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        ProGen2ScoringOutput: Contains a ``CausalModelScoringMetrics`` for each input
            sequence with:

            - ``log_likelihood``, ``avg_log_likelihood``, ``perplexity`` (access via
              attribute ``score.perplexity`` or mapping ``score["perplexity"]``)
            - ``logits``: Per-position logits tensor (seq_len, vocab_size=30) if
              ``return_logits=True``, otherwise ``None``
            - ``vocab``: List of 30 tokens (special + amino acids) giving the
              column order of ``logits`` (always populated)

    Examples:
        >>> # Basic scoring (metrics only, no logits)
        >>> inputs = ProGen2ScoringInput(sequences=["MVLSPADKTN", "MKTLLILAVVAA"])
        >>> config = ProGen2ScoringConfig(model_checkpoint="progen2-large")
        >>> result = run_progen2_score(inputs, config)
        >>> print(f"Perplexity: {result.scores[0]['perplexity']}")
        >>>
        >>> # Scoring with logits for downstream analysis
        >>> config = ProGen2ScoringConfig(return_logits=True)
        >>> result = run_progen2_score(inputs, config)
        >>> print(f"Vocab: {result.scores[0].vocab}")
        >>>
        >>> # Using antibody-specific model
        >>> config = ProGen2ScoringConfig(model_checkpoint="progen2-oas")
        >>> result = run_progen2_score(inputs, config)

    Note:
        - Lower perplexity indicates higher model confidence in the sequence
        - The start token '1' is automatically prepended if not present
        - Set ``return_logits=False`` (default) to save memory when only metrics
          are needed
    """
    logger.debug(f"Using local venv for ProGen2 scoring: {config.model_checkpoint}")
    result = ToolInstance.dispatch(
        "progen2",
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

    logits = result["logits"]

    scores = [
        CausalModelScoringMetrics(
            **metrics,
            logits=logits[i] if logits is not None else None,
            vocab=result["vocab"],
        )
        for i, metrics in enumerate(result["metrics"])
    ]

    return ProGen2ScoringOutput(scores=scores)
