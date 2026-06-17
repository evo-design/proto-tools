"""proto_tools/tools/causal_models/evo2/evo2_score.py.

Evo2 scoring tool.
"""

import logging
from typing import Any, Literal

from pydantic import model_validator

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
# Input:
Evo2ScoringInput = CausalModelScoringInput


# Output:
Evo2ScoringOutput = CausalModelScoringOutput


# Config:
class Evo2ScoringConfig(CausalModelScoringConfig):
    """Configuration for Evo2 DNA sequence scoring.

    Computes autoregressive likelihood by computing P(x_t | x_{<t}) for each
    position and summing the log probabilities. Uses batched processing.

    Attributes:
        model_checkpoint (EVO2_MODEL_CHECKPOINTS): Evo2 weights variant.
        local_path (str | None): Override HuggingFace download with a local weights directory.
        prepend_bos (bool): Prepend a beginning-of-sequence token before scoring.
        return_logits (bool): Include per-position logits in the output.

    Note:
        - Evo2 uses byte-level tokenization with vocab_size=512.
        - Evo2 is a large model; batch_size tuning may be needed for memory.
    """

    @model_validator(mode="after")
    def _validate_40b(self) -> Any:
        if "40b" in self.model_checkpoint:
            raise NotImplementedError(
                f"The {self.model_checkpoint} model requires 2 GPUs with tensor "
                "parallelism, which we haven't implemented into our device "
                "manager system. Use a 7b or 1b variant instead."
            )
        return self

    model_checkpoint: EVO2_MODEL_CHECKPOINTS = ConfigField(
        title="Model Checkpoint",
        default="evo2_7b",
        description="Evo2 weights variant",
        reload_on_change=True,
    )
    local_path: str | None = ConfigField(
        title="Local Checkpoint Path",
        default=None,
        description="Override HuggingFace download with a local weights directory",
        reload_on_change=True,
    )
    prepend_bos: bool = ConfigField(
        title="Prepend BOS",
        default=False,
        description="Prepend a beginning-of-sequence token before scoring",
    )
    timeout: int | None = ConfigField(
        title="Timeout",
        default=1800,
        ge=1,
        description="Maximum execution time in seconds",
        include_in_key=False,
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
    return Evo2ScoringInput(sequences=["ATCGATCG"])


@tool(
    key="evo2-score",
    label="Evo2 Scoring",
    category="causal_models",
    input_class=Evo2ScoringInput,
    config_class=Evo2ScoringConfig,
    output_class=Evo2ScoringOutput,
    metrics_class=CausalModelScoringMetrics,
    description="Score DNA sequences using Evo2 language model",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="scores",
    cacheable=True,
)
def run_evo2_score(
    inputs: Evo2ScoringInput,
    config: Evo2ScoringConfig,
    instance: Any = None,
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

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        Evo2ScoringOutput: Contains a ``CausalModelScoringMetrics`` for each input
            sequence with:

            - ``log_likelihood``, ``avg_log_likelihood``, ``perplexity`` (access via
              attribute ``score.perplexity`` or mapping ``score["perplexity"]``)
            - ``logits``: Per-position logits tensor (seq_len, vocab_size=512) if
              ``return_logits=True``, otherwise ``None``
            - ``vocab``: List of 512 byte-level tokens if ``return_logits=True``,
              otherwise ``None``

    Examples:
        >>> # Basic scoring (metrics only, no logits)
        >>> inputs = Evo2ScoringInput(sequences=["ATCGATCG", "GCTAGCTA"])
        >>> config = Evo2ScoringConfig(model_checkpoint="evo2_7b")
        >>> result = run_evo2_score(inputs, config)
        >>> print(f"Perplexity: {result.scores[0]['perplexity']}")
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
    logger.debug(f"Using local venv for Evo2 scoring: {config.model_checkpoint}")

    result = ToolInstance.dispatch(
        "evo2",
        {
            "operation": "score",
            "sequences": inputs.sequences,
            "model_checkpoint": config.model_checkpoint,
            "local_path": config.local_path,
            "device": config.device,
            "verbose": config.verbose,
            "batch_size": config.batch_size,
            "return_logits": config.return_logits,
            "prepend_bos": config.prepend_bos,
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

    return Evo2ScoringOutput(scores=scores)
