"""AbLang scoring tool."""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import field_validator

from proto_tools.tools.masked_models.ablang.ablang_embeddings import (
    ABLANG_MODEL_CHOICES,
    _resolve_model_choice,
)
from proto_tools.tools.masked_models.shared_data_models import (
    MaskedModelConfig,
    MaskedModelScoringOutput,
    SequenceScores,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ConfigField
from proto_tools.utils.tool_instance import ToolInstance
from proto_tools.utils.tool_io import BaseToolInput, InputField

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
class AbLangScoringInput(BaseToolInput):
    """Input for AbLang antibody sequence scoring.

    Attributes:
        sequences (list[str]): Antibody sequence(s) to score. Format depends
            on the model choice:

            - ``ablang1-heavy``: Heavy chain sequences
            - ``ablang1-light``: Light chain sequences
            - ``ablang2-paired``: Paired ``"heavy|light"`` format
    """

    sequences: list[str] = InputField(
        description="Antibody sequence(s) to score. For paired models, use 'heavy|light' format.",
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, v: Any) -> Any:
        """Convert single string to list."""
        if isinstance(v, str):
            return [v]
        if not v:
            raise ValueError("sequences must not be empty")
        return v


# Output:
AbLangScoringOutput = MaskedModelScoringOutput


class AbLangScoringConfig(MaskedModelConfig):
    """Configuration for AbLang antibody sequence scoring.

    Computes pseudo-log-likelihood scores by masking each position individually
    and computing P(x_i | x_{-i}). Uses the AbLang antibody language model
    for antibody-specific scoring.

    Attributes:
        model_choice (ABLANG_MODEL_CHOICES): AbLang model variant to use:

            - ``"auto"``: Automatically select based on input format (default).
              Paired sequences (``"heavy|light"``) use ``ablang2-paired``,
              single-chain sequences use ``ablang1-heavy``.
            - ``"ablang1-heavy"``: Heavy chain only
            - ``"ablang1-light"``: Light chain only
            - ``"ablang2-paired"``: Paired heavy+light chains

        batch_size (int): Number of sequences per forward pass. Default: ``1``.

        device (str): Device to run on. Default: ``"cuda"``.

        scoring_mode (Literal["pseudo_log_likelihood", "confidence"]): Scoring method to use:

            - ``"pseudo_log_likelihood"``: Mask each position individually for
              true MLM scoring (slower but accurate, default)
            - ``"confidence"``: Single forward pass confidence score (faster)
    """

    model_choice: ABLANG_MODEL_CHOICES = ConfigField(
        title="Model Choice",
        default="auto",
        description="Model variant: 'auto', 'ablang1-heavy', 'ablang1-light', or 'ablang2-paired'",
        reload_on_change=True,
    )
    scoring_mode: Literal["pseudo_log_likelihood", "confidence"] = ConfigField(
        title="Scoring Mode",
        default="pseudo_log_likelihood",
        description="Scoring method: 'pseudo_log_likelihood' (accurate, slower) or 'confidence' (fast)",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> AbLangScoringInput:
    """Minimal valid input for testing and examples."""
    return AbLangScoringInput(sequences=["EVQLVESGGGLVQPGG"])


@tool(
    key="ablang-score",
    label="AbLang Scoring",
    category="masked_models",
    input_class=AbLangScoringInput,
    config_class=AbLangScoringConfig,
    output_class=AbLangScoringOutput,
    description="Score antibody sequences using AbLang language model",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="sequences",
    iterable_output_field="scores",
    cacheable=True,
)
def run_ablang_score(
    inputs: AbLangScoringInput,
    config: AbLangScoringConfig,
    instance: Any = None,
) -> AbLangScoringOutput:
    """Score antibody sequences using AbLang language model.

    Computes pseudo-log-likelihood scores for antibody sequences using AbLang's
    masked language modeling objective. Lower perplexity indicates the sequence
    is more consistent with natural antibody repertoires.

    Args:
        inputs (AbLangScoringInput): Validated input containing antibody
            sequences to score.
        config (AbLangScoringConfig): Scoring configuration specifying model
            variant, scoring mode, and device.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        AbLangScoringOutput: Contains SequenceScores for each input sequence with:

            - ``metrics``: Dict with ``pseudo_log_likelihood`` (and ``confidence``
              if using confidence mode)
            - ``logits``: ``None`` (not returned for scoring)
            - ``vocab``: ``None``

    Examples:
        >>> inputs = AbLangScoringInput(sequences=["EVQLVESGGGLVQPGG"])
        >>> config = AbLangScoringConfig(model_choice="ablang1-heavy")
        >>> result = run_ablang_score(inputs, config)
        >>> print(f"PLL: {result.scores[0].metrics['pseudo_log_likelihood']}")

    Note:
        - Lower pseudo-log-likelihood (more negative) indicates lower model confidence
        - The ``confidence`` mode is faster but less accurate than ``pseudo_log_likelihood``
    """
    resolved_model = _resolve_model_choice(config.model_choice, inputs.sequences)
    logger.debug(f"Using local venv for AbLang scoring: {resolved_model}")
    result = ToolInstance.dispatch(
        "ablang",
        {
            "operation": "score",
            "sequences": inputs.sequences,
            "batch_size": config.batch_size,
            "model_choice": resolved_model,
            "device": config.device,
            "verbose": config.verbose,
            "scoring_mode": config.scoring_mode,
        },
        instance=instance,
        config=config,
    )

    sequence_scores = [
        SequenceScores(
            metrics=metrics,
            logits=None,
            vocab=None,
        )
        for metrics in result["metrics"]
    ]

    return AbLangScoringOutput(scores=sequence_scores)
