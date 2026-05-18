"""AbLang scoring tool."""

import logging
from typing import Any, Literal

from proto_tools.entities.antibody import Antibody
from proto_tools.tools.masked_models.ablang.ablang_embeddings import _resolve_model_choice
from proto_tools.tools.masked_models.shared_data_models import (
    MaskedModelScoringConfig,
    MaskedModelScoringMetrics,
    MaskedModelScoringOutput,
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
        antibodies (list[Antibody]): Antibody sequence(s) to score.
    """

    antibodies: list[Antibody] = InputField(
        description="Antibody sequence(s) to score.",
        min_length=1,
    )


# Output:
AbLangScoringOutput = MaskedModelScoringOutput


class AbLangScoringConfig(MaskedModelScoringConfig):
    """Configuration for AbLang antibody sequence scoring.

    Computes pseudo-log-likelihood scores by masking each position individually
    and computing P(x_i | x_{-i}). The model variant is selected automatically
    based on which chains are provided on each ``Antibody``.

    Attributes:
        batch_size (int): Number of sequences per forward pass.
        device (str): Device to run on.
        return_logits (bool): Include per-position logits in the output (large; disable to
            save memory). Triggers a second ``likelihood``-mode forward pass per batch.
        scoring_mode (Literal["pseudo_log_likelihood", "confidence"]): Scoring method.
            ``"pseudo_log_likelihood"`` masks each position individually (accurate, O(L) passes);
            ``"confidence"`` is a single-pass confidence proxy (faster, less accurate).
    """

    scoring_mode: Literal["pseudo_log_likelihood", "confidence"] = ConfigField(
        title="Scoring Mode",
        default="pseudo_log_likelihood",
        description="Per-position masked PLL (accurate, O(L) passes) vs single-pass confidence proxy",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> AbLangScoringInput:
    """Minimal valid input for testing and examples."""
    return AbLangScoringInput(antibodies=[Antibody(heavy_chain="EVQLVESGGGLVQPGG")])


@tool(
    key="ablang-score",
    label="AbLang Scoring",
    category="masked_models",
    input_class=AbLangScoringInput,
    config_class=AbLangScoringConfig,
    output_class=AbLangScoringOutput,
    metrics_class=MaskedModelScoringMetrics,
    description="Score antibody sequences using AbLang language model",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="antibodies",
    iterable_output_field="scores",
    cacheable=True,
)
def run_ablang_score(
    inputs: AbLangScoringInput,
    config: AbLangScoringConfig,
    instance: Any = None,
) -> AbLangScoringOutput:
    """Score antibody sequences using AbLang language model."""
    sequences = [ab.to_sequence() for ab in inputs.antibodies]
    model_choice = _resolve_model_choice(inputs.antibodies)
    logger.debug("Using local venv for AbLang scoring: %s", model_choice)
    result = ToolInstance.dispatch(
        "ablang",
        {
            "operation": "score",
            "sequences": sequences,
            "batch_size": config.batch_size,
            "model_choice": model_choice,
            "device": config.device,
            "verbose": config.verbose,
            "scoring_mode": config.scoring_mode,
            "return_logits": config.return_logits,
        },
        instance=instance,
        config=config,
    )

    logits_per_seq = result.get("logits")
    vocab = result.get("vocab")
    sequence_scores = [
        MaskedModelScoringMetrics(
            **metrics,
            logits=logits_per_seq[i] if logits_per_seq is not None else None,
            vocab=vocab,
        )
        for i, metrics in enumerate(result["metrics"])
    ]

    return AbLangScoringOutput(scores=sequence_scores)
