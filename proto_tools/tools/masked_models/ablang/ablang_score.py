"""AbLang scoring tool."""

import logging
from typing import Any, Literal

from proto_tools.entities.antibody import Antibody
from proto_tools.tools.masked_models.ablang.ablang_embeddings import _resolve_model_choice
from proto_tools.tools.masked_models.shared_data_models import (
    MaskedModelConfig,
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


class AbLangScoringConfig(MaskedModelConfig):
    """Configuration for AbLang antibody sequence scoring.

    Computes pseudo-log-likelihood scores by masking each position individually
    and computing P(x_i | x_{-i}). The model variant is selected automatically
    based on which chains are provided on each ``Antibody``.

    Attributes:
        batch_size (int): Number of sequences per forward pass. Default: ``1``.
        device (str): Device to run on. Default: ``"cuda"``.
        scoring_mode (Literal["pseudo_log_likelihood", "confidence"]): Scoring method to use:

            - ``"pseudo_log_likelihood"``: Mask each position individually for
              true MLM scoring (slower but accurate, default)
            - ``"confidence"``: Single forward pass confidence score (faster)
    """

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
    return AbLangScoringInput(antibodies=[Antibody(heavy_chain="EVQLVESGGGLVQPGG")])


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
        },
        instance=instance,
        config=config,
    )

    sequence_scores = [MaskedModelScoringMetrics(**metrics) for metrics in result["metrics"]]

    return AbLangScoringOutput(scores=sequence_scores)
