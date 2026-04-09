"""proto_tools/tools/causal_models/evo1/evo1_sample.py.

This module provides a standardized interface for sampling DNA sequences
using the Evo1 language model, supporting multiple model checkpoints
including CRISPR and transposon fine-tuned variants.
"""

import logging
import math
from typing import Any, Literal

from pydantic import Field

from proto_tools.tools.causal_models.shared_data_models import (
    CausalModelSampleConfig,
    CausalModelSampleInput,
    CausalModelSampleOutput,
    SequenceScores,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    ConfigField,
    ToolInstance,
)

logger = logging.getLogger(__name__)

EVO1_MODEL_CHECKPOINTS = Literal[
    "evo-1-8k-base",
    "evo-1-131k-base",
    "evo-1-8k-crispr",
    "evo-1-8k-transposon",
]


# ============================================================================
# Data Models
# ============================================================================
# Input:
Evo1SampleInput = CausalModelSampleInput


# Output:
class Evo1SampleOutput(CausalModelSampleOutput):
    """Output from Evo1 DNA sequence sampling.

    Attributes:
        sequences (list[str]): Generated DNA sequences.
        scores (list[SequenceScores] | None): Scoring metrics per sequence, including
            log_likelihood, avg_log_likelihood, and perplexity.
    """

    scores: list[SequenceScores] | None = Field(default=None, description="Scoring metrics per generated sequence")


# Config:
class Evo1SampleConfig(CausalModelSampleConfig):
    """Configuration for Evo1 DNA sequence sampling.

    Attributes:
        prepend_prompt (bool): Whether to include the input prompt at the
            start of each generated sequence.
        model_name (EVO1_MODEL_CHECKPOINTS): Evo1 model checkpoint to use.
        top_k (int): Number of top tokens to consider for sampling.
        num_tokens (int): Number of tokens to generate per prompt.

    Note:
        Inherits temperature, top_p, prepend_prompt, batch_size, and device
        from CausalModelSampleConfig. Overrides prepend_prompt default to
        ``False`` (Evo1 does not prepend by default).
    """

    prepend_prompt: bool = ConfigField(
        title="Prepend Prompt",
        default=False,
        description="Whether to prepend the input prompt to the generated sequence",
    )
    model_name: EVO1_MODEL_CHECKPOINTS = ConfigField(
        title="Model Name",
        default="evo-1-8k-base",
        description="Evo1 model checkpoint to use",
        reload_on_change=True,
    )
    top_k: int = ConfigField(
        title="Top K",
        default=4,
        ge=1,
        description="Number of top tokens to consider for sampling",
    )
    num_tokens: int = ConfigField(
        title="Number of Tokens",
        default=100,
        ge=1,
        description="Number of tokens to generate per prompt",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return Evo1SampleInput(prompts=["ATCGATCG"])


@tool(
    key="evo1-sample",
    label="Evo1 Sampling",
    category="causal_models",
    input_class=Evo1SampleInput,
    config_class=Evo1SampleConfig,
    output_class=Evo1SampleOutput,
    description="Sample DNA sequences using Evo1 language model",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="prompts",
    iterable_output_field="sequences",
)
def run_evo1_sample(
    inputs: Evo1SampleInput,
    config: Evo1SampleConfig,
    instance: Any = None,
) -> Evo1SampleOutput:
    """Sample DNA sequences using the Evo1 language model.

    Uses the Evo1 model to autoregressively generate DNA sequences from
    prompt sequences. Supports multiple model checkpoints including
    CRISPR and transposon fine-tuned variants.

    Args:
        inputs (Evo1SampleInput): Validated input containing prompt sequences.
        config (Evo1SampleConfig): Sampling configuration including model
            checkpoint, temperature, and top-k parameters.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        Evo1SampleOutput: Generated DNA sequences with optional scores.

    Examples:
        >>> inputs = Evo1SampleInput(prompts=["ATCGATCG"])
        >>> config = Evo1SampleConfig(num_tokens=100, top_k=4)
        >>> result = run_evo1_sample(inputs, config)
        >>> print(f"Generated {len(result.sequences)} sequences")
    """
    logger.debug(f"Using local venv for Evo1 sampling: {config.model_name}")

    result = ToolInstance.dispatch(
        "evo1",
        {
            "model_name": config.model_name,
            "prompts": inputs.prompts,
            "num_tokens": config.num_tokens,
            "top_k": config.top_k,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "batch_size": config.batch_size,
            "device": config.device,
            "verbose": config.verbose,
        },
        instance=instance,
        config=config,
    )

    sequences = result["sequences"]
    raw_scores: list[float] | None = result.get("scores")

    # Prepend prompts if requested
    if config.prepend_prompt:
        sequences = [prompt + seq for prompt, seq in zip(inputs.prompts, sequences, strict=False)]

    # Derive standard score triplet from mean log-prob per token.
    scores: list[SequenceScores] | None = None
    if raw_scores is not None:
        scores = [
            SequenceScores(
                metrics={
                    "log_likelihood": s * config.num_tokens,
                    "avg_log_likelihood": s,
                    "perplexity": math.exp(-s),
                },
            )
            for s in raw_scores
        ]

    return Evo1SampleOutput(
        metadata={
            "prompts": inputs.prompts,
            "model_name": config.model_name,
            "top_k": config.top_k,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "num_tokens": config.num_tokens,
            "prepend_prompt": config.prepend_prompt,
        },
        sequences=sequences,
        scores=scores,
    )
