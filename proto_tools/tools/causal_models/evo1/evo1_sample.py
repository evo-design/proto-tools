"""proto_tools/tools/causal_models/evo1/evo1_sample.py.

This module provides a standardized interface for sampling DNA sequences
using the Evo1 language model, supporting multiple model checkpoints
including CRISPR and transposon fine-tuned variants.
"""

import logging
from typing import Any, Literal

from pydantic import Field

from proto_tools.tools.causal_models.shared_data_models import (
    CausalModelSampleConfig,
    CausalModelSampleInput,
    CausalModelSampleOutput,
    CausalModelScoringMetrics,
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
Evo1SampleInput = CausalModelSampleInput


# Output:
class Evo1SampleOutput(CausalModelSampleOutput):
    """Output from Evo1 DNA sequence sampling.

    Attributes:
        sequences (list[str]): Generated DNA sequences.
        scores (list[CausalModelScoringMetrics] | None): Scoring metrics per
            sequence, including log_likelihood, avg_log_likelihood, and perplexity.
    """

    scores: list[CausalModelScoringMetrics] | None = Field(
        default=None,
        title="Scores",
        description="Scoring metrics per generated sequence",
    )


# Config:
class Evo1SampleConfig(CausalModelSampleConfig):
    """Configuration for Evo1 DNA sequence sampling.

    Attributes:
        prepend_prompt (bool): Prepend the input prompt to each generated sequence;
            when ``False`` (the default), only newly generated tokens are returned.
        temperature (float): Softmax temperature; lower values are more deterministic.
        top_p (float): Nucleus sampling threshold over per-position token probabilities.
        batch_size (int): Number of prompts to process simultaneously on GPU.
        model_name (EVO1_MODEL_CHECKPOINTS): Evo1 weights variant; ``evo-1-8k-*`` variants
            use an 8,192-token context, ``evo-1-131k-base`` extends to 131,072 tokens, and
            ``-crispr``/``-transposon`` are domain fine-tunes.
        top_k (int): Limit sampling to the top-k most probable tokens at each step.
            Defaults to ``4`` (one per DNA base).
        max_new_tokens (int): Maximum number of new tokens to generate per prompt (excludes prompt).
        cached_generation (bool): Use the KV cache for autoregressive generation.
        force_prompt_threshold (int): Number of tokens to prefill in parallel before
            switching to autoregressive prompt forcing; lower values reduce peak memory.
    """

    prepend_prompt: bool = ConfigField(
        title="Prepend Prompt",
        default=False,
        description="Include the input prompt at the start of each generated sequence",
    )
    model_name: EVO1_MODEL_CHECKPOINTS = ConfigField(
        title="Model Name",
        default="evo-1-8k-base",
        description="Evo1 weights variant",
        reload_on_change=True,
    )
    top_k: int = ConfigField(
        title="Top K",
        default=4,
        ge=1,
        description="Limit sampling to the top-k most probable tokens at each step",
    )
    max_new_tokens: int = ConfigField(
        title="Max New Tokens",
        default=100,
        ge=1,
        description="Maximum newly-generated tokens per prompt (excludes the prompt)",
    )
    cached_generation: bool = ConfigField(
        title="Cached Generation",
        default=True,
        description="Use the KV cache for autoregressive generation",
    )
    force_prompt_threshold: int = ConfigField(
        title="Force Prompt Threshold",
        default=128,
        ge=1,
        description="Tokens to prefill in parallel before switching to prompt forcing",
    )
    timeout: int | None = ConfigField(
        title="Timeout",
        default=1800,
        ge=1,
        description="Maximum execution time in seconds",
        include_in_key=False,
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
    stochastic=True,
    example_input=example_input,
    iterable_input_fields=["prompts"],
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
        >>> config = Evo1SampleConfig(max_new_tokens=100, top_k=4)
        >>> result = run_evo1_sample(inputs, config)
        >>> print(f"Generated {len(result.sequences)} sequences")
    """
    logger.debug(f"Using local venv for Evo1 sampling: {config.model_name}")

    result = ToolInstance.dispatch(
        "evo1",
        {
            "operation": "sample",
            "model_name": config.model_name,
            "prompts": inputs.prompts,
            "max_new_tokens": config.max_new_tokens,
            "top_k": config.top_k,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "batch_size": config.batch_size,
            "device": config.device,
            "verbose": config.verbose,
            "seed": config.seed,
            "cached_generation": config.cached_generation,
            "force_prompt_threshold": config.force_prompt_threshold,
        },
        instance=instance,
        config=config,
    )

    sequences = result["sequences"]
    raw_metrics = result.get("metrics")

    if config.prepend_prompt:
        sequences = [prompt + seq for prompt, seq in zip(inputs.prompts, sequences, strict=False)]

    scores: list[CausalModelScoringMetrics] | None = (
        [CausalModelScoringMetrics(**m) for m in raw_metrics] if raw_metrics else None
    )

    return Evo1SampleOutput(
        metadata={
            "prompts": inputs.prompts,
            "model_name": config.model_name,
            "top_k": config.top_k,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_new_tokens,
            "prepend_prompt": config.prepend_prompt,
        },
        sequences=sequences,
        scores=scores,
    )
