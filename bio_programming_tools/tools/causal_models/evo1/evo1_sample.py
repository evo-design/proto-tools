"""
Evo1 DNA sequence sampling.

This module provides a standardized interface for sampling DNA sequences
using the Evo1 language model, supporting multiple model checkpoints
including CRISPR and transposon fine-tuned variants.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, field_validator

from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
)
from bio_programming_tools.utils.tool_instance import ToolInstance

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
class Evo1SampleInput(BaseToolInput):
    """Input object for Evo1 DNA sequence sampling.

    Attributes:
        prompts (List[str]): Prompt sequences for DNA generation.
            Can be a single prompt string or a list of prompt strings.
    """

    prompts: List[str] = InputField(description="Prompt sequences for generation")

    @field_validator("prompts", mode="before")
    @classmethod
    def normalize_prompts(cls, v):
        """Convert single string to list of strings."""
        if isinstance(v, str):
            return [v]
        if not v:
            raise ValueError("prompts must not be empty")
        return v


# Output:
class Evo1SampleOutput(BaseToolOutput):
    """Output from Evo1 DNA sequence sampling.

    Attributes:
        sequences (List[str]): Generated DNA sequences.
        scores (Optional[List[float]]): Mean log-probability scores per sequence.
    """

    sequences: List[str] = Field(description="Generated DNA sequences")
    scores: Optional[List[float]] = Field(
        default=None, description="Mean log-probability scores per sequence"
    )

    @property
    def output_format_options(self) -> List[str]:
        return ["fasta", "txt", "json"]

    @property
    def output_format_default(self) -> str:
        return "fasta"

    def _export_output(self, export_path: str | Path, file_format: str):
        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "fasta":
            with open(path, "w") as f:
                for i, seq in enumerate(self.sequences):
                    f.write(f">seq_{i}\n{seq}\n")
        elif file_format == "txt":
            with open(path, "w") as f:
                for seq in self.sequences:
                    f.write(f"{seq}\n")
        elif file_format == "json":
            with open(path, "w") as f:
                json.dump({"sequences": self.sequences}, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# Config:
class Evo1SampleConfig(BaseConfig):
    """Configuration for Evo1 DNA sequence sampling.

    Attributes:
        model_name: Evo1 model checkpoint to use.
        top_k: Top-k sampling parameter.
        temperature: Sampling temperature.
        top_p: Top-p (nucleus) sampling parameter.
        num_tokens: Number of tokens to generate.
        prepend_prompt: Whether to prepend prompt to output.
        batch_size: Number of sequences to process simultaneously on GPU.
            Larger batches improve throughput but use more GPU memory; reduce
            if encountering out-of-memory errors.
        device: Device to run on.
    """

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
        description="Top-k sampling parameter",
    )
    temperature: float = ConfigField(
        title="Temperature",
        default=1.0,
        gt=0.0,
        description="Sampling temperature",
    )
    top_p: float = ConfigField(
        title="Top P",
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Top-p (nucleus) sampling parameter",
        advanced=True,
    )
    num_tokens: int = ConfigField(
        title="Num Tokens",
        default=100,
        ge=1,
        description="Number of tokens to generate",
    )
    prepend_prompt: bool = ConfigField(
        title="Prepend Prompt",
        default=False,
        description="Whether to prepend prompt to generated sequence",
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Number of sequences to process simultaneously on GPU",
        advanced=True,
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run on",
        hidden=True,
        include_in_key=False,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input():
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
    inputs: Evo1SampleInput, config: Evo1SampleConfig | None = None, instance=None,
) -> Evo1SampleOutput:
    """Sample DNA sequences using the Evo1 language model.

    Uses the Evo1 model to autoregressively generate DNA sequences from
    prompt sequences. Supports multiple model checkpoints including
    CRISPR and transposon fine-tuned variants.

    Args:
        inputs (Evo1SampleInput): Validated input containing prompt sequences.
        config (Evo1SampleConfig): Sampling configuration including model
            checkpoint, temperature, and top-k parameters.

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
    scores = result.get("scores")

    # Prepend prompts if requested
    if config.prepend_prompt:
        sequences = [
            prompt + seq for prompt, seq in zip(inputs.prompts, sequences)
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
