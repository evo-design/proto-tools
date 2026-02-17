"""
Evo1 DNA sequence sampling (in-process mode).

Loads the Evo1 model directly into the current Python process for
interactive use and development. For production/CI use the default
ToolInstance-based tool (key: ``evo1-sample``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, field_validator

from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig, BaseToolInput, BaseToolOutput, ConfigField

from .evo1_cache import get_cached_evo1_model

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

    prompts: List[str] = Field(description="Prompt sequences for generation")

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
    """Configuration for Evo1 DNA sequence sampling (in-process mode).

    Attributes:
        model_name: Evo1 model checkpoint to use.
        top_k: Top-k sampling parameter.
        temperature: Sampling temperature.
        top_p: Top-p (nucleus) sampling parameter.
        num_tokens: Number of tokens to generate.
        prepend_prompt: Whether to prepend prompt to output.
        batch_size: Number of prompts per GPU batch.
        device: Device to run on.
        keep_on_gpu: Whether to keep model loaded after generation.
    """

    model_name: EVO1_MODEL_CHECKPOINTS = ConfigField(
        title="Model Name",
        default="evo-1-8k-base",
        description="Evo1 model checkpoint to use",
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
    batch_size: Optional[int] = ConfigField(
        title="Batch Size",
        default=128,
        ge=1,
        description="Max number of prompts per GPU batch",
        advanced=True,
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run on",
        hidden=True,
    )
    keep_on_gpu: bool = ConfigField(
        title="Keep on GPU",
        default=True,
        description="Whether to keep model on device memory after generation",
        hidden=True,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
@tool(
    key="evo1-sample-in-process",
    label="Evo1 Sampling (In-Process)",
    input=Evo1SampleInput,
    config=Evo1SampleConfig,
    output=Evo1SampleOutput,
    description="Sample DNA sequences using Evo1 language model (in-process mode)",
    uses_gpu=True,
)
def run_evo1_sample(
    inputs: Evo1SampleInput, config: Evo1SampleConfig
) -> Evo1SampleOutput:
    """Sample DNA sequences using the Evo1 language model (in-process).

    Loads the Evo1 model directly into the current process for interactive
    use. Supports model caching via ``keep_on_gpu`` to avoid reloading
    across calls.

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
    model = get_cached_evo1_model(
        model_name=config.model_name,
        device=config.device,
    )

    result = model.sample(
        prompts=inputs.prompts,
        num_tokens=config.num_tokens,
        top_k=config.top_k,
        temperature=config.temperature,
        top_p=config.top_p,
        batch_size=config.batch_size,
        verbose=config.verbose,
    )

    sequences = result["sequences"]
    scores = result.get("scores")

    # Prepend prompts if requested
    if config.prepend_prompt:
        sequences = [
            prompt + seq for prompt, seq in zip(inputs.prompts, sequences)
        ]

    # Unload model if not keeping on GPU
    if not config.keep_on_gpu:
        model.unload()

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
