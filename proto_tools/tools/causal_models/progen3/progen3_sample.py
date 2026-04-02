"""proto_tools/tools/causal_models/progen3/progen3_sample.py.

ProGen3 sampling tool.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)

logger = logging.getLogger(__name__)

PROGEN3_DIRECTION = Literal["forward", "reverse"]
_DIRECTION_TOKEN = {"forward": "1", "reverse": "2"}

PROGEN3_MODEL_CHECKPOINTS = Literal[
    "progen3-112m",
    "progen3-219m",
    "progen3-339m",
    "progen3-762m",
    "progen3-1b",
    "progen3-3b",
]


# ============================================================================
# Data Models
# ============================================================================
class ProGen3SampleInput(BaseToolInput):
    """Input object for ProGen3 protein sequence generation.

    Attributes:
        prompts (list[str]): Amino acid prompt sequences for protein generation.
            Use ``direction`` in ``ProGen3SampleConfig`` to control generation
            direction. Pass an empty string (``""``) for unconditional generation.
            Can be a single string or a list of strings.
    """

    prompts: list[str] = InputField(description="Prompt sequences for generation")

    @field_validator("prompts", mode="before")
    @classmethod
    def validate_prompts(cls, v: Any) -> Any:
        """Coerce a single string to a list and validate non-empty."""
        if isinstance(v, str):
            v = [v]
        if not v:
            raise ValueError("prompts must not be empty")
        return v


class ProGen3SampleOutput(BaseToolOutput):
    """Output from ProGen3 protein sequence generation.

    Attributes:
        sequences (list[str]): Generated protein sequences. Each sequence contains
            amino acid characters. Special tokens and direction indicators
            are always stripped. If ``include_prompt_in_output=True`` (default),
            sequences include the input prompt residues; if ``False``, only newly
            generated residues are returned.
    """

    sequences: list[str] = Field(description="Generated protein sequences")

    @property
    def output_format_options(self) -> list[str]:
        """Return supported export formats."""
        return ["fasta", "txt", "json"]

    @property
    def output_format_default(self) -> str:
        """Default export format."""
        return "fasta"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        path = Path(export_path)

        if file_format == "fasta":
            path = path / "progen3_sequences.fasta" if path.is_dir() else path.with_suffix(".fasta")
            with open(path, "w") as f:
                f.writelines(f">seq_{i}\n{seq}\n" for i, seq in enumerate(self.sequences))

        elif file_format == "txt":
            path = path / "progen3_sequences.txt" if path.is_dir() else path.with_suffix(".txt")
            with open(path, "w") as f:
                f.writelines(f"{seq}\n" for seq in self.sequences)

        elif file_format == "json":
            path = path / "progen3_sequences.json" if path.is_dir() else path.with_suffix(".json")
            import json

            with open(path, "w") as f:
                json.dump({"sequences": self.sequences}, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# Config:
class ProGen3SampleConfig(BaseConfig):
    """Configuration for ProGen3 protein sequence sampling.

    ProGen3 is a Mixture-of-Experts protein language model supporting
    forward (N→C) and reverse (C→N) autoregressive generation.

    Attributes:
        model_checkpoint (PROGEN3_MODEL_CHECKPOINTS): ProGen3 model checkpoint to use. Options:

            - ``"progen3-112m"``: 112M parameters (fastest)
            - ``"progen3-219m"``: 219M parameters
            - ``"progen3-339m"``: 339M parameters
            - ``"progen3-762m"``: 762M parameters
            - ``"progen3-1b"``: 1B parameters
            - ``"progen3-3b"``: 3B parameters (highest quality, slowest)

            Default: ``"progen3-762m"``.

        local_path (str | None): Optional path to local model weights directory.
            If provided, loads model from local filesystem instead of downloading
            from HuggingFace. Default: ``None``.

        direction (PROGEN3_DIRECTION): Generation direction. ``"forward"``
            generates N→C (left to right), ``"reverse"`` generates C→N
            (right to left). Default: ``"forward"``.

        temperature (float): Scales the randomness of sampling:

            - ``< 1.0``: More deterministic (recommended for proteins)
            - ``1.0``: Standard sampling
            - ``> 1.0``: More random and diverse

            Default: 0.2 (following ProGen3 defaults).

        top_p (float): Nucleus sampling parameter. Smallest set of tokens whose
            cumulative probability mass is at least ``top_p``.
            Default: 0.95.

        max_new_tokens (int): Maximum number of new tokens to generate per prompt.
            Does not include prompt tokens. Default: 256.

        min_new_tokens (int): Minimum number of new tokens to generate per prompt.
            Generation will not stop before this many tokens. Default: 1.

        num_sequences (int): Number of sequences to generate per prompt.
            Default: 1.

        include_prompt_in_output (bool): Whether to include the input prompt
            residues in the output sequence. If ``True`` (default), returned
            sequences include both the prompt and newly generated residues. If
            ``False``, only newly generated residues are returned.

        batch_size (int): Number of sequences to process simultaneously on GPU.
            Default: 1.

        device (str): Device to run the model on. Default: ``"cuda"``.
    """

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        hidden=True,
        include_in_key=False,
    )

    model_checkpoint: PROGEN3_MODEL_CHECKPOINTS = ConfigField(
        default="progen3-762m",
        title="Model Checkpoint",
        description="ProGen3 model checkpoint to use",
        reload_on_change=True,
    )
    local_path: str | None = ConfigField(
        default=None,
        title="Local Model Path",
        description="Path to local model weights (if None, downloads from HuggingFace)",
        hidden=True,
        reload_on_change=True,
    )
    direction: PROGEN3_DIRECTION = ConfigField(
        default="forward",
        title="Direction",
        description="Generation direction: 'forward' (N→C) or 'reverse' (C→N)",
    )
    temperature: float = ConfigField(
        default=0.2,
        gt=0.0,
        title="Temperature",
        description="Scales the randomness of sampling.",
    )
    top_p: float = ConfigField(
        default=0.95,
        gt=0.0,
        le=1.0,
        title="Top-p",
        description="Nucleus sampling parameter.",
    )
    max_new_tokens: int = ConfigField(
        default=256,
        ge=1,
        title="Max New Tokens",
        description="Maximum number of new tokens to generate per prompt.",
    )
    min_new_tokens: int = ConfigField(
        default=1,
        ge=1,
        title="Min New Tokens",
        description="Minimum number of new tokens to generate per prompt.",
        advanced=True,
    )
    num_sequences: int = ConfigField(
        default=1,
        ge=1,
        title="Num Sequences",
        description="Number of sequences to generate per prompt.",
    )
    include_prompt_in_output: bool = ConfigField(
        title="Include Prompt in Output",
        default=True,
        description="Whether to include the input prompt residues in the output sequence",
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Number of sequences to process simultaneously on GPU",
        advanced=True,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> ProGen3SampleInput:
    """Minimal valid input for testing and examples."""
    return ProGen3SampleInput(prompts=["MKTL"])


@tool(
    key="progen3-sample",
    label="ProGen3 Sampling",
    category="causal_models",
    input_class=ProGen3SampleInput,
    config_class=ProGen3SampleConfig,
    output_class=ProGen3SampleOutput,
    description="Sample protein sequences using ProGen3 language model",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="prompts",
    iterable_output_field="sequences",
)
def run_progen3_sample(
    inputs: ProGen3SampleInput,
    config: ProGen3SampleConfig | None = None,
    instance: Any = None,
) -> ProGen3SampleOutput:
    """Generate protein sequences using ProGen3 autoregressive language model.

    Uses the ProGen3 Mixture-of-Experts protein language model to
    autoregressively generate protein sequences from prompt sequences.
    Supports forward (N→C) and reverse (C→N) generation via the
    ``direction`` config parameter.

    Args:
        inputs (ProGen3SampleInput): Validated input containing one or more amino acid
            prompt sequences. Pass an empty string for unconditional generation.
        config (ProGen3SampleConfig | None): Validated ProGen3 sampling configuration specifying
            model variant, direction, generation parameters (temperature, top_p),
            and sequence length.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        ProGen3SampleOutput: Structured output containing:
            - ``sequences``: List of generated protein sequences
            - Metadata about generation parameters

    Examples:
        >>> # Forward (N→C) generation (default)
        >>> inputs = ProGen3SampleInput(prompts=["MKTL"])
        >>> config = ProGen3SampleConfig(max_new_tokens=100, temperature=0.2)
        >>> result = run_progen3_sample(inputs, config)

        >>> # Reverse (C→N) generation
        >>> inputs = ProGen3SampleInput(prompts=["RYTE"])
        >>> config = ProGen3SampleConfig(direction="reverse", max_new_tokens=100)
        >>> result = run_progen3_sample(inputs, config)

        >>> # Unconditional generation
        >>> inputs = ProGen3SampleInput(prompts=[""])
        >>> result = run_progen3_sample(inputs)

    Note:
        - Set ``direction="forward"`` for N→C, ``direction="reverse"`` for C→N
        - ProGen3 requires GPU with bfloat16 support (A100/H100 recommended)
        - Model weights are CC BY-NC-SA 4.0 (non-commercial use only)

    See Also:
        - GitHub: https://github.com/Profluent-AI/progen3
        - HuggingFace: https://huggingface.co/Profluent-Bio
    """
    logger.debug(f"Using local venv for ProGen3 sampling: {config.model_checkpoint}")  # type: ignore[union-attr]

    direction_token = _DIRECTION_TOKEN[config.direction]  # type: ignore[union-attr]
    prefixed_prompts = [f"{direction_token}{p}" for p in inputs.prompts]

    result = ToolInstance.dispatch(
        "progen3",
        {
            "operation": "sample",
            "prompts": prefixed_prompts,
            "model_checkpoint": config.model_checkpoint,  # type: ignore[union-attr]
            "local_path": config.local_path,  # type: ignore[union-attr]
            "temperature": config.temperature,  # type: ignore[union-attr]
            "top_p": config.top_p,  # type: ignore[union-attr]
            "max_new_tokens": config.max_new_tokens,  # type: ignore[union-attr]
            "min_new_tokens": config.min_new_tokens,  # type: ignore[union-attr]
            "num_sequences": config.num_sequences,  # type: ignore[union-attr]
            "batch_size": config.batch_size,  # type: ignore[union-attr]
            "device": config.device,  # type: ignore[union-attr]
            "verbose": config.verbose,  # type: ignore[union-attr]
        },
        instance=instance,
        config=config,
    )

    sequences = result["sequences"]

    if not config.include_prompt_in_output:  # type: ignore[union-attr]
        stripped = []
        for seq, aa_prompt, direction in zip(
            sequences,
            result["prompts"],
            result["directions"],
            strict=True,
        ):
            if aa_prompt and direction == "fwd" and seq.startswith(aa_prompt):
                stripped.append(seq[len(aa_prompt) :])
            elif aa_prompt and direction == "rev" and seq.endswith(aa_prompt):
                stripped.append(seq[: -len(aa_prompt)])
            else:
                stripped.append(seq)
        sequences = stripped

    return ProGen3SampleOutput(
        metadata={
            "model_checkpoint": config.model_checkpoint,  # type: ignore[union-attr]
            "direction": config.direction,  # type: ignore[union-attr]
            "temperature": config.temperature,  # type: ignore[union-attr]
            "top_p": config.top_p,  # type: ignore[union-attr]
            "max_new_tokens": config.max_new_tokens,  # type: ignore[union-attr]
            "num_sequences": config.num_sequences,  # type: ignore[union-attr]
            "include_prompt_in_output": config.include_prompt_in_output,  # type: ignore[union-attr]
        },
        sequences=sequences,
    )
