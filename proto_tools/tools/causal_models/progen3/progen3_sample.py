"""ProGen3 sampling tool."""

import logging
from typing import Any, Literal

from proto_tools.tools.causal_models.shared_data_models import (
    CausalModelSampleConfig,
    CausalModelSampleInput,
    CausalModelSampleOutput,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    ConfigField,
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
ProGen3SampleInput = CausalModelSampleInput

ProGen3SampleOutput = CausalModelSampleOutput


# Config:
class ProGen3SampleConfig(CausalModelSampleConfig):
    """Configuration for ProGen3 protein sequence sampling.

    ProGen3 is a Mixture-of-Experts protein language model supporting
    forward (N→C) and reverse (C→N) autoregressive generation.

    Attributes:
        batch_size (int): Maximum number of same-length prompts to process
            simultaneously on GPU.
        model_checkpoint (PROGEN3_MODEL_CHECKPOINTS): ProGen3 weights variant. Sizes range
            from 112M (fastest) to 3B (highest quality).
        local_path (str | None): Override HuggingFace download with a local weights directory.
        direction (PROGEN3_DIRECTION): ``"forward"`` generates N→C, ``"reverse"`` generates C→N.
        temperature (float): Softmax temperature; lower values are more deterministic, higher
            values increase diversity.
        top_p (float): Nucleus sampling threshold over per-position token probabilities.
        max_new_tokens (int): Maximum new tokens to generate per prompt (excludes prompt).
        min_new_tokens (int): Minimum new tokens to generate per prompt before stopping is allowed.
        prepend_prompt (bool): If ``True``, returned sequences include the prompt and newly
            generated residues; if ``False``, only the newly generated residues.
    """

    model_checkpoint: PROGEN3_MODEL_CHECKPOINTS = ConfigField(
        default="progen3-762m",
        title="Model Checkpoint",
        description="ProGen3 weights variant",
        reload_on_change=True,
    )
    local_path: str | None = ConfigField(
        default=None,
        title="Local Model Path",
        description="Override the default download with a local weights directory",
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
        description="Softmax temperature for sampling; lower is more deterministic",
    )
    top_p: float = ConfigField(
        default=0.95,
        gt=0.0,
        le=1.0,
        title="Top-p",
        description="Nucleus sampling cutoff over per-position token probabilities",
    )
    batch_size: int = ConfigField(
        default=1,
        ge=1,
        title="Batch Size",
        description="Same-length prompts per GPU forward pass; raise for throughput, lower if OOM",
    )
    max_new_tokens: int = ConfigField(
        default=256,
        ge=1,
        title="Max New Tokens",
        description="Maximum number of new tokens to generate per prompt",
    )
    min_new_tokens: int = ConfigField(
        default=1,
        ge=1,
        title="Min New Tokens",
        description="Minimum number of new tokens to generate per prompt",
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
    stochastic=True,
    example_input=example_input,
    iterable_input_field="prompts",
    iterable_output_field="sequences",
)
def run_progen3_sample(
    inputs: ProGen3SampleInput,
    config: ProGen3SampleConfig,
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
        config (ProGen3SampleConfig): Validated ProGen3 sampling configuration specifying
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
    logger.debug(f"Using local venv for ProGen3 sampling: {config.model_checkpoint}")

    direction_token = _DIRECTION_TOKEN[config.direction]
    prefixed_prompts = [f"{direction_token}{p}" for p in inputs.prompts]

    result = ToolInstance.dispatch(
        "progen3",
        {
            "operation": "sample",
            "prompts": prefixed_prompts,
            "model_checkpoint": config.model_checkpoint,
            "local_path": config.local_path,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_new_tokens,
            "min_new_tokens": config.min_new_tokens,
            "batch_size": config.batch_size,
            "device": config.device,
            "verbose": config.verbose,
            "seed": config.seed,
        },
        instance=instance,
        config=config,
    )

    sequences = result["sequences"]

    if not config.prepend_prompt:
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
            "model_checkpoint": config.model_checkpoint,
            "direction": config.direction,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_new_tokens,
            "prepend_prompt": config.prepend_prompt,
        },
        sequences=sequences,
    )
