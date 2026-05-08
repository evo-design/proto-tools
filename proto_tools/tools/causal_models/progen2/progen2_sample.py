"""proto_tools/tools/causal_models/progen2/progen2_sample.py.

ProGen2 sampling tool.
"""

import logging
from typing import Any, Literal

from pydantic import Field

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

PROGEN2_MODEL_CHECKPOINTS = Literal[
    "progen2-small",
    "progen2-medium",
    "progen2-base",
    "progen2-oas",
    "progen2-large",
    "progen2-BFD90",
    "progen2-xlarge",
]


# ============================================================================
# Data Models
# ============================================================================
ProGen2SampleInput = CausalModelSampleInput


class ProGen2SampleOutput(CausalModelSampleOutput):
    """Output from ProGen2 protein sequence generation.

    Attributes:
        sequences (list[str]): Generated protein sequences.
        logits (list[list[list[float]]] | None): Per-position logits for each
            generated sequence (shape: [num_sequences, generated_len, vocab_size]).
    """

    logits: list[list[list[float]]] | None = Field(
        default=None, description="Per-position logits for each generated sequence"
    )


# Config:
class ProGen2SampleConfig(CausalModelSampleConfig):
    """Configuration for ProGen2 protein sequence sampling.

    ProGen2 is an autoregressive protein language model with sizes from 151M to 6B
    parameters and specialized variants for antibody (OAS) and broader protein
    families (BFD90).

    Attributes:
        prepend_prompt (bool): Include the input prompt at the start of each generated
            sequence; when ``False``, only newly generated tokens are returned.
        batch_size (int): Number of prompts to process simultaneously on GPU.
        model_checkpoint (PROGEN2_MODEL_CHECKPOINTS): ProGen2 weights variant.
            Sizes range from 151M (small) to 6B (xlarge).
        local_path (str | None): Override the default download with a local weights directory.
        temperature (float): Softmax temperature; lower values are more deterministic.
        top_p (float): Nucleus sampling threshold over per-position token probabilities.
        top_k (int): Top-k truncation; ``0`` disables and uses top-p only.
        max_length (int): Maximum total sequence length including prompt.
        truncate_at_stop (bool): Truncate generated sequences at the first stop token.
        strip_special_tokens (bool): Strip ProGen2 start/stop sentinel tokens (``1``/``2``)
            from output.
        return_logits (bool): Include per-position logits in the output.
        num_samples (int): Independent samples drawn per prompt; raise for diversity.
    """

    temperature: float = ConfigField(
        title="Temperature",
        default=0.2,
        gt=0.0,
        description="Softmax temperature for sampling; lower is more deterministic",
    )
    top_p: float = ConfigField(
        title="Top P",
        default=0.95,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling threshold over per-position token probabilities",
        advanced=True,
    )
    model_checkpoint: PROGEN2_MODEL_CHECKPOINTS = ConfigField(
        default="progen2-large",
        title="Model Checkpoint",
        description="ProGen2 weights variant",
        reload_on_change=True,
    )
    local_path: str | None = ConfigField(
        default=None,
        title="Local Model Path",
        description="Override the default download with a local weights directory",
        hidden=True,
        reload_on_change=True,
    )
    top_k: int = ConfigField(
        default=0,
        ge=0,
        title="Top-k",
        description="Top-k truncation; 0 disables and uses top-p only",
        advanced=True,
    )
    max_length: int = ConfigField(
        default=256,
        ge=1,
        title="Max Length",
        description="Maximum total sequence length including prompt",
    )
    truncate_at_stop: bool = ConfigField(
        default=True,
        title="Truncate at Stop",
        description="Truncate generated sequences at the first stop token",
        advanced=True,
    )
    strip_special_tokens: bool = ConfigField(
        title="Strip Special Tokens",
        default=True,
        description="Strip ProGen2 start/stop sentinel tokens from output",
        advanced=True,
    )
    return_logits: bool = ConfigField(
        title="Return Logits",
        default=False,
        description="Include per-position logits in the output (large; disable to save memory)",
        advanced=True,
    )
    num_samples: int = ConfigField(
        title="Samples Per Prompt",
        default=1,
        ge=1,
        description="Independent samples drawn per prompt; raise for diversity",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return ProGen2SampleInput(prompts=["MKTL"])


@tool(
    key="progen2-sample",
    label="ProGen2 Sampling",
    category="causal_models",
    input_class=ProGen2SampleInput,
    config_class=ProGen2SampleConfig,
    output_class=ProGen2SampleOutput,
    description="Sample protein sequences using ProGen2 language model",
    uses_gpu=True,
    generative=True,
    example_input=example_input,
    iterable_input_field="prompts",
    iterable_output_field="sequences",
)
def run_progen2_sample(
    inputs: ProGen2SampleInput,
    config: ProGen2SampleConfig,
    instance: Any = None,
) -> ProGen2SampleOutput:
    """Generate protein sequences using ProGen2 autoregressive language model.

    Uses the ProGen2 protein language model to autoregressively generate protein
    sequences from prompt sequences. Supports local GPU execution with various
    sampling strategies.

    Args:
        inputs (ProGen2SampleInput): Validated input containing one or more protein
            prompt sequences. Prompts can include ProGen2's special tokens or raw
            amino acid sequences (which will be automatically normalized).
        config (ProGen2SampleConfig): Validated ProGen2 sampling configuration specifying
            model variant, generation parameters (temperature, top-k, top-p),
            sequence length, and output processing options.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        ProGen2SampleOutput: Structured output containing:
            - ``sequences``: List of generated protein sequences
            - Metadata about generation parameters and execution mode

    Examples:
        >>> # Basic protein sequence generation with explicit start token
        >>> inputs = ProGen2SampleInput(prompts=["1MKTL"])
        >>> config = ProGen2SampleConfig(max_length=100, temperature=0.2, top_p=0.95)
        >>> result = run_progen2_sample(inputs, config)
        >>> print(f"Generated: {result.sequences[0]}")

        >>> # Generate from raw amino acids (auto-normalized)
        >>> inputs = ProGen2SampleInput(prompts=["MVLS"])  # Will prepend '1'
        >>> result = run_progen2_sample(inputs, config)

        >>> # Batch generation
        >>> inputs = ProGen2SampleInput(prompts=["1MKTL", "1MVLS", "1GSSGSSG"])
        >>> result = run_progen2_sample(inputs, config)
        >>> print(f"Generated {len(result.sequences)} sequences")

        >>> # Using antibody-specific model
        >>> config = ProGen2SampleConfig(model_checkpoint="progen2-oas", temperature=0.3)
        >>> result = run_progen2_sample(inputs, config)

    Note:
        - ProGen2 uses '1' as start token and '2' as stop token
        - Raw amino acid sequences are automatically normalized (start token prepended)
        - Local execution runs inside a standalone venv via ToolInstance

    See Also:
        - HuggingFace: https://huggingface.co/hugohrban/
        - ProGen2-finetuning GitHub: https://github.com/hugohrban/ProGen2-finetuning
        - Original ProGen2 GitHub: https://github.com/enijkamp/progen2
    """
    logger.debug(f"Using local venv for ProGen2 sampling: {config.model_checkpoint}")
    result = ToolInstance.dispatch(
        "progen2",
        {
            "operation": "sample",
            "prompts": inputs.prompts,
            "model_checkpoint": config.model_checkpoint,
            "local_path": config.local_path,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "max_length": config.max_length,
            "num_samples": config.num_samples,
            "truncate_at_stop": config.truncate_at_stop,
            "strip_special_tokens": config.strip_special_tokens,
            "prepend_prompt": config.prepend_prompt,
            "batch_size": config.batch_size,
            "device": config.device,
            "verbose": config.verbose,
            "return_logits": config.return_logits,
            "seed": config.seed,
        },
        instance=instance,
        config=config,
    )

    return ProGen2SampleOutput(
        metadata={
            "model_checkpoint": config.model_checkpoint,
            "temperature": config.temperature,
            "max_length": config.max_length,
        },
        sequences=result["sequences"],
        logits=result.get("logits"),
    )
