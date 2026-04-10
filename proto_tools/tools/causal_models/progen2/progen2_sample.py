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
    """Configuration object for ProGen2 protein sequence sampling.

    This class defines all configuration parameters for generating protein sequences
    using the ProGen2 autoregressive language model. ProGen2 supports various model
    sizes from 151M to 6B parameters, with specialized variants for antibody sequences
    (OAS) and broader protein families (BFD90).

    Attributes:
        prepend_prompt (bool): Whether to include the input prompt at the
            start of each generated sequence. Default: ``True``.
        batch_size (int): Number of sequences to process simultaneously.
            Default: 1.
        model_checkpoint (PROGEN2_MODEL_CHECKPOINTS): ProGen2 model checkpoint to use. Options:

            - ``"progen2-small"``: 151M parameters (fastest)
            - ``"progen2-medium"``: 754M parameters
            - ``"progen2-oas"``: 754M parameters, trained on OAS antibody sequences
            - ``"progen2-large"``: 2B parameters (default, good balance)
            - ``"progen2-BFD90"``: 2B parameters, trained on BFD90
            - ``"progen2-xlarge"``: 6B parameters (highest quality, slowest)

            Default: ``"progen2-large"``.

        local_path (str | None): Optional path to local model weights directory.
            If provided, loads model from local filesystem instead of downloading
            from HuggingFace. Useful for offline inference or custom model versions.
            Default: ``None`` (download from HuggingFace hugohrban/).

        temperature (float): Sampling temperature. Lower values produce more
            deterministic sequences. Default: 0.2 (following ProGen2 defaults).

        top_p (float): Nucleus sampling threshold. Default: 0.95
            (following ProGen2 recommendations).

        top_k (int): Limits sampling to the top-k most probable tokens at each
            generation step. Set to 0 to disable (use top_p only).
            Default: 0 (disabled).

        max_length (int): Maximum total sequence length including prompt.
            Generation stops when this length is reached or a stop token is encountered.
            Must be at least 1. Default: 256.

        truncate_at_stop (bool): Whether to truncate generated sequences at the
            first stop token ('1' or '2'). If ``True``, returns clean protein
            sequences. Default: ``True``.

        strip_special_tokens (bool): Whether to remove the ProGen2 start and stop
            tokens ('1' or '2') from the output. If ``True``, returns clean amino
            acid sequences. Default: ``True``.

        return_logits (bool): Whether to include per-position logits in the output.
            When ``True``, returns logits for each sequence. When ``False``, only
            returns metrics (saves memory and serialization time). Default: ``False``.

    Note:
        For detailed information on ProGen2, see:
        - HuggingFace: https://huggingface.co/hugohrban/
        - GitHub: https://github.com/hugohrban/ProGen2-finetuning
        - Original GitHub: https://github.com/enijkamp/progen2
        - Original paper: https://www.cell.com/cell-systems/fulltext/S2405-4712(23)00272-7
    """

    temperature: float = ConfigField(
        title="Temperature",
        default=0.2,
        gt=0.0,
        description="Sampling temperature controlling randomness of generation",
    )
    top_p: float = ConfigField(
        title="Top P",
        default=0.95,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling threshold",
        advanced=True,
    )
    model_checkpoint: PROGEN2_MODEL_CHECKPOINTS = ConfigField(
        default="progen2-large",
        title="Model Checkpoint",
        description="ProGen2 model checkpoint to use",
        reload_on_change=True,
    )
    local_path: str | None = ConfigField(
        default=None,
        title="Local Model Path",
        description="Path to local model weights (if None, downloads from HuggingFace)",
        hidden=True,
        reload_on_change=True,
    )
    top_k: int = ConfigField(
        default=0,
        ge=0,
        title="Top-k",
        description="Top-k sampling limit. Set to 0 to disable.",
        advanced=True,
    )
    max_length: int = ConfigField(
        default=256,
        ge=1,
        title="Max Length",
        description="Maximum total sequence length including prompt.",
    )
    truncate_at_stop: bool = ConfigField(
        default=True,
        title="Truncate at Stop",
        description="Whether to truncate sequences at stop tokens.",
        advanced=True,
    )
    strip_special_tokens: bool = ConfigField(
        title="Strip Special Tokens",
        default=True,
        description="Whether to strip start and stop tokens ('1' or '2')",
        advanced=True,
    )
    return_logits: bool = ConfigField(
        title="Return Logits",
        default=False,
        description="Whether to include per-position logits in the output. Disable to save memory.",
        advanced=True,
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
            "truncate_at_stop": config.truncate_at_stop,
            "strip_special_tokens": config.strip_special_tokens,
            "prepend_prompt": config.prepend_prompt,
            "batch_size": config.batch_size,
            "device": config.device,
            "verbose": config.verbose,
            "return_logits": config.return_logits,
            "seed": config.resolved_seed,
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
