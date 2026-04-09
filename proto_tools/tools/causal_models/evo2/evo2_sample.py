"""proto_tools/tools/causal_models/evo2/evo2_sample.py.

Evo2 sampling tool.
"""

import logging
from typing import Any, Literal

from pydantic import Field, model_validator

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

EVO2_MODEL_CHECKPOINTS = Literal[
    "evo2_7b",
    "evo2_20b",
    "evo2_40b",
    "evo2_7b_base",
    "evo2_40b_base",
    "evo2_1b_base",
    "evo2_7b_262k",
    "evo2_7b_microviridae",
]


# ============================================================================
# Data Models
# ============================================================================
Evo2SampleInput = CausalModelSampleInput


class Evo2SampleOutput(CausalModelSampleOutput):
    """Output from Evo2 DNA sequence sampling.

    Attributes:
        sequences (list[str]): Generated DNA sequences.
        logits (list[Any] | None): Per-token logits for each generated sequence
            (shape: [num_sequences, num_generated_tokens, vocab_size]).
        kv_caches (list[dict[str, Any]] | None): KV caches for each sequence,
            enabling continued generation from where sampling left off.
    """

    logits: list[Any] | None = Field(default=None, description="Per-token logits for each generated sequence")
    kv_caches: list[dict[str, Any]] | None = Field(default=None, description="KV caches for continued generation")


# Config:
class Evo2SampleConfig(CausalModelSampleConfig):
    """Configuration object for Evo2 DNA sequence sampling.

    Attributes:
        prepend_prompt (bool): Whether to include the input prompt at the
            start of each generated sequence. Default: ``True``.
        temperature (float): Sampling temperature controlling randomness.
            Default: 1.0.
        top_p (float): Nucleus sampling threshold. Default: 1.0.
        batch_size (int): Number of sequences to process simultaneously.
            Default: 1.
        model_checkpoint (EVO2_MODEL_CHECKPOINTS): Evo2 model checkpoint to use.
            Default: ``"evo2_7b"``.

        local_path (str | None): Optional path to local model weights directory.
            If provided, loads model from local filesystem instead of downloading
            from Hugging Face. Default: ``None``.

        top_k (int): Limits sampling to the top-k most probable tokens at each
            generation step. Default: 4.

        num_tokens (int): Number of new tokens to generate per sequence (does not
            include the prompt tokens). Default: 32.

        cached_generation (bool): Whether to use vortex KV caching for faster
            generation. Default: ``True``.

        force_prompt_threshold (int | None): Optional number of tokens to
            prefill in parallel before switching to autoregressive prompt forcing.
            Default: ``None``.

        max_seqlen (int | None): Optional maximum sequence length to generate.
            Default: ``None``.

        print_generation (bool): Whether to print generated tokens to console in
            real-time as they are generated. Default: ``True``.

        stop_at_eos (bool): Whether to stop generation when an end-of-sequence
            (EOS) token is encountered. Default: ``True``.

        old_kv_cache (dict[str, Any] | None): Dictionary of inference parameters containing
            a pre-computed KV cache from a previous generation run. Default: ``None``.

        return_logits (bool): Whether to include per-token logits in the output.
            Default: ``False``.
    """

    @model_validator(mode="after")
    def _validate_40b(self) -> Any:
        if "40b" in self.model_checkpoint:
            raise NotImplementedError(
                f"The {self.model_checkpoint} model requires 2 GPUs with tensor "
                "parallelism, which we haven't implemented into our device "
                "manager system. Use a 7b or 1b variant instead."
            )
        return self

    # Evo2 model params
    model_checkpoint: EVO2_MODEL_CHECKPOINTS = ConfigField(
        title="Model Checkpoint",
        default="evo2_7b",
        description="Evo2 model checkpoint to use",
        reload_on_change=True,
    )
    local_path: str | None = ConfigField(
        title="Local Checkpoint Path",
        default=None,
        description="Optional path to local model weights",
        hidden=True,
        reload_on_change=True,
    )
    # vortex sampling params
    top_k: int = ConfigField(
        title="Top K",
        default=4,
        ge=1,
        description="Limits sampling to the top-k most probable tokens at each generation step.",
    )
    num_tokens: int = ConfigField(
        title="Num Tokens",
        default=32,
        ge=1,
        description="Number of tokens to generate (Does not include prompt)",
    )
    cached_generation: bool = ConfigField(
        title="Cached Generation",
        default=True,
        description="Whether to use vortex KV caching for generation",
        advanced=True,
    )
    force_prompt_threshold: int | None = ConfigField(
        title="Force Prompt Threshold",
        default=None,
        description="Optional number of tokens to prefill in parallel before switching to prompt forcing.",
        hidden=True,
    )
    max_seqlen: int | None = ConfigField(
        title="Max Sequence Length",
        default=None,
        description="Optional maximum sequence length to generate.",
        advanced=True,
    )
    print_generation: bool = ConfigField(
        title="Print Generation",
        default=True,
        description="Whether to print generation tokens",
        hidden=True,
    )
    stop_at_eos: bool = ConfigField(
        title="Stop at EOS",
        default=True,
        description="Whether to stop at end-of-sequence token",
        advanced=True,
    )
    old_kv_cache: dict[str, Any] | None = ConfigField(
        title="Old KV Cache",
        default=None,
        description="Dictionary of inference parameters to use for cached sampling (KV cache)",
        hidden=True,
    )
    return_logits: bool = ConfigField(
        title="Return Logits",
        default=False,
        description="Whether to include per-token logits in the output. Disable to save memory.",
        advanced=True,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return Evo2SampleInput(prompts=["ATCGATCG"])


@tool(
    key="evo2-sample",
    label="Evo2 Sampling",
    category="causal_models",
    input_class=Evo2SampleInput,
    config_class=Evo2SampleConfig,
    output_class=Evo2SampleOutput,
    description="Sample DNA sequences using Evo2 language model",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="prompts",
    iterable_output_field="sequences",
)
def run_evo2_sample(
    inputs: Evo2SampleInput,
    config: Evo2SampleConfig,
    instance: Any = None,
) -> Evo2SampleOutput:
    """Sample DNA sequences using Evo2 language model.

    Uses the Evo2 7B parameter language model to autoregressively generate
    genomic DNA sequences from prompt sequences. Supports local GPU execution
    with advanced sampling strategies including nucleus sampling and KV caching.

    Args:
        inputs (Evo2SampleInput): Validated input containing one or more DNA
            prompt sequences using Evo2's special prompt format (e.g., ``"+~GA"``).
        config (Evo2SampleConfig): Validated Evo2 sampling configuration specifying
            model variant, generation parameters (temperature, top-k, top-p),
            sequence length, and caching options.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        Evo2SampleOutput: Structured output containing:
            - ``sequences``: List of generated DNA sequences
            - ``logits``: Optional per-token logits for each sequence
            - ``kv_caches``: Optional KV caches for continuing generation
            - Metadata about generation parameters and execution mode

    Examples:
        >>> # Basic DNA sequence generation
        >>> inputs = Evo2SampleInput(prompts=["+~GA"])
        >>> config = Evo2SampleConfig(num_tokens=1000, temperature=0.8, top_k=4)
        >>> result = run_evo2_sample(inputs, config)
        >>> print(f"Generated: {result.sequences[0]}")

    Note:
        - For long sequences, use ``cached_generation=True`` for efficiency
        - Batched generation requires all prompts to have the same length

    See Also:
        - Evo2 GitHub Repository: https://github.com/arcinstitute/evo2
        - Evo2 Website: https://arcinstitute.org/tools/evo
    """
    # Local GPU - use standalone venv
    logger.debug(f"Using local venv for Evo2 sampling: {config.model_checkpoint}")

    # Warn about KV cache limitation in venv mode
    if config.old_kv_cache is not None:
        logger.warning(
            "old_kv_cache provided but standalone venv execution does not support "
            "KV caching. The cache will be ignored."
        )

    result = ToolInstance.dispatch(
        "evo2",
        {
            "operation": "sample",
            "prompts": inputs.prompts,
            "model_checkpoint": config.model_checkpoint,
            "local_path": config.local_path,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "temperature": config.temperature,
            "num_tokens": config.num_tokens,
            "cached_generation": config.cached_generation,
            "force_prompt_threshold": config.force_prompt_threshold,
            "max_seqlen": config.max_seqlen,
            "print_generation": config.print_generation,
            "stop_at_eos": config.stop_at_eos,
            "batch_size": config.batch_size,
            "device": config.device,
            "verbose": config.verbose,
            "return_logits": config.return_logits,
        },
        instance=instance,
        config=config,
    )

    # Serialize tensors to nested lists at tool boundary if needed
    logits = result.get("logits")
    if isinstance(logits, list) and logits and hasattr(logits[0], "tolist"):
        logits = [t.cpu().tolist() for t in logits]
    elif hasattr(logits, "tolist"):
        assert logits is not None
        logits = logits.cpu().tolist()

    # Prepend prompts to generated sequences (vortex generate() returns only newly generated tokens)
    if config.prepend_prompt:
        result["sequences"] = [prompt + seq for prompt, seq in zip(inputs.prompts, result["sequences"], strict=False)]

    return Evo2SampleOutput(
        metadata={
            "prompts": inputs.prompts,
            "model_checkpoint": config.model_checkpoint,
            "local_path": config.local_path,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "temperature": config.temperature,
            "num_tokens": config.num_tokens,
            "cached_generation": config.cached_generation,
            "prepend_prompt": config.prepend_prompt,
        },
        sequences=result["sequences"],
        kv_caches=result["kv_caches"],
        logits=logits,
    )
