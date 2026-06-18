"""proto_tools/tools/causal_models/evo2/evo2_sample.py.

Evo2 sampling tool.
"""

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

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


class Evo2KVCacheRef(BaseModel):
    """Worker-local Evo2 KV-cache handle returned by a persistent worker."""

    type: Literal["evo2_kv_cache"] = Field(
        default="evo2_kv_cache",
        title="Cache Type",
        description="Evo2 KV-cache handle type",
    )
    cache_id: str = Field(
        title="Cache ID",
        description="Identifier for the worker-local KV cache",
    )


class Evo2SampleOutput(CausalModelSampleOutput):
    """Output from Evo2 DNA sequence sampling.

    Attributes:
        sequences (list[str]): Generated DNA sequences.
        logits (list[Any] | None): Per-position logits for each generated sequence
            (shape: [num_sequences, num_generated_tokens, vocab_size]).
        kv_caches (list[Evo2KVCacheRef] | None): Worker-local cache handles
            for continued generation inside the same persistent worker.
    """

    logits: list[Any] | None = Field(
        default=None,
        title="Logits",
        description="Per-step logits for each generated sequence (shape [n_outputs, gen_len, vocab_size])",
    )
    kv_caches: list[Evo2KVCacheRef] | None = Field(
        default=None,
        title="KV Caches",
        description="Opaque worker-local KV cache handles (only valid on the same worker)",
    )


# Config:
class Evo2SampleConfig(CausalModelSampleConfig):
    """Configuration object for Evo2 DNA sequence sampling.

    Attributes:
        prepend_prompt (bool): Include the input prompt at the start of each generated
            sequence; when ``False``, only newly generated tokens are returned.
        temperature (float): Sampling temperature controlling randomness.
        top_p (float): Nucleus sampling threshold over per-position token probabilities.
        batch_size (int): Number of sequences to process simultaneously.
        model_checkpoint (EVO2_MODEL_CHECKPOINTS): Evo2 weights variant.
        local_path (str | None): Override HuggingFace download with a local weights directory.
        top_k (int): Limit sampling to the top-k most probable tokens at each step.
        max_new_tokens (int): Maximum number of new tokens to generate per prompt (excludes prompt).
        cached_generation (bool): Use the model's per-call KV cache during generation.
        force_prompt_threshold (int | None): Tokens to prefill in parallel before switching
            to autoregressive prompt forcing.
        max_seqlen (int | None): Maximum sequence length the KV cache will be sized for.
        skip_special_tokens (bool): Filter EOS/PAD bytes from the detokenized output.
        stop_at_eos (bool): Stop generation when an EOS (id=0) token is sampled.
        old_kv_cache (Evo2KVCacheRef | None): Worker-local KV cache handle returned by a
            previous persistent-worker generation call.
        return_kv_cache (bool): Return worker-local KV cache handles for continued generation.
        return_logits (bool): Include per-position logits in the output.
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

    @model_validator(mode="after")
    def _validate_kv_cache_settings(self) -> "Evo2SampleConfig":
        if (self.old_kv_cache is not None or self.return_kv_cache) and not self.cached_generation:
            raise ValueError(
                "evo2-sample: old_kv_cache and/or return_kv_cache require cached_generation=True; "
                "set cached_generation=True or remove the KV-cache flag."
            )
        return self

    # Evo2 model params
    model_checkpoint: EVO2_MODEL_CHECKPOINTS = ConfigField(
        title="Model Checkpoint",
        default="evo2_7b",
        description="Evo2 weights variant",
        reload_on_change=True,
    )
    local_path: str | None = ConfigField(
        title="Local Checkpoint Path",
        default=None,
        description="Override the default download with a local weights directory",
        reload_on_change=True,
    )
    timeout: int | None = ConfigField(
        title="Timeout",
        default=1800,
        ge=1,
        description="Maximum execution time in seconds",
        include_in_key=False,
    )
    # Sampling params
    top_k: int = ConfigField(
        title="Top K",
        default=4,
        ge=1,
        description="Limit sampling to the top-k most probable tokens at each step",
    )
    max_new_tokens: int = ConfigField(
        title="Max New Tokens",
        default=32,
        ge=1,
        description="Maximum newly-generated tokens per prompt (excludes the prompt)",
    )
    cached_generation: bool = ConfigField(
        title="Cached Generation",
        default=True,
        description="Use the model's per-call KV cache during generation",
    )
    force_prompt_threshold: int | None = ConfigField(
        title="Force Prompt Threshold",
        default=None,
        description="Tokens to prefill in parallel before switching to prompt forcing",
    )
    max_seqlen: int | None = ConfigField(
        title="Max Sequence Length",
        default=None,
        description="Maximum sequence length the KV cache will be sized for",
    )
    skip_special_tokens: bool = ConfigField(
        title="Skip Special Tokens",
        default=False,
        description="Filter EOS/PAD bytes from the detokenized output",
    )
    stop_at_eos: bool = ConfigField(
        title="Stop at EOS",
        default=True,
        description="Stop generation when an EOS (id=0) token is sampled",
    )
    old_kv_cache: Evo2KVCacheRef | None = ConfigField(
        title="Old KV Cache",
        default=None,
        description="Worker-local KV cache handle to use for continued generation",
    )
    return_kv_cache: bool = ConfigField(
        title="Return KV Cache",
        default=False,
        description="Return worker-local KV cache handles for continued generation",
    )
    return_logits: bool = ConfigField(
        title="Return Logits",
        default=False,
        description="Include per-position logits in the output (large; disable to save memory)",
    )

    def cloud_unsupported_reason(self) -> str | None:
        """A local weights directory (``local_path``) isn't present on a hosted worker."""
        if self.local_path:
            return "local_path points to a local weights directory not available on device='cloud'. Unset it, or run locally with device='cpu'."
        return None


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
    stochastic=True,
    example_input=example_input,
    iterable_input_fields=["prompts"],
    iterable_output_field="sequences",
)
def run_evo2_sample(
    inputs: Evo2SampleInput,
    config: Evo2SampleConfig,
    instance: Any = None,
) -> Evo2SampleOutput:
    """Sample DNA sequences using Evo2 language model.

    Uses the configured Evo2 checkpoint (``evo2_7b`` by default) to
    autoregressively generate genomic DNA sequences from prompt sequences.
    Supports local GPU execution with nucleus sampling and KV caching.

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
            - ``logits``: Optional per-position logits for each sequence
            - ``kv_caches``: Optional worker-local KV cache handles for continuing generation
            - Metadata about generation parameters and execution mode

    Examples:
        >>> # Basic DNA sequence generation
        >>> inputs = Evo2SampleInput(prompts=["+~GA"])
        >>> config = Evo2SampleConfig(max_new_tokens=1000, temperature=0.8, top_k=4)
        >>> result = run_evo2_sample(inputs, config)
        >>> print(f"Generated: {result.sequences[0]}")

    Note:
        - For long sequences, use ``cached_generation=True`` for efficiency
        - Prompts of differing lengths in one batch are processed individually;
          uniform-length prompts share a single batched forward pass

    See Also:
        - Evo2 GitHub Repository: https://github.com/arcinstitute/evo2
        - Evo2 Website: https://arcinstitute.org/tools/evo
    """
    # Local GPU - use standalone venv
    logger.debug(f"Using local venv for Evo2 sampling: {config.model_checkpoint}")

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
            "max_new_tokens": config.max_new_tokens,
            "cached_generation": config.cached_generation,
            "force_prompt_threshold": config.force_prompt_threshold,
            "max_seqlen": config.max_seqlen,
            "skip_special_tokens": config.skip_special_tokens,
            "stop_at_eos": config.stop_at_eos,
            "old_kv_cache": config.old_kv_cache.model_dump() if config.old_kv_cache is not None else None,
            "return_kv_cache": config.return_kv_cache,
            "batch_size": config.batch_size,
            "device": config.device,
            "verbose": config.verbose,
            "return_logits": config.return_logits,
            "seed": config.seed,
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

    # Prepend prompts since generation returns only newly-generated tokens.
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
            "max_new_tokens": config.max_new_tokens,
            "cached_generation": config.cached_generation,
            "prepend_prompt": config.prepend_prompt,
        },
        sequences=result["sequences"],
        kv_caches=result["kv_caches"],
        logits=logits,
    )


def release_evo2_kv_caches(
    kv_caches: list[Evo2KVCacheRef | None] | Evo2KVCacheRef | None,
    instance: Any = None,
) -> None:
    """Release Evo2 KV-cache handles held by the persistent worker."""
    if kv_caches is None:
        return
    cache_payload = (
        [cache.model_dump() if cache is not None else None for cache in kv_caches]
        if isinstance(kv_caches, list)
        else kv_caches.model_dump()
    )
    ToolInstance.dispatch(
        "evo2",
        {
            "operation": "release_kv_caches",
            "kv_caches": cache_payload,
        },
        instance=instance,
    )
