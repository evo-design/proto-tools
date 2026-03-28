"""bio_programming_tools/tools/causal_models/evo2/evo2_sample.py

Evo2 sampling tool."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import Field, field_validator, model_validator

from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
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
# Input: Evo2SampleInput
class Evo2SampleInput(BaseToolInput):
    """Input object for Evo2 DNA sequence sampling.

    This class defines the input parameters for generating DNA sequences using
    the Evo2 language model. Evo2 uses prompt sequences to guide autoregressive
    generation of genomic DNA.

    Attributes:
        prompts (list[str]): Prompt sequences for DNA generation.
            Can be provided as:

            - A single prompt string (e.g., ``"GA"``)
            - A list of prompt strings for batch generation (e.g., ``["GA", "GC"]``)

            The model will autoregressively generate DNA continuing from these prompts.
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

# Output: Evo2SampleOutput
class Evo2SampleOutput(BaseToolOutput):
    """Output from Evo2 DNA sequence sampling.

    This class encapsulates the results of Evo2 DNA sequence generation,
    providing generated sequences and optional generation metadata.

    Attributes:
        sequences (list[str]): Generated DNA sequences. Each sequence is a string
            of nucleotides. If ``prepend_prompt=True`` in the config, sequences
            include both the input prompt and newly generated tokens. If ``False``,
            only the newly generated tokens are returned.

        logits (list | None): Per-token logits for each generated sequence.
            Shape: ``[num_sequences, num_generated_tokens, vocab_size]``. Higher
            logit values indicate higher model confidence for that token. Useful for:

            - Analyzing generation uncertainty
            - Computing sequence likelihoods
            - Implementing custom decoding strategies

            Returns ``None`` if logits were not computed or stored.

        kv_caches (list[dict] | None): List of KV cache dictionaries, one per
            sequence. Each cache contains the intermediate attention states from
            generation. Can be passed as ``old_kv_cache`` in a subsequent generation
            call to continue generation from the cached state. Useful for:

            - Continuing generation from a previous run
            - Interactive generation workflows
            - Memory-efficient long sequence generation

            Returns ``None`` if caching was disabled or not used.

    Note:
        Sequences use standard DNA nucleotide characters (A, T, C, G) and may
        include Evo2's special tokens depending on the prompt format.
    """
    sequences: List[str] = Field(description="Generated DNA sequences")
    logits: Optional[List] = Field(
        default=None,
        description="Per-token logits for each sequence",
    )
    kv_caches: Optional[List[Dict]] = Field(
        default=None,
        description="List of KV caches for each sequence",
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
            import json

            # Export minimal sequence data.
            # Logits/KV caches too large for standard export usually.
            data = {
                "sequences": self.sequences
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

# Config:
class Evo2SampleConfig(BaseConfig):
    """Configuration object for Evo2 DNA sequence sampling.

    This class defines all configuration parameters for generating DNA sequences
    using the Evo2 language model. Evo2 is a 7B parameter model trained on genomic
    DNA sequences for autoregressive generation.

    Attributes:
        prepend_prompt (bool): Whether to prepend the input prompt to the generated
            sequence in the output. If ``True``, the returned sequences include both
            the prompt and generated tokens. If ``False``, only the newly generated
            tokens are returned. Default: ``True``.

        model_checkpoint (EVO2_MODEL_CHECKPOINTS): Evo2 model checkpoint to use. Options:

            - ``"evo2_7b"``: 7 billion parameter Evo2 model (default)

            Default: ``"evo2_7b"``.

        local_path (str | None): Optional path to local model weights directory.
            If provided, loads model from local filesystem instead of downloading
            from Hugging Face. Useful for offline inference or custom model versions.
            Default: ``None`` (download from Hugging Face).

        top_k (int): Limits sampling to the top-k most probable tokens at each
            generation step. Lower values make generation more focused and deterministic,
            higher values increase diversity. Must be at least 1. Default: 4.

        top_p (float): Nucleus sampling parameter. Chooses the smallest set of tokens
            whose cumulative probability mass is at least ``top_p``. Common values:

            - ``0.9``: Conservative, high-probability tokens only
            - ``0.95``: Balanced (typical)
            - ``1.0``: No filtering, sample from full distribution (default)

            Range: (0.0, 1.0]. Default: 1.0.

        temperature (float): Scales the randomness of sampling by adjusting the
            sharpness of the probability distribution:

            - ``< 1.0``: Sharper distribution, more deterministic
            - ``1.0``: Standard sampling from model distribution (default)
            - ``> 1.0``: Flatter distribution, more random and diverse

            Must be greater than 0. Default: 1.0.

        num_tokens (int): Number of new tokens to generate per sequence (does not
            include the prompt tokens). For DNA sequences, each token represents
            a nucleotide or short subsequence depending on the tokenizer. Must be
            at least 1. Default: 32.

        cached_generation (bool): Whether to use vortex KV caching for faster
            generation. KV caching stores intermediate attention states to avoid
            recomputation. Recommended for long sequences. Default: ``True``.

        force_prompt_threshold (int | None): Optional number of tokens to
            prefill in parallel before switching to autoregressive prompt forcing.
            This can speed up generation for long prompts. Default: ``None``
            (automatic determination).

        max_seqlen (int | None): Optional maximum sequence length to generate.
            Determines the maximum size of the KV cache. If ``None``, automatically
            calculated as ``prompt_length + num_tokens``. Useful for constraining
            memory usage. Default: ``None``.

        print_generation (bool): Whether to print generated tokens to console in
            real-time as they are generated. Useful for debugging and monitoring
            progress. Default: ``True``.

        stop_at_eos (bool): Whether to stop generation when an end-of-sequence
            (EOS) token is encountered. If ``True``, generation terminates early
            for sequences that naturally end. If ``False``, always generates exactly
            ``num_tokens`` tokens. Default: ``True``.

        old_kv_cache (dict | None): Dictionary of inference parameters containing
            a pre-computed KV cache from a previous generation run. Used for
            continuing generation from a cached state. Default: ``None``.

        batch_size (int): Number of sequences to process simultaneously on GPU.
            Larger batches improve throughput but use more GPU memory; reduce
            if encountering out-of-memory errors. Default: ``1``.

        device (str): Device to run sampling on (``"cuda"``, ``"cpu"``, ``"mps"``).
            Default: ``"cuda"``.

        return_logits (bool): Whether to include per-position logits in the output.
            When ``True``, returns logits for each sequence. When ``False``, only
            returns metrics (saves memory and serialization time). Default: ``False``.

    Note:
        Evo2 is a large model. Smaller batch sizes and shorter sequences
        are recommended if memory is constrained.
    """
    @model_validator(mode="after")
    def _validate_40b(self):
        if "40b" in self.model_checkpoint:
            raise NotImplementedError(
                f"The {self.model_checkpoint} model requires 2 GPUs with tensor "
                "parallelism, which we haven't implemented into our device "
                "manager system. Use a 7b or 1b variant instead."
            )
        return self

    # prompt params
    prepend_prompt: bool = ConfigField(
        title="Prepend Prompt",
        default=True,
        description="Whether to prepend the prompt to the generated sequence in the output",
    )

    # Evo2 model params
    model_checkpoint: EVO2_MODEL_CHECKPOINTS = ConfigField(
        title="Model Checkpoint",
        default="evo2_7b",
        description="Evo2 model checkpoint to use",
        reload_on_change=True,
    )
    local_path: Optional[str] = ConfigField(
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
    top_p: float = ConfigField(
        title="Top P",
        default=1,
        gt=0.0,
        le=1.0,
        description="Chooses the smallest set of tokens whose cumulative probability mass ≥ top-p.",
    )
    temperature: float = ConfigField(
        title="Temperature",
        default=1.0,
        gt=0.0,
        description="Scales the randomness of sampling by adjusting probability distribution sharpness.",
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
    force_prompt_threshold: Optional[int] = ConfigField(
        title="Force Prompt Threshold",
        default=None,
        description="Optional number of tokens to prefill in parallel before switching to prompt forcing.",
        hidden=True,
    )
    max_seqlen: Optional[int] = ConfigField(
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
    old_kv_cache: Optional[Dict] = ConfigField(
        title="Old KV Cache",
        default=None,
        description="Dictionary of inference parameters to use for cached sampling (KV cache)",
        hidden=True,
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Number of sequences to process simultaneously on GPU",
        advanced=True,
    )

    # memory management params
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run on",
        hidden=True,
        include_in_key=False,
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
def example_input():
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
    inputs: Evo2SampleInput, config: Evo2SampleConfig | None = None,
    instance=None,
) -> Evo2SampleOutput:
    """Sample DNA sequences using Evo2 language model.

    Uses the Evo2 7B parameter language model to autoregressively generate
    genomic DNA sequences from prompt sequences. Supports local GPU execution
    with advanced sampling strategies including nucleus sampling and KV caching.

    Args:
        inputs (Evo2SampleInput): Validated input containing one or more DNA
            prompt sequences using Evo2's special prompt format (e.g., ``"+~GA"``).
        config (Evo2SampleConfig | None): Validated Evo2 sampling configuration specifying
            model variant, generation parameters (temperature, top-k, top-p),
            sequence length, and caching options.

    Returns:
        Evo2SampleOutput: Structured output containing:
            - ``sequences``: List of generated DNA sequences
            - ``logits``: Optional per-token logits for each sequence
            - ``kv_caches``: Optional KV caches for continuing generation
            - Metadata about generation parameters and execution mode

    Examples:
        >>> # Basic DNA sequence generation
        >>> inputs = Evo2SampleInput(prompts=["+~GA"])
        >>> config = Evo2SampleConfig(
        ...     num_tokens=1000,
        ...     temperature=0.8,
        ...     top_k=4
        ... )
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
        logits = logits.cpu().tolist()

    # Prepend prompts to generated sequences (vortex generate() returns only newly generated tokens)
    if config.prepend_prompt:
        result["sequences"] = [
            prompt + seq for prompt, seq in zip(inputs.prompts, result["sequences"])
        ]

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
