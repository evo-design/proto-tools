"""bio_programming_tools/tools/causal_models/progen2/progen2_sample.py

ProGen2 sampling tool."""
from __future__ import annotations

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
# Input: ProGen2SampleInput
class ProGen2SampleInput(BaseToolInput):
    """Input object for ProGen2 protein sequence generation.

    This class defines the input parameters for generating protein sequences using
    the ProGen2 autoregressive language model.

    Attributes:
        prompts (list[str]): Prompt sequences for protein generation.
            Can be provided as:

            - A single prompt string (e.g., ``"MKTL"``)
            - A list of prompt strings for batch generation (e.g., ``["MKTL", "MVLS"]``)

            ProGen2 uses special tokens: '1' (start) and '2' (end/stop).
            The start token is automatically prepended by the inference layer
            if not already present.

            The model will autoregressively generate proteins continuing from these prompts.
    """
    prompts: List[str] = InputField(description="Prompt sequences for generation")

    @field_validator("prompts", mode="before")
    @classmethod
    def validate_prompts(cls, v):
        """Coerce a single string to a list and validate non-empty."""
        if isinstance(v, str):
            v = [v]
        if not v:
            raise ValueError("prompts must not be empty")
        return v

# Output: ProGen2SampleOutput
class ProGen2SampleOutput(BaseToolOutput):
    """Output from ProGen2 protein sequence generation.

    This class encapsulates the results of ProGen2 protein sequence generation,
    providing generated sequences and optionally the logits.

    Attributes:
        sequences (list[str]): Generated protein sequences. Each sequence is a string
            of amino acids. Depending on configuration:

            - If ``prepend_prompt=True``: Sequences include both input prompt and
              newly generated tokens
            - If ``prepend_prompt=False``: Only newly generated tokens are returned
            - If ``strip_special_tokens=True``: ProGen2 special tokens ('1', '2')
              are removed
            - If ``truncate_at_stop=True``: Sequences are truncated at stop tokens
        logits (list[list[list[float]]] | None): Per-position logits for each
            generated sequence. Shape is (num_sequences, generated_len, vocab_size).
            Only present if return_logits=True in config.

    Note:
        Sequences use standard amino acid characters. Special tokens ('1' start,
        '2' end) may be present depending on configuration.
    """

    sequences: List[str] = Field(description="Generated protein sequences")
    logits: Optional[List[List[List[float]]]] = Field(
        default=None,
        description="Per-position logits for generated tokens. Only present if return_logits=True.",
    )

    @property
    def output_format_options(self) -> List[str]:
        return ["fasta", "txt", "json"]

    @property
    def output_format_default(self) -> str:
        return "fasta"

    def _export_output(self, export_path: str | Path, file_format: str):
        path = Path(export_path)

        if file_format == "fasta":
            if path.is_dir():
                path = path / "progen2_sequences.fasta"
            with open(path, "w") as f:
                for i, seq in enumerate(self.sequences):
                    f.write(f">seq_{i}\n{seq}\n")

        elif file_format == "txt":
            if path.is_dir():
                path = path / "progen2_sequences.txt"
            with open(path, "w") as f:
                for seq in self.sequences:
                    f.write(f"{seq}\n")

        elif file_format == "json":
            if path.is_dir():
                path = path / "progen2_sequences.json"
            import json
            with open(path, "w") as f:
                json.dump(self.sequences, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

# Config:
class ProGen2SampleConfig(BaseConfig):
    """Configuration object for ProGen2 protein sequence sampling.

    This class defines all configuration parameters for generating protein sequences
    using the ProGen2 autoregressive language model. ProGen2 supports various model
    sizes from 151M to 6B parameters, with specialized variants for antibody sequences
    (OAS) and broader protein families (BFD90).

    Attributes:
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

        temperature (float): Scales the randomness of sampling by adjusting the
            sharpness of the probability distribution:

            - ``< 1.0``: Sharper distribution, more deterministic (recommended for proteins)
            - ``1.0``: Standard sampling from model distribution
            - ``> 1.0``: Flatter distribution, more random and diverse

            Must be greater than 0. Default: 0.2 (following ProGen2 defaults).

        top_p (float): Nucleus sampling parameter. Chooses the smallest set of tokens
            whose cumulative probability mass is at least ``top_p``. Common values:

            - ``0.9``: Conservative, high-probability tokens only
            - ``0.95``: Balanced (default, following ProGen2 recommendations)
            - ``1.0``: No filtering, sample from full distribution

            Range: (0.0, 1.0]. Default: 0.95.

        top_k (int): Limits sampling to the top-k most probable tokens at each
            generation step. Set to 0 to disable (use top_p only).
            Must be >= 0. Default: 0 (disabled).

        max_length (int): Maximum total sequence length including prompt.
            Generation stops when this length is reached or a stop token is encountered.
            Must be at least 1. Default: 256.

        truncate_at_stop (bool): Whether to truncate generated sequences at the
            first stop token ('1' or '2'). If ``True``, returns clean protein
            sequences. Default: ``True``.

        strip_special_tokens (bool): Whether to remove the ProGen2 start and stop
            tokens ('1' or '2') from the output. If ``True``, returns clean amino
            acid sequences. Default: ``True``.

        prepend_prompt (bool): Whether to include the prompt in the returned
            sequence. If ``False``, only newly generated tokens are returned.
            Default: ``True``.

        batch_size (int): Number of sequences to process simultaneously on GPU.
            Larger batches improve throughput but use more GPU memory; reduce
            if encountering out-of-memory errors. Default: ``1``.

        return_logits (bool): Whether to include per-position logits in the output.
            When ``True``, returns logits for each sequence. When ``False``, only
            returns metrics (saves memory and serialization time). Default: ``False``.

        device (str): Device to run the model on (e.g., ``"cuda"``, ``"cpu"``).
            Default: ``"cuda"``.

    Note:
        For detailed information on ProGen2, see:
        - HuggingFace: https://huggingface.co/hugohrban/
        - GitHub: https://github.com/hugohrban/ProGen2-finetuning
        - Original GitHub: https://github.com/enijkamp/progen2
        - Original paper: https://www.cell.com/cell-systems/fulltext/S2405-4712(23)00272-7
    """

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on (e.g., 'cuda', 'cpu')",
        hidden=True,
        include_in_key=False,
    )

    model_checkpoint: PROGEN2_MODEL_CHECKPOINTS = ConfigField(
        default="progen2-large",
        title="Model Checkpoint",
        description="ProGen2 model checkpoint to use",
        reload_on_change=True,
    )
    local_path: Optional[str] = ConfigField(
        default=None,
        title="Local Model Path",
        description="Path to local model weights (if None, downloads from HuggingFace)",
        hidden=True,
        reload_on_change=True,
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
    prepend_prompt: bool = ConfigField(
        default=True,
        title="Prepend Prompt",
        description="Whether to prepend prompt to generation.",
        hidden=True,
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Number of sequences to process simultaneously on GPU",
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
def example_input():
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
    inputs: ProGen2SampleInput, config: ProGen2SampleConfig | None = None,
    instance=None,
) -> ProGen2SampleOutput:
    """Generate protein sequences using ProGen2 autoregressive language model.

    Uses the ProGen2 protein language model to autoregressively generate protein
    sequences from prompt sequences. Supports local GPU execution with various
    sampling strategies.

    Args:
        inputs (ProGen2SampleInput): Validated input containing one or more protein
            prompt sequences. Prompts can include ProGen2's special tokens or raw
            amino acid sequences (which will be automatically normalized).
        config (ProGen2SampleConfig | None): Validated ProGen2 sampling configuration specifying
            model variant, generation parameters (temperature, top-k, top-p),
            sequence length, and output processing options.

    Returns:
        ProGen2SampleOutput: Structured output containing:
            - ``sequences``: List of generated protein sequences
            - Metadata about generation parameters and execution mode

    Examples:
        >>> # Basic protein sequence generation with explicit start token
        >>> inputs = ProGen2SampleInput(prompts=["1MKTL"])
        >>> config = ProGen2SampleConfig(
        ...     max_length=100,
        ...     temperature=0.2,
        ...     top_p=0.95
        ... )
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
        >>> config = ProGen2SampleConfig(
        ...     model_checkpoint="progen2-oas",
        ...     temperature=0.3
        ... )
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
            "device": "cuda",
            "verbose": config.verbose,
            "return_logits": config.return_logits,
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
