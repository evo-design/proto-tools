"""ESM2 sampling tool."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field

from bio_programming_tools.utils.env_manager import EnvManager
from bio_programming_tools.utils.tool_io import BaseToolOutput
from bio_programming_tools.tools.masked_models.shared_data_models import (
    MaskedModelInput,
)
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig, ConfigField, use_cloud_gpu

from .standalone.inference import ESM2_MODEL_CHECKPOINTS

logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================
# Input:
ESM2SampleInput = MaskedModelInput

# Output:
class ESM2SampleOutput(BaseToolOutput):
    """Output from ESM2 protein sequence sampling.

    This class encapsulates the results of ESM2 sequence generation or mutation,
    providing the sampled protein sequences and optionally the logits.

    Attributes:
        sequences (List[str]): Sampled or mutated protein sequences. Each sequence
            is a string of amino acid characters. For de novo generation, these are
            completely new sequences. For mutation, these are modified versions of
            the input sequences with specified positions changed to model-predicted
            alternatives.
        logits (Optional[List[List[List[float]]]]): Per-position logits for each
            sequence. Shape is (num_sequences, seq_len, vocab_size=20). Only present
            if return_logits=True in config.
    """
    sequences: List[str] = Field(
        description="Sampled/mutated protein sequences"
    )
    logits: Optional[List[List[List[float]]]] = Field(
        default=None,
        description="Per-position amino acid logits. Shape: [num_sequences, seq_len, 20].",
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
            with open(path, "w") as f:
                json.dump(self.sequences, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

# Config:
class ESM2SampleConfig(BaseConfig):
    """Configuration object for ESM2 protein sequence sampling.

    This class defines configuration parameters for sampling/generating protein
    sequences using ESM2's learned sequence distributions. Supports both de novo
    generation and guided mutation of existing sequences.

    Attributes:
        model_checkpoint (str): Name of the ESM2 model variant to use. See
            ``ESM2EmbeddingsConfig`` for available options. Larger models produce more
            realistic sequences but are slower. Default: ``"esm2_t33_650M_UR50D"``.

        temperature (float): Sampling temperature controlling randomness in amino
            acid selection:

            - ``< 1.0``: More conservative, higher probability choices (deterministic)
            - ``1.0``: Standard sampling from the model distribution (default)
            - ``> 1.0``: More diverse, explores lower probability choices (creative)

            Lower temperatures produce sequences closer to natural proteins, while
            higher temperatures increase diversity. Default: 1.0.

        decoding_method (str): Method for selecting which positions to mutate:

            - ``"entropy"``: Mutate positions with highest prediction uncertainty (default)
            - ``"max_logit"``: Mutate positions with lowest confidence predictions
            - ``"random"``: Randomly select positions to mutate

            ``"entropy"`` typically produces the most natural-looking sequences.
            Default: ``"entropy"``.

        num_mutations (int): Number of positions to mutate per sequence in each
            iteration. Higher values produce more divergent sequences but may reduce
            biological plausibility. Default: 1.

        batch_size (Optional[int]): Number of sequences to process per batch during inference.
            If None, processes all sequences at once. Larger batches are faster but use more GPU memory.
            Reduce if encountering out-of-memory errors. Default: ``None``.

        device (str): Device to run sampling on (``"cuda"``, ``"cpu"``, ``"mps"``).
            Default: ``"cuda"``.

        verbose (bool): Whether to print progress messages during sampling.
            Default: ``False``.

        return_logits (bool): Whether to include per-position logits in the output.
            When ``True``, returns logits for each sequence. When ``False``, only
            returns metrics (saves memory and serialization time). Default: ``False``.
    """
    model_checkpoint: Literal[ESM2_MODEL_CHECKPOINTS] = ConfigField(
        title="ESM2 Model Checkpoint",
        default="esm2_t33_650M_UR50D",
        description="Name of the ESM2 model variant to use",
    )
    temperature: float = ConfigField(
        title="Sampling Temperature",
        default=1.0,
        description="Sampling temperature for amino acid selection",
        advanced=True,
    )
    decoding_method: Literal["entropy", "max_logit", "random"] = ConfigField(
        title="Decoding Method",
        default="entropy",
        description="Method for selecting positions to mutate",
        advanced=True,
    )
    num_mutations: int = ConfigField(
        title="Number of Mutations",
        default=1,
        description="Number of positions to mutate per sequence",
        ge=1,
        advanced=True,
    )
    batch_size: Optional[int] = ConfigField(
        title="Batch Size",
        default=None,
        description="Number of sequences to process per batch. If None, processes all at once.",
        advanced=True,
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run on",
        hidden=True,
    )
    verbose: bool = ConfigField(
        title="Verbose",
        default=False,
        description="Whether to print status messages",
        hidden=True,
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
@tool(
    key="esm2-sample",
    label="ESM2 Sampling",
    input=ESM2SampleInput,
    config=ESM2SampleConfig,
    output=ESM2SampleOutput,
    description="Sample protein sequences using ESM2 language model",
)
def run_esm2_sample(
    inputs: ESM2SampleInput, config: ESM2SampleConfig
) -> ESM2SampleOutput:
    """Sample or mutate protein sequences using ESM2 language model.

    Uses ESM2's learned sequence distributions to generate new protein sequences
    or intelligently mutate existing ones. Supports multiple decoding strategies
    for controlling the sampling process.

    Args:
        inputs (MaskedModelInput): Validated input containing:
            - Empty list or placeholder sequences for de novo generation
            - Existing protein sequences for guided mutation
        config (ESM2SampleConfig): Validated ESM2 sampling configuration specifying
            model variant, sequence length, temperature, and mutation parameters.

    Returns:
        ESM2SampleOutput: Structured output containing:
            - ``sequences``: List of sampled/mutated protein sequences
            - Metadata about sampling parameters

    Examples:
        >>> # Mutate existing sequences
        >>> inputs = MaskedModelInput(
        ...     sequences=["MVLSPADKTNVKAAW"]
        ... )
        >>> config = ESM2SampleConfig(
        ...     temperature=1.0,
        ...     decoding_method="entropy",
        ...     num_mutations=3,
        ...     verbose=True
        ... )
        >>> result = run_esm2_sample(inputs, config)
        >>> print(f"Original: {inputs.sequences[0]}")
        >>> print(f"Mutated:  {result.sequences[0]}")
        >>>
        >>> # Generate diverse variants with higher temperature
        >>> config = ESM2SampleConfig(
        ...     temperature=1.5,  # More diverse
        ...     num_mutations=5
        ... )
        >>> result = run_esm2_sample(inputs, config)
        >>>
        >>> # Conservative mutations with entropy-based selection
        >>> config = ESM2SampleConfig(
        ...     temperature=0.7,  # More conservative
        ...     decoding_method="entropy",
        ...     num_mutations=1  # Single mutation
        ... )
        >>> result = run_esm2_sample(inputs, config)

    See Also:
        - ESM2 GitHub Repository: https://github.com/facebookresearch/esm

    Note:
        - For protein design workflows, start with low temperatures and few mutations
        - the cloud runtime GPU execution is automatically used when configured via environment
    """

    if use_cloud_gpu():
        logger.debug(f"Using the cloud runtime for ESM2 sampling: {config.model_checkpoint}")
        import _gpu_runtime

        ESM2Service = _gpu_runtime.Cls.from_name("bio-programming", "ESM2Service")
        result = ESM2Service().sample.remote(
            sequences=inputs.sequences,
            temperature=config.temperature,
            decoding_method=config.decoding_method,
            num_mutations=config.num_mutations,
            batch_size=config.batch_size,
            model_checkpoint=config.model_checkpoint,
            verbose=config.verbose,
            return_logits=config.return_logits,
        )
    else:
        logger.debug(f"Using local venv for ESM2 sampling: {config.model_checkpoint}")
        venv_manager = EnvManager("esm2")
        script_path = Path(__file__).parent / "standalone" / "inference.py"
        result = venv_manager.call_standalone_script_in_venv(
            script_path=script_path,
            input_dict={
                "operation": "sample",
                "sequences": inputs.sequences,
                "temperature": config.temperature,
                "decoding_method": config.decoding_method,
                "num_mutations": config.num_mutations,
                "batch_size": config.batch_size,
                "model_checkpoint": config.model_checkpoint,
                "device": config.device,
                "verbose": config.verbose,
                "return_logits": config.return_logits,
            },
            device=config.device,
            verbose=config.verbose,
        )

    return ESM2SampleOutput(
        metadata={
            "model_checkpoint": config.model_checkpoint,
            "num_sequences": len(inputs.sequences),
            "temperature": config.temperature,
            "decoding_method": config.decoding_method,
            "num_mutations": config.num_mutations,
            "used_cloud": use_cloud_gpu(),
        },
        sequences=result["sequences"],
        logits=result["logits"],
    )
