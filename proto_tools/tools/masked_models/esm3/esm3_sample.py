"""proto_tools/tools/masked_models/esm3/esm3_sample.py.

ESM3 sampling tool.
"""

import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from proto_tools.tools.masked_models.masking import (
    MaskingStrategy,
    apply_masking_strategy,
    build_position_score_fn,
)
from proto_tools.tools.masked_models.shared_data_models import (
    MaskedModelInput,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolOutput,
    ConfigField,
    ToolInstance,
    require_hf_token,
)

logger = logging.getLogger(__name__)

ESM3_MODEL_CHECKPOINTS = Literal["esm3_sm_open_v1",]

# ============================================================================
# Data Models
# ============================================================================
# Input:
ESM3SampleInput = MaskedModelInput


# Output:
class ESM3SampleOutput(BaseToolOutput):
    """Output from ESM3 protein sequence sampling.

    This class encapsulates the results of ESM3 sequence generation or mutation,
    providing the sampled protein sequences and optionally the logits.

    Attributes:
        sequences (list[str]): Sampled or mutated protein sequences. Each sequence
            is a string of amino acid characters. For de novo generation, these are
            completely new sequences. For mutation, these are modified versions of
            the input sequences with specified positions changed to model-predicted
            alternatives.
        logits (list[list[list[float]]] | None): Per-position logits for each
            sequence. Shape is (num_sequences, seq_len, vocab_size=20). Only present
            if return_logits=True in config.
    """

    sequences: list[str] = Field(description="Sampled/mutated protein sequences")
    logits: list[list[list[float]]] | None = Field(
        default=None,
        description="Per-position amino acid logits. Shape: [num_sequences, seq_len, 20].",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["fasta", "txt", "json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "fasta"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "fasta":
            with open(path, "w") as f:
                f.writelines(f">seq_{i}\n{seq}\n" for i, seq in enumerate(self.sequences))

        elif file_format == "txt":
            with open(path, "w") as f:
                f.writelines(f"{seq}\n" for seq in self.sequences)

        elif file_format == "json":
            with open(path, "w") as f:
                json.dump(self.sequences, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# Config:
class ESM3SampleConfig(BaseConfig):
    """Configuration for ESM3 protein sequence sampling.

    Attributes:
        model_checkpoint (Literal[ESM3_MODEL_CHECKPOINTS]): ESM3 model checkpoint. Default: ``"esm3_sm_open_v1"``.
        temperature (float): Sampling temperature (< 1.0 conservative, > 1.0 diverse). Default: 1.0.
        masking_strategy (MaskingStrategy): Controls which positions to mask for sampling. Default: random 30%.
        batch_size (int): Sequences per GPU forward pass. Default: 1.
        device (str): Device to run on. Default: ``"cuda"``.
        return_logits (bool): Whether to include per-position logits. Default: ``False``.
    """

    masking_strategy: MaskingStrategy = ConfigField(
        title="Masking Strategy",
        default_factory=MaskingStrategy,
        description="Controls which positions to mask for sampling. Default: random 30%.",
    )
    model_checkpoint: Literal[ESM3_MODEL_CHECKPOINTS] = ConfigField(
        title="Model Checkpoint",
        default="esm3_sm_open_v1",
        description="ESM3 model checkpoint to use",
        reload_on_change=True,
    )
    temperature: float = ConfigField(
        title="Sampling Temperature",
        default=1.0,
        description="Sampling temperature for amino acid selection",
        advanced=True,
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Number of sequences to process simultaneously on GPU",
        advanced=True,
    )
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

    def preprocess(self, inputs: Any) -> Any:
        """Apply masking strategy unless sequences are already pre-masked."""
        position_score_fn = build_position_score_fn("esm3", self.masking_strategy, self.device)
        return apply_masking_strategy(self, inputs, position_score_fn=position_score_fn)


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return ESM3SampleInput(sequences=["MKTL"])


@tool(
    key="esm3-sample",
    label="ESM3 Sampling",
    category="masked_models",
    input_class=ESM3SampleInput,
    config_class=ESM3SampleConfig,
    output_class=ESM3SampleOutput,
    description="Sample protein sequences using ESM3 language model",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="sequences",
    iterable_output_field="sequences",
)
def run_esm3_sample(
    inputs: ESM3SampleInput,
    config: ESM3SampleConfig,
    instance: Any = None,
) -> ESM3SampleOutput:
    """Sample protein sequences using ESM3 language model.

    The ``preprocess`` hook on :class:`ESM3SampleConfig` applies the masking
    strategy before this function runs, so ``inputs.sequences`` already
    contain ``_`` at positions to sample.

    Args:
        inputs (ESM3SampleInput): Protein sequences with ``_`` at designable positions.
        config (ESM3SampleConfig): Sampling configuration.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        ESM3SampleOutput: ESM3SampleOutput with sampled sequences and optional logits.
    """
    require_hf_token("ESM3", "https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1")

    logger.debug(f"Using local for ESM3 sampling: {config.model_checkpoint}")
    result = ToolInstance.dispatch(
        "esm3",
        {
            "operation": "sample",
            "sequences": inputs.sequences,
            "temperature": config.temperature,
            "batch_size": config.batch_size,
            "model_checkpoint": config.model_checkpoint,
            "device": config.device,
            "verbose": config.verbose,
            "return_logits": config.return_logits,
        },
        instance=instance,
        config=config,
    )

    return ESM3SampleOutput(
        metadata={
            "model_checkpoint": config.model_checkpoint,
            "num_sequences": len(inputs.sequences),
            "temperature": config.temperature,
        },
        sequences=result["sequences"],
        logits=result["logits"],
    )
