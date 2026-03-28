"""bio_programming_tools/tools/masked_models/esm2/esm2_sample.py

ESM2 sampling tool."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field

from bio_programming_tools.tools.masked_models.masking import (
    MaskingStrategy,
    apply_masking_strategy,
    build_position_score_fn,
)
from bio_programming_tools.tools.masked_models.shared_data_models import (
    MaskedModelInput,
)
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import (
    BaseConfig,
    BaseToolOutput,
    ConfigField,
    ToolInstance,
)

logger = logging.getLogger(__name__)

ESM2_MODEL_CHECKPOINTS = Literal[
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
    "esm2_t48_15B_UR50D",
]

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
        sequences (list[str]): Sampled or mutated protein sequences. Each sequence
            is a string of amino acid characters. For de novo generation, these are
            completely new sequences. For mutation, these are modified versions of
            the input sequences with specified positions changed to model-predicted
            alternatives.
        logits (list[list[list[float]]] | None): Per-position logits for each
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
    """Configuration for ESM2 protein sequence sampling.

    Attributes:
        model_checkpoint (Literal[ESM2_MODEL_CHECKPOINTS]): ESM2 model variant. Default: ``"esm2_t33_650M_UR50D"``.
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
    model_checkpoint: Literal[ESM2_MODEL_CHECKPOINTS] = ConfigField(
        title="ESM2 Model Checkpoint",
        default="esm2_t33_650M_UR50D",
        description="Name of the ESM2 model variant to use",
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

    def preprocess(self, inputs):
        """Apply masking strategy unless sequences are already pre-masked."""
        position_score_fn = build_position_score_fn("esm2", self.masking_strategy, self.device)
        return apply_masking_strategy(self, inputs, position_score_fn=position_score_fn)


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input():
    """Minimal valid input for testing and examples."""
    return ESM2SampleInput(sequences=["MKTL"])


@tool(
    key="esm2-sample",
    label="ESM2 Sampling",
    category="masked_models",
    input_class=ESM2SampleInput,
    config_class=ESM2SampleConfig,
    output_class=ESM2SampleOutput,
    description="Sample protein sequences using ESM2 language model",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="sequences",
    iterable_output_field="sequences",
)
def run_esm2_sample(
    inputs: ESM2SampleInput, config: ESM2SampleConfig | None = None,
    instance=None,
) -> ESM2SampleOutput:
    """Sample protein sequences using ESM2 language model.

    The ``preprocess`` hook on :class:`ESM2SampleConfig` applies the masking
    strategy before this function runs, so ``inputs.sequences`` already
    contain ``_`` at positions to sample.

    Args:
        inputs (ESM2SampleInput): Protein sequences with ``_`` at designable positions.
        config (ESM2SampleConfig | None): Sampling configuration.

    Returns:
        ESM2SampleOutput: ESM2SampleOutput with sampled sequences and optional logits.
    """
    logger.debug(f"Using local for ESM2 sampling: {config.model_checkpoint}")
    result = ToolInstance.dispatch(
        "esm2",
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

    return ESM2SampleOutput(
        metadata={
            "model_checkpoint": config.model_checkpoint,
            "num_sequences": len(inputs.sequences),
            "temperature": config.temperature,
        },
        sequences=result["sequences"],
        logits=result["logits"],
    )
