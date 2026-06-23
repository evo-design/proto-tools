"""proto_tools/tools/masked_models/esm2/esm2_sample.py.

ESM2 sampling tool.
"""

import logging
from typing import Any, ClassVar, Literal

from pydantic import Field, field_validator

from proto_tools.tools.masked_models.shared_data_models import (
    MaskedModelInput,
    MaskedModelSampleConfig,
    MaskedModelSampleOutput,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.transforms.masking import (
    MaskingInput,
    MaskingStrategy,
    apply_masking_strategy,
    build_position_score_fn,
)
from proto_tools.utils import (
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

ESM2_MAX_SEQ_LENGTH = 1022


# ============================================================================
# Data Models
# ============================================================================
# Input:
class ESM2SampleInput(MaskedModelInput):
    """ESM-2 sampling input.

    Attributes:
        sequences (list[str]): Protein sequence(s) with ``_`` at positions to sample.
            Each must be ≤ 1022 residues (ESM-2's positional-encoding cap);
            over-length inputs raise ``ValueError``.
    """

    @field_validator("sequences")
    @classmethod
    def _validate_max_length(cls, sequences: list[str]) -> list[str]:
        for idx, seq in enumerate(sequences):
            if len(seq) > ESM2_MAX_SEQ_LENGTH:
                raise ValueError(
                    f"esm2: supports sequences up to {ESM2_MAX_SEQ_LENGTH} residues; input {idx} has length {len(seq)}."
                )
        return sequences


# Output:
class ESM2SampleOutput(MaskedModelSampleOutput):
    """Output from ESM2 protein sequence sampling.

    Inherits from ``MaskedModelSampleOutput``.

    Attributes:
        sequences (list[str]): Sampled or mutated protein sequences. Each sequence
            is a string of amino acid characters and is a modified version of the
            input sequence with masked positions changed to model-predicted
            alternatives.
        logits (list[list[list[float]]] | None): Per-position logits for each
            sequence. Shape is (num_sequences, seq_len, vocab_size=20). Only present
            if return_logits=True in config.
    """

    logits: list[list[list[float]]] | None = Field(
        default=None,
        title="Logits",
        description="Per-position amino acid logits. Shape: [num_sequences, seq_len, 20].",
    )


# Config:
class ESM2SampleConfig(MaskedModelSampleConfig):
    """Configuration for ESM2 protein sequence sampling.

    Attributes:
        model_checkpoint (ESM2_MODEL_CHECKPOINTS): ESM2 weights variant.
        masking_strategy (MaskingStrategy): Positions to mask before sampling.
        sampling_method (Literal["single_pass", "iterative_refinement"]): "single_pass"
            fills every mask in one forward; "iterative_refinement" runs an
            iterative MaskGIT-style loop driven by the five settings below.
        temperature (float): Softmax temperature.
        top_p (float): Nucleus threshold (iterative only).
        num_steps (int): Refinement steps (iterative only).
        schedule (Literal["cosine", "linear"]): Unmask schedule (iterative only).
        strategy (Literal["random", "entropy"]): Per-round commit selection (iterative only).
        temperature_annealing (bool): Anneal toward 0 across rounds (iterative only).
        batch_size (int): Sequences per GPU forward pass.
        device (str): Device to run on.
        return_logits (bool): Include per-position logits.
    """

    # Has a model → supplies per-position logits, so model-informed masking is valid.
    masking_inputs: ClassVar[frozenset[MaskingInput]] = frozenset({MaskingInput.LOGITS})

    masking_strategy: MaskingStrategy = ConfigField(
        title="Masking Strategy",
        default_factory=MaskingStrategy,
        description="Strategy for selecting positions to mask for resampling",
    )
    model_checkpoint: ESM2_MODEL_CHECKPOINTS = ConfigField(
        title="ESM2 Model Checkpoint",
        default="esm2_t33_650M_UR50D",
        description="ESM2 weights variant; trade off speed vs sample quality",
        reload_on_change=True,
    )
    sampling_method: Literal["single_pass", "iterative_refinement"] = ConfigField(
        title="Sampling Method",
        default="single_pass",
        description="'single_pass' samples every mask in one forward; 'iterative_refinement' runs the MaskGIT-style loop",
    )
    temperature: float = ConfigField(
        title="Sampling Temperature",
        default=1.0,
        gt=0.0,
        description="Softmax temperature for per-position amino-acid sampling",
    )
    top_p: float = ConfigField(
        title="Top P",
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling threshold; 1.0 disables",
    )
    num_steps: int = ConfigField(
        title="Num Steps",
        default=20,
        ge=1,
        description="Iterative-refinement decoding steps; diminishing returns above 20",
    )
    schedule: Literal["cosine", "linear"] = ConfigField(
        title="Unmask Schedule",
        default="cosine",
        description="Unmask schedule across rounds; 'cosine' fronts more commits late",
    )
    strategy: Literal["random", "entropy"] = ConfigField(
        title="Unmask Strategy",
        default="random",
        description="Position-selection per round; 'entropy' commits the most-confident first",
    )
    temperature_annealing: bool = ConfigField(
        title="Temperature Annealing",
        default=True,
        description="Anneal temperature toward 0 across rounds",
    )
    return_logits: bool = ConfigField(
        title="Return Logits",
        default=False,
        description="Include per-position logits in the output (large; disable to save memory)",
    )

    def cloud_unsupported_reason(self) -> str | None:
        if self.model_checkpoint == "esm2_t48_15B_UR50D":
            return "The 15B variant (esm2_t48_15B_UR50D) isn't available with device='cloud'. Choose a smaller variant, or run locally."
        return None

    def preprocess(self, inputs: Any) -> Any:
        """Apply masking strategy unless sequences are already pre-masked."""
        position_score_fn = build_position_score_fn(
            "esm2", self.masking_strategy, self.device, sampling_checkpoint=self.model_checkpoint
        )
        return apply_masking_strategy(self, inputs, position_score_fn=position_score_fn)


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return ESM2SampleInput(sequences=["MKTL"])


@tool(
    key="esm2-sample",
    label="ESM2 Sampling",
    category="masked_models",
    input_class=ESM2SampleInput,
    config_class=ESM2SampleConfig,
    output_class=ESM2SampleOutput,
    description="Sample masked positions in protein sequences using ESM2 language model",
    uses_gpu=True,
    stochastic=True,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="sequences",
)
def run_esm2_sample(
    inputs: ESM2SampleInput,
    config: ESM2SampleConfig,
    instance: Any = None,
) -> ESM2SampleOutput:
    """Sample masked positions in protein sequences using ESM2.

    The ``preprocess`` hook on :class:`ESM2SampleConfig` applies the masking
    strategy before this function runs, so ``inputs.sequences`` already
    contain ``_`` at positions to sample.

    Args:
        inputs (ESM2SampleInput): Protein sequences with ``_`` at designable positions.
        config (ESM2SampleConfig): Sampling configuration.

        instance (Any): Optional ToolInstance for subprocess execution.

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
            "sampling_method": config.sampling_method,
            "top_p": config.top_p,
            "num_steps": config.num_steps,
            "schedule": config.schedule,
            "strategy": config.strategy,
            "temperature_annealing": config.temperature_annealing,
            "batch_size": config.batch_size,
            "model_checkpoint": config.model_checkpoint,
            "device": config.device,
            "verbose": config.verbose,
            "return_logits": config.return_logits,
            "seed": config.seed,
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
