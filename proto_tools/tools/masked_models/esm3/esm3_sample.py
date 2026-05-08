"""proto_tools/tools/masked_models/esm3/esm3_sample.py.

ESM3 sampling tool.
"""

import logging
from typing import Any, Literal

from pydantic import Field

from proto_tools.tools.masked_models.shared_data_models import (
    MaskedModelInput,
    MaskedModelSampleConfig,
    MaskedModelSampleOutput,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.transforms.masking import (
    MaskingStrategy,
    apply_masking_strategy,
    build_position_score_fn,
)
from proto_tools.utils import (
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
class ESM3SampleOutput(MaskedModelSampleOutput):
    """Output from ESM3 protein sequence sampling.

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
        description="Per-position amino acid logits. Shape: [num_sequences, seq_len, 20].",
    )


# Config:
class ESM3SampleConfig(MaskedModelSampleConfig):
    """Configuration for ESM3 protein sequence sampling.

    Attributes:
        model_checkpoint (ESM3_MODEL_CHECKPOINTS): ESM3 weights variant.
        masking_strategy (MaskingStrategy): Positions to mask before sampling.
        sampling_method (Literal["single_pass", "iterative_refinement"]): "single_pass"
            fills every mask in one forward; "iterative_refinement" dispatches to
            ``model.batch_generate`` and uses the five GenerationConfig knobs below.
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

    masking_strategy: MaskingStrategy = ConfigField(
        title="Masking Strategy",
        default_factory=MaskingStrategy,
        description="Strategy for selecting positions to mask for resampling",
    )
    model_checkpoint: ESM3_MODEL_CHECKPOINTS = ConfigField(
        title="Model Checkpoint",
        default="esm3_sm_open_v1",
        description="ESM3 weights variant",
        reload_on_change=True,
    )
    sampling_method: Literal["single_pass", "iterative_refinement"] = ConfigField(
        title="Sampling Method",
        default="single_pass",
        description="'single_pass' samples every mask in one forward; 'iterative_refinement' uses batch_generate",
        advanced=True,
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
        advanced=True,
        depends_on={"sampling_method": ["iterative_refinement"]},
    )
    num_steps: int = ConfigField(
        title="Num Steps",
        default=20,
        ge=1,
        description="Iterative-refinement decoding steps; diminishing returns above 20",
        advanced=True,
        depends_on={"sampling_method": ["iterative_refinement"]},
    )
    schedule: Literal["cosine", "linear"] = ConfigField(
        title="Unmask Schedule",
        default="cosine",
        description="Unmask schedule across rounds; 'cosine' fronts more commits late",
        advanced=True,
        depends_on={"sampling_method": ["iterative_refinement"]},
    )
    strategy: Literal["random", "entropy"] = ConfigField(
        title="Unmask Strategy",
        default="random",
        description="Position-selection per round; 'entropy' commits the most-confident first",
        advanced=True,
        depends_on={"sampling_method": ["iterative_refinement"]},
    )
    temperature_annealing: bool = ConfigField(
        title="Temperature Annealing",
        default=True,
        description="Anneal temperature toward 0 across rounds",
        advanced=True,
        depends_on={"sampling_method": ["iterative_refinement"]},
    )
    return_logits: bool = ConfigField(
        title="Return Logits",
        default=False,
        description="Include per-position logits in the output (large; disable to save memory)",
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
    description="Sample masked positions in protein sequences using ESM3 language model",
    uses_gpu=True,
    generative=True,
    example_input=example_input,
    iterable_input_field="sequences",
    iterable_output_field="sequences",
)
def run_esm3_sample(
    inputs: ESM3SampleInput,
    config: ESM3SampleConfig,
    instance: Any = None,
) -> ESM3SampleOutput:
    """Sample masked positions in protein sequences using ESM3.

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

    return ESM3SampleOutput(
        metadata={
            "model_checkpoint": config.model_checkpoint,
            "num_sequences": len(inputs.sequences),
            "temperature": config.temperature,
        },
        sequences=result["sequences"],
        logits=result["logits"],
    )
