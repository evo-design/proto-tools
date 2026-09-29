"""gLM2 sample for prepared mixed protein/DNA sequences."""

import logging
from typing import Any, Literal

from proto_tools.tools.masked_models.mixed_data_models import (
    MixedSampleConfig,
    MixedSampleOutput,
    MixedSequenceSampleInput,
    dispatch_mixed_model,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ConfigField

logger = logging.getLogger(__name__)

# Input:
GLM2SampleInput = MixedSequenceSampleInput

# Output:
GLM2SampleOutput = MixedSampleOutput


class GLM2SampleConfig(MixedSampleConfig):
    """Configuration for gLM2 sample.

    Attributes:
        model_checkpoint (Literal['tattabio/gLM2_150M', 'tattabio/gLM2_650M']): Public model checkpoint.
        batch_size (int): Sequences per forward pass.
        masking_strategy (MaskingStrategy): Select editable token positions when no explicit masks are supplied.
        sampling_method (Literal['single_pass', 'iterative_refinement']): Sampling algorithm.
        temperature (float): Temperature within each editable position's modality.
        top_p (float): Nucleus threshold for iterative refinement.
        num_steps (int): Number of iterative refinement rounds.
        schedule (Literal['cosine', 'linear']): Iterative unmask schedule.
        strategy (Literal['random', 'entropy']): Random or confidence-based commitment selection.
        temperature_annealing (bool): Cool temperature during iterative refinement.
        return_logits (bool): Include biological logits from the completed sequence.
    """

    model_checkpoint: Literal["tattabio/gLM2_150M", "tattabio/gLM2_650M"] = ConfigField(
        default="tattabio/gLM2_650M",
        title="Model Checkpoint",
        description="gLM2 checkpoint; downloaded automatically from Hugging Face",
        reload_on_change=True,
    )


def example_input() -> GLM2SampleInput:
    """Return a short prepared locus for examples and infrastructure checks."""
    return GLM2SampleInput(sequences=["<+>MKTL<+>acgt<->ACDE"])


@tool(
    key="glm2-sample",
    label="gLM2 Sample",
    category="masked_models",
    input_class=GLM2SampleInput,
    config_class=GLM2SampleConfig,
    output_class=GLM2SampleOutput,
    description="Sample mixed protein/DNA loci while preserving modality and strand markers with gLM2",
    uses_gpu=True,
    example_input=example_input,
    cacheable=True,
    stochastic=True,
    iterable_input_fields=["sequences", "mask_modalities"],
    iterable_output_field="results",
    max_chunk_size=32,
)
def run_glm2_sample(
    inputs: GLM2SampleInput,
    config: GLM2SampleConfig,
    instance: Any = None,
) -> GLM2SampleOutput:
    """Run gLM2 sample on prepared mixed-modality input.

    Args:
        inputs (GLM2SampleInput): Validated mixed-sequence input.
        config (GLM2SampleConfig): Checkpoint and operation settings.
        instance (Any): Optional persistent tool instance.

    Returns:
        GLM2SampleOutput: Token-aligned sample output.
    """
    logger.debug("Using local worker for gLM2 sample: %s", config.model_checkpoint)
    return GLM2SampleOutput(**dispatch_mixed_model("glm2", "sample", inputs, config, instance))
