"""Minerva sample for prepared mixed protein/DNA sequences."""

import logging
from typing import Any

from proto_tools.tools.masked_models.execution import dispatch_masked_model
from proto_tools.tools.masked_models.minerva.config import MinervaConfig
from proto_tools.tools.masked_models.mixed_configs import MixedSampleConfig
from proto_tools.tools.masked_models.mixed_data_models import (
    MixedSampleOutput,
    MixedSequenceSampleInput,
)
from proto_tools.tools.tool_registry import tool

logger = logging.getLogger(__name__)

# Input:
MinervaSampleInput = MixedSequenceSampleInput

# Output:
MinervaSampleOutput = MixedSampleOutput


class MinervaSampleConfig(MinervaConfig, MixedSampleConfig):
    """Configuration for Minerva sample.

    Attributes:
        model_checkpoint (Literal['gbrixi/minerva-mlm', 'gbrixi/minerva-mlm-8k']): Public model checkpoint.
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


def example_input() -> MinervaSampleInput:
    """Return a short prepared locus for examples and infrastructure checks."""
    return MinervaSampleInput(sequences=["+MKTL+acgt-ACDE"])


@tool(
    key="minerva-sample",
    label="Minerva Sample",
    category="masked_models",
    input_class=MinervaSampleInput,
    config_class=MinervaSampleConfig,
    output_class=MinervaSampleOutput,
    description="Sample mixed protein/DNA loci while preserving modality and strand markers with Minerva",
    uses_gpu=True,
    example_input=example_input,
    cacheable=True,
    stochastic=True,
    iterable_input_fields=["sequences", "mask_modalities"],
    iterable_output_field="results",
    max_chunk_size=32,
)
def run_minerva_sample(
    inputs: MinervaSampleInput,
    config: MinervaSampleConfig,
    instance: Any = None,
) -> MinervaSampleOutput:
    """Run Minerva sample on prepared mixed-modality input.

    Args:
        inputs (MinervaSampleInput): Validated mixed-sequence input.
        config (MinervaSampleConfig): Checkpoint and operation settings.
        instance (Any): Optional persistent tool instance.

    Returns:
        MinervaSampleOutput: Token-aligned sample output.
    """
    logger.debug("Using local worker for Minerva sample: %s", config.model_checkpoint)
    return MinervaSampleOutput(**dispatch_masked_model("minerva", "sample", inputs, config, instance))
