"""IgLM full-length antibody sequence generation tool."""

import logging
from typing import Any

from pydantic import model_validator

from proto_tools.tools.causal_models.iglm.shared_data_models import (
    IgLMConditioningInput,
    IgLMGenerationConfig,
    IgLMGenerationOutput,
    IgLMSamples,
)
from proto_tools.tools.causal_models.shared_data_models import CausalModelSampleInput
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ConfigField, InputField, ToolInstance

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
class IgLMSampleInput(CausalModelSampleInput, IgLMConditioningInput):
    """Input for IgLM full-length antibody sequence generation.

    Attributes:
        prompts (list[str]): N-terminal prefix to continue for each item. An empty
            string generates from the conditioning tokens alone.
        chains (list[IgLMChain]): Chain type conditioning each prompt.
        species (list[IgLMSpecies]): Species-of-origin conditioning each prompt.
    """

    prompts: list[str] = InputField(
        default=[""],
        title="Prompts",
        description="N-terminal prefix per item; an empty string generates from scratch",
        examples=["EVQ", ["EVQ", "QVQ"]],
    )

    @model_validator(mode="after")
    def broadcast_conditioning(self) -> "IgLMSampleInput":
        """Normalize the broadcastable conditioning fields against ``prompts``."""
        return self.broadcast_parallel_fields("prompts")  # type: ignore[return-value]


# Output:
IgLMSampleOutput = IgLMGenerationOutput


# Config:
class IgLMSampleConfig(IgLMGenerationConfig):
    """Configuration for IgLM full-length antibody sequence generation.

    Attributes:
        model_checkpoint (IGLM_MODEL_CHECKPOINTS): IgLM weights variant.
        num_sequences_per_input (int): Distinct sequences to generate per prompt.
        temperature (float): Softmax temperature; lower is more conservative.
        top_p (float): Nucleus sampling cutoff over per-position token probabilities.
        max_length (int): Token budget per generation, including conditioning tokens
            and the prompt.
        prepend_prompt (bool): Keep the prompt at the start of each generated sequence;
            when ``False``, only the newly generated residues are returned.
        batch_size (int): Candidate sequences per forward pass.
        device (str): Device to run the model on.
    """

    prepend_prompt: bool = ConfigField(
        title="Prepend Prompt",
        default=True,
        description="Keep the prompt at the start of each generated sequence",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> IgLMSampleInput:
    """Minimal valid input for testing and examples."""
    return IgLMSampleInput(prompts=["EVQ"], chains=["heavy"], species=["human"])


@tool(
    key="iglm-sample",
    label="IgLM Sequence Generation",
    category="causal_models",
    input_class=IgLMSampleInput,
    config_class=IgLMSampleConfig,
    output_class=IgLMSampleOutput,
    description="Generate full-length antibody sequences with IgLM, conditioned on chain and species",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["prompts", "chains", "species"],
    iterable_output_field="results",
    max_chunk_size=8,
    cacheable=True,
    stochastic=True,
)
def run_iglm_sample(
    inputs: IgLMSampleInput,
    config: IgLMSampleConfig,
    instance: Any = None,
) -> IgLMSampleOutput:
    """Generate full-length antibody sequences with IgLM.

    Each prompt is continued autoregressively from its chain and species
    conditioning tokens until IgLM emits an end-of-sequence token, yielding
    ``config.num_sequences_per_input`` distinct sequences per prompt.

    Args:
        inputs (IgLMSampleInput): Prompts with their per-prompt chain and species
            conditioning. An empty prompt generates from the conditioning alone.
        config (IgLMSampleConfig): Checkpoint and sampling configuration.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        IgLMSampleOutput: One ``IgLMSamples`` bundle per prompt, in input order.

    Examples:
        >>> # Ten human heavy chains starting from EVQ
        >>> inputs = IgLMSampleInput(prompts=["EVQ"], chains=["heavy"], species=["human"])
        >>> result = run_iglm_sample(inputs, IgLMSampleConfig(num_sequences_per_input=10))
        >>> print(result.results[0].sequences[0])

        >>> # Nanobodies: camel heavy chains starting from QVQ
        >>> inputs = IgLMSampleInput(prompts=["QVQ"], chains=["heavy"], species=["camel"])
        >>> result = run_iglm_sample(inputs, IgLMSampleConfig(num_sequences_per_input=100))

        >>> # Mixed conditioning in one call: a human heavy and a human light chain
        >>> inputs = IgLMSampleInput(prompts=["EVQ", "DIQ"], chains=["heavy", "light"])
        >>> result = run_iglm_sample(inputs, IgLMSampleConfig())

    Note:
        - Sequences within a bundle are distinct; malformed generations are discarded,
          mirroring upstream's rejection sampling.
        - Generation stops at ``config.max_length`` tokens, counting the two
          conditioning tokens and the prompt. A candidate that runs out of budget
          before completing IgLM's format is discarded rather than returned, so too
          small a budget surfaces as a "yielded only N of M" error.
    """
    logger.debug("Using local venv for IgLM generation: %s", config.model_checkpoint)
    result = ToolInstance.dispatch(
        "iglm",
        {
            "operation": "sample",
            "prompts": inputs.prompts,
            "chain_tokens": inputs.chain_tokens(),
            "species_tokens": inputs.species_tokens(),
            "model_checkpoint": config.model_checkpoint,
            "num_sequences_per_input": config.num_sequences_per_input,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_length": config.max_length,
            "prepend_prompt": config.prepend_prompt,
            "batch_size": config.batch_size,
            "device": config.device,
            "verbose": config.verbose,
            "seed": config.seed,
        },
        instance=instance,
        config=config,
    )

    return IgLMSampleOutput(
        metadata={
            "model_checkpoint": config.model_checkpoint,
            "num_sequences_per_input": config.num_sequences_per_input,
            "temperature": config.temperature,
        },
        results=[IgLMSamples(sequences=sequences) for sequences in result["sequences"]],
    )
