"""AbLang embeddings tool."""

import logging
from typing import Any

from proto_tools.entities.antibody import Antibody, AntibodyLogits
from proto_tools.tools.masked_models.projection import attach_projections
from proto_tools.tools.masked_models.shared_data_models import (
    MaskedModelEmbeddingsConfig,
    MaskedModelEmbeddingsOutput,
    SequenceEmbedding,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ConfigField
from proto_tools.utils.tool_instance import ToolInstance
from proto_tools.utils.tool_io import BaseToolInput, InputField

logger = logging.getLogger(__name__)


def _chain_config(ab: Antibody | AntibodyLogits) -> tuple[bool, bool]:
    """Return (has_heavy, has_light) for an antibody."""
    return (ab.heavy_chain is not None, ab.light_chain is not None)


def _resolve_model_choice(antibodies: list[Antibody] | list[AntibodyLogits]) -> str:
    """Select the AbLang model variant and validate batch homogeneity."""
    first = _chain_config(antibodies[0])
    if any(_chain_config(ab) != first for ab in antibodies[1:]):
        raise ValueError("ablang: all antibodies in a batch must have the same chain configuration")
    has_heavy, has_light = first
    if has_heavy and has_light:
        return "ablang2-paired"
    if has_light:
        return "ablang1-light"
    return "ablang1-heavy"


# ============================================================================
# Data Models
# ============================================================================
class AbLangEmbeddingsInput(BaseToolInput):
    """Input for AbLang antibody embedding extraction.

    Attributes:
        antibodies (list[Antibody]): Antibody sequence(s) to embed.
    """

    antibodies: list[Antibody] = InputField(
        description="Antibody sequence(s) to embed.",
        min_length=1,
    )


class AbLangEmbeddingsConfig(MaskedModelEmbeddingsConfig):
    """Configuration for AbLang antibody embedding extraction.

    AbLang provides antibody-specific language model embeddings trained on
    paired antibody sequence data from the Observed Antibody Space (OAS).
    The model variant is selected automatically based on which chains are
    provided on each ``Antibody``.

    Attributes:
        batch_size (int): Number of sequences to process per forward pass.
        device (str): Device to run on.
        return_logits (bool): Include per-position amino-acid logits in output (large; disable
            to save memory).
    """

    return_logits: bool = ConfigField(
        title="Return Logits",
        default=False,
        description="Include per-position amino-acid logits in the output (large; disable to save memory)",
    )


class AbLangEmbeddingsOutput(MaskedModelEmbeddingsOutput):
    """Output from AbLang antibody embedding extraction.

    Inherits from ``MaskedModelEmbeddingsOutput``.

    Attributes:
        results (list[SequenceEmbedding]): Per-sequence embedding results. Each
            ``SequenceEmbedding`` contains:

            - ``mean_embedding``: Mean-pooled embedding vector (768-dim for ablang1, 480-dim for ablang2-paired)
            - ``attention_mask``: Binary mask (1 = valid, 0 = padding)
            - ``logits``: Optional per-position amino acid logits
    """


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> AbLangEmbeddingsInput:
    """Minimal valid input for testing and examples."""
    return AbLangEmbeddingsInput(antibodies=[Antibody(heavy_chain="EVQLVESGGGLVQPGG")])


@tool(
    key="ablang-embedding",
    label="AbLang Embeddings",
    category="masked_models",
    input_class=AbLangEmbeddingsInput,
    config_class=AbLangEmbeddingsConfig,
    output_class=AbLangEmbeddingsOutput,
    description="Extract antibody sequence embeddings using AbLang",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="antibodies",
    iterable_output_field="results",
    cacheable=True,
    post_process_iterable=attach_projections,
)
def run_ablang_embeddings(
    inputs: AbLangEmbeddingsInput,
    config: AbLangEmbeddingsConfig,
    instance: Any = None,
) -> AbLangEmbeddingsOutput:
    """Extract antibody sequence embeddings using AbLang."""
    sequences = [ab.to_sequence() for ab in inputs.antibodies]
    model_choice = _resolve_model_choice(inputs.antibodies)
    logger.debug("Using local venv for AbLang embeddings: %s", model_choice)
    outputs = ToolInstance.dispatch(
        "ablang",
        {
            "operation": "embeddings",
            "sequences": sequences,
            "batch_size": config.batch_size,
            "model_choice": model_choice,
            "device": config.device,
            "verbose": config.verbose,
            "return_logits": config.return_logits,
        },
        instance=instance,
        config=config,
    )

    logits_list = outputs["logits"]
    results = [
        SequenceEmbedding(
            mean_embedding=outputs["mean_embeddings"][i],
            attention_mask=outputs["attention_masks"][i],
            logits=logits_list[i] if logits_list is not None else None,
        )
        for i in range(len(sequences))
    ]

    return AbLangEmbeddingsOutput(
        metadata={
            "model_choice": model_choice,
            "num_sequences": len(sequences),
            "batch_size": config.batch_size,
            "device": config.device,
        },
        results=results,
    )
