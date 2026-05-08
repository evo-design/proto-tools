"""AbLang sampling (restore) tool."""

import logging
from typing import Any

from pydantic import Field

from proto_tools.entities.antibody import Antibody
from proto_tools.tools.masked_models.ablang.ablang_embeddings import _resolve_model_choice
from proto_tools.tools.masked_models.shared_data_models import (
    MaskedModelSampleConfig,
    MaskedModelSampleOutput,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ConfigField
from proto_tools.utils.tool_instance import ToolInstance
from proto_tools.utils.tool_io import (
    BaseToolInput,
    InputField,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
class AbLangSampleInput(BaseToolInput):
    """Input for AbLang antibody sequence restoration/sampling.

    Chain sequences should contain underscores (``_``) at positions to be
    restored (predicted) by the model.

    Attributes:
        antibodies (list[Antibody]): Antibody sequence(s) with ``_`` at
            positions to restore.
    """

    antibodies: list[Antibody] = InputField(
        description="Antibody sequence(s) with '_' at positions to restore.",
        min_length=1,
    )


class AbLangSampleConfig(MaskedModelSampleConfig):
    """Configuration for AbLang antibody sequence restoration.

    Controls the restoration of masked positions in antibody sequences
    using AbLang's learned antibody language model distribution. The model
    variant is selected automatically based on which chains are provided.

    Attributes:
        batch_size (int): Number of sequences per forward pass.
        device (str): Device to run on.
        align (bool): Run ANARCI alignment first; enables restoration of unknown numbers of
            missing residues at chain termini.
        return_logits (bool): Include per-position logits in the output (large; disable to
            save memory). Triggers a second ``likelihood``-mode forward pass per batch.
    """

    align: bool = ConfigField(
        title="ANARCI-aligned Restore",
        default=False,
        description="Run ANARCI alignment first; enables extension of unknown-length termini",
        advanced=True,
    )
    return_logits: bool = ConfigField(
        title="Return Logits",
        default=False,
        description="Include per-position logits in the output (large; disable to save memory)",
        advanced=True,
    )


class AbLangSampleOutput(MaskedModelSampleOutput):
    """Output from AbLang antibody sequence restoration.

    Inherits from ``MaskedModelSampleOutput``.

    Attributes:
        sequences (list[str]): Restored antibody sequences with masked
            positions replaced by model predictions.
        logits (list[list[list[float]]] | None): Per-position logits for each restored
            sequence. Shape is (num_sequences, seq_len, vocab_size=20). Only present if
            ``return_logits=True`` in config.
    """

    logits: list[list[list[float]]] | None = Field(
        default=None,
        description="Per-position amino acid logits. Shape: [num_sequences, seq_len, 20].",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> AbLangSampleInput:
    """Minimal valid input for testing and examples."""
    return AbLangSampleInput(antibodies=[Antibody(heavy_chain="EVQL_ESGGGLVQPGG")])


@tool(
    key="ablang-sample",
    label="AbLang Sampling",
    category="masked_models",
    input_class=AbLangSampleInput,
    config_class=AbLangSampleConfig,
    output_class=AbLangSampleOutput,
    description="Restore masked antibody sequence positions using AbLang",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="antibodies",
    iterable_output_field="sequences",
    cacheable=True,
    generative=True,
)
def run_ablang_sample(
    inputs: AbLangSampleInput,
    config: AbLangSampleConfig,
    instance: Any = None,
) -> AbLangSampleOutput:
    """Restore masked positions in antibody sequences using AbLang."""
    sequences = [ab.to_sequence() for ab in inputs.antibodies]
    model_choice = _resolve_model_choice(inputs.antibodies)
    logger.debug("Using local venv for AbLang sampling: %s", model_choice)
    result = ToolInstance.dispatch(
        "ablang",
        {
            "operation": "sample",
            "sequences": sequences,
            "batch_size": config.batch_size,
            "model_choice": model_choice,
            "device": config.device,
            "verbose": config.verbose,
            "align": config.align,
            "return_logits": config.return_logits,
        },
        instance=instance,
        config=config,
    )

    return AbLangSampleOutput(
        metadata={
            "model_choice": model_choice,
            "num_sequences": len(sequences),
        },
        sequences=result["sequences"],
        logits=result["logits"],
    )
