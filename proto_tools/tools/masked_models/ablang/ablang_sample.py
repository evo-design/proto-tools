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
        title="Antibodies",
        description="Antibody sequence(s) with '_' at positions to restore.",
        min_length=1,
    )


class AbLangSampleConfig(MaskedModelSampleConfig):
    """Configuration for AbLang antibody sequence sampling.

    Controls how masked positions in antibody sequences are filled in by
    AbLang's learned antibody language model distribution. The model
    variant is selected automatically based on which chains are provided.

    Attributes:
        batch_size (int): Number of sequences per forward pass.
        device (str): Device to run on.
        temperature (float): Softmax temperature for per-position amino-acid
            sampling. ``temperature == 0`` selects greedy argmax decoding
            (equivalent to ablang's native ``restore`` mode). ``temperature == 1``
            samples from the unscaled model distribution; higher values flatten
            the distribution toward uniform, lower values sharpen toward greedy.
        align (bool): Run ANARCI alignment first; enables restoration of unknown numbers of
            missing residues at chain termini. Forces greedy decoding (ANARCI's
            spread-of-variants logic is incompatible with stochastic sampling).
        return_logits (bool): Include per-position logits in the output (large; disable to
            save memory). Triggers a second ``likelihood``-mode forward pass per batch.
    """

    temperature: float = ConfigField(
        title="Temperature",
        default=1.0,
        ge=0.0,
        description="Softmax temperature for amino-acid sampling. 0 = greedy argmax (ablang restore); >0 = stochastic.",
    )
    align: bool = ConfigField(
        title="ANARCI-aligned Restore",
        default=False,
        description="Run ANARCI alignment first; enables extension of unknown-length termini",
    )
    return_logits: bool = ConfigField(
        title="Return Logits",
        default=False,
        description="Include per-position logits in the output (large; disable to save memory)",
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
        title="Logits",
        description="Per-position amino acid logits. Shape: [num_sequences, seq_len, 20].",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> AbLangSampleInput:
    """Minimal valid input for testing and examples."""
    # Several mask positions so stochastic sampling produces visibly diverse
    # outputs (a single mask + 20-token vocab makes collision likely in 3 draws).
    return AbLangSampleInput(antibodies=[Antibody(heavy_chain="EVQL_E_GG_LVQ_GG")])


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
    iterable_input_fields=["antibodies"],
    iterable_output_field="sequences",
    cacheable=True,
    stochastic=True,
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
            "temperature": config.temperature,
            "seed": config.seed,
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
