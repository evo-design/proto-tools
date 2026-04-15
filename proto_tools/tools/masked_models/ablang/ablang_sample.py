"""AbLang sampling (restore) tool."""

import logging
from pathlib import Path
from typing import Any

from pydantic import Field

from proto_tools.entities.antibody import Antibody
from proto_tools.tools.masked_models.ablang.ablang_embeddings import _resolve_model_choice
from proto_tools.tools.masked_models.shared_data_models import (
    MaskedModelConfig,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils.tool_instance import ToolInstance
from proto_tools.utils.tool_io import (
    BaseToolInput,
    BaseToolOutput,
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


class AbLangSampleConfig(MaskedModelConfig):
    """Configuration for AbLang antibody sequence restoration.

    Controls the restoration of masked positions in antibody sequences
    using AbLang's learned antibody language model distribution. The model
    variant is selected automatically based on which chains are provided.

    Attributes:
        batch_size (int): Number of sequences per forward pass. Default: ``1``.
        device (str): Device to run on. Default: ``"cuda"``.
    """


class AbLangSampleOutput(BaseToolOutput):
    """Output from AbLang antibody sequence restoration.

    Attributes:
        sequences (list[str]): Restored antibody sequences with masked
            positions replaced by model predictions.
    """

    sequences: list[str] = Field(
        description="Restored antibody sequences with masked positions filled in",
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
        """Export restored sequences to the specified file format."""
        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "fasta":
            with open(path, "w") as f:
                f.writelines(f">seq_{i}\n{seq}\n" for i, seq in enumerate(self.sequences))

        elif file_format == "txt":
            with open(path, "w") as f:
                f.writelines(f"{seq}\n" for seq in self.sequences)

        elif file_format == "json":
            import json

            with open(path, "w") as f:
                json.dump(self.sequences, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


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
    )
