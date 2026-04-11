"""AbLang sampling (restore) tool."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from proto_tools.tools.masked_models.ablang.ablang_embeddings import (
    ABLANG_MODEL_CHOICES,
    _resolve_model_choice,
)
from proto_tools.tools.masked_models.shared_data_models import (
    MaskedModelConfig,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ConfigField
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

    Sequences should contain underscores (``_``) at positions to be restored
    (predicted) by the model. The model will replace masked positions with
    its most likely amino acid predictions.

    Attributes:
        sequences (list[str]): Antibody sequence(s) with ``_`` at positions to
            restore. Format depends on the model choice:

            - ``ablang1-heavy``: Heavy chain with masked positions
              (e.g., ``"EVQL_ESGGGLVQPGG"``)
            - ``ablang1-light``: Light chain with masked positions
            - ``ablang2-paired``: Paired format ``"heavy_seq|light_seq"``
    """

    sequences: list[str] = InputField(
        description="Antibody sequence(s) with '_' at positions to restore. For paired models, use 'heavy|light' format.",
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, v: Any) -> Any:
        """Convert single string to list."""
        if isinstance(v, str):
            return [v]
        if not v:
            raise ValueError("sequences must not be empty")
        return v


class AbLangSampleConfig(MaskedModelConfig):
    """Configuration for AbLang antibody sequence restoration.

    Controls the restoration of masked positions in antibody sequences
    using AbLang's learned antibody language model distribution.

    Attributes:
        model_choice (ABLANG_MODEL_CHOICES): AbLang model variant to use:

            - ``"auto"``: Automatically select based on input format (default).
              Paired sequences (``"heavy|light"``) use ``ablang2-paired``,
              single-chain sequences use ``ablang1-heavy``.
            - ``"ablang1-heavy"``: Heavy chain only
            - ``"ablang1-light"``: Light chain only
            - ``"ablang2-paired"``: Paired heavy+light chains

        batch_size (int): Number of sequences per forward pass. Default: ``1``.

        device (str): Device to run on. Default: ``"cuda"``.
    """

    model_choice: ABLANG_MODEL_CHOICES = ConfigField(
        title="Model Choice",
        default="auto",
        description="Model variant: 'auto', 'ablang1-heavy', 'ablang1-light', or 'ablang2-paired'",
        reload_on_change=True,
    )


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
    return AbLangSampleInput(sequences=["EVQL_ESGGGLVQPGG"])


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
    iterable_input_field="sequences",
    iterable_output_field="sequences",
)
def run_ablang_sample(
    inputs: AbLangSampleInput,
    config: AbLangSampleConfig,
    instance: Any = None,
) -> AbLangSampleOutput:
    """Restore masked positions in antibody sequences using AbLang.

    Uses AbLang's learned antibody sequence distribution to predict the most
    likely amino acids at masked (``_``) positions. This is useful for:

    - Restoring missing or uncertain regions in antibody sequences
    - Predicting CDR loop sequences given framework context
    - Generating antibody sequence variants

    Args:
        inputs (AbLangSampleInput): Validated input containing antibody
            sequences with ``_`` at positions to restore.
        config (AbLangSampleConfig): Configuration specifying model variant
            and device.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        AbLangSampleOutput: Restored sequences with masked positions filled
            by model predictions.

    Examples:
        >>> inputs = AbLangSampleInput(sequences=["EVQL_ESGGGLVQPGG"])
        >>> config = AbLangSampleConfig(model_choice="ablang1-heavy")
        >>> result = run_ablang_sample(inputs, config)
        >>> print(f"Restored: {result.sequences[0]}")

    Note:
        - Positions marked with ``_`` will be replaced by model predictions
        - For paired models, both chains can have masked positions
        - Restoration uses greedy decoding (most likely amino acid per position)
    """
    resolved_model = _resolve_model_choice(config.model_choice, inputs.sequences)
    logger.debug(f"Using local venv for AbLang sampling: {resolved_model}")
    result = ToolInstance.dispatch(
        "ablang",
        {
            "operation": "sample",
            "sequences": inputs.sequences,
            "batch_size": config.batch_size,
            "model_choice": resolved_model,
            "device": config.device,
            "verbose": config.verbose,
        },
        instance=instance,
        config=config,
    )

    return AbLangSampleOutput(
        metadata={
            "model_choice": resolved_model,
            "num_sequences": len(inputs.sequences),
        },
        sequences=result["sequences"],
    )
