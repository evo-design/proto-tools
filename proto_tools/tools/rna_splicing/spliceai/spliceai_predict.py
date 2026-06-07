"""SpliceAI raw splice-site probability prediction from DNA sequence."""

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
class SpliceAIPredictInput(BaseToolInput):
    """Input for SpliceAI raw splice-site prediction.

    Attributes:
        sequences (list[str]): DNA sequence(s) to predict on. A single string is
            auto-wrapped into a list. Sequences may be any length; SpliceAI pads
            5000 bp of context on each side internally, so predictions cover every
            input position.
    """

    sequences: list[str] = InputField(
        title="Sequences",
        description="DNA sequence(s) to predict splice-site probabilities for",
        examples=["ACGTACGTACGT", ["ACGT", "TTGGCCAA"]],
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, value: Any) -> Any:
        """Convert a single string to a list and reject empty input."""
        if isinstance(value, str):
            return [value]
        if not value:
            raise ValueError("sequences must not be empty")
        return value


class SpliceAIPredictConfig(BaseConfig):
    """Configuration for SpliceAI raw prediction.

    Attributes:
        device (str): Device to run inference on. SpliceAI (TensorFlow)
            auto-falls-back to CPU when no GPU is visible.
    """

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run inference on (e.g. 'cpu', 'cuda', 'cuda:0')",
        include_in_key=False,
    )


class SpliceAIPredictOutput(BaseToolOutput):
    """Output from SpliceAI raw splice-site prediction.

    Attributes:
        predictions (list[list[list[float]]]): Per sequence, per position, a
            ``[neither, acceptor, donor]`` probability triple. Outer length and
            order match the input ``sequences``; each inner length equals the
            corresponding input sequence's length.
    """

    predictions: list[list[list[float]]] = Field(
        title="Predictions",
        description="Per-sequence, per-position [neither, acceptor, donor] probabilities",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["npy", "json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "npy"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "npy":
            import numpy as np

            # Ragged batches (sequences of differing lengths) need an object array.
            uniform = len({len(seq_pred) for seq_pred in self.predictions}) <= 1
            arr = np.array(self.predictions) if uniform else np.array(self.predictions, dtype=object)
            np.save(path, arr)
            return

        if file_format == "json":
            with open(path, "w") as handle:
                json.dump(self.predictions, handle)
            return

        raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return SpliceAIPredictInput(sequences=["ACGT" * 25])


@tool(
    key="spliceai-predict",
    label="SpliceAI Splice-Site Prediction",
    category="rna_splicing",
    input_class=SpliceAIPredictInput,
    config_class=SpliceAIPredictConfig,
    output_class=SpliceAIPredictOutput,
    description="Predict per-position acceptor/donor splice-site probabilities from DNA sequence with SpliceAI",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="predictions",
    cacheable=True,
)
def run_spliceai_predict(
    inputs: SpliceAIPredictInput,
    config: SpliceAIPredictConfig,
    instance: Any = None,
) -> SpliceAIPredictOutput:
    """Predict per-position [neither, acceptor, donor] splice-site probabilities from DNA sequence.

    Args:
        inputs (SpliceAIPredictInput): DNA sequence(s) to predict on.
        config (SpliceAIPredictConfig): Device setting.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        SpliceAIPredictOutput: Per-sequence, per-position [neither, acceptor, donor] probabilities (ragged across lengths).
    """
    logger.debug("Using local venv for SpliceAI splice-site prediction")

    dispatch_result = ToolInstance.dispatch(
        "spliceai",
        {
            "operation": "predict",
            "sequences": inputs.sequences,
            "device": config.device,
        },
        instance=instance,
        config=config,
    )

    return SpliceAIPredictOutput(predictions=dispatch_result["predictions"])
