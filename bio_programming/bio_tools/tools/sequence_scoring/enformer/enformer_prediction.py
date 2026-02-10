"""Enformer sequence scoring tool."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Literal, Union

from pydantic import ConfigDict, Field, field_validator

from bio_programming.bio_tools.tools.utils import BaseConfig, ConfigField, return_invalid_nucleotide_chars
from bio_programming.bio_tools.tools.infra.env_manager import EnvManager
from bio_programming.bio_tools.tools.infra.tool_io import BaseToolInput, BaseToolOutput
from bio_programming.bio_tools.tools.tool_registry import tool
from bio_programming.bio_tools.tools.utils import use_cloud_gpu

logger = logging.getLogger(__name__)

# Enformer model constants
ENFORMER_CONTEXT = 196_608
ENFORMER_OUTPUT = 896


# ============================================================================
# Data Models
# ============================================================================
# Input:
class EnformerInput(BaseToolInput):
    """Input for Enformer regulatory activity prediction.

    Attributes:
        sequence (str): DNA sequence for Enformer inference. Must be exactly
            196,608 bp and only contain valid nucleotide characters.
    """

    sequence: str = Field(description="DNA sequence to score")

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, sequence: str) -> str:
        """Validate and normalize nucleotide sequence; require length 196,608 bp."""
        if not sequence or not sequence.strip():
            raise ValueError("Sequence cannot be empty")
        sequence = sequence.upper()
        invalid_chars = return_invalid_nucleotide_chars(sequence, additional_valid_chars="N")
        if invalid_chars:
            raise ValueError(
                f"Invalid nucleotide characters in sequence: {', '.join(sorted(invalid_chars))}"
            )
        if len(sequence) != ENFORMER_CONTEXT:
            raise ValueError(
                f"Input sequence must have length {ENFORMER_CONTEXT}, got {len(sequence)}"
            )
        return sequence


# Output:
class EnformerOutput(BaseToolOutput):
    """Output from Enformer prediction.

    Attributes:
        sequence (str): Input DNA sequence that was scored.
        sequence_length (int): Length of the input sequence (always 196,608).
        prediction (List[List[float]]): Predicted signal matrix with shape
            ``[896, num_tracks]``.
        output_tracks (List[int]): Track indices that were extracted.
        species (str): Species used for prediction (``"human"`` or ``"mouse"``).
    """

    sequence: str = Field(description="Input DNA/RNA sequence")
    sequence_length: int = Field(description="Length of input sequence")
    prediction: List[List[float]] = Field(
        description="Predicted activity matrix with shape [896, num_tracks]"
    )
    output_tracks: List[int] = Field(description="Track indices extracted from Enformer")
    species: str = Field(description="Species used for prediction ('human' or 'mouse')")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def output_format_options(self) -> List[str]:
        return ["json", "csv"]

    @property
    def output_format_default(self) -> str:
        return "json"

    def _export_output(self, export_path: Union[Path, str], file_format: str):
        path = Path(export_path).with_suffix(f".{file_format}")
        _metadata_fields = {
            "tool_id", "execution_time", "timestamp", "success",
            "warnings", "errors", "metadata",
        }
        data = {
            k: v for k, v in self.model_dump().items()
            if k not in _metadata_fields
        }
        if file_format == "json":
            import json
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        elif file_format == "csv":
            import csv
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(data.keys())
                writer.writerow(data.values())
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# Config:
class EnformerConfig(BaseConfig):
    """Configuration for Enformer inference.

    Attributes:
        output_tracks (List[int]): Track indices to extract from the Enformer output.
        species (Literal["human", "mouse"]): Species track head to use.
        device (str): Device used for inference.
        verbose (bool): Whether to print status logs.
    """

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on (e.g., 'cuda', 'cpu')",
        hidden=True,
    )
    verbose: bool = ConfigField(
        title="Verbose",
        default=False,
        description="Whether to print status messages during execution",
        hidden=True,
    )
    output_tracks: List[int] = ConfigField(
        title="Output Tracks",
        description="Track indices to extract from model output",
    )
    species: Literal["human", "mouse"] = ConfigField(
        title="Species",
        default="human",
        description="Species track head to use",
        advanced=True,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
@tool(
    key="enformer-prediction",
    label="Enformer Prediction",
    input=EnformerInput,
    config=EnformerConfig,
    output=EnformerOutput,
    description="Gene expression and regulatory activity prediction using Enformer",
)
def run_enformer(inputs: EnformerInput, config: EnformerConfig) -> EnformerOutput:
    """Predict regulatory activity with Enformer.

    Args:
        inputs (EnformerInput): Validated sequence input.
        config (EnformerConfig): Validated runtime and model configuration.

    Returns:
        EnformerOutput: Prediction object with sequence, tracks, and metadata.
    """

    if use_cloud_gpu():
        logger.debug("Using the cloud runtime for Enformer prediction")

        import _gpu_runtime

        EnformerService = _gpu_runtime.Cls.from_name("bio-programming", "EnformerService")
        result = EnformerService().predict.remote(
            sequence=inputs.sequence,
            output_tracks=config.output_tracks,
            species=config.species,
            verbose=config.verbose,
        )
    else:
        logger.debug("Using local venv for Enformer prediction")

        venv_manager = EnvManager("enformer")
        script_path = Path(__file__).parent / "standalone" / "inference.py"
        result = venv_manager.call_standalone_script_in_venv(
            script_path=script_path,
            input_dict={
                "sequence": inputs.sequence,
                "output_tracks": config.output_tracks,
                "species": config.species,
                "device": config.device,
                "verbose": config.verbose,
            },
            device=config.device,
            verbose=config.verbose,
        )

    return EnformerOutput(
        sequence=inputs.sequence,
        sequence_length=len(inputs.sequence),
        prediction=result["prediction"],
        output_tracks=config.output_tracks,
        species=config.species,
    )
