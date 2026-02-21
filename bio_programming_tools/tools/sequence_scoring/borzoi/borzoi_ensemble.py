"""Borzoi ensemble sequence scoring tool."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Literal, Union

from pydantic import ConfigDict, Field, model_validator
from tqdm import tqdm

from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig, ConfigField
from bio_programming_tools.utils.tool_io import BaseToolOutput

from .borzoi_prediction import BorzoiConfig, BorzoiInput, run_borzoi

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
# Input: BorzoiInput

# Output:
class BorzoiEnsembleOutput(BaseToolOutput):
    """Output from Borzoi ensemble prediction.

    Attributes:
        sequence (str): Input DNA sequence that was scored.
        sequence_length (int): Length of the input sequence (always 524,288).
        predictions (List[List[List[float]]]): Stacked predictions with shape
            ``[4, num_tracks, 6144]`` for replicates 0-3.
        output_tracks (List[int]): Track indices used for prediction.
        species (str): Species used for prediction (``"human"`` or ``"mouse"``).
        avg_output_tracks (bool): Whether requested tracks were averaged.
        num_replicates (int): Number of replicates returned (always 4).
    """

    sequence: str = Field(description="Input DNA/RNA sequence")
    sequence_length: int = Field(description="Length of input sequence")
    predictions: List[List[List[float]]] = Field(
        description="Stacked predictions with shape [4, num_tracks, 6144]"
    )
    output_tracks: List[int] = Field(description="Track indices used for prediction")
    species: str = Field(description="Species used for prediction ('human' or 'mouse')")
    avg_output_tracks: bool = Field(description="Whether track outputs were averaged")
    num_replicates: int = Field(
        default=4, description="Number of Borzoi replicates returned"
    )

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
class BorzoiEnsembleConfig(BaseConfig):
    """Configuration for Borzoi ensemble prediction.

    Attributes:
        output_tracks (List[int]): Track indices to extract from model output.
        species (Literal["human", "mouse"]): Species model to use.
        avg_output_tracks (bool): Whether to average selected tracks.
        use_flash_attn (bool): Whether to run FlashAttention-backed models.
        device (str): Device used for inference.
    """

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on (e.g., 'cuda', 'cpu')",
        hidden=True,
    )
    output_tracks: List[int] = ConfigField(
        title="Output Tracks",
        description="Track indices to extract from model output",
    )
    species: Literal["human", "mouse"] = ConfigField(
        title="Species",
        default="human",
        description="Species model to use",
    )
    avg_output_tracks: bool = ConfigField(
        title="Average Tracks",
        default=True,
        description="Whether to average selected tracks into one output",
        advanced=True,
    )
    use_flash_attn: bool = ConfigField(
        title="Use FlashAttention",
        default=True,
        description="Whether to use FlashAttention models",
        hidden=True,
    )

    @model_validator(mode="after")
    def validate_mouse_flash_attn(self) -> BorzoiEnsembleConfig:
        """Mouse Borzoi checkpoints do not support FlashAttention."""
        if self.species == "mouse" and self.use_flash_attn:
            raise ValueError(
                "FlashAttention (use_flash_attn=True) is not available for mouse models."
            )
        return self


# ============================================================================
# Tool Implementation
# ============================================================================
@tool(
    key="borzoi-ensemble",
    label="Borzoi Ensemble",
    category="sequence_scoring",
    input=BorzoiInput,
    config=BorzoiEnsembleConfig,
    output=BorzoiEnsembleOutput,
    description="Regulatory activity prediction using all 4 Borzoi replicates",
    uses_gpu=True,
)
def run_borzoi_ensemble(
    inputs: BorzoiInput, config: BorzoiEnsembleConfig,
    instance=None,
) -> BorzoiEnsembleOutput:
    """Predict regulatory activity using all Borzoi replicates.

    Args:
        inputs (BorzoiInput): Validated sequence input.
        config (BorzoiEnsembleConfig): Validated runtime and model configuration.

    Returns:
        BorzoiEnsembleOutput: Stacked predictions from Borzoi replicates 0-3.
    """

    if config.use_flash_attn and not config.device.startswith("cuda"):
        raise ValueError("Must run on GPU to use FlashAttention with Borzoi")

    logger.debug("Using local execution for Borzoi ensemble prediction")

    predictions: List[List[List[float]]] = []
    iterator = tqdm(
        range(4),
        desc="Borzoi replicates",
        unit="replicate",
        total=4,
        disable=not config.verbose,
    )

    for replicate in iterator:
        replicate_config = BorzoiConfig(
            output_tracks=config.output_tracks,
            species=config.species,
            replicate=str(replicate),
            avg_output_tracks=config.avg_output_tracks,
            use_flash_attn=config.use_flash_attn,
            device=config.device,
            verbose=config.verbose,
        )
        replicate_output = run_borzoi(inputs, replicate_config)
        predictions.append(replicate_output.prediction)

    return BorzoiEnsembleOutput(
        sequence=inputs.sequence,
        sequence_length=len(inputs.sequence),
        predictions=predictions,
        output_tracks=config.output_tracks,
        species=config.species,
        avg_output_tracks=config.avg_output_tracks,
        num_replicates=4,
    )
