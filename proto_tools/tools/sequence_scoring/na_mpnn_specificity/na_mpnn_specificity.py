"""NA-MPNN protein-DNA specificity prediction tool."""

import csv
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

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
# Input
# ============================================================================
class NAMPNNSpecificityInput(BaseToolInput):
    """Input for NA-MPNN specificity prediction.

    Attributes:
        pdb_paths (list[str]): PDB file paths for protein-DNA complexes to score.
            A bare string is normalized into a single-element list.
    """

    pdb_paths: list[str] = InputField(
        title="PDB Paths",
        description="PDB file paths for protein-DNA complexes to score for DNA specificity",
        min_length=1,
    )

    @field_validator("pdb_paths", mode="before")
    @classmethod
    def normalize_paths(cls, value: Any) -> Any:
        """Normalize a single path string into a list."""
        if isinstance(value, str):
            return [value]
        return value

    def __len__(self) -> int:
        """Return the number of input PDB paths."""
        return len(self.pdb_paths)


# ============================================================================
# Output
# ============================================================================
class NAMPNNSpecificityResult(BaseModel):
    """Canonicalized NA-MPNN specificity result for one structure.

    Attributes:
        input_name (str): Basename (stem) of the scored input structure.
        source_method (str): Source method tag, always ``"na_mpnn"``.
        output_npz_path (str): Path to the canonical ``.npz`` output file.
        predicted_ppm (list[list[float]]): Canonical DNA PPM (``L x 4``) in A,C,G,T order.
        true_sequence (list[int]): Canonical DNA truth indices in ``0..3`` per row.
        mask (list[int]): Valid-residue mask for canonical rows.
        dna_mask (list[int]): DNA-residue mask for canonical rows.
        chain_labels (list[int]): Canonical chain IDs per canonical row.
    """

    input_name: str = Field(title="Input Name", description="Basename of the scored input structure")
    source_method: str = Field(title="Source Method", description="Source method tag for this prediction")
    output_npz_path: str = Field(title="Output NPZ Path", description="Path to canonical NPZ output")
    predicted_ppm: list[list[float]] = Field(
        title="Predicted PPM", description="Canonical DNA PPM (L x 4) in A,C,G,T order"
    )
    true_sequence: list[int] = Field(title="True Sequence", description="Canonical DNA truth indices in 0..3")
    mask: list[int] = Field(title="Mask", description="Valid residue mask for canonical rows")
    dna_mask: list[int] = Field(title="DNA Mask", description="DNA residue mask for canonical rows")
    chain_labels: list[int] = Field(title="Chain Labels", description="Canonical chain IDs for canonical rows")


class NAMPNNSpecificityOutput(BaseToolOutput):
    """Output from NA-MPNN specificity prediction.

    Attributes:
        results (list[NAMPNNSpecificityResult]): Canonicalized result per input structure,
            index-aligned with ``pdb_paths``.
    """

    results: list[NAMPNNSpecificityResult] = Field(
        title="Results",
        description="Canonicalized specificity result per input structure",
    )

    def __len__(self) -> int:
        """Return the number of per-structure results."""
        return len(self.results)

    def __getitem__(self, index: int) -> NAMPNNSpecificityResult:
        """Return the per-structure result at ``index``."""
        return self.results[index]

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json", "csv"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: Path | str, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")
        rows = [result.model_dump() for result in self.results]
        if file_format == "json":
            with open(path, "w") as handle:
                json.dump(rows, handle, indent=2)
            return
        if file_format == "csv":
            fieldnames = [
                "input_name",
                "source_method",
                "output_npz_path",
                "predicted_ppm",
                "true_sequence",
                "mask",
                "dna_mask",
                "chain_labels",
            ]
            with open(path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()}
                    )
            return
        raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Config
# ============================================================================
class NAMPNNSpecificityConfig(BaseConfig):
    """Configuration for NA-MPNN specificity prediction.

    The NA-MPNN checkout and its in-tree specificity checkpoint are provisioned
    automatically by the standalone setup into the managed weights cache, so no
    repository or checkpoint path is configured here.

    Attributes:
        batch_size (int): Batch size per inference call.
        number_of_batches (int): Number of prediction batches to run.
        temperature (float): Sampling temperature for NA-MPNN.
        omit_aa (str): Residues to omit during NA-MPNN sampling.
        design_na_only (bool): Restrict design tokens to nucleic acid positions.
        output_directory (str | None): Optional directory for canonical NPZ artifacts.
        keep_intermediate (bool): Keep intermediate raw NA-MPNN output directories.
        device (str): Device to run inference on (inherited).
    """

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run NA-MPNN inference on",
        include_in_key=False,
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Batch size used for each NA-MPNN prediction pass",
    )
    number_of_batches: int = ConfigField(
        title="Num Batches",
        default=1,
        ge=1,
        description="Number of NA-MPNN prediction batches to run",
    )
    temperature: float = ConfigField(
        title="Temperature",
        default=0.1,
        gt=0.0,
        description="Sampling temperature for NA-MPNN",
    )
    omit_aa: str = ConfigField(
        title="Omit Residues",
        default="",
        description="Residues omitted during NA-MPNN sampling",
    )
    design_na_only: bool = ConfigField(
        title="Design NA Only",
        default=True,
        description="Restrict design to nucleic acid positions",
    )
    output_directory: str | None = ConfigField(
        title="Output Directory",
        default=None,
        description="Optional output directory for canonical NPZ artifacts",
        include_in_key=False,
    )
    keep_intermediate: bool = ConfigField(
        title="Keep Intermediate",
        default=False,
        description="Keep intermediate raw NA-MPNN output directories",
    )

    def cloud_unsupported_reason(self) -> str | None:
        """NA-MPNN needs a local repo checkout + checkpoint that can't be staged to cloud."""
        return (
            "NA-MPNN requires a local repository checkout and checkpoint on disk, which "
            "can't be staged to device='cloud'. Run locally with device='cuda' or 'cpu'."
        )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> NAMPNNSpecificityInput:
    """Minimal valid input for testing and examples."""
    return NAMPNNSpecificityInput(pdb_paths=["protein_dna_complex.pdb"])


@tool(
    key="na-mpnn-specificity",
    label="NA-MPNN Specificity",
    category="sequence_scoring",
    input_class=NAMPNNSpecificityInput,
    config_class=NAMPNNSpecificityConfig,
    output_class=NAMPNNSpecificityOutput,
    description="Predict protein-DNA specificity (DNA-only A,C,G,T PPM) using NA-MPNN",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["pdb_paths"],
    iterable_output_field="results",
)
def run_na_mpnn_specificity(
    inputs: NAMPNNSpecificityInput,
    config: NAMPNNSpecificityConfig,
    instance: Any = None,
) -> NAMPNNSpecificityOutput:
    """Run NA-MPNN specificity prediction and return canonicalized outputs.

    Args:
        inputs (NAMPNNSpecificityInput): Input structure paths for prediction.
        config (NAMPNNSpecificityConfig): Runtime configuration values.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        NAMPNNSpecificityOutput: Canonicalized specificity outputs per structure,
            index-aligned with ``inputs.pdb_paths``.
    """
    logger.debug("Using local venv for NA-MPNN specificity prediction")

    output_data = ToolInstance.dispatch(
        "na_mpnn_specificity",
        {
            "pdb_paths": inputs.pdb_paths,
            "batch_size": config.batch_size,
            "number_of_batches": config.number_of_batches,
            "temperature": config.temperature,
            "omit_aa": config.omit_aa,
            "design_na_only": config.design_na_only,
            "output_directory": config.output_directory,
            "keep_intermediate": config.keep_intermediate,
            "device": config.device,
            "verbose": config.verbose,
        },
        instance=instance,
        config=config,
    )

    results = [NAMPNNSpecificityResult(**item) for item in output_data["results"]]
    if len(results) != len(inputs.pdb_paths):
        raise ValueError(f"Expected {len(inputs.pdb_paths)} NA-MPNN results, got {len(results)}")
    return NAMPNNSpecificityOutput(
        results=results,
        metadata={"num_inputs": len(results)},
    )
