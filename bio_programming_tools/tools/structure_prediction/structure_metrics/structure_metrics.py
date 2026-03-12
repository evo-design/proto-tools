"""
Structure quality metrics from PDB files.

Computes structural quality metrics including longest alpha helix length
and gyration radius from PDB files, used to filter out disordered or
artifactual predicted protein structures.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import BaseModel, Field, field_validator

from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    InputField,
)


# ============================================================================
# Data Models
# ============================================================================
class StructureMetrics(BaseModel):
    """Metrics for a single predicted structure."""

    pdb_path: str = Field(description="Path to the PDB file analyzed")
    longest_alpha_helix: int = Field(
        description="Length of the longest alpha helix (in residues)"
    )
    gyration_radius: float = Field(
        description="Radius of gyration of the structure (in Angstroms)"
    )


# Input:
class StructureMetricsInput(BaseToolInput):
    """Input for structure quality metrics computation.

    Attributes:
        pdb_paths (List[str]): List of paths to PDB files to analyze.
    """

    pdb_paths: List[str] = InputField(
        description="List of PDB file paths to compute structure metrics for"
    )

    @field_validator("pdb_paths", mode="before")
    @classmethod
    def normalize_paths(cls, value) -> List[str]:
        """Normalize a single path to a list."""
        if isinstance(value, str):
            return [value]
        return [str(p) for p in value]


# Output:
class StructureMetricsOutput(BaseToolOutput):
    """Output from structure metrics computation.

    Attributes:
        metrics (List[StructureMetrics]): Per-structure metrics including
            longest alpha helix length and gyration radius.
    """

    metrics: List[StructureMetrics] = Field(
        default_factory=list,
        description="Per-structure quality metrics",
    )

    @property
    def output_format_options(self) -> List[str]:
        return ["csv", "json"]

    @property
    def output_format_default(self) -> str:
        return "csv"

    def _export_output(self, export_path: str | Path, file_format: str):
        import pandas as pd

        path = Path(export_path).with_suffix(f".{file_format}")
        df = pd.DataFrame([m.model_dump() for m in self.metrics])
        if file_format == "csv":
            df.to_csv(path, index=False)
        elif file_format == "json":
            df.to_json(path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# Config:
class StructureMetricsConfig(BaseConfig):
    """Configuration for structure metrics computation.

    No configuration parameters are needed — metrics are computed
    directly from the PDB structure.
    """


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input():
    """Minimal valid input for testing and examples."""
    return StructureMetricsInput(pdb_paths=[str(Path(__file__).parents[4] / "tests" / "dummy_data" / "structure_prediction_test_examples" / "gfp.fasta")])


@tool(
    key="structure-metrics",
    label="Structure Quality Metrics",
    category="structure_prediction",
    input_class=StructureMetricsInput,
    config_class=StructureMetricsConfig,
    output_class=StructureMetricsOutput,
    description="Compute structural quality metrics (longest alpha helix, gyration radius) from PDB files",
    example_input=example_input,
    iterable_input_field="pdb_paths",
    iterable_output_field="metrics",
    cacheable=True,
)
def run_structure_metrics(
    inputs: StructureMetricsInput, config: StructureMetricsConfig | None = None, instance=None,
) -> StructureMetricsOutput:
    """Compute structural quality metrics from PDB files.

    Uses biotite to annotate secondary structure elements (SSE) and compute
    the radius of gyration for each input PDB structure. These metrics are
    used to filter out structures with unusually long alpha helices (artifact)
    or high gyration radius (disordered).

    Args:
        inputs (StructureMetricsInput): Validated input containing PDB file paths.
        config (StructureMetricsConfig): Configuration (no parameters needed).

    Returns:
        StructureMetricsOutput: Per-structure metrics.

    Examples:
        >>> inputs = StructureMetricsInput(pdb_paths=["/path/to/structure.pdb"])
        >>> config = StructureMetricsConfig()
        >>> result = run_structure_metrics(inputs, config)
        >>> print(result.metrics[0].longest_alpha_helix)
        >>> print(result.metrics[0].gyration_radius)
    """
    from bio_programming_tools.utils.tool_instance import ToolInstance

    input_data = {
        "pdb_paths": inputs.pdb_paths,
    }

    input_data["device"] = "cpu"
    output_data = ToolInstance.dispatch(
        "structure_metrics",
        input_data,
        instance=instance,
        config=config,
    )

    metrics = [StructureMetrics(**m) for m in output_data["metrics"]]

    return StructureMetricsOutput(
        metadata={"num_structures": len(inputs.pdb_paths)},
        metrics=metrics,
    )
