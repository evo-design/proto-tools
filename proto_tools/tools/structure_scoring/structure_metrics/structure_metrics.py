"""Structural quality metrics (helix/sheet/loop %, longest helix, gyration radius) from PDB files.

Used to filter out disordered or artifactual predicted protein structures.
"""

from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, field_validator

from proto_tools.entities.structures import Structure
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    InputField,
)
from proto_tools.utils.tool_io import Metrics, MetricSpec


# ============================================================================
# Data Models
# ============================================================================
class StructureQualityMetrics(Metrics):
    """Structural quality metrics for a single PDB.

    Metrics documented in ``metric_spec``:
        longest_alpha_helix (int): Length of the longest alpha helix (residues).
        gyration_radius (float): Radius of gyration of the structure (Å).
        helix_pct (float): Percentage of residues in alpha helices (0-100).
        sheet_pct (float): Percentage of residues in beta sheets (0-100).
        loop_pct (float): Percentage of residues in loops/coil (0-100).
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "longest_alpha_helix": {
            "availability": "always",
            "type": "int",
            "min": 0,
            "max": None,
            "unit": "residues",
            "better_values_are": "context-dependent",
        },
        "gyration_radius": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": None,
            "unit": "Å",
            "better_values_are": "context-dependent",
        },
        "helix_pct": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 100.0,
            "unit": "%",
            "better_values_are": "context-dependent",
        },
        "sheet_pct": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 100.0,
            "unit": "%",
            "better_values_are": "context-dependent",
        },
        "loop_pct": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 100.0,
            "unit": "%",
            "better_values_are": "context-dependent",
        },
    }


# Input:
class StructureMetricsInput(BaseToolInput):
    """Input for structure quality metrics computation.

    Attributes:
        structures (list[Structure]): Structures to analyze. Accepts file paths,
            raw PDB/CIF content strings, or ``Structure`` objects per item.
    """

    structures: list[Structure] = InputField(
        title="Structures",
        description="Structures to compute structure metrics for",
    )

    @field_validator("structures", mode="before")
    @classmethod
    def _wrap_single_in_list(cls, value: Any) -> list[Any]:
        """Wrap a single Structure / path / dict input in a list; per-item Structure coercion handles paths and content."""
        if isinstance(value, (str, Path, Structure, dict)):
            return [value]
        return list(value)


# Output:
class StructureMetricsOutput(BaseToolOutput):
    """Output from structure metrics computation.

    Attributes:
        metrics (list[StructureQualityMetrics]): Per-structure quality metrics,
            index-aligned with ``inputs.structures``.
    """

    metrics: list[StructureQualityMetrics] = Field(
        default_factory=list,
        title="Quality Metrics",
        description="Per-structure quality metrics",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["csv", "json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "csv"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
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

    No configuration parameters are needed. Metrics are computed
    directly from the PDB structure.
    """


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return StructureMetricsInput(structures=[Structure.from_file(Path(__file__).parent / "example_input_fixture.pdb")])


@tool(
    key="structure-metrics",
    label="Structure Quality Metrics",
    category="structure_scoring",
    input_class=StructureMetricsInput,
    config_class=StructureMetricsConfig,
    output_class=StructureMetricsOutput,
    metrics_class=StructureQualityMetrics,
    description="Compute structural quality metrics (SS percentages, longest helix, gyration radius) from PDB files",
    example_input=example_input,
    iterable_input_fields=["structures"],
    iterable_output_field="metrics",
    cacheable=True,
)
def run_structure_metrics(
    inputs: StructureMetricsInput,
    config: StructureMetricsConfig,  # noqa: ARG001
    instance: Any = None,  # noqa: ARG001
) -> StructureMetricsOutput:
    """Compute structural quality metrics from PDB structures.

    Computes secondary structure percentages, longest alpha helix, and radius of
    gyration for each input structure. Used to filter out disordered or artifactual
    predicted structures.

    Args:
        inputs (StructureMetricsInput): Structures to analyze.
        config (StructureMetricsConfig): No parameters needed.
        instance (Any): Unused (no standalone dispatch).

    Returns:
        StructureMetricsOutput: Per-structure quality metrics, index-aligned with inputs.structures.
    """
    metrics = []
    for struct in inputs.structures:
        ss = struct.secondary_structure_percentages()
        metrics.append(
            StructureQualityMetrics(
                longest_alpha_helix=struct.longest_alpha_helix(),
                gyration_radius=struct.gyration_radius(),
                helix_pct=ss["helix"],
                sheet_pct=ss["sheet"],
                loop_pct=ss["loop"],
            )
        )

    return StructureMetricsOutput(
        metadata={"num_structures": len(inputs.structures)},
        metrics=metrics,
    )
