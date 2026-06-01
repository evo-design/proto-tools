"""PyMOL-backed RMSD alignment tool."""

import json
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field

from proto_tools.entities.structures import Structure
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)
from proto_tools.utils.tool_io import Metrics, MetricSpec

PyMOLAlignmentMethod = Literal["cealign", "align"]


class PyMOLRMSDInput(BaseToolInput):
    """Input for PyMOL RMSD alignment.

    Attributes:
        target_structure (Structure): Target/reference structure.
        mobile_structure (Structure): Mobile/query structure to align against the target.

    Both fields accept a ``Structure`` object, a file path, or raw PDB/CIF
    content; each is normalised to a ``Structure``.
    """

    target_structure: Structure = InputField(
        title="Target Structure",
        description="Target/reference structure (Structure object, file path, or raw PDB/CIF string)",
    )
    mobile_structure: Structure = InputField(
        title="Mobile Structure",
        description="Mobile/query structure (Structure object, file path, or raw PDB/CIF string)",
    )


class PyMOLRMSDConfig(BaseConfig):
    """Configuration for PyMOL RMSD alignment.

    Attributes:
        method (PyMOLAlignmentMethod): PyMOL alignment routine to use.
        target_selection (str): PyMOL selection for the target/reference structure.
        mobile_selection (str): PyMOL selection for the mobile/query structure.
        failure_rmsd (float): RMSD returned when PyMOL cannot align the structures.
    """

    method: PyMOLAlignmentMethod = ConfigField(
        title="PyMOL Alignment Method",
        default="cealign",
        description="PyMOL alignment routine to use for RMSD calculation.",
    )
    target_selection: str = ConfigField(
        title="Target Selection", default="target", description="PyMOL target selection, e.g. 'target and name CA'."
    )
    mobile_selection: str = ConfigField(
        title="Mobile Selection", default="mobile", description="PyMOL mobile selection, e.g. 'mobile and name CA'."
    )
    failure_rmsd: float = ConfigField(
        title="Failure RMSD",
        default=999.0,
        description="RMSD returned when PyMOL cannot align the requested structures.",
        include_in_key=False,
    )


class PyMOLRMSDMetrics(Metrics):
    """RMSD metrics emitted by PyMOL.

    Metrics documented in ``metric_spec``:
        rmsd (float): Post-alignment RMSD in Angstroms.
        aligned_length (int): CE alignment length, available for ``cealign``.
        aligned_atoms (int): Final aligned atom count, available for ``align``.
        alignment_cycles (int): Outlier-rejection cycles run by ``align``.
        alignment_score (float): Raw sequence/structure alignment score from ``align``.
        pre_refinement_rmsd (float): Initial RMSD before ``align`` refinement.
        pre_refinement_aligned_atoms (int): Initial aligned atom count before refinement.
        aligned_residues (int): Residue pairs aligned by ``align``.
    """

    primary_metric: str | None = Field(
        default="rmsd",
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )
    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "rmsd": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": None,
            "unit": "angstrom",
            "better_values_are": "lower",
        },
        "aligned_length": {
            "availability": "cealign only",
            "type": "int",
            "min": 0,
            "max": None,
            "better_values_are": "context-dependent",
        },
        "aligned_atoms": {
            "availability": "align only",
            "type": "int",
            "min": 0,
            "max": None,
            "better_values_are": "context-dependent",
        },
        "alignment_cycles": {
            "availability": "align only",
            "type": "int",
            "min": 0,
            "max": None,
            "better_values_are": "context-dependent",
        },
        "alignment_score": {
            "availability": "align only",
            "type": "float",
            "min": None,
            "max": None,
            "better_values_are": "higher",
        },
        "pre_refinement_rmsd": {
            "availability": "align only",
            "type": "float",
            "min": 0.0,
            "max": None,
            "unit": "angstrom",
            "better_values_are": "lower",
        },
        "pre_refinement_aligned_atoms": {
            "availability": "align only",
            "type": "int",
            "min": 0,
            "max": None,
            "better_values_are": "context-dependent",
        },
        "aligned_residues": {
            "availability": "align only",
            "type": "int",
            "min": 0,
            "max": None,
            "better_values_are": "context-dependent",
        },
    }


class PyMOLRMSDOutput(BaseToolOutput):
    """Output from PyMOL RMSD alignment.

    Attributes:
        method (PyMOLAlignmentMethod): PyMOL alignment method used.
        metrics (PyMOLRMSDMetrics): RMSD alignment metrics.
    """

    method: PyMOLAlignmentMethod = Field(title="Method", description="PyMOL alignment method used.")
    metrics: PyMOLRMSDMetrics = Field(
        default_factory=PyMOLRMSDMetrics,
        title="RMSD Metrics",
        description="RMSD alignment metrics.",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return supported output formats."""
        return ["json"]

    @property
    def output_format_default(self) -> str:
        """Return default output format."""
        return "json"

    def _export_output(self, export_path: Path | str, file_format: str) -> None:
        if file_format != "json":
            raise ValueError(f"Unsupported format: {file_format}")
        path = Path(export_path).with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"method": self.method, **dict(self.metrics.items())}, f, indent=2)


_EXAMPLE_PDB_PATH = Path(__file__).parents[1] / "example_input_fixture.pdb"


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    structure = Structure.from_file(_EXAMPLE_PDB_PATH)
    return PyMOLRMSDInput(target_structure=structure, mobile_structure=structure)


@tool(
    key="pymol-rmsd-alignment",
    label="PyMOL RMSD Alignment",
    category="structure_alignment",
    input_class=PyMOLRMSDInput,
    config_class=PyMOLRMSDConfig,
    output_class=PyMOLRMSDOutput,
    metrics_class=PyMOLRMSDMetrics,
    uses_gpu=False,
    description="Pairwise structure RMSD alignment using PyMOL cealign or align.",
    example_input=example_input,
)
def run_pymol_rmsd_alignment(
    inputs: PyMOLRMSDInput,
    config: PyMOLRMSDConfig,
    instance: Any = None,
) -> PyMOLRMSDOutput:
    """Run PyMOL RMSD alignment on two PDB structures."""
    input_data = {
        "target_pdb_text": inputs.target_structure.structure_pdb,
        "mobile_pdb_text": inputs.mobile_structure.structure_pdb,
        "method": config.method,
        "target_selection": config.target_selection,
        "mobile_selection": config.mobile_selection,
        "failure_rmsd": config.failure_rmsd,
        "device": "cpu",
    }
    output_data = ToolInstance.dispatch(
        "pymol_rmsd",
        input_data,
        instance=instance,
        config=config,
    )

    return PyMOLRMSDOutput(
        method=output_data.get("method", config.method),
        metrics=PyMOLRMSDMetrics(
            rmsd=output_data["rmsd"],
            aligned_length=output_data.get("aligned_length"),
            aligned_atoms=output_data.get("aligned_atoms"),
            alignment_cycles=output_data.get("alignment_cycles"),
            alignment_score=output_data.get("alignment_score"),
            pre_refinement_rmsd=output_data.get("pre_refinement_rmsd"),
            pre_refinement_aligned_atoms=output_data.get("pre_refinement_aligned_atoms"),
            aligned_residues=output_data.get("aligned_residues"),
        ),
        metadata={
            "tool": "pymol_rmsd",
            "method": output_data.get("method", config.method),
            "target_selection": config.target_selection,
            "mobile_selection": config.mobile_selection,
            **({"alignment_error": output_data["alignment_error"]} if output_data.get("alignment_error") else {}),
        },
    )
