"""PyRosetta Solvent Accessible Surface Area (SASA) scoring tool."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from proto_tools.entities.structures import Structure
from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_relax import PyRosettaRelaxConfig
from proto_tools.tools.structure_scoring.pyrosetta.shared_data_models import (
    ScoringStructureInput,
    prepare_pdb_and_chain_maps,
    relax_inputs_via_pyrosetta,
    remap_per_residue_chain_ids,
    warn_about_dropped_residues,
)
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

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
class ResidueSASA(BaseModel):
    """SASA value for a single residue.

    Attributes:
        chain_id (str): Chain identifier.
        residue_index (int): 1-indexed residue position.
        residue_name (str): Three-letter residue code.
        sasa (float): Solvent accessible surface area in Angstroms squared.
    """

    model_config = ConfigDict(extra="forbid")

    chain_id: str = Field(title="Chain ID", description="Chain identifier")
    residue_index: int = Field(title="Residue Index", description="1-indexed residue position")
    residue_name: str = Field(title="Residue Name", description="Three-letter residue code")
    sasa: float = Field(title="SASA", description="Per-residue solvent-accessible surface area in Å²")


class PyRosettaSASAMetrics(Metrics):
    """SASA result for a single structure.

    Metrics documented in ``metric_spec``:
        total_sasa (float): Total solvent accessible surface area in Å². When
            ``chains_to_score`` is set on the input, this is the sum over the selected
            residues only, not the whole-pose SASA. (Contrast with
            ``pyrosetta-energy``, whose ``total_energy`` is always the whole-pose
            total regardless of chain selection — SASA can be meaningfully
            summed over a residue subset, energy cannot.)

    Attributes:
        per_residue (list[ResidueSASA]): Per-residue SASA breakdown. Declared
            as a real field (not a metric) because each entry carries
            chain/residue identifiers alongside the SASA value.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "total_sasa": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": None,
            "unit": "Å²",
            "better_values_are": "context-dependent",
        },
    }
    primary_metric: str | None = Field(
        default="total_sasa",
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )

    per_residue: list[ResidueSASA] = Field(
        title="Per-Residue SASA",
        description="Per-residue SASA breakdown",
    )


class PyRosettaSASAInput(BaseToolInput):
    """Input for PyRosetta SASA computation.

    Attributes:
        inputs (list[ScoringStructureInput]): Protein structures to analyze,
            each with optional chain selection. Accepts bare Structure objects,
            PDB file paths, or PDB content strings for convenience.
    """

    inputs: list[ScoringStructureInput] = InputField(
        title="Structures",
        description="Protein structures with optional chain selection for SASA computation",
    )

    @field_validator("inputs", mode="before")
    @classmethod
    def normalize_inputs(cls, value: Any) -> Any:
        """Normalize a single structure/input to a list."""
        if isinstance(value, (str, Path, Structure, ScoringStructureInput)):
            value = [value]
        if isinstance(value, dict):
            value = [value]
        return value


class PyRosettaSASAConfig(BaseConfig):
    """Configuration for PyRosetta SASA computation.

    Attributes:
        probe_radius (float): Radius of the solvent probe sphere in Angstroms.
            Standard water probe is 1.4 A.
        pre_relax_structures (bool): If ``True``, run ``pyrosetta-relax`` on
            each input structure before scoring. Default ``False``.
        relax_config (PyRosettaRelaxConfig): Settings used when
            ``pre_relax_structures=True``. Ignored otherwise.
    """

    probe_radius: float = ConfigField(
        title="Probe Radius",
        default=1.4,
        gt=0.0,
        description="Solvent probe radius in Angstroms (standard water = 1.4)",
    )
    pre_relax_structures: bool = ConfigField(
        title="Pre-relax Structures",
        default=False,
        description="If True, run pyrosetta-relax on each input structure before scoring.",
    )
    relax_config: PyRosettaRelaxConfig = ConfigField(
        default_factory=PyRosettaRelaxConfig,
        title="Relax Config",
        description="Settings used when pre_relax_structures=True. Ignored otherwise.",
    )

    def preprocess(self, inputs: PyRosettaSASAInput) -> PyRosettaSASAInput:  # type: ignore[override]
        """Apply optional FastRelax preprocess to input structures."""
        if not self.pre_relax_structures:
            return inputs
        return inputs.model_copy(update={"inputs": relax_inputs_via_pyrosetta(inputs.inputs, self.relax_config)})

    @property
    def cpus_per_instance(self) -> int | None:
        """Opt in to ToolPool CPU fan-out — PyRosetta runs single-threaded per pose."""
        return 1


class PyRosettaSASAOutput(BaseToolOutput):
    """Output from PyRosetta SASA computation.

    Attributes:
        results (list[PyRosettaSASAMetrics]): SASA results, one per input structure.
    """

    results: list[PyRosettaSASAMetrics] = Field(
        default_factory=list,
        title="SASA Results",
        description="SASA results, one per input structure",
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
        rows = [
            {
                "structure_index": i,
                "chain_id": res.chain_id,
                "residue_index": res.residue_index,
                "residue_name": res.residue_name,
                "sasa": res.sasa,
                "total_sasa": result["total_sasa"],
            }
            for i, result in enumerate(self.results)
            for res in result.per_residue
        ]
        df = pd.DataFrame(rows)
        if file_format == "csv":
            df.to_csv(path, index=False)
        elif file_format == "json":
            df.to_json(path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return PyRosettaSASAInput(
        inputs=[
            ScoringStructureInput(
                structure=Structure(structure=str(Path(__file__).parent / "examples" / "example.pdb"))
            )
        ]
    )


@tool(
    key="pyrosetta-sasa",
    label="PyRosetta SASA",
    category="structure_scoring",
    input_class=PyRosettaSASAInput,
    config_class=PyRosettaSASAConfig,
    output_class=PyRosettaSASAOutput,
    metrics_class=PyRosettaSASAMetrics,
    description="Compute Solvent Accessible Surface Area (SASA) for protein structures using PyRosetta",
    uses_gpu=False,
    example_input=example_input,
    iterable_input_field="inputs",
    iterable_output_field="results",
    cacheable=True,
)
def run_pyrosetta_sasa(
    inputs: PyRosettaSASAInput,
    config: PyRosettaSASAConfig | None = None,
    instance: ToolInstance | None = None,
) -> PyRosettaSASAOutput:
    """Compute Solvent Accessible Surface Area (SASA) using PyRosetta.

    Calculates total and per-residue SASA using the SasaCalc module with
    a configurable probe radius. SASA measures the surface area of a protein
    accessible to solvent molecules.

    Args:
        inputs (PyRosettaSASAInput): Structures to analyze with optional chain selection.
        config (PyRosettaSASAConfig | None): Configuration with probe radius.
        instance (ToolInstance | None): Optional ToolInstance for persistent execution.

    Returns:
        PyRosettaSASAOutput: Total and per-residue SASA for each input structure.
    """
    logger.debug("Using local venv for PyRosetta SASA computation")

    pdb_contents, chain_ids_list, pdb_to_mmcif_maps = prepare_pdb_and_chain_maps(inputs.inputs)

    input_data = {
        "operation": "sasa",
        "pdb_contents": pdb_contents,
        "chain_ids_list": chain_ids_list,
        "probe_radius": config.probe_radius,  # type: ignore[union-attr]
        "device": "cpu",
    }

    output_data = ToolInstance.dispatch(
        "pyrosetta",
        input_data,
        instance=instance,
        config=config,
    )

    warn_about_dropped_residues(output_data["results"])
    remap_per_residue_chain_ids(output_data["results"], pdb_to_mmcif_maps)
    results = [PyRosettaSASAMetrics(**r) for r in output_data["results"]]

    return PyRosettaSASAOutput(
        metadata={
            "num_structures": len(inputs.inputs),
            "probe_radius": config.probe_radius,  # type: ignore[union-attr]
        },
        results=results,
    )
