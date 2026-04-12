"""PyRosetta energy scoring tool with optional FastRelax."""

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from proto_tools.entities.structures import Structure
from proto_tools.tools.structure_scoring.pyrosetta.shared_data_models import (
    ScoringStructureInput,
    prepare_pdb_and_chain_maps,
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

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
class ResidueEnergy(BaseModel):
    """Energy contribution of a single residue, in the context of the full structure.

    When a chain selection is used, the reported energy is the residue's
    contribution *within the full complex* (including pair interactions with
    unselected chains), not the energy of the chain scored in isolation. See
    :class:`EnergyResult` for details.

    Attributes:
        chain_id (str): Chain identifier.
        residue_index (int): 1-indexed residue position.
        residue_name (str): Three-letter residue code.
        total_energy (float): Total energy for this residue in REU.
    """

    model_config = ConfigDict(extra="forbid")

    chain_id: str = Field(description="Chain identifier")
    residue_index: int = Field(description="1-indexed residue position")
    residue_name: str = Field(description="Three-letter residue code")
    total_energy: float = Field(description="Total residue energy in REU")


class EnergyResult(BaseModel):
    """Energy scoring result for a single structure.

    ``total_energy`` and ``energy_terms`` are always computed on the full pose,
    regardless of any chain selection on the input (the full structure is
    required for the physics to be meaningful). A ``chain_ids`` selection only
    filters which residues appear in ``per_residue``; each entry still reflects
    that residue's contribution within the full complex. To score a chain as
    if it were isolated, extract it into its own Structure first.

    Attributes:
        total_energy (float): Total Rosetta energy in REU (Rosetta Energy Units).
            Always the whole-pose total, independent of chain selection.
        energy_terms (dict[str, float]): Breakdown by score term (fa_atr, fa_rep, etc.).
            Always the whole-pose terms, independent of chain selection.
        per_residue (list[ResidueEnergy]): Per-residue energy breakdown, filtered
            to the selected chains when ``chain_ids`` is set. Energies reflect
            each residue's contribution in the context of the full complex.
        relaxed (bool): Whether FastRelax was applied before scoring.
    """

    model_config = ConfigDict(extra="forbid")

    total_energy: float = Field(description="Total Rosetta energy in REU")
    energy_terms: dict[str, float] = Field(description="Energy breakdown by score term (fa_atr, fa_rep, etc.)")
    per_residue: list[ResidueEnergy] = Field(description="Per-residue energy breakdown")
    relaxed: bool = Field(description="Whether FastRelax was applied before scoring")


class PyRosettaEnergyInput(BaseToolInput):
    """Input for PyRosetta energy scoring.

    Attributes:
        inputs (list[ScoringStructureInput]): Protein structures to score,
            each with optional chain selection. Accepts bare Structure objects,
            PDB file paths, or PDB content strings for convenience.
    """

    inputs: list[ScoringStructureInput] = InputField(
        description="Protein structures with optional chain selection for energy scoring"
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


class PyRosettaEnergyConfig(BaseConfig):
    """Configuration for PyRosetta energy scoring.

    Attributes:
        scorefxn (str): Rosetta score function name. 'ref2015' is the current
            community standard.
        relax (bool): Whether to run FastRelax before scoring. Recommended for
            raw PDB structures to resolve clashes and strain.
        relax_cycles (int): Number of FastRelax cycles. More cycles improve
            convergence but increase runtime.
        constrain_to_start (bool): Whether to constrain relaxation to starting
            coordinates. Prevents large structural deviations.
    """

    scorefxn: str = ConfigField(
        title="Score Function",
        default="ref2015",
        description="Rosetta score function name",
        examples=["ref2015", "beta_nov16", "ref2015_cart"],
    )
    relax: bool = ConfigField(
        title="Relax Before Scoring",
        default=True,
        description="Run FastRelax before scoring to resolve clashes",
    )
    relax_cycles: int = ConfigField(
        title="Relax Cycles",
        default=5,
        ge=1,
        le=15,
        description="Number of FastRelax repeats (more = better convergence, slower)",
        advanced=True,
    )
    constrain_to_start: bool = ConfigField(
        title="Constrain to Start",
        default=True,
        description="Constrain relaxation to starting coordinates",
        advanced=True,
    )


class PyRosettaEnergyOutput(BaseToolOutput):
    """Output from PyRosetta energy scoring.

    Attributes:
        results (list[EnergyResult]): Energy scores, one per input structure.
    """

    results: list[EnergyResult] = Field(
        default_factory=list,
        description="Energy scores, one per input structure",
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
                "total_energy": result.total_energy,
                "relaxed": result.relaxed,
                "chain_id": res.chain_id,
                "residue_index": res.residue_index,
                "residue_name": res.residue_name,
                "residue_energy": res.total_energy,
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
    return PyRosettaEnergyInput(
        inputs=[
            ScoringStructureInput(
                structure=Structure(structure=str(Path(__file__).parent / "examples" / "example.pdb"))
            )
        ]
    )


@tool(
    key="pyrosetta-energy",
    label="PyRosetta Energy Score",
    category="structure_scoring",
    input_class=PyRosettaEnergyInput,
    config_class=PyRosettaEnergyConfig,
    output_class=PyRosettaEnergyOutput,
    description="Compute Rosetta energy scores for protein structures with optional FastRelax",
    uses_gpu=False,
    example_input=example_input,
    iterable_input_field="inputs",
    iterable_output_field="results",
    cacheable=True,
)
def run_pyrosetta_energy(
    inputs: PyRosettaEnergyInput,
    config: PyRosettaEnergyConfig | None = None,
    instance: ToolInstance | None = None,
) -> PyRosettaEnergyOutput:
    """Compute Rosetta energy scores using PyRosetta.

    Scores protein structures using the specified Rosetta score function.
    Optionally applies FastRelax before scoring to resolve steric clashes
    and strain in raw PDB structures.

    Chain selection only filters the per-residue breakdown; the whole pose
    is always scored, so ``total_energy``/``energy_terms`` are always the
    full-complex values and selected-residue energies are in-complex
    contributions, not isolated-chain energies. See :class:`EnergyResult`
    for details.

    Args:
        inputs (PyRosettaEnergyInput): Structures to score with optional chain selection.
        config (PyRosettaEnergyConfig | None): Configuration for score function and relaxation.
        instance (ToolInstance | None): Optional ToolInstance for persistent execution.

    Returns:
        PyRosettaEnergyOutput: Energy scores with per-residue breakdown.
    """
    logger.debug("Using local venv for PyRosetta energy scoring")

    seed = config.seed if config.seed is not None else config.get_random_int()  # type: ignore[union-attr]
    pdb_contents, chain_ids_list, pdb_to_mmcif_maps = prepare_pdb_and_chain_maps(inputs.inputs)

    input_data = {
        "operation": "energy",
        "pdb_contents": pdb_contents,
        "chain_ids_list": chain_ids_list,
        "scorefxn": config.scorefxn,  # type: ignore[union-attr]
        "relax": config.relax,  # type: ignore[union-attr]
        "relax_cycles": config.relax_cycles,  # type: ignore[union-attr]
        "constrain_to_start": config.constrain_to_start,  # type: ignore[union-attr]
        "seed": seed,
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
    results = [EnergyResult(**r) for r in output_data["results"]]

    return PyRosettaEnergyOutput(
        metadata={
            "num_structures": len(inputs.inputs),
            "scorefxn": config.scorefxn,  # type: ignore[union-attr]
            "relax": config.relax,  # type: ignore[union-attr]
        },
        results=results,
    )
