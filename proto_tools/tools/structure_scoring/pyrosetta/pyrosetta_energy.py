"""PyRosetta energy scoring tool, with opt-in FastRelax preprocess."""

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

    chain_id: str = Field(title="Chain ID", description="Chain identifier")
    residue_index: int = Field(title="Residue Index", description="1-indexed residue position")
    residue_name: str = Field(title="Residue Name", description="Three-letter residue code")
    total_energy: float = Field(
        title="Total Energy (REU)", description="Per-residue total energy in Rosetta Energy Units (REU)"
    )


class PyRosettaEnergyMetrics(Metrics):
    """Energy scoring result for a single structure.

    ``total_energy`` and ``energy_terms`` are always computed on the full pose,
    regardless of any chain selection on the input (the full structure is
    required for the physics to be meaningful). A ``chains_to_score`` selection only
    filters which residues appear in ``per_residue``; each entry still reflects
    that residue's contribution within the full complex. To score a chain as
    if it were isolated, extract it into its own Structure first.

    Metrics documented in ``metric_spec``:
        total_energy (float): Total Rosetta energy in REU (Rosetta Energy Units).
            Always the whole-pose total, independent of chain selection.

    Attributes:
        energy_terms (dict[str, float]): Breakdown by score term (fa_atr, fa_rep, etc.).
            Always the whole-pose terms. Declared as a real field (not a metric)
            because it's a named-term breakdown, not a scalar quantity.
        per_residue (list[ResidueEnergy]): Per-residue energy breakdown, filtered
            to the selected chains when ``chains_to_score`` is set. Declared as a
            real field because each entry carries chain/residue identifiers
            alongside the energy value.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "total_energy": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": None,
            "unit": "REU",
            "better_values_are": "lower",
        },
    }
    primary_metric: str | None = Field(
        default="total_energy",
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )

    energy_terms: dict[str, float] = Field(
        title="Energy Terms",
        description="Energy breakdown by score term (fa_atr, fa_rep, etc.)",
    )
    per_residue: list[ResidueEnergy] = Field(
        title="Per-Residue Energies",
        description="Per-residue energy breakdown",
    )


class PyRosettaEnergyInput(BaseToolInput):
    """Input for PyRosetta energy scoring.

    Attributes:
        inputs (list[ScoringStructureInput]): Protein structures to score,
            each with optional chain selection. Accepts bare Structure objects,
            PDB file paths, or PDB content strings for convenience.
    """

    inputs: list[ScoringStructureInput] = InputField(
        title="Structures",
        description="Protein structures with optional chain selection for energy scoring",
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
        scorefxn (str): Rosetta score function name. ``ref2015`` is the current
            community standard.
        pre_relax_structures (bool): If ``True``, run ``pyrosetta-relax`` on
            each input structure before scoring (the actual settings come from
            :attr:`relax_config`). Default ``False`` — energy is reported on
            the input structure as-given. Set to ``True`` for raw predicted
            structures with steric clashes that would otherwise inflate
            ``fa_rep``.
        relax_config (PyRosettaRelaxConfig): Settings used when
            ``pre_relax_structures=True``. Ignored otherwise.
    """

    scorefxn: str = ConfigField(
        title="Score Function",
        default="ref2015",
        description="Rosetta score function name",
        examples=["ref2015", "beta_nov16", "ref2015_cart"],
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

    def preprocess(self, inputs: PyRosettaEnergyInput) -> PyRosettaEnergyInput:  # type: ignore[override]
        """Apply optional FastRelax preprocess to input structures."""
        if not self.pre_relax_structures:
            return inputs
        return inputs.model_copy(update={"inputs": relax_inputs_via_pyrosetta(inputs.inputs, self.relax_config)})

    @property
    def cpus_per_instance(self) -> int | None:
        """Opt in to ToolPool CPU fan-out — PyRosetta runs single-threaded per pose."""
        return 1


class PyRosettaEnergyOutput(BaseToolOutput):
    """Output from PyRosetta energy scoring.

    Attributes:
        results (list[PyRosettaEnergyMetrics]): Energy scores, one per input
            structure. Each entry carries ``total_energy`` + ``relaxed`` as
            specced metrics plus ``energy_terms`` and ``per_residue`` as
            declared non-metric fields.
    """

    results: list[PyRosettaEnergyMetrics] = Field(
        default_factory=list,
        title="Energy Results",
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
                "total_energy": result["total_energy"],
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
    metrics_class=PyRosettaEnergyMetrics,
    description="Compute Rosetta energy scores for protein structures (with optional FastRelax preprocess via config.pre_relax_structures)",
    uses_gpu=False,
    example_input=example_input,
    iterable_input_fields=["inputs"],
    iterable_output_field="results",
    cacheable=True,
)
def run_pyrosetta_energy(
    inputs: PyRosettaEnergyInput,
    config: PyRosettaEnergyConfig | None = None,
    instance: ToolInstance | None = None,
) -> PyRosettaEnergyOutput:
    """Compute Rosetta energy scores using PyRosetta.

    Scores each protein structure using the specified Rosetta score function.
    To resolve steric clashes and strain before scoring, set
    ``config.pre_relax_structures=True`` — the framework's ``Config.preprocess``
    hook then dispatches ``pyrosetta-relax`` and substitutes the relaxed
    structures before this function runs.

    Chain selection only filters the per-residue breakdown; the whole pose
    is always scored, so ``total_energy``/``energy_terms`` are always the
    full-complex values and selected-residue energies are in-complex
    contributions, not isolated-chain energies. See :class:`PyRosettaEnergyMetrics`
    for details.

    Args:
        inputs (PyRosettaEnergyInput): Structures to score with optional chain selection.
        config (PyRosettaEnergyConfig | None): Score function + optional relax preprocess.
        instance (ToolInstance | None): Optional ToolInstance for persistent execution.

    Returns:
        PyRosettaEnergyOutput: Energy scores with per-residue breakdown.
    """
    logger.debug("Using local venv for PyRosetta energy scoring")

    pdb_contents, chain_ids_list, pdb_to_mmcif_maps = prepare_pdb_and_chain_maps(inputs.inputs)

    input_data = {
        "operation": "energy",
        "pdb_contents": pdb_contents,
        "chain_ids_list": chain_ids_list,
        "scorefxn": config.scorefxn,  # type: ignore[union-attr]
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
    results = [PyRosettaEnergyMetrics(**r) for r in output_data["results"]]

    return PyRosettaEnergyOutput(
        metadata={
            "num_structures": len(inputs.inputs),
            "scorefxn": config.scorefxn,  # type: ignore[union-attr]
            "pre_relax_structures": config.pre_relax_structures,  # type: ignore[union-attr]
        },
        results=results,
    )
