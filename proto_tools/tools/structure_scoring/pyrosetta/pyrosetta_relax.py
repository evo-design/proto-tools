"""PyRosetta FastRelax wrapper that returns a relaxed Structure.

Companion to ``pyrosetta_energy``: where the energy tool runs FastRelax as an
opt-in pre-step and discards the relaxed pose, this tool is the explicit
relax-and-return path. The output ``Structure`` is meant to be chained into
the existing ``pyrosetta-energy`` / ``pyrosetta-sap`` / ``pyrosetta-sasa``
tools (or downstream geometric filters) without re-relaxing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from proto_tools.entities.structures import Structure
from proto_tools.tools.structure_scoring.pyrosetta.shared_data_models import (
    ScoringStructureInput,
    prepare_pdb_and_chain_maps,
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
class RelaxResult(BaseModel):
    """Relaxed structure plus run metadata for a single input.

    Attributes:
        relaxed_structure (Structure): Relaxed coordinates. Drop-in replacement
            for the input: chain labels match (PyRosetta internally shortens
            multi-character chain IDs like ``"Heavy"`` → ``"A"`` for PDB
            compatibility, but the wrapper restores the originals via
            :meth:`Structure.with_renamed_chains` before returning), and the
            source format is preserved (PDB in → PDB out, CIF in → CIF out).
        relax_cycles (int): Number of FastRelax repeats actually applied.
    """

    model_config = ConfigDict(extra="forbid")

    relaxed_structure: Structure = Field(description="Relaxed structure (drop-in replacement for the input)")
    relax_cycles: int = Field(description="Number of FastRelax repeats applied")


class PyRosettaRelaxMetrics(Metrics):
    """FastRelax result for a single structure.

    Metrics documented in ``metric_spec``:
        total_score (float): Total Rosetta energy of the relaxed pose, in REU.
            Computed with the same score function used for the relax minimization.

    Attributes:
        relax (RelaxResult): Relaxed structure and run metadata. Declared as a
            real field (not a metric) because it carries structured data, not
            a scalar.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "total_score": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": None,
            "unit": "REU",
        },
    }
    primary_metric: str | None = "total_score"

    relax: RelaxResult = Field(description="Relaxed structure and run metadata")


class PyRosettaRelaxInput(BaseToolInput):
    """Input for PyRosetta FastRelax.

    Attributes:
        inputs (list[ScoringStructureInput]): Protein structures to relax.
            Accepts bare ``Structure`` objects, PDB file paths, or PDB content
            strings for convenience.
    """

    inputs: list[ScoringStructureInput] = InputField(
        description="Protein structures to relax",
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


class PyRosettaRelaxConfig(BaseConfig):
    """Configuration for PyRosetta FastRelax.

    Attributes:
        scorefxn (str): Rosetta score function name. ``ref2015`` is the current
            community standard.
        relax_cycles (int): Number of FastRelax repeats. Germinal uses ``1`` for
            speed in cofolding filter pipelines; raise for better convergence
            at the cost of runtime.
        constrain_to_start (bool): When ``True``, add a coordinate-constraint
            term to the relax score function and call
            ``constrain_relax_to_start_coords(True)`` on the FastRelax mover so
            atoms stay near their input positions. Recommended for filter use
            cases where large geometric deviations would defeat the purpose.
        max_iter (int | None): Maximum minimizer iterations per relax cycle.
            ``None`` uses PyRosetta's default (2500). Upstream BindCraft uses
            200 for faster turnaround in binder-design pipelines.
        disable_jumps (bool): Lock inter-chain rigid-body DOFs so chains
            cannot translate or rotate relative to each other during relax.
        min_type (str | None): Optional minimizer type forwarded to
            ``FastRelax.min_type``. BindCraft uses
            ``"lbfgs_armijo_nonmonotone"``.
        align_to_start (bool): If ``True``, align the relaxed pose back to the
            input pose after FastRelax. BindCraft does this before saving its
            relaxed PDBs so coordinates remain in the original frame.
        copy_b_factors_from_start (bool): If ``True``, copy the input pose's
            per-residue B-factors onto the relaxed pose. BindCraft uses this
            to preserve AF2 pLDDT values after relaxation.
    """

    scorefxn: str = ConfigField(
        title="Score Function",
        default="ref2015",
        description="Rosetta score function name",
        examples=["ref2015", "beta_nov16", "ref2015_cart"],
    )
    relax_cycles: int = ConfigField(
        title="Relax Cycles",
        default=1,
        ge=1,
        le=15,
        description="Number of FastRelax repeats (more = better convergence, slower)",
    )
    constrain_to_start: bool = ConfigField(
        title="Constrain to Start",
        default=True,
        description="Constrain relaxation to starting coordinates",
    )
    max_iter: int | None = ConfigField(
        default=None,
        ge=1,
        title="Max Iterations",
        description="Maximum minimizer iterations per relax cycle. None uses PyRosetta default (2500).",
    )
    disable_jumps: bool = ConfigField(
        default=False,
        title="Disable Jumps",
        description="Lock inter-chain rigid-body DOFs during relaxation.",
    )
    min_type: str | None = ConfigField(
        default=None,
        title="Minimizer Type",
        description="Optional FastRelax minimizer type.",
    )
    align_to_start: bool = ConfigField(
        default=False,
        title="Align to Start",
        description="Align the relaxed pose back onto the starting pose after FastRelax.",
    )
    copy_b_factors_from_start: bool = ConfigField(
        default=False,
        title="Copy Start B-factors",
        description="Copy per-residue B-factors from the starting pose to the relaxed pose.",
    )

    @property
    def cpus_per_instance(self) -> int | None:
        """Opt in to ToolPool CPU fan-out — PyRosetta runs single-threaded per pose."""
        return 1


class PyRosettaRelaxOutput(BaseToolOutput):
    """Output from PyRosetta FastRelax.

    Attributes:
        results (list[PyRosettaRelaxMetrics]): One entry per input structure,
            in input order. Each carries ``total_score`` + ``relaxed`` as
            specced metrics plus ``relax`` (a :class:`RelaxResult`) carrying
            the relaxed ``Structure``.
    """

    results: list[PyRosettaRelaxMetrics] = Field(
        default_factory=list,
        description="Relax results, one per input structure",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["pdb", "json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "pdb"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        path = Path(export_path)
        if file_format == "pdb":
            # One PDB file per input. For a single input, write `<path>.pdb`;
            # otherwise suffix with the index so callers see distinct files.
            single = len(self.results) == 1
            for i, result in enumerate(self.results):
                target = path.with_suffix(".pdb") if single else path.with_name(f"{path.name}_{i}.pdb")
                result.relax.relaxed_structure.write_pdb(target)
        elif file_format == "json":
            target = path.with_suffix(".json")
            target.write_text(self.model_dump_json(indent=2))
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return PyRosettaRelaxInput(
        inputs=[
            ScoringStructureInput(
                structure=Structure(structure=str(Path(__file__).parent / "examples" / "example.pdb"))
            )
        ]
    )


@tool(
    key="pyrosetta-relax",
    label="PyRosetta FastRelax",
    category="structure_scoring",
    input_class=PyRosettaRelaxInput,
    config_class=PyRosettaRelaxConfig,
    output_class=PyRosettaRelaxOutput,
    metrics_class=PyRosettaRelaxMetrics,
    description="Run PyRosetta FastRelax on a structure and return the relaxed Structure plus its total score",
    uses_gpu=False,
    example_input=example_input,
    iterable_input_field="inputs",
    iterable_output_field="results",
    cacheable=True,
    stochastic=True,
)
def run_pyrosetta_relax(
    inputs: PyRosettaRelaxInput,
    config: PyRosettaRelaxConfig | None = None,
    instance: ToolInstance | None = None,
) -> PyRosettaRelaxOutput:
    """Run FastRelax on protein structures and return the relaxed coordinates.

    Designed for cofolding filter pipelines (Germinal-style binder design)
    where downstream geometric / energetic gates need to operate on a
    relaxed pose. The relaxed Structure chains cleanly into
    ``run_pyrosetta_energy``, ``run_pyrosetta_sap``, ``run_pyrosetta_sasa``,
    or Structure-aware non-PyRosetta tools.

    Args:
        inputs (PyRosettaRelaxInput): Structures to relax.
        config (PyRosettaRelaxConfig | None): Score function + relax knobs.
        instance (ToolInstance | None): Optional ToolInstance for persistent execution.

    Returns:
        PyRosettaRelaxOutput: One :class:`PyRosettaRelaxMetrics` per input,
            carrying ``total_score`` + ``relaxed`` as scalar metrics and the
            relaxed :class:`Structure` as a structured field.
    """
    logger.debug("Using local venv for PyRosetta FastRelax")

    seed = config.seed if config.seed is not None else config.get_random_int()  # type: ignore[union-attr]
    pdb_contents, _, pdb_to_mmcif_maps = prepare_pdb_and_chain_maps(inputs.inputs)

    input_data = {
        "operation": "relax",
        "pdb_contents": pdb_contents,
        # chain_ids_list intentionally omitted: relax operates on the whole pose,
        # not a chain selection. Future: add per-chain MoveMap toggles if needed.
        "scorefxn": config.scorefxn,  # type: ignore[union-attr]
        "relax_cycles": config.relax_cycles,  # type: ignore[union-attr]
        "constrain_to_start": config.constrain_to_start,  # type: ignore[union-attr]
        "max_iter": config.max_iter,  # type: ignore[union-attr]
        "disable_jumps": config.disable_jumps,  # type: ignore[union-attr]
        "min_type": config.min_type,  # type: ignore[union-attr]
        "align_to_start": config.align_to_start,  # type: ignore[union-attr]
        "copy_b_factors_from_start": config.copy_b_factors_from_start,  # type: ignore[union-attr]
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

    results = []
    for raw, pdb_to_mmcif, inp in zip(output_data["results"], pdb_to_mmcif_maps, inputs.inputs, strict=True):
        # PyRosetta emits PDB with shortened single-char chain labels, which
        # with_renamed_chains restores to the originals. That method preserves
        # source format and rejects multi-char targets on PDB Structures, so
        # CIF-originating inputs must be coerced to CIF before the rename.
        target_format = inp.structure.structure_format or "pdb"
        relaxed = Structure(structure=raw["relaxed_pdb"], structure_format="pdb")
        if target_format == "cif":
            relaxed = Structure(structure=relaxed.structure_cif, structure_format="cif")
        relaxed = relaxed.with_renamed_chains(pdb_to_mmcif)
        results.append(
            PyRosettaRelaxMetrics(
                total_score=raw["total_score"],
                relax=RelaxResult(
                    relaxed_structure=relaxed,
                    relax_cycles=raw["relax_cycles"],
                ),
            )
        )

    return PyRosettaRelaxOutput(
        metadata={
            "num_structures": len(inputs.inputs),
            "scorefxn": config.scorefxn,  # type: ignore[union-attr]
            "relax_cycles": config.relax_cycles,  # type: ignore[union-attr]
        },
        results=results,
    )
