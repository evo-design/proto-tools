"""PyRosetta interface analyzer tool, with opt-in FastRelax preprocess."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from proto_tools.entities.structures import Structure
from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_relax import PyRosettaRelaxConfig
from proto_tools.tools.structure_scoring.pyrosetta.shared_data_models import (
    MAX_CHAINS_FOR_PDB,
    ScoringStructureInput,
    relax_inputs_via_pyrosetta,
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
class PyRosettaInterfaceAnalyzerMetrics(Metrics):
    """Interface-analysis result for a single two-chain complex.

    Metrics are computed on the full pose with the target and binder chains
    specified on the corresponding :class:`InterfaceStructureInput`.

    Metrics documented in ``metric_spec``:
        interface_sc (float): Shape complementarity across the interface, on
            a [0, 1] scale (1 = perfect fit).
        interface_hbonds (int): Count of hydrogen bonds across the interface.
        interface_dG (float): Binding ΔG in Rosetta Energy Units (REU).
        interface_dSASA (float): Interface buried SASA in Å².
        interface_packstat (float): Interface packing statistic on a [0, 1]
            scale (higher = better packing).
        interface_hydrophobicity (float): Percentage of interface residues
            that are apolar or aromatic (set ``"ACFILMPVWY"``), in [0, 100].
        surface_hydrophobicity (float): Fraction of binder-chain surface
            residues that are apolar or aromatic, in [0, 1].
        delta_unsat_hbonds (int | None): Buried unsatisfied hydrogen bonds
            across the interface via Rosetta's BuriedUnsatHbonds filter with
            DAlphaBall SASA. ``None`` when DAlphaBall is not installed.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "interface_sc": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "unit": None,
            "better_values_are": "higher",
        },
        "interface_hbonds": {
            "availability": "always",
            "type": "int",
            "min": 0,
            "max": None,
            "unit": "count",
            "better_values_are": "higher",
        },
        "interface_dG": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": None,
            "unit": "REU",
            "better_values_are": "lower",
        },
        "interface_dSASA": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": None,
            "unit": "Å²",
            "better_values_are": "context-dependent",
        },
        "interface_packstat": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "unit": None,
            "better_values_are": "higher",
        },
        "interface_hydrophobicity": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 100.0,
            "unit": "%",
            "better_values_are": "context-dependent",
        },
        "surface_hydrophobicity": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "unit": None,
            "better_values_are": "lower",
        },
        "delta_unsat_hbonds": {
            "availability": "optional",
            "type": "int",
            "min": 0,
            "max": None,
            "unit": "count",
            "better_values_are": "lower",
        },
    }
    primary_metric: str | None = Field(
        default="interface_dG",
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )


class InterfaceStructureInput(BaseModel):
    """A two-chain complex paired with the chain labels that define its interface.

    Chain labels are validated at input-construction time against the structure's
    own chain IDs — passing a label absent from the structure raises
    ``pydantic.ValidationError`` immediately rather than surfacing as a dispatch
    error. Multi-character mmCIF labels (e.g. ``"Heavy"``) are accepted; the
    tool layer shortens them to PDB-compatible single characters before
    dispatching to PyRosetta and reports metrics in a namespace-independent way.

    Attributes:
        structure (Structure): The complex to analyze. Accepts a ``Structure``
            object, a file path, or a PDB/CIF content string.
        target_chains (list[str]): Target-side chain label(s); multiple chains
            score the binder against all of them (binder-vs-rest). Default ``["A"]``.
        binder_chain (str): Binder-side chain label. Default ``"B"``.
    """

    model_config = ConfigDict(extra="forbid")

    structure: Structure = Field(
        title="Complex Structure",
        description="Protein complex (file path, content string, or Structure object).",
    )
    target_chains: list[str] = Field(
        default_factory=lambda: ["A"],
        title="Target Chains",
        description="Target-side chain label(s); multiple = binder-vs-rest.",
    )
    binder_chain: str = Field(
        default="B",
        title="Binder Chain",
        description="Chain label for the binder side of the interface.",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_structure(cls, data: Any) -> Any:
        """Accept a bare ``str``/``Path``/``Structure`` as shorthand for a default-chains complex."""
        if isinstance(data, (str, Path)):
            return {"structure": Structure(structure=str(data))}
        if isinstance(data, Structure):
            return {"structure": data}
        if isinstance(data, dict):
            structure = data.get("structure")
            if isinstance(structure, (str, Path)):
                data = {**data, "structure": Structure(structure=str(structure))}
        return data

    @model_validator(mode="after")
    def _validate_interface_chains(self) -> InterfaceStructureInput:
        """Assert target/binder chains exist, are disjoint, and the structure fits in PDB."""
        if not self.target_chains:
            raise ValueError("target_chains must name at least one chain.")
        if self.binder_chain in self.target_chains:
            raise ValueError(
                f"binder_chain {self.binder_chain!r} cannot also be a target chain ({self.target_chains}).",
            )
        available = set(self.structure.get_chain_ids())
        # PyRosetta runs on PDB-format poses, which cap chain IDs at 62 unique
        # single-character labels. Rejecting up front mirrors ScoringStructureInput.
        if len(available) > MAX_CHAINS_FOR_PDB:
            raise ValueError(
                f"Structure has {len(available)} chains, but PyRosetta scoring "
                f"requires PDB format which supports at most {MAX_CHAINS_FOR_PDB} "
                f"single-character chain IDs.",
            )
        missing = {self.binder_chain, *self.target_chains} - available
        if missing:
            raise ValueError(
                f"Chain(s) {sorted(missing)} not found in structure. Available chains: {sorted(available)}",
            )
        return self


class PyRosettaInterfaceAnalyzerInput(BaseToolInput):
    """Input for PyRosetta interface analysis.

    Attributes:
        inputs (list[InterfaceStructureInput]): Complexes to analyze, each paired
            with the ``target_chains`` and ``binder_chain`` labels that define its
            interface. A bare ``Structure`` / path / content string / dict is
            wrapped into a single-element list with default chains ``["A"]`` / ``"B"``.
    """

    inputs: list[InterfaceStructureInput] = InputField(
        title="Complexes",
        description="Complexes to analyze, each with target_chains/binder_chain labels",
    )

    @field_validator("inputs", mode="before")
    @classmethod
    def normalize_inputs(cls, value: Any) -> Any:
        """Normalize a single structure/input to a list."""
        if isinstance(value, (str, Path, Structure, InterfaceStructureInput)):
            value = [value]
        if isinstance(value, dict):
            value = [value]
        return value


class PyRosettaInterfaceAnalyzerConfig(BaseConfig):
    """Configuration for PyRosetta interface analysis.

    Interface chains are specified on each :class:`InterfaceStructureInput` in
    ``inputs``, not here — this config only carries cross-input settings
    (score function and the optional FastRelax preprocess).

    Attributes:
        scorefxn (str): Rosetta score function name. ``ref2015`` is the current
            community standard.
        pre_relax_structures (bool): If ``True``, run ``pyrosetta-relax`` on
            each input structure before analyzing (settings come from
            :attr:`relax_config`). Default ``False`` — the interface is
            analyzed on the input structure as-given. Set to ``True`` for raw
            predicted complexes with steric clashes that would otherwise
            distort ``interface_dG`` and related energy-based metrics.
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
        description="If True, run pyrosetta-relax on each input structure before analyzing.",
    )
    relax_config: PyRosettaRelaxConfig = ConfigField(
        default_factory=PyRosettaRelaxConfig,
        title="Relax Config",
        description="Settings used when pre_relax_structures=True. Ignored otherwise.",
    )

    def preprocess(self, inputs: PyRosettaInterfaceAnalyzerInput) -> PyRosettaInterfaceAnalyzerInput:  # type: ignore[override]
        """Apply optional FastRelax preprocess to each input structure.

        Runs ``relax_inputs_via_pyrosetta`` on structures stripped into
        :class:`ScoringStructureInput`s, then rebuilds each
        :class:`InterfaceStructureInput` with the relaxed structure and its
        original target/binder chain labels preserved.
        """
        if not self.pre_relax_structures:
            return inputs
        ssi_inputs = [ScoringStructureInput(structure=inp.structure) for inp in inputs.inputs]
        relaxed = relax_inputs_via_pyrosetta(ssi_inputs, self.relax_config)
        return inputs.model_copy(
            update={
                "inputs": [
                    InterfaceStructureInput(
                        structure=relaxed[i].structure,
                        target_chains=inp.target_chains,
                        binder_chain=inp.binder_chain,
                    )
                    for i, inp in enumerate(inputs.inputs)
                ],
            },
        )

    @property
    def cpus_per_instance(self) -> int | None:
        """Opt in to ToolPool CPU fan-out — PyRosetta runs single-threaded per pose."""
        return 1


class PyRosettaInterfaceAnalyzerOutput(BaseToolOutput):
    """Output from PyRosetta interface analysis.

    Attributes:
        results (list[PyRosettaInterfaceAnalyzerMetrics]): Interface-analysis
            metrics, one per input structure.
    """

    results: list[PyRosettaInterfaceAnalyzerMetrics] = Field(
        default_factory=list,
        title="Interface Metrics",
        description="Interface-analysis metrics, one per input structure",
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
        rows = [{"structure_index": i, **dict(result)} for i, result in enumerate(self.results)]
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
    """Minimal valid input for testing and examples.

    Uses a small two-chain complex fixture (chains A + B) shipped at the
    pyrosetta family directory root.
    """
    return PyRosettaInterfaceAnalyzerInput(
        inputs=[
            InterfaceStructureInput(
                structure=Structure(structure=str(Path(__file__).parent / "example_complex.pdb")),
            ),
        ],
    )


@tool(
    key="pyrosetta-interface-analyzer",
    label="PyRosetta Interface Analyzer",
    category="structure_scoring",
    input_class=PyRosettaInterfaceAnalyzerInput,
    config_class=PyRosettaInterfaceAnalyzerConfig,
    output_class=PyRosettaInterfaceAnalyzerOutput,
    metrics_class=PyRosettaInterfaceAnalyzerMetrics,
    description=(
        "Compute interface-quality metrics for a two-chain complex via Rosetta's "
        "InterfaceAnalyzerMover + LayerSelector (with optional FastRelax "
        "preprocess via config.pre_relax_structures)"
    ),
    uses_gpu=False,
    example_input=example_input,
    iterable_input_fields=["inputs"],
    iterable_output_field="results",
    cacheable=True,
    stochastic=True,
)
def run_pyrosetta_interface_analyzer(
    inputs: PyRosettaInterfaceAnalyzerInput,
    config: PyRosettaInterfaceAnalyzerConfig | None = None,
    instance: ToolInstance | None = None,
) -> PyRosettaInterfaceAnalyzerOutput:
    """Compute interface-quality metrics for two-chain complexes using PyRosetta.

    Runs Rosetta's ``InterfaceAnalyzerMover`` to compute shape complementarity,
    interface H-bond count, binding ΔG, buried SASA, and packing statistic;
    computes ``interface_hydrophobicity`` from interface-residue AA composition
    (binder residues with any atom within 4.0 Å of any target atom); and
    computes ``surface_hydrophobicity`` from ``LayerSelector(pick_surface=True)``
    applied to the binder sub-pose.

    The interface for each complex is defined by the ``target_chains`` and
    ``binder_chain`` fields on the corresponding :class:`InterfaceStructureInput`
    (multiple target chains score the binder against all of them, binder-vs-rest).
    Chain-label validity is enforced at input construction. To analyze an
    already-relaxed pose, pass it in directly; to relax-then-analyze in one
    dispatch, set ``config.pre_relax_structures=True``.

    Args:
        inputs (PyRosettaInterfaceAnalyzerInput): Two-chain complexes to analyze,
            each carrying its own target/binder chain labels.
        config (PyRosettaInterfaceAnalyzerConfig | None): Score function + optional
            relax preprocess.
        instance (ToolInstance | None): Optional ToolInstance for persistent execution.

    Returns:
        PyRosettaInterfaceAnalyzerOutput: Interface-analysis metrics, one per input.
    """
    logger.debug("Using local venv for PyRosetta interface analysis")

    pdb_contents: list[str] = []
    target_sides: list[list[str]] = []
    binder_chains: list[str] = []
    for inp in inputs.inputs:
        pdb_content, mmcif_to_pdb = inp.structure.to_pdb_with_chain_mapping()
        pdb_contents.append(pdb_content)
        # PDB-shortened target chains form one binder-vs-rest interface side.
        target_sides.append([mmcif_to_pdb[chain] for chain in inp.target_chains])
        binder_chains.append(mmcif_to_pdb[inp.binder_chain])

    input_data = {
        "operation": "interface_analyzer",
        "pdb_contents": pdb_contents,
        "binder_chains": binder_chains,
        "target_chains": target_sides,
        "scorefxn": config.scorefxn,  # type: ignore[union-attr]
        "seed": config.seed,  # type: ignore[union-attr]
        "device": "cpu",
    }

    output_data = ToolInstance.dispatch(
        "pyrosetta",
        input_data,
        instance=instance,
        config=config,
    )

    warn_about_dropped_residues(output_data["results"])
    results = [PyRosettaInterfaceAnalyzerMetrics(**r) for r in output_data["results"]]

    return PyRosettaInterfaceAnalyzerOutput(
        metadata={
            "num_structures": len(inputs.inputs),
            "scorefxn": config.scorefxn,  # type: ignore[union-attr]
            "pre_relax_structures": config.pre_relax_structures,  # type: ignore[union-attr]
        },
        results=results,
    )
