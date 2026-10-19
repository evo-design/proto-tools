"""Tests for PyRosetta energy scoring tool."""

from pathlib import Path

import pytest

from proto_tools.entities.structures import Structure
from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_energy import (
    PyRosettaEnergyConfig,
    PyRosettaEnergyInput,
    ResidueEnergy,
    run_pyrosetta_energy,
)
from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_sap import PyRosettaSAPInput, run_pyrosetta_sap
from proto_tools.tools.structure_scoring.pyrosetta.shared_data_models import ScoringStructureInput
from proto_tools.utils.tool_instance import ToolInstance
from tests.conftest import benchmark_twice
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output

TEST_PDB = str(Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb")
TEST_CIF_MULTICHAIN = str(Path(__file__).parent.parent / "dummy_data" / "renin.cif")

# Reference totals for TEST_PDB from a standalone PyRosetta process initialized with only the flags each score function needs.
REF2015_TOTAL = -602.176
BETA_NOV16_TOTAL = -373.803

# ── Validation ────────────────────────────────────────────────────────────────


def test_energy_input_normalizes_single_structure():
    structure = Structure(structure=TEST_PDB)
    inp = PyRosettaEnergyInput(inputs=structure)
    assert len(inp.inputs) == 1
    assert isinstance(inp.inputs[0], ScoringStructureInput)


def test_energy_input_rejects_invalid_chain():
    with pytest.raises(ValueError, match="not in structure"):
        PyRosettaEnergyInput(inputs=[{"structure": TEST_PDB, "chains_to_score": ["Z"]}])


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_run_pyrosetta_energy_on_pdb():
    """Default path: no relax preprocess; score the input structure as-given."""
    structure = Structure(structure=TEST_PDB)
    result = run_pyrosetta_energy(PyRosettaEnergyInput(inputs=[structure]))
    assert_metrics_in_spec(result)

    assert result.success
    assert result.tool_id == "pyrosetta-energy"
    assert len(result.results) == 1

    energy = result.results[0]
    # Raw (un-relaxed) AF3 prediction may have steric clashes inflating fa_rep,
    # so the total range is wider than for a relaxed pose.
    assert isinstance(energy.total_energy, float)

    assert isinstance(energy.energy_terms, dict)
    assert len(energy.energy_terms) > 5
    assert "fa_atr" in energy.energy_terms
    assert "fa_rep" in energy.energy_terms
    assert energy.energy_terms["fa_atr"] < 0

    assert len(energy.per_residue) == 340

    residue = energy.per_residue[0]
    assert isinstance(residue, ResidueEnergy)
    assert residue.residue_index >= 1


@pytest.mark.integration
def test_run_pyrosetta_energy_chain_selection_filters_residues():
    """Chain A selection should return fewer residues than whole complex."""
    from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_energy import (
        PyRosettaEnergyConfig,
    )

    no_relax = PyRosettaEnergyConfig()
    whole = run_pyrosetta_energy(PyRosettaEnergyInput(inputs=[TEST_CIF_MULTICHAIN]), no_relax)
    chain_a = run_pyrosetta_energy(
        PyRosettaEnergyInput(inputs=[{"structure": TEST_CIF_MULTICHAIN, "chains_to_score": ["A"]}]),
        no_relax,
    )

    assert whole.success and chain_a.success

    whole_res = whole.results[0]
    chain_a_res = chain_a.results[0]

    assert len(chain_a_res.per_residue) < len(whole_res.per_residue), (
        "Chain A should have fewer residues than whole complex"
    )
    assert all(r.chain_id == "A" for r in chain_a_res.per_residue), "All residues should be from chain A"


@pytest.mark.integration
def test_run_pyrosetta_energy_with_pre_relax_preprocess():
    """Setting pre_relax_structures=True should run pyrosetta-relax before scoring.

    The relaxed pose's total energy should differ meaningfully from the raw
    (un-relaxed) pose, since FastRelax resolves steric clashes that inflate fa_rep.
    """
    from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_energy import (
        PyRosettaEnergyConfig,
    )
    from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_relax import (
        PyRosettaRelaxConfig,
    )

    structure = Structure(structure=TEST_PDB)

    raw = run_pyrosetta_energy(PyRosettaEnergyInput(inputs=[structure]))
    relaxed = run_pyrosetta_energy(
        PyRosettaEnergyInput(inputs=[structure]),
        PyRosettaEnergyConfig(
            pre_relax_structures=True,
            relax_config=PyRosettaRelaxConfig(relax_cycles=1, seed=42),
        ),
    )

    assert raw.success and relaxed.success
    # Relax should drop the energy meaningfully (>1 REU) by resolving clashes.
    assert relaxed.results[0].total_energy < raw.results[0].total_energy - 1.0, (
        f"Relaxed energy {relaxed.results[0].total_energy} should be lower than raw {raw.results[0].total_energy}"
    )


@pytest.mark.integration
def test_ref2015_energy_matches_reference_rosetta_init():
    """ref2015 scores with REF2015's database parameters, untouched by the beta_nov16 corrections."""
    result = run_pyrosetta_energy(PyRosettaEnergyInput(inputs=[TEST_PDB]))
    assert result.results[0].total_energy == pytest.approx(REF2015_TOTAL, abs=0.01)


@pytest.mark.integration
def test_persistent_worker_scores_each_scorefxn_with_matching_corrections():
    """One persistent worker serves beta_nov16, SAP, and ref2015 calls in any order with correct energies."""
    inputs = PyRosettaEnergyInput(inputs=[TEST_PDB])
    beta = PyRosettaEnergyConfig(scorefxn="beta_nov16")
    with ToolInstance.persist_tool("pyrosetta"):
        beta_first = run_pyrosetta_energy(inputs, beta).results[0].total_energy
        assert run_pyrosetta_sap(PyRosettaSAPInput(inputs=[TEST_PDB])).success
        ref2015 = run_pyrosetta_energy(inputs, PyRosettaEnergyConfig()).results[0].total_energy
        beta_again = run_pyrosetta_energy(inputs, beta).results[0].total_energy

    assert ref2015 == pytest.approx(REF2015_TOTAL, abs=0.01)
    assert beta_first == pytest.approx(BETA_NOV16_TOTAL, abs=0.01)
    assert beta_again == pytest.approx(BETA_NOV16_TOTAL, abs=0.01)


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("pyrosetta-energy")
@pytest.mark.slow
def test_pyrosetta_energy_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark pyrosetta-energy: 5 distinct renin_af3 copies (~340 aa each), no preprocess relax (cold + warm)."""
    structures = [Structure(structure=TEST_PDB, metrics={"_bench_id": i}) for i in range(5)]
    inputs = PyRosettaEnergyInput(inputs=structures)

    result = benchmark_twice(request, "pyrosetta", lambda: run_pyrosetta_energy(inputs))
    validate_output(result)

    assert result.tool_id == "pyrosetta-energy"
    assert len(result.results) == 5
    for r in result.results:
        assert isinstance(r.total_energy, float)
