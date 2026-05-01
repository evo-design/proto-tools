"""Tests for PyRosetta energy scoring tool."""

from pathlib import Path

import pytest

from proto_tools.entities.structures import Structure
from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_energy import (
    PyRosettaEnergyInput,
    ResidueEnergy,
    run_pyrosetta_energy,
)
from proto_tools.tools.structure_scoring.pyrosetta.shared_data_models import ScoringStructureInput
from tests.conftest import benchmark_twice
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output

TEST_PDB = str(Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb")
TEST_CIF_MULTICHAIN = str(Path(__file__).parent.parent / "dummy_data" / "renin.cif")

# ── Validation ────────────────────────────────────────────────────────────────


def test_energy_input_normalizes_single_structure():
    structure = Structure(structure=TEST_PDB)
    inp = PyRosettaEnergyInput(inputs=structure)
    assert len(inp.inputs) == 1
    assert isinstance(inp.inputs[0], ScoringStructureInput)


def test_energy_input_rejects_invalid_chain():
    with pytest.raises(ValueError, match="not found in structure"):
        PyRosettaEnergyInput(inputs=[{"structure": TEST_PDB, "chain_ids": ["Z"]}])


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
        PyRosettaEnergyInput(inputs=[{"structure": TEST_CIF_MULTICHAIN, "chain_ids": ["A"]}]),
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


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("pyrosetta-energy")
@pytest.mark.slow
def test_pyrosetta_energy_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark pyrosetta-energy: 5 distinct renin_af3 copies (~340 aa each), no preprocess relax (cold + warm)."""
    # Distinct metrics break the iterable-input dedup so all 5 copies actually compute.
    structures = [Structure(structure=TEST_PDB, metrics={"_bench_id": i}) for i in range(5)]
    inputs = PyRosettaEnergyInput(inputs=structures)

    result = benchmark_twice(request, "pyrosetta", lambda: run_pyrosetta_energy(inputs))
    validate_output(result)

    assert result.tool_id == "pyrosetta-energy"
    assert len(result.results) == 5
    for r in result.results:
        assert isinstance(r.total_energy, float)
