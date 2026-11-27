"""Tests for PyRosetta SAP scoring tool."""

from pathlib import Path

import pytest

from proto_tools.entities.structures import Structure
from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_sap import (
    PyRosettaSAPInput,
    ResidueSAP,
    run_pyrosetta_sap,
)
from proto_tools.tools.structure_scoring.pyrosetta.shared_data_models import ScoringStructureInput
from tests.conftest import benchmark_twice
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output

TEST_PDB = str(Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb")
TEST_CIF_MULTICHAIN = str(Path(__file__).parent.parent / "dummy_data" / "renin.cif")

# ── Validation ────────────────────────────────────────────────────────────────


def test_sap_input_normalizes_single_structure():
    structure = Structure(structure=TEST_PDB)
    inp = PyRosettaSAPInput(inputs=structure)
    assert len(inp.inputs) == 1
    assert isinstance(inp.inputs[0], ScoringStructureInput)


def test_sap_input_accepts_bare_path():
    inp = PyRosettaSAPInput(inputs=TEST_PDB)
    assert len(inp.inputs) == 1
    assert isinstance(inp.inputs[0].structure, Structure)


def test_sap_input_accepts_dict_with_chains_to_score():
    inp = PyRosettaSAPInput(inputs=[{"structure": TEST_PDB, "chains_to_score": ["A"]}])
    assert inp.inputs[0].chains_to_score is not None
    assert inp.inputs[0].chains_to_score.chains == ["A"]


def test_sap_input_rejects_invalid_chain():
    with pytest.raises(ValueError, match="not in structure"):
        PyRosettaSAPInput(inputs=[{"structure": TEST_PDB, "chains_to_score": ["Z"]}])


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_run_pyrosetta_sap_on_pdb():
    structure = Structure(structure=TEST_PDB)
    result = run_pyrosetta_sap(PyRosettaSAPInput(inputs=[structure]))
    assert_metrics_in_spec(result)

    assert result.success
    assert result.tool_id == "pyrosetta-sap"
    assert len(result.results) == 1

    sap = result.results[0]
    assert isinstance(sap.sap_score, float)
    assert sap.sap_score <= 300, f"SAP score {sap.sap_score} outside expected range"

    assert len(sap.per_residue) == 340
    residue = sap.per_residue[0]
    assert isinstance(residue, ResidueSAP)
    assert residue.residue_index >= 1
    assert isinstance(residue.sap_score, float)


@pytest.mark.integration
def test_run_pyrosetta_sap_chain_selection_changes_score():
    """Chain A score should differ from whole-complex score on a multi-chain structure."""
    whole = run_pyrosetta_sap(PyRosettaSAPInput(inputs=[TEST_CIF_MULTICHAIN]))
    chain_a = run_pyrosetta_sap(
        PyRosettaSAPInput(inputs=[{"structure": TEST_CIF_MULTICHAIN, "chains_to_score": ["A"]}])
    )

    assert whole.success and chain_a.success
    assert whole.results[0].sap_score != chain_a.results[0].sap_score, (
        "Chain A SAP should differ from whole-complex SAP"
    )
    assert len(chain_a.results[0].per_residue) < len(whole.results[0].per_residue), (
        "Chain A should have fewer per-residue entries than whole complex"
    )
    assert all(r.chain_id == "A" for r in chain_a.results[0].per_residue), (
        "All per-residue entries should be from chain A"
    )


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("pyrosetta-sap")
@pytest.mark.slow
def test_pyrosetta_sap_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark pyrosetta-sap: 5 distinct renin_af3 copies (~340 aa each) (cold + warm)."""
    structures = [Structure(structure=TEST_PDB, metrics={"_bench_id": i}) for i in range(5)]
    inputs = PyRosettaSAPInput(inputs=structures)

    result = benchmark_twice(request, "pyrosetta", lambda: run_pyrosetta_sap(inputs))
    validate_output(result)

    assert result.tool_id == "pyrosetta-sap"
    assert len(result.results) == 5
    for r in result.results:
        assert isinstance(r.sap_score, float)
