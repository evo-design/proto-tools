"""Tests for PyRosetta SASA scoring tool."""

from pathlib import Path

import pytest

from proto_tools.entities.structures import Structure
from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_sasa import (
    PyRosettaSASAInput,
    ResidueSASA,
    run_pyrosetta_sasa,
)
from proto_tools.tools.structure_scoring.pyrosetta.shared_data_models import ScoringStructureInput
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec

TEST_PDB = str(Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb")
TEST_CIF_MULTICHAIN = str(Path(__file__).parent.parent / "dummy_data" / "renin.cif")

# ── Validation ────────────────────────────────────────────────────────────────


def test_sasa_input_normalizes_single_structure():
    structure = Structure(structure=TEST_PDB)
    inp = PyRosettaSASAInput(inputs=structure)
    assert len(inp.inputs) == 1
    assert isinstance(inp.inputs[0], ScoringStructureInput)


def test_sasa_input_rejects_invalid_chain():
    with pytest.raises(ValueError, match="not found in structure"):
        PyRosettaSASAInput(inputs=[{"structure": TEST_PDB, "chain_ids": ["Z"]}])


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_run_pyrosetta_sasa_on_pdb():
    structure = Structure(structure=TEST_PDB)
    result = run_pyrosetta_sasa(PyRosettaSASAInput(inputs=[structure]))
    assert_metrics_in_spec(result)

    assert result.success
    assert result.tool_id == "pyrosetta-sasa"
    assert len(result.results) == 1

    sasa = result.results[0]
    assert 5_000 <= sasa.total_sasa <= 25_000, f"Total SASA {sasa.total_sasa} outside expected range"
    assert len(sasa.per_residue) == 340

    residue = sasa.per_residue[0]
    assert isinstance(residue, ResidueSASA)
    assert residue.residue_index >= 1
    assert 0 <= residue.sasa <= 300, f"Residue SASA {residue.sasa} outside expected range"


@pytest.mark.integration
def test_run_pyrosetta_sasa_chain_selection_filters_residues():
    """Chain A selection should return fewer residues and different SASA than whole complex."""
    whole = run_pyrosetta_sasa(PyRosettaSASAInput(inputs=[TEST_CIF_MULTICHAIN]))
    chain_a = run_pyrosetta_sasa(PyRosettaSASAInput(inputs=[{"structure": TEST_CIF_MULTICHAIN, "chain_ids": ["A"]}]))

    assert whole.success and chain_a.success

    whole_res = whole.results[0]
    chain_a_res = chain_a.results[0]

    assert len(chain_a_res.per_residue) < len(whole_res.per_residue), (
        "Chain A should have fewer residues than whole complex"
    )
    assert all(r.chain_id == "A" for r in chain_a_res.per_residue), "All residues should be from chain A"
    assert chain_a_res.total_sasa < whole_res.total_sasa, "Chain A SASA should be less than whole complex SASA"
