"""Tests for StructureEnsemble."""

from pathlib import Path

import pytest

from proto_tools.entities.structures import Structure, StructureEnsemble

_SAMPLE_SEQUENCE = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
_TEST_PDB_FILE = Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb"


@pytest.fixture()
def structure():
    return Structure.from_file(_TEST_PDB_FILE)


def test_ensemble_creation(structure):
    ensemble = StructureEnsemble(structures=[structure] * 5, sequence=_SAMPLE_SEQUENCE)
    assert len(ensemble.structures) == 5
    assert ensemble.sequence == _SAMPLE_SEQUENCE


def test_approx_equal_matching(structure):
    a = StructureEnsemble(structures=[structure, structure], sequence=_SAMPLE_SEQUENCE)
    b = StructureEnsemble(structures=[structure, structure], sequence=_SAMPLE_SEQUENCE)
    a.approx_equal(b)


@pytest.mark.parametrize(
    "a_count,a_seq,b_count,b_seq,match",
    [
        (1, "AAA", 1, "BBB", "sequences differ"),
        (1, _SAMPLE_SEQUENCE, 2, _SAMPLE_SEQUENCE, "structure count differs"),
    ],
    ids=["different-sequences", "different-counts"],
)
def test_approx_equal_mismatch(structure, a_count, a_seq, b_count, b_seq, match):
    a = StructureEnsemble(structures=[structure] * a_count, sequence=a_seq)
    b = StructureEnsemble(structures=[structure] * b_count, sequence=b_seq)
    with pytest.raises(AssertionError, match=match):
        a.approx_equal(b)
