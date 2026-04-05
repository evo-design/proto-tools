"""tests/structure_tests/test_structure_ensemble.py.

Tests for StructureEnsemble.
"""

from pathlib import Path

from proto_tools.entities.structures import Structure, StructureEnsemble

_SAMPLE_SEQUENCE = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
_TEST_PDB_FILE = Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb"


def test_ensemble_creation():
    """Test creating a StructureEnsemble with real structures."""
    structure = Structure.from_file(_TEST_PDB_FILE)
    ensemble = StructureEnsemble(
        structures=[structure] * 5,
        sequence=_SAMPLE_SEQUENCE,
    )
    assert len(ensemble.structures) == 5
    assert ensemble.sequence == _SAMPLE_SEQUENCE
