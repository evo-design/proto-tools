"""Tests for StructureEnsemble."""

from unittest.mock import Mock

from bio_programming_tools.entities.structures import Structure, StructureEnsemble

_SAMPLE_SEQUENCE = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"


def test_ensemble_creation():
    """Test creating a StructureEnsemble with mock structures."""
    mock_structure = Mock(spec=Structure)
    ensemble = StructureEnsemble(
        structures=[mock_structure] * 5,
        sequence=_SAMPLE_SEQUENCE,
    )
    assert len(ensemble.structures) == 5
    assert ensemble.sequence == _SAMPLE_SEQUENCE
