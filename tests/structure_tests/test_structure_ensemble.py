"""
Unit tests for StructureEnsemble class.
"""
from unittest.mock import Mock

import pytest

from bio_programming_tools.entities.structures import Structure, StructureEnsemble


@pytest.fixture
def sample_sequence():
    """A short valid protein sequence for testing."""
    return "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"

@pytest.fixture
def sample_pdb_content():
    """Minimal valid PDB content for testing."""
    return """ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  MET A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   MET A   1       2.009   1.420   0.000  1.00  0.00           C
END
"""


class TestStructureEnsemble:
    """Tests for StructureEnsemble output schema."""

    def test_creation(self, sample_pdb_content, sample_sequence):
        """Test creating a StructureEnsemble."""
        mock_structure = Mock(spec=Structure)

        ensemble = StructureEnsemble(
            structures=[mock_structure] * 5,
            sequence=sample_sequence,
        )

        assert len(ensemble.structures) == 5
        assert ensemble.sequence == sample_sequence
