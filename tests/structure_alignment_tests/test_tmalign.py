"""Tests for TMalign structure alignment tool."""
from __future__ import annotations

from pathlib import Path

import pytest

from bio_programming_tools.tools.structure_alignment.tmalign import (
    TMalignConfig,
    TMalignInput,
    run_tmalign,
)
from tests.tool_infra_tests.test_export_functionality import validate_output

DUMMY_DATA = Path(__file__).parent.parent / "dummy_data"
PDB_1 = (DUMMY_DATA / "test_structure_similarity.pdb").read_text()
PDB_2 = (DUMMY_DATA / "renin_af3.pdb").read_text()


class TestTMalignIntegration:
    """Integration tests requiring TMalign installation."""

    @pytest.mark.include_in_env_report
    def test_aligns_two_structures(self):
        """Align two different PDB structures and verify TM-scores."""
        inputs = TMalignInput(pdb_text_1=PDB_1, pdb_text_2=PDB_2)
        result = run_tmalign(inputs, TMalignConfig())

        validate_output(result)
        assert 0.0 <= result.tm_score_chain_1 <= 1.0
        assert 0.0 <= result.tm_score_chain_2 <= 1.0

    @pytest.mark.include_in_env_report
    def test_self_alignment_perfect_score(self):
        """Aligning a structure to itself should give TM-score = 1.0."""
        inputs = TMalignInput(pdb_text_1=PDB_1, pdb_text_2=PDB_1)
        result = run_tmalign(inputs, TMalignConfig())

        validate_output(result)
        assert result.tm_score_chain_1 == pytest.approx(1.0, abs=0.01)
        assert result.tm_score_chain_2 == pytest.approx(1.0, abs=0.01)
