"""tests/structure_alignment_tests/test_tmalign.py

Tests for TMalign structure alignment tool."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from bio_programming_tools.tools.structure_alignment.tmalign import (
    TMalignConfig,
    TMalignInput,
    run_tmalign,
)
from tests.tool_infra_tests.test_export_functionality import validate_output

_DUMMY_DATA = Path(__file__).parent.parent / "dummy_data"
_PDB_1_PATH = _DUMMY_DATA / "test_structure_similarity.pdb"
_PDB_2_PATH = _DUMMY_DATA / "renin_af3.pdb"
_MINIMAL_PDB = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n"


# ── Validation ────────────────────────────────────────────────────────────────


def test_tmalign_input_rejects_missing_pdb_text_1():
    with pytest.raises(ValidationError, match="pdb_text_1"):
        TMalignInput(pdb_text_2=_MINIMAL_PDB)


def test_tmalign_input_rejects_missing_pdb_text_2():
    with pytest.raises(ValidationError, match="pdb_text_2"):
        TMalignInput(pdb_text_1=_MINIMAL_PDB)


def test_tmalign_input_rejects_extra_fields():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TMalignInput(pdb_text_1=_MINIMAL_PDB, pdb_text_2=_MINIMAL_PDB, extra_field="x")


# ---------------------------------------------------------------------------
# Integration tests


@pytest.mark.integration
@pytest.mark.include_in_env_report(category="structure_alignment")
def test_tmalign_aligns_two_structures():
    """Align two different PDB structures and verify TM-scores."""
    pdb_1 = _PDB_1_PATH.read_text()
    pdb_2 = _PDB_2_PATH.read_text()
    inputs = TMalignInput(pdb_text_1=pdb_1, pdb_text_2=pdb_2)
    result = run_tmalign(inputs, TMalignConfig())

    validate_output(result)
    assert 0.0 <= result.tm_score_chain_1 <= 1.0
    assert 0.0 <= result.tm_score_chain_2 <= 1.0


@pytest.mark.integration
@pytest.mark.include_in_env_report(category="structure_alignment")
def test_tmalign_self_alignment_perfect_score():
    """Aligning a structure to itself should give TM-score = 1.0."""
    pdb_1 = _PDB_1_PATH.read_text()
    inputs = TMalignInput(pdb_text_1=pdb_1, pdb_text_2=pdb_1)
    result = run_tmalign(inputs, TMalignConfig())

    validate_output(result)
    assert result.tm_score_chain_1 == pytest.approx(1.0, abs=0.01)
    assert result.tm_score_chain_2 == pytest.approx(1.0, abs=0.01)
