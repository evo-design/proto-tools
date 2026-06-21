"""tests/structure_alignment_tests/test_usalign.py.

Tests for USalign structure alignment tool.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from proto_tools.tools.structure_alignment.usalign import (
    USalignConfig,
    USalignInput,
    run_usalign,
)
from tests.conftest import benchmark_twice
from tests.tool_infra_tests.test_export_functionality import validate_output

_DUMMY_DATA = Path(__file__).parent.parent / "dummy_data"
_PDB_1_PATH = _DUMMY_DATA / "test_structure_similarity.pdb"
_PDB_2_PATH = _DUMMY_DATA / "renin_af3.pdb"
_MINIMAL_PDB = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n"


# ── Validation ────────────────────────────────────────────────────────────────


def test_usalign_input_rejects_missing_query_structure():
    with pytest.raises(ValidationError, match="query_structure"):
        USalignInput(reference_structure=_MINIMAL_PDB)


def test_usalign_input_rejects_missing_reference_structure():
    with pytest.raises(ValidationError, match="reference_structure"):
        USalignInput(query_structure=_MINIMAL_PDB)


def test_usalign_input_rejects_extra_fields():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        USalignInput(query_structure=_MINIMAL_PDB, reference_structure=_MINIMAL_PDB, extra_field="x")


# ---------------------------------------------------------------------------
# Integration tests


@pytest.mark.integration
def test_usalign_aligns_two_structures():
    """Align two different PDB structures and verify TM-scores."""
    pdb_1 = _PDB_1_PATH.read_text()
    pdb_2 = _PDB_2_PATH.read_text()
    inputs = USalignInput(query_structure=pdb_1, reference_structure=pdb_2)
    result = run_usalign(inputs, USalignConfig())

    validate_output(result)
    assert 0.0 <= result.tm_score_structure_1 <= 1.0
    assert 0.0 <= result.tm_score_structure_2 <= 1.0
    # The superposition transform is parsed from the ``-m`` matrix the binary writes.
    assert result.superposition is not None
    assert len(result.superposition.rotation) == 3
    assert all(len(row) == 3 for row in result.superposition.rotation)
    assert len(result.superposition.translation) == 3


@pytest.mark.integration
def test_usalign_self_alignment_perfect_score():
    """Aligning a structure to itself gives TM-score = 1.0 and an ~identity superposition."""
    pdb_1 = _PDB_1_PATH.read_text()
    inputs = USalignInput(query_structure=pdb_1, reference_structure=pdb_1)
    result = run_usalign(inputs, USalignConfig())

    validate_output(result)
    assert result.tm_score_structure_1 == pytest.approx(1.0, abs=0.01)
    assert result.tm_score_structure_2 == pytest.approx(1.0, abs=0.01)
    # Self-alignment ⇒ ~identity rotation and ~zero translation.
    assert result.superposition is not None
    for i, row in enumerate(result.superposition.rotation):
        for j, value in enumerate(row):
            assert value == pytest.approx(1.0 if i == j else 0.0, abs=1e-3)
    assert result.superposition.translation == pytest.approx([0.0, 0.0, 0.0], abs=1e-3)


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("usalign-alignment")
@pytest.mark.slow
def test_usalign_alignment_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark usalign-alignment: 20 sequential alignments of pdl1 vs renin_af3 (~340 aa) (cold + warm)."""
    pdb_1 = _PDB_1_PATH.read_text()
    pdb_2 = _PDB_2_PATH.read_text()
    inputs = USalignInput(query_structure=pdb_1, reference_structure=pdb_2)
    config = USalignConfig()

    def run_batch():
        last = None
        for _ in range(20):
            last = run_usalign(inputs, config)
        return last

    result = benchmark_twice(request, "usalign", run_batch)
    validate_output(result)

    assert result.tool_id == "usalign-alignment"
    assert 0.0 <= result.tm_score_structure_1 <= 1.0
