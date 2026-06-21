"""tests/structure_alignment_tests/test_tmalign.py.

Tests for TMalign structure alignment tool.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from proto_tools.tools.structure_alignment.tmalign import (
    TMalignConfig,
    TMalignInput,
    run_tmalign,
)
from tests.conftest import benchmark_twice
from tests.tool_infra_tests.test_export_functionality import validate_output

_DUMMY_DATA = Path(__file__).parent.parent / "dummy_data"
_PDB_1_PATH = _DUMMY_DATA / "test_structure_similarity.pdb"
_PDB_2_PATH = _DUMMY_DATA / "renin_af3.pdb"
_MINIMAL_PDB = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n"


# ── Validation ────────────────────────────────────────────────────────────────


def test_tmalign_input_rejects_missing_query_structure():
    with pytest.raises(ValidationError, match="query_structure"):
        TMalignInput(reference_structure=_MINIMAL_PDB)


def test_tmalign_input_rejects_missing_reference_structure():
    with pytest.raises(ValidationError, match="reference_structure"):
        TMalignInput(query_structure=_MINIMAL_PDB)


def test_tmalign_input_rejects_extra_fields():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TMalignInput(query_structure=_MINIMAL_PDB, reference_structure=_MINIMAL_PDB, extra_field="x")


# ---------------------------------------------------------------------------
# Integration tests


@pytest.mark.integration
def test_tmalign_aligns_two_structures():
    """Align two different PDB structures and verify TM-scores."""
    pdb_1 = _PDB_1_PATH.read_text()
    pdb_2 = _PDB_2_PATH.read_text()
    inputs = TMalignInput(query_structure=pdb_1, reference_structure=pdb_2)
    result = run_tmalign(inputs, TMalignConfig())

    validate_output(result)
    assert 0.0 <= result.tm_score_chain_1 <= 1.0
    assert 0.0 <= result.tm_score_chain_2 <= 1.0
    # The superposition transform is parsed from the ``-m`` matrix the binary writes.
    assert result.superposition is not None
    assert len(result.superposition.rotation) == 3
    assert all(len(row) == 3 for row in result.superposition.rotation)
    assert len(result.superposition.translation) == 3


@pytest.mark.integration
def test_tmalign_self_alignment_perfect_score():
    """Aligning a structure to itself gives TM-score = 1.0 and an ~identity superposition."""
    pdb_1 = _PDB_1_PATH.read_text()
    inputs = TMalignInput(query_structure=pdb_1, reference_structure=pdb_1)
    result = run_tmalign(inputs, TMalignConfig())

    validate_output(result)
    assert result.tm_score_chain_1 == pytest.approx(1.0, abs=0.01)
    assert result.tm_score_chain_2 == pytest.approx(1.0, abs=0.01)
    # Self-alignment ⇒ ~identity rotation and ~zero translation.
    assert result.superposition is not None
    for i, row in enumerate(result.superposition.rotation):
        for j, value in enumerate(row):
            assert value == pytest.approx(1.0 if i == j else 0.0, abs=1e-3)
    assert result.superposition.translation == pytest.approx([0.0, 0.0, 0.0], abs=1e-3)


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("tmalign-alignment")
@pytest.mark.slow
def test_tmalign_alignment_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark tmalign-alignment: 20 sequential alignments of pdl1 vs renin_af3 (~340 aa) (cold + warm)."""
    pdb_1 = _PDB_1_PATH.read_text()
    pdb_2 = _PDB_2_PATH.read_text()
    inputs = TMalignInput(query_structure=pdb_1, reference_structure=pdb_2)
    config = TMalignConfig()

    def run_batch():
        last = None
        for _ in range(20):
            last = run_tmalign(inputs, config)
        return last

    result = benchmark_twice(request, "tmalign", run_batch)
    validate_output(result)

    assert result.tool_id == "tmalign-alignment"
    assert 0.0 <= result.tm_score_chain_1 <= 1.0
