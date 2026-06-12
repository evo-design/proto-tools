"""tests/structure_alignment_tests/test_foldseek_rbh.py.

Tests for foldseek-rbh (local-only reciprocal-best-hits structural search).
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from proto_tools.tools.structure_alignment import (
    FoldseekRBHConfig,
    FoldseekRBHInput,
    run_foldseek_rbh,
)
from tests.conftest import benchmark_twice
from tests.tool_infra_tests.test_export_functionality import validate_output

_TINY_PDB = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
_FIXTURES = Path(__file__).parent.parent / "dummy_data"


# ── run_foldseek_rbh (mocked dispatch) ───────────────────────────────────────


def test_run_foldseek_rbh_dispatches_easy_rbh():
    """The wrapper sends operation=easy_rbh + structure_text + local_db; output is parsed FoldseekHits."""
    inputs = FoldseekRBHInput(structure=_TINY_PDB)
    config = FoldseekRBHConfig(local_db="/path/to/db", sensitivity=7.0, num_threads=8)

    with patch("proto_tools.tools.structure_alignment.foldseek.foldseek_rbh.ToolInstance.dispatch") as mock_dispatch:
        mock_dispatch.return_value = {"stdout": "query\t1abc_A\t75.0\t100\t5\t1\t1\t100\t10\t110\t1e-30\t150.0\n"}
        output = run_foldseek_rbh(inputs, config)

    assert output.success
    assert output.num_hits == 1
    assert output.hits[0].target_id == "1abc_A"
    assert output.target_db == "/path/to/db"

    payload = mock_dispatch.call_args.args[1]
    assert payload == {
        "operation": "easy_rbh",
        "structure_text": _TINY_PDB,
        "local_db": "/path/to/db",
        "evalue": 10.0,  # default
        "sensitivity": 7.0,  # user-supplied
        "max_seqs": 1000,  # default
        "alignment_type": 2,  # default 3Di+AA
        "cov": 0.0,
        "cov_mode": 0,
        "tmscore_threshold": 0.0,
        "lddt_threshold": 0.0,
        "num_threads": 8,
        "use_gpu": False,
        "device": "cpu",
    }


def test_run_foldseek_rbh_normalizes_pident_to_fraction():
    """The standalone forces pident (0-100); the parser normalizes to [0, 1]."""
    inputs = FoldseekRBHInput(structure=_TINY_PDB)
    config = FoldseekRBHConfig(local_db="/path/to/db")

    with patch("proto_tools.tools.structure_alignment.foldseek.foldseek_rbh.ToolInstance.dispatch") as mock_dispatch:
        # pident=85.5 (0-100 scale, as the standalone forces with --format-output pident,...)
        mock_dispatch.return_value = {"stdout": "query\t2xyz_A\t85.5\t100\t5\t1\t1\t100\t10\t110\t1e-30\t200.0\n"}
        output = run_foldseek_rbh(inputs, config)

    assert output.hits[0].sequence_identity == pytest.approx(0.855)


def test_foldseek_rbh_config_requires_local_db():
    """Pydantic rejects FoldseekRBHConfig without local_db at construction time (no remote analog)."""
    with pytest.raises(ValidationError, match="local_db is required"):
        FoldseekRBHConfig()


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_foldseek_rbh_end_to_end_with_directory_target_db(tmp_path):
    """Local end-to-end: RBH the renin fixture against a tmp directory of PDB targets (self-match guaranteed)."""
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    (target_dir / "renin.pdb").write_text((_FIXTURES / "renin_af3.pdb").read_text())
    (target_dir / "test_struct.pdb").write_text((_FIXTURES / "test_structure_similarity.pdb").read_text())

    output = run_foldseek_rbh(
        FoldseekRBHInput(structure=(_FIXTURES / "renin_af3.pdb").read_text()),
        FoldseekRBHConfig(local_db=str(target_dir), num_threads=2),
    )

    assert output.success, f"errors: {output.errors}"
    assert output.target_db == str(target_dir)
    assert output.num_hits >= 1, "self-RBH must produce at least the self-match"
    # The self-match must appear with near-identical sequence_identity.
    self_hits = [h for h in output.hits if "renin" in h.target_id.lower()]
    assert self_hits, f"renin self-match missing; got {[h.target_id for h in output.hits]}"
    assert self_hits[0].sequence_identity > 0.99
    # Schema invariants on every hit.
    for hit in output.hits:
        assert 0.0 <= hit.sequence_identity <= 1.0
        assert hit.alignment_length >= 0
        assert hit.evalue >= 0.0


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("foldseek-rbh")
@pytest.mark.slow
def test_foldseek_rbh_benchmark(request: pytest.FixtureRequest, tmp_path) -> None:
    """Benchmark foldseek-rbh: 8 sequential RBH searches of renin (~340 aa) vs a 30-structure target DB (cold + warm)."""
    src = [
        (_FIXTURES / "renin_af3.pdb").read_text(),
        (_FIXTURES / "pdl1.pdb").read_text(),
        (_FIXTURES / "test_structure_similarity.pdb").read_text(),
    ]
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    for i in range(30):
        (target_dir / f"target_{i}.pdb").write_text(src[i % len(src)])

    inputs = FoldseekRBHInput(structure=src[0])
    config = FoldseekRBHConfig(local_db=str(target_dir), num_threads=4)

    def run_batch():
        last = None
        for _ in range(8):
            last = run_foldseek_rbh(inputs, config)
        return last

    result = benchmark_twice(request, "foldseek", run_batch)
    validate_output(result)

    assert result.tool_id == "foldseek-rbh"
    assert result.target_db == str(target_dir)
    assert result.num_hits == len(result.hits)
    assert result.num_hits >= 1, "self-RBH must produce at least the renin self-match"
    assert max(h.sequence_identity for h in result.hits) > 0.99, "renin self-match should be present"
    for hit in result.hits:
        assert 0.0 <= hit.sequence_identity <= 1.0
        assert hit.alignment_length >= 0
        assert hit.evalue >= 0.0
