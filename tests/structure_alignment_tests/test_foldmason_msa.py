"""tests/structure_alignment_tests/test_foldmason_msa.py.

Tests for foldmason-msa (remote + local multiple structure alignment).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from proto_tools.tools.structure_alignment import (
    FoldmasonMSAConfig,
    FoldmasonMSAInput,
    run_foldmason_msa,
)
from proto_tools.tools.structure_alignment.foldmason.foldmason_msa import (
    _msa_dimensions,
    _parse_msa_response_json,
)
from tests.conftest import benchmark_twice
from tests.tool_infra_tests.test_export_functionality import validate_output

_TINY_PDB = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
_FIXTURES = Path(__file__).parent.parent / "dummy_data"


# ── _parse_msa_response_json ─────────────────────────────────────────────────


def test_parse_msa_response_json_builds_fastas_and_tree_from_entries():
    """Server JSON entries become FASTAs ordered by entry order; tree passes through."""
    payload = {
        "entries": [
            {"name": "alpha", "aa": "MQR-K", "ss": "DLDLD", "ca": "..."},
            {"name": "beta", "aa": "MQR-K", "ss": "DLDLD", "ca": "..."},
            {"name": "gamma", "aa": "M--EK", "ss": "D--LD", "ca": "..."},
        ],
        "tree": "((alpha,beta),gamma);",
        "scores": [0.8, 0.8, 0.6, 0.5, 0.7],
        "statistics": {"msaLDDT": 0.68},
    }

    aa_fasta, three_di_fasta, newick, num_seqs, aln_len = _parse_msa_response_json(payload)

    assert num_seqs == 3
    assert aln_len == 5
    assert ">alpha" in aa_fasta and "MQR-K" in aa_fasta
    assert ">beta" in aa_fasta
    assert ">gamma" in aa_fasta and "M--EK" in aa_fasta
    assert ">alpha" in three_di_fasta and "DLDLD" in three_di_fasta
    assert newick == "((alpha,beta),gamma);"


# ── _msa_dimensions (FASTA helper) ───────────────────────────────────────────


def test_msa_dimensions_handles_single_line_records():
    """Single-line FASTA: count headers + length of first sequence."""
    fasta = ">a\nMKRY\n>b\nMKRY\n"
    assert _msa_dimensions(fasta) == (2, 4)


def test_msa_dimensions_handles_multiline_records():
    """Multi-line FASTA records sum across all lines of the first record."""
    fasta = ">a\nMKR\nY\n>b\nMKRY\n"
    assert _msa_dimensions(fasta) == (2, 4)


def test_msa_dimensions_handles_empty_input():
    """Empty FASTA → (0, 0)."""
    assert _msa_dimensions("") == (0, 0)


# ── run_foldmason_msa (local, mocked dispatch) ───────────────────────────────


def test_local_mode_dispatches_easy_msa():
    """search_mode='local' dispatches operation=easy_msa and parses the standalone output."""
    inputs = FoldmasonMSAInput(structures=[_TINY_PDB, _TINY_PDB])
    config = FoldmasonMSAConfig(search_mode="local", gap_open=15, num_threads=8)

    with patch("proto_tools.tools.structure_alignment.foldmason.foldmason_msa.ToolInstance.dispatch") as mock_dispatch:
        mock_dispatch.return_value = {
            "aa_msa_fasta": ">structure_0\nMK\n>structure_1\nMK\n",
            "three_di_msa_fasta": ">structure_0\nDD\n>structure_1\nDD\n",
            "newick_tree": "(structure_0,structure_1);\n",
        }
        output = run_foldmason_msa(inputs, config)

    assert output.success
    assert output.ticket_id == ""  # local mode
    assert output.result_url == ""
    assert output.num_sequences == 2
    assert output.alignment_length == 2
    assert "((structure_0,structure_1));" not in output.newick_tree  # sanity
    assert output.newick_tree.strip() == "(structure_0,structure_1);"

    payload = mock_dispatch.call_args.args[1]
    assert payload["operation"] == "easy_msa"
    assert payload["structures"] == [_TINY_PDB, _TINY_PDB]
    assert payload["structure_ids"] == ["structure_0", "structure_1"]
    assert payload["gap_open"] == 15  # user override
    assert payload["gap_extend"] == 1  # default
    assert payload["refine_iters"] == 0
    assert payload["precluster"] is False
    assert payload["guide_tree_newick"] is None
    assert payload["num_threads"] == 8


def test_local_mode_accepts_structure_objects_and_paths(tmp_path):
    """`structures` dual-accept: Structure objects and file paths normalize to PDB on the wire."""
    from proto_tools.entities import Structure

    pdb_file = tmp_path / "s.pdb"
    pdb_file.write_text(_TINY_PDB)
    structure_obj = Structure(structure=_TINY_PDB)

    inputs = FoldmasonMSAInput(structures=[structure_obj, pdb_file])
    assert all(type(s).__name__ == "Structure" for s in inputs.structures)

    with patch("proto_tools.tools.structure_alignment.foldmason.foldmason_msa.ToolInstance.dispatch") as mock_dispatch:
        mock_dispatch.return_value = {
            "aa_msa_fasta": ">structure_0\nM\n>structure_1\nM\n",
            "three_di_msa_fasta": ">structure_0\nD\n>structure_1\nD\n",
            "newick_tree": "(structure_0,structure_1);\n",
        }
        run_foldmason_msa(inputs, FoldmasonMSAConfig(search_mode="local"))

    # Both entries normalize to PDB text on the wire payload.
    assert mock_dispatch.call_args.args[1]["structures"] == [_TINY_PDB, _TINY_PDB]


# ── run_foldmason_msa (remote, mocked HTTP) ──────────────────────────────────


def test_remote_mode_submits_polls_and_parses_json():
    """Remote mode submits, polls to COMPLETE, and parses the result JSON into FASTAs."""
    inputs = FoldmasonMSAInput(structures=[_TINY_PDB, _TINY_PDB], structure_ids=["q1", "q2"])
    config = FoldmasonMSAConfig()

    submit_response = MagicMock()
    submit_response.status_code = 200
    submit_response.raise_for_status.return_value = None
    submit_response.json.return_value = {"id": "tk-123"}

    result_response = MagicMock()
    result_response.raise_for_status.return_value = None
    result_response.json.return_value = {
        "entries": [
            {"name": "q1", "aa": "MK", "ss": "DD", "ca": ""},
            {"name": "q2", "aa": "MK", "ss": "DD", "ca": ""},
        ],
        "tree": "(q1,q2);",
        "scores": [0.8, 0.8],
        "statistics": {"msaLDDT": 0.8},
    }

    fake_session = MagicMock()
    fake_session.post.return_value = submit_response
    fake_session.get.return_value = result_response

    with (
        patch(
            "proto_tools.tools.structure_alignment.foldmason.foldmason_msa.build_http_session",
            return_value=fake_session,
        ),
        patch("proto_tools.tools.structure_alignment.foldmason.foldmason_msa.poll_until_complete"),
    ):
        output = run_foldmason_msa(inputs, config)

    assert output.success
    assert output.ticket_id == "tk-123"
    assert output.result_url == "https://search.foldseek.com/api/result/foldmason/tk-123"
    assert output.num_sequences == 2
    assert output.alignment_length == 2
    assert ">q1" in output.aa_msa_fasta and ">q2" in output.aa_msa_fasta
    assert output.newick_tree == "(q1,q2);"

    # Submit posted to the dedicated foldmason endpoint with multipart fileNames[] + queries[]
    fake_session.post.assert_called_once()
    submit_url = fake_session.post.call_args.args[0]
    assert submit_url == "https://search.foldseek.com/api/ticket/foldmason"
    files = fake_session.post.call_args.kwargs["files"]
    field_names = [name for name, _ in files]
    assert field_names == ["fileNames[]", "queries[]", "fileNames[]", "queries[]"]


def test_remote_submit_raises_on_missing_ticket_id():
    """Server response without an `id` field is a real schema regression and raises."""
    inputs = FoldmasonMSAInput(structures=[_TINY_PDB, _TINY_PDB])
    config = FoldmasonMSAConfig()

    submit_response = MagicMock()
    submit_response.status_code = 200
    submit_response.raise_for_status.return_value = None
    submit_response.json.return_value = {"status": "ERROR"}  # no "id"

    fake_session = MagicMock()
    fake_session.post.return_value = submit_response

    with patch(
        "proto_tools.tools.structure_alignment.foldmason.foldmason_msa.build_http_session",
        return_value=fake_session,
    ):
        with pytest.raises(Exception, match="no ticket ID"):
            run_foldmason_msa(inputs, config)


# ── Validation tests ─────────────────────────────────────────────────────────


def test_input_rejects_id_count_mismatch():
    """structure_ids must match structures length."""
    with pytest.raises(ValidationError, match="structure_ids length"):
        FoldmasonMSAInput(structures=[_TINY_PDB, _TINY_PDB], structure_ids=["only_one"])


def test_input_requires_min_two_structures():
    """A single structure can't form an MSA."""
    with pytest.raises(ValidationError, match="at least 2"):
        FoldmasonMSAInput(structures=[_TINY_PDB])


@pytest.mark.parametrize(
    "bad_id",
    ["../escape", "/etc/passwd", "..", ".", "path/with/slash", "path\\with\\backslash", ""],
    ids=["dot-dot-prefix", "absolute-path", "dotdot", "dot", "fwd-slash", "back-slash", "empty"],
)
def test_input_rejects_unsafe_structure_ids(bad_id):
    """structure_ids are written as `{id}.pdb` files; reject path-traversal-shaped values."""
    with pytest.raises(ValidationError, match="not a safe filename"):
        FoldmasonMSAInput(structures=[_TINY_PDB, _TINY_PDB], structure_ids=[bad_id, "ok"])


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.skip_ci  # downloads foldmason binary from mmseqs.com; upstream is flaky in GH Actions
def test_foldmason_msa_local_with_real_fixtures():
    """Local end-to-end: align 3 real PDBs via the foldmason binary, verify MSA shape + Newick leaves."""
    structures = [
        (_FIXTURES / "renin_af3.pdb").read_text(),
        (_FIXTURES / "test_structure_similarity.pdb").read_text(),
        (_FIXTURES / "renin_af3.pdb").read_text(),
    ]
    ids = ["renin1", "tester", "renin2"]

    output = run_foldmason_msa(
        FoldmasonMSAInput(structures=structures, structure_ids=ids),
        FoldmasonMSAConfig(search_mode="local", num_threads=2),
    )

    assert output.success, f"errors: {output.errors}"
    assert output.ticket_id == ""  # local mode produces no ticket
    assert output.num_sequences == 3
    assert output.alignment_length > 0
    # All input IDs round-trip into the AA FASTA + Newick.
    for sid in ids:
        assert f">{sid}" in output.aa_msa_fasta
        assert sid in output.newick_tree


@pytest.mark.integration
def test_foldmason_msa_remote_with_real_fixtures():
    """Remote end-to-end: 3 PDBs → live foldmason server → schema-validated MSA.

    The server caches results by structure content, so when we submit the same
    fixtures repeatedly the entry names (and ticket ID) reflect the *first*
    submission of those bytes — user-supplied ``structure_ids`` are dropped on
    a cache hit. Assertions cover the protocol shape + MSA dimensions only.
    """
    structures = [
        (_FIXTURES / "renin_af3.pdb").read_text(),
        (_FIXTURES / "test_structure_similarity.pdb").read_text(),
        (_FIXTURES / "renin_af3.pdb").read_text(),
    ]

    output = run_foldmason_msa(
        FoldmasonMSAInput(structures=structures, structure_ids=["renin1", "tester", "renin2"]),
        FoldmasonMSAConfig(timeout_seconds=300.0),
    )

    assert output.success, f"errors: {output.errors}"
    assert output.ticket_id and output.result_url.endswith(output.ticket_id)
    assert output.num_sequences == 3
    assert output.alignment_length > 0
    # Every record header should appear at the start of a line in the FASTA + the Newick string.
    headers = [line[1:] for line in output.aa_msa_fasta.splitlines() if line.startswith(">")]
    assert len(headers) == 3
    for h in headers:
        assert h in output.newick_tree, f"header {h!r} missing from Newick {output.newick_tree!r}"


# ── Benchmark ──────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("foldmason-msa")
@pytest.mark.slow
def test_foldmason_msa_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark foldmason-msa: 5x local easy-msa over a 90-structure MSA (3 folds cycled) (cold + warm)."""
    pdbs = [
        (_FIXTURES / "renin_af3.pdb").read_text(),
        (_FIXTURES / "test_structure_similarity.pdb").read_text(),
        (_FIXTURES / "pdl1.pdb").read_text(),
    ]
    n = 90
    structures = [pdbs[i % len(pdbs)] for i in range(n)]
    structure_ids = [f"struct_{i}" for i in range(n)]
    inputs = FoldmasonMSAInput(structures=structures, structure_ids=structure_ids)
    config = FoldmasonMSAConfig(search_mode="local", num_threads=4)

    def run_batch():
        last = None
        for _ in range(5):
            last = run_foldmason_msa(inputs, config)
        return last

    result = benchmark_twice(request, "foldmason", run_batch)
    validate_output(result)

    assert result.tool_id == "foldmason-msa"
    assert result.success, f"errors: {result.errors}"
    assert result.ticket_id == ""  # local mode produces no ticket
    assert result.num_sequences == 4 * n // 3
    assert result.alignment_length > 0
