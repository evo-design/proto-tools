"""tests/structure_alignment_tests/test_foldseek_cluster.py.

Tests for foldseek-cluster (local-only structural clustering).
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from proto_tools.tools.structure_alignment import (
    FoldseekClusterConfig,
    FoldseekClusterInput,
    run_foldseek_cluster,
)
from proto_tools.tools.structure_alignment.foldseek.foldseek_cluster import _parse_cluster_tsv

_TINY_PDB = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
_FIXTURES = Path(__file__).parent.parent / "dummy_data"


# ── _parse_cluster_tsv ────────────────────────────────────────────────────────


def test_parse_cluster_tsv_groups_members_by_representative():
    """Multiple lines per representative collapse into one FoldseekCluster with all members."""
    tsv = "rep1\tmem1\nrep1\tmem2\nrep1\trep1\nrep2\tmem3\nrep2\trep2\n"

    clusters = _parse_cluster_tsv(tsv)

    assert len(clusters) == 2
    by_rep = {c.representative_id: c for c in clusters}
    assert sorted(by_rep["rep1"].member_ids) == ["mem1", "mem2", "rep1"]
    assert sorted(by_rep["rep2"].member_ids) == ["mem3", "rep2"]


def test_parse_cluster_tsv_skips_blank_and_short_lines():
    """Empty / single-column lines are silently dropped (defensive)."""
    tsv = "rep1\tmem1\n\nshort_line\nrep1\tmem2\n"

    clusters = _parse_cluster_tsv(tsv)

    assert len(clusters) == 1
    assert sorted(clusters[0].member_ids) == ["mem1", "mem2"]


# ── run_foldseek_cluster (mocked dispatch) ────────────────────────────────────


def test_run_foldseek_cluster_dispatches_easy_cluster():
    """The wrapper sends operation=easy_cluster + structures + ids; output is parsed clusters."""
    inputs = FoldseekClusterInput(structures=[_TINY_PDB, _TINY_PDB, _TINY_PDB])

    with patch(
        "proto_tools.tools.structure_alignment.foldseek.foldseek_cluster.ToolInstance.dispatch"
    ) as mock_dispatch:
        mock_dispatch.return_value = {
            "clusters_tsv": "structure_0\tstructure_0\nstructure_0\tstructure_1\nstructure_2\tstructure_2\n"
        }
        output = run_foldseek_cluster(inputs, FoldseekClusterConfig())

    assert output.success
    assert output.num_clusters == 2
    assert output.num_structures == 3

    # The dispatch call shape — toolkit + operation + payload
    call = mock_dispatch.call_args
    assert call.args[0] == "foldseek"
    payload = call.args[1]
    assert payload["operation"] == "easy_cluster"
    assert payload["structures"] == [_TINY_PDB, _TINY_PDB, _TINY_PDB]
    assert payload["structure_ids"] == ["structure_0", "structure_1", "structure_2"]
    assert payload["min_seq_id"] == 0.0  # default lets 3Di structural similarity dominate


def test_run_foldseek_cluster_uses_user_supplied_ids():
    """User-supplied structure_ids are forwarded verbatim to the standalone."""
    inputs = FoldseekClusterInput(
        structures=[_TINY_PDB, _TINY_PDB],
        structure_ids=["my_protein_a", "my_protein_b"],
    )

    with patch(
        "proto_tools.tools.structure_alignment.foldseek.foldseek_cluster.ToolInstance.dispatch"
    ) as mock_dispatch:
        mock_dispatch.return_value = {"clusters_tsv": "my_protein_a\tmy_protein_a\nmy_protein_a\tmy_protein_b\n"}
        output = run_foldseek_cluster(inputs, FoldseekClusterConfig())

    assert output.success
    assert output.num_clusters == 1
    assert sorted(output.clusters[0].member_ids) == ["my_protein_a", "my_protein_b"]
    assert mock_dispatch.call_args.args[1]["structure_ids"] == ["my_protein_a", "my_protein_b"]


def test_foldseek_cluster_input_rejects_id_count_mismatch():
    """structure_ids must match structures length — Pydantic-level validation."""
    with pytest.raises(ValidationError, match="structure_ids length"):
        FoldseekClusterInput(structures=[_TINY_PDB, _TINY_PDB], structure_ids=["only_one_id"])


def test_foldseek_cluster_input_requires_min_two_structures():
    """A single structure can't be clustered; Pydantic rejects at validation time."""
    with pytest.raises(ValidationError, match="at least 2"):
        FoldseekClusterInput(structures=[_TINY_PDB])


@pytest.mark.parametrize(
    "bad_id",
    ["../escape", "/etc/passwd", "..", ".", "path/with/slash", "path\\with\\backslash", ""],
    ids=["dot-dot-prefix", "absolute-path", "dotdot", "dot", "fwd-slash", "back-slash", "empty"],
)
def test_foldseek_cluster_input_rejects_unsafe_structure_ids(bad_id):
    """structure_ids are written as `{id}.pdb` files; reject path-traversal-shaped values."""
    with pytest.raises(ValidationError, match="not a safe filename"):
        FoldseekClusterInput(
            structures=[_TINY_PDB, _TINY_PDB],
            structure_ids=[bad_id, "ok"],
        )


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_foldseek_cluster_end_to_end_with_real_pdb_fixtures():
    """Local end-to-end: cluster two real single-chain PDB fixtures via the foldseek binary."""
    structures = [
        (_FIXTURES / "renin_af3.pdb").read_text(),
        (_FIXTURES / "test_structure_similarity.pdb").read_text(),
    ]

    output = run_foldseek_cluster(
        FoldseekClusterInput(structures=structures, structure_ids=["renin", "test"]),
        FoldseekClusterConfig(),
    )

    assert output.success, f"errors: {output.errors}"
    assert output.num_structures == 2
    assert output.num_clusters >= 1  # Can be 1 or 2 depending on structural similarity
    # Every input ID appears in exactly one cluster (modulo Foldseek's _chain suffixing).
    all_members = {m for cluster in output.clusters for m in cluster.member_ids}
    assert {"renin", "test"}.issubset(all_members) or all(
        any(m.startswith(prefix) for m in all_members) for prefix in ("renin", "test")
    )
