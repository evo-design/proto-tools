"""tests/structure_alignment_tests/test_foldseek_multimercluster.py.

Tests for foldseek-multimercluster (local-only multimer structural clustering).
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from proto_tools.tools.structure_alignment import (
    FoldseekMultimerClusterConfig,
    FoldseekMultimerClusterInput,
    run_foldseek_multimercluster,
)

_TINY_MULTIMER_PDB = (
    "ATOM      1  CA  MET A   1       0.000   0.000   0.000  1.00  0.00\n"
    "ATOM      2  CA  MET B   1      10.000  10.000  10.000  1.00  0.00\n"
)
_FIXTURES = Path(__file__).parent.parent / "dummy_data"


# ── run_foldseek_multimercluster (mocked dispatch) ───────────────────────────


def test_run_foldseek_multimercluster_dispatches_easy_multimercluster():
    """The wrapper sends operation=easy_multimercluster + structures + thresholds; output is parsed clusters + FASTA."""
    inputs = FoldseekMultimerClusterInput(structures=[_TINY_MULTIMER_PDB, _TINY_MULTIMER_PDB, _TINY_MULTIMER_PDB])

    with patch(
        "proto_tools.tools.structure_alignment.foldseek.foldseek_multimercluster.ToolInstance.dispatch"
    ) as mock_dispatch:
        mock_dispatch.return_value = {
            "clusters_tsv": "multimer_0\tmultimer_0\nmultimer_0\tmultimer_1\nmultimer_2\tmultimer_2\n",
            "rep_seq_fasta": ">multimer_0_A\nMSEQ\n>multimer_2_A\nMSEQ\n",
        }
        output = run_foldseek_multimercluster(inputs, FoldseekMultimerClusterConfig())

    assert output.success
    assert output.num_clusters == 2
    assert output.num_multimers == 3
    assert "multimer_0_A" in output.rep_seq_fasta

    call = mock_dispatch.call_args
    assert call.args[0] == "foldseek"
    payload = call.args[1]
    assert payload["operation"] == "easy_multimercluster"
    assert payload["structures"] == [_TINY_MULTIMER_PDB] * 3
    assert payload["structure_ids"] == ["multimer_0", "multimer_1", "multimer_2"]
    # Defaults from MultimerCluster::setMultimerClusterDefaults — verified against foldseek source.
    assert payload["multimer_tm_threshold"] == 0.65
    assert payload["chain_tm_threshold"] == 0.001
    assert payload["interface_lddt_threshold"] == 0.5


def test_run_foldseek_multimercluster_uses_user_supplied_ids():
    """User-supplied structure_ids are forwarded verbatim to the standalone."""
    inputs = FoldseekMultimerClusterInput(
        structures=[_TINY_MULTIMER_PDB, _TINY_MULTIMER_PDB],
        structure_ids=["complex_alpha", "complex_beta"],
    )

    with patch(
        "proto_tools.tools.structure_alignment.foldseek.foldseek_multimercluster.ToolInstance.dispatch"
    ) as mock_dispatch:
        mock_dispatch.return_value = {
            "clusters_tsv": "complex_alpha\tcomplex_alpha\ncomplex_alpha\tcomplex_beta\n",
            "rep_seq_fasta": ">complex_alpha_A\nMSEQ\n",
        }
        output = run_foldseek_multimercluster(inputs, FoldseekMultimerClusterConfig())

    assert output.num_clusters == 1
    assert sorted(output.clusters[0].member_ids) == ["complex_alpha", "complex_beta"]
    assert mock_dispatch.call_args.args[1]["structure_ids"] == ["complex_alpha", "complex_beta"]


def test_foldseek_multimercluster_input_rejects_id_count_mismatch():
    """structure_ids must match structures length — Pydantic-level validation."""
    with pytest.raises(ValidationError, match="structure_ids length"):
        FoldseekMultimerClusterInput(
            structures=[_TINY_MULTIMER_PDB, _TINY_MULTIMER_PDB],
            structure_ids=["only_one"],
        )


def test_foldseek_multimercluster_input_requires_min_two_structures():
    """A single multimer can't be clustered; Pydantic rejects at validation time."""
    with pytest.raises(ValidationError, match="at least 2"):
        FoldseekMultimerClusterInput(structures=[_TINY_MULTIMER_PDB])


@pytest.mark.parametrize(
    "bad_id",
    ["../escape", "/etc/passwd", "..", ".", "path/with/slash", "path\\with\\backslash", ""],
    ids=["dot-dot-prefix", "absolute-path", "dotdot", "dot", "fwd-slash", "back-slash", "empty"],
)
def test_foldseek_multimercluster_input_rejects_unsafe_structure_ids(bad_id):
    """structure_ids are written as `{id}.pdb` files; reject path-traversal-shaped values."""
    with pytest.raises(ValidationError, match="not a safe filename"):
        FoldseekMultimerClusterInput(
            structures=[_TINY_MULTIMER_PDB, _TINY_MULTIMER_PDB],
            structure_ids=[bad_id, "ok"],
        )


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_foldseek_multimercluster_end_to_end_with_real_multimer_fixtures():
    """Local end-to-end: cluster two copies of the pdl1 2-chain fixture (self-match guaranteed) via the foldseek binary.

    Asserts schema invariants only — not specific cluster IDs — since cluster
    representative choice can shift with foldseek version. The two duplicates
    must land in the same cluster (multimer_tm=1.0 between identical copies).
    """
    multimer_pdb = (_FIXTURES / "pdl1.pdb").read_text()

    output = run_foldseek_multimercluster(
        FoldseekMultimerClusterInput(
            structures=[multimer_pdb, multimer_pdb],
            structure_ids=["pdl1_a", "pdl1_b"],
        ),
        FoldseekMultimerClusterConfig(num_threads=2),
    )

    assert output.success, f"errors: {output.errors}"
    assert output.num_multimers == 2
    assert output.num_clusters == 1, "two identical multimers should land in one cluster"
    # Foldseek's rep_seq_fasta uses '#multimer_id' separators followed by '>chain_id' chain records.
    assert ">" in output.rep_seq_fasta, "rep_seq_fasta must contain at least one chain record"

    cluster = output.clusters[0]
    # Foldseek encodes IDs as {multimer_id}_{chain}; both inputs must appear as some prefix.
    assert all(any(m.startswith(prefix) for m in cluster.member_ids) for prefix in ("pdl1_a", "pdl1_b"))
