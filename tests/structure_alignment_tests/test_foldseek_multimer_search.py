"""tests/structure_alignment_tests/test_foldseek_multimer_search.py.

Tests for foldseek-multimer-search (remote + local).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from proto_tools.tools.structure_alignment import (
    FoldseekMultimerSearchConfig,
    FoldseekMultimerSearchInput,
    run_foldseek_multimer_search,
)

_MULTIMER_FIXTURE = Path(__file__).parent.parent / "dummy_data" / "pdl1.pdb"
_TINY_MULTIMER_PDB = (
    "ATOM      1  CA  MET A   1       0.000   0.000   0.000  1.00  0.00\n"
    "ATOM      2  CA  MET B   1      10.000  10.000  10.000  1.00  0.00\n"
)


# ── run_foldseek_multimer_search (remote, mocked HTTP) ───────────────────────


def test_remote_multimer_search_wraps_mode_with_complex_prefix():
    """Remote multimer mode wraps config.mode='3diaa' as wire-mode 'complex-3diaa' on submit."""
    inputs = FoldseekMultimerSearchInput(structure_text=_TINY_MULTIMER_PDB)

    submitted_modes: list[str] = []

    def fake_submit(structure_text, databases, mode, session):
        submitted_modes.append(mode)
        return "mt-123"

    fake_archive_response = MagicMock()
    fake_archive_response.raise_for_status.return_value = None
    fake_archive_response.content = b""

    fake_session = MagicMock()
    fake_session.get.return_value = fake_archive_response

    with (
        patch("proto_tools.tools.structure_alignment.foldseek.foldseek_multimer_search._submit", fake_submit),
        patch(
            "proto_tools.tools.structure_alignment.foldseek.foldseek_multimer_search.build_http_session",
            return_value=fake_session,
        ),
        patch("proto_tools.tools.structure_alignment.foldseek.foldseek_multimer_search.poll_until_complete"),
        patch(
            "proto_tools.tools.structure_alignment.foldseek.foldseek_multimer_search._parse_m8_archive",
            return_value=[],
        ),
    ):
        output = run_foldseek_multimer_search(inputs, FoldseekMultimerSearchConfig())

    assert output.success
    assert output.ticket_id == "mt-123"
    assert submitted_modes == ["complex-3diaa"]


# ── run_foldseek_multimer_search (local, mocked dispatch) ────────────────────


def test_local_multimer_search_dispatches_easy_multimersearch():
    """Local mode dispatches operation=easy_multimersearch with structure + local_db."""
    inputs = FoldseekMultimerSearchInput(structure_text=_TINY_MULTIMER_PDB)
    config = FoldseekMultimerSearchConfig(search_mode="local", local_db="/path/to/db")

    with patch(
        "proto_tools.tools.structure_alignment.foldseek.foldseek_multimer_search.ToolInstance.dispatch"
    ) as mock_dispatch:
        mock_dispatch.return_value = {"stdout": "query\t1abc_A\t75.0\t100\t5\t1\t1\t100\t10\t110\t1e-30\t150.0\n"}
        output = run_foldseek_multimer_search(inputs, config)

    assert output.success
    assert output.ticket_id == ""  # no ticket in local mode
    assert output.num_hits == 1
    assert output.hits[0].target_id == "1abc_A"
    assert output.databases_queried == ["/path/to/db"]

    payload = mock_dispatch.call_args.args[1]
    assert payload["operation"] == "easy_multimersearch"
    assert payload["local_db"] == "/path/to/db"


def test_local_mode_validator_requires_local_db():
    """Pydantic rejects search_mode='local' without local_db at config-construction time."""
    with pytest.raises(ValidationError, match="local_db is required"):
        FoldseekMultimerSearchConfig(search_mode="local")


# Note: invalid-ticket-id error path is covered by `test_submit_raises_on_invalid_id`
# in test_foldseek_search.py — multimer reuses the same `_submit` helper.


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_remote_multimer_search_finds_complex_neighbors_for_pdl1():
    """End-to-end: a real 2-chain PDB → Foldseek multimer pdb100 → schema-validated hits.

    Hits search.foldseek.com/foldmulti via the same /api/ticket endpoint with
    mode='complex-3diaa'. Restricted to pdb100 to keep runtime modest. Asserts
    protocol correctness + per-hit schema invariants — does NOT assert specific
    PDB IDs since multimer-aware cluster representatives shift over time.
    """
    multimer_pdb = _MULTIMER_FIXTURE.read_text()

    output = run_foldseek_multimer_search(
        FoldseekMultimerSearchInput(structure_text=multimer_pdb),
        FoldseekMultimerSearchConfig(databases=["pdb100"], timeout_seconds=600.0),
    )

    # Tool wrapper invariants
    assert output.success, f"errors: {output.errors}"
    assert output.ticket_id and output.result_url.endswith(output.ticket_id)
    assert output.databases_queried == ["pdb100"]
    assert output.num_hits == len(output.hits) > 0

    # Schema invariants on every hit
    for hit in output.hits:
        assert hit.database == "pdb100"
        assert 0.0 <= hit.sequence_identity <= 1.0
        assert hit.alignment_length >= 0
        assert hit.evalue >= 0.0
        assert 1 <= hit.query_start <= hit.query_end
        assert 1 <= hit.target_start <= hit.target_end

    # Sanity: best hit aligns a meaningful fraction of the query.
    best = min(output.hits, key=lambda h: h.evalue)
    assert best.alignment_length >= 30, (
        f"Best hit alignment too short ({best.alignment_length} aa); multimer search may be miscalibrated"
    )


@pytest.mark.integration
def test_local_multimer_search_with_directory_db(tmp_path):
    """Local end-to-end: pdl1 2-chain fixture as both query and target (self-match guaranteed)."""
    multimer_pdb = _MULTIMER_FIXTURE.read_text()
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    (target_dir / "pdl1.pdb").write_text(multimer_pdb)

    output = run_foldseek_multimer_search(
        FoldseekMultimerSearchInput(structure_text=multimer_pdb),
        FoldseekMultimerSearchConfig(search_mode="local", local_db=str(target_dir), num_threads=2),
    )

    assert output.success, f"errors: {output.errors}"
    assert output.ticket_id == ""
    assert output.databases_queried == [str(target_dir)]
    assert output.num_hits >= 1
    # Self-search should produce a near-perfect identity match somewhere in hits.
    best = max(output.hits, key=lambda h: h.sequence_identity)
    assert best.sequence_identity > 0.99
