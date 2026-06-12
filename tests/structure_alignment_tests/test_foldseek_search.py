"""tests/structure_alignment_tests/test_foldseek_search.py.

Tests for the Foldseek server-search wrapper.
"""

import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from proto_tools.tools.database_retrieval import (
    AlphaFoldDBFetchConfig,
    AlphaFoldDBFetchInput,
    run_alphafold_db_fetch,
)
from proto_tools.tools.structure_alignment import (
    FoldseekSearchConfig,
    FoldseekSearchInput,
    run_foldseek_search,
)
from proto_tools.tools.structure_alignment.foldseek.foldseek_search import (
    FoldseekHit,
    _parse_m8_archive,
    _parse_m8_text,
    _submit,
)
from tests.conftest import benchmark_twice
from tests.tool_infra_tests.test_export_functionality import validate_output


def _make_archive(files: dict[str, str]) -> bytes:
    """Build an in-memory tar.gz archive from {filename: tab-separated text} pairs."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, text in files.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _row(target_id: str, identity_pct: str = "55.0") -> str:
    """Build one tab-separated M8 row with sensible defaults."""
    return "\t".join(["query", target_id, identity_pct, "120", "10", "2", "1", "120", "5", "125", "1e-30", "150.0"])


# ── _parse_m8_text (malformed-row handling + coordinate validation) ───────────


def test_parse_m8_text_skips_malformed_rows():
    """Non-numeric fields and out-of-range coordinates are skipped, not raised."""
    good = "q\t1abc_A\t55.0\t120\t10\t2\t1\t120\t5\t125\t1e-30\t150.0"
    non_numeric = "q\t1abc_B\tNOTNUM\t120\t10\t2\t1\t120\t5\t125\t1e-30\t150.0"
    inverted = "q\t1abc_C\t55.0\t120\t10\t2\t120\t1\t5\t125\t1e-30\t150.0"
    zero_coord = "q\t1abc_D\t55.0\t120\t10\t2\t0\t120\t5\t125\t1e-30\t150.0"
    hits = _parse_m8_text("\n".join([good, non_numeric, inverted, zero_coord]), "testdb")
    assert [h.target_id for h in hits] == ["1abc_A"]


def test_foldseek_hit_rejects_invalid_coordinates():
    """FoldseekHit enforces 1-indexed coordinates with start <= end."""
    base = {
        "database": "d",
        "target_id": "t",
        "sequence_identity": 0.5,
        "alignment_length": 10,
        "mismatches": 0,
        "gap_openings": 0,
        "query_start": 1,
        "query_end": 10,
        "target_start": 1,
        "target_end": 10,
        "evalue": 1e-5,
        "bit_score": 50.0,
    }
    FoldseekHit(**base)
    with pytest.raises(ValidationError):
        FoldseekHit(**{**base, "query_start": 20})
    with pytest.raises(ValidationError):
        FoldseekHit(**{**base, "target_start": 0})


# ── _parse_m8_archive ─────────────────────────────────────────────────────────


def test_parse_m8_archive_extracts_hits_per_database():
    """Hits are partitioned by db, identity is normalized to a fraction, short rows are skipped."""
    archive = _make_archive(
        {
            "alis_pdb100.m8": _row("1tup_A", "55.0") + "\n" + "only\ttwo\tfields\n",
            "alis_afdb50.m8": _row("AF-P04637-F1", "100.0") + "\n",
        }
    )

    hits = _parse_m8_archive(archive)

    assert len(hits) == 2
    by_db = {h.database: h for h in hits}
    assert by_db["pdb100"].target_id == "1tup_A"
    assert by_db["pdb100"].sequence_identity == pytest.approx(0.55)
    assert by_db["afdb50"].target_id == "AF-P04637-F1"
    assert by_db["afdb50"].sequence_identity == pytest.approx(1.0)


@pytest.mark.parametrize(
    "extra_files,expected_count",
    [
        ({}, 0),
        ({"alis_pdb100.m8": _row("1tup_A") + "\n"}, 1),
    ],
    ids=["no-m8-files", "m8-and-non-m8-mixed"],
)
def test_parse_m8_archive_filters_to_m8_files(extra_files, expected_count):
    """Only `.m8` archive entries are parsed; other files are silently ignored."""
    archive = _make_archive({"config.json": '{"version": 1}', **extra_files})
    assert len(_parse_m8_archive(archive)) == expected_count


def test_parse_m8_archive_raises_on_non_tarball_bytes():
    """An HTML error page from the server raises a clear ValueError, not an opaque TarError."""
    html = b"<html><body>503 Service Unavailable</body></html>"
    with pytest.raises(ValueError, match=r"not a valid tar\.gz archive"):
        _parse_m8_archive(html)


# ── _submit ───────────────────────────────────────────────────────────────────


def test_submit_parses_ticket_id_and_encodes_databases():
    """Submit returns the ticket ID and encodes each database as a separate database[] form entry."""
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status.return_value = None
    response.json.return_value = {"id": "ticket-abc-123", "status": "PENDING"}
    session = MagicMock()
    session.post.return_value = response

    ticket_id = _submit("PDB...", ["pdb100", "afdb50"], "tmalign", session)

    assert ticket_id == "ticket-abc-123"
    call = session.post.call_args
    assert "files" in call.kwargs
    assert call.kwargs["data"] == [
        ("database[]", "pdb100"),
        ("database[]", "afdb50"),
        ("mode", "tmalign"),
    ]


# ── Local mode (mocked dispatch) ─────────────────────────────────────────────


def test_local_mode_dispatches_easy_search():
    """search_mode='local' dispatches operation=easy_search with structure_text + local_db."""
    from unittest.mock import patch

    pdb_text = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
    inputs = FoldseekSearchInput(structure=pdb_text)
    config = FoldseekSearchConfig(search_mode="local", local_db="/path/to/db", num_threads=8)

    with patch("proto_tools.tools.structure_alignment.foldseek.foldseek_search.ToolInstance.dispatch") as mock_dispatch:
        mock_dispatch.return_value = {"stdout": "query\t1abc_A\t50.0\t100\t5\t1\t1\t100\t10\t110\t1e-30\t150.0\n"}
        output = run_foldseek_search(inputs, config)

    assert output.success
    assert output.ticket_id == ""  # local mode doesn't issue a ticket
    assert output.num_hits == 1
    assert output.hits[0].target_id == "1abc_A"
    assert output.databases_queried == ["/path/to/db"]

    payload = mock_dispatch.call_args.args[1]
    assert payload == {
        "operation": "easy_search",
        "structure_text": pdb_text,
        "local_db": "/path/to/db",
        "evalue": 10.0,
        "sensitivity": 9.5,
        "max_seqs": 1000,
        "alignment_type": 2,
        "tmscore_threshold": 0.0,
        "lddt_threshold": 0.0,
        "num_threads": 8,
        "use_gpu": False,
        "device": "cpu",
    }


def test_local_mode_validator_requires_local_db():
    """Pydantic rejects search_mode='local' without local_db at config-construction time."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="local_db is required"):
        FoldseekSearchConfig(search_mode="local")


@pytest.mark.parametrize(
    "payload,error_pattern",
    [
        ({"status": "ERROR"}, "no ticket ID"),  # missing id
        ({"id": ""}, "no ticket ID"),  # empty id
    ],
    ids=["missing-id", "empty-id"],
)
def test_submit_raises_on_invalid_id(payload, error_pattern):
    """A submit response without a non-empty `id` field is a real schema regression → ValueError."""
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    session = MagicMock()
    session.post.return_value = response

    with pytest.raises(ValueError, match=error_pattern):
        _submit("PDB...", ["pdb100"], "3diaa", session)


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_foldseek_search_end_to_end_with_tp53():
    """End-to-end: TP53 from AFDB → Foldseek pdb100 → schema-validated structural hits.

    Hits search.foldseek.com live (submit, poll, download, parse). TP53 is a
    well-known protein design target; restricting to pdb100 keeps runtime ~10-30s
    (afdb50 alone takes minutes). Asserts protocol correctness and schema
    invariants — does NOT assert specific PDB IDs because pdb100 is a
    redundancy-clustered DB whose representatives shift over time.
    """
    afdb = run_alphafold_db_fetch(
        AlphaFoldDBFetchInput(uniprot_id="P04637"),
        AlphaFoldDBFetchConfig(structure_format="pdb"),
    )
    assert afdb.success and afdb.structure is not None

    output = run_foldseek_search(
        FoldseekSearchInput(structure=afdb.structure.structure_pdb),
        FoldseekSearchConfig(databases=["pdb100"], timeout_seconds=600.0),
    )

    # Tool wrapper invariants — protocol-level correctness
    assert output.success
    assert output.ticket_id and output.result_url.endswith(output.ticket_id)
    assert output.databases_queried == ["pdb100"]
    assert output.num_hits == len(output.hits) > 0

    # Schema invariants on every hit — catches M8 column drift
    for hit in output.hits:
        assert hit.database == "pdb100"
        assert 0.0 <= hit.sequence_identity <= 1.0
        assert hit.alignment_length >= 0
        assert hit.evalue >= 0.0
        assert 1 <= hit.query_start <= hit.query_end
        assert 1 <= hit.target_start <= hit.target_end

    # Sanity check: the best hit aligns a meaningful fraction of the query.
    # Catches a regression where Foldseek returns only spurious 1-residue matches.
    best = min(output.hits, key=lambda h: h.evalue)
    assert best.alignment_length >= 30, (
        f"Best hit alignment too short ({best.alignment_length} aa); Foldseek scoring may be miscalibrated"
    )


@pytest.mark.integration
def test_foldseek_search_local_mode_with_directory_db(tmp_path):
    """Local end-to-end: search the renin fixture against a tmp directory of PDB targets (self-match guaranteed)."""
    fixtures = Path(__file__).parent.parent / "dummy_data"
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    (target_dir / "renin.pdb").write_text((fixtures / "renin_af3.pdb").read_text())
    (target_dir / "test_struct.pdb").write_text((fixtures / "test_structure_similarity.pdb").read_text())

    output = run_foldseek_search(
        FoldseekSearchInput(structure=(fixtures / "renin_af3.pdb").read_text()),
        FoldseekSearchConfig(search_mode="local", local_db=str(target_dir), num_threads=2),
    )

    assert output.success, f"errors: {output.errors}"
    assert output.ticket_id == ""  # local mode produces no ticket
    assert output.databases_queried == [str(target_dir)]
    assert output.num_hits >= 1
    # Self-search should put the renin self-match at high identity / very low e-value.
    self_hits = [h for h in output.hits if "renin" in h.target_id.lower()]
    assert self_hits, f"renin self-match missing; got {[h.target_id for h in output.hits]}"
    assert self_hits[0].sequence_identity > 0.99


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("foldseek-search")
@pytest.mark.slow
def test_foldseek_search_benchmark(request: pytest.FixtureRequest, tmp_path) -> None:
    """Benchmark foldseek-search: 20 sequential local easy-search runs of renin (~340 aa) vs a 39-structure PDB DB (cold + warm)."""
    fixtures = Path(__file__).parent.parent / "dummy_data"
    src = [
        (fixtures / "renin_af3.pdb").read_text(),
        (fixtures / "test_structure_similarity.pdb").read_text(),
        (fixtures / "pdl1.pdb").read_text(),
    ]
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    for i in range(39):
        (target_dir / f"target_{i}.pdb").write_text(src[i % len(src)])

    inputs = FoldseekSearchInput(structure=src[0])
    config = FoldseekSearchConfig(search_mode="local", local_db=str(target_dir), num_threads=4)

    def run_batch():
        last = None
        for _ in range(20):
            last = run_foldseek_search(inputs, config)
        return last

    result = benchmark_twice(request, "foldseek", run_batch)
    validate_output(result)

    assert result.tool_id == "foldseek-search"
    assert result.ticket_id == ""  # local mode produces no ticket
    assert result.databases_queried == [str(target_dir)]
    assert result.num_hits == len(result.hits) >= 1
    assert max(h.sequence_identity for h in result.hits) > 0.99, "renin self-match should be present"
