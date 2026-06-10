"""tests/database_retrieval_tests/test_alphafold_db_fetch.py.

Tests for the AlphaFold DB fetch tool.
"""

import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from proto_tools.entities.structures.structure import BFactorType, Structure
from proto_tools.tools.database_retrieval import (
    AlphaFoldDBFetchConfig,
    AlphaFoldDBFetchInput,
    UniProtFetchConfig,
    UniProtFetchInput,
    run_alphafold_db_fetch,
    run_uniprot_fetch,
)
from proto_tools.tools.database_retrieval.alphafold_db.alphafold_db_fetch import (
    AlphaFoldDBFetchOutput,
    _fetch_pae,
    _fetch_plddt,
    _fetch_prediction,
)
from proto_tools.tools.tool_registry import _make_error_output
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec


def _mock_session(payload, status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    session = MagicMock()
    session.get.return_value = response
    return session


def test_fetch_plddt_parses_confidence_score_array():
    session = _mock_session(
        {"residueNumber": [1, 2, 3], "confidenceScore": [42.5, 50.1, 91.0], "confidenceCategory": ["D", "D", "H"]}
    )
    result = _fetch_plddt("https://example/plddt.json", session)
    assert result == [42.5, 50.1, 91.0]


def test_fetch_pae_parses_predicted_aligned_error_matrix():
    session = _mock_session([{"predicted_aligned_error": [[0.1, 1.2], [1.2, 0.0]]}])
    result = _fetch_pae("https://example/pae.json", session)
    assert result == [[0.1, 1.2], [1.2, 0.0]]


@pytest.mark.parametrize(
    "fetch_fn, payload, error_pattern",
    [
        (_fetch_plddt, {"residueNumber": [1]}, "confidenceScore"),
        (_fetch_pae, [], "empty or wrong shape"),
        (_fetch_pae, [{"foo": "bar"}], "predicted_aligned_error"),
    ],
    ids=["plddt-missing-scores", "pae-empty-list", "pae-missing-matrix"],
)
def test_fetch_parsers_reject_malformed_payloads(fetch_fn, payload, error_pattern):
    """Each parser must surface a clear ValueError when AFDB's JSON shape is unexpected."""
    session = _mock_session(payload)
    with pytest.raises(ValueError, match=error_pattern):
        fetch_fn("https://example/file.json", session)


def test_fetch_prediction_returns_none_on_404():
    """A 404 from the AFDB API short-circuits without parsing the body."""
    session = _mock_session(payload=None, status_code=404)
    result = _fetch_prediction("https://example/api", session)
    assert result is None
    session.get.return_value.json.assert_not_called()


# ---------------------------------------------------------------------------
# CSV manifest + sidecar export tests (mock-only, no live API)
# ---------------------------------------------------------------------------

_FAKE_PDB_BODY = (
    "HEADER    TEST                                    01-JAN-26   FAKE              \n"
    "ATOM      1  CA  MET A   1      11.111  22.222  33.333  1.00 91.20           C  \n"
    "TER       2      MET A   1                                                      \n"
    "END                                                                             \n"
)


def _make_structure(structure_format: str = "pdb") -> Structure:
    pdb_struct = Structure(
        structure=_FAKE_PDB_BODY,
        structure_format="pdb",
        b_factor_type=BFactorType.PLDDT,
        source="alphafold-db-fetch",
    )
    if structure_format == "pdb":
        return pdb_struct
    return Structure(
        structure=pdb_struct.structure_cif,
        structure_format="cif",
        b_factor_type=BFactorType.PLDDT,
        source="alphafold-db-fetch",
    )


# Sentinel so callers can pass ``structure=None`` explicitly to omit it.
_DEFAULT = object()


def _make_full_output(
    *,
    structure=_DEFAULT,
    msa_a3m: str | None = ">P04637\nMEEPQSDPSVE\n",
    raw_entry: dict | None = None,
) -> AlphaFoldDBFetchOutput:
    """Build a fully populated success Output for export-tests."""
    output = AlphaFoldDBFetchOutput(
        uniprot_accession="P04637",
        entry_id="AF-P04637-F1",
        sequence="MEEPQSDPSVE",
        sequence_length=11,
        sequence_start=1,
        sequence_end=11,
        latest_version=4,
        pdb_url="https://example/file.pdb",
        cif_url="https://example/file.cif",
        pae_doc_url="https://example/pae.json",
        plddt_doc_url="https://example/plddt.json",
        pae_image_url="https://example/pae.png",
        structure=_make_structure() if structure is _DEFAULT else structure,
        msa_a3m=msa_a3m,
        source_url="https://example/api",
        raw_entry={"modelEntityId": "AF-P04637-F1", "extra": "value"} if raw_entry is None else raw_entry,
    )
    output.success = True
    return output


def _read_csv_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    return rows[0]


@pytest.mark.parametrize("structure_format", ["pdb", "cif"])
def test_csv_export_writes_manifest_with_basename_path_columns(structure_format):
    """Happy path: 3 sidecars written; structure suffix follows ``structure_format``; paths are basenames."""
    output = _make_full_output(structure=_make_structure(structure_format=structure_format))
    with tempfile.TemporaryDirectory() as tmp:
        output.export(name="result", export_path=tmp, file_format="csv")
        d = Path(tmp)
        assert {p.name for p in d.iterdir()} == {
            "result.a3m",
            f"result.{structure_format}",
            "result.csv",
            "result_raw.json",
        }
        row = _read_csv_row(d / "result.csv")
        assert row["structure_path"] == f"result.{structure_format}"
        assert row["msa_path"] == "result.a3m"
        assert row["raw_path"] == "result_raw.json"
        for col in ("structure_path", "msa_path", "raw_path"):
            assert "/" not in row[col] and "\\" not in row[col]


@pytest.mark.parametrize(
    "kwargs,absent_file,empty_col",
    [
        ({"structure": None}, "result.pdb", "structure_path"),
        ({"msa_a3m": None}, "result.a3m", "msa_path"),
        ({"raw_entry": {}}, "result_raw.json", "raw_path"),
    ],
)
def test_csv_export_omits_sidecar_when_field_unset(kwargs, absent_file, empty_col):
    """Sidecars are written only when their field is populated; the column stays empty otherwise."""
    output = _make_full_output(**kwargs)
    with tempfile.TemporaryDirectory() as tmp:
        output.export(name="result", export_path=tmp, file_format="csv")
        assert not (Path(tmp) / absent_file).exists()
        assert _read_csv_row(Path(tmp) / "result.csv")[empty_col] == ""


def test_csv_export_round_trips_blob_bodies():
    """Sidecar contents equal the source fields exactly (PDB / A3M byte-for-byte; raw_entry JSON-equal)."""
    output = _make_full_output()
    with tempfile.TemporaryDirectory() as tmp:
        output.export(name="result", export_path=tmp, file_format="csv")
        d = Path(tmp)
        assert (d / "result.pdb").read_text(encoding="utf-8") == output.structure.structure
        assert (d / "result.a3m").read_text(encoding="utf-8") == output.msa_a3m
        with (d / "result_raw.json").open(encoding="utf-8") as f:
            assert json.load(f) == output.raw_entry


def test_csv_export_on_failed_output_writes_metadata_only():
    """Failed outputs (success=False) export without crashing; no sidecars, empty path columns."""
    failed = _make_error_output(
        output_class=AlphaFoldDBFetchOutput,
        key="alphafold-db-fetch",
        start_time=0.0,
        exception=ValueError("no prediction"),
        traceback_str="",
    )
    with tempfile.TemporaryDirectory() as tmp:
        failed.export(name="result", export_path=tmp, file_format="csv")
        d = Path(tmp)
        assert sorted(p.name for p in d.iterdir()) == ["result.csv"]
        row = _read_csv_row(d / "result.csv")
        assert row["structure_path"] == row["msa_path"] == row["raw_path"] == ""


def test_json_export_unchanged_after_sidecar_addition():
    """Regression guard: JSON export still emits one inline file with all blobs."""
    output = _make_full_output()
    with tempfile.TemporaryDirectory() as tmp:
        output.export(name="result", export_path=tmp, file_format="json")
        d = Path(tmp)
        assert sorted(p.name for p in d.iterdir()) == ["result.json"]
        payload = json.loads((d / "result.json").read_text(encoding="utf-8"))
        assert payload["msa_a3m"] == output.msa_a3m
        assert payload["raw_entry"] == output.raw_entry
        assert payload["structure"] is not None


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_alphafold_db_fetch_csv_export_against_live_api_yields_usable_sidecars():
    """Live AFDB fetch + CSV export: each sidecar re-parses to match the manifest row."""
    output = run_alphafold_db_fetch(
        AlphaFoldDBFetchInput(uniprot_id="P04637"),
        AlphaFoldDBFetchConfig(include_msa=True),
    )
    assert output.success

    with tempfile.TemporaryDirectory() as tmp:
        output.export(name="tp53", export_path=tmp, file_format="csv")
        d = Path(tmp)
        assert {p.name for p in d.iterdir()} == {"tp53.csv", "tp53.pdb", "tp53.a3m", "tp53_raw.json"}

        row = _read_csv_row(d / "tp53.csv")
        assert row["entry_id"] == "AF-P04637-F1"
        assert row["structure_path"] == "tp53.pdb"
        assert row["msa_path"] == "tp53.a3m"
        assert row["raw_path"] == "tp53_raw.json"

        rehydrated = Structure(
            structure=(d / "tp53.pdb").read_text(encoding="utf-8"),
            structure_format="pdb",
            b_factor_type=BFactorType.PLDDT,
            source="alphafold-db-fetch",
        )
        chain_seqs = rehydrated.get_chain_sequences()
        assert chain_seqs[next(iter(chain_seqs))] == row["sequence"]

        a3m_lines = (d / "tp53.a3m").read_text(encoding="utf-8").splitlines()
        a3m_query = "".join(c for c in a3m_lines[1] if c.isalpha() and c.isupper())
        assert a3m_query == row["sequence"]


@pytest.mark.integration
def test_alphafold_db_fetch_full_p04637_record():
    """Default fetch returns a complete record with a parsed PDB Structure and length-matched pLDDT."""
    output = run_alphafold_db_fetch(AlphaFoldDBFetchInput(uniprot_id="P04637"), AlphaFoldDBFetchConfig())
    assert output.success
    assert output.tool_id == "alphafold-db-fetch"
    assert output.uniprot_accession == "P04637"
    assert output.entry_id == "AF-P04637-F1"
    assert output.sequence.startswith("M") and len(output.sequence) == 393  # human TP53 canonical length
    assert output.sequence_length == 393
    assert output.sequence_start == 1 and output.sequence_end == 393
    assert output.mean_plddt is not None and 0.0 <= output.mean_plddt <= 100.0
    assert output.latest_version >= 4

    # Structure shape: parsed Structure with PDB body, PLDDT B-factors, AFDB source
    assert output.structure is not None
    assert output.structure.structure_format == "pdb"
    assert output.structure.b_factor_type == BFactorType.PLDDT
    assert output.structure.source == "alphafold-db-fetch"

    # Structure body must parse into a real PDB record, not just contain the literal "ATOM"
    atom_lines = [line for line in output.structure.structure.splitlines() if line.startswith("ATOM ")]
    assert len(atom_lines) > 1000  # 393 residues * many atoms each
    # First ATOM line follows the PDB column convention: cols 13-16 = atom name, cols 18-20 = residue name
    first_atom = atom_lines[0]
    assert first_atom[13:16].strip() in {"N", "CA", "C", "O"}
    assert first_atom[17:20].strip() == "MET"  # TP53 starts with methionine

    # pLDDT array must align 1:1 with the sequence and live on structure.metrics
    plddt_per_residue = output.structure.metrics["plddt_per_residue"]
    assert len(plddt_per_residue) == output.sequence_length
    assert all(0 <= v <= 100 for v in plddt_per_residue)
    # Mean of the array equals the reported global mean (within float tolerance)
    derived_mean = sum(plddt_per_residue) / len(plddt_per_residue)
    assert derived_mean == pytest.approx(output.structure.metrics["avg_plddt"], abs=0.5)
    # avg_plddt on the Structure mirrors the top-level mean_plddt metadata
    assert output.structure.metrics["avg_plddt"] == pytest.approx(output.mean_plddt, abs=1e-6)

    # PAE and MSA are opt-in
    assert "pae" not in output.structure.metrics
    assert output.msa_a3m is None

    # Spec contract: every always-available metric is present, typed, and in-range
    assert_metrics_in_spec(output)


@pytest.mark.integration
def test_alphafold_db_fetch_with_pae_returns_square_matrix():
    """PAE matrix must be square with side equal to sequence length."""
    output = run_alphafold_db_fetch(
        AlphaFoldDBFetchInput(uniprot_id="P04637"),
        AlphaFoldDBFetchConfig(include_pae=True),
    )
    assert output.success
    assert output.structure is not None
    pae = output.structure.metrics["pae"]
    n = output.sequence_length
    assert len(pae) == n
    assert all(len(row) == n for row in pae)
    # Diagonal entries are conventionally near-zero (a residue's error against itself)
    assert all(pae[i][i] < 5.0 for i in range(0, n, 50))

    # With include_pae=True, pae is now also a spec-validated metric
    assert_metrics_in_spec(output)


@pytest.mark.integration
def test_alphafold_db_fetch_with_msa_returns_a3m_text():
    """include_msa=True returns parseable A3M-format MSA contents."""
    output = run_alphafold_db_fetch(
        AlphaFoldDBFetchInput(uniprot_id="P04637"),
        AlphaFoldDBFetchConfig(include_msa=True),
    )
    assert output.success
    assert output.msa_a3m is not None
    # A3M begins with a header line ('>...') and the first sequence record corresponds
    # to the query — its ungapped/uppercase residues must equal the entry's sequence.
    lines = output.msa_a3m.splitlines()
    assert lines[0].startswith(">")
    query_seq = "".join(c for c in lines[1] if c.isalpha() and c.isupper())
    assert query_seq == output.sequence


@pytest.mark.integration
def test_alphafold_db_fetch_cif_format():
    """structure_format='cif' returns a parseable mmCIF body."""
    output = run_alphafold_db_fetch(
        AlphaFoldDBFetchInput(uniprot_id="P04637"),
        AlphaFoldDBFetchConfig(structure_format="cif"),
    )
    assert output.success
    assert output.structure is not None
    assert output.structure.structure_format == "cif"
    assert output.structure.structure.startswith("data_")  # mmCIF data block header
    assert "_atom_site." in output.structure.structure  # atom loop column prefix


@pytest.mark.integration
def test_alphafold_db_fetch_unknown_accession_returns_failure():
    """A nonexistent accession raises with a clear error."""
    with pytest.raises(Exception, match="AlphaFold DB has no prediction"):
        run_alphafold_db_fetch(AlphaFoldDBFetchInput(uniprot_id="Q0Q0Q0"), AlphaFoldDBFetchConfig())


@pytest.mark.integration
def test_workflow_uniprot_then_afdb_yields_consistent_template():
    """End-to-end workflow mirroring a sequence-for-template lookup.

    Steps:
      1. Resolve KRAS by gene symbol via UniProt (no accession provided).
      2. Use the returned canonical UniProt accession to pull AFDB structure + pLDDT.
      3. Verify the two sources agree on the canonical sequence (a real design
         workflow would feed both into a generator/constraint, so disagreement here
         would silently corrupt every downstream design).

    This is the multi-tool contract the previous tests don't cover: each tool in
    isolation is verified, but the cross-tool sequence-consistency assertion only
    holds if both wrappers normalize the canonical isoform identically.
    """
    # 1. UniProt: resolve KRAS in human. `prefer_pdb_crossref=True` biases the
    #    ranker toward Swiss-Prot reviewed entries (which carry structures);
    #    without it, the default ranking can return an unreviewed TrEMBL hit.
    uniprot = run_uniprot_fetch(
        UniProtFetchInput(target_name="KRAS", organism="Homo sapiens", prefer_pdb_crossref=True),
        UniProtFetchConfig(),
    )
    assert uniprot.success
    assert uniprot.accession == "P01116"  # canonical UniProt accession for human KRAS
    assert "reviewed" in (uniprot.entry_type or "").lower()
    assert uniprot.sequence is not None
    assert uniprot.length == 189  # KRAS canonical length

    # 2. AFDB: pull structure + pLDDT for the resolved accession
    afdb = run_alphafold_db_fetch(
        AlphaFoldDBFetchInput(uniprot_id=uniprot.accession),
        AlphaFoldDBFetchConfig(),
    )
    assert afdb.success
    assert afdb.entry_id == "AF-P01116-F1"

    # 3. Sequence consistency between the two sources
    assert afdb.sequence == uniprot.sequence, (
        "UniProt and AFDB returned different canonical sequences for the same accession; "
        "downstream design tools would silently use mismatched coordinates."
    )
    assert afdb.sequence_length == uniprot.length

    # 4. KRAS Switch I + Switch II loops are well-defined; nucleotide-binding pocket
    #    residues should be high-confidence. Confirm pLDDT is informative (not flat).
    assert afdb.structure is not None
    plddt = afdb.structure.metrics["plddt_per_residue"]
    assert max(plddt) - min(plddt) > 30, "pLDDT array is suspiciously flat for KRAS"


@pytest.mark.integration
def test_alphafold_db_fetch_multi_isoform_picks_canonical(caplog):
    """TP53 (P04637) has multiple annotated isoforms in AFDB; the wrapper must pick the canonical first record.

    The AFDB API response is a list of records: the canonical isoform first
    (entry_id 'AF-P04637-F1', full 393 aa), followed by 8 alternative isoforms
    (entry_id 'AF-P04637-2-F1' through '-9-F1', shorter). This test pins that
    behavior so a future API change that re-orders the response would be caught,
    and asserts that the multi-record warning is emitted naming the canonical
    record so downstream callers can disambiguate.
    """
    import logging

    with caplog.at_level(
        logging.WARNING, logger="proto_tools.tools.database_retrieval.alphafold_db.alphafold_db_fetch"
    ):
        output = run_alphafold_db_fetch(AlphaFoldDBFetchInput(uniprot_id="P04637"), AlphaFoldDBFetchConfig())

    assert output.success
    assert output.entry_id == "AF-P04637-F1"  # canonical: no isoform number, fragment 1
    assert output.sequence_length == 393  # full canonical TP53 length, not a shorter isoform

    # Multi-record warning must name the canonical record so callers know which one was selected
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("AF-P04637-F1" in w and "Other isoforms available" in w for w in warnings), (
        f"expected multi-record warning naming the canonical record; got: {warnings}"
    )


@pytest.mark.integration
def test_alphafold_db_fetch_metadata_only_skips_payload_downloads():
    """include_structure=False returns metadata only.

    This is the canonical batch / probe workflow: hit AFDB to confirm an entry
    exists and grab the URLs + mean pLDDT without paying for the structure
    text (~250 KB) or per-residue pLDDT (~3 KB).
    """
    output = run_alphafold_db_fetch(
        AlphaFoldDBFetchInput(uniprot_id="P04637"),
        AlphaFoldDBFetchConfig(include_structure=False, include_pae=False, include_msa=False),
    )
    assert output.success
    assert output.entry_id == "AF-P04637-F1"
    assert output.mean_plddt is not None  # always in the metadata
    # All URLs are still populated regardless of which payloads were skipped
    assert output.pdb_url and output.cif_url and output.plddt_doc_url and output.pae_doc_url
    # Heavy payloads are skipped — no Structure, no MSA
    assert output.structure is None
    assert output.msa_a3m is None


@pytest.mark.integration
def test_alphafold_db_fetch_isoform_selects_non_canonical():
    """Setting `isoform=2` returns the AF-P04637-2-F1 record, not the canonical."""
    output = run_alphafold_db_fetch(
        AlphaFoldDBFetchInput(uniprot_id="P04637", isoform=2),
        AlphaFoldDBFetchConfig(),
    )
    assert output.success
    assert output.entry_id == "AF-P04637-2-F1"
    # Isoform 2 is shorter than canonical (341 aa vs 393 aa per the live API)
    assert output.sequence_length == 341
    assert output.sequence_length != 393


@pytest.mark.integration
def test_alphafold_db_fetch_invalid_isoform_fails_loudly():
    """Asking for an isoform that doesn't exist raises a clear error."""
    with pytest.raises(Exception, match="Isoform 99 not available"):
        run_alphafold_db_fetch(
            AlphaFoldDBFetchInput(uniprot_id="P04637", isoform=99),
            AlphaFoldDBFetchConfig(),
        )


@pytest.mark.integration
def test_alphafold_db_fetch_structure_composes_with_structure_pipeline():
    """The fetched Structure is drop-in compatible with structure-consuming tools.

    This pins the original motivation of the refactor: anything typed
    ``Structure`` (TM-align, US-align, inverse folding, structure-scoring,
    PyRosetta wrappers) accepts ``output.structure`` directly without an
    intermediate ``Structure(structure=text, structure_format=...)`` wrap.
    """
    output = run_alphafold_db_fetch(AlphaFoldDBFetchInput(uniprot_id="P04637"), AlphaFoldDBFetchConfig())
    assert output.success
    assert isinstance(output.structure, Structure)

    # Lazy gemmi parse must succeed and expose chain content
    sequences = output.structure.get_chain_sequences()
    assert sequences, "expected at least one chain"
    chain_id = next(iter(sequences))
    assert sequences[chain_id] == output.sequence

    # Per-residue pLDDT can also be derived from the B-factor column (normalized to 0-1)
    derived = output.structure.per_residue_plddt
    assert derived is not None
    assert len(derived) == output.sequence_length
    assert all(0.0 <= v <= 1.0 for v in derived)
