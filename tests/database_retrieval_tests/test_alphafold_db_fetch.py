"""tests/database_retrieval_tests/test_alphafold_db_fetch.py.

Tests for the AlphaFold DB fetch tool.
"""

from unittest.mock import MagicMock

import pytest

from proto_tools.tools.database_retrieval import (
    AlphaFoldDBFetchConfig,
    AlphaFoldDBFetchInput,
    UniProtFetchConfig,
    UniProtFetchInput,
    run_alphafold_db_fetch,
    run_uniprot_fetch,
)
from proto_tools.tools.database_retrieval.alphafold_db.alphafold_db_fetch import (
    _fetch_pae,
    _fetch_plddt,
    _fetch_prediction,
)


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
    result = _fetch_plddt("https://example/plddt.json", AlphaFoldDBFetchConfig(), session)
    assert result == [42.5, 50.1, 91.0]


def test_fetch_pae_parses_predicted_aligned_error_matrix():
    session = _mock_session([{"predicted_aligned_error": [[0.1, 1.2], [1.2, 0.0]]}])
    result = _fetch_pae("https://example/pae.json", AlphaFoldDBFetchConfig(), session)
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
        fetch_fn("https://example/file.json", AlphaFoldDBFetchConfig(), session)


def test_fetch_prediction_returns_none_on_404():
    """A 404 from the AFDB API short-circuits without parsing the body."""
    session = _mock_session(payload=None, status_code=404)
    result = _fetch_prediction("https://example/api", AlphaFoldDBFetchConfig(), session)
    assert result is None
    session.get.return_value.json.assert_not_called()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_alphafold_db_fetch_full_p04637_record():
    """Default fetch returns a complete record with parseable PDB structure and a length-matched pLDDT."""
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

    # Structure text must parse into a real PDB record, not just contain the literal "ATOM"
    atom_lines = [line for line in output.structure_text.splitlines() if line.startswith("ATOM ")]
    assert len(atom_lines) > 1000  # 393 residues * many atoms each
    # First ATOM line follows the PDB column convention: cols 13-16 = atom name, cols 18-20 = residue name
    first_atom = atom_lines[0]
    assert first_atom[13:16].strip() in {"N", "CA", "C", "O"}
    assert first_atom[17:20].strip() == "MET"  # TP53 starts with methionine

    # pLDDT array must align 1:1 with the sequence
    assert output.plddt_per_residue is not None
    assert len(output.plddt_per_residue) == output.sequence_length
    assert all(0 <= v <= 100 for v in output.plddt_per_residue)
    # Mean of the array equals the reported global mean (within float tolerance)
    derived_mean = sum(output.plddt_per_residue) / len(output.plddt_per_residue)
    assert derived_mean == pytest.approx(output.mean_plddt, abs=0.5)

    # PAE is opt-in
    assert output.pae_matrix is None


@pytest.mark.integration
def test_alphafold_db_fetch_with_pae_returns_square_matrix():
    """PAE matrix must be square with side equal to sequence length."""
    output = run_alphafold_db_fetch(
        AlphaFoldDBFetchInput(uniprot_id="P04637"),
        AlphaFoldDBFetchConfig(include_pae=True),
    )
    assert output.success
    assert output.pae_matrix is not None
    n = output.sequence_length
    assert len(output.pae_matrix) == n
    assert all(len(row) == n for row in output.pae_matrix)
    # Diagonal entries are conventionally near-zero (a residue's error against itself)
    assert all(output.pae_matrix[i][i] < 5.0 for i in range(0, n, 50))


@pytest.mark.integration
def test_alphafold_db_fetch_cif_format():
    """structure_format='cif' returns a parseable mmCIF body."""
    output = run_alphafold_db_fetch(
        AlphaFoldDBFetchInput(uniprot_id="P04637"),
        AlphaFoldDBFetchConfig(structure_format="cif"),
    )
    assert output.success
    assert output.structure_format == "cif"
    assert output.structure_text.startswith("data_")  # mmCIF data block header
    assert "_atom_site." in output.structure_text  # atom loop column prefix


@pytest.mark.integration
def test_alphafold_db_fetch_metadata_only_skips_payload_downloads():
    """Disabling all three payloads still populates URLs but skips structure / pLDDT / PAE bodies."""
    output = run_alphafold_db_fetch(
        AlphaFoldDBFetchInput(uniprot_id="P04637"),
        AlphaFoldDBFetchConfig(include_structure=False, include_plddt=False, include_pae=False),
    )
    assert output.success
    assert output.structure_text is None
    assert output.plddt_per_residue is None
    assert output.pae_matrix is None
    # Metadata still fully populated
    assert output.entry_id == "AF-P04637-F1"
    assert output.pdb_url and output.cif_url and output.plddt_doc_url and output.pae_doc_url


@pytest.mark.integration
def test_alphafold_db_fetch_unknown_accession_returns_failure():
    """A nonexistent accession surfaces as a failed Output with a clear error."""
    output = run_alphafold_db_fetch(AlphaFoldDBFetchInput(uniprot_id="Q0Q0Q0"), AlphaFoldDBFetchConfig())
    assert output.success is False
    assert output.tool_id == "alphafold-db-fetch"
    assert any("AlphaFold DB has no prediction" in err for err in output.errors)


@pytest.mark.integration
def test_workflow_uniprot_then_afdb_yields_consistent_template():
    """End-to-end workflow mirroring prototyper's `fetch_target_sequence_for_template` eval.

    Steps:
      1. Resolve KRAS by gene symbol via UniProt (no accession provided).
      2. Use the returned canonical UniProt accession to pull AFDB structure + pLDDT.
      3. Verify the two sources agree on the canonical sequence (a real prototyper
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
    plddt = afdb.plddt_per_residue
    assert plddt is not None
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
    assert any("AF-P04637-F1" in w and "alternative isoforms or additional fragments" in w for w in warnings), (
        f"expected multi-record warning naming the canonical record; got: {warnings}"
    )
