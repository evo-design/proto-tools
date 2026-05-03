"""tests/database_retrieval_tests/test_uniprot_fetch.py.

Tests for the UniProt fetch tool.
"""

import pytest
from pydantic import ValidationError

from proto_tools.tools.database_retrieval import (
    UniProtFetchConfig,
    UniProtFetchInput,
    run_uniprot_fetch,
)
from proto_tools.tools.database_retrieval.uniprot.uniprot_fetch import (
    _entry_priority,
    _extract_gene_names,
    _extract_pdb_crossrefs,
)


def test_uniprot_fetch_input_requires_id_or_name():
    with pytest.raises(
        ValidationError,
        match="Provide either uniprot_id or both target_name and organism",
    ):
        UniProtFetchInput()


def test_uniprot_fetch_input_name_without_organism():
    with pytest.raises(
        ValidationError,
        match="Provide either uniprot_id or both target_name and organism",
    ):
        UniProtFetchInput(target_name="TP53")


def test_extract_gene_names():
    entry = {
        "genes": [
            {"geneName": {"value": "TP53"}},
            {"geneName": {"value": "P53"}},
        ]
    }
    names = _extract_gene_names(entry)
    assert "tp53" in names
    assert "p53" in names


def test_extract_gene_names_empty():
    assert _extract_gene_names({}) == set()
    assert _extract_gene_names({"genes": "not_a_list"}) == set()


def test_extract_pdb_crossrefs():
    entry = {
        "uniProtKBCrossReferences": [
            {"database": "PDB", "id": "1TSR"},
            {"database": "PDB", "id": "2OCJ"},
            {"database": "EMBL", "id": "M14694"},
        ]
    }
    pdb_ids = _extract_pdb_crossrefs(entry)
    assert pdb_ids == ["1TSR", "2OCJ"]


def test_extract_pdb_crossrefs_empty():
    assert _extract_pdb_crossrefs({}) == []


def test_entry_priority_distinguishes_reviewed_from_unreviewed():
    """Regression: substring "reviewed" in entryType used to match TrEMBL too.

    The pre-fix substring check `"reviewed" in entry_type` was True for both
    "UniProtKB reviewed (Swiss-Prot)" and "UniProtKB unreviewed (TrEMBL)",
    silently inverting the ranking and promoting longer TrEMBL accessions
    over the canonical Swiss-Prot entry.
    """
    swissprot = {
        "primaryAccession": "P04637",
        "entryType": "UniProtKB reviewed (Swiss-Prot)",
        "genes": [{"geneName": {"value": "TP53"}}],
        "uniProtKBCrossReferences": [{"database": "PDB", "id": "1TUP"}],
    }
    trembl = {
        "primaryAccession": "A0A0U1RQC9",
        "entryType": "UniProtKB unreviewed (TrEMBL)",
        "genes": [{"geneName": {"value": "TP53"}}],
        "uniProtKBCrossReferences": [],
    }
    # Position 2 of the priority tuple is the `reviewed` flag (0/1).
    assert _entry_priority(swissprot, "TP53", prefer_pdb_crossref=False)[2] == 1
    assert _entry_priority(trembl, "TP53", prefer_pdb_crossref=False)[2] == 0
    # And max() picks the Swiss-Prot entry, not the longer TrEMBL accession.
    winner = max(
        [swissprot, trembl],
        key=lambda e: _entry_priority(e, "TP53", prefer_pdb_crossref=False),
    )
    assert winner["primaryAccession"] == "P04637"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_uniprot_fetch_by_accession():
    """Fetch TP53 protein by UniProt accession."""
    inputs = UniProtFetchInput(uniprot_id="P04637")
    output = run_uniprot_fetch(inputs, UniProtFetchConfig())
    assert output.accession == "P04637"
    assert output.sequence is not None
    assert output.sequence.startswith("M")
    assert len(output.pdb_crossrefs) > 0


@pytest.mark.integration
def test_uniprot_fetch_by_name():
    """Search UniProt for lacI in E. coli."""
    inputs = UniProtFetchInput(
        target_name="lacI",
        organism="Escherichia coli",
    )
    output = run_uniprot_fetch(inputs, UniProtFetchConfig())
    assert output.accession
    assert output.sequence is not None
    assert output.sequence.startswith("M")


@pytest.mark.integration
@pytest.mark.parametrize(
    "target_name, organism, expected_accession",
    [
        ("TP53", "Homo sapiens", "P04637"),
        ("KRAS", "Homo sapiens", "P01116"),
        ("BRCA1", "Homo sapiens", "P38398"),
    ],
    ids=["TP53", "KRAS", "BRCA1"],
)
def test_uniprot_fetch_by_name_returns_canonical_swissprot(target_name, organism, expected_accession):
    """Default symbol search returns the canonical Swiss-Prot entry, not a TrEMBL hit.

    Pre-fix, the substring bug surfaced long TrEMBL accessions for these
    common human genes; the ranker now picks Swiss-Prot without needing
    `prefer_pdb_crossref=True`.
    """
    output = run_uniprot_fetch(
        UniProtFetchInput(target_name=target_name, organism=organism),
        UniProtFetchConfig(),
    )
    assert output.success
    assert output.accession == expected_accession
    assert "swiss-prot" in (output.entry_type or "").lower()
