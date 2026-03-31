"""tests/database_retrieval_tests/test_uniprot_fetch.py

Tests for the UniProt fetch tool."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from proto_tools.tools.database_retrieval import (
    UniProtFetchConfig,
    UniProtFetchInput,
    run_uniprot_fetch,
)
from proto_tools.tools.database_retrieval.uniprot.uniprot_fetch import (
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
