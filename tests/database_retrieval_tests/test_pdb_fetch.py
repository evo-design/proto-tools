"""tests/database_retrieval_tests/test_pdb_fetch.py.

Tests for the PDB fetch tools (fetch-entry, fetch-fasta).
"""

import pytest

from proto_tools.tools.database_retrieval import (
    PdbFetchConfig,
    PdbFetchEntryInput,
    PdbFetchFastaInput,
    run_pdb_fetch_entry,
    run_pdb_fetch_fasta,
)
from proto_tools.tools.database_retrieval.pdb.shared_data_models import (
    _chain_ids_from_header,
    _is_protein_sequence,
)


def test_is_protein_sequence_protein():
    assert _is_protein_sequence("MKPVTLYDVAEYAGVSYQTV") is True


def test_is_protein_sequence_dna():
    assert _is_protein_sequence("ATGCATGCATGC") is False


def test_is_protein_sequence_rna():
    assert _is_protein_sequence("AUGCAUGCAUGC") is False


def test_is_protein_sequence_empty():
    assert _is_protein_sequence("") is False


@pytest.mark.parametrize(
    "header,expected",
    [
        (
            "1LBG_2|Chains E[auth A], F[auth B], G[auth C], H[auth D]|PROTEIN|Escherichia coli (562)",
            ["A", "B", "C", "D"],
        ),
        ("1ABC_1|Chains A, B|PROTEIN", ["A", "B"]),
    ],
    ids=["with-auth-markers", "bare-labels"],
)
def test_chain_ids_from_header(header: str, expected: list[str]) -> None:
    """Pin the chain-id parser: prefer [auth X], fall back to bare letters."""
    assert _chain_ids_from_header(header) == expected


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_pdb_fetch_entry_metadata():
    """Fetch PDB entry metadata for a known structure."""
    inputs = PdbFetchEntryInput(pdb_id="1LBG")
    output = run_pdb_fetch_entry(inputs, PdbFetchConfig())
    assert output.title is not None
    assert output.method is not None
    assert output.source_url == "https://data.rcsb.org/rest/v1/core/entry/1LBG"


@pytest.mark.integration
def test_pdb_fetch_fasta():
    """Fetch PDB FASTA chains for a known structure."""
    inputs = PdbFetchFastaInput(pdb_id="1LBG")
    output = run_pdb_fetch_fasta(inputs, PdbFetchConfig())
    assert len(output.chains) > 0
    protein_chains = [c for c in output.chains if c.is_protein]
    assert len(protein_chains) > 0
