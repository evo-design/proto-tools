"""tests/database_retrieval_tests/test_pdb_fetch.py

Tests for the PDB fetch tools (fetch-entry, fetch-fasta)."""

from __future__ import annotations

import pytest

from bio_programming_tools.tools.database_retrieval import (
    PdbFetchConfig,
    PdbFetchEntryInput,
    PdbFetchFastaInput,
    run_pdb_fetch_entry,
    run_pdb_fetch_fasta,
)
from bio_programming_tools.tools.database_retrieval.pdb.shared_data_models import (
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
