"""tests/database_retrieval_tests/test_sequence_fetch.py

Tests for the multi-source sequence fetch tool."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import bio_programming_tools.tools.database_retrieval.sequence_fetch.sequence_fetch as sf_module
from bio_programming_tools.tools.database_retrieval import (
    SequenceFetchConfig,
    SequenceFetchInput,
    run_sequence_fetch,
)
from bio_programming_tools.tools.database_retrieval.ncbi.shared_data_models import (
    _parse_fasta_records,
)
from bio_programming_tools.tools.database_retrieval.sequence_fetch.sequence_fetch import (
    _ncbi_gene_term,
    _ncbi_term,
    _select_best_record,
)


def test_sequence_fetch_input_normalizes_single_request():
    inputs = SequenceFetchInput(
        requests={
            "target_name": "lacI",
            "organism": "Escherichia coli",
            "sequence_types": "protein",
        }
    )

    assert len(inputs.requests) == 1
    assert inputs.requests[0].sequence_types == ["protein"]


def test_sequence_fetch_rejects_ncrna_as_protein_request():
    inputs = SequenceFetchInput(
        requests=[
            {
                "request_id": "bad_case",
                "target_name": "RyhB sRNA",
                "organism": "Escherichia coli",
                "sequence_types": ["protein"],
            }
        ]
    )

    output = run_sequence_fetch(inputs, SequenceFetchConfig())

    assert len(output.results) == 1
    assert output.results[0].status == "failed"
    assert any("TYPE_MISMATCH" in error for error in output.results[0].errors)


def test_pdb_protein_fetch_filters_dna_chains(monkeypatch):
    """PDB FASTA with DNA chain first must still return protein chain."""

    fake_records = [
        ("1LBG_1|Chain A|Lac operator DNA|Escherichia coli", "GAATTGTGAGCGCTCACAATT"),
        ("1LBG_2|Chain B|Lac repressor|Escherichia coli", "MKPVTLYDVAEYAGVSYQTVSRVVNQASHVSAKTREKVEAAMAELNYIPNR"),
    ]

    def fake_fetch_pdb_fasta(pdb_id, config, session):
        return fake_records

    monkeypatch.setattr(sf_module, "_fetch_pdb_fasta", fake_fetch_pdb_fasta)

    inputs = SequenceFetchInput(
        requests=[
            {
                "request_id": "pdb_test",
                "target_name": "lacI",
                "organism": "Escherichia coli",
                "sequence_types": ["protein"],
                "pdb_id": "1LBG",
            }
        ]
    )

    output = run_sequence_fetch(inputs, SequenceFetchConfig())
    item = output.results[0]

    assert item.status in {"success", "warning"}
    assert len(item.fetched_sequences) == 1

    seq = item.fetched_sequences[0]
    assert seq.sequence_type == "protein"
    assert seq.sequence.startswith("MKPVTLYDVAEYAGVSYQTVSRVVNQASHVSAKTREKVEAAMAELNYIPNR")
    assert "GAATTGTGAGCGCTCACAATT" not in seq.sequence


def test_pdb_protein_fetch_fails_when_no_protein_chains(monkeypatch):
    """PDB FASTA with only DNA/RNA chains must report NOT_FOUND."""

    fake_records = [
        ("1ABC_1|Chain A|DNA|organism", "GAATTGTGAGCGCTCACAATT"),
        ("1ABC_2|Chain B|RNA|organism", "GAUUCGAUUCGAUUCG"),
    ]

    def fake_fetch_pdb_fasta(pdb_id, config, session):
        return fake_records

    monkeypatch.setattr(sf_module, "_fetch_pdb_fasta", fake_fetch_pdb_fasta)

    inputs = SequenceFetchInput(
        requests=[
            {
                "request_id": "no_protein",
                "target_name": "test",
                "organism": "test org",
                "sequence_types": ["protein"],
                "pdb_id": "1ABC",
            }
        ]
    )

    output = run_sequence_fetch(inputs, SequenceFetchConfig())
    item = output.results[0]

    assert item.status == "failed"
    assert any("NOT_FOUND" in e for e in item.errors)


def test_select_best_record_prefers_name_match():
    records = _parse_fasta_records(
        ">rec1 unrelated protein\nAAAA\n"
        ">rec2 lacI repressor\nMMMM\n"
    )
    selected = _select_best_record(records, "lacI")
    assert selected is not None
    assert selected.sequence == "MMMM"


def test_select_best_record_fallback():
    records = _parse_fasta_records(
        ">rec1 some protein\nAAAA\n"
        ">rec2 another protein\nMMMM\n"
    )
    selected = _select_best_record(records, "noMatch")
    assert selected is not None
    assert selected.sequence == "AAAA"


def test_select_best_record_empty():
    assert _select_best_record([], "test") is None


def test_ncbi_term_format():
    term = _ncbi_term("lacI", "Escherichia coli")
    assert '"lacI"[Gene]' in term
    assert '"Escherichia coli"[Organism]' in term


def test_ncbi_gene_term_format():
    term = _ncbi_gene_term("TP53", "Homo sapiens")
    assert '"TP53"[Gene Name]' in term
    assert '"Homo sapiens"[Organism]' in term


# ---------------------------------------------------------------------------
# Integration tests — call real upstream APIs (NCBI, UniProt, PDB)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_sequence_fetch_protein_by_uniprot_id():
    """Fetch lacI protein from UniProt by accession ID."""
    inputs = SequenceFetchInput(
        requests=[
            {
                "request_id": "lacI_protein",
                "target_name": "lacI",
                "organism": "Escherichia coli",
                "sequence_types": ["protein"],
                "uniprot_id": "P0A6X3",
            }
        ]
    )

    output = run_sequence_fetch(inputs, SequenceFetchConfig())

    assert output.num_requests == 1
    item = output.results[0]
    assert item.status in {"success", "warning"}
    assert len(item.fetched_sequences) == 1

    seq = item.fetched_sequences[0]
    assert seq.source_database == "uniprot"
    assert seq.accession == "P0A6X3"
    assert seq.sequence.startswith("M")
    assert seq.length > 50


@pytest.mark.integration
def test_sequence_fetch_protein_and_structure_by_name():
    """Fetch TP53 protein + structure by gene name and organism."""
    inputs = SequenceFetchInput(
        requests=[
            {
                "request_id": "tp53_name_only",
                "target_name": "TP53",
                "organism": "Homo sapiens",
                "sequence_types": ["protein", "structure"],
            }
        ]
    )

    output = run_sequence_fetch(inputs, SequenceFetchConfig())
    item = output.results[0]

    assert item.status in {"success", "warning"}
    assert len(item.fetched_sequences) >= 1
    assert len(item.fetched_structures) >= 1

    # Should resolve a UniProt ID and a PDB ID
    assert "uniprot_id" in item.resolved_ids
    assert "pdb_id" in item.resolved_ids

    seq = item.fetched_sequences[0]
    assert seq.sequence_type == "protein"
    assert seq.sequence.startswith("M")

    struct = item.fetched_structures[0]
    assert struct.pdb_id
    assert struct.source_url


@pytest.mark.integration
def test_sequence_fetch_export_fasta(tmp_path):
    """Fetch a real protein and export to FASTA format."""
    inputs = SequenceFetchInput(
        requests=[
            {
                "request_id": "tp53",
                "target_name": "TP53",
                "organism": "Homo sapiens",
                "sequence_types": ["protein"],
                "uniprot_id": "P04637",
            }
        ]
    )

    output = run_sequence_fetch(inputs, SequenceFetchConfig())
    assert output.results[0].status in {"success", "warning"}

    output.export(name="tp53_batch", export_path=tmp_path, file_format="fasta")

    fasta_path = Path(tmp_path) / "tp53_batch.fasta"
    assert fasta_path.exists()
    content = fasta_path.read_text()
    assert content.startswith(">")
    assert "tp53|TP53|protein" in content
    assert "MEEPQ" in content


@pytest.mark.integration
def test_sequence_fetch_dna_genomic_by_name():
    """Fetch lacI genomic DNA from NCBI by gene name."""
    inputs = SequenceFetchInput(
        requests=[
            {
                "request_id": "lacI_genomic",
                "target_name": "lacI",
                "organism": "Escherichia coli",
                "sequence_types": ["dna_genomic"],
            }
        ]
    )

    output = run_sequence_fetch(inputs, SequenceFetchConfig())
    item = output.results[0]

    assert item.status in {"success", "warning"}
    assert len(item.fetched_sequences) == 1

    seq = item.fetched_sequences[0]
    assert seq.sequence_type == "dna_genomic"
    assert seq.source_database == "ncbi"
    assert re.fullmatch(r"[ATGCNatgcn]+", seq.sequence)
    assert seq.length > 500


@pytest.mark.integration
def test_sequence_fetch_premrna_minus_strand_no_double_rc():
    """Regression: pre-mRNA on minus-strand must NOT double reverse-complement.

    Uses a real NCBI fetch of a short lacI region on the minus strand of
    E. coli K-12 (NC_000913.3). The genomic DNA fetch already asks NCBI
    for strand=2 (reverse complement), so _fetch_rna_premrna must simply
    convert T->U — not RC again.

    We fetch the same region as both dna_genomic and rna_premrna and verify
    that the pre-mRNA is exactly the T->U conversion of the genomic DNA.
    """
    # Small region of lacI on the minus strand of E. coli K-12
    coords = "NC_000913.3:366428-366528:-"

    genomic_inputs = SequenceFetchInput(
        requests=[
            {
                "request_id": "lacI_dna",
                "target_name": "lacI",
                "organism": "Escherichia coli",
                "sequence_types": ["dna_genomic"],
                "genomic_coordinates": coords,
            }
        ]
    )

    premrna_inputs = SequenceFetchInput(
        requests=[
            {
                "request_id": "lacI_premrna",
                "target_name": "lacI",
                "organism": "Escherichia coli",
                "sequence_types": ["rna_premrna"],
                "genomic_coordinates": coords,
            }
        ]
    )

    config = SequenceFetchConfig()

    genomic_output = run_sequence_fetch(genomic_inputs, config)
    premrna_output = run_sequence_fetch(premrna_inputs, config)

    genomic_item = genomic_output.results[0]
    premrna_item = premrna_output.results[0]

    assert genomic_item.status in {"success", "warning"}
    assert premrna_item.status in {"success", "warning"}
    assert len(genomic_item.fetched_sequences) == 1
    assert len(premrna_item.fetched_sequences) == 1

    dna_seq = genomic_item.fetched_sequences[0].sequence
    rna_seq = premrna_item.fetched_sequences[0].sequence

    # pre-mRNA must be exactly T->U of the genomic DNA (same strand)
    expected_rna = dna_seq.replace("T", "U")
    assert rna_seq == expected_rna, (
        f"pre-mRNA should be T->U of genomic DNA.\n"
        f"  DNA:          {dna_seq[:60]}...\n"
        f"  Expected RNA: {expected_rna[:60]}...\n"
        f"  Got RNA:      {rna_seq[:60]}..."
    )
    assert premrna_item.fetched_sequences[0].sequence_type == "rna_premrna"
    assert premrna_item.fetched_sequences[0].inferred is True
