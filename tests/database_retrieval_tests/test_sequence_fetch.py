"""tests/database_retrieval_tests/test_sequence_fetch.py.

Tests for the multi-source sequence fetch tool.
"""

import re
from pathlib import Path

import pytest

import proto_tools.tools.database_retrieval.sequence_fetch.sequence_fetch as sf_module
from proto_tools.tools.database_retrieval import (
    SequenceFetchConfig,
    SequenceFetchInput,
    run_sequence_fetch,
)
from proto_tools.tools.database_retrieval.ncbi.shared_data_models import (
    _parse_fasta_records,
)
from proto_tools.tools.database_retrieval.sequence_fetch.sequence_fetch import (
    SequenceFetchRequest,
    _ncbi_gene_term,
    _ncbi_term,
    _select_best_record,
    parse_accession,
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


def test_sequence_fetch_request_needs_a_name_and_organism_or_an_identifier():
    with pytest.raises(ValueError, match="Provide both target_name and organism, or an accession"):
        SequenceFetchInput(requests=[{"target_name": "lacI", "sequence_types": ["protein"]}])


@pytest.mark.parametrize(
    ("accession", "expected"),
    [
        ("P04637", ("uniprot", "P04637")),
        ("p04637-2", ("uniprot", "P04637-2")),
        ("P04637.2", ("uniprot", "P04637")),
        ("A0A023GPI8", ("uniprot", "A0A023GPI8")),
        ("2lyz", ("pdb", "2LYZ")),
        ("pdb_00001tsr", ("pdb", "1TSR")),
        ("NC_000017.11", ("genomic", "NC_000017.11")),
        ("NG_017013.2", ("genomic", "NG_017013.2")),
        ("NZ_CP012345.1", ("genomic", "NZ_CP012345.1")),
        ("NZ_AABC02000001.1", ("genomic", "NZ_AABC02000001.1")),
        ("NM_000546.6", ("transcript", "NM_000546.6")),
        ("XR_001234567.1", ("transcript", "XR_001234567.1")),
        ("NP_000537.3", ("protein", "NP_000537.3")),
        ("WP_000123456.1", ("protein", "WP_000123456.1")),
        ("AAA12345.1", ("protein", "AAA12345.1")),
        ("ABC1234567", ("protein", "ABC1234567")),
        ("U49845", ("nucleotide", "U49845")),
        ("AF123456.2", ("nucleotide", "AF123456.2")),
        ("CP01234567", ("nucleotide", "CP01234567")),
        ("AAAA02000001", ("nucleotide", "AAAA02000001")),
        ("7157", ("gene", "7157")),
        ("Gene: 7157", ("gene", "7157")),
        ("uniprot:p04637", ("uniprot", "P04637")),
        ("pdb:1234", ("pdb", "1234")),
    ],
)
def test_parse_accession_reads_the_kind_from_the_format(accession, expected):
    assert parse_accession(accession) == expected


@pytest.mark.parametrize(
    ("accession", "message"),
    [
        ("lysozyme", "Unrecognised accession 'lysozyme'"),
        ("NC_000913.3:1-100", "genomic coordinates go in genomic_coordinates"),
        ("gene:", "needs one accession, without spaces, after its kind prefix"),
        ("gene:TP53", "an NCBI Gene ID is a number"),
    ],
)
def test_parse_accession_rejects_what_it_cannot_route(accession, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        parse_accession(accession)


def test_sequence_fetch_request_takes_at_most_one_accession_of_each_kind():
    with pytest.raises(ValueError, match="at most one uniprot accession per request, got P04637 and P69905"):
        SequenceFetchRequest(accessions=["P04637", "P69905"], sequence_types=["protein"])


def test_sequence_fetch_request_rejects_the_removed_id_fields():
    with pytest.raises(ValueError, match="uniprot_id"):
        SequenceFetchRequest(uniprot_id="P04637", target_name="TP53", organism="Homo sapiens", sequence_types="protein")


@pytest.mark.parametrize(
    ("accessions", "sequence_type", "expected_fetch"),
    [
        (["NP_000537.3"], "protein", ("protein", "NP_000537.3", "fasta")),
        (["AAA12345"], "protein", ("protein", "AAA12345", "fasta")),
        (["NM_000546.6"], "rna_transcript", ("nuccore", "NM_000546.6", "fasta")),
        (["NM_000546.6"], "dna_cds", ("nuccore", "NM_000546.6", "fasta_cds_na")),
        (["NC_000017.11"], "dna_genomic", ("nuccore", "NC_000017.11", "fasta")),
        (["U49845"], "dna_genomic", ("nuccore", "U49845", "fasta")),
        (["U49845"], "rna_transcript", ("nuccore", "U49845", "fasta")),
        (["NC_000017.11", "NM_000546.6"], "dna_genomic", ("nuccore", "NC_000017.11", "fasta")),
        (["NC_000017.11", "NM_000546.6"], "rna_transcript", ("nuccore", "NM_000546.6", "fasta")),
        (["NP_000537.3"], "dna_cds", ("protein", "NP_000537.3", "fasta_cds_na")),
    ],
)
def test_sequence_fetch_routes_each_accession_kind_to_its_ncbi_database(
    monkeypatch, accessions, sequence_type, expected_fetch
):
    fetches = []

    def fake_efetch(db, identifier, rettype, config, session, **kwargs):
        fetches.append((db, identifier, rettype))
        return f">{identifier} test record\nATGAAATAA\n", f"https://example.org/{identifier}"

    monkeypatch.setattr(sf_module, "_ncbi_efetch", fake_efetch)

    output = run_sequence_fetch(
        SequenceFetchInput(requests=[{"accessions": accessions, "sequence_types": sequence_type}])
    )

    assert output.results[0].errors == []
    assert fetches[0] == expected_fetch


@pytest.mark.parametrize("sequence_type", ["rna_transcript", "dna_cds"])
def test_sequence_fetch_never_reads_a_genomic_accession_as_a_transcript(sequence_type):
    output = run_sequence_fetch(
        SequenceFetchInput(requests=[{"accessions": ["NC_000017.11"], "sequence_types": sequence_type}])
    )

    assert output.results[0].status == "failed"
    assert "needs a search by name" in output.results[0].errors[0]


def test_sequence_fetch_takes_genomic_dna_for_a_gene_id_from_its_locus(monkeypatch):
    fetches = []
    monkeypatch.setattr(
        sf_module,
        "_ncbi_esummary",
        lambda db, identifier, config, session: (
            {identifier: {"genomicinfo": [{"chraccver": "NC_000017.11", "chrstart": 7687489, "chrstop": 7668401}]}},
            "https://example.org/esummary",
        ),
    )

    def fake_efetch(db, identifier, rettype, config, session, **kwargs):
        fetches.append((db, identifier, kwargs))
        return f">{identifier} test record\nATGAAATAA\n", f"https://example.org/{identifier}"

    monkeypatch.setattr(sf_module, "_ncbi_efetch", fake_efetch)

    output = run_sequence_fetch(SequenceFetchInput(requests=[{"accessions": "7157", "sequence_types": "dna_genomic"}]))

    assert output.results[0].resolved_ids["gene_id"] == "7157"
    assert fetches == [("nuccore", "NC_000017.11", {"seq_start": 7668402, "seq_stop": 7687490, "strand": "-"})]


def test_sequence_fetch_structure_by_pdb_id_alone(monkeypatch):
    monkeypatch.setattr(
        sf_module,
        "_fetch_pdb_entry",
        lambda pdb_id, session: {"title": "Hen egg-white lysozyme", "method": "X-RAY DIFFRACTION", "resolution": 2.0},
    )

    output = run_sequence_fetch(
        SequenceFetchInput(requests=[{"request_id": "hen", "accessions": "2lyz", "sequence_types": ["structure"]}])
    )

    item = output.results[0]
    assert item.status == "success"
    assert item.fetched_structures[0].pdb_id == "2LYZ"
    assert item.target_name is None


def test_sequence_fetch_says_which_types_need_a_name_when_only_an_identifier_is_given():
    output = run_sequence_fetch(
        SequenceFetchInput(requests=[{"accessions": ["2LYZ"], "sequence_types": ["dna_genomic"]}])
    )

    item = output.results[0]
    assert item.status == "failed"
    assert item.errors == [
        "NOT_FOUND[dna_genomic]: this sequence type needs a search by name, so give target_name and organism, "
        "or an accession for it"
    ]


def test_pdb_protein_fetch_filters_dna_chains(monkeypatch):
    """PDB FASTA with DNA chain first must still return protein chain."""
    fake_records = [
        ("1LBG_1|Chain A|Lac operator DNA|Escherichia coli", "GAATTGTGAGCGCTCACAATT"),
        ("1LBG_2|Chain B|Lac repressor|Escherichia coli", "MKPVTLYDVAEYAGVSYQTVSRVVNQASHVSAKTREKVEAAMAELNYIPNR"),
    ]

    def fake_fetch_pdb_fasta(pdb_id, session):
        return fake_records

    monkeypatch.setattr(sf_module, "_fetch_pdb_fasta", fake_fetch_pdb_fasta)

    inputs = SequenceFetchInput(
        requests=[
            {
                "request_id": "pdb_test",
                "sequence_types": ["protein"],
                "accessions": ["1LBG"],
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

    def fake_fetch_pdb_fasta(pdb_id, session):
        return fake_records

    monkeypatch.setattr(sf_module, "_fetch_pdb_fasta", fake_fetch_pdb_fasta)

    inputs = SequenceFetchInput(
        requests=[
            {
                "request_id": "no_protein",
                "sequence_types": ["protein"],
                "accessions": ["1ABC"],
            }
        ]
    )

    output = run_sequence_fetch(inputs, SequenceFetchConfig())
    item = output.results[0]

    assert item.status == "failed"
    assert any("NOT_FOUND" in e for e in item.errors)


def test_sequence_fetch_raises_when_its_timeout_elapses(monkeypatch):
    """A fetch cut off by the call's timeout fails the call, not just that item."""
    from proto_tools.utils import DeadlineExceededError

    def timed_out(pdb_id, session):
        raise DeadlineExceededError("Timed out requesting PDB: the tool's timeout elapsed")

    monkeypatch.setattr(sf_module, "_fetch_pdb_fasta", timed_out)
    inputs = SequenceFetchInput(
        requests=[
            {
                "request_id": "slow",
                "target_name": "test",
                "organism": "test org",
                "sequence_types": ["protein"],
                "accessions": ["1ABC"],
            }
        ]
    )

    with pytest.raises(TimeoutError):
        run_sequence_fetch(inputs, SequenceFetchConfig())


def test_select_best_record_prefers_name_match():
    records = _parse_fasta_records(">rec1 unrelated protein\nAAAA\n>rec2 lacI repressor\nMMMM\n")
    selected = _select_best_record(records, "lacI")
    assert selected is not None
    assert selected.sequence == "MMMM"


def test_select_best_record_fallback():
    records = _parse_fasta_records(">rec1 some protein\nAAAA\n>rec2 another protein\nMMMM\n")
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
# Integration tests: call real upstream APIs (NCBI, UniProt, PDB)
# ---------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.uses_env
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


@pytest.mark.network
@pytest.mark.uses_env
def test_sequence_fetch_export_fasta(tmp_path):
    """Fetch a protein from UniProt by accession and export it to FASTA format."""
    inputs = SequenceFetchInput(
        requests=[
            {
                "request_id": "tp53",
                "target_name": "TP53",
                "organism": "Homo sapiens",
                "sequence_types": ["protein"],
                "accessions": ["P04637"],
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
    assert seq.accession == "P04637"
    assert seq.sequence.startswith("M")

    output.export(name="tp53_batch", export_path=tmp_path, file_format="fasta")

    fasta_path = Path(tmp_path) / "tp53_batch.fasta"
    assert fasta_path.exists()
    content = fasta_path.read_text()
    assert content.startswith(">")
    assert "tp53|TP53|protein" in content
    assert "MEEPQ" in content


@pytest.mark.network
@pytest.mark.skip_ci
@pytest.mark.uses_env
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


@pytest.mark.network
@pytest.mark.skip_ci
@pytest.mark.uses_env
def test_sequence_fetch_premrna_minus_strand_no_double_rc():
    """Regression: pre-mRNA on minus-strand must NOT double reverse-complement.

    Uses a real NCBI fetch of a short lacI region on the minus strand of
    E. coli K-12 (NC_000913.3). The genomic DNA fetch already asks NCBI
    for strand=2 (reverse complement), so _fetch_rna_premrna must simply
    convert T->U, not RC again.

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


def test_sequence_fetch_config_env_var_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """NCBI_API_KEY / NCBI_EMAIL env vars populate the orchestrator config defaults."""
    monkeypatch.setenv("NCBI_API_KEY", "env-key-123")
    monkeypatch.setenv("NCBI_EMAIL", "env@example.org")
    cfg = SequenceFetchConfig()
    assert cfg.ncbi_api_key == "env-key-123"
    assert cfg.ncbi_email == "env@example.org"
