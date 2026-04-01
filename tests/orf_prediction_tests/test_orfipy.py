"""tests/orf_prediction_tests/test_orfipy.py.

Tests for Orfipy ORF prediction tool.
"""

from pathlib import Path

import pytest

from proto_tools.tools.orf_prediction import (
    OrfipyConfig,
    OrfipyInput,
    OrfipyOutput,
    run_orfipy_prediction,
)
from proto_tools.tools.orf_prediction.orf import ORF
from tests.tool_infra_tests.test_export_functionality import validate_output

TEST_DATA_DIR = Path("tests/dummy_data")
ORFIPY_AA_FILE = TEST_DATA_DIR / "test_orfipy_aa.faa"
ORFIPY_NT_FILE = TEST_DATA_DIR / "test_orfipy_nt.fna"


def _create_sample_orf(
    parent_id: str = "seq_0",
    orf_id: str = "ORF.1",
) -> ORF:
    """Helper to create a sample ORF for testing."""
    return ORF(
        parent_id=parent_id,
        orf_id=orf_id,
        strand="+",
        frame=1,
        amino_acid_sequence="MVLS",
        nucleotide_sequence="ATGGTGCTGAGC",
        amino_acid_length=4,
        nucleotide_length=12,
        nucleotide_start=1,
        nucleotide_end=12,
    )


# ── Parsing ────────────────────────────────────────────────────────────


def test_parsing_with_test_data():
    if not ORFIPY_AA_FILE.exists() or not ORFIPY_NT_FILE.exists():
        pytest.skip("Test data files not available")

    from Bio import SeqIO

    from proto_tools.tools.orf_prediction.orfipy.standalone.run import (
        _parse_orfipy_header,
    )

    aa_records = list(SeqIO.parse(str(ORFIPY_AA_FILE), "fasta"))
    nt_records = list(SeqIO.parse(str(ORFIPY_NT_FILE), "fasta"))
    assert len(aa_records) == len(nt_records)

    results = []
    for aa_record, nt_record in zip(aa_records, nt_records, strict=False):
        parsed_info = _parse_orfipy_header(aa_record.description)
        if parsed_info:
            results.append(
                {
                    "parent_id": "dna_seq_1",
                    "orf_id": parsed_info["orf_id"],
                    "strand": parsed_info["strand"],
                    "frame": parsed_info["frame"],
                    "amino_acid_sequence": str(aa_record.seq).rstrip("*"),
                    "nucleotide_sequence": str(nt_record.seq),
                    "amino_acid_length": len(str(aa_record.seq).rstrip("*")),
                    "nucleotide_length": len(str(nt_record.seq)),
                    "nucleotide_start": parsed_info["start"] + 1,
                    "nucleotide_end": parsed_info["end"],
                }
            )

    assert len(results) == 4

    first_row = results[0]
    assert first_row["parent_id"] == "dna_seq_1"
    assert first_row["orf_id"] == "ORF.1"
    assert first_row["amino_acid_sequence"].startswith("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGK")
    assert first_row["nucleotide_sequence"].startswith("ATGGTGCTGAGCCCGGCGGACAAGACCAACGTGAAGGCGGCGTGGGGCAAG")


@pytest.mark.parametrize(
    "header, expected",
    [
        (
            "dna_seq_1_ORF.1 [0-180](+) frame:1",
            {
                "parent_id": "dna_seq_1",
                "orf_id": "ORF.1",
                "start": 0,
                "end": 180,
                "strand": "+",
                "frame": 1,
            },
        ),
        (
            "complex-name_ORF.15 [100-250](-) frame:2",
            {
                "parent_id": "complex-name",
                "orf_id": "ORF.15",
                "start": 100,
                "end": 250,
                "strand": "-",
                "frame": 2,
            },
        ),
        ("invalid header", None),
    ],
)
def test_header_parsing(header, expected):
    from proto_tools.tools.orf_prediction.orfipy.standalone.run import (
        _parse_orfipy_header as parse_header,
    )

    result = parse_header(header)

    if expected:
        assert result is not None
        assert result["parent_id"] == expected["parent_id"]
        assert result["orf_id"] == expected["orf_id"]
        assert result["start"] == expected["start"]
        assert result["end"] == expected["end"]
        assert result["strand"] == expected["strand"]
    else:
        assert result is None


def test_test_data_integrity():
    """AA and NT test data files have matching headers."""
    if not ORFIPY_AA_FILE.exists() or not ORFIPY_NT_FILE.exists():
        pytest.skip("Test data files not available")

    with open(ORFIPY_AA_FILE) as f:
        aa_lines = f.readlines()
    with open(ORFIPY_NT_FILE) as f:
        nt_lines = f.readlines()

    aa_headers = [line for line in aa_lines if line.startswith(">")]
    nt_headers = [line for line in nt_lines if line.startswith(">")]

    assert len(aa_headers) == len(nt_headers), "AA and NT files should have same number of sequences"
    for aa_header, nt_header in zip(aa_headers, nt_headers, strict=False):
        assert aa_header.strip() == nt_header.strip(), (
            f"Headers don't match: {aa_header.strip()} vs {nt_header.strip()}"
        )


# ── Integration ────────────────────────────────────────────────────────


@pytest.mark.integration
def test_full_workflow():
    inp = OrfipyInput(sequences="ATGGTGCTGAGCCCGGCGGACAAGACCAACGTGAAGGCGGCGTGGGGCAAGTGA")
    config = OrfipyConfig(min_len=30)
    result = run_orfipy_prediction(inp, config)

    validate_output(result)
    assert result.tool_id == "orfipy-prediction"
    assert result.num_orfs >= 0
    assert result.predicted_orfs is not None
    assert len(result.predicted_orfs) == 1

    total_orfs_in_list = sum(len(sr) for sr in result.predicted_orfs)
    assert result.num_orfs == total_orfs_in_list

    # Check that predicted_orfs is a list
    assert isinstance(result.predicted_orfs, list)
    assert len(result.predicted_orfs) == 1

    if result.num_orfs > 0:
        first_orf_model = result.predicted_orfs[0][0]
        assert isinstance(first_orf_model, ORF)
        assert first_orf_model.amino_acid_sequence is not None


@pytest.mark.integration
def test_custom_sequence_ids_preserved():
    unique_seq = "ATGAAACCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGTAG"
    inp = OrfipyInput(
        sequences=[unique_seq],
        sequence_ids=["my_custom_gene"],
    )
    config = OrfipyConfig(min_len=30)
    result = run_orfipy_prediction(inp, config)

    validate_output(result)
    if result.num_orfs > 0:
        for orf in result.predicted_orfs[0]:
            assert orf.parent_id == "my_custom_gene"


@pytest.mark.integration
def test_default_sequence_ids_when_not_provided():
    inp = OrfipyInput(sequences=["ATGGTGCTGAGCCCGGCGGACAAGACCAACGTGAAGGCGGCGTGGGGCAAGTGA"])
    config = OrfipyConfig(min_len=30)
    result = run_orfipy_prediction(inp, config)

    validate_output(result)
    if result.num_orfs > 0:
        for orf in result.predicted_orfs[0]:
            assert orf.parent_id == "seq_0"


@pytest.mark.integration
def test_multiple_sequences_with_custom_ids():
    inp = OrfipyInput(
        sequences=[
            "ATGCCCAAATTTGGGCCCAAATTTGGGCCCAAATTTGGGTAG",
            "ATGTTTCCCGGGAAATTTCCCGGGTAA",
        ],
        sequence_ids=["gene_a", "gene_b"],
    )
    config = OrfipyConfig(min_len=12)
    result = run_orfipy_prediction(inp, config)

    validate_output(result)
    assert len(result.predicted_orfs) == 2
    if result.predicted_orfs[0]:
        assert result.predicted_orfs[0][0].parent_id == "gene_a"
    if result.predicted_orfs[1]:
        assert result.predicted_orfs[1][0].parent_id == "gene_b"


def test_sequence_ids_length_mismatch_raises():
    from proto_tools.utils import resolve_sequence_ids

    with pytest.raises(ValueError, match="must match"):
        resolve_sequence_ids(["ATGAAA", "ATGBBB"], ["only_one"])


# ── Computed fields ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "orfs_per_sequence,expected_total",
    [
        ([], 0),
        ([0], 0),
        ([1], 1),
        ([3], 3),
        ([0, 0, 0], 0),
        ([2, 1, 3], 6),
        ([0, 2, 0], 2),
    ],
)
def test_computed_fields_count(orfs_per_sequence, expected_total):
    predicted_orfs = [
        [_create_sample_orf(f"seq_{i}", f"ORF.{j}") for j in range(count)] for i, count in enumerate(orfs_per_sequence)
    ]

    output = OrfipyOutput(
        predicted_orfs=predicted_orfs,
        tool_id="orfipy-prediction",
        execution_time=0.1,
        success=True,
    )

    assert output.num_orfs == expected_total
    # Verify predicted_orfs structure matches expected count
    total_from_list = sum(len(orfs) for orfs in output.predicted_orfs)
    assert total_from_list == expected_total


def test_results_df_columns_and_content():
    output = OrfipyOutput(
        predicted_orfs=[[_create_sample_orf("seq_0", "ORF.1")]],
        tool_id="orfipy-prediction",
        execution_time=0.1,
        success=True,
    )

    # Check ORF object attributes directly
    orf = output.predicted_orfs[0][0]
    assert orf.parent_id == "seq_0"
    assert orf.orf_id == "ORF.1"
    assert orf.amino_acid_sequence == "MVLS"
    assert orf.nucleotide_sequence == "ATGGTGCTGAGC"
    assert orf.strand == "+"
    assert orf.frame == 1
    assert orf.amino_acid_length == 4
    assert orf.nucleotide_length == 12
    assert orf.nucleotide_start == 1
    assert orf.nucleotide_end == 12


def test_cache_reconstruction():
    """Output works when reconstructed from cache (only predicted_orfs passed)."""
    orfs = [_create_sample_orf("seq_0", "ORF.1"), _create_sample_orf("seq_0", "ORF.2")]

    output = OrfipyOutput(
        tool_id="orfipy-prediction",
        execution_time=0.0,
        success=True,
        warnings=[],
        metadata={},
        predicted_orfs=[orfs],
    )

    assert output.num_orfs == 2
    assert len(output.predicted_orfs[0]) == 2
    assert output.predicted_orfs[0][0].orf_id == "ORF.1"
    assert output.predicted_orfs[0][1].orf_id == "ORF.2"
