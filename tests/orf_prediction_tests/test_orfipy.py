"""Tests for Orfipy ORF prediction tool."""

from typing import get_args

import pytest

from proto_tools.tools.orf_prediction import (
    ORFIPY_TRANSLATION_TABLE_MAP,
    OrfipyConfig,
    OrfipyInput,
    OrfipyOutput,
    OrfipyTranslationTable,
    run_orfipy_prediction,
)
from proto_tools.tools.orf_prediction.orf import ORF
from tests.tool_infra_tests.test_export_functionality import validate_output


def _create_sample_orf(parent_id: str = "seq_0", orf_id: str = "ORF.1") -> ORF:
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


@pytest.mark.parametrize(
    "header, expected",
    [
        (
            "dna_seq_1_ORF.1 [0-180](+) frame:1",
            {"parent_id": "dna_seq_1", "orf_id": "ORF.1", "start": 0, "end": 180, "strand": "+", "frame": 1},
        ),
        (
            "complex-name_ORF.15 [100-250](-) frame:2",
            {"parent_id": "complex-name", "orf_id": "ORF.15", "start": 100, "end": 250, "strand": "-", "frame": 2},
        ),
        ("invalid header", None),
    ],
)
def test_header_parsing(header, expected):
    from proto_tools.tools.orf_prediction.orfipy.standalone.run import _parse_orfipy_header

    assert _parse_orfipy_header(header) == expected


# ── Integration ────────────────────────────────────────────────────────


@pytest.mark.integration
def test_full_workflow():
    inp = OrfipyInput(sequences="ATGGTGCTGAGCCCGGCGGACAAGACCAACGTGAAGGCGGCGTGGGGCAAGTGA")
    result = run_orfipy_prediction(inp, OrfipyConfig(min_len=30))

    validate_output(result)
    assert result.tool_id == "orfipy-prediction"
    assert len(result.predicted_orfs) == 1
    assert result.num_orfs == sum(len(sr) for sr in result.predicted_orfs)

    if result.num_orfs > 0:
        assert isinstance(result.predicted_orfs[0][0], ORF)


@pytest.mark.integration
def test_custom_sequence_ids_preserved():
    inp = OrfipyInput(
        sequences=["ATGAAACCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGTAG"],
        sequence_ids=["my_custom_gene"],
    )
    result = run_orfipy_prediction(inp, OrfipyConfig(min_len=30))

    validate_output(result)
    if result.num_orfs > 0:
        for orf in result.predicted_orfs[0]:
            assert orf.parent_id == "my_custom_gene"


@pytest.mark.integration
def test_multiple_sequences_with_custom_ids():
    inp = OrfipyInput(
        sequences=[
            "ATGCCCAAATTTGGGCCCAAATTTGGGCCCAAATTTGGGTAG",
            "ATGTTTCCCGGGAAATTTCCCGGGTAA",
        ],
        sequence_ids=["gene_a", "gene_b"],
    )
    result = run_orfipy_prediction(inp, OrfipyConfig(min_len=12))

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
    "orfs_per_sequence, expected_total",
    [
        ([], 0),
        ([1], 1),
        ([2, 1, 3], 6),
        ([0, 2, 0], 2),
    ],
)
def test_computed_fields_count(orfs_per_sequence, expected_total):
    predicted_orfs = [
        [_create_sample_orf(f"seq_{i}", f"ORF.{j}") for j in range(count)] for i, count in enumerate(orfs_per_sequence)
    ]
    output = OrfipyOutput(predicted_orfs=predicted_orfs, tool_id="orfipy-prediction", execution_time=0.1, success=True)
    assert output.num_orfs == expected_total


def test_cache_reconstruction():
    """Output works when reconstructed from cache (only predicted_orfs passed)."""
    orfs = [_create_sample_orf("seq_0", "ORF.1"), _create_sample_orf("seq_0", "ORF.2")]
    output = OrfipyOutput(
        tool_id="orfipy-prediction", execution_time=0.0, success=True, warnings=[], metadata={}, predicted_orfs=[orfs]
    )
    assert output.num_orfs == 2
    assert output.predicted_orfs[0][0].orf_id == "ORF.1"
    assert output.predicted_orfs[0][1].orf_id == "ORF.2"


def test_translation_table_map_matches_literal():
    assert set(get_args(OrfipyTranslationTable)) == set(ORFIPY_TRANSLATION_TABLE_MAP.keys())
