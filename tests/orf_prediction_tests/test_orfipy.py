"""Tests for Orfipy ORF prediction tool and ORF data model."""

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
from tests.conftest import benchmark_twice, random_dna_sequences
from tests.tool_infra_tests.test_export_functionality import validate_output

_VALID_ORF_KWARGS = {
    "parent_id": "seq_0",
    "orf_id": "gene_1",
    "strand": "+",
    "frame": 1,
    "amino_acid_sequence": "MKT",
    "nucleotide_sequence": "ATGAAGACT",
    "amino_acid_length": 3,
    "nucleotide_length": 9,
    "nucleotide_start": 1,
    "nucleotide_end": 9,
}


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


# ── ORF data model ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"nucleotide_start": 0}, "Must be > 0"),
        ({"nucleotide_end": 0}, "Must be > 0"),
        ({"nucleotide_start": 10, "nucleotide_end": 5}, "must be < end"),
    ],
    ids=["start=0", "end=0", "start>end"],
)
def test_orf_coordinate_validation(overrides, match):
    with pytest.raises(ValueError, match=match):
        ORF(**{**_VALID_ORF_KWARGS, **overrides})


@pytest.mark.parametrize(
    "metrics,nuc_seq,expected",
    [
        ({"gc_content": 55.0}, "ATGAAGACT", 55.0),
        ({}, "GGCC", 100.0),
    ],
    ids=["from-metrics", "computed"],
)
def test_orf_gc_content(metrics, nuc_seq, expected):
    orf = ORF(**{**_VALID_ORF_KWARGS, "metrics": metrics, "nucleotide_sequence": nuc_seq})
    assert orf.gc_content == pytest.approx(expected)


def test_orf_metric_attribute_access_and_missing():
    orf = ORF(**{**_VALID_ORF_KWARGS, "metrics": {"start_type": "ATG"}})
    assert orf.start_type == "ATG"
    with pytest.raises(AttributeError, match="no attribute"):
        _ = orf.nonexistent_field


def test_orf_to_flat_dict():
    orf = ORF(**{**_VALID_ORF_KWARGS, "metrics": {"start_type": "ATG", "score": 0.9}})
    flat = orf.to_flat_dict()
    assert flat["id"] == "seq_0_gene_1"
    assert "gc_content" in flat
    assert flat["start_type"] == "ATG"
    assert flat["score"] == 0.9
    assert "metrics" not in flat


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


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("orfipy-prediction")
@pytest.mark.slow
def test_orfipy_prediction_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark orfipy-prediction: 50 random 10 kbp DNA sequences (cold + warm)."""
    sequences = random_dna_sequences(n=50, length=10_000, seed=0)
    inputs = OrfipyInput(sequences=sequences)
    config = OrfipyConfig(min_len=100)

    result = benchmark_twice(request, "orfipy", lambda: run_orfipy_prediction(inputs, config))
    validate_output(result)

    assert result.tool_id == "orfipy-prediction"
    assert len(result.predicted_orfs) == 50
