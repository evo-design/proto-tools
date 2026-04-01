"""tests/orf_prediction_tests/test_prodigal.py.

Tests for Prodigal ORF prediction tool.
"""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from proto_tools.tools.orf_prediction import (
    ORF,
    ProdigalConfig,
    ProdigalInput,
    ProdigalOutput,
    run_prodigal_prediction,
)
from proto_tools.utils.tool_cache import ToolCache, _program_tool_cache
from tests.tool_infra_tests.test_export_functionality import validate_output

# ── Input validation ───────────────────────────────────────────────────


def test_input_rejects_invalid_dna_characters():
    with pytest.raises(ValidationError, match="Invalid DNA characters"):
        ProdigalInput(input_sequences="ATGCEFGHIJK")


def test_input_rejects_protein_sequence():
    with pytest.raises(ValidationError, match="Invalid DNA characters"):
        ProdigalInput(input_sequences="MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT")


def test_input_accepts_valid_dna():
    inp = ProdigalInput(input_sequences="ATGCGTAAATAG")
    assert inp.input_sequences == ["ATGCGTAAATAG"]


def test_input_uppercases_dna():
    inp = ProdigalInput(input_sequences="atgcgtaaatag")
    assert inp.input_sequences == ["ATGCGTAAATAG"]


def test_input_uppercases_mixed_case():
    inp = ProdigalInput(input_sequences="AtGcGtAaAtAg")
    assert inp.input_sequences == ["ATGCGTAAATAG"]


def test_input_normalizes_string_to_list():
    inp = ProdigalInput(input_sequences="ATGCGT")
    assert isinstance(inp.input_sequences, list)
    assert len(inp.input_sequences) == 1


def test_input_accepts_list_of_sequences():
    inp = ProdigalInput(input_sequences=["ATGCGTAAATAG", "ATGGCATAA"])
    assert inp.input_sequences == ["ATGCGTAAATAG", "ATGGCATAA"]


def test_input_validates_each_sequence_in_list():
    with pytest.raises(ValidationError, match="Invalid DNA characters"):
        ProdigalInput(input_sequences=["ATGCGT", "ATGEFG"])


def test_input_uppercases_all_sequences_in_list():
    inp = ProdigalInput(input_sequences=["atgcgt", "gcataa"])
    assert inp.input_sequences == ["ATGCGT", "GCATAA"]


# ── Integration ────────────────────────────────────────────────────────


@pytest.mark.integration
def test_simple_gene_prediction():
    inp = ProdigalInput(input_sequences="ATGCGTAAATAA")
    result = run_prodigal_prediction(inp, ProdigalConfig())

    validate_output(result)
    assert isinstance(result, ProdigalOutput)
    assert result.num_orfs >= 0
    assert len(result.predicted_orfs) == 1


@pytest.mark.integration
def test_prediction_with_meta_mode_false():
    """Single-genome mode requires longer sequence (100kb+) for training."""
    sequence = "ATGCGTAAATAA" * 8400
    inp = ProdigalInput(input_sequences=sequence)
    result = run_prodigal_prediction(inp, ProdigalConfig(meta_mode=False))

    validate_output(result)
    assert isinstance(result, ProdigalOutput)


@pytest.mark.integration
@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"closed_ends": True},
        {"translation_table": 4},
    ],
    ids=["closed_ends", "translation_table_4"],
)
def test_prediction_with_config_options(config_kwargs):
    inp = ProdigalInput(input_sequences="ATGCGTAAATAA" * 100)
    result = run_prodigal_prediction(inp, ProdigalConfig(**config_kwargs))

    validate_output(result)
    assert isinstance(result, ProdigalOutput)


@pytest.mark.integration
def test_empty_prediction():
    """Sequence with no genes returns zero ORFs."""
    inp = ProdigalInput(input_sequences="AAAAAAAAAA")
    result = run_prodigal_prediction(inp, ProdigalConfig())

    validate_output(result)
    assert isinstance(result, ProdigalOutput)
    assert result.num_orfs >= 0


@pytest.mark.integration
def test_batch_prediction_multiple_sequences():
    sequences = ["ATGCGTAAATAA" * 50, "ATGGCATAA" * 50, "ATGAAACGT" * 50]
    inp = ProdigalInput(input_sequences=sequences)
    result = run_prodigal_prediction(inp, ProdigalConfig())

    validate_output(result)
    assert isinstance(result, ProdigalOutput)
    assert len(result.predicted_orfs) == 3
    total_orfs = sum(len(orfs) for orfs in result.predicted_orfs)
    assert total_orfs == result.num_orfs


@pytest.mark.integration
def test_orf_structure_fields():
    """ORFs have all expected fields with correct types."""
    inp = ProdigalInput(input_sequences="ATGCGTAAATAA" * 50)
    result = run_prodigal_prediction(inp, ProdigalConfig())

    validate_output(result)

    if result.num_orfs > 0 and result.predicted_orfs:
        orf = result.predicted_orfs[0][0]
        assert isinstance(orf, ORF)

        expected_fields = [
            "parent_id",
            "orf_id",
            "amino_acid_sequence",
            "nucleotide_sequence",
            "amino_acid_length",
            "nucleotide_length",
            "nucleotide_start",
            "nucleotide_end",
            "strand",
            "frame",
            "gc_content",
            "start_type",
            "rbs_motif",
            "partial_begin",
            "partial_end",
        ]
        for field in expected_fields:
            assert hasattr(orf, field), f"Missing field: {field}"

        assert isinstance(orf.amino_acid_length, int)
        assert isinstance(orf.nucleotide_length, int)
        assert isinstance(orf.nucleotide_start, int)
        assert isinstance(orf.nucleotide_end, int)
        assert isinstance(orf.gc_content, float)

        # ID format
        assert orf.parent_id.startswith("seq_")
        assert "gene_" in orf.orf_id

        # Strand format
        for o in result.predicted_orfs[0]:
            assert o.strand in ["+", "-"]

        # Protein sequence format
        for o in result.predicted_orfs[0]:
            assert isinstance(o.amino_acid_sequence, str)
            assert len(o.amino_acid_sequence) > 0
            assert not o.amino_acid_sequence.endswith("*")


@pytest.mark.integration
def test_end_to_end_prediction():
    """Realistic sequence fragment with structure validation."""
    sequence = (
        "ATGACCATGATTACGGATTCACTGGCCGTCGTTTTACAACGTCGTGACTGG"
        "GAAAACCCTGGCGTTACCCAACTTAATCGCCTTGCAGCACATCCCCCTTTC"
        "GCCAGCTGGCGTAATAGCGAAGAGGCCCGCACCGATCGCCCTTCCCAACAG"
        "TTGCGCAGCCTGAATGGCGAATGGCGCTTTGCCTGGTTTCCGGCACCAGAA"
        "GCGGTGCCGGAAAGCTGGCTGGAGTGCGATCTTCCTGAGGCCGATACTGTC"
        "GTCGTCCCCTCAAACTGGCAGATGCACGGTTACGATGCGCCCATCTACACC"
        "AACGTGACCTATCCCATTACGGTCAATCCGCCGTTTGTTCCCACGGAGAAT"
        "CCGACGGGTTGTTACTCGCTCACATTTAATGTTGATGAAAGCTGGCTACAG"
        "GAAGGCCAGACGCGAATTATTTTTGATGGCGTTAACTCGGCGTTTCATCTG"
        "TGGTGCAACGGGCGCTGGGTCGGTTACGGCCAGGACAGTCGTTTGCCGTCT"
        "TAA"
    )

    inp = ProdigalInput(input_sequences=sequence)
    result = run_prodigal_prediction(inp, ProdigalConfig(meta_mode=True))

    validate_output(result)
    assert isinstance(result, ProdigalOutput)

    if result.num_orfs > 0 and result.predicted_orfs:
        orf = result.predicted_orfs[0][0]
        assert orf.parent_id.startswith("seq_")
        assert isinstance(orf.amino_acid_sequence, str)
        assert orf.amino_acid_length > 0
        assert orf.strand in ["+", "-"]


@pytest.mark.integration
def test_comparison_with_direct_pyrodigal():
    try:
        import pyrodigal
    except ImportError:
        pytest.skip("pyrodigal not available in base environment")

    sequence = "ATGCGTAAATAA" * 50
    inp = ProdigalInput(input_sequences=sequence)
    our_result = run_prodigal_prediction(inp, ProdigalConfig(meta_mode=True))

    validate_output(our_result)

    gene_finder = pyrodigal.GeneFinder(meta=True)
    direct_genes = gene_finder.find_genes(sequence.encode("utf-8"))
    assert our_result.num_orfs == len(direct_genes)


@pytest.mark.integration
def test_batch_processing_consistency():
    """Batch processing gives same results as individual processing."""
    sequences = ["ATGCGTAAATAA" * 50, "ATGGCATAA" * 50]

    inp_batch = ProdigalInput(input_sequences=sequences)
    config = ProdigalConfig()
    result_batch = run_prodigal_prediction(inp_batch, config)

    results_individual = []
    for seq in sequences:
        inp_single = ProdigalInput(input_sequences=seq)
        result_single = run_prodigal_prediction(inp_single, config)
        results_individual.append(result_single)

    assert result_batch.num_orfs == sum(r.num_orfs for r in results_individual)
    assert len(result_batch.predicted_orfs) == len(sequences)

    for batch_orfs, single_result in zip(result_batch.predicted_orfs, results_individual, strict=False):
        assert len(batch_orfs) == single_result.num_orfs


@pytest.mark.integration
def test_lowercase_input_produces_valid_output():
    inp = ProdigalInput(input_sequences="atgcgtaaataa" * 50)
    result = run_prodigal_prediction(inp, ProdigalConfig())

    validate_output(result)
    assert result.num_orfs >= 0


@pytest.mark.integration
def test_caching_behavior():
    """Results are cached and computation is skipped on second call."""
    cache = ToolCache()
    token = _program_tool_cache.set(cache)

    try:
        sequence = "ATGCGTAAATAA" * 50
        inp = ProdigalInput(input_sequences=sequence)
        config = ProdigalConfig()

        from proto_tools.utils.tool_instance import ToolInstance

        real_dispatch = ToolInstance.dispatch

        with patch.object(ToolInstance, "dispatch", side_effect=real_dispatch, autospec=True) as mock_call:
            result1 = run_prodigal_prediction(inp, config)
            assert result1.success is True
            assert mock_call.call_count == 1

            orfs1 = result1.predicted_orfs[0]

            result2 = run_prodigal_prediction(inp, config)
            assert result2.success is True
            assert mock_call.call_count == 1  # cached

            orfs2 = result2.predicted_orfs[0]
            assert len(orfs1) == len(orfs2)
            if orfs1:
                assert orfs1[0].amino_acid_sequence == orfs2[0].amino_acid_sequence

            inp_diff = ProdigalInput(input_sequences=sequence + "ATGC")
            result3 = run_prodigal_prediction(inp_diff, config)
            assert result3.success is True
            assert mock_call.call_count == 2

    finally:
        _program_tool_cache.reset(token)
