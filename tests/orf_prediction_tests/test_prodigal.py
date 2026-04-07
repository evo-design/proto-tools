"""Tests for Prodigal ORF prediction tool."""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from proto_tools.tools.orf_prediction import (
    ORF,
    TRANSLATION_TABLE_MAP,
    ProdigalConfig,
    ProdigalInput,
    ProdigalOutput,
    TranslationTable,
    run_prodigal_prediction,
)
from proto_tools.utils.tool_cache import ToolCache, _program_tool_cache
from tests.tool_infra_tests.test_export_functionality import validate_output

# ── Input validation ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "seq",
    ["ATGCEFGHIJK", "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT"],
    ids=["invalid_chars", "protein_sequence"],
)
def test_input_rejects_non_dna(seq):
    with pytest.raises(ValidationError, match="Invalid DNA characters"):
        ProdigalInput(input_sequences=seq)


def test_input_normalizes_and_uppercases():
    inp = ProdigalInput(input_sequences="atGcGtAaAtAg")
    assert inp.input_sequences == ["ATGCGTAAATAG"]


def test_input_accepts_and_validates_list():
    inp = ProdigalInput(input_sequences=["atgcgt", "gcataa"])
    assert inp.input_sequences == ["ATGCGT", "GCATAA"]

    with pytest.raises(ValidationError, match="Invalid DNA characters"):
        ProdigalInput(input_sequences=["ATGCGT", "ATGEFG"])


# ── Config validation ─────────────────────────────────────────────────


@pytest.mark.parametrize("bad_value", ["invalid_table", 11])
def test_config_rejects_bad_translation_table(bad_value):
    """String names required — raw integers and unknown strings both rejected."""
    with pytest.raises(ValidationError):
        ProdigalConfig(translation_table=bad_value)


def test_config_defaults():
    config = ProdigalConfig()
    assert config.translation_table == "bacterial"
    assert config.meta_mode is True


def test_translation_table_map_matches_literal():
    from typing import get_args

    assert set(get_args(TranslationTable)) == set(TRANSLATION_TABLE_MAP.keys())


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
    sequence = "ATGCGTAAATAA" * 8400
    inp = ProdigalInput(input_sequences=sequence)
    result = run_prodigal_prediction(inp, ProdigalConfig(meta_mode=False))

    validate_output(result)
    assert isinstance(result, ProdigalOutput)


@pytest.mark.integration
def test_empty_prediction():
    """Sequence with no genes returns zero ORFs."""
    inp = ProdigalInput(input_sequences="AAAAAAAAAA")
    result = run_prodigal_prediction(inp, ProdigalConfig())

    validate_output(result)
    assert result.num_orfs == 0


@pytest.mark.integration
@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"closed_ends": True},
        {"translation_table": "mycoplasma"},
    ],
    ids=["closed_ends", "translation_table_mycoplasma"],
)
def test_prediction_with_config_options(config_kwargs):
    inp = ProdigalInput(input_sequences="ATGCGTAAATAA" * 100)
    result = run_prodigal_prediction(inp, ProdigalConfig(**config_kwargs))

    validate_output(result)
    assert isinstance(result, ProdigalOutput)


@pytest.mark.integration
def test_batch_prediction():
    sequences = ["ATGCGTAAATAA" * 50, "ATGGCATAA" * 50, "ATGAAACGT" * 50]
    inp = ProdigalInput(input_sequences=sequences)
    result = run_prodigal_prediction(inp, ProdigalConfig())

    validate_output(result)
    assert len(result.predicted_orfs) == 3
    assert sum(len(orfs) for orfs in result.predicted_orfs) == result.num_orfs


@pytest.mark.integration
def test_orf_structure_fields():
    inp = ProdigalInput(input_sequences="ATGCGTAAATAA" * 50)
    result = run_prodigal_prediction(inp, ProdigalConfig())

    validate_output(result)

    if result.num_orfs > 0 and result.predicted_orfs:
        orf = result.predicted_orfs[0][0]
        assert isinstance(orf, ORF)
        assert orf.parent_id.startswith("seq_")
        assert "gene_" in orf.orf_id
        assert orf.strand in ["+", "-"]
        assert isinstance(orf.gc_content, float)
        assert isinstance(orf.amino_acid_sequence, str)
        assert not orf.amino_acid_sequence.endswith("*")


@pytest.mark.integration
def test_end_to_end_prediction():
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
    result = run_prodigal_prediction(inp, ProdigalConfig())

    validate_output(result)
    if result.num_orfs > 0:
        orf = result.predicted_orfs[0][0]
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
    """Batch gives same results as individual processing."""
    sequences = ["ATGCGTAAATAA" * 50, "ATGGCATAA" * 50]

    config = ProdigalConfig()
    result_batch = run_prodigal_prediction(ProdigalInput(input_sequences=sequences), config)

    results_individual = [run_prodigal_prediction(ProdigalInput(input_sequences=seq), config) for seq in sequences]

    assert result_batch.num_orfs == sum(r.num_orfs for r in results_individual)
    for batch_orfs, single_result in zip(result_batch.predicted_orfs, results_individual, strict=False):
        assert len(batch_orfs) == single_result.num_orfs


@pytest.mark.integration
def test_caching_behavior():
    cache = ToolCache()
    token = _program_tool_cache.set(cache)

    try:
        inp = ProdigalInput(input_sequences="ATGCGTAAATAA" * 50)
        config = ProdigalConfig()

        from proto_tools.utils.tool_instance import ToolInstance

        real_dispatch = ToolInstance.dispatch

        with patch.object(ToolInstance, "dispatch", side_effect=real_dispatch, autospec=True) as mock_call:
            result1 = run_prodigal_prediction(inp, config)
            assert result1.success is True
            assert mock_call.call_count == 1

            result2 = run_prodigal_prediction(inp, config)
            assert result2.success is True
            assert mock_call.call_count == 1  # cached

            result3 = run_prodigal_prediction(ProdigalInput(input_sequences="ATGCGTAAATAA" * 50 + "ATGC"), config)
            assert result3.success is True
            assert mock_call.call_count == 2
    finally:
        _program_tool_cache.reset(token)
