"""tests/gene_annotation_tests/test_mmseqs.py

Tests for the MMseqs2 tools."""

from pathlib import Path

import pandas as pd
import pytest

from bio_programming_tools.tools.gene_annotation.mmseqs import (
    MmseqsClusteringConfig,
    MmseqsClusteringInput,
    MmseqsClusteringOutput,
    MmseqsClusterResult,
    MmseqsHit,
    MmseqsSearchGenomesConfig,
    MmseqsSearchGenomesInput,
    MmseqsSearchGenomesOutput,
    MmseqsSearchProteinsConfig,
    MmseqsSearchProteinsInput,
    MmseqsSearchProteinsOutput,
    MmseqsSequenceSearchResult,
    run_mmseqs_clustering,
    run_mmseqs_search_genomes,
    run_mmseqs_search_proteins,
)
from bio_programming_tools.tools.gene_annotation.mmseqs.search_proteins import (
    _build_sequence_search_results,
    _filter_top_hits,
    _parse_m8_output,
)
from tests.tool_infra_tests.test_export_functionality import validate_output

TEST_DATA_DIR = Path("tests/dummy_data")
PROTEIN_FASTA = TEST_DATA_DIR / "test_protein_sequences.faa"
M8_FILE = TEST_DATA_DIR / "test_mmseqs_results.m8"


# ── Schema properties ────────────────────────────────────────────────────


def test_sequence_search_result_properties():
    """Computed properties: num_hits, has_hits, top_hit."""
    hits = [
        MmseqsHit(target_id="db_1", pident=95.0, evalue=1e-50),
        MmseqsHit(target_id="db_2", pident=85.0, evalue=1e-30),
    ]
    result = MmseqsSequenceSearchResult(
        query_id="seq_0", query_sequence="MVLSPADKTN", hits=hits,
    )
    assert result.num_hits == 2
    assert result.has_hits is True
    assert result.top_hit.pident == 95.0


def test_sequence_search_result_no_hits():
    result = MmseqsSequenceSearchResult(
        query_id="seq_0", query_sequence="MVLSPADKTN", hits=[],
    )
    assert result.num_hits == 0
    assert result.has_hits is False
    assert result.top_hit is None


def test_search_proteins_output_iteration():
    results = [
        MmseqsSequenceSearchResult(query_id="seq_0", query_sequence="SEQ1", hits=[]),
        MmseqsSequenceSearchResult(query_id="seq_1", query_sequence="SEQ2", hits=[]),
    ]
    output = MmseqsSearchProteinsOutput(results=results, metadata={})

    assert len(output) == 2
    assert output[0].query_sequence == "SEQ1"
    assert output[1].query_sequence == "SEQ2"
    assert list(output) == results
    assert output.total_hits == 0


def test_clustering_output_representative_indices():
    results = [
        MmseqsClusterResult(sequence_id="seq_0", input_sequence="S1", cluster_id="seq_0", is_representative=True),
        MmseqsClusterResult(sequence_id="seq_1", input_sequence="S2", cluster_id="seq_0", is_representative=False),
        MmseqsClusterResult(sequence_id="seq_2", input_sequence="S3", cluster_id="seq_2", is_representative=True),
    ]
    output = MmseqsClusteringOutput(results=results, metadata={})

    assert output.representative_indices == [0, 2]
    assert output.num_clusters == 2


# ── Helper functions ─────────────────────────────────────────────────────


def test_parse_m8_valid_string():
    if not M8_FILE.exists():
        pytest.skip("Test data file not found")

    raw_output = M8_FILE.read_text()
    df = _parse_m8_output(raw_output)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 6
    assert list(df.columns) == ["query", "target", "pident", "evalue"]
    assert "protein_seq_1" in df["query"].values
    assert "protein_seq_2" in df["query"].values


def test_parse_m8_empty_string():
    df = _parse_m8_output("")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == ["query", "target", "pident", "evalue"]


def test_parse_m8_whitespace_only():
    df = _parse_m8_output("   \n\n  ")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == ["query", "target", "pident", "evalue"]


def test_filter_top_hits_by_highest_pident():
    data = {
        "query": ["q1", "q1", "q2", "q2", "q3"],
        "target": ["t1", "t2", "t3", "t4", "t5"],
        "evalue": [1e-10, 1e-20, 1e-5, 1e-5, 1e-100],
        "pident": [90.0, 80.0, 95.0, 98.0, 99.0],
    }
    df = pd.DataFrame(data)
    filtered_df = _filter_top_hits(df)

    assert len(filtered_df) == 3
    assert filtered_df[filtered_df["query"] == "q1"].pident.iloc[0] == 90.0
    assert filtered_df[filtered_df["query"] == "q2"].pident.iloc[0] == 98.0
    assert filtered_df[filtered_df["query"] == "q3"].pident.iloc[0] == 99.0


def test_filter_top_hits_empty_dataframe():
    df = pd.DataFrame(columns=["query", "target", "pident", "evalue"])
    filtered_df = _filter_top_hits(df)
    assert isinstance(filtered_df, pd.DataFrame)
    assert len(filtered_df) == 0


def test_filter_top_hits_single_hit_per_query():
    data = {"query": ["q1", "q2", "q3"], "target": ["t1", "t2", "t3"], "pident": [90.0, 85.0, 95.0], "evalue": [1e-10, 1e-15, 1e-20]}
    df = pd.DataFrame(data)
    filtered_df = _filter_top_hits(df)
    assert len(filtered_df) == 3


def test_filter_top_hits_ties_in_pident():
    data = {"query": ["q1", "q1", "q1"], "target": ["t1", "t2", "t3"], "pident": [90.0, 90.0, 85.0], "evalue": [1e-10, 1e-15, 1e-20]}
    df = pd.DataFrame(data)
    filtered_df = _filter_top_hits(df)
    # Should keep one of the 90.0% hits (first by pandas default)
    assert len(filtered_df) == 1
    assert filtered_df.iloc[0]["pident"] == 90.0


def test_build_results_with_hits():
    sequences = ["MVLSPADKTN", "MKLLVVAAAA"]
    sequence_ids = ["seq_0", "seq_1"]
    df = pd.DataFrame({
        "query": ["seq_0", "seq_0", "seq_1"],
        "target": ["db_1", "db_2", "db_3"],
        "pident": [95.0, 85.0, 90.0],
        "evalue": [1e-50, 1e-30, 1e-40],
    })

    results = _build_sequence_search_results(sequences, sequence_ids, df)

    assert len(results) == 2
    assert results[0].query_id == "seq_0"
    assert results[0].query_sequence == "MVLSPADKTN"
    assert results[0].num_hits == 2
    assert results[0].top_hit.pident == 95.0
    assert results[1].query_id == "seq_1"
    assert results[1].query_sequence == "MKLLVVAAAA"
    assert results[1].num_hits == 1


def test_build_results_no_hits():
    sequences = ["MVLSPADKTN", "MKLLVVAAAA"]
    sequence_ids = ["seq_0", "seq_1"]
    df = pd.DataFrame(columns=["query", "target", "pident", "evalue"])

    results = _build_sequence_search_results(sequences, sequence_ids, df)

    assert len(results) == 2
    assert all(r.num_hits == 0 for r in results)
    assert all(r.top_hit is None for r in results)


def test_build_results_partial_hits():
    """Only some sequences have hits; missing ones get empty results."""
    sequences = ["SEQ1", "SEQ2", "SEQ3"]
    sequence_ids = ["seq_0", "seq_1", "seq_2"]
    df = pd.DataFrame({
        "query": ["seq_0", "seq_2"],
        "target": ["db_1", "db_2"],
        "pident": [95.0, 90.0],
        "evalue": [1e-50, 1e-40],
    })

    results = _build_sequence_search_results(sequences, sequence_ids, df)

    assert len(results) == 3
    assert results[0].num_hits == 1
    assert results[1].num_hits == 0
    assert results[2].num_hits == 1


def test_build_results_with_custom_ids():
    sequences = ["MVLSPADKTN", "MKLLVVAAAA"]
    sequence_ids = ["protein_a", "protein_b"]
    df = pd.DataFrame({
        "query": ["protein_a", "protein_b"],
        "target": ["db_1", "db_2"],
        "pident": [95.0, 90.0],
        "evalue": [1e-50, 1e-40],
    })

    results = _build_sequence_search_results(sequences, sequence_ids, df)

    assert len(results) == 2
    assert results[0].query_id == "protein_a"
    assert results[1].query_id == "protein_b"


# ── M8 workflow (pure Python, local test data) ───────────────────────────


def test_workflow_with_real_test_data():
    """End-to-end parse → filter pipeline on local M8 file."""
    if not M8_FILE.exists() or not PROTEIN_FASTA.exists():
        pytest.skip("Test data files not found")

    raw_output = M8_FILE.read_text()
    df = _parse_m8_output(raw_output)
    filtered_df = _filter_top_hits(df)

    assert len(filtered_df) <= len(df)
    query_counts = filtered_df.groupby("query").size()
    assert all(count == 1 for count in query_counts)


def test_column_names_consistency():
    """Column names match MMseqs2 M8 format."""
    if not M8_FILE.exists():
        pytest.skip("Test data file not found")

    raw_output = M8_FILE.read_text()
    df = _parse_m8_output(raw_output)
    assert list(df.columns) == ["query", "target", "pident", "evalue"]


# ── Input validation ─────────────────────────────────────────────────────


def test_search_proteins_input_valid(tmp_path):
    db_file = tmp_path / "database.faa"
    db_file.write_text(">db1\nMVLSPADKTN\n")

    inputs = MmseqsSearchProteinsInput(
        query_sequences=["MVLSPADKTN", "MKLLVVAAAA"],
        mmseqs_db=str(db_file),
    )
    assert len(inputs.query_sequences) == 2


def test_search_proteins_input_rejects_empty(tmp_path):
    db_file = tmp_path / "database.faa"
    db_file.write_text(">db1\nMVLSPADKTN\n")

    with pytest.raises(ValueError, match="query_sequences list cannot be empty"):
        MmseqsSearchProteinsInput(
            query_sequences=[], mmseqs_db=str(db_file),
        )


def test_search_proteins_input_rejects_non_list(tmp_path):
    db_file = tmp_path / "database.faa"
    db_file.write_text(">db1\nMVLSPADKTN\n")

    with pytest.raises(ValueError, match="query_sequences must be a list"):
        MmseqsSearchProteinsInput(
            query_sequences="single_string", mmseqs_db=str(db_file),
        )


def test_search_genomes_input_rejects_empty_query():
    with pytest.raises(ValueError, match="query_genomes list cannot be empty"):
        MmseqsSearchGenomesInput(query_genomes=[], target_genomes=["ATGC"])


def test_search_genomes_input_rejects_empty_target():
    with pytest.raises(ValueError, match="target_genomes list cannot be empty"):
        MmseqsSearchGenomesInput(query_genomes=["ATGC"], target_genomes=[])


def test_clustering_input_rejects_empty():
    with pytest.raises(ValueError, match="input_sequences list cannot be empty"):
        MmseqsClusteringInput(input_sequences=[])


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.include_in_env_report(category="gene_annotation")
def test_mmseqs_search_proteins_execution(tmp_path):
    db_file = tmp_path / "database.faa"
    db_file.write_text(">db1\nMVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT\n")

    inputs = MmseqsSearchProteinsInput(
        query_sequences=[
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",
        ],
        mmseqs_db=str(db_file),
    )
    config = MmseqsSearchProteinsConfig(threads=2)
    result = run_mmseqs_search_proteins(inputs, config)

    validate_output(result)

    assert isinstance(result, MmseqsSearchProteinsOutput)
    assert len(result) == 2
    assert result[0].num_hits >= 1
    assert result[1].num_hits >= 1


@pytest.mark.integration
def test_mmseqs_search_proteins_no_hits(tmp_path):
    db_file = tmp_path / "database.faa"
    db_file.write_text(">db1\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n")

    inputs = MmseqsSearchProteinsInput(
        query_sequences=["WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW"],
        mmseqs_db=str(db_file),
    )
    config = MmseqsSearchProteinsConfig(threads=2)
    result = run_mmseqs_search_proteins(inputs, config)

    assert isinstance(result, MmseqsSearchProteinsOutput)
    assert result.success is True
    assert len(result) == 1
    assert result[0].num_hits == 0


@pytest.mark.integration
def test_mmseqs_clustering_execution():
    sequences = [
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",  # Identical to first
        "MKLLVVAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",  # Different
    ]

    inputs = MmseqsClusteringInput(input_sequences=sequences)
    config = MmseqsClusteringConfig(min_seq_id=0.95)
    result = run_mmseqs_clustering(inputs, config)

    validate_output(result)

    assert isinstance(result, MmseqsClusteringOutput)
    assert len(result) == 3
    assert result.num_clusters == 2

    assert result[0].cluster_id == result[1].cluster_id
    assert result[2].cluster_id != result[0].cluster_id


@pytest.mark.integration
def test_mmseqs_clustering_all_identical():
    sequences = [
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",
    ]

    inputs = MmseqsClusteringInput(input_sequences=sequences)
    config = MmseqsClusteringConfig(min_seq_id=0.95)
    result = run_mmseqs_clustering(inputs, config)

    assert result.num_clusters == 1
    assert all(r.cluster_id == result[0].cluster_id for r in result)
    assert len(result.representative_indices) == 1


@pytest.mark.skip_ci
@pytest.mark.integration
def test_mmseqs_search_genomes_execution():
    query_seqs = [
        "ATGGTGCTGTCTCCTGCCGACAAGACCAACGTCAAGGCCGCCTGGGGTAAGGTCATGGTGCTGTCTCCTGCCGACAAGACCAACGTCAAGGCCGCCTGGGGTAAGGTC",
        "ATGAAGCTGCTGGTGGTGGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCATGAAGCTGCTGGTGGTGGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCC",
    ]
    target_seqs = [
        "ATGGTGCTGTCTCCTGCCGACAAGACCAACGTCAAGGCCGCCTGGGGTAAGGTCATGGTGCTGTCTCCTGCCGACAAGACCAACGTCAAGGCCGCCTGGGGTAAGGTC",
        "ATGAAGCTGCTGGTGGTGGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCATGAAGCTGCTGGTGGTGGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCC",
    ]

    inputs = MmseqsSearchGenomesInput(query_genomes=query_seqs, target_genomes=target_seqs)
    config = MmseqsSearchGenomesConfig()
    result = run_mmseqs_search_genomes(inputs, config)

    validate_output(result)

    assert isinstance(result, MmseqsSearchGenomesOutput)
    assert len(result) == 2


@pytest.mark.integration
def test_single_sequence_protein_search(tmp_path):
    db_file = tmp_path / "database.faa"
    db_file.write_text(">db1\nMVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT\n")

    inputs = MmseqsSearchProteinsInput(
        query_sequences=["MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT"],
        mmseqs_db=str(db_file),
    )
    config = MmseqsSearchProteinsConfig(threads=2)
    result = run_mmseqs_search_proteins(inputs, config)

    assert len(result) == 1
    assert result[0].num_hits >= 1


@pytest.mark.integration
def test_single_sequence_clustering():
    inputs = MmseqsClusteringInput(
        input_sequences=["MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT"],
    )
    config = MmseqsClusteringConfig(min_seq_id=0.95)
    result = run_mmseqs_clustering(inputs, config)

    assert len(result) == 1
    assert result.num_clusters == 1
    assert result[0].is_representative is True
