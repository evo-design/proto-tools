"""tests/gene_annotation_tests/test_pyhmmer.py

Tests for the PyHMMER tools."""

from pathlib import Path

import pandas as pd
import pytest
from Bio import SeqIO

from proto_tools.tools.gene_annotation.pyhmmer import (
    PyHmmerOutput,
    PyHmmscanConfig,
    PyHmmscanInput,
    PyHmmsearchConfig,
    PyHmmsearchInput,
    PyJackhmmerConfig,
    PyJackhmmerInput,
    PyNhmmerConfig,
    PyNhmmerInput,
    PyPhmmerConfig,
    PyPhmmerInput,
    run_pyhmmer_hmmscan,
    run_pyhmmer_hmmsearch,
    run_pyhmmer_jackhmmer,
    run_pyhmmer_nhmmer,
    run_pyhmmer_phmmer,
)
from proto_tools.tools.gene_annotation.pyhmmer.shared_data_models import (
    _build_dataframes,
    _convert_dtypes,
)
from tests.tool_infra_tests.test_export_functionality import validate_output

_DATA_DIR = Path(__file__).parent.parent / "dummy_data"
TEST_HMM_FILE = _DATA_DIR / "test_multiple_hmm.hmm"

with open(_DATA_DIR / "test_sequences_for_pyhmmer.fasta") as _f:
    SAMPLE_SEQUENCES = [str(seq.seq) for seq in SeqIO.parse(_f, "fasta")]

with open(_DATA_DIR / "test_dna_sequences.fna") as _f:
    SAMPLE_DNA_SEQUENCES = [str(seq.seq) for seq in SeqIO.parse(_f, "fasta")]


# ── Input validation ─────────────────────────────────────────────────────


def test_hmmsearch_input_single_sequence():
    """Single sequence string is normalized to a list."""
    inputs = PyHmmsearchInput(
        hmm=str(TEST_HMM_FILE), sequences=SAMPLE_SEQUENCES[0]
    )
    assert inputs.sequences == [SAMPLE_SEQUENCES[0]]


def test_hmmsearch_input_list_sequences():
    sequence_list = SAMPLE_SEQUENCES[:2]
    inputs = PyHmmsearchInput(hmm=str(TEST_HMM_FILE), sequences=sequence_list)
    assert inputs.sequences == sequence_list


def test_hmmsearch_input_missing_hmm_file():
    with pytest.raises(ValueError, match="HMM file not found"):
        PyHmmsearchInput(hmm="/nonexistent/test.hmm", sequences=SAMPLE_SEQUENCES)


def test_hmmsearch_input_empty_hmm_path():
    with pytest.raises(ValueError, match="HMM path is not a file"):
        PyHmmsearchInput(hmm="", sequences=SAMPLE_SEQUENCES)


def test_hmmsearch_input_empty_sequences():
    with pytest.raises(ValueError, match="At least one sequence is required"):
        PyHmmsearchInput(hmm=str(TEST_HMM_FILE), sequences=[])


def test_phmmer_input_single_strings():
    """Single strings are normalized to lists."""
    inputs = PyPhmmerInput(
        sequences="MVLSPADKTNVKAAW", target_sequences="ATCGATCGATCGAT"
    )
    assert inputs.sequences == ["MVLSPADKTNVKAAW"]
    assert inputs.target_sequences == ["ATCGATCGATCGAT"]


def test_jackhmmer_config_invalid_max_iterations():
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        PyJackhmmerConfig(max_iterations=0)


# ── Helper functions ─────────────────────────────────────────────────────


def test_convert_dtypes_sequence_format():
    df = pd.DataFrame(
        {
            "target_name": ["test"],
            "evalue": ["1.2e-10"],
            "score": ["45.6"],
            "bias": ["0.1"],
        }
    )
    result = _convert_dtypes(df, is_domain=False)

    assert result["evalue"].dtype == float
    assert result["score"].dtype == float
    assert result["bias"].dtype == float


def test_convert_dtypes_domain_format():
    df = pd.DataFrame(
        {
            "target_name": ["test"],
            "domain_idx": ["1"],
            "c_evalue": ["2.5e-11"],
            "domain_score": ["44.2"],
            "target_length": ["150"],
            "hmm_from": ["5"],
            "hmm_to": ["78"],
        }
    )
    result = _convert_dtypes(df, is_domain=True)

    assert result["domain_idx"].iloc[0] == 1.0
    assert result["c_evalue"].dtype == float
    assert result["domain_score"].dtype == float
    assert result["target_length"].iloc[0] == 150.0
    assert result["hmm_from"].iloc[0] == 5.0
    assert result["hmm_to"].iloc[0] == 78.0


def test_build_dataframes_empty():
    seq_df, dom_df = _build_dataframes([], [])
    assert seq_df is None
    assert dom_df is None


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.include_in_env_report(category="gene_annotation")
def test_pyhmmer_hmmsearch_success():
    inputs = PyHmmsearchInput(
        hmm=str(TEST_HMM_FILE),
        sequences=SAMPLE_SEQUENCES,
    )
    config = PyHmmsearchConfig(
        evalue_threshold=1000.0,
        domain_evalue_threshold=1000.0,
    )
    result = run_pyhmmer_hmmsearch(inputs, config)

    validate_output(result)

    assert isinstance(result, PyHmmerOutput)
    assert result.tool_id == "pyhmmer-hmmsearch"
    assert result.num_sequence_hits == 2
    assert result.num_domain_hits == 2


@pytest.mark.integration
def test_pyhmmer_hmmsearch_no_hits():
    """Very restrictive thresholds should produce no hits."""
    inputs = PyHmmsearchInput(
        hmm=str(TEST_HMM_FILE),
        sequences=SAMPLE_SEQUENCES,
    )
    config = PyHmmsearchConfig(
        evalue_threshold=1e-100,
        domain_evalue_threshold=1e-100,
    )
    result = run_pyhmmer_hmmsearch(inputs, config)

    assert result.success is True
    assert result.num_sequence_hits == 0
    assert result.num_domain_hits == 0


@pytest.mark.integration
def test_pyhmmer_hmmscan_success():
    inputs = PyHmmscanInput(
        hmm_db=str(TEST_HMM_FILE),
        sequences=SAMPLE_SEQUENCES,
    )
    result = run_pyhmmer_hmmscan(inputs, PyHmmscanConfig())

    validate_output(result)


@pytest.mark.integration
def test_pyhmmer_phmmer_success():
    inputs = PyPhmmerInput(
        sequences=SAMPLE_SEQUENCES[:2],
        target_sequences=SAMPLE_SEQUENCES[2:],
    )
    result = run_pyhmmer_phmmer(inputs, PyPhmmerConfig())

    validate_output(result, check_export=False)


@pytest.mark.integration
def test_pyhmmer_nhmmer_success():
    inputs = PyNhmmerInput(
        sequences=SAMPLE_DNA_SEQUENCES[:1],
        target_sequences=SAMPLE_DNA_SEQUENCES,
    )
    config = PyNhmmerConfig(
        evalue_threshold=1000.0,
        domain_evalue_threshold=1000.0,
    )
    result = run_pyhmmer_nhmmer(inputs, config)

    validate_output(result, check_export=False)

    assert isinstance(result, PyHmmerOutput)
    assert result.tool_id == "pyhmmer-nhmmer"
    assert result.num_sequence_hits >= 1


@pytest.mark.integration
def test_pyhmmer_jackhmmer_success():
    inputs = PyJackhmmerInput(
        sequences=SAMPLE_SEQUENCES[:1],
        target_sequences=SAMPLE_SEQUENCES,
    )
    config = PyJackhmmerConfig(
        max_iterations=2,
        evalue_threshold=1000.0,
        domain_evalue_threshold=1000.0,
    )
    result = run_pyhmmer_jackhmmer(inputs, config)

    validate_output(result, check_export=False)

    assert isinstance(result, PyHmmerOutput)
    assert result.tool_id == "pyhmmer-jackhmmer"
    assert result.num_sequence_hits >= 1
    assert result.metadata["max_iterations"] == 2
    assert len(result.metadata["iterations_per_query"]) == 1
