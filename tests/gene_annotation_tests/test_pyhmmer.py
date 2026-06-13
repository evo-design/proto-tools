"""tests/gene_annotation_tests/test_pyhmmer.py.

Tests for the PyHMMER tools.
"""

from pathlib import Path

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
    _build_hit_models,
)
from tests.conftest import benchmark_twice, random_protein_sequences
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
    inputs = PyHmmsearchInput(hmm=str(TEST_HMM_FILE), sequences=SAMPLE_SEQUENCES[0])
    assert inputs.sequences == [SAMPLE_SEQUENCES[0]]


def test_hmmsearch_input_list_sequences():
    sequence_list = SAMPLE_SEQUENCES[:2]
    inputs = PyHmmsearchInput(hmm=str(TEST_HMM_FILE), sequences=sequence_list)
    assert inputs.sequences == sequence_list


def test_hmmsearch_input_empty_sequences():
    with pytest.raises(ValueError, match="At least one sequence is required"):
        PyHmmsearchInput(hmm=str(TEST_HMM_FILE), sequences=[])


def test_phmmer_input_single_strings():
    """Single strings are normalized to lists."""
    inputs = PyPhmmerInput(sequences="MVLSPADKTNVKAAW", target_sequences="ATCGATCGATCGAT")
    assert inputs.sequences == ["MVLSPADKTNVKAAW"]
    assert inputs.target_sequences == ["ATCGATCGATCGAT"]


def test_jackhmmer_config_invalid_max_iterations():
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        PyJackhmmerConfig(max_iterations=0)


# ── Helper functions ─────────────────────────────────────────────────────


def test_build_hit_models_empty():
    seq_hits, dom_hits = _build_hit_models([], [])
    assert seq_hits == []
    assert dom_hits == []


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
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


@pytest.mark.integration
def test_pyhmmer_hmmsearch_bit_cutoffs_filters_hits():
    """``bit_cutoffs='gathering'`` consumes the HMM's GA threshold; the test HMM has GA cutoffs."""
    inputs = PyHmmsearchInput(hmm=str(TEST_HMM_FILE), sequences=SAMPLE_SEQUENCES)
    permissive = run_pyhmmer_hmmsearch(
        inputs,
        PyHmmsearchConfig(evalue_threshold=1000.0, domain_evalue_threshold=1000.0),
    )
    gathering = run_pyhmmer_hmmsearch(inputs, PyHmmsearchConfig(bit_cutoffs="gathering"))
    assert gathering.success
    assert gathering.metadata["bit_cutoffs"] == "gathering"
    # Gathering thresholds are stricter than evalue=1000 → ≤ permissive count
    assert gathering.num_sequence_hits <= permissive.num_sequence_hits


@pytest.mark.integration
def test_pyhmmer_hmmsearch_skip_filters_monotonic():
    """``skip_filters=True`` must return ≥ as many hits as ``False`` at the same E-value.

    Pins the F1=F2=F3=1.0 + bias_filter=False mapping. A regression where any
    of those four kwargs got dropped or typo'd would cause this to fail.
    """
    inputs = PyHmmsearchInput(hmm=str(TEST_HMM_FILE), sequences=SAMPLE_SEQUENCES)
    e_threshold = 100.0  # permissive enough that filter changes affect counts
    filtered = run_pyhmmer_hmmsearch(inputs, PyHmmsearchConfig(skip_filters=False, evalue_threshold=e_threshold))
    unfiltered = run_pyhmmer_hmmsearch(inputs, PyHmmsearchConfig(skip_filters=True, evalue_threshold=e_threshold))
    assert filtered.success and unfiltered.success
    assert unfiltered.num_sequence_hits >= filtered.num_sequence_hits


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark("pyhmmer-hmmscan")
@pytest.mark.slow
def test_pyhmmer_hmmscan_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark pyhmmer-hmmscan: 25000 proteins (2 real + rest random 300-aa) scanned against the test HMM db (cold + warm)."""
    n_queries = 25000
    # Seed with the 2 real sequences that hit the test HMM db so the export is non-empty; the rest are random noise.
    sequences = [*SAMPLE_SEQUENCES, *random_protein_sequences(n=n_queries - len(SAMPLE_SEQUENCES), length=300, seed=0)]
    inputs = PyHmmscanInput(hmm_db=str(TEST_HMM_FILE), sequences=sequences)
    config = PyHmmscanConfig(
        num_threads=4,
        evalue_threshold=1000.0,
        domain_evalue_threshold=1000.0,
    )

    result = benchmark_twice(request, "pyhmmer", lambda: run_pyhmmer_hmmscan(inputs, config))
    validate_output(result)

    assert isinstance(result, PyHmmerOutput)
    assert result.tool_id == "pyhmmer-hmmscan"
    assert result.num_sequence_hits == len(result.sequence_hits)
    assert result.num_domain_hits == len(result.domain_hits)
    assert result.num_sequence_hits >= len(SAMPLE_SEQUENCES)  # the seeded real sequences must hit
    assert result.metadata["num_queries"] == n_queries


@pytest.mark.benchmark("pyhmmer-hmmsearch")
@pytest.mark.slow
def test_pyhmmer_hmmsearch_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark pyhmmer-hmmsearch: multi-HMM file searched against 150000 proteins (2 real + rest random 350aa) (cold + warm)."""
    n_sequences = 150000
    # Seed with the 2 real sequences that hit the test HMM db so the export is non-empty; the rest are random noise.
    sequences = [*SAMPLE_SEQUENCES, *random_protein_sequences(n_sequences - len(SAMPLE_SEQUENCES), 350, seed=0)]
    inputs = PyHmmsearchInput(hmm=str(TEST_HMM_FILE), sequences=sequences)
    config = PyHmmsearchConfig(
        num_threads=0,
        evalue_threshold=1000.0,
        domain_evalue_threshold=1000.0,
    )

    result = benchmark_twice(request, "pyhmmer", lambda: run_pyhmmer_hmmsearch(inputs, config))
    validate_output(result)

    assert isinstance(result, PyHmmerOutput)
    assert result.tool_id == "pyhmmer-hmmsearch"
    assert result.num_sequence_hits == len(result.sequence_hits)
    assert result.num_domain_hits == len(result.domain_hits)
    assert result.num_sequence_hits >= len(SAMPLE_SEQUENCES)  # the seeded real sequences must hit
    assert result.metadata["num_sequences"] == n_sequences
