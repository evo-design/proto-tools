"""tests/causal_models_tests/test_evo2.py.

Tests for Evo2.
"""

import numpy as np
import pytest

from proto_tools.tools.causal_models.evo2 import (
    Evo2SampleConfig,
    Evo2SampleInput,
    Evo2ScoringConfig,
    Evo2ScoringInput,
    run_evo2_sample,
    run_evo2_score,
)
from proto_tools.utils.standalone_helpers_source.standalone_helpers.serialization import DNA_NUCLEOTIDES
from tests.conftest import benchmark_twice, make_persistent_fixture, random_dna_sequences
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output

_persistent_tool = make_persistent_fixture("evo2")


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.uses_gpu
def test_evo2_sample_tool():
    """Test the evo2 sampling tool with run_evo2_sample."""
    prompts = ["ATCG", "GCTA"]
    inputs = Evo2SampleInput(prompts=prompts)
    config = Evo2SampleConfig(
        model_checkpoint="evo2_7b",
        max_new_tokens=50,
        temperature=0.8,
        top_k=4,
        top_p=1.0,
        cached_generation=True,
        verbose=False,
    )

    result = run_evo2_sample(inputs=inputs, config=config)
    validate_output(result)

    assert result.tool_id == "evo2-sample"
    assert result.metadata["model_checkpoint"] == "evo2_7b"
    assert result.metadata["max_new_tokens"] == 50
    assert result.metadata["temperature"] == 0.8
    assert len(result.sequences) == 2

    valid_chars = set("ATCGN")
    for i, (prompt, seq) in enumerate(zip(prompts, result.sequences, strict=False)):
        assert seq.startswith(prompt), f"Sequence {i} should start with '{prompt}'"
        assert set(seq.upper()).issubset(valid_chars)
        assert len(seq) > len(prompt), f"Sequence {i} should be longer than prompt"


@pytest.mark.uses_gpu
def test_evo2_sample_prepend_prompt():
    """Test that prepend_prompt controls whether prompt is included."""
    prompt = "ATCGATCG"

    result_with = run_evo2_sample(
        inputs=Evo2SampleInput(prompts=[prompt]),
        config=Evo2SampleConfig(
            model_checkpoint="evo2_7b",
            max_new_tokens=50,
            prepend_prompt=True,
            verbose=False,
        ),
    )
    assert result_with.sequences[0].startswith(prompt)

    result_without = run_evo2_sample(
        inputs=Evo2SampleInput(prompts=[prompt]),
        config=Evo2SampleConfig(
            model_checkpoint="evo2_7b",
            max_new_tokens=50,
            prepend_prompt=False,
            verbose=False,
        ),
    )
    assert not result_without.sequences[0].startswith(prompt)
    assert len(result_without.sequences[0]) > 0


# ── Batched sampling tests ────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_evo2_sample_batched_tool():
    """Test batched sampling with tool layer (run_evo2_sample)."""
    prompts = ["ATCG", "GCTA", "AAAA", "GGGG", "CCCC", "TTTT"]
    inputs = Evo2SampleInput(prompts=prompts)
    config = Evo2SampleConfig(
        model_checkpoint="evo2_7b",
        max_new_tokens=50,
        temperature=1.0,
        batch_size=2,
        verbose=False,
    )

    result = run_evo2_sample(inputs=inputs, config=config)
    validate_output(result)

    assert len(result.sequences) == 6

    for i, (prompt, seq) in enumerate(zip(prompts, result.sequences, strict=False)):
        assert seq.startswith(prompt), f"Sequence {i} should start with '{prompt}'"
        assert len(seq) > len(prompt)


# ── Scoring tests ─────────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_evo2_score_metrics_consistency():
    """Test that scoring metrics are mathematically consistent."""
    inputs = Evo2ScoringInput(sequences=["ATCGATCGATCGATCG"])
    config = Evo2ScoringConfig(model_checkpoint="evo2_7b", verbose=False, return_logits=True)

    result = run_evo2_score(inputs=inputs, config=config)
    score = result.scores[0]

    expected_perplexity = np.exp(-score.avg_log_likelihood)
    np.testing.assert_allclose(score.perplexity, expected_perplexity, rtol=1e-5)

    assert score.avg_log_likelihood <= 0, "avg_log_likelihood should be <= 0"
    assert score.log_likelihood <= score.avg_log_likelihood, "log_likelihood should be <= avg_log_likelihood"


@pytest.mark.uses_gpu
def test_evo2_score_batched_tool():
    """Test batched scoring with tool layer (run_evo2_score)."""
    sequences = ["ATCGATCG", "GCTAGCTA", "AAAACCCC", "TTTTGGGG", "CCCCAAAA", "GGGGTTTT"]
    inputs = Evo2ScoringInput(sequences=sequences)
    config = Evo2ScoringConfig(
        model_checkpoint="evo2_7b",
        batch_size=2,
        verbose=False,
        return_logits=True,
    )

    result = run_evo2_score(inputs=inputs, config=config)
    validate_output(result)
    assert_metrics_in_spec(result)

    assert len(result.scores) == 6

    for _seq, score in zip(sequences, result.scores, strict=False):
        assert score.logits is not None, "Logits should be present when return_logits=True"
        assert len(score.logits) > 0
        assert len(score.logits[0]) >= 4


@pytest.mark.uses_gpu
def test_evo2_score_batch_size_consistency():
    """Test that different batch_sizes produce consistent results.

    Note: Evo2 uses a Hyena-based state-space architecture which can have small
    numerical differences across batch sizes due to internal state computations.
    We use a relaxed tolerance (1%) compared to transformer models like ProGen2.
    """
    sequences = ["ATCGATCG", "GCTAGCTA", "AAAACCCC", "TTTTGGGG"]
    inputs = Evo2ScoringInput(sequences=sequences)

    results = {
        bs: run_evo2_score(
            inputs=inputs,
            config=Evo2ScoringConfig(model_checkpoint="evo2_7b", batch_size=bs, verbose=False),
        )
        for bs in [1, 2, 4]
    }

    for i in range(4):
        perplexities = [results[bs].scores[i].perplexity for bs in [1, 2, 4]]
        log_likelihoods = [results[bs].scores[i].log_likelihood for bs in [1, 2, 4]]

        np.testing.assert_allclose(perplexities[0], perplexities[1], rtol=1e-2)
        np.testing.assert_allclose(perplexities[0], perplexities[2], rtol=1e-2)
        np.testing.assert_allclose(log_likelihoods[0], log_likelihoods[1], rtol=1e-2)
        np.testing.assert_allclose(log_likelihoods[0], log_likelihoods[2], rtol=1e-2)


@pytest.mark.uses_gpu
def test_evo2_score_tool():
    """Test the evo2 scoring tool with run_evo2_score."""
    sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA"]
    inputs = Evo2ScoringInput(sequences=sequences)
    config = Evo2ScoringConfig(
        model_checkpoint="evo2_7b",
        verbose=False,
        return_logits=True,
    )

    result = run_evo2_score(inputs=inputs, config=config)
    validate_output(result)
    assert_metrics_in_spec(result)

    assert result.tool_id == "evo2-score", "Tool ID should be correct"
    assert len(result.scores) == 2, "Should score 2 sequences"
    assert len(result.vocab) == 512, "Evo2 vocab should be 512 (byte-level)"
    assert result.vocab[65] == "A" and result.vocab[67] == "C"
    assert result.vocab[71] == "G" and result.vocab[84] == "T"

    for i, score in enumerate(result.scores):
        assert score.logits is not None, f"Sequence {i} should have logits"
        assert len(score.logits[0]) == 512, "Inner logits list should have 512 elements (Evo2 byte-level vocab)"


@pytest.mark.uses_gpu
def test_evo2_score_logits_disabled_by_default():
    """Test that logits are None when return_logits=False (default)."""
    sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA"]
    inputs = Evo2ScoringInput(sequences=sequences)
    config = Evo2ScoringConfig(
        model_checkpoint="evo2_7b",
        verbose=False,
    )

    result = run_evo2_score(inputs=inputs, config=config)
    validate_output(result)

    for score in result.scores:
        assert score.logits is None, "Logits should be None when return_logits=False"


@pytest.mark.uses_gpu
def test_evo2_20b_score_smoke():
    """The 20b checkpoint loads and scores; every other test uses 7b to avoid worker restarts."""
    sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA"]
    result = run_evo2_score(
        inputs=Evo2ScoringInput(sequences=sequences),
        config=Evo2ScoringConfig(model_checkpoint="evo2_20b", verbose=False, return_logits=True),
    )
    validate_output(result)
    assert_metrics_in_spec(result)

    assert len(result.scores) == 2
    for seq, score in zip(sequences, result.scores, strict=True):
        assert score.logits is not None
        assert len(score.logits) == len(seq)
        assert len(score.logits[0]) == 512


# ── Benchmarks ────────────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_evo2_score_single_sequence():
    """Test evo2 scoring with a single sequence (string input)."""
    inputs = Evo2ScoringInput(sequences="ATCGATCGATCGATCGATCG")
    config = Evo2ScoringConfig(
        model_checkpoint="evo2_7b",
        verbose=False,
    )

    result = run_evo2_score(inputs=inputs, config=config)
    validate_output(result)
    assert_metrics_in_spec(result)

    assert len(result.scores) == 1


@pytest.mark.benchmark("evo2-sample")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_evo2_sample_benchmark(request):
    """Benchmark evo2-sample: 16 DNA prompts x 1024 nt, generating 512 tokens each (cold + warm)."""
    prompts = random_dna_sequences(n=16, length=1024, seed=0)
    inputs = Evo2SampleInput(prompts=prompts)
    config = Evo2SampleConfig(
        model_checkpoint="evo2_7b",
        max_new_tokens=512,
        temperature=1.0,
        batch_size=8,
        verbose=False,
    )

    result = benchmark_twice(request, "evo2", lambda: run_evo2_sample(inputs=inputs, config=config))

    assert result.tool_id == "evo2-sample"
    assert len(result.sequences) == 16, "Should have 16 sampled sequences"
    valid_chars = set(DNA_NUCLEOTIDES) | set(DNA_NUCLEOTIDES.lower()) | set("Nn")
    for seq in result.sequences:
        assert isinstance(seq, str) and len(seq) >= 1024, "Output should include the 1024nt prompt at minimum"
        invalid = set(seq) - valid_chars
        assert not invalid, f"Non-DNA characters in output: {invalid}"


@pytest.mark.benchmark("evo2-score")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_evo2_score_benchmark(request):
    """Benchmark evo2-score: 24 DNA sequences x 4096 nt (cold + warm)."""
    sequences = random_dna_sequences(n=24, length=4096, seed=1)
    inputs = Evo2ScoringInput(sequences=sequences)
    config = Evo2ScoringConfig(
        model_checkpoint="evo2_7b",
        batch_size=16,
        return_logits=True,
        verbose=False,
    )

    result = benchmark_twice(request, "evo2", lambda: run_evo2_score(inputs=inputs, config=config))
    assert_metrics_in_spec(result)

    assert result.tool_id == "evo2-score"
    assert len(result.scores) == 24
    for score in result.scores:
        assert score.logits is not None
        assert len(score.logits) == 4096
        assert len(score.logits[0]) == 512  # Evo2 byte-level vocab


@pytest.mark.uses_gpu
def test_evo2_continuation_handles_fork_and_release():
    """Continue a cached prefix twice, then reject its released worker-local handle."""
    from proto_tools.tools.causal_models.evo2.evo2_sample import release_evo2_kv_caches
    from proto_tools.utils import ToolInstance

    with ToolInstance.persist_tool("evo2") as instance:
        config = Evo2SampleConfig(max_new_tokens=8, top_k=1, stop_at_eos=False, return_kv_cache=True, seed=17)
        first = run_evo2_sample(Evo2SampleInput(prompts=["ACGTACGT"]), config, instance=instance)
        handle = first.results[0].kv_cache
        assert handle is not None
        continued = config.model_copy(update={"old_kv_cache": handle})
        forks = run_evo2_sample(
            Evo2SampleInput(prompts=[first.sequences[0], first.sequences[0] + "A"]), continued, instance=instance
        )
        assert [len(s) for s in forks.sequences] == [24, 25]
        assert len({item.kv_cache.cache_id for item in forks.results if item.kv_cache}) == 2
        release_evo2_kv_caches(handle, instance=instance)
        with pytest.raises(Exception, match="no longer available"):
            # A distinct prompt avoids the framework's memoized seeded result.
            run_evo2_sample(Evo2SampleInput(prompts=[first.sequences[0] + "C"]), continued, instance=instance)
        release_evo2_kv_caches([item.kv_cache for item in forks.results], instance=instance)
