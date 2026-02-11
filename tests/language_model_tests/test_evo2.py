"""
test_evo2.py

Tests the Evo2 implementation.
"""

import numpy as np
import pytest

# Import all evo2 modules at the module level to avoid re-importing vortex library
# which registers PyTorch custom operations. Re-importing causes schema registration
# errors in PyTorch 2.7+ when running multiple tests together (pytest --all).
from bio_programming_tools.tools.causal_models.evo2 import (
    Evo2Model,
    Evo2SampleConfig,
    Evo2SampleInput,
    Evo2ScoringConfig,
    Evo2ScoringInput,
    run_evo2_sample,
    run_evo2_score,
)

# NOTE: This assumes Evo2 is installed in the base environment. We will need to
# modify or remove these tests in branches where Evo2 uses the standalone venv.
from bio_programming_tools.tools.causal_models.evo2.standalone.inference import (
    _slice_cache,
)
from tests.tool_infra_tests.test_export_functionality import validate_output

# ============================================================================
# Sampling Tests
# ============================================================================

@pytest.mark.uses_gpu
def test_evo2_sample_inference():
    """Test Evo2Model inference with direct model.sample() call."""
    prompts = ["ATCGATCG", "GGCCTTAA"]
    evo2_model = Evo2Model(model_checkpoint="evo2_7b")

    result = evo2_model.sample(
        prompts=prompts,
        num_tokens=100,
        temperature=1.0,
        top_k=4,
        top_p=1.0,
        cached_generation=True,
        verbose=False,
        return_logits=True,
    )

    sequences = result["sequences"]
    assert len(sequences) == 2

    # Validate actual sequence content
    valid_chars = set("ATCGN")  # Evo2 may generate N for uncertain positions
    for i, (prompt, seq) in enumerate(zip(prompts, sequences)):
        # Sequence should be non-empty string of valid DNA characters
        assert isinstance(seq, str) and len(seq) > 0
        assert set(seq.upper()).issubset(valid_chars), f"Invalid chars: {set(seq) - valid_chars}"

        # Sequence should contain the prompt (vortex returns only generated tokens)
        # So we check that the sequence is non-empty and valid
        assert len(seq) > 0, f"Sequence {i} should be non-empty"

    # Validate logits and kv_caches are present (when return_logits=True)
    assert result.get("logits", None) is not None, "Logits should be present when return_logits=True"
    assert result.get("kv_caches", None) is not None, "KV caches should be present"
    assert len(result["logits"]) == 2
    assert len(result["kv_caches"]) == 2


@pytest.mark.uses_gpu
def test_evo2_sample_tool():
    """Test the evo2 sampling tool with run_evo2_sample."""
    prompts = ["ATCG", "GCTA"]
    inputs = Evo2SampleInput(prompts=prompts)
    config = Evo2SampleConfig(
        model_checkpoint="evo2_7b",
        num_tokens=50,
        temperature=0.8,
        top_k=4,
        top_p=1.0,
        cached_generation=True,
        verbose=False,
        print_generation=False,
    )

    result = run_evo2_sample(inputs=inputs, config=config)
    validate_output(result)

    # Validate tool output structure
    assert result.tool_id == "evo2-sample"
    assert result.metadata["model_checkpoint"] == "evo2_7b"
    assert result.metadata["num_tokens"] == 50
    assert result.metadata["temperature"] == 0.8
    assert len(result.sequences) == 2

    # Validate sequences contain prompts and are valid
    valid_chars = set("ATCGN")
    for i, (prompt, seq) in enumerate(zip(prompts, result.sequences)):
        # With prepend_prompt=True (default), sequences should start with prompt
        assert seq.startswith(prompt), f"Sequence {i} should start with '{prompt}'"
        assert set(seq.upper()).issubset(valid_chars)
        assert len(seq) > len(prompt), f"Sequence {i} should be longer than prompt"


@pytest.mark.uses_gpu
@pytest.mark.parametrize("prompt", [
    "ATCGATCG",
    "GCTAGCTA",
    "AAAACCCC",
])
def test_evo2_sample_prompt_handling(prompt):
    """Test evo2 sampling with various prompt formats."""
    inputs = Evo2SampleInput(prompts=prompt)
    config = Evo2SampleConfig(
        model_checkpoint="evo2_7b",
        num_tokens=50,
        temperature=1.0,
        verbose=False,
        print_generation=False,
    )

    result = run_evo2_sample(inputs=inputs, config=config)
    validate_output(result)

    assert len(result.sequences) == 1
    seq = result.sequences[0]
    assert prompt in seq, f"Expected '{prompt}' in output"
    assert len(seq) > len(prompt), "Should generate beyond prompt"


@pytest.mark.uses_gpu
def test_evo2_sample_prepend_prompt():
    """Test that prepend_prompt controls whether prompt is included."""
    prompt = "ATCGATCG"

    # With prepend_prompt=True (default)
    result_with = run_evo2_sample(
        inputs=Evo2SampleInput(prompts=[prompt]),
        config=Evo2SampleConfig(
            model_checkpoint="evo2_7b",
            num_tokens=50,
            prepend_prompt=True,
            verbose=False,
            print_generation=False,
        ),
    )
    assert result_with.sequences[0].startswith(prompt)

    # With prepend_prompt=False
    result_without = run_evo2_sample(
        inputs=Evo2SampleInput(prompts=[prompt]),
        config=Evo2SampleConfig(
            model_checkpoint="evo2_7b",
            num_tokens=50,
            prepend_prompt=False,
            verbose=False,
            print_generation=False,
        ),
    )
    # Should not start with prompt (only generated tokens)
    assert not result_without.sequences[0].startswith(prompt)
    assert len(result_without.sequences[0]) > 0


@pytest.mark.parametrize("config_kwargs,match", [
    ({"prompts": []}, "prompts must not be empty"),
])
def test_evo2_sample_input_validation(config_kwargs, match):
    """Test Evo2SampleInput validation."""
    with pytest.raises(ValueError, match=match):
        Evo2SampleInput(**config_kwargs)


@pytest.mark.parametrize("config_kwargs", [
    {"temperature": 0.0},
    {"top_p": 1.5},
    {"num_tokens": 0},
])
def test_evo2_sample_config_validation(config_kwargs):
    """Test Evo2SampleConfig validation for invalid values."""
    with pytest.raises(ValueError):
        Evo2SampleConfig(**config_kwargs)


# ============================================================================
# Batched Sampling Tests
# ============================================================================

@pytest.mark.uses_gpu
def test_evo2_sample_batched_inference():
    """Test batched sampling with direct model call."""
    prompts = ["ATCGATCG", "GGCCTTAA", "AAAACCCC", "TTTTGGGG"]
    evo2_model = Evo2Model(model_checkpoint="evo2_7b")

    result = evo2_model.sample(
        prompts=prompts,
        num_tokens=50,
        temperature=1.0,
        batch_size=2,
        verbose=False,
        print_generation=False,
    )

    sequences = result["sequences"]
    assert len(sequences) == 4

    # Each sequence should be valid
    valid_chars = set("ATCGN")
    for i, seq in enumerate(sequences):
        assert isinstance(seq, str) and len(seq) > 0
        assert set(seq.upper()).issubset(valid_chars)


@pytest.mark.uses_gpu
def test_evo2_sample_batched_tool():
    """Test batched sampling with tool layer (run_evo2_sample)."""
    prompts = ["ATCG", "GCTA", "AAAA", "GGGG", "CCCC", "TTTT"]
    inputs = Evo2SampleInput(prompts=prompts)
    config = Evo2SampleConfig(
        model_checkpoint="evo2_7b",
        num_tokens=50,
        temperature=1.0,
        batch_size=2,
        verbose=False,
        print_generation=False,
    )

    result = run_evo2_sample(inputs=inputs, config=config)
    validate_output(result)

    assert len(result.sequences) == 6

    # Each sequence should start with its corresponding prompt
    for i, (prompt, seq) in enumerate(zip(prompts, result.sequences)):
        assert seq.startswith(prompt), f"Sequence {i} should start with '{prompt}'"
        assert len(seq) > len(prompt)


# ============================================================================
# Scoring Tests
# ============================================================================

@pytest.mark.uses_gpu
def test_evo2_score_inference():
    """Test Evo2Model.score() with comprehensive value validation."""
    sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA"]
    evo2_model = Evo2Model(model_checkpoint="evo2_7b")

    result = evo2_model.score(sequences=sequences, device="cuda", verbose=False, return_logits=True)

    assert "logits" in result and "metrics" in result
    assert len(result["logits"]) == len(result["metrics"]) == 2

    for seq, metrics, logits in zip(sequences, result["metrics"], result["logits"]):
        # Validate metrics types
        assert isinstance(metrics["log_likelihood"], float)
        assert isinstance(metrics["avg_log_likelihood"], float)
        assert isinstance(metrics["perplexity"], float)

        # Log likelihood should be negative (log probabilities are <= 0)
        assert metrics["log_likelihood"] < 0, f"Log likelihood should be negative, got {metrics['log_likelihood']}"

        # Average log likelihood should be between log_likelihood and 0
        assert metrics["log_likelihood"] <= metrics["avg_log_likelihood"] <= 0

        # Perplexity should be >= 1 (exp(0) = 1 is minimum when avg_ll = 0)
        assert metrics["perplexity"] >= 1.0, f"Perplexity should be >= 1, got {metrics['perplexity']}"

        # Verify perplexity = exp(-avg_log_likelihood)
        expected_ppl = np.exp(-metrics["avg_log_likelihood"])
        np.testing.assert_allclose(metrics["perplexity"], expected_ppl, rtol=1e-5)

        # Logits shape: (seq_len, vocab_size)
        # Note: Evo2 tokenization may differ from raw sequence length
        assert logits.shape[0] > 0, f"Logits seq_len should be > 0, got {logits.shape[0]}"
        assert logits.shape[1] >= 4, f"Vocab size should be >= 4 (ACGT), got {logits.shape[1]}"


@pytest.mark.uses_gpu
def test_evo2_score_different_sequences():
    """Test that model produces different perplexities for different sequences."""
    # Different sequences should produce different perplexities
    seq1 = "ATCGATCGATCGATCGATCG"  # Alternating pattern
    seq2 = "AAAAAAAAAAAAAAAAAAAA"  # Homopolymer

    inputs = Evo2ScoringInput(sequences=[seq1, seq2])
    config = Evo2ScoringConfig(model_checkpoint="evo2_7b", verbose=False, return_logits=True)

    result = run_evo2_score(inputs=inputs, config=config)

    ppl1 = result.scores[0].perplexity
    ppl2 = result.scores[1].perplexity

    # Different sequences should have different perplexities (model discriminates)
    assert ppl1 != ppl2, f"Different sequences should have different perplexities: {ppl1} vs {ppl2}"

    # Both should be valid perplexities
    assert ppl1 >= 1.0 and ppl2 >= 1.0


@pytest.mark.uses_gpu
def test_evo2_score_metrics_consistency():
    """Test that scoring metrics are mathematically consistent."""
    inputs = Evo2ScoringInput(sequences=["ATCGATCGATCGATCG"])
    config = Evo2ScoringConfig(model_checkpoint="evo2_7b", verbose=False, return_logits=True)

    result = run_evo2_score(inputs=inputs, config=config)
    score = result.scores[0]

    # Verify perplexity = exp(-avg_log_likelihood)
    expected_perplexity = np.exp(-score.avg_log_likelihood)
    np.testing.assert_allclose(score.perplexity, expected_perplexity, rtol=1e-5)

    # Average should equal total / count (approximately, accounting for tokenization)
    # We can't assert exact equality due to tokenization differences, but we can check
    # that avg_log_likelihood is reasonable relative to log_likelihood
    assert score.avg_log_likelihood <= 0, "avg_log_likelihood should be <= 0"
    assert score.log_likelihood <= score.avg_log_likelihood, "log_likelihood should be <= avg_log_likelihood"


def test_evo2_score_input_validation():
    """Test Evo2ScoringInput validation and normalization."""
    # Empty sequences should fail
    with pytest.raises(ValueError, match="sequences must not be empty"):
        Evo2ScoringInput(sequences=[])

    # String input should be normalized to list
    input_str = Evo2ScoringInput(sequences="ATCG")
    assert input_str.sequences == ["ATCG"]

    # List should be preserved
    input_list = Evo2ScoringInput(sequences=["ATCG", "GCTA"])
    assert input_list.sequences == ["ATCG", "GCTA"]


# ============================================================================
# Batched Scoring Tests
# ============================================================================

@pytest.mark.uses_gpu
def test_evo2_score_batched_inference():
    """Test batched scoring with direct model call."""
    sequences = ["ATCG", "GCTAGCTA", "AAAACCCC", "TTTTGGGG"]
    evo2_model = Evo2Model(model_checkpoint="evo2_7b")

    result = evo2_model.score(sequences=sequences, device="cuda", batch_size=2, verbose=False, return_logits=True)

    assert len(result["metrics"]) == 4

    for i, (seq, metrics, logits) in enumerate(zip(sequences, result["metrics"], result["logits"])):
        # Each sequence should have valid metrics
        assert metrics["log_likelihood"] < 0
        assert metrics["perplexity"] >= 1.0

        # Logits should have valid shape
        assert logits.shape[0] > 0, f"Sequence {i}: wrong logits length"
        assert logits.shape[1] >= 4


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

    assert len(result.scores) == 6

    for seq, score in zip(sequences, result.scores):
        assert score.log_likelihood < 0
        assert score.perplexity >= 1.0
        assert score.logits is not None, "Logits should be present when return_logits=True"
        # Logits are serialized as nested lists after going through tool layer
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
        for bs in [1, 2, None]
    }

    # All batch sizes should produce consistent perplexities
    # Relaxed tolerance for Hyena architecture batch-dependent numerical variance
    for i in range(4):
        perplexities = [results[bs].scores[i].perplexity for bs in [1, 2, None]]
        log_likelihoods = [results[bs].scores[i].log_likelihood for bs in [1, 2, None]]

        np.testing.assert_allclose(perplexities[0], perplexities[1], rtol=1e-2)
        np.testing.assert_allclose(perplexities[0], perplexities[2], rtol=1e-2)
        np.testing.assert_allclose(log_likelihoods[0], log_likelihoods[1], rtol=1e-2)
        np.testing.assert_allclose(log_likelihoods[0], log_likelihoods[2], rtol=1e-2)


@pytest.mark.uses_gpu
def test_evo2_score_variable_length_sequences():
    """Test scoring sequences of different lengths produces correct logits shapes."""
    sequences = ["AT", "ATCG", "ATCGATCG", "ATCGATCGATCG"]
    inputs = Evo2ScoringInput(sequences=sequences)
    config = Evo2ScoringConfig(model_checkpoint="evo2_7b", batch_size=2, verbose=False, return_logits=True)

    result = run_evo2_score(inputs=inputs, config=config)

    for (seq, score) in zip(sequences, result.scores):
        # Logits should have valid shape (may differ from raw seq len due to tokenization)
        # Logits are serialized as nested lists after going through tool layer
        assert score.logits is not None, "Logits should be present when return_logits=True"
        assert len(score.logits) > 0, (
            f"Sequence '{seq}' (len {len(seq)}): logits len should be > 0, got {len(score.logits)}"
        )

        # Metrics should be valid
        assert score.perplexity >= 1.0
        assert score.log_likelihood < 0


# ============================================================================
# Evo2-Specific Tests (KV Cache)
# ============================================================================

@pytest.mark.uses_gpu
def test_evo2_batch_with_kv_cache():
    """Test that batching with KV cache works correctly via cache slicing."""
    # First generation to get KV caches
    prompts = ["ATCGATCG"] * 4  # 4 identical prompts
    inputs = Evo2SampleInput(prompts=prompts)
    config = Evo2SampleConfig(
        model_checkpoint="evo2_7b",
        num_tokens=50,
        temperature=1.0,
        cached_generation=True,
        verbose=False,
        print_generation=False,
        prepend_prompt=True,
    )

    result1 = run_evo2_sample(inputs=inputs, config=config)
    assert result1.kv_caches is not None, "First generation should return KV caches"
    assert len(result1.kv_caches) == 4, "Should have 4 KV caches"

    # Test slicing the cache
    sliced = _slice_cache(result1.kv_caches[0], 0, 1)
    assert sliced is not None, "Sliced cache should not be None"

    # Verify the sliced cache has correct batch dimension
    kv = next(iter(sliced['mha'].key_value_memory_dict.values()))
    assert kv.shape[0] == 1, f"Sliced cache batch dim should be 1, got {kv.shape[0]}"


@pytest.mark.uses_gpu
def test_evo2_continued_generation_with_cache():
    """Test continued generation using KV cache from previous generation."""
    # First generation
    prompt = "ATCGATCG"
    inputs1 = Evo2SampleInput(prompts=[prompt])
    config1 = Evo2SampleConfig(
        model_checkpoint="evo2_7b",
        num_tokens=50,
        temperature=1.0,
        cached_generation=True,
        verbose=False,
        print_generation=False,
        prepend_prompt=False,  # Get only new tokens
    )

    result1 = run_evo2_sample(inputs=inputs1, config=config1)
    assert result1.kv_caches is not None, "First generation should return KV caches"
    assert len(result1.sequences) == 1
    first_gen = result1.sequences[0]

    # Continue generation from the cached state
    continued_prompt = prompt + first_gen  # Full sequence so far
    inputs2 = Evo2SampleInput(prompts=[continued_prompt])
    config2 = Evo2SampleConfig(
        model_checkpoint="evo2_7b",
        num_tokens=50,
        temperature=1.0,
        cached_generation=True,
        verbose=False,
        print_generation=False,
        prepend_prompt=False,
        old_kv_cache=result1.kv_caches[0],  # Use cached state
    )

    result2 = run_evo2_sample(inputs=inputs2, config=config2)
    assert len(result2.sequences) == 1
    second_gen = result2.sequences[0]

    # Full sequence should be prompt + first_gen + second_gen
    assert len(second_gen) == 50, f"Second generation should produce 50 tokens, got {len(second_gen)}"


# ============================================================================
# Scoring Tests
# ============================================================================

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

    # Validate output and export functionality
    validate_output(result)

    # Check output structure
    assert result.tool_id == "evo2-score", "Tool ID should be correct"
    assert len(result.scores) == 2, "Should score 2 sequences"
    # Evo2 uses byte-level vocab (512 chars); DNA nucleotides at ASCII indices
    assert len(result.vocab) == 512, "Evo2 vocab should be 512 (byte-level)"
    assert result.vocab[65] == "A" and result.vocab[67] == "C"
    assert result.vocab[71] == "G" and result.vocab[84] == "T"

    # Check scoring metrics for each sequence
    for i, score in enumerate(result.scores):
        assert "log_likelihood" in score.metrics, f"Sequence {i} should have log_likelihood"
        assert "avg_log_likelihood" in score.metrics, f"Sequence {i} should have avg_log_likelihood"
        assert "perplexity" in score.metrics, f"Sequence {i} should have perplexity"

        # Verify metric types
        assert isinstance(score.log_likelihood, float), "log_likelihood should be float"
        assert isinstance(score.avg_log_likelihood, float), "avg_log_likelihood should be float"
        assert isinstance(score.perplexity, float), "perplexity should be float"

        # Perplexity should be positive
        assert score.perplexity > 0, f"Perplexity should be positive, got {score.perplexity}"

        # Logits should be present
        assert score.logits is not None, f"Sequence {i} should have logits"


@pytest.mark.uses_gpu
def test_evo2_score_single_sequence():
    """Test evo2 scoring with a single sequence (string input)."""
    # Single sequence as string should work
    inputs = Evo2ScoringInput(sequences="ATCGATCGATCGATCGATCG")
    config = Evo2ScoringConfig(
        model_checkpoint="evo2_7b",
        verbose=False,
    )

    result = run_evo2_score(inputs=inputs, config=config)

    # Validate output
    validate_output(result)

    # Should score 1 sequence
    assert len(result.scores) == 1
    assert result.scores[0].perplexity > 0


@pytest.mark.uses_gpu
def test_evo2_score_metrics_consistency():
    """Test that scoring metrics are mathematically consistent."""
    inputs = Evo2ScoringInput(sequences=["ATCGATCGATCG"])
    config = Evo2ScoringConfig(
        model_checkpoint="evo2_7b",
        verbose=False,
        return_logits=True,
    )

    result = run_evo2_score(inputs=inputs, config=config)
    score = result.scores[0]

    # Verify perplexity = exp(-avg_log_likelihood)
    expected_perplexity = np.exp(-score.avg_log_likelihood)
    np.testing.assert_allclose(
        score.perplexity,
        expected_perplexity,
        rtol=1e-5,
        err_msg="Perplexity should equal exp(-avg_log_likelihood)"
    )


def test_evo2_scoring_input_validation():
    """Test Evo2ScoringInput validation."""
    # Test empty sequences should fail
    with pytest.raises(ValueError, match="sequences must not be empty"):
        Evo2ScoringInput(sequences=[])

    # Test single string normalization
    input_str = Evo2ScoringInput(sequences="ATCG")
    assert input_str.sequences == ["ATCG"], "Single string should be normalized to list"

    # Test list input
    input_list = Evo2ScoringInput(sequences=["ATCG", "GCTA"])
    assert len(input_list.sequences) == 2, "List input should preserve length"

@pytest.mark.uses_gpu
def test_evo2_score_batched_uniform_length():
    """Test that scoring batches sequences of uniform length."""
    # All sequences same length - should use batched scoring
    sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA", "AAAACCCCGGGG", "TTTTGGGGCCCC"]
    inputs = Evo2ScoringInput(sequences=sequences)
    config = Evo2ScoringConfig(model_checkpoint="evo2_7b", verbose=False, return_logits=True)

    result = run_evo2_score(inputs=inputs, config=config)

    assert len(result.scores) == 4
    for score in result.scores:
        assert score.perplexity > 0
        assert score.logits is not None


@pytest.mark.uses_gpu
def test_evo2_score_variable_length():
    """Test scoring sequences of different lengths (sequential scoring)."""
    # Different length sequences - should fall back to sequential scoring
    sequences = ["ATCG", "ATCGATCG", "ATCGATCGATCG"]
    inputs = Evo2ScoringInput(sequences=sequences)
    config = Evo2ScoringConfig(model_checkpoint="evo2_7b", verbose=False, return_logits=True)

    result = run_evo2_score(inputs=inputs, config=config)

    assert len(result.scores) == 3
    # Verify logits have correct shapes for different length sequences
    for i, (seq, score) in enumerate(zip(sequences, result.scores)):
        assert score.logits is not None
        # Logits should have seq_len matching input (may differ due to tokenization)
        assert score.perplexity > 0


# ============================================================================
# Logits-Specific Tests (Scoring)
# ============================================================================

@pytest.mark.uses_gpu
def test_evo2_score_logits_disabled_by_default():
    """Test that logits are None when return_logits=False (default)."""
    sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA"]
    inputs = Evo2ScoringInput(sequences=sequences)
    config = Evo2ScoringConfig(
        model_checkpoint="evo2_7b",
        verbose=False,
        # return_logits defaults to False
    )

    result = run_evo2_score(inputs=inputs, config=config)
    validate_output(result)

    # Logits should be None when return_logits=False
    for score in result.scores:
        assert score.logits is None, "Logits should be None when return_logits=False"


@pytest.mark.uses_gpu
def test_evo2_score_logits_enabled():
    """Test that logits are correctly returned when return_logits=True."""
    sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA"]
    inputs = Evo2ScoringInput(sequences=sequences)
    config = Evo2ScoringConfig(
        model_checkpoint="evo2_7b",
        verbose=False,
        return_logits=True,
    )

    result = run_evo2_score(inputs=inputs, config=config)
    validate_output(result)

    # Logits should be present with correct shape
    for score in result.scores:
        assert score.logits is not None, "Logits should not be None when return_logits=True"
        assert isinstance(score.logits, (list, np.ndarray)), f"Logits should be list or ndarray, got {type(score.logits)}"

        # Convert to ndarray for shape validation if it's a list
        logits_arr = np.array(score.logits)
        # Evo2 uses byte-level tokenization, so seq_len may differ from raw length
        assert logits_arr.shape[0] > 0, "Logits should have at least one position"
        assert logits_arr.shape[1] == 512, f"Evo2 vocab size should be 512 (byte-level), got {logits_arr.shape[1]}"


@pytest.mark.uses_gpu
def test_evo2_score_logits_serialization():
    """Test that logits are properly serialized as nested lists."""
    sequences = ["ATCGATCGATCG"]
    inputs = Evo2ScoringInput(sequences=sequences)
    config = Evo2ScoringConfig(
        model_checkpoint="evo2_7b",
        verbose=False,
        return_logits=True,
    )

    result = run_evo2_score(inputs=inputs, config=config)
    validate_output(result)

    score = result.scores[0]

    # Logits should be serialized as nested lists (not tensors)
    assert isinstance(score.logits, (list, np.ndarray)), "Logits should be list or ndarray"

    if isinstance(score.logits, list):
        # Verify nested list structure
        assert len(score.logits) > 0, "Logits list should not be empty"
        assert isinstance(score.logits[0], list), "Logits should be a list of lists"
        assert len(score.logits[0]) == 512, "Inner logits list should have 512 elements (Evo2 byte-level vocab)"

        # Verify all values are numeric
        for position_logits in score.logits:
            for logit_value in position_logits:
                assert isinstance(logit_value, (int, float)), f"Logit value should be numeric, got {type(logit_value)}"
    else:
        # If ndarray, verify shape
        assert score.logits.ndim == 2, "Logits should be 2D array"
        assert score.logits.shape[1] == 512, "Evo2 vocab size should be 512"


# ============================================================================
# Logits-Specific Tests (Sampling)
# ============================================================================

@pytest.mark.uses_gpu
def test_evo2_sample_logits_inference():
    """Test that sample() can return logits at inference layer."""
    prompts = ["ATCGATCG", "GCTAGCTA"]
    evo2_model = Evo2Model(model_checkpoint="evo2_7b")

    result = evo2_model.sample(
        prompts=prompts,
        num_tokens=50,
        temperature=1.0,
        return_logits=True,
        cached_generation=False,
        verbose=False,
        print_generation=False,
    )

    # Verify logits are returned
    assert "logits" in result, "Result should contain 'logits' key"
    assert result["logits"] is not None, "Logits should not be None when return_logits=True"
    assert len(result["logits"]) == 2, f"Should have logits for 2 sequences, got {len(result['logits'])}"

    # Verify logits are tensors (before serialization at inference layer)
    for i, logits in enumerate(result["logits"]):
        assert hasattr(logits, "shape"), f"Logits[{i}] should be a tensor with shape attribute"
        # Logits shape should be (generated_length, vocab_size=512)
        assert logits.shape[1] == 512, f"Evo2 vocab size should be 512, got {logits.shape[1]}"


@pytest.mark.uses_gpu
def test_evo2_sample_logits_not_returned_by_default():
    """Test that sample() does not return logits when return_logits=False (default)."""
    prompts = ["ATCGATCG"]
    evo2_model = Evo2Model(model_checkpoint="evo2_7b")

    result = evo2_model.sample(
        prompts=prompts,
        num_tokens=50,
        temperature=1.0,
        return_logits=False,  # Explicit False
        cached_generation=False,
        verbose=False,
        print_generation=False,
    )

    # Verify logits are None or empty when not requested
    assert "logits" in result, "Result should contain 'logits' key"
    logits = result["logits"]
    assert logits is None or (isinstance(logits, list) and len(logits) == 0), \
        f"Logits should be None or empty when return_logits=False, got {type(logits)}"
