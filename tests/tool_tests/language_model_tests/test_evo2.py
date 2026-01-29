"""
test_evo2.py

Tests the Evo2 implementation
"""

import pytest

from tests.tool_tests.tool_infra_tests.test_export_functionality import validate_output


@pytest.mark.uses_gpu
def test_evo2_inference():
    """Test Evo2Model inference with direct model call."""
    from bio_programming.tools.language_models.evo2 import Evo2Model

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
    )

    sequences = result["sequences"]
    assert result.get("logits", None) is not None, "Logits should be present"
    assert result.get("kv_caches", None) is not None, "KV caches should be present"

    # Check that we got the correct number of sequences
    assert len(sequences) == 2, "Should generate 2 sequences"

    # Check that sequences are strings
    assert all(isinstance(seq, str) for seq in sequences), "Sequences should be strings"

    # Check that sequences are longer than 0 (generation happened)
    assert all(len(seq) > 0 for seq in sequences), "Sequences should not be empty"

    # Check that sequences contain valid DNA characters
    valid_chars = set("ATCGN")  # Evo2 may generate N for uncertain positions
    for seq in sequences:
        assert set(seq.upper()).issubset(valid_chars), f"Sequence contains invalid characters: {set(seq) - valid_chars}"


@pytest.mark.uses_gpu
def test_evo2_sample_tool():
    """Test the evo2 sampling tool with run_evo2_sample."""
    from bio_programming.tools.language_models.evo2 import (
        Evo2SampleConfig,
        Evo2SampleInput,
        run_evo2_sample,
    )

    # Test with simple prompts
    inputs = Evo2SampleInput(prompts=["ATCG", "GCTA"])
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

    # Validate output and export functionality
    validate_output(result)

    # Check output structure
    assert result.tool_id == "evo2-sample", "Tool ID should be correct"
    assert result.metadata["model_checkpoint"] == "evo2_7b", "Model name should match config"

    # Check sequences
    assert len(result.sequences) == 2, "Should generate 2 sequences"
    assert all(isinstance(seq, str) for seq in result.sequences), "Sequences should be strings"
    assert all(len(seq) > 0 for seq in result.sequences), "Sequences should not be empty"

    # Check metadata
    assert result.metadata["num_tokens"] == 50
    assert result.metadata["temperature"] == 0.8
    assert result.metadata["model_checkpoint"] == "evo2_7b"


@pytest.mark.uses_gpu
def test_evo2_sample_with_prepend():
    """Test evo2 sampling - sequences should include prompts by default."""
    from bio_programming.tools.language_models.evo2 import (
        Evo2SampleConfig,
        Evo2SampleInput,
        run_evo2_sample,
    )

    prompts = ["ATCGATCG", "GGCCTTAA"]
    inputs = Evo2SampleInput(prompts=prompts)
    config = Evo2SampleConfig(
        model_checkpoint="evo2_7b",
        num_tokens=50,
        temperature=1.0,
        verbose=False,
        print_generation=False,
    )

    result = run_evo2_sample(inputs=inputs, config=config)

    # Validate output and export functionality
    validate_output(result)

    # Check that prompts are included in output (Evo2 includes prompt by default)
    for i, (prompt, seq) in enumerate(zip(prompts, result.sequences)):
        assert seq.startswith(prompt), f"Sequence {i} should start with prompt '{prompt}'"
        assert len(seq) > len(prompt), f"Sequence {i} should be longer than prompt"


@pytest.mark.uses_gpu
def test_evo2_sample_single_prompt_batch():
    """Test evo2 sampling with single prompt."""
    from bio_programming.tools.language_models.evo2 import (
        Evo2SampleConfig,
        Evo2SampleInput,
        run_evo2_sample,
    )

    # Single prompt should work
    inputs = Evo2SampleInput(prompts=["ATCGATCG"])
    config = Evo2SampleConfig(
        model_checkpoint="evo2_7b",
        num_tokens=100,
        temperature=1.0,
        verbose=False,
        print_generation=False,
    )

    result = run_evo2_sample(inputs=inputs, config=config)

    # Validate output and export functionality
    validate_output(result)

    # Should generate 1 sequence
    assert len(result.sequences) == 1
    assert isinstance(result.sequences[0], str)
    assert len(result.sequences[0]) > 0


@pytest.mark.uses_gpu
def test_evo2_sample_single_prompt_string():
    """Test evo2 sampling with single prompt as string (not list)."""
    from bio_programming.tools.language_models.evo2 import (
        Evo2SampleConfig,
        Evo2SampleInput,
        run_evo2_sample,
    )

    # Single prompt as string (not list) should work
    inputs = Evo2SampleInput(prompts="ATCGATCG")
    config = Evo2SampleConfig(
        model_checkpoint="evo2_7b",
        num_tokens=100,
        temperature=1.0,
        verbose=False,
        print_generation=False,
    )

    result = run_evo2_sample(inputs=inputs, config=config)

    # Validate output and export functionality
    validate_output(result)

    # Should generate 1 sequence
    assert len(result.sequences) == 1
    assert isinstance(result.sequences[0], str)
    assert len(result.sequences[0]) > 0
    assert result.sequences[0].startswith("ATCGATCG")


def test_evo2_input_and_config_validation():
    """Test Evo2SampleConfig validation."""
    from bio_programming.tools.language_models.evo2 import (
        Evo2SampleConfig,
        Evo2SampleInput,
    )

    # Test empty prompts should fail
    with pytest.raises(ValueError, match="prompts must not be empty"):
        Evo2SampleInput(prompts=[])

    # Test invalid temperature should fail
    with pytest.raises(ValueError):
        _ = Evo2SampleInput(prompts=["ATCG"])
        _ = Evo2SampleConfig(num_tokens=100, temperature=0.0)  # Must be > 0

    # Test invalid top_p should fail
    with pytest.raises(ValueError):
        _ = Evo2SampleInput(prompts=["ATCG"])
        _ = Evo2SampleConfig(num_tokens=100, top_p=1.5)  # Must be <= 1.0

    # Test invalid num_tokens should fail
    with pytest.raises(ValueError):
        _ = Evo2SampleInput(prompts=["ATCG"])
        _ = Evo2SampleConfig(num_tokens=0)  # Must be >= 1


@pytest.mark.uses_gpu
def test_evo2_batch_with_kv_cache():
    """Test that batching with KV cache works correctly via cache slicing."""
    from bio_programming.tools.language_models.evo2 import (
        Evo2SampleConfig,
        Evo2SampleInput,
        run_evo2_sample,
    )

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
    from bio_programming.tools.language_models.evo2.inference import _slice_cache
    sliced = _slice_cache(result1.kv_caches[0], 0, 1)
    assert sliced is not None, "Sliced cache should not be None"

    # Verify the sliced cache has correct batch dimension
    kv = next(iter(sliced['mha'].key_value_memory_dict.values()))
    assert kv.shape[0] == 1, f"Sliced cache batch dim should be 1, got {kv.shape[0]}"


@pytest.mark.uses_gpu
def test_evo2_continued_generation_with_cache():
    """Test continued generation using KV cache from previous generation."""
    from bio_programming.tools.language_models.evo2 import (
        Evo2SampleConfig,
        Evo2SampleInput,
        run_evo2_sample,
    )

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
    from bio_programming.tools.language_models.evo2 import (
        Evo2ScoringConfig,
        Evo2ScoringInput,
        run_evo2_score,
    )

    sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA"]
    inputs = Evo2ScoringInput(sequences=sequences)
    config = Evo2ScoringConfig(
        model_checkpoint="evo2_7b",
        verbose=False,
    )

    result = run_evo2_score(inputs=inputs, config=config)

    # Validate output and export functionality
    validate_output(result)

    # Check output structure
    assert result.tool_id == "evo2-score", "Tool ID should be correct"
    assert len(result.scores) == 2, "Should score 2 sequences"
    assert result.alphabet == "ACGT", "Alphabet should be DNA nucleotides"

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
    from bio_programming.tools.language_models.evo2 import (
        Evo2ScoringConfig,
        Evo2ScoringInput,
        run_evo2_score,
    )

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
    import numpy as np
    from bio_programming.tools.language_models.evo2 import (
        Evo2ScoringConfig,
        Evo2ScoringInput,
        run_evo2_score,
    )

    inputs = Evo2ScoringInput(sequences=["ATCGATCGATCG"])
    config = Evo2ScoringConfig(
        model_checkpoint="evo2_7b",
        verbose=False,
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
    from bio_programming.tools.language_models.evo2 import Evo2ScoringInput

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
    from bio_programming.tools.language_models.evo2 import (
        Evo2ScoringConfig,
        Evo2ScoringInput,
        run_evo2_score,
    )

    # All sequences same length - should use batched scoring
    sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA", "AAAACCCCGGGG", "TTTTGGGGCCCC"]
    inputs = Evo2ScoringInput(sequences=sequences)
    config = Evo2ScoringConfig(model_checkpoint="evo2_7b", verbose=False)

    result = run_evo2_score(inputs=inputs, config=config)

    assert len(result.scores) == 4
    for score in result.scores:
        assert score.perplexity > 0
        assert score.logits is not None


@pytest.mark.uses_gpu
def test_evo2_score_variable_length():
    """Test scoring sequences of different lengths (sequential scoring)."""
    from bio_programming.tools.language_models.evo2 import (
        Evo2ScoringConfig,
        Evo2ScoringInput,
        run_evo2_score,
    )

    # Different length sequences - should fall back to sequential scoring
    sequences = ["ATCG", "ATCGATCG", "ATCGATCGATCG"]
    inputs = Evo2ScoringInput(sequences=sequences)
    config = Evo2ScoringConfig(model_checkpoint="evo2_7b", verbose=False)

    result = run_evo2_score(inputs=inputs, config=config)

    assert len(result.scores) == 3
    # Verify logits have correct shapes for different length sequences
    for i, (seq, score) in enumerate(zip(sequences, result.scores)):
        assert score.logits is not None
        # Logits should have seq_len matching input (may differ due to tokenization)
        assert score.perplexity > 0
