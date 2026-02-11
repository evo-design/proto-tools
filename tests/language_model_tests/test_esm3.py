"""
test_esm3.py

Tests the ESM3 implementation
"""

import numpy as np
import pytest

from bio_programming_tools.tools.masked_models.esm3 import (
    ESM3EmbeddingsConfig,
    ESM3EmbeddingsInput,
    ESM3ScoringConfig,
    ESM3ScoringInput,
    ESM3StructurePredictionConfig,
    ESM3StructurePredictionInput,
    run_esm3_embeddings,
    run_esm3_score,
    run_esm3_structure_prediction,
)
from tests.tool_infra_tests.test_export_functionality import validate_output

GFP = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"


# ============================================================================
# Embedding Tests
# ============================================================================

@pytest.mark.uses_gpu
def test_esm3_forward_pass():
    sequences = ["TARGET"] * 10 + ["TEST"] * 30

    inputs = ESM3EmbeddingsInput(sequences=sequences)
    config = ESM3EmbeddingsConfig(batch_size=2, return_logits=True)

    result = run_esm3_embeddings(inputs=inputs, config=config)
    assert result.success, "ESM3 embeddings failed"

    # Check mean embedding shape (converted to numpy array for shape checking)
    mean_embeddings = np.array(result.mean_embeddings)
    assert mean_embeddings.shape == (
        40,
        1536,
    ), "Avg embedding shape is not correct"

    # Check attention mask shape
    attention_masks = np.array(result.attention_masks)
    assert attention_masks.shape == (
        40,
        6,
    ), "Attention mask shape is not correct"

    # Check logit shape
    logits = np.array(result.logits)
    assert logits.shape == (40, 6, 20), "Logit shape is not correct"


@pytest.mark.uses_gpu
def test_esm3_predict_structure():
    sequences = [GFP] * 2

    inputs = ESM3StructurePredictionInput(sequences=sequences)
    config = ESM3StructurePredictionConfig(batch_size=2)

    result = run_esm3_structure_prediction(inputs=inputs, config=config)

    assert len(result.structures) == 2


# ============================================================================
# Scoring Tests
# ============================================================================

@pytest.mark.uses_gpu
def test_esm3_score_inference():
    """Test run_esm3_score() with comprehensive value validation."""
    sequences = ["MKTAYIAKQR", "EVQLVESGGS"]
    inputs = ESM3ScoringInput(sequences=sequences)
    config = ESM3ScoringConfig(verbose=False, return_logits=True)

    result = run_esm3_score(inputs=inputs, config=config)

    assert len(result.scores) == 2
    assert isinstance(result.vocab, list)

    for seq, score in zip(sequences, result.scores):
        # Validate metrics types
        assert isinstance(score.log_likelihood, float)
        assert isinstance(score.avg_log_likelihood, float)
        assert isinstance(score.perplexity, float)

        # Log likelihood should be negative (log probabilities are <= 0)
        assert (
            score.log_likelihood < 0
        ), f"Log likelihood should be negative, got {score.log_likelihood}"

        # Average log likelihood should be between log_likelihood and 0
        assert score.log_likelihood <= score.avg_log_likelihood <= 0

        # Perplexity should be >= 1 (exp(0) = 1 is minimum when avg_ll = 0)
        assert score.perplexity >= 1.0, f"Perplexity should be >= 1, got {score.perplexity}"

        # Verify perplexity = exp(-avg_log_likelihood)
        expected_ppl = np.exp(-score.avg_log_likelihood)
        np.testing.assert_allclose(score.perplexity, expected_ppl, rtol=1e-5)

        # Logits shape: (seq_len, vocab_size=20 for standard amino acids)
        logits = np.array(score.logits)
        assert logits.shape[0] == len(seq), f"Logits seq_len should be {len(seq)}, got {logits.shape[0]}"
        assert logits.shape[1] == len(
            result.vocab
        ), f"Vocab size should match vocab list, got {logits.shape[1]}"


@pytest.mark.uses_gpu
def test_esm3_score_tool():
    """Test the esm3 scoring tool with run_esm3_score."""
    sequences = ["MKTAYIAKQR", "EVQLVESGGS"]
    inputs = ESM3ScoringInput(sequences=sequences)
    config = ESM3ScoringConfig(batch_size=32, verbose=False, return_logits=True)

    result = run_esm3_score(inputs=inputs, config=config)
    validate_output(result)

    # Validate tool output structure
    assert result.tool_id == "esm3-score"
    assert len(result.scores) == 2
    # ESM3 returns AA-only vocab (20 standard amino acids) as a list
    assert isinstance(result.vocab, list), f"Vocab should be a list, got {type(result.vocab)}"
    assert len(result.vocab) == 20, f"ESM3 vocab should have 20 tokens, got {len(result.vocab)}"

    for seq, score in zip(sequences, result.scores):
        # Check metrics
        assert "log_likelihood" in score.metrics
        assert "avg_log_likelihood" in score.metrics
        assert "perplexity" in score.metrics

        # Validate values
        assert score.perplexity >= 1.0
        assert score.log_likelihood < 0

        # Logits should be present with correct shape (as nested lists)
        assert score.logits is not None
        assert isinstance(score.logits, list), f"Logits should be a list, got {type(score.logits)}"
        assert len(score.logits) == len(seq), f"Logits length should be {len(seq)}, got {len(score.logits)}"
        assert len(score.logits[0]) == 20, f"Logits vocab size should be 20, got {len(score.logits[0])}"


@pytest.mark.uses_gpu
def test_esm3_score_different_sequences():
    """Test that model produces different perplexities for different sequences."""
    # Different sequences should produce different perplexities
    seq1 = "MVLSPADKTNVKAAW"  # Natural-looking sequence
    seq2 = "AAAAAAAAAAAAAAAA"  # Homopolymer

    inputs = ESM3ScoringInput(sequences=[seq1, seq2])
    config = ESM3ScoringConfig(verbose=False, return_logits=True)

    result = run_esm3_score(inputs=inputs, config=config)

    ppl1 = result.scores[0].perplexity
    ppl2 = result.scores[1].perplexity

    # Different sequences should have different perplexities
    assert ppl1 != ppl2, f"Different sequences should have different perplexities: {ppl1} vs {ppl2}"

    # Both should be valid perplexities
    assert ppl1 >= 1.0 and ppl2 >= 1.0


@pytest.mark.uses_gpu
def test_esm3_score_metrics_consistency():
    """Test that scoring metrics are mathematically consistent."""
    inputs = ESM3ScoringInput(sequences=["MVLSPADKTNVKAAW"])
    config = ESM3ScoringConfig(verbose=False, return_logits=True)

    result = run_esm3_score(inputs=inputs, config=config)
    score = result.scores[0]

    # Verify perplexity = exp(-avg_log_likelihood)
    expected_perplexity = np.exp(-score.avg_log_likelihood)
    np.testing.assert_allclose(score.perplexity, expected_perplexity, rtol=1e-5)

    # Verify avg = total / length
    seq_len = 15  # MVLSPADKTNVKAAW
    expected_avg = score.log_likelihood / seq_len
    np.testing.assert_allclose(score.avg_log_likelihood, expected_avg, rtol=1e-5)


@pytest.mark.uses_gpu
def test_esm3_score_batched():
    """Test batched scoring with different batch sizes."""
    sequences = ["MKTAYIAKQR", "EVQLVESGGS", "MVLSPADKTN", "GSSGSSGSS"]
    inputs = ESM3ScoringInput(sequences=sequences)
    config = ESM3ScoringConfig(batch_size=2, verbose=False, return_logits=True)

    result = run_esm3_score(inputs=inputs, config=config)

    assert len(result.scores) == 4

    for seq, score in zip(sequences, result.scores):
        assert score.perplexity >= 1.0
        assert score.log_likelihood < 0
        logits = np.array(score.logits)
        assert logits.shape[0] == len(seq)


@pytest.mark.uses_gpu
def test_esm3_score_variable_length():
    """Test scoring sequences of different lengths."""
    sequences = ["MK", "MKTA", "MKTAYIAK", "MKTAYIAKQRQISFVK"]
    inputs = ESM3ScoringInput(sequences=sequences)
    config = ESM3ScoringConfig(verbose=False, return_logits=True)

    result = run_esm3_score(inputs=inputs, config=config)

    for seq, score in zip(sequences, result.scores):
        # Logits should have correct shape for each sequence (as nested lists)
        assert isinstance(score.logits, list), f"Logits should be a list, got {type(score.logits)}"
        assert len(score.logits) == len(seq), (
            f"Sequence '{seq}' (len {len(seq)}): logits len should be {len(seq)}, got {len(score.logits)}"
        )
        assert len(score.logits[0]) == 20, f"Logits vocab size should be 20, got {len(score.logits[0])}"

        # Metrics should be valid
        assert score.perplexity >= 1.0
        assert score.log_likelihood < 0


@pytest.mark.uses_gpu
def test_esm3_score_single_sequence():
    """Test esm3 scoring with a single sequence (string input)."""
    # Single sequence should work
    inputs = ESM3ScoringInput(sequences="MKTAYIAKQRQISFVKSHFS")
    config = ESM3ScoringConfig(verbose=False, return_logits=True)

    result = run_esm3_score(inputs=inputs, config=config)
    validate_output(result)

    # Should score 1 sequence
    assert len(result.scores) == 1
    assert result.scores[0].perplexity > 0
    assert result.scores[0].logits is not None


# ============================================================================
# Logits-Specific Tests
# ============================================================================

@pytest.mark.uses_gpu
def test_esm3_score_logits_disabled_by_default():
    """Test that logits are None when return_logits=False (default)."""
    sequences = ["MKTAYIAKQR", "EVQLVESGGS"]
    inputs = ESM3ScoringInput(sequences=sequences)
    config = ESM3ScoringConfig(
        verbose=False,
        # return_logits defaults to False
    )

    result = run_esm3_score(inputs=inputs, config=config)
    validate_output(result)

    # Logits should be None when return_logits=False
    for score in result.scores:
        assert score.logits is None, "Logits should be None when return_logits=False"


@pytest.mark.uses_gpu
def test_esm3_score_logits_enabled():
    """Test that logits are correctly returned when return_logits=True."""
    sequences = ["MKTAYIAKQR", "EVQLVESGGS"]
    inputs = ESM3ScoringInput(sequences=sequences)
    config = ESM3ScoringConfig(
        verbose=False,
        return_logits=True,
    )

    result = run_esm3_score(inputs=inputs, config=config)
    validate_output(result)

    # Logits should be present with correct shape
    for seq, score in zip(sequences, result.scores):
        assert score.logits is not None, "Logits should not be None when return_logits=True"
        assert isinstance(score.logits, (list, np.ndarray)), f"Logits should be list or ndarray, got {type(score.logits)}"

        # Convert to ndarray for shape validation if it's a list
        logits_arr = np.array(score.logits)
        assert logits_arr.shape[0] == len(seq), f"Logits length should be {len(seq)}, got {logits_arr.shape[0]}"
        assert logits_arr.shape[1] == 20, f"Logits vocab size should be 20, got {logits_arr.shape[1]}"


@pytest.mark.uses_gpu
def test_esm3_score_logits_serialization():
    """Test that logits are properly serialized as nested lists."""
    sequences = ["MKTAYIAKQR"]
    inputs = ESM3ScoringInput(sequences=sequences)
    config = ESM3ScoringConfig(
        verbose=False,
        return_logits=True,
    )

    result = run_esm3_score(inputs=inputs, config=config)
    validate_output(result)

    score = result.scores[0]

    # Logits should be serialized as nested lists (not tensors)
    assert isinstance(score.logits, (list, np.ndarray)), "Logits should be list or ndarray"

    if isinstance(score.logits, list):
        # Verify nested list structure
        assert len(score.logits) > 0, "Logits list should not be empty"
        assert isinstance(score.logits[0], list), "Logits should be a list of lists"
        assert len(score.logits[0]) == 20, "Inner logits list should have 20 elements (vocab size)"

        # Verify all values are numeric
        for position_logits in score.logits:
            for logit_value in position_logits:
                assert isinstance(logit_value, (int, float)), f"Logit value should be numeric, got {type(logit_value)}"
    else:
        # If ndarray, verify shape
        assert score.logits.ndim == 2, "Logits should be 2D array"
        assert score.logits.shape[1] == 20, "Vocab size should be 20"
