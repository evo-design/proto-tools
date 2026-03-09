"""Tests for ESM3."""

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
from tests.conftest import make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import validate_output

_GFP = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"


_persistent_tool = make_persistent_fixture("esm3")


# ── Input validation ─────────────────────────────────────────────────────────

def test_esm3_scoring_input_normalizes_single_string():
    inp = ESM3ScoringInput(sequences="MKTAYIAKQR")
    assert isinstance(inp.sequences, list)
    assert inp.sequences == ["MKTAYIAKQR"]


def test_esm3_embeddings_input_normalizes_single_string():
    inp = ESM3EmbeddingsInput(sequences="MKTAYIAKQR")
    assert isinstance(inp.sequences, list)
    assert inp.sequences == ["MKTAYIAKQR"]


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

# ── Embedding tests ───────────────────────────────────────────────────────────

@pytest.mark.uses_gpu
def test_esm3_forward_pass():
    sequences = ["TARGET"] * 10 + ["TEST"] * 30

    inputs = ESM3EmbeddingsInput(sequences=sequences)
    config = ESM3EmbeddingsConfig(batch_size=2, return_logits=True)

    result = run_esm3_embeddings(inputs=inputs, config=config)
    assert result.success, "ESM3 embeddings failed"

    assert len(result.mean_embeddings) == 40, "Should have 40 sequences"
    assert len(result.mean_embeddings[0]) == 1536, "Embedding dimension should be 1536"

    assert len(result.attention_masks) == 40, "Should have 40 attention masks"
    assert len(result.attention_masks[0]) == 6, "Attention mask length should be 6"

    assert len(result.logits) == 40, "Should have 40 logit arrays"
    assert len(result.logits[0]) == 6, "Logit sequence length should be 6"
    assert len(result.logits[0][0]) == 20, "Logit vocab size should be 20"


@pytest.mark.uses_gpu
def test_esm3_predict_structure():
    sequences = [_GFP] * 2

    inputs = ESM3StructurePredictionInput(sequences=sequences)
    config = ESM3StructurePredictionConfig(batch_size=2)

    result = run_esm3_structure_prediction(inputs=inputs, config=config)

    assert len(result.structures) == 2


# ── Scoring tests ─────────────────────────────────────────────────────────────

@pytest.mark.include_in_env_report(category="masked_models")
@pytest.mark.uses_gpu
def test_esm3_score_tool():
    """Test the esm3 scoring tool with run_esm3_score."""
    sequences = ["MKTAYIAKQR", "EVQLVESGGS"]
    inputs = ESM3ScoringInput(sequences=sequences)
    config = ESM3ScoringConfig(batch_size=32, verbose=False, return_logits=True)

    result = run_esm3_score(inputs=inputs, config=config)
    validate_output(result)

    assert result.tool_id == "esm3-score"
    assert len(result.scores) == 2
    assert isinstance(result.vocab, list), f"Vocab should be a list, got {type(result.vocab)}"
    assert len(result.vocab) == 20, f"ESM3 vocab should have 20 tokens, got {len(result.vocab)}"

    for seq, score in zip(sequences, result.scores):
        assert "log_likelihood" in score.metrics
        assert "avg_log_likelihood" in score.metrics
        assert "perplexity" in score.metrics

        assert score.perplexity >= 1.0
        assert score.log_likelihood < 0

        assert score.logits is not None
        assert isinstance(score.logits, list), f"Logits should be a list, got {type(score.logits)}"
        assert len(score.logits) == len(seq), f"Logits length should be {len(seq)}, got {len(score.logits)}"
        assert len(score.logits[0]) == 20, f"Logits vocab size should be 20, got {len(score.logits[0])}"

    for score in result.scores:
        expected_ppl = np.exp(-score.avg_log_likelihood)
        np.testing.assert_allclose(score.perplexity, expected_ppl, rtol=1e-5)


@pytest.mark.uses_gpu
def test_esm3_score_different_sequences():
    """Test that model produces different perplexities for different sequences."""
    seq1 = "MVLSPADKTNVKAAW"
    seq2 = "AAAAAAAAAAAAAAAA"

    inputs = ESM3ScoringInput(sequences=[seq1, seq2])
    config = ESM3ScoringConfig(verbose=False, return_logits=True)

    result = run_esm3_score(inputs=inputs, config=config)

    ppl1 = result.scores[0].perplexity
    ppl2 = result.scores[1].perplexity

    assert ppl1 != ppl2, f"Different sequences should have different perplexities: {ppl1} vs {ppl2}"
    assert ppl1 >= 1.0 and ppl2 >= 1.0


@pytest.mark.uses_gpu
def test_esm3_score_metrics_consistency():
    """Test that scoring metrics are mathematically consistent."""
    _seq = "MVLSPADKTNVKAAW"
    inputs = ESM3ScoringInput(sequences=[_seq])
    config = ESM3ScoringConfig(verbose=False, return_logits=True)

    result = run_esm3_score(inputs=inputs, config=config)
    score = result.scores[0]

    expected_perplexity = np.exp(-score.avg_log_likelihood)
    np.testing.assert_allclose(score.perplexity, expected_perplexity, rtol=1e-5)

    expected_avg = score.log_likelihood / len(_seq)
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
        assert isinstance(score.logits, list), f"Logits should be a list, got {type(score.logits)}"
        assert len(score.logits) == len(seq), (
            f"Sequence '{seq}' (len {len(seq)}): logits len should be {len(seq)}, got {len(score.logits)}"
        )
        assert len(score.logits[0]) == 20, f"Logits vocab size should be 20, got {len(score.logits[0])}"

        assert score.perplexity >= 1.0
        assert score.log_likelihood < 0


@pytest.mark.uses_gpu
def test_esm3_score_single_sequence():
    """Test esm3 scoring with a single sequence (string input)."""
    inputs = ESM3ScoringInput(sequences="MKTAYIAKQRQISFVKSHFS")
    config = ESM3ScoringConfig(verbose=False, return_logits=True)

    result = run_esm3_score(inputs=inputs, config=config)
    validate_output(result)

    assert len(result.scores) == 1
    assert result.scores[0].perplexity >= 1.0
    assert result.scores[0].logits is not None


# ── Logits-specific tests ─────────────────────────────────────────────────────

@pytest.mark.uses_gpu
def test_esm3_score_logits_disabled_by_default():
    """Test that logits are None when return_logits=False (default)."""
    sequences = ["MKTAYIAKQR", "EVQLVESGGS"]
    inputs = ESM3ScoringInput(sequences=sequences)
    config = ESM3ScoringConfig(
        verbose=False,
    )

    result = run_esm3_score(inputs=inputs, config=config)
    validate_output(result)

    for score in result.scores:
        assert score.logits is None, "Logits should be None when return_logits=False"


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

    assert isinstance(score.logits, list), "Logits should be a list"
    assert len(score.logits) > 0, "Logits list should not be empty"
    assert isinstance(score.logits[0], list), "Logits should be a list of lists"
    assert len(score.logits[0]) == 20, "Inner logits list should have 20 elements (vocab size)"

    for position_logits in score.logits:
        for logit_value in position_logits:
            assert isinstance(logit_value, (int, float)), f"Logit value should be numeric, got {type(logit_value)}"
