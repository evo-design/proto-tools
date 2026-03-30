"""tests/masked_models_tests/test_esm2.py

Tests for ESM2."""
import numpy as np
import pytest

from bio_programming_tools.tools.masked_models.esm2 import (
    ESM2EmbeddingsConfig,
    ESM2EmbeddingsInput,
    ESM2ScoringConfig,
    ESM2ScoringInput,
    run_esm2_embeddings,
    run_esm2_score,
)
from tests.conftest import make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import validate_output

_persistent_tool = make_persistent_fixture("esm2")


# ── Input validation ─────────────────────────────────────────────────────────

def test_esm2_scoring_input_normalizes_single_string():
    inp = ESM2ScoringInput(sequences="MKTAYIAKQR")
    assert isinstance(inp.sequences, list)
    assert inp.sequences == ["MKTAYIAKQR"]


def test_esm2_embeddings_input_normalizes_single_string():
    inp = ESM2EmbeddingsInput(sequences="MKTAYIAKQR")
    assert isinstance(inp.sequences, list)
    assert inp.sequences == ["MKTAYIAKQR"]


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

# ── Embedding tests ───────────────────────────────────────────────────────────

@pytest.mark.uses_gpu
def test_esm2_forward_pass():
    sequences = ["TARGET"] * 10 + ["TEST"] * 30

    inputs = ESM2EmbeddingsInput(sequences=sequences)
    config = ESM2EmbeddingsConfig(model_checkpoint="esm2_t33_650M_UR50D", batch_size=2, return_logits=True)

    result = run_esm2_embeddings(inputs=inputs, config=config)

    # SequenceEmbedding bundle (primary field)
    assert len(result.results) == 40, "Should have 40 SequenceEmbedding objects"
    assert len(result.results[0].mean_embedding) == 1280, "Embedding dimension should be 1280"
    assert len(result.results[0].attention_mask) == 6, "Attention mask length should be 6"
    assert result.results[0].logits is not None
    assert len(result.results[0].logits) == 6, "Logit sequence length should be 6"

    # Logit details
    assert len(result.results[0].logits[0]) == 20, "Logit vocab size should be 20"


# ── Scoring tests ─────────────────────────────────────────────────────────────

@pytest.mark.include_in_env_report(category="masked_models")
@pytest.mark.uses_gpu
def test_esm2_score_tool():
    """Test the esm2 scoring tool with run_esm2_score."""
    sequences = ["MKTAYIAKQR", "EVQLVESGGS"]
    inputs = ESM2ScoringInput(sequences=sequences)
    config = ESM2ScoringConfig(
        model_checkpoint="esm2_t33_650M_UR50D",
        batch_size=32,
        verbose=False,
        return_logits=True,
    )

    result = run_esm2_score(inputs=inputs, config=config)
    validate_output(result)

    assert result.tool_id == "esm2-score"
    assert len(result.scores) == 2
    assert isinstance(result.vocab, list), f"Vocab should be a list, got {type(result.vocab)}"
    assert len(result.vocab) == 20, f"ESM2 vocab should have 20 tokens, got {len(result.vocab)}"

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
def test_esm2_score_different_sequences():
    """Test that model produces different perplexities for different sequences."""
    seq1 = "MVLSPADKTNVKAAW"
    seq2 = "AAAAAAAAAAAAAAAA"

    inputs = ESM2ScoringInput(sequences=[seq1, seq2])
    config = ESM2ScoringConfig(model_checkpoint="esm2_t33_650M_UR50D", verbose=False, return_logits=True)

    result = run_esm2_score(inputs=inputs, config=config)

    ppl1 = result.scores[0].perplexity
    ppl2 = result.scores[1].perplexity

    assert ppl1 != ppl2, f"Different sequences should have different perplexities: {ppl1} vs {ppl2}"
    assert ppl1 >= 1.0 and ppl2 >= 1.0


@pytest.mark.uses_gpu
def test_esm2_score_metrics_consistency():
    """Test that scoring metrics are mathematically consistent."""
    _seq = "MVLSPADKTNVKAAW"
    inputs = ESM2ScoringInput(sequences=[_seq])
    config = ESM2ScoringConfig(model_checkpoint="esm2_t33_650M_UR50D", verbose=False, return_logits=True)

    result = run_esm2_score(inputs=inputs, config=config)
    score = result.scores[0]

    expected_perplexity = np.exp(-score.avg_log_likelihood)
    np.testing.assert_allclose(score.perplexity, expected_perplexity, rtol=1e-5)

    expected_avg = score.log_likelihood / len(_seq)
    np.testing.assert_allclose(score.avg_log_likelihood, expected_avg, rtol=1e-5)


@pytest.mark.uses_gpu
def test_esm2_score_batched():
    """Test batched scoring with different batch sizes."""
    sequences = ["MKTAYIAKQR", "EVQLVESGGS", "MVLSPADKTN", "GSSGSSGSS"]
    inputs = ESM2ScoringInput(sequences=sequences)
    config = ESM2ScoringConfig(model_checkpoint="esm2_t33_650M_UR50D", batch_size=2, verbose=False, return_logits=True)

    result = run_esm2_score(inputs=inputs, config=config)

    assert len(result.scores) == 4

    for seq, score in zip(sequences, result.scores):
        assert score.perplexity >= 1.0
        assert score.log_likelihood < 0
        logits = np.array(score.logits)
        assert logits.shape[0] == len(seq)
        assert logits.shape[1] == 20


@pytest.mark.uses_gpu
def test_esm2_score_variable_length():
    """Test scoring sequences of different lengths."""
    sequences = ["MK", "MKTA", "MKTAYIAK", "MKTAYIAKQRQISFVK"]
    inputs = ESM2ScoringInput(sequences=sequences)
    config = ESM2ScoringConfig(model_checkpoint="esm2_t33_650M_UR50D", verbose=False, return_logits=True)

    result = run_esm2_score(inputs=inputs, config=config)

    for seq, score in zip(sequences, result.scores):
        assert isinstance(score.logits, list), f"Logits should be a list, got {type(score.logits)}"
        assert len(score.logits) == len(seq), (
            f"Sequence '{seq}' (len {len(seq)}): logits len should be {len(seq)}, got {len(score.logits)}"
        )
        assert len(score.logits[0]) == 20, f"Logits vocab size should be 20, got {len(score.logits[0])}"

        assert score.perplexity >= 1.0
        assert score.log_likelihood < 0


@pytest.mark.uses_gpu
def test_esm2_score_single_sequence():
    """Test esm2 scoring with a single sequence (string input)."""
    inputs = ESM2ScoringInput(sequences="MKTAYIAKQRQISFVKSHFS")
    config = ESM2ScoringConfig(model_checkpoint="esm2_t33_650M_UR50D", verbose=False, return_logits=True)

    result = run_esm2_score(inputs=inputs, config=config)
    validate_output(result)

    assert len(result.scores) == 1
    assert result.scores[0].perplexity >= 1.0
    assert result.scores[0].logits is not None


# ── Logits-specific tests ─────────────────────────────────────────────────────

@pytest.mark.uses_gpu
def test_esm2_score_logits_disabled_by_default():
    """Test that logits are None when return_logits=False (default)."""
    sequences = ["MKTAYIAKQR", "EVQLVESGGS"]
    inputs = ESM2ScoringInput(sequences=sequences)
    config = ESM2ScoringConfig(
        model_checkpoint="esm2_t33_650M_UR50D",
        verbose=False,
    )

    result = run_esm2_score(inputs=inputs, config=config)
    validate_output(result)

    for score in result.scores:
        assert score.logits is None, "Logits should be None when return_logits=False"


@pytest.mark.uses_gpu
def test_esm2_score_logits_serialization():
    """Test that logits are properly serialized as nested lists."""
    sequences = ["MKTAYIAKQR"]
    inputs = ESM2ScoringInput(sequences=sequences)
    config = ESM2ScoringConfig(
        model_checkpoint="esm2_t33_650M_UR50D",
        verbose=False,
        return_logits=True,
    )

    result = run_esm2_score(inputs=inputs, config=config)
    validate_output(result)

    score = result.scores[0]

    assert isinstance(score.logits, list), "Logits should be a list"
    assert len(score.logits) > 0, "Logits list should not be empty"
    assert isinstance(score.logits[0], list), "Logits should be a list of lists"
    assert len(score.logits[0]) == 20, "Inner logits list should have 20 elements (vocab size)"

    for position_logits in score.logits:
        for logit_value in position_logits:
            assert isinstance(logit_value, (int, float)), f"Logit value should be numeric, got {type(logit_value)}"
