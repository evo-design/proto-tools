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
from tests.conftest import make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import validate_output

_persistent_tool = make_persistent_fixture("evo2")

_EVO2_TEST_CHECKPOINTS = ["evo2_7b", "evo2_20b"]


# ── Sample input/config validation ────────────────────────────────────────────


@pytest.mark.parametrize(
    "input_kwargs,match",
    [
        ({"prompts": []}, "prompts must not be empty"),
    ],
)
def test_evo2_sample_input_validation(input_kwargs, match):
    with pytest.raises(ValueError, match=match):
        Evo2SampleInput(**input_kwargs)


@pytest.mark.parametrize(
    "config_kwargs,match",
    [
        ({"temperature": 0.0}, "greater than 0"),
        ({"top_p": 1.5}, "less than or equal to 1"),
        ({"num_tokens": 0}, "greater than or equal to 1"),
    ],
)
def test_evo2_sample_config_validation(config_kwargs, match):
    with pytest.raises(ValueError, match=match):
        Evo2SampleConfig(**config_kwargs)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.uses_gpu
def test_evo2_sample_tool(model_checkpoint="evo2_7b"):
    """Test the evo2 sampling tool with run_evo2_sample."""
    prompts = ["ATCG", "GCTA"]
    inputs = Evo2SampleInput(prompts=prompts)
    config = Evo2SampleConfig(
        model_checkpoint=model_checkpoint,
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

    assert result.tool_id == "evo2-sample"
    assert result.metadata["model_checkpoint"] == model_checkpoint
    assert result.metadata["num_tokens"] == 50
    assert result.metadata["temperature"] == 0.8
    assert len(result.sequences) == 2

    valid_chars = set("ATCGN")
    for i, (prompt, seq) in enumerate(zip(prompts, result.sequences, strict=False)):
        assert seq.startswith(prompt), f"Sequence {i} should start with '{prompt}'"
        assert set(seq.upper()).issubset(valid_chars)
        assert len(seq) > len(prompt), f"Sequence {i} should be longer than prompt"


@pytest.mark.uses_gpu
@pytest.mark.parametrize("model_checkpoint", _EVO2_TEST_CHECKPOINTS)
@pytest.mark.parametrize(
    "prompt",
    [
        "ATCGATCG",
        "GCTAGCTA",
        "AAAACCCC",
    ],
)
def test_evo2_sample_prompt_handling(prompt, model_checkpoint):
    """Test evo2 sampling with various prompt formats."""
    inputs = Evo2SampleInput(prompts=prompt)
    config = Evo2SampleConfig(
        model_checkpoint=model_checkpoint,
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
@pytest.mark.parametrize("model_checkpoint", _EVO2_TEST_CHECKPOINTS)
def test_evo2_sample_prepend_prompt(model_checkpoint):
    """Test that prepend_prompt controls whether prompt is included."""
    prompt = "ATCGATCG"

    result_with = run_evo2_sample(
        inputs=Evo2SampleInput(prompts=[prompt]),
        config=Evo2SampleConfig(
            model_checkpoint=model_checkpoint,
            num_tokens=50,
            prepend_prompt=True,
            verbose=False,
            print_generation=False,
        ),
    )
    assert result_with.sequences[0].startswith(prompt)

    result_without = run_evo2_sample(
        inputs=Evo2SampleInput(prompts=[prompt]),
        config=Evo2SampleConfig(
            model_checkpoint=model_checkpoint,
            num_tokens=50,
            prepend_prompt=False,
            verbose=False,
            print_generation=False,
        ),
    )
    assert not result_without.sequences[0].startswith(prompt)
    assert len(result_without.sequences[0]) > 0


# ── Batched sampling tests ────────────────────────────────────────────────────


@pytest.mark.uses_gpu
@pytest.mark.parametrize("model_checkpoint", _EVO2_TEST_CHECKPOINTS)
def test_evo2_sample_batched_tool(model_checkpoint):
    """Test batched sampling with tool layer (run_evo2_sample)."""
    prompts = ["ATCG", "GCTA", "AAAA", "GGGG", "CCCC", "TTTT"]
    inputs = Evo2SampleInput(prompts=prompts)
    config = Evo2SampleConfig(
        model_checkpoint=model_checkpoint,
        num_tokens=50,
        temperature=1.0,
        batch_size=2,
        verbose=False,
        print_generation=False,
    )

    result = run_evo2_sample(inputs=inputs, config=config)
    validate_output(result)

    assert len(result.sequences) == 6

    for i, (prompt, seq) in enumerate(zip(prompts, result.sequences, strict=False)):
        assert seq.startswith(prompt), f"Sequence {i} should start with '{prompt}'"
        assert len(seq) > len(prompt)


# ── Scoring tests ─────────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
@pytest.mark.parametrize("model_checkpoint", _EVO2_TEST_CHECKPOINTS)
def test_evo2_score_different_sequences(model_checkpoint):
    """Test that model produces different perplexities for different sequences."""
    seq1 = "ATCGATCGATCGATCGATCG"
    seq2 = "AAAAAAAAAAAAAAAAAAAA"

    inputs = Evo2ScoringInput(sequences=[seq1, seq2])
    config = Evo2ScoringConfig(model_checkpoint=model_checkpoint, verbose=False, return_logits=True)

    result = run_evo2_score(inputs=inputs, config=config)

    ppl1 = result.scores[0].perplexity
    ppl2 = result.scores[1].perplexity

    assert ppl1 != ppl2, f"Different sequences should have different perplexities: {ppl1} vs {ppl2}"
    assert ppl1 >= 1.0 and ppl2 >= 1.0


@pytest.mark.uses_gpu
@pytest.mark.parametrize("model_checkpoint", _EVO2_TEST_CHECKPOINTS)
def test_evo2_score_metrics_consistency(model_checkpoint):
    """Test that scoring metrics are mathematically consistent."""
    inputs = Evo2ScoringInput(sequences=["ATCGATCGATCGATCG"])
    config = Evo2ScoringConfig(model_checkpoint=model_checkpoint, verbose=False, return_logits=True)

    result = run_evo2_score(inputs=inputs, config=config)
    score = result.scores[0]

    expected_perplexity = np.exp(-score.avg_log_likelihood)
    np.testing.assert_allclose(score.perplexity, expected_perplexity, rtol=1e-5)

    assert score.avg_log_likelihood <= 0, "avg_log_likelihood should be <= 0"
    assert score.log_likelihood <= score.avg_log_likelihood, "log_likelihood should be <= avg_log_likelihood"


# ── Batched scoring tests ────────────────────────────────────────────────────


@pytest.mark.uses_gpu
@pytest.mark.parametrize("model_checkpoint", _EVO2_TEST_CHECKPOINTS)
def test_evo2_score_batched_tool(model_checkpoint):
    """Test batched scoring with tool layer (run_evo2_score)."""
    sequences = ["ATCGATCG", "GCTAGCTA", "AAAACCCC", "TTTTGGGG", "CCCCAAAA", "GGGGTTTT"]
    inputs = Evo2ScoringInput(sequences=sequences)
    config = Evo2ScoringConfig(
        model_checkpoint=model_checkpoint,
        batch_size=2,
        verbose=False,
        return_logits=True,
    )

    result = run_evo2_score(inputs=inputs, config=config)
    validate_output(result)

    assert len(result.scores) == 6

    for _seq, score in zip(sequences, result.scores, strict=False):
        assert score.log_likelihood < 0
        assert score.perplexity >= 1.0
        assert score.logits is not None, "Logits should be present when return_logits=True"
        assert len(score.logits) > 0
        assert len(score.logits[0]) >= 4


@pytest.mark.uses_gpu
@pytest.mark.parametrize("model_checkpoint", _EVO2_TEST_CHECKPOINTS)
def test_evo2_score_batch_size_consistency(model_checkpoint):
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
            config=Evo2ScoringConfig(model_checkpoint=model_checkpoint, batch_size=bs, verbose=False),
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
@pytest.mark.parametrize("model_checkpoint", _EVO2_TEST_CHECKPOINTS)
def test_evo2_score_variable_length_sequences(model_checkpoint):
    """Test scoring sequences of different lengths produces correct logits shapes."""
    sequences = ["AT", "ATCG", "ATCGATCG", "ATCGATCGATCG"]
    inputs = Evo2ScoringInput(sequences=sequences)
    config = Evo2ScoringConfig(model_checkpoint=model_checkpoint, batch_size=2, verbose=False, return_logits=True)

    result = run_evo2_score(inputs=inputs, config=config)

    for seq, score in zip(sequences, result.scores, strict=False):
        assert score.logits is not None, "Logits should be present when return_logits=True"
        assert len(score.logits) > 0, (
            f"Sequence '{seq}' (len {len(seq)}): logits len should be > 0, got {len(score.logits)}"
        )

        assert score.perplexity >= 1.0
        assert score.log_likelihood < 0


@pytest.mark.uses_gpu
@pytest.mark.parametrize("model_checkpoint", _EVO2_TEST_CHECKPOINTS)
def test_evo2_score_tool(model_checkpoint):
    """Test the evo2 scoring tool with run_evo2_score."""
    sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA"]
    inputs = Evo2ScoringInput(sequences=sequences)
    config = Evo2ScoringConfig(
        model_checkpoint=model_checkpoint,
        verbose=False,
        return_logits=True,
    )

    result = run_evo2_score(inputs=inputs, config=config)
    validate_output(result)

    assert result.tool_id == "evo2-score", "Tool ID should be correct"
    assert len(result.scores) == 2, "Should score 2 sequences"
    assert len(result.vocab) == 512, "Evo2 vocab should be 512 (byte-level)"
    assert result.vocab[65] == "A" and result.vocab[67] == "C"
    assert result.vocab[71] == "G" and result.vocab[84] == "T"

    for i, score in enumerate(result.scores):
        assert isinstance(score.log_likelihood, float), "log_likelihood should be float"
        assert isinstance(score.avg_log_likelihood, float), "avg_log_likelihood should be float"
        assert isinstance(score.perplexity, float), "perplexity should be float"

        assert score.perplexity >= 1.0, f"Perplexity should be >= 1.0, got {score.perplexity}"
        assert score.logits is not None, f"Sequence {i} should have logits"


@pytest.mark.uses_gpu
@pytest.mark.parametrize("model_checkpoint", _EVO2_TEST_CHECKPOINTS)
def test_evo2_score_single_sequence(model_checkpoint):
    """Test evo2 scoring with a single sequence (string input)."""
    inputs = Evo2ScoringInput(sequences="ATCGATCGATCGATCGATCG")
    config = Evo2ScoringConfig(
        model_checkpoint=model_checkpoint,
        verbose=False,
    )

    result = run_evo2_score(inputs=inputs, config=config)
    validate_output(result)

    assert len(result.scores) == 1
    assert result.scores[0].perplexity >= 1.0


# ── Logits-specific tests (scoring) ──────────────────────────────────────────


@pytest.mark.uses_gpu
@pytest.mark.parametrize("model_checkpoint", _EVO2_TEST_CHECKPOINTS)
def test_evo2_score_logits_disabled_by_default(model_checkpoint):
    """Test that logits are None when return_logits=False (default)."""
    sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA"]
    inputs = Evo2ScoringInput(sequences=sequences)
    config = Evo2ScoringConfig(
        model_checkpoint=model_checkpoint,
        verbose=False,
    )

    result = run_evo2_score(inputs=inputs, config=config)
    validate_output(result)

    for score in result.scores:
        assert score.logits is None, "Logits should be None when return_logits=False"


@pytest.mark.uses_gpu
@pytest.mark.parametrize("model_checkpoint", _EVO2_TEST_CHECKPOINTS)
def test_evo2_score_logits_serialization(model_checkpoint):
    """Test that logits are properly serialized as nested lists."""
    sequences = ["ATCGATCGATCG"]
    inputs = Evo2ScoringInput(sequences=sequences)
    config = Evo2ScoringConfig(
        model_checkpoint=model_checkpoint,
        verbose=False,
        return_logits=True,
    )

    result = run_evo2_score(inputs=inputs, config=config)
    validate_output(result)

    score = result.scores[0]

    assert isinstance(score.logits, list), "Logits should be a list"
    assert len(score.logits) > 0, "Logits list should not be empty"
    assert isinstance(score.logits[0], list), "Logits should be a list of lists"
    assert len(score.logits[0]) == 512, "Inner logits list should have 512 elements (Evo2 byte-level vocab)"

    for position_logits in score.logits:
        for logit_value in position_logits:
            assert isinstance(logit_value, (int, float)), f"Logit value should be numeric, got {type(logit_value)}"
