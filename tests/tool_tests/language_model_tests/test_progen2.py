"""
test_progen2.py

Tests the ProGen2 implementation.
"""

import numpy as np
import pytest

from tests.tool_tests.tool_infra_tests.test_export_functionality import validate_output


# ============================================================================
# Sampling Tests
# ============================================================================

@pytest.mark.uses_gpu
def test_progen2_sample_inference():
    """Test ProGen2Model inference with direct model.sample() call."""
    from bio_programming.tools.language_models.progen2 import ProGen2Model
    from bio_programming.language.core.sequence import PROTEIN_AMINO_ACIDS

    prompts = ["1MKTLV", "1EVQLVE"]
    progen2_model = ProGen2Model(model_checkpoint="progen2-small")

    result = progen2_model.sample(
        prompts=prompts,
        max_length=100,
        temperature=0.2,
        top_p=0.95,
        top_k=0,
        truncate_at_stop=True,
        strip_special_tokens=True,
        prepend_prompt=True,
        verbose=False,
    )

    sequences = result["sequences"]
    assert len(sequences) == 2

    # Validate actual sequence content
    valid_chars = set(PROTEIN_AMINO_ACIDS)
    for i, (prompt, seq) in enumerate(zip(prompts, sequences)):
        # Sequence should be non-empty string of valid amino acids
        assert isinstance(seq, str) and len(seq) > 0
        assert set(seq.upper()).issubset(valid_chars), f"Invalid chars: {set(seq) - valid_chars}"

        # Sequence should contain the prompt (minus start token)
        prompt_aa = prompt[1:]  # Remove '1' start token
        assert seq.startswith(prompt_aa), f"Sequence {i} should start with prompt '{prompt_aa}'"

        # Sequence length should not exceed max_length (prompt + generated)
        # Note: max_length includes the entire sequence
        assert len(seq) <= 100, f"Sequence {i} exceeds max_length"


@pytest.mark.uses_gpu
def test_progen2_sample_tool():
    """Test the progen2 sampling tool with run_progen2_sample."""
    from bio_programming.tools.language_models.progen2 import (
        ProGen2SampleConfig,
        ProGen2SampleInput,
        run_progen2_sample,
    )
    from bio_programming.language.core.sequence import PROTEIN_AMINO_ACIDS

    prompts = ["1MKTL", "1EVQLV"]
    inputs = ProGen2SampleInput(prompts=prompts)
    config = ProGen2SampleConfig(
        model_checkpoint="progen2-small",
        max_length=100,
        temperature=0.2,
        top_p=0.95,
        verbose=False,
    )

    result = run_progen2_sample(inputs=inputs, config=config)
    validate_output(result)

    # Validate tool output structure
    assert result.tool_id == "progen2-sample"
    assert result.metadata["model_checkpoint"] == "progen2-small"
    assert result.metadata["max_length"] == 100
    assert result.metadata["temperature"] == 0.2
    assert len(result.sequences) == 2

    # Validate sequences contain prompts and are valid
    valid_chars = set(PROTEIN_AMINO_ACIDS)
    for i, (prompt, seq) in enumerate(zip(prompts, result.sequences)):
        prompt_aa = prompt[1:]
        assert seq.startswith(prompt_aa), f"Sequence {i} should start with '{prompt_aa}'"
        assert set(seq.upper()).issubset(valid_chars)
        assert len(seq) <= 100


@pytest.mark.uses_gpu
@pytest.mark.parametrize("prompt,expected_prefix", [
    ("1MKTLV", "MKTLV"),   # With start token
    ("MKTLV", "MKTLV"),    # Raw amino acids (auto-normalized)
    ("<|pf03668|>1MEVVIVTGMSGAGK", "MEVVIVTGMSGAGK"),  # With domain tag
])
def test_progen2_sample_prompt_handling(prompt, expected_prefix):
    """Test progen2 sampling with various prompt formats."""
    from bio_programming.tools.language_models.progen2 import (
        ProGen2SampleConfig,
        ProGen2SampleInput,
        run_progen2_sample,
    )

    inputs = ProGen2SampleInput(prompts=prompt)
    config = ProGen2SampleConfig(
        model_checkpoint="progen2-small",
        max_length=100,
        temperature=0.2,
        verbose=False,
    )

    result = run_progen2_sample(inputs=inputs, config=config)
    validate_output(result)

    assert len(result.sequences) == 1
    seq = result.sequences[0]
    assert expected_prefix in seq, f"Expected '{expected_prefix}' in output"
    assert len(seq) > len(expected_prefix), "Should generate beyond prompt"


@pytest.mark.uses_gpu
def test_progen2_sample_max_length_respected():
    """Test that max_length is strictly respected."""
    from bio_programming.tools.language_models.progen2 import (
        ProGen2SampleConfig,
        ProGen2SampleInput,
        run_progen2_sample,
    )

    inputs = ProGen2SampleInput(prompts="1M")  # Short prompt
    config = ProGen2SampleConfig(
        model_checkpoint="progen2-small",
        max_length=20,  # Short max length
        temperature=0.2,
        verbose=False,
    )

    result = run_progen2_sample(inputs=inputs, config=config)

    # Sequence (after stripping special tokens) should not exceed max_length
    assert len(result.sequences[0]) <= 20


@pytest.mark.uses_gpu
def test_progen2_sample_special_token_stripping():
    """Test special token stripping behavior."""
    from bio_programming.tools.language_models.progen2 import (
        ProGen2SampleConfig,
        ProGen2SampleInput,
        run_progen2_sample,
    )
    from bio_programming.language.core.sequence import PROTEIN_AMINO_ACIDS

    # Test with stripping (default)
    result_stripped = run_progen2_sample(
        inputs=ProGen2SampleInput(prompts="1MKTLV"),
        config=ProGen2SampleConfig(
            model_checkpoint="progen2-small",
            max_length=50,
            strip_special_tokens=True,
            verbose=False,
        ),
    )

    # Test without stripping
    result_unstripped = run_progen2_sample(
        inputs=ProGen2SampleInput(prompts="1MKTLV"),
        config=ProGen2SampleConfig(
            model_checkpoint="progen2-small",
            max_length=50,
            strip_special_tokens=False,
            verbose=False,
        ),
    )

    # Stripped should NOT start with '1' and should contain only amino acids
    assert not result_stripped.sequences[0].startswith("1"), "Stripped sequence should not start with '1'"
    assert set(result_stripped.sequences[0]).issubset(set(PROTEIN_AMINO_ACIDS)), "Stripped should only contain amino acids"

    # Unstripped SHOULD start with '1' (and may end with '2')
    assert result_unstripped.sequences[0].startswith("1"), "Unstripped sequence should start with '1'"


@pytest.mark.parametrize("config_kwargs,match", [
    ({"prompts": []}, "prompts must not be empty"),
])
def test_progen2_sample_input_validation(config_kwargs, match):
    """Test ProGen2SampleInput validation."""
    from bio_programming.tools.language_models.progen2 import ProGen2SampleInput

    with pytest.raises(ValueError, match=match):
        ProGen2SampleInput(**config_kwargs)


@pytest.mark.parametrize("config_kwargs", [
    {"temperature": 0.0},
    {"top_p": 1.5},
    {"max_length": 0},
])
def test_progen2_sample_config_validation(config_kwargs):
    """Test ProGen2SampleConfig validation for invalid values."""
    from bio_programming.tools.language_models.progen2 import ProGen2SampleConfig

    with pytest.raises(ValueError):
        ProGen2SampleConfig(**config_kwargs)


@pytest.mark.uses_gpu
def test_progen2_sample_special_tokens():
    """Test ProGen2 with domain tag in prompt."""
    from bio_programming.tools.language_models.progen2 import (
        ProGen2SampleConfig,
        ProGen2SampleInput,
        run_progen2_sample,
    )

    inputs = ProGen2SampleInput(prompts="<|pf03668|>1MEVVIVTGMSGAGK")
    config = ProGen2SampleConfig(
        model_checkpoint="progen2-small",
        max_length=100,
        temperature=0.2,
        verbose=False,
    )

    result = run_progen2_sample(inputs=inputs, config=config)
    validate_output(result)

    assert len(result.sequences) == 1
    assert len(result.sequences[0]) > 0


@pytest.mark.uses_gpu
def test_progen2_sample_without_special_token_stripping():
    """Test ProGen2 without stripping special tokens."""
    from bio_programming.tools.language_models.progen2 import (
        ProGen2SampleConfig,
        ProGen2SampleInput,
        run_progen2_sample,
    )

    inputs = ProGen2SampleInput(prompts="1MKTLV")
    config = ProGen2SampleConfig(
        model_checkpoint="progen2-small",
        max_length=100,
        temperature=0.2,
        strip_special_tokens=False,
        verbose=False,
    )

    result = run_progen2_sample(inputs=inputs, config=config)
    validate_output(result)

    assert len(result.sequences) == 1
    assert result.sequences[0].startswith("1")


# ============================================================================
# Batched Sampling Tests
# ============================================================================

@pytest.mark.uses_gpu
def test_progen2_sample_batched_inference():
    """Test batched sampling with direct model call."""
    from bio_programming.tools.language_models.progen2 import ProGen2Model

    prompts = ["1MKTLV", "1EVQLV", "1AAAAA", "1GGGGG"]
    progen2_model = ProGen2Model(model_checkpoint="progen2-small")

    result = progen2_model.sample(
        prompts=prompts,
        max_length=50,
        temperature=0.2,
        batch_size=2,
        verbose=False,
    )

    sequences = result["sequences"]
    assert len(sequences) == 4

    # Each sequence should start with its corresponding prompt
    for i, (prompt, seq) in enumerate(zip(prompts, sequences)):
        prompt_aa = prompt[1:]
        assert seq.startswith(prompt_aa), f"Sequence {i} should start with '{prompt_aa}', got '{seq[:10]}'"


@pytest.mark.uses_gpu
def test_progen2_sample_batched_tool():
    """Test batched sampling with tool layer (run_progen2_sample)."""
    from bio_programming.tools.language_models.progen2 import (
        ProGen2SampleConfig,
        ProGen2SampleInput,
        run_progen2_sample,
    )

    prompts = ["1MKTL", "1EVQL", "1AAAA", "1GGGG", "1LLLL", "1VVVV"]
    inputs = ProGen2SampleInput(prompts=prompts)
    config = ProGen2SampleConfig(
        model_checkpoint="progen2-small",
        max_length=50,
        temperature=0.2,
        batch_size=2,
        verbose=False,
    )

    result = run_progen2_sample(inputs=inputs, config=config)
    validate_output(result)

    assert len(result.sequences) == 6

    # Each sequence should start with its corresponding prompt
    for i, (prompt, seq) in enumerate(zip(prompts, result.sequences)):
        prompt_aa = prompt[1:]
        assert seq.startswith(prompt_aa), f"Sequence {i} should start with '{prompt_aa}'"
        assert len(seq) <= 50


# ============================================================================
# Scoring Tests
# ============================================================================

@pytest.mark.uses_gpu
def test_progen2_score_inference():
    """Test ProGen2Model.score() with comprehensive value validation."""
    from bio_programming.tools.language_models.progen2 import ProGen2Model

    sequences = ["MKTLVIVTGA", "EVQLVESGGGLVQ"]
    progen2_model = ProGen2Model(model_checkpoint="progen2-small")

    result = progen2_model.score(sequences=sequences, device="cuda", verbose=False)

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

        # Logits shape: (seq_len + 1, vocab_size) - +1 for start token
        expected_seq_len = len(seq) + 1  # sequence + start token
        assert logits.shape[0] == expected_seq_len, f"Logits seq_len should be {expected_seq_len}, got {logits.shape[0]}"
        assert logits.shape[1] > 20, f"Vocab size should be > 20 (amino acids), got {logits.shape[1]}"


@pytest.mark.uses_gpu
def test_progen2_score_tool():
    """Test the progen2 scoring tool with comprehensive validation."""
    from bio_programming.tools.language_models.progen2 import (
        ProGen2ScoringConfig,
        ProGen2ScoringInput,
        run_progen2_score,
    )

    sequences = ["MKTLVIVTGASGAGK", "EVQLVESGGGLVQPG"]
    inputs = ProGen2ScoringInput(sequences=sequences)
    config = ProGen2ScoringConfig(model_checkpoint="progen2-small", verbose=False)

    result = run_progen2_score(inputs=inputs, config=config)
    validate_output(result)

    assert result.tool_id == "progen2-score"
    assert result.alphabet == "ACDEFGHIKLMNPQRSTVWY"
    assert len(result.scores) == 2

    for seq, score in zip(sequences, result.scores):
        # Validate metric types and ranges
        assert isinstance(score.log_likelihood, float)
        assert isinstance(score.avg_log_likelihood, float)
        assert isinstance(score.perplexity, float)

        assert score.log_likelihood < 0
        assert score.log_likelihood <= score.avg_log_likelihood <= 0
        assert score.perplexity >= 1.0

        # Logits should have correct shape
        assert score.logits is not None
        assert score.logits.shape[0] == len(seq) + 1  # +1 for start token


@pytest.mark.uses_gpu
def test_progen2_score_different_sequences():
    """Test that model produces different perplexities for different sequences."""
    from bio_programming.tools.language_models.progen2 import (
        ProGen2ScoringConfig,
        ProGen2ScoringInput,
        run_progen2_score,
    )

    # Different sequences should produce different perplexities
    seq1 = "EVQLVESGGGLVQPGGSLRLSCAASGFTFS"  # VH domain start
    seq2 = "MKTLVIVTGASGAGKSTIVNLLAQRFGKED"  # Different sequence

    inputs = ProGen2ScoringInput(sequences=[seq1, seq2])
    config = ProGen2ScoringConfig(model_checkpoint="progen2-small", verbose=False)

    result = run_progen2_score(inputs=inputs, config=config)

    ppl1 = result.scores[0].perplexity
    ppl2 = result.scores[1].perplexity

    # Different sequences should have different perplexities (model discriminates)
    assert ppl1 != ppl2, f"Different sequences should have different perplexities: {ppl1} vs {ppl2}"

    # Both should be valid perplexities
    assert ppl1 >= 1.0 and ppl2 >= 1.0


@pytest.mark.uses_gpu
def test_progen2_score_metrics_consistency():
    """Test that scoring metrics are mathematically consistent."""
    from bio_programming.tools.language_models.progen2 import (
        ProGen2ScoringConfig,
        ProGen2ScoringInput,
        run_progen2_score,
    )

    inputs = ProGen2ScoringInput(sequences=["MKTLVIVTGASGAGK"])
    config = ProGen2ScoringConfig(model_checkpoint="progen2-small", verbose=False)

    result = run_progen2_score(inputs=inputs, config=config)
    score = result.scores[0]

    # Verify perplexity = exp(-avg_log_likelihood)
    expected_perplexity = np.exp(-score.avg_log_likelihood)
    np.testing.assert_allclose(score.perplexity, expected_perplexity, rtol=1e-5)

    # Average should equal total / count
    # For a 15-char sequence with start token, we predict 15 positions
    seq_len = 15  # MKTLVIVTGASGAGK
    expected_avg = score.log_likelihood / seq_len
    np.testing.assert_allclose(score.avg_log_likelihood, expected_avg, rtol=1e-4)


@pytest.mark.uses_gpu
def test_progen2_score_auto_prepends_start_token():
    """Test that scoring auto-prepends start token and produces identical results."""
    from bio_programming.tools.language_models.progen2 import (
        ProGen2ScoringConfig,
        ProGen2ScoringInput,
        run_progen2_score,
    )

    config = ProGen2ScoringConfig(model_checkpoint="progen2-small", verbose=False)

    # Without start token
    result_no_token = run_progen2_score(
        inputs=ProGen2ScoringInput(sequences=["MKTLVIVTGA"]),
        config=config,
    )

    # With explicit start token
    result_with_token = run_progen2_score(
        inputs=ProGen2ScoringInput(sequences=["1MKTLVIVTGA"]),
        config=config,
    )

    # All metrics should be identical
    np.testing.assert_allclose(
        result_no_token.scores[0].log_likelihood,
        result_with_token.scores[0].log_likelihood,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        result_no_token.scores[0].perplexity,
        result_with_token.scores[0].perplexity,
        rtol=1e-5,
    )


def test_progen2_score_input_validation():
    """Test ProGen2ScoringInput validation and normalization."""
    from bio_programming.tools.language_models.progen2 import ProGen2ScoringInput

    # Empty sequences should fail
    with pytest.raises(ValueError, match="sequences must not be empty"):
        ProGen2ScoringInput(sequences=[])

    # String input should be normalized to list
    input_str = ProGen2ScoringInput(sequences="MKTL")
    assert input_str.sequences == ["MKTL"]

    # List should be preserved
    input_list = ProGen2ScoringInput(sequences=["MKTL", "EVQL"])
    assert input_list.sequences == ["MKTL", "EVQL"]


# ============================================================================
# Batched Scoring Tests
# ============================================================================

@pytest.mark.uses_gpu
def test_progen2_score_batched_inference():
    """Test batched scoring with direct model call."""
    from bio_programming.tools.language_models.progen2 import ProGen2Model

    sequences = ["MKTL", "EVQLVESGGS", "AAAACCCC", "TTTTGGGG"]
    progen2_model = ProGen2Model(model_checkpoint="progen2-small")

    result = progen2_model.score(sequences=sequences, device="cuda", batch_size=2, verbose=False)

    assert len(result["metrics"]) == 4

    for i, (seq, metrics, logits) in enumerate(zip(sequences, result["metrics"], result["logits"])):
        # Each sequence should have valid metrics
        assert metrics["log_likelihood"] < 0
        assert metrics["perplexity"] >= 1.0

        # Logits should match sequence length
        assert logits.shape[0] == len(seq) + 1, f"Sequence {i}: wrong logits length"


@pytest.mark.uses_gpu
def test_progen2_score_batched_tool():
    """Test batched scoring with tool layer (run_progen2_score)."""
    from bio_programming.tools.language_models.progen2 import (
        ProGen2ScoringConfig,
        ProGen2ScoringInput,
        run_progen2_score,
    )

    sequences = ["MKTLVIVTGA", "EVQLVESGGS", "AAAACCCCGG", "TTTTGGGGCC", "LLLLLLLLLL", "VVVVVVVVVV"]
    inputs = ProGen2ScoringInput(sequences=sequences)
    config = ProGen2ScoringConfig(
        model_checkpoint="progen2-small",
        batch_size=2,
        verbose=False,
    )

    result = run_progen2_score(inputs=inputs, config=config)
    validate_output(result)

    assert len(result.scores) == 6

    for seq, score in zip(sequences, result.scores):
        assert score.log_likelihood < 0
        assert score.perplexity >= 1.0
        assert score.logits is not None
        assert score.logits.shape[0] == len(seq) + 1


@pytest.mark.uses_gpu
def test_progen2_score_batch_size_consistency():
    """Test that different batch_sizes produce identical results."""
    from bio_programming.tools.language_models.progen2 import (
        ProGen2ScoringConfig,
        ProGen2ScoringInput,
        run_progen2_score,
    )

    sequences = ["MKTLVIVTGA", "EVQLVESGGS", "AAAACCCCGG", "TTTTGGGGCC"]
    inputs = ProGen2ScoringInput(sequences=sequences)

    results = {
        bs: run_progen2_score(
            inputs=inputs,
            config=ProGen2ScoringConfig(model_checkpoint="progen2-small", batch_size=bs, verbose=False),
        )
        for bs in [1, 2, None]
    }

    # All batch sizes should produce identical perplexities
    for i in range(4):
        perplexities = [results[bs].scores[i].perplexity for bs in [1, 2, None]]
        log_likelihoods = [results[bs].scores[i].log_likelihood for bs in [1, 2, None]]

        np.testing.assert_allclose(perplexities[0], perplexities[1], rtol=1e-5)
        np.testing.assert_allclose(perplexities[0], perplexities[2], rtol=1e-5)
        np.testing.assert_allclose(log_likelihoods[0], log_likelihoods[1], rtol=1e-5)
        np.testing.assert_allclose(log_likelihoods[0], log_likelihoods[2], rtol=1e-5)


@pytest.mark.uses_gpu
def test_progen2_score_variable_length_sequences():
    """Test scoring sequences of different lengths produces correct logits shapes."""
    from bio_programming.tools.language_models.progen2 import (
        ProGen2ScoringConfig,
        ProGen2ScoringInput,
        run_progen2_score,
    )

    sequences = ["MK", "MKTL", "MKTLVIVT", "MKTLVIVTGASG"]
    inputs = ProGen2ScoringInput(sequences=sequences)
    config = ProGen2ScoringConfig(model_checkpoint="progen2-small", batch_size=2, verbose=False)

    result = run_progen2_score(inputs=inputs, config=config)

    for (seq, score) in zip(sequences, result.scores):
        # Logits should match each sequence's length
        expected_len = len(seq) + 1  # +1 for start token
        assert score.logits.shape[0] == expected_len, (
            f"Sequence '{seq}' (len {len(seq)}): expected logits len {expected_len}, got {score.logits.shape[0]}"
        )

        # Metrics should be valid
        assert score.perplexity >= 1.0
        assert score.log_likelihood < 0
