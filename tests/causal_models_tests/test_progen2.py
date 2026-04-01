"""tests/causal_models_tests/test_progen2.py.

Tests for ProGen2.
"""

import numpy as np
import pytest

from proto_tools.tools.causal_models.progen2 import (
    ProGen2SampleConfig,
    ProGen2SampleInput,
    ProGen2ScoringConfig,
    ProGen2ScoringInput,
    run_progen2_sample,
    run_progen2_score,
)
from proto_tools.utils import PROTEIN_AMINO_ACIDS
from tests.conftest import make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import validate_output

_persistent_tool = make_persistent_fixture("progen2")


# ── Sample input/config validation ────────────────────────────────────────────


def test_progen2_sample_input_normalizes_single_string():
    inp = ProGen2SampleInput(prompts="MKTLV")
    assert isinstance(inp.prompts, list)
    assert inp.prompts == ["MKTLV"]


@pytest.mark.parametrize(
    "input_kwargs,match",
    [
        ({"prompts": []}, "prompts must not be empty"),
    ],
)
def test_progen2_sample_input_validation(input_kwargs, match):
    with pytest.raises(ValueError, match=match):
        ProGen2SampleInput(**input_kwargs)


@pytest.mark.parametrize(
    "config_kwargs,match",
    [
        ({"temperature": 0.0}, "greater than 0"),
        ({"top_p": 1.5}, "less than or equal to 1"),
        ({"max_length": 0}, "greater than or equal to 1"),
    ],
)
def test_progen2_sample_config_validation(config_kwargs, match):
    with pytest.raises(ValueError, match=match):
        ProGen2SampleConfig(**config_kwargs)


# ── Scoring input validation ─────────────────────────────────────────────────


def test_progen2_score_input_validation():
    """Test ProGen2ScoringInput validation and normalization."""
    with pytest.raises(ValueError, match="sequences must not be empty"):
        ProGen2ScoringInput(sequences=[])

    input_str = ProGen2ScoringInput(sequences="MKTL")
    assert input_str.sequences == ["MKTL"]

    input_list = ProGen2ScoringInput(sequences=["MKTL", "EVQL"])
    assert input_list.sequences == ["MKTL", "EVQL"]


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

# ── Sampling tests ────────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_progen2_sample_basic():
    """Test basic ProGen2 sampling via run_progen2_sample."""
    prompts = ["1MKTLV", "1EVQLVE"]
    inputs = ProGen2SampleInput(prompts=prompts)
    config = ProGen2SampleConfig(
        model_checkpoint="progen2-small",
        max_length=100,
        temperature=0.2,
        top_p=0.95,
        top_k=0,
        truncate_at_stop=True,
        strip_special_tokens=True,
        prepend_prompt=True,
        verbose=False,
    )

    result = run_progen2_sample(inputs=inputs, config=config)
    validate_output(result)

    assert result.tool_id == "progen2-sample"
    assert result.metadata["model_checkpoint"] == "progen2-small"
    assert result.metadata["max_length"] == 100
    assert result.metadata["temperature"] == 0.2

    assert len(result.sequences) == 2

    valid_chars = set(PROTEIN_AMINO_ACIDS)
    for i, (prompt, seq) in enumerate(zip(prompts, result.sequences, strict=False)):
        assert isinstance(seq, str) and len(seq) > 0
        assert set(seq.upper()).issubset(valid_chars), f"Invalid chars: {set(seq) - valid_chars}"

        prompt_aa = prompt[1:]  # Remove '1' start token
        assert seq.startswith(prompt_aa), f"Sequence {i} should start with prompt '{prompt_aa}'"

        assert len(seq) <= 100, f"Sequence {i} exceeds max_length"


@pytest.mark.uses_gpu
@pytest.mark.parametrize(
    "prompt,expected_prefix",
    [
        ("1MKTLV", "MKTLV"),
        ("MKTLV", "MKTLV"),
        ("<|pf03668|>1MEVVIVTGMSGAGK", "MEVVIVTGMSGAGK"),
    ],
)
def test_progen2_sample_prompt_handling(prompt, expected_prefix):
    """Test progen2 sampling with various prompt formats."""
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
    inputs = ProGen2SampleInput(prompts="1M")
    config = ProGen2SampleConfig(
        model_checkpoint="progen2-small",
        max_length=20,
        temperature=0.2,
        verbose=False,
    )

    result = run_progen2_sample(inputs=inputs, config=config)

    assert len(result.sequences[0]) <= 20


@pytest.mark.uses_gpu
def test_progen2_sample_special_token_stripping():
    """Test special token stripping behavior."""
    result_stripped = run_progen2_sample(
        inputs=ProGen2SampleInput(prompts="1MKTLV"),
        config=ProGen2SampleConfig(
            model_checkpoint="progen2-small",
            max_length=50,
            strip_special_tokens=True,
            verbose=False,
        ),
    )

    result_unstripped = run_progen2_sample(
        inputs=ProGen2SampleInput(prompts="1MKTLV"),
        config=ProGen2SampleConfig(
            model_checkpoint="progen2-small",
            max_length=50,
            strip_special_tokens=False,
            verbose=False,
        ),
    )

    assert not result_stripped.sequences[0].startswith("1"), "Stripped sequence should not start with '1'"
    assert set(result_stripped.sequences[0]).issubset(set(PROTEIN_AMINO_ACIDS)), (
        "Stripped should only contain amino acids"
    )

    assert result_unstripped.sequences[0].startswith("1"), "Unstripped sequence should start with '1'"


# ── Batched sampling tests ────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_progen2_sample_batched():
    """Test batched sampling via run_progen2_sample with batch_size."""
    prompts = ["1MKTLV", "1EVQLV", "1AAAAA", "1GGGGG"]
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

    assert len(result.sequences) == 4

    for i, (prompt, seq) in enumerate(zip(prompts, result.sequences, strict=False)):
        prompt_aa = prompt[1:]
        assert seq.startswith(prompt_aa), f"Sequence {i} should start with '{prompt_aa}', got '{seq[:10]}'"


@pytest.mark.uses_gpu
def test_progen2_sample_batched_many():
    """Test batched sampling with more prompts than batch_size."""
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

    for i, (prompt, seq) in enumerate(zip(prompts, result.sequences, strict=False)):
        prompt_aa = prompt[1:]
        assert seq.startswith(prompt_aa), f"Sequence {i} should start with '{prompt_aa}'"
        assert len(seq) <= 50


# ── Scoring tests ─────────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_progen2_score_tool():
    """Test the progen2 scoring tool with comprehensive validation."""
    sequences = ["MKTLVIVTGASGAGK", "EVQLVESGGGLVQPG"]
    inputs = ProGen2ScoringInput(sequences=sequences)
    config = ProGen2ScoringConfig(model_checkpoint="progen2-small", verbose=False, return_logits=True)

    result = run_progen2_score(inputs=inputs, config=config)
    validate_output(result)

    assert result.tool_id == "progen2-score"
    assert len(result.vocab) == 30
    assert result.vocab[5] == "A" and result.vocab[7] == "C"
    assert result.vocab[22] == "S" and result.vocab[28] == "Y"
    assert len(result.scores) == 2

    for seq, score in zip(sequences, result.scores, strict=False):
        assert isinstance(score.log_likelihood, float)
        assert isinstance(score.avg_log_likelihood, float)
        assert isinstance(score.perplexity, float)

        assert score.log_likelihood < 0
        assert score.log_likelihood <= score.avg_log_likelihood <= 0
        assert score.perplexity >= 1.0

        assert score.logits is not None
        assert len(score.logits) == len(seq) + 1  # +1 for start token
        assert len(score.logits[0]) == 30

    for score in result.scores:
        expected_ppl = np.exp(-score.avg_log_likelihood)
        np.testing.assert_allclose(score.perplexity, expected_ppl, rtol=1e-5)


@pytest.mark.uses_gpu
def test_progen2_score_different_sequences():
    """Test that model produces different perplexities for different sequences."""
    seq1 = "EVQLVESGGGLVQPGGSLRLSCAASGFTFS"
    seq2 = "MKTLVIVTGASGAGKSTIVNLLAQRFGKED"

    inputs = ProGen2ScoringInput(sequences=[seq1, seq2])
    config = ProGen2ScoringConfig(model_checkpoint="progen2-small", verbose=False, return_logits=True)

    result = run_progen2_score(inputs=inputs, config=config)

    ppl1 = result.scores[0].perplexity
    ppl2 = result.scores[1].perplexity

    assert ppl1 != ppl2, f"Different sequences should have different perplexities: {ppl1} vs {ppl2}"
    assert ppl1 >= 1.0 and ppl2 >= 1.0


@pytest.mark.uses_gpu
def test_progen2_score_metrics_consistency():
    """Test that scoring metrics are mathematically consistent."""
    _seq = "MKTLVIVTGASGAGK"
    inputs = ProGen2ScoringInput(sequences=[_seq])
    config = ProGen2ScoringConfig(model_checkpoint="progen2-small", verbose=False, return_logits=True)

    result = run_progen2_score(inputs=inputs, config=config)
    score = result.scores[0]

    expected_perplexity = np.exp(-score.avg_log_likelihood)
    np.testing.assert_allclose(score.perplexity, expected_perplexity, rtol=1e-5)

    expected_avg = score.log_likelihood / len(_seq)
    np.testing.assert_allclose(score.avg_log_likelihood, expected_avg, rtol=1e-4)


@pytest.mark.uses_gpu
def test_progen2_score_auto_prepends_start_token():
    """Test that scoring auto-prepends start token and produces identical results."""
    config = ProGen2ScoringConfig(model_checkpoint="progen2-small", verbose=False)

    result_no_token = run_progen2_score(
        inputs=ProGen2ScoringInput(sequences=["MKTLVIVTGA"]),
        config=config,
    )

    result_with_token = run_progen2_score(
        inputs=ProGen2ScoringInput(sequences=["1MKTLVIVTGA"]),
        config=config,
    )

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


# ── Batched scoring tests ────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_progen2_score_batched():
    """Test batched scoring via run_progen2_score with batch_size."""
    sequences = ["MKTL", "EVQLVESGGS", "AAAACCCC", "TTTTGGGG", "LLLLLLLLLL", "VVVVVVVVVV"]
    inputs = ProGen2ScoringInput(sequences=sequences)
    config = ProGen2ScoringConfig(
        model_checkpoint="progen2-small",
        batch_size=2,
        verbose=False,
        return_logits=True,
    )

    result = run_progen2_score(inputs=inputs, config=config)
    validate_output(result)

    assert len(result.scores) == 6

    for i, (seq, score) in enumerate(zip(sequences, result.scores, strict=False)):
        assert score.log_likelihood < 0
        assert score.perplexity >= 1.0

        assert score.logits is not None
        assert len(score.logits) == len(seq) + 1, f"Sequence {i}: wrong logits length"


@pytest.mark.uses_gpu
def test_progen2_score_batch_size_consistency():
    """Test that different batch_sizes produce identical results."""
    sequences = ["MKTLVIVTGA", "EVQLVESGGS", "AAAACCCCGG", "TTTTGGGGCC"]
    inputs = ProGen2ScoringInput(sequences=sequences)

    results = {
        bs: run_progen2_score(
            inputs=inputs,
            config=ProGen2ScoringConfig(model_checkpoint="progen2-small", batch_size=bs, verbose=False),
        )
        for bs in [1, 2, 4]
    }

    for i in range(4):
        perplexities = [results[bs].scores[i].perplexity for bs in [1, 2, 4]]
        log_likelihoods = [results[bs].scores[i].log_likelihood for bs in [1, 2, 4]]

        np.testing.assert_allclose(perplexities[0], perplexities[1], rtol=1e-5)
        np.testing.assert_allclose(perplexities[0], perplexities[2], rtol=1e-5)
        np.testing.assert_allclose(log_likelihoods[0], log_likelihoods[1], rtol=1e-5)
        np.testing.assert_allclose(log_likelihoods[0], log_likelihoods[2], rtol=1e-5)


@pytest.mark.uses_gpu
def test_progen2_score_variable_length_sequences():
    """Test scoring sequences of different lengths produces correct logits shapes."""
    sequences = ["MK", "MKTL", "MKTLVIVT", "MKTLVIVTGASG"]
    inputs = ProGen2ScoringInput(sequences=sequences)
    config = ProGen2ScoringConfig(model_checkpoint="progen2-small", batch_size=2, verbose=False, return_logits=True)

    result = run_progen2_score(inputs=inputs, config=config)

    for seq, score in zip(sequences, result.scores, strict=False):
        expected_len = len(seq) + 1  # +1 for start token
        assert len(score.logits) == expected_len, (
            f"Sequence '{seq}' (len {len(seq)}): expected logits len {expected_len}, got {len(score.logits)}"
        )
        assert len(score.logits[0]) == 30, "ProGen2 vocab size should be 30"

        assert score.perplexity >= 1.0
        assert score.log_likelihood < 0


# ── Logits-specific tests (scoring) ──────────────────────────────────────────


@pytest.mark.uses_gpu
def test_progen2_score_logits_disabled_by_default():
    """Test that logits are None when return_logits=False (default)."""
    sequences = ["MKTLVIVTGA", "EVQLVESGGS"]
    inputs = ProGen2ScoringInput(sequences=sequences)
    config = ProGen2ScoringConfig(
        model_checkpoint="progen2-small",
        verbose=False,
    )

    result = run_progen2_score(inputs=inputs, config=config)
    validate_output(result)

    for score in result.scores:
        assert score.logits is None, "Logits should be None when return_logits=False"


@pytest.mark.uses_gpu
def test_progen2_score_logits_serialization():
    """Test that logits are properly serialized as nested lists."""
    sequences = ["MKTLVIVTGA"]
    inputs = ProGen2ScoringInput(sequences=sequences)
    config = ProGen2ScoringConfig(
        model_checkpoint="progen2-small",
        verbose=False,
        return_logits=True,
    )

    result = run_progen2_score(inputs=inputs, config=config)
    validate_output(result)

    score = result.scores[0]

    assert isinstance(score.logits, list), "Logits should be a list"
    assert len(score.logits) > 0, "Logits list should not be empty"
    assert isinstance(score.logits[0], list), "Logits should be a list of lists"
    assert len(score.logits[0]) == 30, "Inner logits list should have 30 elements (ProGen2 vocab size)"

    for position_logits in score.logits:
        for logit_value in position_logits:
            assert isinstance(logit_value, (int, float)), f"Logit value should be numeric, got {type(logit_value)}"


# ── Logits-specific tests (sampling) ─────────────────────────────────────────


@pytest.mark.uses_gpu
def test_progen2_sample_logits_returned():
    """Test that sampling returns logits when return_logits=True."""
    prompts = ["1MKTLV", "1EVQLVE"]
    inputs = ProGen2SampleInput(prompts=prompts)
    config = ProGen2SampleConfig(
        model_checkpoint="progen2-small",
        max_length=50,
        temperature=0.2,
        top_p=0.95,
        return_logits=True,
        verbose=False,
    )

    result = run_progen2_sample(inputs=inputs, config=config)
    validate_output(result)

    assert result.logits is not None, "Logits should not be None when return_logits=True"
    assert len(result.logits) == 2, f"Should have logits for 2 sequences, got {len(result.logits)}"

    for i, seq_logits in enumerate(result.logits):
        assert len(seq_logits) > 0, f"Logits[{i}] should have at least one position"
        assert len(seq_logits[0]) == 30, f"Logits vocab size should be 30, got {len(seq_logits[0])}"


@pytest.mark.uses_gpu
def test_progen2_sample_logits_not_returned_by_default():
    """Test that sampling does not return logits when return_logits=False (default)."""
    inputs = ProGen2SampleInput(prompts=["1MKTLV"])
    config = ProGen2SampleConfig(
        model_checkpoint="progen2-small",
        max_length=50,
        temperature=0.2,
        return_logits=False,
        verbose=False,
    )

    result = run_progen2_sample(inputs=inputs, config=config)
    validate_output(result)

    assert result.logits is None, f"Logits should be None when return_logits=False, got {type(result.logits)}"
