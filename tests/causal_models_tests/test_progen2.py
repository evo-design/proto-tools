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
from tests.conftest import benchmark_twice, make_persistent_fixture, random_protein_sequences
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
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
        ({"max_new_tokens": 0}, "greater than or equal to 1"),
    ],
)
def test_progen2_sample_config_validation(config_kwargs, match):
    with pytest.raises(ValueError, match=match):
        ProGen2SampleConfig(**config_kwargs)


def test_progen2_sample_dispatches_one_sequence_per_prompt(monkeypatch):
    """Multi-prompt input goes through a single dispatch carrying the full batch."""
    captured_payloads = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        assert toolkit == "progen2"
        assert config is not None
        assert instance is None
        captured_payloads.append(payload)
        return {"sequences": [f"{p}_sample" for p in payload["prompts"]], "logits": None}

    monkeypatch.setattr(
        "proto_tools.tools.causal_models.progen2.progen2_sample.ToolInstance.dispatch",
        fake_dispatch,
    )

    result = run_progen2_sample(
        ProGen2SampleInput(prompts=["1AAAA", "1CCCC"]),
        ProGen2SampleConfig(seed=7),
    )

    assert result.sequences == ["1AAAA_sample", "1CCCC_sample"]
    assert [payload["prompts"] for payload in captured_payloads] == [["1AAAA", "1CCCC"]]


@pytest.mark.parametrize("terminal_token_id", [3, 4], ids=["start-token", "end-token"])
def test_progen2_standalone_sample_strips_generated_terminal_tokens(terminal_token_id):
    """Sampling strips ProGen2 terminal sentinels from either output edge."""
    from proto_tools.tools.causal_models.progen2.standalone.inference import (
        PROGEN2_TERMINAL_TOKENS,
        PROGEN2_VOCAB,
        ProGen2Model,
    )

    decoded = "".join(PROGEN2_VOCAB[token_id] for token_id in [3, 5, 7, terminal_token_id])
    sequence = ProGen2Model()._truncate_at_terminals(decoded).strip(PROGEN2_TERMINAL_TOKENS)
    assert sequence == "AC"


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
        max_new_tokens=100,
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
    assert result.metadata["max_new_tokens"] == 100
    assert result.metadata["temperature"] == 0.2

    assert len(result.sequences) == 2

    valid_chars = set(PROTEIN_AMINO_ACIDS)
    for i, (prompt, seq) in enumerate(zip(prompts, result.sequences, strict=False)):
        assert isinstance(seq, str) and len(seq) > 0
        assert set(seq.upper()).issubset(valid_chars), f"Invalid chars: {set(seq) - valid_chars}"

        prompt_aa = prompt[1:]  # Remove '1' start token
        assert seq.startswith(prompt_aa), f"Sequence {i} should start with prompt '{prompt_aa}'"

        assert len(seq) <= len(prompt_aa) + 100, f"Sequence {i} exceeds prompt + max_new_tokens"


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
        max_new_tokens=100,
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
def test_progen2_sample_max_new_tokens_respected():
    """Test that max_new_tokens caps the newly generated portion of the output."""
    inputs = ProGen2SampleInput(prompts="1M")
    config = ProGen2SampleConfig(
        model_checkpoint="progen2-small",
        max_new_tokens=20,
        temperature=0.2,
        verbose=False,
    )

    result = run_progen2_sample(inputs=inputs, config=config)

    # "1M" decodes to "M" after the start sentinel is stripped; output ≤ prompt + new tokens.
    assert len(result.sequences[0]) <= 1 + 20


@pytest.mark.uses_gpu
def test_progen2_sample_special_token_stripping():
    """Test special token stripping behavior."""
    result_stripped = run_progen2_sample(
        inputs=ProGen2SampleInput(prompts="1MKTLV"),
        config=ProGen2SampleConfig(
            model_checkpoint="progen2-small",
            max_new_tokens=50,
            strip_special_tokens=True,
            verbose=False,
        ),
    )

    result_unstripped = run_progen2_sample(
        inputs=ProGen2SampleInput(prompts="1MKTLV"),
        config=ProGen2SampleConfig(
            model_checkpoint="progen2-small",
            max_new_tokens=50,
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
        max_new_tokens=50,
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
        max_new_tokens=50,
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
        assert len(seq) <= len(prompt_aa) + 50


# ── Scoring tests ─────────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_progen2_score_tool():
    """Test the progen2 scoring tool with comprehensive validation."""
    sequences = ["MKTLVIVTGASGAGK", "EVQLVESGGGLVQPG"]
    inputs = ProGen2ScoringInput(sequences=sequences)
    config = ProGen2ScoringConfig(model_checkpoint="progen2-small", verbose=False, return_logits=True)

    result = run_progen2_score(inputs=inputs, config=config)
    validate_output(result)
    assert_metrics_in_spec(result)

    assert result.tool_id == "progen2-score"
    assert len(result.vocab) == 30
    assert result.vocab[5] == "A" and result.vocab[7] == "C"
    assert result.vocab[22] == "S" and result.vocab[28] == "Y"
    assert len(result.scores) == 2

    for seq, score in zip(sequences, result.scores, strict=False):
        assert score.log_likelihood <= score.avg_log_likelihood <= 0

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
    assert_metrics_in_spec(result)

    ppl1 = result.scores[0].perplexity
    ppl2 = result.scores[1].perplexity

    assert ppl1 != ppl2, f"Different sequences should have different perplexities: {ppl1} vs {ppl2}"


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
    assert_metrics_in_spec(result)

    assert len(result.scores) == 6

    for i, (seq, score) in enumerate(zip(sequences, result.scores, strict=False)):
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
    assert_metrics_in_spec(result)

    for seq, score in zip(sequences, result.scores, strict=False):
        expected_len = len(seq) + 1  # +1 for start token
        assert len(score.logits) == expected_len, (
            f"Sequence '{seq}' (len {len(seq)}): expected logits len {expected_len}, got {len(score.logits)}"
        )
        assert len(score.logits[0]) == 30, "ProGen2 vocab size should be 30"


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
        max_new_tokens=50,
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
        max_new_tokens=50,
        temperature=0.2,
        return_logits=False,
        verbose=False,
    )

    result = run_progen2_sample(inputs=inputs, config=config)
    validate_output(result)

    assert result.logits is None, f"Logits should be None when return_logits=False, got {type(result.logits)}"


# ── Benchmarks ──────────────────────────────────────────────────────────────


@pytest.mark.benchmark("progen2-sample")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_progen2_sample_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark progen2-sample on 50 length-250 prompts generating up to 450 new tokens (cold + warm)."""
    prompts = random_protein_sequences(n=50, length=250, seed=0)
    inputs = ProGen2SampleInput(prompts=prompts)
    config = ProGen2SampleConfig(
        model_checkpoint="progen2-medium",
        batch_size=16,
        max_new_tokens=450,
        temperature=0.2,
        top_p=0.95,
        verbose=False,
    )

    result = benchmark_twice(request, "progen2", lambda: run_progen2_sample(inputs=inputs, config=config))

    assert len(result.sequences) == 50, "Should have 50 generated sequences"
    for sampled in result.sequences:
        assert len(sampled) > 0, "Generated sequence should be non-empty"
        assert all(aa in PROTEIN_AMINO_ACIDS for aa in sampled), "All residues should be standard amino acids"


@pytest.mark.benchmark("progen2-score")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_progen2_score_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark progen2-score on 200 sequences of length 500 (cold + warm)."""
    sequences = random_protein_sequences(n=200, length=500, seed=1)
    inputs = ProGen2ScoringInput(sequences=sequences)
    config = ProGen2ScoringConfig(
        model_checkpoint="progen2-medium",
        batch_size=32,
        return_logits=False,
        verbose=False,
    )

    result = benchmark_twice(request, "progen2", lambda: run_progen2_score(inputs=inputs, config=config))
    assert_metrics_in_spec(result)

    assert result.tool_id == "progen2-score"
    assert len(result.scores) == 200
    for score in result.scores:
        assert score["perplexity"] >= 1.0
