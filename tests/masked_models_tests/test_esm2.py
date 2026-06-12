"""tests/masked_models_tests/test_esm2.py.

Tests for ESM2.
"""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from proto_tools.tools.masked_models.esm2 import (
    ESM2EmbeddingsConfig,
    ESM2EmbeddingsInput,
    ESM2GradientConfig,
    ESM2GradientInput,
    ESM2SampleConfig,
    ESM2SampleInput,
    ESM2ScoringConfig,
    ESM2ScoringInput,
    run_esm2_embeddings,
    run_esm2_gradient,
    run_esm2_sample,
    run_esm2_score,
)
from proto_tools.tools.masked_models.esm2.esm2_gradient import ESM2GradientOutput
from proto_tools.utils import one_hot_protein_logits
from proto_tools.utils.standalone_helpers_source.standalone_helpers.serialization import AMINO_ACIDS_LIST
from tests.conftest import benchmark_twice, make_persistent_fixture, random_protein_sequences
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
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


def test_esm2_gradient_input_validation():
    inp = ESM2GradientInput(logits=[[0.0] * 20, [1.0] * 20])
    assert inp.logits == [[0.0] * 20, [1.0] * 20]
    assert inp.temperature is None

    inp_with_temp = ESM2GradientInput(logits=[[0.0] * 20], temperature=0.6)
    assert inp_with_temp.temperature == 0.6

    with pytest.raises(ValidationError):
        ESM2GradientInput(logits=[[0.0] * 20], temperature=-0.5)
    with pytest.raises(ValidationError):
        ESM2GradientInput(logits=[[0.0] * 20], temperature=0.0)
    with pytest.raises(ValidationError, match="20 columns"):
        ESM2GradientInput(logits=[[0.0] * 19])


def test_esm2_gradient_dispatch_contract(monkeypatch):
    captured: dict[str, object] = {}

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured["toolkit"] = toolkit
        captured["payload"] = payload
        n = len(payload.get("logits", []))
        return {
            "gradient": [[0.0] * 20] * n,
            "loss": 0.5,
            "metrics": {"log_likelihood": -1.0, "avg_log_likelihood": -0.5, "perplexity": np.exp(0.5)},
            "vocab": AMINO_ACIDS_LIST,
        }

    monkeypatch.setattr(
        "proto_tools.tools.masked_models.esm2.esm2_gradient.ToolInstance.dispatch",
        fake_dispatch,
    )

    run_esm2_gradient(
        ESM2GradientInput(logits=[[0.0] * 20] * 3, temperature=0.6),
        ESM2GradientConfig(model_checkpoint="esm2_t6_8M_UR50D", use_ste=True, batch_size=2, device="cpu"),
    )

    assert captured["toolkit"] == "esm2"
    assert captured["payload"]["operation"] == "compute_gradient"
    assert captured["payload"]["logits"] == [[0.0] * 20] * 3
    assert captured["payload"]["temperature"] == 0.6
    assert captured["payload"]["use_ste"] is True
    assert captured["payload"]["compute_gradient"] is True
    assert captured["payload"]["batch_size"] == 2
    assert captured["payload"]["model_checkpoint"] == "esm2_t6_8M_UR50D"
    assert captured["payload"]["device"] == "cpu"


def test_esm2_gradient_forward_mode_dispatch_contract(monkeypatch):
    captured: dict[str, object] = {}

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured.update(payload=payload)
        return {
            "gradient": None,
            "loss": 0.5,
            "metrics": {"log_likelihood": -1.0, "avg_log_likelihood": -0.5, "perplexity": np.exp(0.5)},
            "vocab": AMINO_ACIDS_LIST,
        }

    monkeypatch.setattr(
        "proto_tools.tools.masked_models.esm2.esm2_gradient.ToolInstance.dispatch",
        fake_dispatch,
    )

    result = run_esm2_gradient(
        ESM2GradientInput(logits=[[0.0] * 20] * 3),
        ESM2GradientConfig(compute_gradient=False, device="cpu"),
    )

    assert captured["payload"]["compute_gradient"] is False
    assert result.gradient is None
    assert result.loss == 0.5
    assert result.metrics["avg_log_likelihood"] == -0.5


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

# ── Gradient tests ───────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_esm2_gradient_single_sequence():
    """Test ESM2 gradient shape, finiteness, and metric consistency."""
    seq_len = 4
    result = run_esm2_gradient(
        ESM2GradientInput(logits=[[0.0] * 20] * seq_len, temperature=0.6),
        ESM2GradientConfig(model_checkpoint="esm2_t6_8M_UR50D", batch_size=2),
    )
    validate_output(result)

    assert result.tool_id == "esm2-gradient"
    assert result.gradient is not None
    assert len(result.gradient) == seq_len
    assert all(len(row) == 20 for row in result.gradient)
    assert all(math.isfinite(v) for row in result.gradient for v in row)
    assert any(v != 0.0 for row in result.gradient for v in row)
    assert result.loss > 0
    assert result.metrics["avg_log_likelihood"] == pytest.approx(-result.loss, rel=1e-6)
    assert result.metrics["perplexity"] == pytest.approx(math.exp(result.loss), rel=1e-6)
    assert result.vocab == AMINO_ACIDS_LIST


@pytest.mark.uses_gpu
def test_esm2_gradient_forward_mode_matches_backward_loss():
    """compute_gradient=False should keep the scalar objective identical."""
    inputs = ESM2GradientInput(logits=[[0.0] * 20] * 4, temperature=0.6)
    backward = run_esm2_gradient(
        inputs,
        ESM2GradientConfig(model_checkpoint="esm2_t6_8M_UR50D", batch_size=2, seed=42),
    )
    forward = run_esm2_gradient(
        inputs,
        ESM2GradientConfig(model_checkpoint="esm2_t6_8M_UR50D", batch_size=2, seed=42, compute_gradient=False),
    )

    assert backward.gradient is not None
    assert forward.gradient is None
    assert forward.loss == pytest.approx(backward.loss, rel=1e-6)
    assert forward.metrics["avg_log_likelihood"] == pytest.approx(backward.metrics["avg_log_likelihood"], rel=1e-6)


@pytest.mark.uses_gpu
def test_esm2_score_and_gradient_agree_on_pll():
    """esm2-score and esm2-gradient forward mode should agree on one-hot discrete input."""
    sequence = "MKTL"
    one_hot = [[10.0 if aa == obs else 0.0 for aa in AMINO_ACIDS_LIST] for obs in sequence]

    score_result = run_esm2_score(
        ESM2ScoringInput(sequences=[sequence]),
        ESM2ScoringConfig(model_checkpoint="esm2_t6_8M_UR50D", batch_size=2),
    )
    grad_result = run_esm2_gradient(
        ESM2GradientInput(logits=one_hot, temperature=0.6),
        ESM2GradientConfig(model_checkpoint="esm2_t6_8M_UR50D", use_ste=True, batch_size=2, compute_gradient=False),
    )

    assert -grad_result.loss == pytest.approx(score_result.scores[0].avg_log_likelihood, rel=1e-5)


@pytest.mark.benchmark("esm2-gradient")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_esm2_gradient_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark esm2-gradient: 20 backward passes over a length-500 logits matrix (cold + warm)."""
    sequence = random_protein_sequences(n=1, length=500, seed=3)[0]
    inputs = ESM2GradientInput(logits=one_hot_protein_logits(sequence, sharpness=2.0), temperature=0.6)
    config = ESM2GradientConfig(model_checkpoint="esm2_t33_650M_UR50D", batch_size=32)

    def run() -> ESM2GradientOutput:
        last: ESM2GradientOutput | None = None
        for _ in range(20):
            last = run_esm2_gradient(inputs, config)
        assert last is not None
        return last

    result = benchmark_twice(request, "esm2", run)
    validate_output(result)

    assert result.tool_id == "esm2-gradient"
    assert result.gradient is not None
    assert len(result.gradient) == 500
    assert all(len(row) == 20 for row in result.gradient)
    assert result.loss > 0


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


@pytest.mark.benchmark("esm2-embedding")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_esm2_embedding_benchmark(request):
    """Benchmark esm2-embedding on 100 sequences of length 300 (cold + warm)."""
    sequences = random_protein_sequences(n=100, length=300, seed=0)
    inputs = ESM2EmbeddingsInput(sequences=sequences)
    config = ESM2EmbeddingsConfig(model_checkpoint="esm2_t33_650M_UR50D", batch_size=32, return_logits=True)

    result = benchmark_twice(request, "esm2", lambda: run_esm2_embeddings(inputs=inputs, config=config))

    assert len(result.results) == 100, "Should have 100 SequenceEmbedding objects"
    assert len(result.results[0].mean_embedding) == 1280, "Embedding dimension should be 1280"
    assert len(result.results[0].attention_mask) == 300, "Attention mask length should be 300"
    assert result.results[0].logits is not None
    assert len(result.results[0].logits) == 300, "Logit sequence length should be 300"
    assert len(result.results[0].logits[0]) == 20, "Logit vocab size should be 20"


# ── Sampling tests ───────────────────────────────────────────────────────────


@pytest.mark.benchmark("esm2-sample")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_esm2_sample_benchmark(request):
    """Benchmark esm2-sample on 50 sequences of length 200 (cold + warm)."""
    sequences = random_protein_sequences(n=50, length=200, seed=1)
    inputs = ESM2SampleInput(sequences=sequences)
    config = ESM2SampleConfig(model_checkpoint="esm2_t33_650M_UR50D", batch_size=16, temperature=1.0)

    result = benchmark_twice(request, "esm2", lambda: run_esm2_sample(inputs=inputs, config=config))

    assert len(result.sequences) == 50, "Should have 50 sampled sequences"
    for sampled in result.sequences:
        assert len(sampled) == 200, "Sampled sequence should preserve length"
        assert "_" not in sampled, "Sampled sequence should have no remaining mask tokens"
        assert all(aa in AMINO_ACIDS_LIST for aa in sampled), "All residues should be standard amino acids"


# ── Scoring tests ─────────────────────────────────────────────────────────────


@pytest.mark.benchmark("esm2-score")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_esm2_score_benchmark(request):
    """Benchmark esm2-score on 100 sequences of length 300 (cold + warm)."""
    sequences = random_protein_sequences(n=100, length=300, seed=2)
    inputs = ESM2ScoringInput(sequences=sequences)
    config = ESM2ScoringConfig(
        model_checkpoint="esm2_t33_650M_UR50D",
        batch_size=32,
        verbose=False,
        return_logits=True,
    )

    result = benchmark_twice(request, "esm2", lambda: run_esm2_score(inputs=inputs, config=config))
    assert_metrics_in_spec(result)

    assert result.tool_id == "esm2-score"
    assert len(result.scores) == 100
    for score in result.scores:
        assert score.logits is not None
        assert len(score.logits) == 300
        assert len(score.logits[0]) == 20


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
    assert_metrics_in_spec(result)

    assert result.tool_id == "esm2-score"
    assert len(result.scores) == 2
    assert isinstance(result.vocab, list), f"Vocab should be a list, got {type(result.vocab)}"
    assert len(result.vocab) == 20, f"ESM2 vocab should have 20 tokens, got {len(result.vocab)}"

    for seq, score in zip(sequences, result.scores, strict=False):
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
    assert_metrics_in_spec(result)

    ppl1 = result.scores[0].perplexity
    ppl2 = result.scores[1].perplexity

    assert ppl1 != ppl2, f"Different sequences should have different perplexities: {ppl1} vs {ppl2}"


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
    assert_metrics_in_spec(result)

    assert len(result.scores) == 4

    for seq, score in zip(sequences, result.scores, strict=False):
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
    assert_metrics_in_spec(result)

    for seq, score in zip(sequences, result.scores, strict=False):
        assert isinstance(score.logits, list), f"Logits should be a list, got {type(score.logits)}"
        assert len(score.logits) == len(seq), (
            f"Sequence '{seq}' (len {len(seq)}): logits len should be {len(seq)}, got {len(score.logits)}"
        )
        assert len(score.logits[0]) == 20, f"Logits vocab size should be 20, got {len(score.logits[0])}"


@pytest.mark.uses_gpu
def test_esm2_score_single_sequence():
    """Test esm2 scoring with a single sequence (string input)."""
    inputs = ESM2ScoringInput(sequences="MKTAYIAKQRQISFVKSHFS")
    config = ESM2ScoringConfig(model_checkpoint="esm2_t33_650M_UR50D", verbose=False, return_logits=True)

    result = run_esm2_score(inputs=inputs, config=config)
    validate_output(result)
    assert_metrics_in_spec(result)

    assert len(result.scores) == 1
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
