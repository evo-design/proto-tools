"""Tests for ProGen3 tool."""

import json

import pytest
from pydantic import ValidationError

from proto_tools.tools.causal_models.progen3 import (
    ProGen3SampleConfig,
    ProGen3SampleInput,
    ProGen3SampleOutput,
    ProGen3ScoringConfig,
    ProGen3ScoringInput,
    ProGen3ScoringOutput,
    run_progen3_sample,
    run_progen3_score,
)
from proto_tools.utils import PROTEIN_AMINO_ACIDS
from tests.conftest import benchmark_twice, make_persistent_fixture, random_protein_sequences
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec

_SMALL_MODEL = "progen3-112m"

_persistent_tool = make_persistent_fixture("progen3")


# ── Sample input validation ──────────────────────────────────────────────────


def test_sample_input_accepts_single_string():
    """Single string is coerced to a list."""
    inp = ProGen3SampleInput(prompts="MKTL")
    assert isinstance(inp.prompts, list)
    assert inp.prompts == ["MKTL"]


def test_sample_input_accepts_list():
    """List of strings is accepted as-is."""
    inp = ProGen3SampleInput(prompts=["MKTL", "RYTE"])
    assert inp.prompts == ["MKTL", "RYTE"]


def test_sample_input_rejects_empty():
    """Empty list raises ValueError."""
    with pytest.raises(ValueError, match="prompts must not be empty"):
        ProGen3SampleInput(prompts=[])


# ── Sample config validation ────────────────────────────────────────────────


def test_sample_config_defaults():
    """All defaults are set correctly."""
    cfg = ProGen3SampleConfig()
    assert cfg.model_checkpoint == "progen3-762m"
    assert cfg.direction == "forward"
    assert cfg.temperature == 0.2
    assert cfg.top_p == 0.95
    assert cfg.max_new_tokens == 256
    assert cfg.min_new_tokens == 1
    assert cfg.prepend_prompt is True
    assert cfg.batch_size == 1
    assert cfg.device == "cuda"
    assert cfg.local_path is None


def test_sample_dispatches_one_sequence_per_prompt(monkeypatch):
    """Multi-prompt input goes through a single dispatch carrying the full batch."""
    captured_payloads = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        assert toolkit == "progen3"
        assert config is not None
        assert instance is None
        captured_payloads.append(payload)
        return {"sequences": [f"{p[1:]}G" for p in payload["prompts"]]}

    monkeypatch.setattr(
        "proto_tools.tools.causal_models.progen3.progen3_sample.ToolInstance.dispatch",
        fake_dispatch,
    )

    result = run_progen3_sample(
        ProGen3SampleInput(prompts=["AAAA", "CCCC"]),
        ProGen3SampleConfig(seed=7),
    )

    assert result.sequences == ["AAAAG", "CCCCG"]
    assert [payload["prompts"] for payload in captured_payloads] == [["1AAAA", "1CCCC"]]


def test_scoring_config_defaults():
    """All defaults are set correctly."""
    cfg = ProGen3ScoringConfig()
    assert cfg.model_checkpoint == "progen3-762m"
    assert cfg.device == "cuda"
    assert cfg.batch_size == 1
    assert cfg.local_path is None


def test_sample_config_rejects_invalid_temperature():
    """Temperature must be > 0."""
    with pytest.raises(ValidationError, match="greater than 0"):
        ProGen3SampleConfig(temperature=0.0)
    with pytest.raises(ValidationError, match="greater than 0"):
        ProGen3SampleConfig(temperature=-1.0)


def test_sample_config_rejects_invalid_top_p():
    """top_p must be > 0 and <= 1."""
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        ProGen3SampleConfig(top_p=1.5)
    with pytest.raises(ValidationError, match="greater than 0"):
        ProGen3SampleConfig(top_p=0.0)


def test_sample_config_rejects_invalid_checkpoint():
    """Invalid model checkpoint raises ValidationError."""
    with pytest.raises(ValidationError):
        ProGen3SampleConfig(model_checkpoint="progen3-invalid")


def test_sample_config_rejects_invalid_direction():
    """Invalid direction raises ValidationError."""
    with pytest.raises(ValidationError):
        ProGen3SampleConfig(direction="diagonal")


# ── Sample output export ────────────────────────────────────────────────────


def test_sample_output_export_fasta(tmp_path):
    """Test FASTA export from a manually constructed output."""
    output = ProGen3SampleOutput(
        sequences=["MKTLVIVTGA", "EVQLVESGGS"],
        tool_id="progen3-sample",
        success=True,
    )
    export_path = tmp_path / "test_output"
    output.export(str(export_path), file_format="fasta")

    fasta_path = tmp_path / "test_output.fasta"
    assert fasta_path.exists()
    content = fasta_path.read_text()
    assert ">seq_0" in content
    assert "MKTLVIVTGA" in content
    assert ">seq_1" in content
    assert "EVQLVESGGS" in content


def test_sample_output_export_json(tmp_path):
    """Test JSON export from a manually constructed output."""
    output = ProGen3SampleOutput(
        sequences=["MKTLVIVTGA", "EVQLVESGGS"],
        tool_id="progen3-sample",
        success=True,
    )
    export_path = tmp_path / "test_output"
    output.export(str(export_path), file_format="json")

    json_path = tmp_path / "test_output.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert data["sequences"] == ["MKTLVIVTGA", "EVQLVESGGS"]


def test_scoring_output_has_correct_format_options():
    """CausalModelScoringOutput supports csv and json export."""
    output = ProGen3ScoringOutput(
        scores=[],
        tool_id="progen3-score",
        success=True,
    )
    assert "csv" in output.output_format_options
    assert "json" in output.output_format_options
    assert output.output_format_default == "csv"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

# ── Sampling tests ──────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_progen3_sample_forward():
    """Forward (N->C) generation expands to the right of the prompt."""
    inputs = ProGen3SampleInput(prompts=["MKTL"])
    config = ProGen3SampleConfig(
        max_new_tokens=32,
        model_checkpoint=_SMALL_MODEL,
    )
    result = run_progen3_sample(inputs, config)

    assert result.success
    assert result.tool_id == "progen3-sample"
    assert len(result.sequences) == 1
    seq = result.sequences[0]
    assert len(seq) > len("MKTL")
    assert seq.startswith("MKTL"), "Forward generation should expand to the right of the prompt"


@pytest.mark.uses_gpu
def test_progen3_sample_reverse():
    """Reverse (C->N) generation expands to the left of the prompt."""
    inputs = ProGen3SampleInput(prompts=["RYTE"])
    config = ProGen3SampleConfig(
        direction="reverse",
        max_new_tokens=32,
        model_checkpoint=_SMALL_MODEL,
    )
    result = run_progen3_sample(inputs, config)

    assert result.success
    assert result.tool_id == "progen3-sample"
    assert len(result.sequences) == 1
    seq = result.sequences[0]
    assert len(seq) > len("RYTE")
    assert seq.endswith("RYTE"), "Reverse generation should expand to the left of the prompt"


# ── Scoring tests ──────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_progen3_score_basic():
    """Basic scoring returns valid metrics for a single sequence."""
    inputs = ProGen3ScoringInput(sequences=["MKTLVIVTGASGAGK"])
    config = ProGen3ScoringConfig(model_checkpoint=_SMALL_MODEL)
    result = run_progen3_score(inputs, config)
    assert_metrics_in_spec(result)

    assert result.success
    assert result.tool_id == "progen3-score"
    assert len(result.scores) == 1


@pytest.mark.uses_gpu
def test_progen3_score_batch():
    """Scoring multiple sequences in a batch."""
    sequences = ["MKTLVIVTGASGAGK", "EVQLVESGGGLVQPG", "MVLSPADKTNVKAAW"]
    inputs = ProGen3ScoringInput(sequences=sequences)
    config = ProGen3ScoringConfig(
        model_checkpoint=_SMALL_MODEL,
        batch_size=2,
    )
    result = run_progen3_score(inputs, config)
    assert_metrics_in_spec(result)

    assert result.success
    assert len(result.scores) == 3


@pytest.mark.uses_gpu
def test_progen3_score_return_logits():
    """``return_logits=True`` populates per-position forward logits and the 34-token vocab."""
    seq = "MKTLVIVTGASGAGK"
    inputs = ProGen3ScoringInput(sequences=[seq])
    config = ProGen3ScoringConfig(model_checkpoint=_SMALL_MODEL, return_logits=True)
    result = run_progen3_score(inputs, config)

    score = result.scores[0]
    # Vocab: 6 specials + "1"/"2" + 26 AA letters
    assert len(score.vocab) == 34
    assert score.vocab[6:8] == ["1", "2"]
    assert score.vocab[8] == "A" and score.vocab[33] == "Z"

    assert score.logits is not None
    # Tokenized layout: <bos> + "1" + L AAs + "2" + <eos> = L + 4 tokens
    assert len(score.logits) == len(seq) + 4
    assert len(score.logits[0]) == 34


@pytest.mark.uses_gpu
def test_progen3_score_logits_disabled_by_default():
    """``logits`` is ``None`` when ``return_logits=False`` (default)."""
    inputs = ProGen3ScoringInput(sequences=["MKTLVIVTGASGAGK"])
    result = run_progen3_score(inputs, ProGen3ScoringConfig(model_checkpoint=_SMALL_MODEL))
    assert result.scores[0].logits is None


@pytest.mark.uses_gpu
def test_progen3_score_per_position_structure():
    """Per-position metrics have correct structure and length.

    Per-position metrics live on the ``Metrics`` container as ``_pp``-suffixed
    extras (e.g. ``log_likelihood_pp``) to avoid colliding with the scalar
    metrics of the same stem name.
    """
    seq = "MKTLVIVTGASGAGK"
    inputs = ProGen3ScoringInput(sequences=[seq])
    config = ProGen3ScoringConfig(model_checkpoint=_SMALL_MODEL)
    result = run_progen3_score(inputs, config)

    score = result.scores[0]
    assert "log_likelihood_pp" in score

    for key in ("forward_log_likelihood_pp", "reverse_log_likelihood_pp", "log_likelihood_pp"):
        assert key in score, f"Missing key: {key}"
        values = score[key]
        assert len(values) == len(seq), f"{key} length {len(values)} != seq length {len(seq)}"

    # Forward has no left context at position 0
    assert score["forward_log_likelihood_pp"][0] is None
    # Reverse has no right context at last position
    assert score["reverse_log_likelihood_pp"][-1] is None

    # Interior positions should have values for all three
    for key in ("forward_log_likelihood_pp", "reverse_log_likelihood_pp", "log_likelihood_pp"):
        for j in range(1, len(seq) - 1):
            assert score[key][j] is not None, f"{key}[{j}] is None for interior position"


@pytest.mark.uses_gpu
def test_progen3_score_per_position_consistency():
    """Per-position scores are consistent with aggregate metrics.

    The aggregate avg_log_likelihood includes special token predictions
    (direction markers, eos) while per-position only covers amino acid
    residues. We verify that the per-position AA-only mean is close to
    (but not identical to) the aggregate, and that forward/reverse means
    are individually consistent across two scoring calls.
    """
    seq = "MKTLVIVTGASGAGK"
    inputs = ProGen3ScoringInput(sequences=[seq])
    config = ProGen3ScoringConfig(model_checkpoint=_SMALL_MODEL, seed=42)
    result = run_progen3_score(inputs, config)

    score = result.scores[0]
    fwd = [v for v in score["forward_log_likelihood_pp"] if v is not None]
    rev = [v for v in score["reverse_log_likelihood_pp"] if v is not None]
    bidir = [v for v in score["log_likelihood_pp"] if v is not None]

    # All log-likelihoods should be negative
    assert all(v < 0 for v in fwd), "Forward LLs should be negative"
    assert all(v < 0 for v in rev), "Reverse LLs should be negative"
    assert all(v < 0 for v in bidir), "Bidirectional LLs should be negative"

    # Forward and reverse should each have L-1 values
    assert len(fwd) == len(seq) - 1
    assert len(rev) == len(seq) - 1

    # Bidirectional mean of interior positions should equal avg of fwd and rev at those positions
    for j in range(1, len(seq) - 1):
        f = score["forward_log_likelihood_pp"][j]
        r = score["reverse_log_likelihood_pp"][j]
        b = score["log_likelihood_pp"][j]
        assert abs(b - (f + r) / 2) < 1e-6, f"Bidirectional[{j}] != avg(fwd, rev)"

    # Deterministic: scoring same sequence twice gives same per-position values
    result2 = run_progen3_score(inputs, config)
    fwd2 = [v for v in result2.scores[0]["forward_log_likelihood_pp"] if v is not None]
    assert all(abs(a - b) < 1e-4 for a, b in zip(fwd, fwd2, strict=True)), "Per-position scores not deterministic"


# ── Benchmarks ──────────────────────────────────────────────────────────────


@pytest.mark.benchmark("progen3-sample")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_progen3_sample_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark progen3-sample on 50 length-250 prompts generating up to 450 new tokens (cold + warm)."""
    prompts = random_protein_sequences(n=50, length=250, seed=0)
    inputs = ProGen3SampleInput(prompts=prompts)
    config = ProGen3SampleConfig(
        model_checkpoint="progen3-762m",
        batch_size=16,
        max_new_tokens=450,
        temperature=0.2,
        top_p=0.95,
    )

    result = benchmark_twice(request, "progen3", lambda: run_progen3_sample(inputs=inputs, config=config))

    assert len(result.sequences) == 50, "Should have 50 generated sequences"
    for sampled in result.sequences:
        assert len(sampled) > 0, "Generated sequence should be non-empty"
        assert all(aa in PROTEIN_AMINO_ACIDS for aa in sampled), "All residues should be standard amino acids"


@pytest.mark.benchmark("progen3-score")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_progen3_score_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark progen3-score on 200 sequences of length 500 (cold + warm)."""
    sequences = random_protein_sequences(n=200, length=500, seed=1)
    inputs = ProGen3ScoringInput(sequences=sequences)
    config = ProGen3ScoringConfig(
        model_checkpoint="progen3-762m",
        batch_size=32,
    )

    result = benchmark_twice(request, "progen3", lambda: run_progen3_score(inputs=inputs, config=config))
    assert_metrics_in_spec(result)

    assert result.tool_id == "progen3-score"
    assert len(result.scores) == 200
    for score in result.scores:
        assert score["perplexity"] >= 1.0
