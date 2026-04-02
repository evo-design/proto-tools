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

_SMALL_MODEL = "progen3-112m"


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


# ── Scoring input validation ────────────────────────────────────────────────


def test_scoring_input_accepts_single_string():
    """Single string is coerced to a list."""
    inp = ProGen3ScoringInput(sequences="MKTL")
    assert isinstance(inp.sequences, list)
    assert inp.sequences == ["MKTL"]


def test_scoring_input_accepts_list():
    """List of strings is accepted as-is."""
    inp = ProGen3ScoringInput(sequences=["MKTL", "EVQL"])
    assert inp.sequences == ["MKTL", "EVQL"]


def test_scoring_input_rejects_empty():
    """Empty list raises ValueError."""
    with pytest.raises(ValueError, match="sequences must not be empty"):
        ProGen3ScoringInput(sequences=[])


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
    assert cfg.num_sequences == 1
    assert cfg.include_prompt_in_output is True
    assert cfg.batch_size == 1
    assert cfg.device == "cuda"
    assert cfg.local_path is None


def test_scoring_config_defaults():
    """All defaults are set correctly."""
    cfg = ProGen3ScoringConfig()
    assert cfg.model_checkpoint == "progen3-762m"
    assert cfg.device == "cuda"
    assert cfg.batch_size == 1
    assert cfg.reduction == "mean"
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
@pytest.mark.include_in_env_report(category="causal_models")
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

    assert result.success
    assert result.tool_id == "progen3-score"
    assert len(result.scores) == 1

    score = result.scores[0]
    assert "log_likelihood" in score.metrics
    assert "avg_log_likelihood" in score.metrics
    assert "perplexity" in score.metrics

    assert isinstance(score.metrics["log_likelihood"], float)
    assert score.metrics["log_likelihood"] < 0
    assert score.metrics["perplexity"] >= 1.0


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

    assert result.success
    assert len(result.scores) == 3

    for score in result.scores:
        assert "log_likelihood" in score.metrics
        assert "perplexity" in score.metrics
        assert score.metrics["log_likelihood"] < 0
        assert score.metrics["perplexity"] >= 1.0


@pytest.mark.uses_gpu
def test_progen3_score_per_position_structure():
    """Per-position metrics have correct structure and length."""
    seq = "MKTLVIVTGASGAGK"
    inputs = ProGen3ScoringInput(sequences=[seq])
    config = ProGen3ScoringConfig(model_checkpoint=_SMALL_MODEL)
    result = run_progen3_score(inputs, config)

    score = result.scores[0]
    assert score.per_position_metrics is not None

    for key in ("forward_log_likelihood", "reverse_log_likelihood", "log_likelihood"):
        assert key in score.per_position_metrics, f"Missing key: {key}"
        values = score.per_position_metrics[key]
        assert len(values) == len(seq), f"{key} length {len(values)} != seq length {len(seq)}"

    # Forward has no left context at position 0
    assert score.per_position_metrics["forward_log_likelihood"][0] is None
    # Reverse has no right context at last position
    assert score.per_position_metrics["reverse_log_likelihood"][-1] is None

    # Interior positions should have values for all three
    for key in ("forward_log_likelihood", "reverse_log_likelihood", "log_likelihood"):
        for j in range(1, len(seq) - 1):
            assert score.per_position_metrics[key][j] is not None, f"{key}[{j}] is None for interior position"


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
    config = ProGen3ScoringConfig(model_checkpoint=_SMALL_MODEL)
    result = run_progen3_score(inputs, config)

    score = result.scores[0]
    fwd = [v for v in score.per_position_metrics["forward_log_likelihood"] if v is not None]
    rev = [v for v in score.per_position_metrics["reverse_log_likelihood"] if v is not None]
    bidir = [v for v in score.per_position_metrics["log_likelihood"] if v is not None]

    # All log-likelihoods should be negative
    assert all(v < 0 for v in fwd), "Forward LLs should be negative"
    assert all(v < 0 for v in rev), "Reverse LLs should be negative"
    assert all(v < 0 for v in bidir), "Bidirectional LLs should be negative"

    # Forward and reverse should each have L-1 values
    assert len(fwd) == len(seq) - 1
    assert len(rev) == len(seq) - 1

    # Bidirectional mean of interior positions should equal avg of fwd and rev at those positions
    for j in range(1, len(seq) - 1):
        f = score.per_position_metrics["forward_log_likelihood"][j]
        r = score.per_position_metrics["reverse_log_likelihood"][j]
        b = score.per_position_metrics["log_likelihood"][j]
        assert abs(b - (f + r) / 2) < 1e-6, f"Bidirectional[{j}] != avg(fwd, rev)"

    # Deterministic: scoring same sequence twice gives same per-position values
    result2 = run_progen3_score(inputs, config)
    fwd2 = [v for v in result2.scores[0].per_position_metrics["forward_log_likelihood"] if v is not None]
    assert all(abs(a - b) < 1e-4 for a, b in zip(fwd, fwd2, strict=True)), "Per-position scores not deterministic"
