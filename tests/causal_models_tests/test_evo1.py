"""Tests for Evo1."""

import json

import numpy as np
import pytest

from bio_programming_tools.tools.causal_models.evo1 import (
    Evo1SampleConfig,
    Evo1SampleInput,
    Evo1SampleOutput,
    Evo1ScoringConfig,
    Evo1ScoringInput,
    Evo1ScoringOutput,
    run_evo1_sample,
    run_evo1_score,
)
from bio_programming_tools.tools.causal_models.shared_data_models import SequenceScores
from tests.conftest import make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import validate_output


_persistent_tool = make_persistent_fixture("evo1")


def _make_mock_scoring_output():
    """Create a mock scoring output for testing."""
    scores = [
        SequenceScores(
            metrics={
                "log_likelihood": -10.5,
                "avg_log_likelihood": -1.05,
                "perplexity": 2.86,
            },
        ),
        SequenceScores(
            metrics={
                "log_likelihood": -12.3,
                "avg_log_likelihood": -1.23,
                "perplexity": 3.42,
            },
        ),
    ]
    return Evo1ScoringOutput(scores=scores)


# ── Sample input validation ───────────────────────────────────────────────────

def test_evo1_sample_input_normalizes_single_string():
    inp = Evo1SampleInput(prompts="ATCGATCG")
    assert isinstance(inp.prompts, list)
    assert len(inp.prompts) == 1
    assert inp.prompts[0] == "ATCGATCG"


def test_evo1_sample_input_preserves_list():
    inp = Evo1SampleInput(prompts=["ATCG", "GCTA"])
    assert len(inp.prompts) == 2


def test_evo1_sample_input_rejects_empty():
    with pytest.raises(ValueError, match="prompts must not be empty"):
        Evo1SampleInput(prompts=[])


# ── Sample config validation ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    "config_kwargs,match",
    [
        ({"temperature": 0.0}, "greater than 0"),
        ({"top_p": 1.5}, "less than or equal to 1"),
        ({"num_tokens": 0}, "greater than or equal to 1"),
        ({"top_k": 0}, "greater than or equal to 1"),
    ],
)
def test_evo1_sample_config_rejects_invalid_values(config_kwargs, match):
    with pytest.raises(ValueError, match=match):
        Evo1SampleConfig(**config_kwargs)


# ── Sample output export ─────────────────────────────────────────────────────

def test_evo1_sample_export_fasta(tmp_path):
    output = Evo1SampleOutput(
        sequences=["ATCGATCG", "GCTAGCTA"], scores=[-1.0, -1.5]
    )
    output.export(name="test", export_path=tmp_path, file_format="fasta")
    fasta_file = tmp_path / "test.fasta"
    assert fasta_file.exists()
    content = fasta_file.read_text()
    assert ">seq_0" in content
    assert ">seq_1" in content
    assert "ATCGATCG" in content
    assert "GCTAGCTA" in content


def test_evo1_sample_export_json(tmp_path):
    output = Evo1SampleOutput(sequences=["ATCGATCG"], scores=[-1.0])
    output.export(name="test", export_path=tmp_path, file_format="json")
    json_file = tmp_path / "test.json"
    assert json_file.exists()
    data = json.loads(json_file.read_text())
    assert "sequences" in data
    assert data["sequences"] == ["ATCGATCG"]


def test_evo1_sample_export_txt(tmp_path):
    output = Evo1SampleOutput(
        sequences=["ATCGATCG", "GCTAGCTA"], scores=[-1.0, -1.5]
    )
    output.export(name="test", export_path=tmp_path, file_format="txt")
    txt_file = tmp_path / "test.txt"
    assert txt_file.exists()
    lines = txt_file.read_text().strip().split("\n")
    assert len(lines) == 2
    assert lines[0] == "ATCGATCG"
    assert lines[1] == "GCTAGCTA"


# ── Scoring input validation ─────────────────────────────────────────────────

def test_evo1_scoring_input_normalizes_single_string():
    inp = Evo1ScoringInput(sequences="ATCGATCG")
    assert isinstance(inp.sequences, list)
    assert len(inp.sequences) == 1
    assert inp.sequences[0] == "ATCGATCG"


def test_evo1_scoring_input_preserves_list():
    inp = Evo1ScoringInput(sequences=["ATCG", "GCTA"])
    assert len(inp.sequences) == 2


def test_evo1_scoring_input_rejects_empty():
    with pytest.raises(ValueError, match="sequences must not be empty"):
        Evo1ScoringInput(sequences=[])


# ── Scoring output export ────────────────────────────────────────────────────

def test_evo1_scoring_export_csv(tmp_path):
    output = _make_mock_scoring_output()
    output.export(name="test", export_path=tmp_path, file_format="csv")
    csv_file = tmp_path / "test.csv"
    assert csv_file.exists()
    content = csv_file.read_text()
    assert "log_likelihood" in content
    assert "avg_log_likelihood" in content
    assert "perplexity" in content


def test_evo1_scoring_export_json(tmp_path):
    output = _make_mock_scoring_output()
    output.export(name="test", export_path=tmp_path, file_format="json")
    json_file = tmp_path / "test.json"
    assert json_file.exists()
    data = json.loads(json_file.read_text())
    assert isinstance(data, list)
    assert len(data) == 2
    assert "log_likelihood" in data[0]


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.include_in_env_report(category="causal_models")
@pytest.mark.uses_gpu
def test_evo1_sample_tool():
    """Test the evo1 sampling tool end-to-end: inference, output structure, and export."""
    prompts = ["ATCG", "GCTA"]
    inputs = Evo1SampleInput(prompts=prompts)
    config = Evo1SampleConfig(
        model_name="evo-1-8k-base",
        num_tokens=50,
        temperature=1.0,
        top_k=4,
        verbose=False,
    )

    result = run_evo1_sample(inputs=inputs, config=config)
    validate_output(result)

    assert result.tool_id == "evo1-sample"
    assert result.metadata["model_name"] == "evo-1-8k-base"
    assert result.metadata["num_tokens"] == 50
    assert len(result.sequences) == 2
    assert result.scores is not None
    assert len(result.scores) == 2

    valid_chars = set("ACGTacgt")
    for seq in result.sequences:
        assert isinstance(seq, str) and len(seq) > 0
        invalid = set(seq) - valid_chars
        assert not invalid, f"Non-DNA characters in output: {invalid}"

    for score in result.scores:
        assert float(score) < 0


@pytest.mark.uses_gpu
def test_evo1_sample_batched():
    """Test batched sampling with batch_size=2 on 6 prompts (3 batches)."""
    prompts = ["ATCG", "GCTA", "AAAA", "GGGG", "CCCC", "TTTT"]
    inputs = Evo1SampleInput(prompts=prompts)
    config = Evo1SampleConfig(
        num_tokens=50,
        temperature=1.0,
        batch_size=2,
        verbose=False,
    )

    result = run_evo1_sample(inputs=inputs, config=config)
    validate_output(result)

    assert len(result.sequences) == 6
    assert result.scores is not None
    assert len(result.scores) == 6


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_evo1_sample_prepend_prompt():
    """Test that prepend_prompt controls whether prompt is included."""
    prompt = "ATCGATCG"

    result_with = run_evo1_sample(
        inputs=Evo1SampleInput(prompts=[prompt]),
        config=Evo1SampleConfig(num_tokens=50, prepend_prompt=True, verbose=False),
    )
    assert result_with.sequences[0].startswith(prompt)

    result_without = run_evo1_sample(
        inputs=Evo1SampleInput(prompts=[prompt]),
        config=Evo1SampleConfig(num_tokens=50, prepend_prompt=False, verbose=False),
    )
    assert len(result_without.sequences[0]) > 0


@pytest.mark.uses_gpu
def test_evo1_score_tool():
    """Test evo1 scoring: output structure, metrics, batching, and consistency."""
    sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA"]
    inputs = Evo1ScoringInput(sequences=sequences)
    config = Evo1ScoringConfig(
        model_name="evo-1-8k-base",
        return_logits=True,
        verbose=False,
    )

    result = run_evo1_score(inputs=inputs, config=config)
    validate_output(result)

    assert result.tool_id == "evo1-score"
    assert len(result.scores) == 2

    for score in result.scores:
        assert isinstance(score.log_likelihood, float)
        assert isinstance(score.avg_log_likelihood, float)
        assert isinstance(score.perplexity, float)
        assert score.log_likelihood < 0
        assert score.perplexity >= 1.0

    score = result.scores[0]
    expected_perplexity = np.exp(-score.avg_log_likelihood)
    np.testing.assert_allclose(score.perplexity, expected_perplexity, rtol=1e-5)

    assert score.logits is not None
    logits_arr = np.array(score.logits)
    assert logits_arr.shape == (len(sequences[0]), 512)
    assert score.vocab is not None
    assert len(score.vocab) == 512
    assert score.vocab[65] == "A"
    assert score.vocab[67] == "C"
    assert score.vocab[71] == "G"
    assert score.vocab[84] == "T"


@pytest.mark.uses_gpu
def test_evo1_score_batched():
    """Test batched scoring with batch_size=2 on 6 sequences."""
    sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA", "AAAACCCCGGGG",
                  "TTTTGGGGCCCC", "CCCCAAAATTTT", "GGGGTTTTAAAA"]
    inputs = Evo1ScoringInput(sequences=sequences)
    config = Evo1ScoringConfig(
        model_name="evo-1-8k-base",
        batch_size=2,
        return_logits=True,
        verbose=False,
    )

    result = run_evo1_score(inputs=inputs, config=config)
    validate_output(result)

    assert len(result.scores) == 6

    for score in result.scores:
        assert isinstance(score.log_likelihood, float)
        assert score.log_likelihood < 0
        assert score.perplexity >= 1.0
        assert score.logits is not None
