"""
test_evo1.py

Tests the Evo1 implementation.
"""

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
)
from bio_programming_tools.tools.causal_models.shared_data_models import SequenceScores
from tests.tool_infra_tests.test_export_functionality import validate_output


def _get_in_process_tools():
    """Lazy import of in-process tools (requires torch)."""
    from bio_programming_tools.tools.causal_models.evo1._in_process_mode import (
        run_evo1_sample as run_evo1_sample_ip,
        run_evo1_score as run_evo1_score_ip,
    )
    from bio_programming_tools.tools.causal_models.evo1._in_process_mode import (
        Evo1ScoringConfig as Evo1ScoringConfigIP,
    )

    return run_evo1_sample_ip, run_evo1_score_ip, Evo1ScoringConfigIP

# ============================================================================
# Input Validation Tests (no GPU needed)
# ============================================================================


class TestEvo1SampleInput:
    """Tests for Evo1SampleInput validation and normalization."""

    def test_single_string_normalization(self):
        """Single string should be normalized to list."""
        inp = Evo1SampleInput(prompts="ATCGATCG")
        assert isinstance(inp.prompts, list)
        assert len(inp.prompts) == 1
        assert inp.prompts[0] == "ATCGATCG"

    def test_list_input_preserved(self):
        """List input should be preserved as-is."""
        inp = Evo1SampleInput(prompts=["ATCG", "GCTA"])
        assert len(inp.prompts) == 2

    def test_empty_prompts_raises(self):
        """Empty prompts should raise ValueError."""
        with pytest.raises(ValueError, match="prompts must not be empty"):
            Evo1SampleInput(prompts=[])


# ============================================================================
# Config Validation Tests (no GPU needed)
# ============================================================================


class TestEvo1SampleConfig:
    """Tests for Evo1SampleConfig validation."""

    def test_default_batch_size(self):
        """Default batch_size should be 128."""
        config = Evo1SampleConfig()
        assert config.batch_size == 128

    def test_default_values(self):
        """Verify default config values."""
        config = Evo1SampleConfig()
        assert config.model_name == "evo-1-8k-base"
        assert config.top_k == 4
        assert config.temperature == 1.0
        assert config.top_p == 1.0
        assert config.num_tokens == 100
        assert config.prepend_prompt is False
        assert config.device == "cuda"

    def test_custom_batch_size(self):
        """Custom batch_size should be accepted."""
        config = Evo1SampleConfig(batch_size=32)
        assert config.batch_size == 32

    def test_batch_size_none(self):
        """batch_size=None should be accepted."""
        config = Evo1SampleConfig(batch_size=None)
        assert config.batch_size is None

    @pytest.mark.parametrize(
        "config_kwargs",
        [
            {"temperature": 0.0},
            {"top_p": 1.5},
            {"num_tokens": 0},
            {"top_k": 0},
        ],
    )
    def test_invalid_config_raises(self, config_kwargs):
        """Invalid config values should raise ValueError."""
        with pytest.raises(ValueError):
            Evo1SampleConfig(**config_kwargs)


# ============================================================================
# Sampling Tests (GPU required)
# ============================================================================


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

    # Output structure
    assert result.tool_id == "evo1-sample"
    assert result.metadata["model_name"] == "evo-1-8k-base"
    assert result.metadata["num_tokens"] == 50
    assert len(result.sequences) == 2
    assert result.scores is not None
    assert len(result.scores) == 2

    # Sequences are valid DNA
    valid_chars = set("ATCGNatcgn")
    for seq in result.sequences:
        assert isinstance(seq, str) and len(seq) > 0
        assert set(seq).issubset(valid_chars)

    # Scores are negative log-probabilities
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


# ============================================================================
# Export Tests (no GPU needed)
# ============================================================================


class TestEvo1SampleOutput:
    """Tests for Evo1SampleOutput export functionality."""

    def test_output_format_options(self):
        """Verify supported export formats."""
        output = Evo1SampleOutput(sequences=["ATCG"], scores=[-1.0])
        assert "fasta" in output.output_format_options
        assert "txt" in output.output_format_options
        assert "json" in output.output_format_options

    def test_default_format_is_fasta(self):
        """Default export format should be fasta."""
        output = Evo1SampleOutput(sequences=["ATCG"], scores=[-1.0])
        assert output.output_format_default == "fasta"

    def test_export_fasta(self, tmp_path):
        """Test FASTA export creates valid file."""
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

    def test_export_json(self, tmp_path):
        """Test JSON export creates valid file."""
        import json

        output = Evo1SampleOutput(sequences=["ATCGATCG"], scores=[-1.0])
        output.export(name="test", export_path=tmp_path, file_format="json")
        json_file = tmp_path / "test.json"
        assert json_file.exists()
        data = json.loads(json_file.read_text())
        assert "sequences" in data
        assert data["sequences"] == ["ATCGATCG"]

    def test_export_txt(self, tmp_path):
        """Test TXT export creates valid file."""
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


# ============================================================================
# Scoring Input Validation Tests (no GPU needed)
# ============================================================================


class TestEvo1ScoringInput:
    """Tests for Evo1ScoringInput validation and normalization."""

    def test_single_string_normalization(self):
        """Single string should be normalized to list."""
        inp = Evo1ScoringInput(sequences="ATCGATCG")
        assert isinstance(inp.sequences, list)
        assert len(inp.sequences) == 1
        assert inp.sequences[0] == "ATCGATCG"

    def test_list_input_preserved(self):
        """List input should be preserved as-is."""
        inp = Evo1ScoringInput(sequences=["ATCG", "GCTA"])
        assert len(inp.sequences) == 2

    def test_empty_sequences_raises(self):
        """Empty sequences should raise ValueError."""
        with pytest.raises(ValueError, match="sequences must not be empty"):
            Evo1ScoringInput(sequences=[])


# ============================================================================
# Scoring Config Validation Tests (no GPU needed)
# ============================================================================


class TestEvo1ScoringConfig:
    """Tests for Evo1ScoringConfig validation."""

    def test_default_values(self):
        """Verify default config values."""
        config = Evo1ScoringConfig()
        assert config.model_name == "evo-1-8k-base"
        assert config.batch_size is None
        assert config.device == "cuda"
        assert config.return_logits is False
        assert config.verbose is False

    def test_custom_batch_size(self):
        """Custom batch_size should be accepted."""
        config = Evo1ScoringConfig(batch_size=4)
        assert config.batch_size == 4

    def test_custom_model_name(self):
        """Custom model_name should be accepted."""
        config = Evo1ScoringConfig(model_name="evo-1-8k-crispr")
        assert config.model_name == "evo-1-8k-crispr"


# ============================================================================
# Scoring Output / Export Tests (no GPU needed)
# ============================================================================


class TestEvo1ScoringOutput:
    """Tests for Evo1ScoringOutput export functionality."""

    def _make_mock_output(self):
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

    def test_output_format_options(self):
        """Verify supported export formats."""
        output = self._make_mock_output()
        assert "csv" in output.output_format_options
        assert "json" in output.output_format_options

    def test_default_format_is_csv(self):
        """Default export format should be csv."""
        output = self._make_mock_output()
        assert output.output_format_default == "csv"

    def test_export_csv(self, tmp_path):
        """Test CSV export creates valid file."""
        output = self._make_mock_output()
        output.export(name="test", export_path=tmp_path, file_format="csv")
        csv_file = tmp_path / "test.csv"
        assert csv_file.exists()
        content = csv_file.read_text()
        assert "log_likelihood" in content
        assert "avg_log_likelihood" in content
        assert "perplexity" in content

    def test_export_json(self, tmp_path):
        """Test JSON export creates valid file."""
        output = self._make_mock_output()
        output.export(name="test", export_path=tmp_path, file_format="json")
        json_file = tmp_path / "test.json"
        assert json_file.exists()
        data = json.loads(json_file.read_text())
        assert isinstance(data, list)
        assert len(data) == 2
        assert "log_likelihood" in data[0]


# ============================================================================
# Scoring Tests (GPU required)
# ============================================================================


@pytest.mark.uses_gpu
def test_evo1_score_tool():
    """Test the evo1 scoring tool end-to-end (in-process mode)."""
    _, run_evo1_score_ip, Evo1ScoringConfigIP = _get_in_process_tools()

    sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA"]
    inputs = Evo1ScoringInput(sequences=sequences)
    config = Evo1ScoringConfigIP(
        model_name="evo-1-8k-base",
        verbose=False,
    )

    result = run_evo1_score_ip(inputs=inputs, config=config)
    validate_output(result)

    # Check output structure
    assert result.tool_id == "evo1-score-in-process"
    assert len(result.scores) == 2

    # Check scoring metrics for each sequence
    for i, score in enumerate(result.scores):
        assert "log_likelihood" in score.metrics
        assert "avg_log_likelihood" in score.metrics
        assert "perplexity" in score.metrics

        assert isinstance(score.log_likelihood, float)
        assert isinstance(score.avg_log_likelihood, float)
        assert isinstance(score.perplexity, float)

        # Log-likelihood should be negative
        assert score.log_likelihood < 0
        # Perplexity should be positive
        assert score.perplexity > 0

    # Logits should be None by default
    for score in result.scores:
        assert score.logits is None


@pytest.mark.uses_gpu
def test_evo1_score_with_logits():
    """Test scoring with return_logits=True returns correct shapes."""
    _, run_evo1_score_ip, Evo1ScoringConfigIP = _get_in_process_tools()

    sequences = ["ATCGATCGATCG"]
    inputs = Evo1ScoringInput(sequences=sequences)
    config = Evo1ScoringConfigIP(
        model_name="evo-1-8k-base",
        verbose=False,
        return_logits=True,
    )

    result = run_evo1_score_ip(inputs=inputs, config=config)
    validate_output(result)

    score = result.scores[0]
    assert score.logits is not None
    logits_arr = np.array(score.logits)
    # Logits shape: (seq_len, vocab_size=512)
    assert logits_arr.shape[0] == len(sequences[0])
    assert logits_arr.shape[1] == 512

    # Vocab should have 512 entries
    assert score.vocab is not None
    assert len(score.vocab) == 512
    # Verify DNA nucleotide positions (ASCII values)
    assert score.vocab[65] == "A"
    assert score.vocab[67] == "C"
    assert score.vocab[71] == "G"
    assert score.vocab[84] == "T"


@pytest.mark.uses_gpu
def test_evo1_score_batched():
    """Test batched scoring with batch_size=2 on 6 sequences."""
    _, run_evo1_score_ip, Evo1ScoringConfigIP = _get_in_process_tools()

    sequences = [
        "ATCGATCG",
        "GCTAGCTA",
        "AAAACCCC",
        "GGGGTTTT",
        "ACGTACGT",
        "TGCATGCA",
    ]
    inputs = Evo1ScoringInput(sequences=sequences)
    config = Evo1ScoringConfigIP(
        model_name="evo-1-8k-base",
        batch_size=2,
        verbose=False,
    )

    result = run_evo1_score_ip(inputs=inputs, config=config)
    validate_output(result)

    assert len(result.scores) == 6
    for score in result.scores:
        assert score.perplexity > 0
        assert score.log_likelihood < 0


@pytest.mark.uses_gpu
def test_evo1_score_metrics_consistency():
    """Test that scoring metrics are mathematically consistent."""
    _, run_evo1_score_ip, Evo1ScoringConfigIP = _get_in_process_tools()

    inputs = Evo1ScoringInput(sequences=["ATCGATCGATCG"])
    config = Evo1ScoringConfigIP(
        model_name="evo-1-8k-base",
        verbose=False,
    )

    result = run_evo1_score_ip(inputs=inputs, config=config)
    score = result.scores[0]

    # Verify perplexity = exp(-avg_log_likelihood)
    expected_perplexity = np.exp(-score.avg_log_likelihood)
    np.testing.assert_allclose(
        score.perplexity,
        expected_perplexity,
        rtol=1e-5,
        err_msg="Perplexity should equal exp(-avg_log_likelihood)",
    )


@pytest.mark.uses_gpu
def test_evo1_score_model_reuse():
    """Test that sample and score share the cached model instance."""
    run_evo1_sample_ip, run_evo1_score_ip, Evo1ScoringConfigIP = (
        _get_in_process_tools()
    )
    from bio_programming_tools.tools.causal_models.evo1._in_process_mode.evo1_cache import (
        get_cached_evo1_model,
    )
    from bio_programming_tools.tools.causal_models.evo1._in_process_mode.evo1_sample import (
        Evo1SampleConfig as Evo1SampleConfigIP,
    )

    model_name = "evo-1-8k-base"
    device = "cuda"

    # First call: sample (loads model)
    sample_inputs = Evo1SampleInput(prompts=["ATCG"])
    sample_config = Evo1SampleConfigIP(
        model_name=model_name,
        num_tokens=10,
        device=device,
        verbose=False,
    )
    run_evo1_sample_ip(inputs=sample_inputs, config=sample_config)

    # Get cached model reference
    model_after_sample = get_cached_evo1_model(
        model_name=model_name, device=device
    )

    # Second call: score (should reuse model)
    score_inputs = Evo1ScoringInput(sequences=["ATCG"])
    score_config = Evo1ScoringConfigIP(
        model_name=model_name,
        device=device,
        verbose=False,
    )
    run_evo1_score_ip(inputs=score_inputs, config=score_config)

    # Get cached model reference again
    model_after_score = get_cached_evo1_model(
        model_name=model_name, device=device
    )

    # Should be the exact same object (no duplicate loading)
    assert model_after_sample is model_after_score
