"""
test_evo1.py

Tests the Evo1 implementation.
"""

import pytest

from bio_programming_tools.tools.causal_models.evo1 import (
    Evo1SampleConfig,
    Evo1SampleInput,
    Evo1SampleOutput,
    run_evo1_sample,
)
from tests.tool_infra_tests.test_export_functionality import validate_output


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

    @pytest.mark.parametrize("config_kwargs", [
        {"temperature": 0.0},
        {"top_p": 1.5},
        {"num_tokens": 0},
        {"top_k": 0},
    ])
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
        output = Evo1SampleOutput(
            sequences=["ATCG"], scores=[-1.0]
        )
        assert "fasta" in output.output_format_options
        assert "txt" in output.output_format_options
        assert "json" in output.output_format_options

    def test_default_format_is_fasta(self):
        """Default export format should be fasta."""
        output = Evo1SampleOutput(
            sequences=["ATCG"], scores=[-1.0]
        )
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

        output = Evo1SampleOutput(
            sequences=["ATCGATCG"], scores=[-1.0]
        )
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
