"""Tests for AbLang antibody language model tools."""

import math

import pytest
from pydantic import ValidationError

from proto_tools.tools.masked_models.ablang import (
    AbLangEmbeddingsConfig,
    AbLangEmbeddingsInput,
    AbLangGerminalGradientConfig,
    AbLangGerminalGradientInput,
    AbLangSampleConfig,
    AbLangSampleInput,
    AbLangScoringConfig,
    AbLangScoringInput,
    run_ablang_embeddings,
    run_ablang_germinal_gradient,
    run_ablang_sample,
    run_ablang_score,
)
from proto_tools.utils.sequence import PROTEIN_AMINO_ACIDS
from tests.conftest import make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import validate_output

_VALID_AAS = set(PROTEIN_AMINO_ACIDS)
_CANONICAL_VOCAB = list(PROTEIN_AMINO_ACIDS)

_persistent_tool = make_persistent_fixture("ablang")

# Full antibody sequences (heavy + light with constant regions)
HEAVY_FULL = "AVKLVQAGGGVVQPGRSLRLSCIASGFTFSNYGMHWVRQAPGKGLEWVAVIWYNGSRTYYGDSVKGRFTISRDNSKRTLYMQMNSLRTEDTAVYYCARDPDILTAFSFDYWGQGVLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC"
LIGHT_FULL = "SYELTQPPSVSVSPGQTARITCSANALPNQYAYWYQQKPGRAPVMVIYKDTQRPSGIPQRFSSSTSGTTVTLTISGVQAEDEADYYCQAWDNSASIFGGGTKLTVLGQPKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPIKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTECS"
PAIRED_SEQ = f"{HEAVY_FULL}|{LIGHT_FULL}"

# Variable domain sequences for ablang1 models (max_position_embeddings=160)
VH_SEQ = "AVKLVQAGGGVVQPGRSLRLSCIASGFTFSNYGMHWVRQAPGKGLEWVAVIWYNGSRTYYGDSVKGRFTISRDNSKRTLYMQMNSLRTEDTAVYYCARDPDILTAFSFDYWGQGVLVTVSS"
VL_SEQ = "SYELTQPPSVSVSPGQTARITCSANALPNQYAYWYQQKPGRAPVMVIYKDTQRPSGIPQRFSSSTSGTTVTLTISGVQAEDEADYYCQAWDNSASIFGGGTKLTVLG"


# ── Input validation ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("input_cls", "sequence"),
    [
        (AbLangEmbeddingsInput, "EVQLVESGGGLVQPGG"),
        (AbLangScoringInput, "EVQLVESGGGLVQPGG"),
        (AbLangSampleInput, "EVQL_ESGGGLVQPGG"),
    ],
)
def test_ablang_input_normalizes_single_string(input_cls, sequence):
    inp = input_cls(sequences=sequence)
    assert isinstance(inp.sequences, list)
    assert inp.sequences == [sequence]


@pytest.mark.parametrize("input_cls", [AbLangEmbeddingsInput, AbLangScoringInput, AbLangSampleInput])
def test_ablang_empty_input_raises(input_cls):
    with pytest.raises(ValueError, match="sequences must not be empty"):
        input_cls(sequences=[])


def test_ablang_germinal_gradient_input_requires_20_aa_columns():
    logits = [[0.0] * 20, [1.0] * 20]
    inp = AbLangGerminalGradientInput(logits=logits, temperature=0.5)
    assert inp.logits == logits

    with pytest.raises(ValidationError, match="20 columns"):
        AbLangGerminalGradientInput(logits=[[0.0] * 19], temperature=1.0)


def test_ablang_germinal_gradient_config_validates_single_chain_variable_fragment_layout():
    config = AbLangGerminalGradientConfig(
        use_single_chain_variable_fragment=True,
        heavy_chain_length=4,
        light_chain_length=3,
    )
    assert config.heavy_chain_length == 4
    assert config.light_chain_length == 3

    with pytest.raises(
        ValidationError,
        match="heavy_chain_length and light_chain_length are required when use_single_chain_variable_fragment=True",
    ):
        AbLangGerminalGradientConfig(use_single_chain_variable_fragment=True)

    with pytest.raises(
        ValidationError,
        match="only supported when use_single_chain_variable_fragment=True",
    ):
        AbLangGerminalGradientConfig(use_single_chain_variable_fragment=False, heavy_chain_length=4)


def test_ablang_germinal_gradient_dispatch_contract(monkeypatch):
    captured: dict[str, object] = {}

    def fake_dispatch(tool_name, payload, *, instance=None, config=None):
        captured["tool_name"] = tool_name
        captured["payload"] = payload
        captured["instance"] = instance
        captured["config"] = config
        return {
            "gradient": [[0.1] * 20, [0.2] * 20],
            "loss": 0.75,
            "metrics": {"log_likelihood": -0.75, "model_choice": "ablang2-paired"},
            "vocab": list(PROTEIN_AMINO_ACIDS),
        }

    monkeypatch.setattr(
        "proto_tools.tools.masked_models.ablang.ablang_germinal_gradient.ToolInstance.dispatch",
        fake_dispatch,
    )

    inputs = AbLangGerminalGradientInput(logits=[[0.0] * 20, [1.0] * 20], temperature=0.8)
    config = AbLangGerminalGradientConfig(
        use_single_chain_variable_fragment=True,
        heavy_chain_first=False,
        heavy_chain_length=1,
        light_chain_length=1,
        seed=17,
        device="cpu",
    )
    result = run_ablang_germinal_gradient(inputs=inputs, config=config)

    validate_output(result)
    assert captured["tool_name"] == "ablang"
    assert captured["payload"] == {
        "operation": "compute_germinal_gradient",
        "logits": inputs.logits,
        "temperature": 0.8,
        "use_single_chain_variable_fragment": True,
        "heavy_chain_first": False,
        "heavy_chain_length": 1,
        "light_chain_length": 1,
        "seed": 17,
        "device": "cpu",
        "verbose": False,
    }
    assert result.gradient == [[0.1] * 20, [0.2] * 20]
    assert result.loss == 0.75
    assert result.metrics["log_likelihood"] == -0.75
    assert result.vocab == _CANONICAL_VOCAB


# ── Embedding tests ──────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_ablang_embeddings_heavy():
    """Test ablang1-heavy embeddings with multiple variable-length sequences."""
    seqs = [VH_SEQ, VH_SEQ[:50]]
    result = run_ablang_embeddings(
        AbLangEmbeddingsInput(sequences=seqs),
        AbLangEmbeddingsConfig(model_choice="ablang1-heavy", batch_size=2),
    )
    validate_output(result)

    assert result.tool_id == "ablang-embedding"
    assert len(result.results) == 2

    for i, seq in enumerate(seqs):
        emb = result.results[i]
        assert len(emb.mean_embedding) == 768
        assert all(math.isfinite(v) for v in emb.mean_embedding)
        assert all(v in (0, 1) for v in emb.attention_mask)
        assert sum(emb.attention_mask) >= len(seq)

    assert result.results[0].mean_embedding != result.results[1].mean_embedding


@pytest.mark.uses_gpu
def test_ablang_embeddings_light():
    """Test ablang1-light embeddings produce correct 768-dim vectors."""
    result = run_ablang_embeddings(
        AbLangEmbeddingsInput(sequences=[VL_SEQ]),
        AbLangEmbeddingsConfig(model_choice="ablang1-light"),
    )
    validate_output(result)

    emb = result.results[0]
    assert len(emb.mean_embedding) == 768
    assert all(math.isfinite(v) for v in emb.mean_embedding)
    assert all(v in (0, 1) for v in emb.attention_mask)
    assert sum(emb.attention_mask) >= len(VL_SEQ)


@pytest.mark.uses_gpu
def test_ablang_embeddings_paired():
    """Test ablang2-paired embeddings produce correct 480-dim vectors."""
    result = run_ablang_embeddings(
        AbLangEmbeddingsInput(sequences=[PAIRED_SEQ]),
        AbLangEmbeddingsConfig(model_choice="ablang2-paired"),
    )
    validate_output(result)

    emb = result.results[0]
    assert len(emb.mean_embedding) == 480
    assert all(math.isfinite(v) for v in emb.mean_embedding)
    assert all(v in (0, 1) for v in emb.attention_mask)
    assert sum(emb.attention_mask) >= len(HEAVY_FULL) + len(LIGHT_FULL)


@pytest.mark.uses_gpu
def test_ablang_embeddings_auto_routing():
    """Test that auto model routing selects correct model and embedding dim."""
    single_result = run_ablang_embeddings(
        AbLangEmbeddingsInput(sequences=[VH_SEQ]),
        AbLangEmbeddingsConfig(model_choice="auto"),
    )
    assert single_result.success
    assert single_result.metadata["model_choice"] == "ablang1-heavy"
    assert len(single_result.results[0].mean_embedding) == 768

    paired_result = run_ablang_embeddings(
        AbLangEmbeddingsInput(sequences=[PAIRED_SEQ]),
        AbLangEmbeddingsConfig(model_choice="auto"),
    )
    assert paired_result.success
    assert paired_result.metadata["model_choice"] == "ablang2-paired"
    assert len(paired_result.results[0].mean_embedding) == 480


# ── Scoring tests ────────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_ablang_score_heavy():
    """Test ablang1-heavy PLL: natural VH scores higher than poly-A."""
    result = run_ablang_score(
        AbLangScoringInput(sequences=[VH_SEQ, "A" * 20]),
        AbLangScoringConfig(model_choice="ablang1-heavy", scoring_mode="pseudo_log_likelihood"),
    )
    validate_output(result)

    pll_natural = result.scores[0]["pseudo_log_likelihood"]
    pll_poly_a = result.scores[1]["pseudo_log_likelihood"]

    assert result.tool_id == "ablang-score"
    assert math.isfinite(pll_natural) and pll_natural < 0
    assert math.isfinite(pll_poly_a) and pll_poly_a < 0
    assert pll_natural > pll_poly_a


@pytest.mark.uses_gpu
def test_ablang_score_light():
    """Test ablang1-light PLL: natural VL scores higher than poly-A."""
    result = run_ablang_score(
        AbLangScoringInput(sequences=[VL_SEQ, "A" * 20]),
        AbLangScoringConfig(model_choice="ablang1-light", scoring_mode="pseudo_log_likelihood"),
    )
    validate_output(result)

    pll_natural = result.scores[0]["pseudo_log_likelihood"]
    pll_poly_a = result.scores[1]["pseudo_log_likelihood"]

    assert math.isfinite(pll_natural) and pll_natural < 0
    assert math.isfinite(pll_poly_a) and pll_poly_a < 0
    assert pll_natural > pll_poly_a


@pytest.mark.uses_gpu
def test_ablang_score_paired():
    """Test ablang2-paired PLL produces finite negative scores."""
    result = run_ablang_score(
        AbLangScoringInput(sequences=[PAIRED_SEQ]),
        AbLangScoringConfig(model_choice="ablang2-paired", scoring_mode="pseudo_log_likelihood"),
    )
    validate_output(result)

    pll = result.scores[0]["pseudo_log_likelihood"]
    assert math.isfinite(pll) and pll < 0


@pytest.mark.uses_gpu
def test_ablang_score_confidence_mode():
    """Test confidence scoring returns finite scores for all three models."""
    for model, seq in [("ablang1-heavy", VH_SEQ), ("ablang1-light", VL_SEQ), ("ablang2-paired", PAIRED_SEQ)]:
        result = run_ablang_score(
            AbLangScoringInput(sequences=[seq]),
            AbLangScoringConfig(model_choice=model, scoring_mode="confidence"),
        )
        validate_output(result)
        assert math.isfinite(result.scores[0]["confidence"])


# ── Sampling tests ───────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_ablang_sample_heavy():
    """Test ablang1-heavy restoration: valid AAs, length preserved, unmasked positions unchanged."""
    mask_pos = 4
    masked_seq = VH_SEQ[:mask_pos] + "_" + VH_SEQ[mask_pos + 1 :]
    result = run_ablang_sample(
        AbLangSampleInput(sequences=[masked_seq]),
        AbLangSampleConfig(model_choice="ablang1-heavy"),
    )
    validate_output(result)

    restored = result.sequences[0]
    assert result.tool_id == "ablang-sample"
    assert len(restored) == len(VH_SEQ)
    assert "_" not in restored
    assert set(restored) <= _VALID_AAS
    assert restored[:mask_pos] == VH_SEQ[:mask_pos]
    assert restored[mask_pos + 1 :] == VH_SEQ[mask_pos + 1 :]


@pytest.mark.uses_gpu
def test_ablang_sample_light():
    """Test ablang1-light restoration: valid AAs, length preserved, unmasked unchanged."""
    mask_pos = 4
    masked_seq = VL_SEQ[:mask_pos] + "_" + VL_SEQ[mask_pos + 1 :]
    result = run_ablang_sample(
        AbLangSampleInput(sequences=[masked_seq]),
        AbLangSampleConfig(model_choice="ablang1-light"),
    )
    validate_output(result)

    restored = result.sequences[0]
    assert len(restored) == len(VL_SEQ)
    assert "_" not in restored
    assert set(restored) <= _VALID_AAS
    assert restored[:mask_pos] == VL_SEQ[:mask_pos]
    assert restored[mask_pos + 1 :] == VL_SEQ[mask_pos + 1 :]


@pytest.mark.uses_gpu
def test_ablang_sample_paired():
    """Test ablang2-paired restoration: valid AAs, length preserved, unmasked unchanged."""
    mask_pos = 4
    masked_heavy = HEAVY_FULL[:mask_pos] + "_" + HEAVY_FULL[mask_pos + 1 :]
    masked_paired = masked_heavy + "|" + LIGHT_FULL
    result = run_ablang_sample(
        AbLangSampleInput(sequences=[masked_paired]),
        AbLangSampleConfig(model_choice="ablang2-paired"),
    )
    validate_output(result)

    restored = result.sequences[0]
    residues_only = restored.replace("|", "")
    assert "_" not in restored
    assert len(restored) == len(masked_paired)
    assert set(residues_only) <= _VALID_AAS
    assert restored[:mask_pos] == HEAVY_FULL[:mask_pos]
    assert restored[mask_pos + 1 : len(HEAVY_FULL)] == HEAVY_FULL[mask_pos + 1 :]


# ── Germinal gradient tests ─────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_ablang_germinal_gradient_vhh():
    """Test Germinal VHH gradient: shape, finiteness, and metric consistency."""
    seq_len = 10
    result = run_ablang_germinal_gradient(
        AbLangGerminalGradientInput(logits=[[0.0] * 20] * seq_len, temperature=0.6),
        AbLangGerminalGradientConfig(use_single_chain_variable_fragment=False),
    )
    validate_output(result)

    assert result.tool_id == "ablang-germinal-gradient"
    assert len(result.gradient) == seq_len
    assert all(len(row) == 20 for row in result.gradient)
    assert all(math.isfinite(v) for row in result.gradient for v in row)
    assert any(v != 0.0 for row in result.gradient for v in row)
    assert math.isfinite(result.loss) and result.loss > 0
    assert result.vocab == _CANONICAL_VOCAB
    assert result.metrics["sequence_length"] == seq_len
    assert result.metrics["effective_sequence_length"] == seq_len
    assert result.metrics["linker_length"] == 0
    assert result.metrics["model_choice"] == "ablang1-heavy"
    assert result.metrics["objective"] == "germinal_shifted_cross_entropy"
    assert result.metrics["log_likelihood"] == pytest.approx(-result.loss, rel=1e-6)


@pytest.mark.parametrize(
    ("heavy_chain_first", "expected_linker_start"),
    [(True, 4), (False, 5)],
)
@pytest.mark.uses_gpu
def test_ablang_germinal_gradient_single_chain_variable_fragment_zeroes_linker(
    heavy_chain_first: bool,
    expected_linker_start: int,
):
    """Test Germinal paired-chain gradients zero-pad the linker for both chain orders."""
    heavy_chain_length = 4
    linker_len = 3
    light_chain_length = 5
    seq_len = heavy_chain_length + linker_len + light_chain_length

    result = run_ablang_germinal_gradient(
        AbLangGerminalGradientInput(logits=[[0.0] * 20] * seq_len, temperature=0.6),
        AbLangGerminalGradientConfig(
            use_single_chain_variable_fragment=True,
            heavy_chain_length=heavy_chain_length,
            light_chain_length=light_chain_length,
            heavy_chain_first=heavy_chain_first,
        ),
    )
    validate_output(result)

    linker_rows = result.gradient[expected_linker_start : expected_linker_start + linker_len]
    prefix_length = heavy_chain_length if heavy_chain_first else light_chain_length
    suffix_length = light_chain_length if heavy_chain_first else heavy_chain_length
    non_linker_rows = result.gradient[:prefix_length] + result.gradient[-suffix_length:]

    assert len(result.gradient) == seq_len
    assert all(len(row) == 20 for row in result.gradient)
    assert all(math.isfinite(v) for row in result.gradient for v in row)
    assert all(v == 0.0 for row in linker_rows for v in row)
    assert any(v != 0.0 for row in non_linker_rows for v in row)
    assert result.vocab == _CANONICAL_VOCAB
    assert result.metrics["sequence_length"] == seq_len
    assert result.metrics["effective_sequence_length"] == heavy_chain_length + light_chain_length
    assert result.metrics["linker_length"] == linker_len
    assert result.metrics["use_single_chain_variable_fragment"] is True
    assert result.metrics["model_choice"] == "ablang2-paired"
    assert result.metrics["objective"] == "germinal_shifted_cross_entropy"


# ── Batched tests ────────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_ablang_batched_operations():
    """Test that batch_size > 1 works for embeddings, scoring, and sampling."""
    seqs = [VH_SEQ, VH_SEQ[:50], VH_SEQ[:80]]

    emb_result = run_ablang_embeddings(
        AbLangEmbeddingsInput(sequences=seqs),
        AbLangEmbeddingsConfig(model_choice="ablang1-heavy", batch_size=2),
    )
    assert emb_result.success
    assert len(emb_result.results) == 3
    assert all(len(r.mean_embedding) == 768 for r in emb_result.results)

    score_result = run_ablang_score(
        AbLangScoringInput(sequences=seqs),
        AbLangScoringConfig(model_choice="ablang1-heavy", batch_size=2),
    )
    assert score_result.success
    assert len(score_result.scores) == 3
    assert all("pseudo_log_likelihood" in s for s in score_result.scores)

    masked_seqs = [s[:4] + "_" + s[5:] for s in seqs]
    sample_result = run_ablang_sample(
        AbLangSampleInput(sequences=masked_seqs),
        AbLangSampleConfig(model_choice="ablang1-heavy", batch_size=2),
    )
    assert sample_result.success
    assert len(sample_result.sequences) == 3
    assert all("_" not in s for s in sample_result.sequences)
