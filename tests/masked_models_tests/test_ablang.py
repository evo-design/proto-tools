"""Tests for AbLang antibody language model tool."""

import pytest

from proto_tools.tools.masked_models.ablang import (
    AbLangEmbeddingsConfig,
    AbLangEmbeddingsInput,
    AbLangSampleConfig,
    AbLangSampleInput,
    AbLangScoringConfig,
    AbLangScoringInput,
    run_ablang_embeddings,
    run_ablang_sample,
    run_ablang_score,
)
from tests.conftest import make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import validate_output

_persistent_tool = make_persistent_fixture("ablang")

# Full antibody sequences (heavy + light with constant regions)
HEAVY_FULL = "AVKLVQAGGGVVQPGRSLRLSCIASGFTFSNYGMHWVRQAPGKGLEWVAVIWYNGSRTYYGDSVKGRFTISRDNSKRTLYMQMNSLRTEDTAVYYCARDPDILTAFSFDYWGQGVLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC"
LIGHT_FULL = "SYELTQPPSVSVSPGQTARITCSANALPNQYAYWYQQKPGRAPVMVIYKDTQRPSGIPQRFSSSTSGTTVTLTISGVQAEDEADYYCQAWDNSASIFGGGTKLTVLGQPKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPIKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTECS"
PAIRED_SEQ = f"{HEAVY_FULL}|{LIGHT_FULL}"

# Variable domain sequences for ablang1 models (max_position_embeddings=160)
VH_SEQ = "AVKLVQAGGGVVQPGRSLRLSCIASGFTFSNYGMHWVRQAPGKGLEWVAVIWYNGSRTYYGDSVKGRFTISRDNSKRTLYMQMNSLRTEDTAVYYCARDPDILTAFSFDYWGQGVLVTVSS"
VL_SEQ = "SYELTQPPSVSVSPGQTARITCSANALPNQYAYWYQQKPGRAPVMVIYKDTQRPSGIPQRFSSSTSGTTVTLTISGVQAEDEADYYCQAWDNSASIFGGGTKLTVLG"


# ── Input validation ─────────────────────────────────────────────────────────


def test_ablang_embeddings_input_normalizes_single_string():
    inp = AbLangEmbeddingsInput(sequences="EVQLVESGGGLVQPGG")
    assert isinstance(inp.sequences, list)
    assert inp.sequences == ["EVQLVESGGGLVQPGG"]


def test_ablang_scoring_input_normalizes_single_string():
    inp = AbLangScoringInput(sequences="EVQLVESGGGLVQPGG")
    assert isinstance(inp.sequences, list)
    assert inp.sequences == ["EVQLVESGGGLVQPGG"]


def test_ablang_sample_input_normalizes_single_string():
    inp = AbLangSampleInput(sequences="EVQL_ESGGGLVQPGG")
    assert isinstance(inp.sequences, list)
    assert inp.sequences == ["EVQL_ESGGGLVQPGG"]


def test_ablang_empty_input_raises():
    with pytest.raises(ValueError, match="sequences must not be empty"):
        AbLangEmbeddingsInput(sequences=[])
    with pytest.raises(ValueError, match="sequences must not be empty"):
        AbLangScoringInput(sequences=[])
    with pytest.raises(ValueError, match="sequences must not be empty"):
        AbLangSampleInput(sequences=[])


# ── Embedding tests ──────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_ablang_embeddings_heavy():
    """Test AbLang embeddings with ablang1-heavy model."""
    inputs = AbLangEmbeddingsInput(sequences=[VH_SEQ, VH_SEQ[:50]])
    config = AbLangEmbeddingsConfig(
        model_choice="ablang1-heavy",
        batch_size=1,
    )

    result = run_ablang_embeddings(inputs=inputs, config=config)
    validate_output(result)

    assert result.tool_id == "ablang-embedding"
    assert len(result.results) == 2
    assert len(result.results[0].mean_embedding) == 768
    assert result.results[0].attention_mask is not None


@pytest.mark.uses_gpu
def test_ablang_embeddings_light():
    """Test AbLang embeddings with ablang1-light model."""
    inputs = AbLangEmbeddingsInput(sequences=[VL_SEQ])
    config = AbLangEmbeddingsConfig(
        model_choice="ablang1-light",
        batch_size=1,
    )

    result = run_ablang_embeddings(inputs=inputs, config=config)
    validate_output(result)

    assert result.tool_id == "ablang-embedding"
    assert len(result.results) == 1
    assert len(result.results[0].mean_embedding) == 768


@pytest.mark.uses_gpu
def test_ablang_embeddings_paired():
    """Test AbLang embeddings with ablang2-paired model."""
    inputs = AbLangEmbeddingsInput(sequences=[PAIRED_SEQ])
    config = AbLangEmbeddingsConfig(
        model_choice="ablang2-paired",
        batch_size=1,
    )

    result = run_ablang_embeddings(inputs=inputs, config=config)
    validate_output(result)

    assert len(result.results) == 1
    assert len(result.results[0].mean_embedding) == 480


@pytest.mark.uses_gpu
def test_ablang_embeddings_auto_routing():
    """Test that model_choice='auto' routes paired sequences to ablang2-paired."""
    # Single chain → should auto-route to ablang1-heavy
    single_result = run_ablang_embeddings(
        AbLangEmbeddingsInput(sequences=[VH_SEQ]),
        AbLangEmbeddingsConfig(model_choice="auto"),
    )
    assert single_result.success
    assert single_result.metadata["model_choice"] == "ablang1-heavy"
    assert len(single_result.results[0].mean_embedding) == 768

    # Paired → should auto-route to ablang2-paired
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
    """Test AbLang scoring with ablang1-heavy model."""
    inputs = AbLangScoringInput(sequences=[VH_SEQ])
    config = AbLangScoringConfig(
        model_choice="ablang1-heavy",
        scoring_mode="pseudo_log_likelihood",
    )

    result = run_ablang_score(inputs=inputs, config=config)
    validate_output(result)

    assert result.tool_id == "ablang-score"
    assert len(result.scores) == 1
    assert "pseudo_log_likelihood" in result.scores[0].metrics


@pytest.mark.uses_gpu
def test_ablang_score_light():
    """Test AbLang scoring with ablang1-light model."""
    inputs = AbLangScoringInput(sequences=[VL_SEQ])
    config = AbLangScoringConfig(
        model_choice="ablang1-light",
        scoring_mode="pseudo_log_likelihood",
    )

    result = run_ablang_score(inputs=inputs, config=config)
    validate_output(result)

    assert result.tool_id == "ablang-score"
    assert len(result.scores) == 1
    assert "pseudo_log_likelihood" in result.scores[0].metrics


@pytest.mark.uses_gpu
def test_ablang_score_paired():
    """Test AbLang scoring with ablang2-paired model."""
    inputs = AbLangScoringInput(sequences=[PAIRED_SEQ])
    config = AbLangScoringConfig(
        model_choice="ablang2-paired",
        scoring_mode="pseudo_log_likelihood",
    )

    result = run_ablang_score(inputs=inputs, config=config)
    validate_output(result)

    assert result.tool_id == "ablang-score"
    assert len(result.scores) == 1
    assert "pseudo_log_likelihood" in result.scores[0].metrics


@pytest.mark.uses_gpu
def test_ablang_score_different_sequences():
    """Test that different sequences produce different scores."""
    seq1 = VH_SEQ
    seq2 = "A" * 20

    inputs = AbLangScoringInput(sequences=[seq1, seq2])
    config = AbLangScoringConfig(
        model_choice="ablang1-heavy",
        scoring_mode="pseudo_log_likelihood",
    )

    result = run_ablang_score(inputs=inputs, config=config)

    assert len(result.scores) == 2
    pll1 = result.scores[0].metrics["pseudo_log_likelihood"]
    pll2 = result.scores[1].metrics["pseudo_log_likelihood"]
    assert pll1 != pll2, f"Different sequences should have different scores: {pll1} vs {pll2}"


@pytest.mark.uses_gpu
def test_ablang_score_confidence_mode():
    """Test AbLang scoring with confidence mode."""
    inputs = AbLangScoringInput(sequences=[VH_SEQ])
    config = AbLangScoringConfig(
        model_choice="ablang1-heavy",
        scoring_mode="confidence",
    )

    result = run_ablang_score(inputs=inputs, config=config)
    validate_output(result)

    assert len(result.scores) == 1
    assert "confidence" in result.scores[0].metrics


# ── Sampling tests ───────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_ablang_sample_heavy():
    """Test AbLang sequence restoration with ablang1-heavy model."""
    masked_seq = VH_SEQ[:4] + "_" + VH_SEQ[5:]
    inputs = AbLangSampleInput(sequences=[masked_seq])
    config = AbLangSampleConfig(
        model_choice="ablang1-heavy",
    )

    result = run_ablang_sample(inputs=inputs, config=config)
    validate_output(result)

    assert result.tool_id == "ablang-sample"
    assert len(result.sequences) == 1
    assert "_" not in result.sequences[0], "Restored sequence should not contain '_' mask tokens"
    assert len(result.sequences[0]) == len(masked_seq), "Restored sequence length should match input length"


@pytest.mark.uses_gpu
def test_ablang_sample_light():
    """Test AbLang sequence restoration with ablang1-light model."""
    masked_seq = VL_SEQ[:4] + "_" + VL_SEQ[5:]
    inputs = AbLangSampleInput(sequences=[masked_seq])
    config = AbLangSampleConfig(
        model_choice="ablang1-light",
    )

    result = run_ablang_sample(inputs=inputs, config=config)
    validate_output(result)

    assert result.tool_id == "ablang-sample"
    assert len(result.sequences) == 1
    assert "_" not in result.sequences[0]
    assert len(result.sequences[0]) == len(masked_seq)


@pytest.mark.uses_gpu
def test_ablang_sample_paired():
    """Test AbLang sequence restoration with ablang2-paired model."""
    # Mask a position in the heavy chain
    masked_paired = HEAVY_FULL[:4] + "_" + HEAVY_FULL[5:] + "|" + LIGHT_FULL
    inputs = AbLangSampleInput(sequences=[masked_paired])
    config = AbLangSampleConfig(
        model_choice="ablang2-paired",
    )

    result = run_ablang_sample(inputs=inputs, config=config)
    validate_output(result)

    assert result.tool_id == "ablang-sample"
    assert len(result.sequences) == 1
    assert "_" not in result.sequences[0]


# ── Batched tests ───────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_ablang_batched_operations():
    """Test that batch_size > 1 works for embeddings, scoring, and sampling."""
    seqs = [VH_SEQ, VH_SEQ[:50], VH_SEQ[:80]]

    # Embeddings with batch_size=2 (3 seqs → 2 batches)
    emb_result = run_ablang_embeddings(
        AbLangEmbeddingsInput(sequences=seqs),
        AbLangEmbeddingsConfig(model_choice="ablang1-heavy", batch_size=2),
    )
    assert emb_result.success
    assert len(emb_result.results) == 3
    assert all(len(r.mean_embedding) == 768 for r in emb_result.results)

    # Scoring with batch_size=2
    score_result = run_ablang_score(
        AbLangScoringInput(sequences=seqs),
        AbLangScoringConfig(model_choice="ablang1-heavy", batch_size=2),
    )
    assert score_result.success
    assert len(score_result.scores) == 3
    assert all("pseudo_log_likelihood" in s.metrics for s in score_result.scores)

    # Sampling with batch_size=2
    masked_seqs = [s[:4] + "_" + s[5:] for s in seqs]
    sample_result = run_ablang_sample(
        AbLangSampleInput(sequences=masked_seqs),
        AbLangSampleConfig(model_choice="ablang1-heavy", batch_size=2),
    )
    assert sample_result.success
    assert len(sample_result.sequences) == 3
    assert all("_" not in s for s in sample_result.sequences)
