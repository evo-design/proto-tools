"""tests/masked_models_tests/test_esmc.py.

Tests for ESM C (Cambrian) embeddings tool.
"""

import pytest

from proto_tools.tools.masked_models.esmc import (
    ESMCEmbeddingsConfig,
    ESMCEmbeddingsInput,
    run_esmc_embeddings,
)
from tests.conftest import benchmark_twice, make_persistent_fixture, random_protein_sequences
from tests.tool_infra_tests.test_export_functionality import validate_output

_persistent_tool = make_persistent_fixture("esmc")

# (checkpoint, hidden_dim): 300M is open license, 600M is non-commercial.
_CHECKPOINTS = [("esmc_300m", 960), ("esmc_600m", 1152)]
_CHECKPOINT_IDS = [c for c, _ in _CHECKPOINTS]


@pytest.mark.uses_gpu
@pytest.mark.parametrize(("checkpoint", "embedding_dim"), _CHECKPOINTS)
def test_esmc_forward_pass_shapes(checkpoint, embedding_dim):
    """End-to-end embedding extraction returns the expected shapes."""
    sequences = ["MKTAYIAKQR", "GSSGSSGSS"]
    inputs = ESMCEmbeddingsInput(sequences=sequences)
    config = ESMCEmbeddingsConfig(model_checkpoint=checkpoint, batch_size=2, return_logits=True)

    result = run_esmc_embeddings(inputs=inputs, config=config)
    validate_output(result)

    assert result.tool_id == "esmc-embedding"
    assert len(result.results) == 2

    first = result.results[0]
    assert len(first.mean_embedding) == embedding_dim
    # Attention mask covers the input residues (BOS/EOS stripped)
    assert len(first.attention_mask) == max(len(s) for s in sequences)
    # Logits are over the 20 amino acids
    assert first.logits is not None
    assert len(first.logits[0]) == 20


@pytest.mark.uses_gpu
@pytest.mark.parametrize("checkpoint", _CHECKPOINT_IDS)
def test_esmc_logits_disabled_by_default(checkpoint):
    inputs = ESMCEmbeddingsInput(sequences=["MKTAYIAKQR"])
    result = run_esmc_embeddings(inputs=inputs, config=ESMCEmbeddingsConfig(model_checkpoint=checkpoint))
    assert result.results[0].logits is None


@pytest.mark.uses_gpu
@pytest.mark.parametrize("checkpoint", _CHECKPOINT_IDS)
def test_esmc_different_sequences_produce_different_embeddings(checkpoint):
    """Sanity check: distinct sequences should not collapse to the same embedding."""
    inputs = ESMCEmbeddingsInput(sequences=["MVLSPADKTNVKAAW", "AAAAAAAAAAAAAAA"])
    result = run_esmc_embeddings(inputs=inputs, config=ESMCEmbeddingsConfig(model_checkpoint=checkpoint))

    a = result.results[0].mean_embedding
    b = result.results[1].mean_embedding
    assert any(abs(x - y) > 1e-3 for x, y in zip(a, b, strict=True))


@pytest.mark.uses_gpu
@pytest.mark.parametrize("checkpoint", _CHECKPOINT_IDS)
def test_esmc_persistent_worker_reuse(checkpoint):
    """Two back-to-back dispatches under the module-scoped persistence fixture."""
    inputs = ESMCEmbeddingsInput(sequences=["MKTAYIAKQR"])
    config = ESMCEmbeddingsConfig(model_checkpoint=checkpoint)

    r1 = run_esmc_embeddings(inputs=inputs, config=config)
    r2 = run_esmc_embeddings(inputs=inputs, config=config)

    assert r1.success and r2.success
    # Identical inputs through one persistent worker → identical embeddings
    assert r1.results[0].mean_embedding == r2.results[0].mean_embedding


# ── Benchmarks ────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("esmc-embedding")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_esmc_embedding_benchmark(request):
    """Benchmark esmc-embedding on 100 sequences of length 300 (cold + warm)."""
    sequences = random_protein_sequences(n=100, length=300, seed=0)
    inputs = ESMCEmbeddingsInput(sequences=sequences)
    config = ESMCEmbeddingsConfig(model_checkpoint="esmc_300m", batch_size=32, return_logits=True)

    result = benchmark_twice(request, "esmc", lambda: run_esmc_embeddings(inputs=inputs, config=config))

    assert result.tool_id == "esmc-embedding"
    assert len(result.results) == 100, "Should have 100 SequenceEmbedding objects"
    assert len(result.results[0].mean_embedding) == 960, "esmc_300m embedding dimension should be 960"
    assert len(result.results[0].attention_mask) == 300, "Attention mask length should be 300"
    assert result.results[0].logits is not None
    assert len(result.results[0].logits) == 300, "Logit sequence length should be 300"
    assert len(result.results[0].logits[0]) == 20, "Logit vocab size should be 20"
