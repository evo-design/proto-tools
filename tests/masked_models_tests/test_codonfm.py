"""tests/masked_models_tests/test_codonfm.py.

Tests for CodonFM (Encodon): codon-level masked-LM fitness, embeddings, mutation scoring,
and the masked pseudo-log-likelihood gradient.
"""

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from proto_tools.tools.masked_models.codonfm import (
    CODONFM_CODON_VOCAB,
    CODONFM_NUM_CODONS,
    CodonFMEmbeddingsConfig,
    CodonFMEmbeddingsInput,
    CodonFMFitnessConfig,
    CodonFMFitnessInput,
    CodonFMGradientConfig,
    CodonFMGradientInput,
    CodonFMSampleConfig,
    CodonFMSampleInput,
    CodonFMScoreConfig,
    CodonFMScoreInput,
    one_hot_codon_logits,
    resolve_checkpoint_source,
    run_codonfm_embeddings,
    run_codonfm_fitness,
    run_codonfm_gradient,
    run_codonfm_sample,
    run_codonfm_score,
)
from proto_tools.tools.masked_models.codonfm.shared_data_models import CODONFM_CHECKPOINTS, CODONFM_MAX_NT
from tests.conftest import benchmark_twice, make_persistent_fixture
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output

_persistent_tool = make_persistent_fixture("codonfm")

# A short in-frame CDS (ATG start) reused across the model integration tests.
_CDS = "ATGGTGAGCAAGGGCGAGGAGCTGTTCACC"  # 30 nt, 10 codons


def _skip_if_no_gpu() -> None:
    from proto_tools.utils.device import number_of_visible_gpus

    if number_of_visible_gpus() < 1:
        pytest.skip("CodonFM GPU test requires a visible CUDA GPU")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_codonfm_input_normalizes_and_validates() -> None:
    """A CodonFM sequence input maps RNA U to T and rejects ambiguous or invalid characters."""
    inputs = CodonFMFitnessInput(sequences="augg ug")  # whitespace stripped, U -> T
    assert inputs.sequences == ["ATGGTG"]
    assert len(inputs) == 1

    with pytest.raises(ValidationError, match="Invalid nucleotide characters"):
        CodonFMFitnessInput(sequences="ATGX")
    with pytest.raises(ValidationError, match="Invalid nucleotide characters"):
        CodonFMFitnessInput(sequences="ATGNNN")
    with pytest.raises(ValidationError, match="multiple of 3"):
        CodonFMFitnessInput(sequences="ATGG")  # not codon-aligned
    with pytest.raises(ValidationError, match="up to"):
        CodonFMEmbeddingsInput(sequences="A" * (CODONFM_MAX_NT + 3))  # over the codon cap


def test_codonfm_score_input_validates_mutation() -> None:
    """A score mutation normalizes codons, checks the position, and enforces ref/sequence agreement."""
    inp = CodonFMScoreInput(
        mutations=[{"sequence": "ATGGTGAGC", "codon_position": 2, "ref_codon": "gtg", "alt_codon": "gua"}]
    )
    mutation = inp.mutations[0]
    assert mutation.ref_codon == "GTG"  # uppercased
    assert mutation.alt_codon == "GTA"  # U -> T
    assert len(inp) == 1

    with pytest.raises(ValidationError, match="does not match"):
        CodonFMScoreInput(
            mutations=[{"sequence": "ATGGTGAGC", "codon_position": 2, "ref_codon": "AAA", "alt_codon": "GTA"}]
        )
    with pytest.raises(ValidationError, match="out of range"):
        CodonFMScoreInput(
            mutations=[{"sequence": "ATGGTG", "codon_position": 5, "ref_codon": "GTG", "alt_codon": "GTA"}]
        )
    with pytest.raises(ValidationError, match="3-nucleotide"):
        CodonFMScoreInput(
            mutations=[{"sequence": "ATGGTG", "codon_position": 1, "ref_codon": "GT", "alt_codon": "GTA"}]
        )


def test_codonfm_one_hot_codon_logits_matches_vocab_order() -> None:
    """The one-hot helper builds (L, 64) rows in CODONFM_CODON_VOCAB column order and rejects N."""
    assert CODONFM_NUM_CODONS == 64
    assert CODONFM_CODON_VOCAB[0] == "AAA" and CODONFM_CODON_VOCAB[-1] == "TTT"

    logits = one_hot_codon_logits("ATGGTG", sharpness=5.0)
    assert len(logits) == 2 and all(len(row) == 64 for row in logits)
    assert logits[0][CODONFM_CODON_VOCAB.index("ATG")] == 5.0
    assert logits[1][CODONFM_CODON_VOCAB.index("GTG")] == 5.0
    assert sum(1 for v in logits[0] if v != 0.0) == 1  # exactly one hot column

    with pytest.raises(ValueError, match="Invalid nucleotide characters"):
        one_hot_codon_logits("ATGNNN")
    with pytest.raises(ValueError, match="finite"):
        one_hot_codon_logits("ATG", sharpness=float("nan"))


def test_codonfm_gradient_input_validation() -> None:
    """The gradient input requires a non-empty L x 64 matrix and a positive temperature."""
    inp = CodonFMGradientInput(logits=[[0.0] * 64, [1.0] * 64])
    assert inp.temperature == 1.0

    assert CodonFMGradientInput(logits=[[0.0] * 64], temperature=0.6).temperature == 0.6
    with pytest.raises(ValidationError):
        CodonFMGradientInput(logits=[[0.0] * 64], temperature=0.0)
    with pytest.raises(ValidationError):
        CodonFMGradientInput(logits=[[0.0] * 64], temperature=-1.0)
    with pytest.raises(ValidationError, match="64 columns"):
        CodonFMGradientInput(logits=[[0.0] * 63])
    with pytest.raises(ValidationError, match="at least one"):
        CodonFMGradientInput(logits=[])
    with pytest.raises(ValidationError, match="finite"):
        CodonFMGradientInput(logits=[[float("nan")] * 64])
    with pytest.raises(ValidationError):
        CodonFMGradientInput(logits=[[0.0] * 64], temperature=float("inf"))
    with pytest.raises(ValidationError, match="probability distribution"):
        CodonFMGradientInput(logits=[[0.0] * 64], temperature=None)
    assert CodonFMGradientInput(logits=[[1.0 / 64] * 64], temperature=None).temperature is None


def test_codonfm_vendored_import_closure_is_present() -> None:
    """All documented vendored Python modules, including src/data, ship in the source tree."""
    root = Path(__file__).parents[2] / "proto_tools/tools/masked_models/codonfm/standalone/src"
    assert len(list(root.rglob("*.py"))) == 27
    for relative in (
        "data/__init__.py",
        "data/datamodule.py",
        "data/metadata.py",
        "data/preprocess/__init__.py",
        "data/preprocess/codon_sequence.py",
        "data/preprocess/mutation_pred.py",
        "data/stateful_dataset.py",
    ):
        assert (root / relative).is_file()


def _load_codonfm_worker():
    """Load the lightweight worker module without constructing its standalone environment."""
    root = Path(__file__).parents[2]
    standalone = root / "proto_tools/tools/masked_models/codonfm/standalone"
    helpers = root / "proto_tools/utils/standalone_helpers_source"
    for path in (str(standalone), str(helpers)):
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location("_codonfm_test_worker", standalone / "inference.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_codonfm_worker_padding_and_early_validation(monkeypatch) -> None:
    """Worker contexts satisfy upstream's /8 invariant and bad requests fail before model loading."""
    worker = _load_codonfm_worker()
    model = worker.CodonFMModel()
    assert model._context_length(["ATG" * 10]) == 16
    assert model._context_length(["ATG" * 2046]) == 2048

    loaded = False

    def fail_if_loaded(**_kwargs):
        nonlocal loaded
        loaded = True

    monkeypatch.setattr(worker._MODEL, "load", fail_if_loaded)
    with pytest.raises(ValueError, match="unknown operation"):
        worker.dispatch({"operation": "bogus", "device": "cpu"})
    assert loaded is False
    with pytest.raises(ValueError, match="positive integer"):
        worker.dispatch({"operation": "fitness", "device": "cpu", "batch_size": 0, "sequences": ["ATG"]})
    assert loaded is False


def test_codonfm_worker_gradient_is_padded_finite_and_does_not_touch_parameter_grads(monkeypatch) -> None:
    """The relaxed worker path uses /8 padding and returns only input gradients."""
    torch = pytest.importorskip("torch")
    worker = _load_codonfm_worker()

    class FakeEmbedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.word_embeddings = torch.nn.Embedding(69, 8, padding_idx=67)
            self.post_ln = torch.nn.LayerNorm(8)
            self.dropout = torch.nn.Dropout(0.0)

        def forward(self, input_ids):
            return self.dropout(self.post_ln(self.word_embeddings(input_ids)))

    class FakeModel(torch.nn.Module):
        def __init__(self, embedding):
            super().__init__()
            self.embedding = embedding
            self.head = torch.nn.Linear(8, 69)

        def forward(self, batch):
            input_ids = next(value for key, value in batch.items() if "input" in str(key).lower())
            assert input_ids.shape[-1] % 8 == 0
            return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

    embedding = FakeEmbedding()
    fake_model = FakeModel(embedding)
    fake_model.requires_grad_(False)
    tokenizer = SimpleNamespace(
        codons=CODONFM_CODON_VOCAB,
        encoder={
            **{codon: idx for idx, codon in enumerate(CODONFM_CODON_VOCAB)},
            "<CLS>": 64,
            "<SEP>": 65,
            "<MASK>": 66,
            "<PAD>": 67,
            "<UNK>": 68,
        },
        cls_token_id=64,
        sep_token_id=65,
        mask_token_id=66,
        pad_token_id=67,
    )
    model = worker.CodonFMModel()
    model.inference = SimpleNamespace(model=fake_model)
    model.tokenizer = tokenizer
    model.device = "cpu"
    monkeypatch.setattr(model, "_embedding_layer", lambda: embedding)

    result = model.compute_gradient([[0.0] * 64 for _ in range(3)], temperature=1.0, batch_size=2, device="cpu")
    assert len(result["gradient"]) == 3
    assert all(math.isfinite(value) for row in result["gradient"] for value in row)
    assert all(parameter.grad is None for parameter in fake_model.parameters())


def test_codonfm_resolve_checkpoint_source_covers_all_aliases() -> None:
    """Every checkpoint alias resolves to public HF URLs plus a stable cache subdir."""
    for alias, entry in CODONFM_CHECKPOINTS.items():
        safetensors_url, config_url, filename, subdir = resolve_checkpoint_source(alias)
        assert safetensors_url.startswith("https://huggingface.co/")
        assert safetensors_url.endswith(entry["safetensors"])
        assert config_url.endswith("/config.json")
        assert filename == entry["safetensors"]
        assert subdir.startswith(f"codonfm-{alias}-")
    # Distinct checkpoints get distinct cache subdirs (no cross-checkpoint clobbering).
    subdirs = {resolve_checkpoint_source(alias)[3] for alias in CODONFM_CHECKPOINTS}
    assert len(subdirs) == len(CODONFM_CHECKPOINTS)


# ---------------------------------------------------------------------------
# Dispatch contracts (mocked worker)
# ---------------------------------------------------------------------------
def test_codonfm_fitness_dispatch_contract(monkeypatch) -> None:
    """run_codonfm_fitness dispatches the fitness op and maps per-sequence scores."""
    captured: dict[str, object] = {}

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured["toolkit"] = toolkit
        captured["payload"] = payload
        return {"fitness": [-0.5 for _ in payload["sequences"]]}

    monkeypatch.setattr(
        "proto_tools.tools.masked_models.codonfm.codonfm_fitness.ToolInstance.dispatch",
        staticmethod(fake_dispatch),
    )

    result = run_codonfm_fitness(
        CodonFMFitnessInput(sequences=[_CDS, "ATGGCCACC"]),
        CodonFMFitnessConfig(model_checkpoint="encodon_80m", batch_size=2, device="cpu"),
    )

    assert captured["toolkit"] == "codonfm"
    assert captured["payload"]["operation"] == "fitness"
    assert captured["payload"]["sequences"] == [_CDS, "ATGGCCACC"]
    assert captured["payload"]["safetensors_url"].endswith("NV-CodonFM-Encodon-80M-v1.safetensors")
    assert captured["payload"]["config_url"].endswith("/config.json")
    assert [r.fitness for r in result.results] == [-0.5, -0.5]
    assert [r.sequence_length for r in result.results] == [30, 9]


def test_codonfm_embeddings_dispatch_contract(monkeypatch) -> None:
    """run_codonfm_embeddings dispatches the embeddings op and maps per-sequence vectors."""
    captured: dict[str, object] = {}

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured["payload"] = payload
        return {"embeddings": [[0.1, 0.2, 0.3] for _ in payload["sequences"]]}

    monkeypatch.setattr(
        "proto_tools.tools.masked_models.codonfm.codonfm_embeddings.ToolInstance.dispatch",
        staticmethod(fake_dispatch),
    )

    result = run_codonfm_embeddings(
        CodonFMEmbeddingsInput(sequences=[_CDS]),
        CodonFMEmbeddingsConfig(device="cpu"),
    )

    assert captured["payload"]["operation"] == "embeddings"
    assert result.results[0].embedding == [0.1, 0.2, 0.3]
    assert result.results[0].sequence_length == 30


def test_codonfm_score_dispatch_contract(monkeypatch) -> None:
    """run_codonfm_score dispatches normalized mutations and maps ref/alt log-likelihoods + LLR."""
    captured: dict[str, object] = {}

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured["payload"] = payload
        return {
            "mutations": [
                {"ref_log_likelihood": -0.2, "alt_log_likelihood": -1.2, "llr": 1.0} for _ in payload["mutations"]
            ]
        }

    monkeypatch.setattr(
        "proto_tools.tools.masked_models.codonfm.codonfm_score.ToolInstance.dispatch",
        staticmethod(fake_dispatch),
    )

    result = run_codonfm_score(
        CodonFMScoreInput(
            mutations=[{"sequence": "ATGGTGAGC", "codon_position": 2, "ref_codon": "GTG", "alt_codon": "GTA"}]
        ),
        CodonFMScoreConfig(device="cpu"),
    )

    assert captured["payload"]["operation"] == "score"
    assert captured["payload"]["mutations"][0] == {
        "sequence": "ATGGTGAGC",
        "codon_position": 1,
        "ref_codon": "GTG",
        "alt_codon": "GTA",
    }
    assert result.results[0].llr == 1.0
    assert result.results[0].sequence == "ATGGTGAGC"
    assert result.results[0].sequence_length == 9
    assert result.results[0].codon_position == 2
    assert result.results[0].ref_codon == "GTG"


def test_codonfm_gradient_dispatch_contract(monkeypatch) -> None:
    """run_codonfm_gradient forwards logits/temperature/ste and returns the gradient bundle."""
    captured: dict[str, object] = {}

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured["toolkit"] = toolkit
        captured["payload"] = payload
        n = len(payload["logits"])
        return {
            "gradient": [[0.0] * 64 for _ in range(n)],
            "loss": 0.5,
            "metrics": {"log_likelihood": -5.0, "avg_log_likelihood": -0.5, "perplexity": float(np.exp(0.5))},
            "vocab": CODONFM_CODON_VOCAB,
        }

    monkeypatch.setattr(
        "proto_tools.tools.masked_models.codonfm.codonfm_gradient.ToolInstance.dispatch",
        staticmethod(fake_dispatch),
    )

    result = run_codonfm_gradient(
        CodonFMGradientInput(logits=[[0.0] * 64] * 3, temperature=0.6),
        CodonFMGradientConfig(model_checkpoint="encodon_80m", use_ste=True, batch_size=2, device="cpu"),
    )

    assert captured["toolkit"] == "codonfm"
    assert captured["payload"]["operation"] == "gradient"
    assert captured["payload"]["logits"] == [[0.0] * 64] * 3
    assert captured["payload"]["temperature"] == 0.6
    assert captured["payload"]["use_ste"] is True
    assert captured["payload"]["compute_gradient"] is True
    assert captured["payload"]["batch_size"] == 2
    assert captured["payload"]["device"] == "cpu"
    assert result.gradient is not None and len(result.gradient) == 3
    assert result.vocab == CODONFM_CODON_VOCAB


def test_codonfm_gradient_forward_mode_dispatch_contract(monkeypatch) -> None:
    """compute_gradient=False forwards the flag and returns gradient=None with the scalar objective."""
    captured: dict[str, object] = {}

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured["payload"] = payload
        return {
            "gradient": None,
            "loss": 0.5,
            "metrics": {"log_likelihood": -5.0, "avg_log_likelihood": -0.5, "perplexity": float(np.exp(0.5))},
            "vocab": CODONFM_CODON_VOCAB,
        }

    monkeypatch.setattr(
        "proto_tools.tools.masked_models.codonfm.codonfm_gradient.ToolInstance.dispatch",
        staticmethod(fake_dispatch),
    )

    result = run_codonfm_gradient(
        CodonFMGradientInput(logits=[[0.0] * 64] * 3),
        CodonFMGradientConfig(compute_gradient=False, device="cpu"),
    )

    assert captured["payload"]["compute_gradient"] is False
    assert result.gradient is None
    assert result.loss == 0.5
    assert result.metrics["avg_log_likelihood"] == -0.5


def test_codonfm_sample_dispatch_contract(monkeypatch) -> None:
    """run_codonfm_sample forwards masking/temperature config and returns the resampled sequences."""
    captured: dict[str, object] = {}

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured["toolkit"] = toolkit
        captured["payload"] = payload
        # Echo one mutated sequence per input (length preserved by the real worker).
        return {"sequences": list(payload["sequences"])}

    monkeypatch.setattr(
        "proto_tools.tools.masked_models.codonfm.codonfm_sample.ToolInstance.dispatch",
        staticmethod(fake_dispatch),
    )

    result = run_codonfm_sample(
        CodonFMSampleInput(sequences=[_CDS, "ATGGCCACC"]),
        CodonFMSampleConfig(
            model_checkpoint="encodon_80m", num_mutations=2, temperature=1.2, batch_size=2, device="cpu"
        ),
    )

    assert captured["toolkit"] == "codonfm"
    assert captured["payload"]["operation"] == "sample"
    assert captured["payload"]["sequences"] == [_CDS, "ATGGCCACC"]
    assert captured["payload"]["num_mutations"] == 2
    assert captured["payload"]["mask_fraction"] == 0.15
    assert captured["payload"]["temperature"] == 1.2
    assert [item.sequence for item in result.results] == [_CDS, "ATGGCCACC"]
    assert list(result.sequences) == [_CDS, "ATGGCCACC"]
    assert len(result) == 2

    with pytest.raises(ValueError, match="exceeds"):
        run_codonfm_sample(
            CodonFMSampleInput(sequences=["ATG"]),
            CodonFMSampleConfig(num_mutations=2, device="cpu"),
        )


# ---------------------------------------------------------------------------
# Integration (real model — GPU + public checkpoint)
# ---------------------------------------------------------------------------
@pytest.mark.uses_gpu
@pytest.mark.slow
def test_codonfm_fitness_real_gpu() -> None:
    """Real GPU smoke test for CodonFM fitness through the tool worker."""
    _skip_if_no_gpu()

    result = run_codonfm_fitness(
        CodonFMFitnessInput(sequences=[_CDS, "ATGGCCACCGTG"]),
        CodonFMFitnessConfig(model_checkpoint="encodon_80m", batch_size=2, device="cuda"),
    )
    validate_output(result)
    assert result.tool_id == "codonfm-fitness"
    assert len(result.results) == 2
    assert all(math.isfinite(r.fitness) for r in result.results)
    assert_metrics_in_spec(result)


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_codonfm_embeddings_real_gpu() -> None:
    """Real GPU smoke test for CodonFM CLS embeddings."""
    _skip_if_no_gpu()

    result = run_codonfm_embeddings(
        CodonFMEmbeddingsInput(sequences=[_CDS]),
        CodonFMEmbeddingsConfig(model_checkpoint="encodon_80m", device="cuda"),
    )
    assert result.tool_id == "codonfm-embedding"
    embedding = result.results[0].embedding
    assert len(embedding) > 0
    assert all(math.isfinite(v) for v in embedding)


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_codonfm_score_real_gpu() -> None:
    """Real GPU smoke test for CodonFM mutation scoring (a synonymous vs missense substitution)."""
    _skip_if_no_gpu()

    result = run_codonfm_score(
        CodonFMScoreInput(
            mutations=[
                {"sequence": _CDS, "codon_position": 2, "ref_codon": "GTG", "alt_codon": "GTA"},  # synonymous
                {"sequence": _CDS, "codon_position": 2, "ref_codon": "GTG", "alt_codon": "AAA"},  # missense
            ]
        ),
        CodonFMScoreConfig(model_checkpoint="encodon_80m", batch_size=2, device="cuda"),
    )
    assert len(result.results) == 2
    for row in result.results:
        assert math.isfinite(row.ref_log_likelihood)
        assert math.isfinite(row.alt_log_likelihood)
        assert row.llr == pytest.approx(row.ref_log_likelihood - row.alt_log_likelihood, rel=1e-5)


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_codonfm_gradient_real_gpu() -> None:
    """Real GPU smoke test for the differentiable masked-PLL codon objective."""
    _skip_if_no_gpu()

    result = run_codonfm_gradient(
        CodonFMGradientInput(logits=one_hot_codon_logits(_CDS, sharpness=2.0), temperature=0.6),
        CodonFMGradientConfig(model_checkpoint="encodon_80m", batch_size=4, device="cuda"),
    )
    validate_output(result)
    assert result.tool_id == "codonfm-gradient"
    assert result.gradient is not None
    assert len(result.gradient) == len(_CDS) // 3
    assert all(len(row) == 64 for row in result.gradient)
    assert all(math.isfinite(v) for row in result.gradient for v in row)
    assert any(v != 0.0 for row in result.gradient for v in row)
    assert result.loss > 0
    assert result.metrics["avg_log_likelihood"] == pytest.approx(-result.loss, rel=1e-6)
    assert result.vocab == CODONFM_CODON_VOCAB


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_codonfm_sample_real_gpu() -> None:
    """Real GPU smoke test: masked-codon resampling preserves length and stays codon-aligned."""
    _skip_if_no_gpu()

    result = run_codonfm_sample(
        CodonFMSampleInput(sequences=[_CDS]),
        CodonFMSampleConfig(model_checkpoint="encodon_80m", num_mutations=3, temperature=1.0, device="cuda", seed=0),
    )
    assert len(result.sequences) == 1
    sampled = result.sequences[0]
    assert len(sampled) == len(_CDS)  # length preserved
    assert len(sampled) % 3 == 0
    assert set(sampled) <= set("ACGT")


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_codonfm_gradient_forward_mode_matches_backward_loss() -> None:
    """compute_gradient=False keeps the scalar masked-PLL objective identical to the backward pass."""
    _skip_if_no_gpu()

    inputs = CodonFMGradientInput(logits=one_hot_codon_logits(_CDS, sharpness=2.0), temperature=0.6)
    backward = run_codonfm_gradient(
        inputs, CodonFMGradientConfig(model_checkpoint="encodon_80m", batch_size=4, seed=42, device="cuda")
    )
    forward = run_codonfm_gradient(
        inputs,
        CodonFMGradientConfig(
            model_checkpoint="encodon_80m", batch_size=4, seed=42, compute_gradient=False, device="cuda"
        ),
    )
    assert backward.gradient is not None and forward.gradient is None
    assert forward.loss == pytest.approx(backward.loss, rel=1e-6)


@pytest.mark.benchmark("codonfm-fitness")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_codonfm_fitness_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark codonfm-fitness on a batch of length-300 CDS (cold + warm)."""
    from tests.conftest import random_dna_sequences

    sequences = [s[: 300 - (len(s) % 3)] for s in random_dna_sequences(n=512, length=300, seed=0)]
    inputs = CodonFMFitnessInput(sequences=sequences)
    config = CodonFMFitnessConfig(model_checkpoint="encodon_80m", batch_size=32)

    result = benchmark_twice(request, "codonfm", lambda: run_codonfm_fitness(inputs, config))
    validate_output(result)
    assert result.tool_id == "codonfm-fitness"
    assert len(result.results) == 512


@pytest.mark.benchmark("codonfm-embedding")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_codonfm_embedding_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark codonfm-embedding on 100 length-300 coding sequences (cold + warm)."""
    sequences = ["ATG" * 100 for _ in range(100)]
    inputs = CodonFMEmbeddingsInput(sequences=sequences)
    config = CodonFMEmbeddingsConfig(model_checkpoint="encodon_80m", batch_size=16)

    result = benchmark_twice(request, "codonfm", lambda: run_codonfm_embeddings(inputs, config))
    validate_output(result)
    assert result.tool_id == "codonfm-embedding"
    assert len(result.results) == 100


@pytest.mark.benchmark("codonfm-score")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_codonfm_score_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark codonfm-score on 100 mutations in a length-300 coding sequence (cold + warm)."""
    sequence = "ATG" * 100
    inputs = CodonFMScoreInput(
        mutations=[
            {"sequence": sequence, "codon_position": position, "ref_codon": "ATG", "alt_codon": "GCC"}
            for position in range(1, 101)
        ]
    )
    config = CodonFMScoreConfig(model_checkpoint="encodon_80m", batch_size=16)

    result = benchmark_twice(request, "codonfm", lambda: run_codonfm_score(inputs, config))
    validate_output(result)
    assert result.tool_id == "codonfm-score"
    assert len(result.results) == 100


@pytest.mark.benchmark("codonfm-gradient")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_codonfm_gradient_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark codonfm-gradient on a length-100-codon relaxed sequence (cold + warm)."""
    inputs = CodonFMGradientInput(logits=one_hot_codon_logits("ATG" * 100, sharpness=2.0), temperature=0.6)
    config = CodonFMGradientConfig(model_checkpoint="encodon_80m", batch_size=32)

    result = benchmark_twice(request, "codonfm", lambda: run_codonfm_gradient(inputs, config))
    validate_output(result)
    assert result.tool_id == "codonfm-gradient"
    assert result.gradient is not None and len(result.gradient) == 100


@pytest.mark.benchmark("codonfm-sample")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_codonfm_sample_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark codonfm-sample on 50 length-300 coding sequences (cold + warm)."""
    sequences = ["ATG" * 100 for _ in range(50)]
    inputs = CodonFMSampleInput(sequences=sequences)
    config = CodonFMSampleConfig(model_checkpoint="encodon_80m", num_mutations=10, batch_size=16, seed=0)

    result = benchmark_twice(request, "codonfm", lambda: run_codonfm_sample(inputs, config))
    validate_output(result)
    assert result.tool_id == "codonfm-sample"
    assert len(result.sequences) == 50
