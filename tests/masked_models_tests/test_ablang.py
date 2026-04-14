"""Tests for AbLang antibody language model tools."""

import math
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from proto_tools.tools.masked_models.ablang import (
    AbLangEmbeddingsConfig,
    AbLangEmbeddingsInput,
    AbLangGradientConfig,
    AbLangGradientInput,
    AbLangSampleConfig,
    AbLangSampleInput,
    AbLangScoringConfig,
    AbLangScoringInput,
    run_ablang_embeddings,
    run_ablang_gradient,
    run_ablang_sample,
    run_ablang_score,
)
from proto_tools.utils.sequence import PROTEIN_AMINO_ACIDS
from tests.conftest import make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import validate_output

_VALID_AAS = set(PROTEIN_AMINO_ACIDS)
_CANONICAL_VOCAB = list(PROTEIN_AMINO_ACIDS)

# AbLang's amino acid ordering, from https://github.com/TobiasHeOl/AbLang2/blob/main/ablang2/models/ablang2/vocab.py
ABLANG_VOCAB_ORDER = list("ARNDCQEGHILKMFPSTWYV")
_ABLANG_VOCAB_SIZE = len(ABLANG_VOCAB_ORDER) + 1  # +1 for "|" separator
_PIPE_TOKEN_ID = len(ABLANG_VOCAB_ORDER)


def _import_ablang_inference():
    """Lazy import of the standalone inference module (requires standalone_helpers stub)."""
    sys.modules.setdefault("standalone_helpers", types.SimpleNamespace(serialize_output=lambda value: value))
    from proto_tools.tools.masked_models.ablang.standalone import inference as ablang_inference

    return ablang_inference


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


def test_ablang_gradient_input_requires_20_aa_columns():
    logits = [[0.0] * 20, [1.0] * 20]
    inp = AbLangGradientInput(logits=logits, temperature=0.5)
    assert inp.logits == logits

    with pytest.raises(ValidationError, match="20 columns"):
        AbLangGradientInput(logits=[[0.0] * 19], temperature=1.0)


def test_ablang_gradient_config_validates_paired_config():
    config = AbLangGradientConfig(
        model_choice="ablang2-paired",
        chain_break_position=4,
    )
    assert config.chain_break_position == 4

    with pytest.raises(
        ValidationError,
        match="chain_break_position is required when model_choice='ablang2-paired'",
    ):
        AbLangGradientConfig(model_choice="ablang2-paired")

    with pytest.raises(
        ValidationError,
        match="chain_break_position is only supported when model_choice='ablang2-paired'",
    ):
        AbLangGradientConfig(model_choice="ablang1-heavy", chain_break_position=4)


def test_ablang_gradient_dispatch_contract(monkeypatch):
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
        "proto_tools.tools.masked_models.ablang.ablang_gradient.ToolInstance.dispatch",
        fake_dispatch,
    )

    inputs = AbLangGradientInput(logits=[[0.0] * 20, [1.0] * 20], temperature=0.8)
    config = AbLangGradientConfig(
        model_choice="ablang2-paired",
        chain_break_position=1,
        seed=17,
        device="cpu",
    )
    result = run_ablang_gradient(inputs=inputs, config=config)

    validate_output(result)
    assert captured["tool_name"] == "ablang"
    assert captured["payload"] == {
        "operation": "compute_gradient",
        "logits": inputs.logits,
        "temperature": 0.8,
        "model_choice": "ablang2-paired",
        "chain_break_position": 1,
        "seed": 17,
        "device": "cpu",
        "verbose": False,
    }
    assert result.gradient == [[0.1] * 20, [0.2] * 20]
    assert result.loss == 0.75
    assert result.metrics["log_likelihood"] == -0.75
    assert result.vocab == _CANONICAL_VOCAB


# ── Gradient unit tests (CPU, fake model) ────────────────────────────────────


class _FakeHookHandle:
    def __init__(self, layer: "_FakeEmbedLayer") -> None:
        self.layer = layer

    def remove(self) -> None:
        self.layer.hook = None


class _FakeEmbedLayer:
    def __init__(self, weight: Any) -> None:
        self.weight = weight
        self.hook = None

    def register_forward_hook(self, hook):
        self.hook = hook
        return _FakeHookHandle(self)

    def __call__(self, token_ids: Any) -> Any:
        output = self.weight[token_ids]
        if self.hook is not None:
            return self.hook(self, (token_ids,), output)
        return output


class _FakeAbLangPaired:
    def __init__(self, embed_layer: _FakeEmbedLayer, projection: Any) -> None:
        self.embed_layer = embed_layer
        self.projection = projection

    def get_aa_embeddings(self) -> _FakeEmbedLayer:
        return self.embed_layer

    def __call__(self, token_ids: Any) -> Any:
        import torch

        embeddings = self.embed_layer(token_ids)
        return torch.einsum("bld,df->blf", embeddings, self.projection)


def _build_fake_paired_model() -> tuple:
    import torch

    ablang_inference = _import_ablang_inference()

    weight = torch.arange(_ABLANG_VOCAB_SIZE * 3, dtype=torch.float32).reshape(_ABLANG_VOCAB_SIZE, 3) / 10.0
    projection = torch.arange(3 * _ABLANG_VOCAB_SIZE, dtype=torch.float32).reshape(3, _ABLANG_VOCAB_SIZE) / 7.0
    embed_layer = _FakeEmbedLayer(weight)

    model = ablang_inference.AbLangModel(model_choice="ablang2-paired")
    model._loaded = True
    model.device = "cpu"
    model.model = SimpleNamespace(AbLang=_FakeAbLangPaired(embed_layer, projection))
    model._ablang_vocab = {aa: idx for idx, aa in enumerate(ABLANG_VOCAB_ORDER)} | {"|": _PIPE_TOKEN_ID}
    return model, weight, projection


def _manual_expected_gradient(
    logits_list: list[list[float]],
    *,
    chain_break_position: int | None,
    weight: Any,
    projection: Any,
) -> tuple:
    """Compute the expected gradient by manually reproducing the math."""
    import torch
    import torch.nn.functional as F

    seq_dist = torch.tensor(logits_list, dtype=torch.float32, requires_grad=True)

    mapping = torch.zeros((len(_CANONICAL_VOCAB), _ABLANG_VOCAB_SIZE), dtype=torch.float32)
    for idx, aa in enumerate(_CANONICAL_VOCAB):
        mapping[idx, ABLANG_VOCAB_ORDER.index(aa)] = 1.0
    mapped = seq_dist @ mapping

    token_ids = mapped.argmax(dim=-1).detach()
    residue_embeddings = mapped @ weight

    if chain_break_position is not None:
        residue_embeddings = torch.cat(
            [
                residue_embeddings[:chain_break_position],
                weight[_PIPE_TOKEN_ID].unsqueeze(0),
                residue_embeddings[chain_break_position:],
            ],
            dim=0,
        )
        token_ids = torch.cat(
            [
                token_ids[:chain_break_position],
                torch.tensor([_PIPE_TOKEN_ID], dtype=torch.long),
                token_ids[chain_break_position:],
            ],
            dim=0,
        )

    logits_out = torch.einsum("bld,df->blf", residue_embeddings.unsqueeze(0), projection)
    shift_logits = logits_out[:, :-1, :]
    shift_labels = token_ids.unsqueeze(0)[:, 1:]
    position_losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    ).reshape(shift_labels.shape)
    loss = position_losses[:, 1:-1].mean()
    (gradient,) = torch.autograd.grad(loss, seq_dist)

    return gradient, loss.item()


_UNIT_LOGITS = [
    [0.2 + i / 50.0 for i in range(20)],
    [0.5 - i / 60.0 for i in range(20)],
    [0.1 + i / 70.0 for i in range(20)],
    [0.3 - i / 80.0 for i in range(20)],
    [0.4 + i / 90.0 for i in range(20)],
    [0.6 - i / 100.0 for i in range(20)],
    [0.7 + i / 110.0 for i in range(20)],
]


@pytest.mark.parametrize("chain_break_position", [2, None])
def test_compute_gradient_matches_objective_math(chain_break_position: int | None) -> None:
    """Validate the exact shifted-cross-entropy math with and without chain separator."""
    torch = pytest.importorskip("torch")
    model, weight, projection = _build_fake_paired_model()

    result = model.compute_gradient(
        logits_list=_UNIT_LOGITS,
        chain_break_position=chain_break_position,
        seed=None,
        device="cpu",
    )

    expected_gradient, expected_loss = _manual_expected_gradient(
        _UNIT_LOGITS,
        chain_break_position=chain_break_position,
        weight=weight,
        projection=projection,
    )

    result_gradient = torch.tensor(result["gradient"], dtype=torch.float32)
    torch.testing.assert_close(result_gradient, expected_gradient, rtol=1e-5, atol=1e-6)
    assert result["loss"] == pytest.approx(expected_loss, rel=1e-6)
    assert result["metrics"]["log_likelihood"] == pytest.approx(-expected_loss, rel=1e-6)
    assert result["metrics"]["sequence_length"] == len(_UNIT_LOGITS)
    assert result["vocab"] == _CANONICAL_VOCAB


@pytest.mark.parametrize(
    ("model_choice", "expected_model_choice"),
    [("ablang1-heavy", "ablang1-heavy"), ("ablang2-paired", "ablang2-paired")],
)
def test_dispatch_routes_gradient_to_expected_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    model_choice: str,
    expected_model_choice: str,
) -> None:
    """Dispatch should instantiate the requested model variant."""
    created_model_choices: list[str] = []

    class _FakeDispatchedModel:
        def __init__(self, model_choice: str) -> None:
            created_model_choices.append(model_choice)
            self.model_choice = model_choice
            self._loaded = False

        def compute_gradient(self, **_kwargs):
            return {
                "gradient": [[0.0] * 20],
                "loss": 0.0,
                "metrics": {"model_choice": self.model_choice},
                "vocab": list(PROTEIN_AMINO_ACIDS),
            }

    ablang_inference = _import_ablang_inference()
    monkeypatch.setattr(ablang_inference, "AbLangModel", _FakeDispatchedModel)
    monkeypatch.setattr(ablang_inference, "_model", None)

    result = ablang_inference.dispatch(
        {
            "operation": "compute_gradient",
            "logits": [[0.0] * 20],
            "temperature": 1.0,
            "model_choice": model_choice,
            "chain_break_position": 1 if model_choice == "ablang2-paired" else None,
            "seed": None,
            "device": "cpu",
            "verbose": False,
        }
    )

    assert created_model_choices == [expected_model_choice]
    assert result["metrics"]["model_choice"] == expected_model_choice


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


# ── Gradient tests ──────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_ablang_gradient_single_chain():
    """Test single-chain gradient: shape, finiteness, and metric consistency."""
    seq_len = 10
    result = run_ablang_gradient(
        AbLangGradientInput(logits=[[0.0] * 20] * seq_len, temperature=0.6),
        AbLangGradientConfig(model_choice="ablang1-heavy"),
    )
    validate_output(result)

    assert result.tool_id == "ablang-gradient"
    assert len(result.gradient) == seq_len
    assert all(len(row) == 20 for row in result.gradient)
    assert all(math.isfinite(v) for row in result.gradient for v in row)
    assert any(v != 0.0 for row in result.gradient for v in row)
    assert math.isfinite(result.loss) and result.loss > 0
    assert result.vocab == _CANONICAL_VOCAB
    assert result.metrics["sequence_length"] == seq_len
    assert result.metrics["model_choice"] == "ablang1-heavy"
    assert result.metrics["objective"] == "shifted_cross_entropy"
    assert result.metrics["log_likelihood"] == pytest.approx(-result.loss, rel=1e-6)


@pytest.mark.uses_gpu
def test_ablang_gradient_paired_with_chain_break():
    """Test paired-chain gradient with chain_break_position."""
    heavy_len = 4
    light_len = 5
    seq_len = heavy_len + light_len

    result = run_ablang_gradient(
        AbLangGradientInput(logits=[[0.0] * 20] * seq_len, temperature=0.6),
        AbLangGradientConfig(
            model_choice="ablang2-paired",
            chain_break_position=heavy_len,
        ),
    )
    validate_output(result)

    assert result.tool_id == "ablang-gradient"
    assert len(result.gradient) == seq_len
    assert all(len(row) == 20 for row in result.gradient)
    assert all(math.isfinite(v) for row in result.gradient for v in row)
    assert any(v != 0.0 for row in result.gradient for v in row)
    assert result.vocab == _CANONICAL_VOCAB
    assert result.metrics["sequence_length"] == seq_len
    assert result.metrics["model_choice"] == "ablang2-paired"
    assert result.metrics["objective"] == "shifted_cross_entropy"


@pytest.mark.uses_gpu
def test_ablang_gradient_descent_reduces_loss():
    """Taking a step in the negative gradient direction should reduce the loss."""
    seq_len = 10
    initial_logits = [[0.0] * 20] * seq_len

    result_0 = run_ablang_gradient(
        AbLangGradientInput(logits=initial_logits, temperature=0.7),
        AbLangGradientConfig(model_choice="ablang1-heavy", seed=42),
    )

    lr = 10.0
    stepped_logits = [[initial_logits[i][j] - lr * result_0.gradient[i][j] for j in range(20)] for i in range(seq_len)]

    result_1 = run_ablang_gradient(
        AbLangGradientInput(logits=stepped_logits, temperature=0.7),
        AbLangGradientConfig(model_choice="ablang1-heavy", seed=42),
    )

    assert result_1.loss < result_0.loss


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
