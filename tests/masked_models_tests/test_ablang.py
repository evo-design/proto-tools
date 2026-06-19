"""Tests for AbLang antibody language model tools."""

import math
import random
import re
import sys
import types

import pytest
from pydantic import ValidationError

from proto_tools.entities.antibody import Antibody, AntibodyLogits
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
from proto_tools.utils.sequence import PROTEIN_AMINO_ACIDS, one_hot_protein_logits
from tests.conftest import benchmark_twice, make_persistent_fixture
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output

_VALID_AAS = set(PROTEIN_AMINO_ACIDS)
_CANONICAL_VOCAB = list(PROTEIN_AMINO_ACIDS)


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


def test_ablang_gradient_input_validation():
    inp = AbLangGradientInput(antibody=AntibodyLogits(heavy_chain=[[0.0] * 20, [1.0] * 20]))
    assert inp.antibody.heavy_chain == [[0.0] * 20, [1.0] * 20]
    assert inp.temperature is None

    inp_with_temp = AbLangGradientInput(antibody=AntibodyLogits(heavy_chain=[[0.0] * 20]), temperature=0.6)
    assert inp_with_temp.temperature == 0.6

    with pytest.raises(ValidationError):
        AbLangGradientInput(antibody=AntibodyLogits(heavy_chain=[[0.0] * 20]), temperature=-0.5)
    with pytest.raises(ValidationError):
        AbLangGradientInput(antibody=AntibodyLogits(heavy_chain=[[0.0] * 20]), temperature=0.0)

    with pytest.raises(ValidationError, match="20 columns"):
        AntibodyLogits(heavy_chain=[[0.0] * 19])
    with pytest.raises(ValidationError, match="20 columns"):
        AntibodyLogits(heavy_chain=[[0.0] * 20], light_chain=[[0.0] * 19])
    with pytest.raises(ValidationError, match="At least one"):
        AntibodyLogits()

    paired = AntibodyLogits(heavy_chain=[[0.0] * 20] * 3, light_chain=[[0.0] * 20] * 2)
    assert len(paired.heavy_chain) == 3
    assert len(paired.light_chain) == 2


def test_ablang_gradient_dispatch_contract(monkeypatch):
    captured: dict[str, object] = {}

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured["payload"] = payload
        n = len(payload.get("logits", []))
        return {
            "gradient": [[0.0] * 20] * n,
            "loss": 0.5,
            "metrics": {
                "log_likelihood": -0.5 * n,
                "avg_log_likelihood": -0.5,
                "perplexity": math.exp(0.5),
                "model_choice": payload.get("model_choice"),
            },
            "vocab": list(PROTEIN_AMINO_ACIDS),
        }

    monkeypatch.setattr(
        "proto_tools.tools.masked_models.ablang.ablang_gradient.ToolInstance.dispatch",
        fake_dispatch,
    )

    heavy, light = [[0.0] * 20], [[1.0] * 20]

    # Paired: chains concatenated, model auto-selected
    run_ablang_gradient(AbLangGradientInput(antibody=AntibodyLogits(heavy_chain=heavy, light_chain=light)))
    assert captured["payload"]["logits"] == heavy + light
    assert captured["payload"]["chain_break_position"] == 1
    assert captured["payload"]["model_choice"] == "ablang2-paired"
    assert captured["payload"]["temperature"] is None
    assert captured["payload"]["use_ste"] is False
    assert captured["payload"]["compute_gradient"] is True  # default
    assert captured["payload"]["batch_size"] is None  # default auto

    # Heavy only, with temperature
    run_ablang_gradient(AbLangGradientInput(antibody=AntibodyLogits(heavy_chain=heavy), temperature=0.6))
    assert captured["payload"]["logits"] == heavy
    assert captured["payload"]["chain_break_position"] is None
    assert captured["payload"]["model_choice"] == "ablang1-heavy"
    assert captured["payload"]["temperature"] == 0.6

    # Light only
    run_ablang_gradient(AbLangGradientInput(antibody=AntibodyLogits(light_chain=light)))
    assert captured["payload"]["logits"] == light
    assert captured["payload"]["chain_break_position"] is None
    assert captured["payload"]["model_choice"] == "ablang1-light"


def test_ablang_forward_mode_dispatch_contract(monkeypatch):
    """compute_gradient=False forwards the flag and returns gradient=None."""
    captured: dict[str, object] = {}

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured.update(payload=payload)
        n = len(payload.get("logits", []))
        return {
            "gradient": None,
            "loss": 0.5,
            "metrics": {
                "log_likelihood": -0.5 * n,
                "avg_log_likelihood": -0.5,
                "perplexity": math.exp(0.5),
                "sequence_length": n,
            },
            "vocab": list(PROTEIN_AMINO_ACIDS),
        }

    monkeypatch.setattr(
        "proto_tools.tools.masked_models.ablang.ablang_gradient.ToolInstance.dispatch",
        fake_dispatch,
    )

    result = run_ablang_gradient(
        AbLangGradientInput(antibody=AntibodyLogits(heavy_chain=[[0.0] * 20] * 3)),
        AbLangGradientConfig(compute_gradient=False),
    )

    assert captured["payload"]["compute_gradient"] is False
    assert result.gradient is None
    assert result.loss == 0.5
    assert result.metrics["avg_log_likelihood"] == -0.5


# ── Gradient dispatch tests (subprocess venv, GPU-only) ──────────────────────


_UNIT_LOGITS = [
    [0.2 + i / 50.0 for i in range(20)],
    [0.5 - i / 60.0 for i in range(20)],
    [0.1 + i / 70.0 for i in range(20)],
    [0.3 - i / 80.0 for i in range(20)],
    [0.4 + i / 90.0 for i in range(20)],
    [0.6 - i / 100.0 for i in range(20)],
    [0.7 + i / 110.0 for i in range(20)],
]


@pytest.mark.uses_gpu
@pytest.mark.parametrize("paired", [True, False], ids=["paired", "single"])
@pytest.mark.parametrize(
    ("temperature", "use_ste"),
    [(None, False), (0.6, False), (None, True), (0.6, True)],
    ids=["default", "temp_only", "ste_only", "temp_and_ste"],
)
def test_compute_gradient_dispatch(
    paired: bool,
    temperature: float | None,
    use_ste: bool,
) -> None:
    """Validate compute_gradient across temperature/STE/chain modes via real dispatch."""
    if paired:
        antibody = AntibodyLogits(heavy_chain=_UNIT_LOGITS[:2], light_chain=_UNIT_LOGITS[2:])
        expected_len = len(_UNIT_LOGITS)
    else:
        antibody = AntibodyLogits(heavy_chain=_UNIT_LOGITS)
        expected_len = len(_UNIT_LOGITS)

    result = run_ablang_gradient(
        AbLangGradientInput(antibody=antibody, temperature=temperature),
        AbLangGradientConfig(use_ste=use_ste),
    )
    validate_output(result)

    assert result.gradient is not None
    assert len(result.gradient) == expected_len
    assert all(len(row) == 20 for row in result.gradient)
    assert all(math.isfinite(v) for row in result.gradient for v in row)
    assert result.loss > 0
    assert math.isfinite(result.loss)
    assert result.metrics["avg_log_likelihood"] == pytest.approx(-result.loss, rel=1e-6)
    assert result.metrics["log_likelihood"] == pytest.approx(-result.loss * expected_len, rel=1e-6)
    assert result.metrics["perplexity"] == pytest.approx(math.exp(result.loss), rel=1e-6)
    assert result.metrics["sequence_length"] == expected_len
    assert result.vocab == _CANONICAL_VOCAB


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
            "temperature": None,
            "use_ste": False,
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
        AbLangEmbeddingsInput(antibodies=[Antibody(heavy_chain=s) for s in seqs]),
        AbLangEmbeddingsConfig(batch_size=2),
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
        AbLangEmbeddingsInput(antibodies=[Antibody(light_chain=VL_SEQ)]),
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
        AbLangEmbeddingsInput(antibodies=[Antibody(heavy_chain=HEAVY_FULL, light_chain=LIGHT_FULL)]),
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
        AbLangEmbeddingsInput(antibodies=[Antibody(heavy_chain=VH_SEQ)]),
    )
    assert single_result.success
    assert single_result.metadata["model_choice"] == "ablang1-heavy"
    assert len(single_result.results[0].mean_embedding) == 768

    paired_result = run_ablang_embeddings(
        AbLangEmbeddingsInput(antibodies=[Antibody(heavy_chain=HEAVY_FULL, light_chain=LIGHT_FULL)]),
    )
    assert paired_result.success
    assert paired_result.metadata["model_choice"] == "ablang2-paired"
    assert len(paired_result.results[0].mean_embedding) == 480


# ── Cross-tool consistency ──────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_ablang_score_and_gradient_agree_on_pll():
    """ablang-score (string input) and ablang-gradient (one-hot logits, backprop=False) return the same mean NLL."""
    sequence = "EVQLVESGGGLVQPGGSLRL"
    aa_order = list("ACDEFGHIKLMNPQRSTVWY")

    score_result = run_ablang_score(
        AbLangScoringInput(antibodies=[Antibody(heavy_chain=sequence)]),
        AbLangScoringConfig(scoring_mode="pseudo_log_likelihood"),
    )
    score_pll = score_result.scores[0]["avg_log_likelihood"]

    one_hot = [[10.0 if aa_order[j] == aa else 0.0 for j in range(20)] for aa in sequence]
    grad_result = run_ablang_gradient(
        AbLangGradientInput(antibody=AntibodyLogits(heavy_chain=one_hot), temperature=0.6),
        AbLangGradientConfig(use_ste=True, compute_gradient=False),
    )
    grad_pll = grad_result.metrics["avg_log_likelihood"]

    assert score_pll == pytest.approx(grad_pll, rel=1e-3)


# ── Scoring tests ────────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_ablang_score_heavy():
    """Test ablang1-heavy PLL: natural VH scores higher than poly-A."""
    result = run_ablang_score(
        AbLangScoringInput(antibodies=[Antibody(heavy_chain=VH_SEQ), Antibody(heavy_chain="A" * 20)]),
        AbLangScoringConfig(scoring_mode="pseudo_log_likelihood"),
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
        AbLangScoringInput(antibodies=[Antibody(light_chain=VL_SEQ), Antibody(light_chain="A" * 20)]),
        AbLangScoringConfig(scoring_mode="pseudo_log_likelihood"),
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
        AbLangScoringInput(antibodies=[Antibody(heavy_chain=HEAVY_FULL, light_chain=LIGHT_FULL)]),
        AbLangScoringConfig(scoring_mode="pseudo_log_likelihood"),
    )
    validate_output(result)

    pll = result.scores[0]["pseudo_log_likelihood"]
    assert math.isfinite(pll) and pll < 0


@pytest.mark.uses_gpu
def test_ablang_score_return_logits():
    """``return_logits=True`` populates per-position AA logits; ``vocab`` is always populated."""
    inputs = AbLangScoringInput(antibodies=[Antibody(heavy_chain=VH_SEQ), Antibody(heavy_chain=VH_SEQ[:50])])

    default = run_ablang_score(inputs, AbLangScoringConfig())
    validate_output(default)
    assert all(s.logits is None for s in default.scores)
    assert all(s.vocab == list(PROTEIN_AMINO_ACIDS) for s in default.scores)

    with_logits = run_ablang_score(inputs, AbLangScoringConfig(return_logits=True))
    validate_output(with_logits)
    assert with_logits.vocab is not None and len(with_logits.vocab) == 20
    for score in with_logits.scores:
        assert score.logits is not None
        assert all(len(row) == 20 for row in score.logits)
        assert all(math.isfinite(v) for row in score.logits for v in row)
        assert score.vocab == list(PROTEIN_AMINO_ACIDS)


@pytest.mark.uses_gpu
def test_ablang_sample_return_logits():
    """``return_logits=True`` returns per-position AA logits over the restored sequences."""
    masked = [
        Antibody(heavy_chain=VH_SEQ[:4] + "_" + VH_SEQ[5:]),
        Antibody(heavy_chain=VH_SEQ[:50]),
    ]
    inputs = AbLangSampleInput(antibodies=masked)

    default = run_ablang_sample(inputs, AbLangSampleConfig())
    validate_output(default)
    assert default.logits is None

    with_logits = run_ablang_sample(inputs, AbLangSampleConfig(return_logits=True))
    validate_output(with_logits)
    assert with_logits.logits is not None
    assert len(with_logits.logits) == len(masked)
    for restored, seq_logits in zip(with_logits.sequences, with_logits.logits, strict=True):
        # Restored output rows include format-time special tokens; AA columns are 20.
        assert all(len(row) == 20 for row in seq_logits)
        assert all(math.isfinite(v) for row in seq_logits for v in row)
        assert len(seq_logits) >= len(restored)


@pytest.mark.uses_gpu
def test_ablang_score_confidence_mode():
    """Test confidence scoring returns finite scores for all chain configurations."""
    for ab in [
        Antibody(heavy_chain=VH_SEQ),
        Antibody(light_chain=VL_SEQ),
        Antibody(heavy_chain=HEAVY_FULL, light_chain=LIGHT_FULL),
    ]:
        result = run_ablang_score(
            AbLangScoringInput(antibodies=[ab]),
            AbLangScoringConfig(scoring_mode="confidence"),
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
        AbLangSampleInput(antibodies=[Antibody(heavy_chain=masked_seq)]),
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
        AbLangSampleInput(antibodies=[Antibody(light_chain=masked_seq)]),
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
    result = run_ablang_sample(
        AbLangSampleInput(antibodies=[Antibody(heavy_chain=masked_heavy, light_chain=LIGHT_FULL)]),
    )
    validate_output(result)

    restored = result.sequences[0]
    residues_only = restored.replace("|", "")
    assert "_" not in restored
    assert len(restored) == len(masked_heavy) + 1 + len(LIGHT_FULL)
    assert set(residues_only) <= _VALID_AAS
    assert restored[:mask_pos] == HEAVY_FULL[:mask_pos]
    assert restored[mask_pos + 1 : len(HEAVY_FULL)] == HEAVY_FULL[mask_pos + 1 :]


# ── Gradient tests ──────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_ablang_gradient_single_chain():
    """Test single-chain gradient: shape, finiteness, and metric consistency."""
    seq_len = 10
    result = run_ablang_gradient(
        AbLangGradientInput(antibody=AntibodyLogits(heavy_chain=[[0.0] * 20] * seq_len), temperature=0.6),
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
    assert result.metrics["objective"] == "masked_pll"
    assert result.metrics["avg_log_likelihood"] == pytest.approx(-result.loss, rel=1e-6)


@pytest.mark.uses_gpu
def test_ablang_gradient_paired_chains():
    """Test paired-chain gradient with separate heavy and light chain inputs."""
    heavy_len = 4
    light_len = 5

    result = run_ablang_gradient(
        AbLangGradientInput(
            antibody=AntibodyLogits(heavy_chain=[[0.0] * 20] * heavy_len, light_chain=[[0.0] * 20] * light_len),
            temperature=0.6,
        ),
    )
    validate_output(result)

    assert result.tool_id == "ablang-gradient"
    assert len(result.gradient) == heavy_len + light_len
    assert all(len(row) == 20 for row in result.gradient)
    assert all(math.isfinite(v) for row in result.gradient for v in row)
    assert any(v != 0.0 for row in result.gradient for v in row)
    assert result.vocab == _CANONICAL_VOCAB
    assert result.metrics["sequence_length"] == heavy_len + light_len
    assert result.metrics["model_choice"] == "ablang2-paired"
    assert result.metrics["objective"] == "masked_pll"


@pytest.mark.uses_gpu
def test_ablang_gradient_forward_mode_matches_backward_loss():
    """Forward and backward use the same masked PLL objective — losses must match."""
    seq_len = 10
    logits = [[0.0] * 20] * seq_len
    ab = AntibodyLogits(heavy_chain=logits)

    backward = run_ablang_gradient(
        AbLangGradientInput(antibody=ab, temperature=0.6),
        AbLangGradientConfig(seed=42),
    )
    forward = run_ablang_gradient(
        AbLangGradientInput(antibody=ab, temperature=0.6),
        AbLangGradientConfig(seed=42, compute_gradient=False),
    )
    validate_output(backward)
    validate_output(forward)

    assert backward.gradient is not None
    assert forward.gradient is None
    assert forward.loss == pytest.approx(backward.loss, rel=1e-5)
    assert forward.metrics["log_likelihood"] == pytest.approx(backward.metrics["log_likelihood"], rel=1e-5)
    assert forward.metrics["sequence_length"] == backward.metrics["sequence_length"]


@pytest.mark.uses_gpu
def test_ablang_gradient_descent_reduces_loss():
    """Taking a step in the negative gradient direction should reduce the loss."""
    seq_len = 10
    initial_logits = [[0.0] * 20] * seq_len

    result_0 = run_ablang_gradient(
        AbLangGradientInput(antibody=AntibodyLogits(heavy_chain=initial_logits), temperature=0.7),
        AbLangGradientConfig(seed=42),
    )

    lr = 10.0
    stepped_logits = [[initial_logits[i][j] - lr * result_0.gradient[i][j] for j in range(20)] for i in range(seq_len)]

    result_1 = run_ablang_gradient(
        AbLangGradientInput(antibody=AntibodyLogits(heavy_chain=stepped_logits), temperature=0.7),
        AbLangGradientConfig(seed=42),
    )

    assert result_1.loss < result_0.loss


@pytest.mark.uses_gpu
@pytest.mark.parametrize("batch_size", [1, 3, None], ids=["chunk1", "chunk3", "auto"])
def test_ablang_gradient_batch_size_equivalence(batch_size: int | None) -> None:
    """Different chunk sizes must produce equivalent loss and gradient."""
    logits = [[0.1 * (i + j) for j in range(20)] for i in range(5)]
    ab = AntibodyLogits(heavy_chain=logits)

    ref = run_ablang_gradient(
        AbLangGradientInput(antibody=ab, temperature=0.6),
        AbLangGradientConfig(seed=42, batch_size=None),
    )
    result = run_ablang_gradient(
        AbLangGradientInput(antibody=ab, temperature=0.6),
        AbLangGradientConfig(seed=42, batch_size=batch_size),
    )

    assert result.loss == pytest.approx(ref.loss, rel=1e-4, abs=1e-6)
    for row_r, row_ref in zip(result.gradient, ref.gradient, strict=True):
        for v_r, v_ref in zip(row_r, row_ref, strict=True):
            assert v_r == pytest.approx(v_ref, rel=1e-4, abs=1e-6)


# ── Batched tests ────────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_ablang_batched_operations():
    """Test that batch_size > 1 works for embeddings, scoring, and sampling."""
    antibodies = [Antibody(heavy_chain=s) for s in [VH_SEQ, VH_SEQ[:50], VH_SEQ[:80]]]

    emb_result = run_ablang_embeddings(
        AbLangEmbeddingsInput(antibodies=antibodies),
        AbLangEmbeddingsConfig(batch_size=2),
    )
    assert emb_result.success
    assert len(emb_result.results) == 3
    assert all(len(r.mean_embedding) == 768 for r in emb_result.results)

    score_result = run_ablang_score(
        AbLangScoringInput(antibodies=antibodies),
        AbLangScoringConfig(batch_size=2),
    )
    assert score_result.success
    assert len(score_result.scores) == 3
    assert all("pseudo_log_likelihood" in s for s in score_result.scores)

    masked_abs = [Antibody(heavy_chain=s[:4] + "_" + s[5:]) for s in [VH_SEQ, VH_SEQ[:50], VH_SEQ[:80]]]
    sample_result = run_ablang_sample(
        AbLangSampleInput(antibodies=masked_abs),
        AbLangSampleConfig(batch_size=2),
    )
    assert sample_result.success
    assert len(sample_result.sequences) == 3
    assert all("_" not in s for s in sample_result.sequences)


# ── Benchmark fixtures ──────────────────────────────────────────────────────
# Real therapeutic antibody variable-domain sequences (publicly documented).
# Used to scale benchmark inputs without sacrificing antibody-shape realism —
# every input is a real-shaped antibody, optionally diversified by mutating
# CDR3 (the most variable region in real antibody libraries).

# Trastuzumab (Herceptin) — anti-HER2 — VH and VL.
_VH_TRASTUZUMAB = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
_VL_TRASTUZUMAB = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)

# Pertuzumab (Perjeta) — anti-HER2 — VH and VL.
_VH_PERTUZUMAB = "EVQLVESGGGLVQPGGSLRLSCAASGFTFTDYTMDWVRQAPGKGLEWVADVNPNSGGSIYNQRFKGRFTLSVDRSKNTLYLQMNSLRAEDTAVYYCARNLGPSFYFDYWGQGTLVTVSS"
_VL_PERTUZUMAB = (
    "DIQMTQSPSSLSASVGDRVTITCKASQDVSIGVAWYQQKPGKAPKLLIYSASYRYTGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYYIYPYTFGQGTKVEIK"
)

_VH_TEMPLATES = [_VH_TRASTUZUMAB, _VH_PERTUZUMAB, VH_SEQ]
_VL_TEMPLATES = [_VL_TRASTUZUMAB, _VL_PERTUZUMAB, VL_SEQ]


def _diversify_cdr3(seq: str, rng: random.Random) -> str:
    """Mutate 5 contiguous positions just before the WG[QK]GT framework boundary.

    CDR3 is the most variable region in real antibody libraries; randomising
    a small CDR3 window keeps framework regions intact while producing
    distinct, antibody-shaped sequences for benchmark workloads.
    """
    m = re.search(r"WG[QK]GT", seq)
    cdr3_end = m.start() if m else len(seq) - 12
    start = max(0, cdr3_end - 8)
    seq_list = list(seq)
    for pos in range(start, min(start + 5, cdr3_end)):
        seq_list[pos] = rng.choice(PROTEIN_AMINO_ACIDS)
    return "".join(seq_list)


def _benchmark_paired_antibodies(n: int, seed: int) -> list[Antibody]:
    """Generate n paired antibodies cycling through real templates with CDR3 diversification."""
    rng = random.Random(seed)
    return [
        Antibody(
            heavy_chain=_diversify_cdr3(rng.choice(_VH_TEMPLATES), rng),
            light_chain=rng.choice(_VL_TEMPLATES),
        )
        for _ in range(n)
    ]


def _mask_cdr3(seq: str) -> str:
    """Replace 5 contiguous CDR3 positions with the AbLang mask token (``_``)."""
    m = re.search(r"WG[QK]GT", seq)
    cdr3_end = m.start() if m else len(seq) - 12
    start = max(0, cdr3_end - 8)
    seq_list = list(seq)
    for pos in range(start, min(start + 5, cdr3_end)):
        seq_list[pos] = "_"
    return "".join(seq_list)


# ── Benchmarks ──────────────────────────────────────────────────────────────


@pytest.mark.benchmark("ablang-embedding")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_ablang_embeddings_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark ablang-embeddings on 100 paired antibodies (cold + warm)."""
    antibodies = _benchmark_paired_antibodies(n=100, seed=0)
    inputs = AbLangEmbeddingsInput(antibodies=antibodies)
    config = AbLangEmbeddingsConfig(batch_size=32)

    result = benchmark_twice(request, "ablang", lambda: run_ablang_embeddings(inputs=inputs, config=config))

    assert len(result.results) == 100
    # ablang2-paired emits 480-dim embeddings.
    assert len(result.results[0].mean_embedding) == 480


@pytest.mark.benchmark("ablang-score")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_ablang_score_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark ablang-score on 100 paired antibodies (cold + warm)."""
    antibodies = _benchmark_paired_antibodies(n=100, seed=1)
    inputs = AbLangScoringInput(antibodies=antibodies)
    config = AbLangScoringConfig(batch_size=32)

    result = benchmark_twice(request, "ablang", lambda: run_ablang_score(inputs=inputs, config=config))
    assert_metrics_in_spec(result)

    assert result.tool_id == "ablang-score"
    assert len(result.scores) == 100
    for score in result.scores:
        assert score["perplexity"] >= 1.0


@pytest.mark.benchmark("ablang-sample")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_ablang_sample_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark ablang-sample on 50 heavy chains with CDR3 masked (cold + warm)."""
    rng = random.Random(2)
    masked = [Antibody(heavy_chain=_mask_cdr3(_diversify_cdr3(rng.choice(_VH_TEMPLATES), rng))) for _ in range(50)]
    inputs = AbLangSampleInput(antibodies=masked)
    config = AbLangSampleConfig(batch_size=16)

    result = benchmark_twice(request, "ablang", lambda: run_ablang_sample(inputs=inputs, config=config))

    assert len(result.sequences) == 50
    for sampled in result.sequences:
        assert "_" not in sampled, "All masks should be filled"
        assert all(aa in _VALID_AAS for aa in sampled), "All residues should be valid amino acids"


@pytest.mark.benchmark("ablang-gradient")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_ablang_gradient_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark ablang-gradient on a full-length paired antibody as logits (cold + warm)."""
    rng = random.Random(3)
    heavy_logits = one_hot_protein_logits(_diversify_cdr3(rng.choice(_VH_TEMPLATES), rng))
    light_logits = one_hot_protein_logits(rng.choice(_VL_TEMPLATES))
    total_len = len(heavy_logits) + len(light_logits)
    inputs = AbLangGradientInput(antibody=AntibodyLogits(heavy_chain=heavy_logits, light_chain=light_logits))
    config = AbLangGradientConfig()

    result = benchmark_twice(request, "ablang", lambda: run_ablang_gradient(inputs=inputs, config=config))

    assert result.tool_id == "ablang-gradient"
    # Output gradient is a flat (Lh + Ll, 20) matrix — chains concatenated.
    assert result.gradient is not None
    assert len(result.gradient) == total_len
    assert all(len(row) == 20 for row in result.gradient)
    assert all(math.isfinite(v) for row in result.gradient for v in row)
    assert result.metrics["sequence_length"] == total_len
    assert result.metrics["model_choice"] == "ablang2-paired"
