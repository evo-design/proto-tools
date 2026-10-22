"""Tests for AbLang antibody language model tools."""

import math
import random
import re
import sys
import types
from collections import Counter
from typing import Any

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
    inp = AbLangGradientInput(inputs=AntibodyLogits(heavy_chain=[[0.0] * 20, [1.0] * 20]))
    assert inp.inputs[0].heavy_chain == [[0.0] * 20, [1.0] * 20]
    assert inp.temperature == 1.0

    inp_with_temp = AbLangGradientInput(inputs=AntibodyLogits(heavy_chain=[[0.0] * 20]), temperature=0.6)
    assert inp_with_temp.temperature == 0.6

    with pytest.raises(ValidationError):
        AbLangGradientInput(inputs=AntibodyLogits(heavy_chain=[[0.0] * 20]), temperature=-0.5)
    with pytest.raises(ValidationError):
        AbLangGradientInput(inputs=AntibodyLogits(heavy_chain=[[0.0] * 20]), temperature=0.0)

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
        return {
            "results": [
                {
                    "gradient": [[0.0] * 20] * len(item["logits"]),
                    "loss": 0.5,
                    "metrics": {
                        "log_likelihood": -0.5 * len(item["logits"]),
                        "avg_log_likelihood": -0.5,
                        "perplexity": math.exp(0.5),
                        "model_choice": payload.get("model_choice"),
                    },
                }
                for item in payload["items"]
            ],
            "vocab": list(PROTEIN_AMINO_ACIDS),
        }

    monkeypatch.setattr(
        "proto_tools.tools.masked_models.ablang.ablang_gradient.ToolInstance.dispatch",
        fake_dispatch,
    )

    heavy, light = [[0.0] * 20], [[1.0] * 20]

    # Paired: chains concatenated, model auto-selected
    run_ablang_gradient(AbLangGradientInput(inputs=AntibodyLogits(heavy_chain=heavy, light_chain=light)))
    assert captured["payload"]["items"] == [
        {"logits": heavy + light, "chain_break_position": len(heavy), "model_choice": "ablang2-paired"}
    ]
    assert captured["payload"]["temperature"] == 1.0  # shared default
    assert captured["payload"]["soft"] == 1.0
    assert captured["payload"]["hard"] == 0.0
    assert captured["payload"]["compute_gradient"] is True  # default
    assert captured["payload"]["batch_size"] is None  # default auto

    # Heavy only, with temperature
    run_ablang_gradient(AbLangGradientInput(inputs=AntibodyLogits(heavy_chain=heavy), temperature=0.6))
    assert captured["payload"]["items"] == [
        {"logits": heavy, "chain_break_position": None, "model_choice": "ablang1-heavy"}
    ]
    assert captured["payload"]["temperature"] == 0.6

    # Light only
    run_ablang_gradient(AbLangGradientInput(inputs=AntibodyLogits(light_chain=light)))
    assert captured["payload"]["items"] == [
        {"logits": light, "chain_break_position": None, "model_choice": "ablang1-light"}
    ]


def test_ablang_forward_mode_dispatch_contract(monkeypatch):
    """compute_gradient=False forwards the flag and returns gradient=None."""
    captured: dict[str, object] = {}

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured.update(payload=payload)
        return {
            "results": [
                {
                    "gradient": None,
                    "loss": 0.5,
                    "metrics": {
                        "log_likelihood": -0.5 * len(item["logits"]),
                        "avg_log_likelihood": -0.5,
                        "perplexity": math.exp(0.5),
                        "sequence_length": len(item["logits"]),
                    },
                }
                for item in payload["items"]
            ],
            "vocab": list(PROTEIN_AMINO_ACIDS),
        }

    monkeypatch.setattr(
        "proto_tools.tools.masked_models.ablang.ablang_gradient.ToolInstance.dispatch",
        fake_dispatch,
    )

    result = run_ablang_gradient(
        AbLangGradientInput(inputs=AntibodyLogits(heavy_chain=[[0.0] * 20] * 3)),
        AbLangGradientConfig(compute_gradient=False),
    )

    assert captured["payload"]["compute_gradient"] is False
    assert result.results[0].gradient is None
    assert result.results[0].loss == 0.5
    assert result.results[0].metrics["avg_log_likelihood"] == -0.5


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
def test_compute_gradient_dispatch(paired: bool) -> None:
    """Validate compute_gradient with temperature and STE on both chain modes via real dispatch."""
    temperature, hard = 0.6, 1.0
    if paired:
        antibody = AntibodyLogits(heavy_chain=_UNIT_LOGITS[:2], light_chain=_UNIT_LOGITS[2:])
        expected_len = len(_UNIT_LOGITS)
    else:
        antibody = AntibodyLogits(heavy_chain=_UNIT_LOGITS)
        expected_len = len(_UNIT_LOGITS)

    result = run_ablang_gradient(
        AbLangGradientInput(inputs=antibody, temperature=temperature),
        AbLangGradientConfig(hard=hard),
    )
    validate_output(result)

    assert result.results[0].gradient is not None
    assert len(result.results[0].gradient) == expected_len
    assert all(len(row) == 20 for row in result.results[0].gradient)
    assert all(math.isfinite(v) for row in result.results[0].gradient for v in row)
    assert result.results[0].loss > 0
    assert math.isfinite(result.results[0].loss)
    assert result.results[0].metrics["avg_log_likelihood"] == pytest.approx(-result.results[0].loss, rel=1e-6)
    assert result.results[0].metrics["log_likelihood"] == pytest.approx(
        -result.results[0].loss * expected_len, rel=1e-6
    )
    assert result.results[0].metrics["perplexity"] == pytest.approx(math.exp(result.results[0].loss), rel=1e-6)
    assert result.results[0].metrics["sequence_length"] == expected_len
    assert result.vocab == _CANONICAL_VOCAB


# ── Mixed chain configurations ───────────────────────────────────────────────


def _recording_model_class(events: list[tuple[str, str]]) -> type:
    """Build a stand-in AbLangModel that records checkpoint loads and unloads in order.

    Selecting the right checkpoint is not the same as managing memory: a worker that loads a
    second variant without releasing the first would still produce correct numbers. Recording
    both lets a test assert that only one checkpoint is ever resident.

    Args:
        events (list[tuple[str, str]]): Collects ``("load", choice)`` and ``("unload", choice)``
            in the order the worker performs them.

    Returns:
        type: A class matching the slice of ``AbLangModel`` the dispatch path uses.
    """

    class _RecordingModel:
        def __init__(self, model_choice: str) -> None:
            events.append(("load", model_choice))
            self.model_choice = model_choice
            self._loaded = True

        def unload(self) -> None:
            events.append(("unload", self.model_choice))
            self._loaded = False

        def compute_gradient(self, *, logits_list, chain_break_position, **_kwargs):
            # The loss identifies the item, so a reordered reassembly is visible.
            return {
                "gradient": [[float(len(logits_list))] * 20],
                "loss": float(logits_list[0][0]),
                "metrics": {"model_choice": self.model_choice, "chain_break_position": chain_break_position},
                "vocab": list(PROTEIN_AMINO_ACIDS),
            }

    return _RecordingModel


def _mixed_items() -> list[dict[str, Any]]:
    """Return heavy, light, paired, heavy, paired items, interleaved on purpose."""
    plans = [
        ("ablang1-heavy", None),
        ("ablang1-light", None),
        ("ablang2-paired", 2),
        ("ablang1-heavy", None),
        ("ablang2-paired", 3),
    ]
    return [
        {
            "logits": [[float(index)] * 20] * (index + 2),
            "chain_break_position": break_position,
            "model_choice": model_choice,
        }
        for index, (model_choice, break_position) in enumerate(plans)
    ]


def _dispatch_mixed(ablang_inference: Any, items: list[dict[str, Any]]) -> dict[str, Any]:
    """Run one gradient dispatch over the given items."""
    return ablang_inference.dispatch(
        {
            "operation": "compute_gradient",
            "items": items,
            "temperature": 1.0,
            "soft": 1.0,
            "hard": 0.0,
            "seed": None,
            "device": "cpu",
            "verbose": False,
        }
    )


def test_gradient_tags_each_antibody_with_its_own_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The payload names a checkpoint per antibody, chosen from the chains it carries."""
    captured: dict[str, Any] = {}

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured["payload"] = payload
        return {
            "results": [{"gradient": None, "loss": 0.0, "metrics": {}} for _ in payload["items"]],
            "vocab": list(PROTEIN_AMINO_ACIDS),
        }

    monkeypatch.setattr(
        "proto_tools.tools.masked_models.ablang.ablang_gradient.ToolInstance.dispatch",
        fake_dispatch,
    )

    run_ablang_gradient(
        AbLangGradientInput(
            inputs=[
                AntibodyLogits(heavy_chain=[[0.0] * 20] * 3),
                AntibodyLogits(light_chain=[[0.0] * 20] * 4),
                AntibodyLogits(heavy_chain=[[0.0] * 20] * 3, light_chain=[[0.0] * 20] * 4),
            ]
        ),
        AbLangGradientConfig(compute_gradient=False, device="cpu"),
    )

    payload = captured["payload"]
    assert [item["model_choice"] for item in payload["items"]] == [
        "ablang1-heavy",
        "ablang1-light",
        "ablang2-paired",
    ]
    assert "model_choice" not in payload, "the checkpoint is per item, not per call"
    # The paired antibody is the only one that carries a split point.
    assert [item["chain_break_position"] for item in payload["items"]] == [None, None, 3]


def test_worker_loads_each_checkpoint_once_and_keeps_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mixed chunk groups by checkpoint, loads each once, and still replies in input order.

    The contract is that each required checkpoint is loaded once and results keep input order —
    not that the groups run in any particular sequence, so the order they are visited in is
    deliberately not asserted. A future pass that started with the resident checkpoint to save a
    reload would still satisfy this.
    """
    events: list[tuple[str, str]] = []
    ablang_inference = _import_ablang_inference()
    monkeypatch.setattr(ablang_inference, "AbLangModel", _recording_model_class(events))
    monkeypatch.setattr(ablang_inference, "_model", None)

    items = _mixed_items()
    result = _dispatch_mixed(ablang_inference, items)

    loaded = Counter(choice for event, choice in events if event == "load")
    assert loaded == Counter({"ablang1-heavy": 1, "ablang1-light": 1, "ablang2-paired": 1}), (
        f"each distinct checkpoint must load exactly once, in any order; got {events}"
    )
    assert [item["model_choice"] for item in items] == [
        entry["metrics"]["model_choice"] for entry in result["results"]
    ], "results must line up with the items that produced them"
    # The loss carries the item's index, so a reordered reassembly fails here.
    assert [entry["loss"] for entry in result["results"]] == [0.0, 1.0, 2.0, 3.0, 4.0]
    # Only one checkpoint is resident: every load but the first follows an unload.
    assert [event for event, _ in events] == ["load", "unload", "load", "unload", "load"], (
        f"a checkpoint must be released before the next is loaded; got {events}"
    )


def test_persistent_worker_reuses_then_switches_checkpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """Across calls the worker keeps a resident checkpoint and only swaps when the need changes."""
    events: list[tuple[str, str]] = []
    ablang_inference = _import_ablang_inference()
    monkeypatch.setattr(ablang_inference, "AbLangModel", _recording_model_class(events))
    monkeypatch.setattr(ablang_inference, "_model", None)

    heavy_item = {"logits": [[0.0] * 20] * 3, "chain_break_position": None, "model_choice": "ablang1-heavy"}
    paired_item = {"logits": [[1.0] * 20] * 5, "chain_break_position": 2, "model_choice": "ablang2-paired"}

    _dispatch_mixed(ablang_inference, [heavy_item])
    _dispatch_mixed(ablang_inference, [heavy_item])
    assert events == [("load", "ablang1-heavy")], f"a repeat call reuses the resident checkpoint; got {events}"

    _dispatch_mixed(ablang_inference, [paired_item])
    assert events == [
        ("load", "ablang1-heavy"),
        ("unload", "ablang1-heavy"),
        ("load", "ablang2-paired"),
    ], f"a new configuration releases the old checkpoint before loading; got {events}"

    _dispatch_mixed(ablang_inference, [heavy_item])
    assert events[-2:] == [("unload", "ablang2-paired"), ("load", "ablang1-heavy")], (
        f"swapping back releases and reloads, since only one stays resident; got {events}"
    )


def test_mixed_batch_gives_the_same_results_on_one_worker_and_two(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mixed batch is valid on any number of workers and returns identical ordered results.

    This is the regression the per-item checkpoint fixes. Previously the checkpoint was resolved
    for the whole batch inside the tool function, which runs *after* ToolPool has partitioned:
    a heavy-only and a light-only candidate were rejected when one worker received both and
    accepted when two workers each received one, so validity depended on device assignment.
    """
    from proto_tools.utils.device_manager import DeviceManager
    from proto_tools.utils.tool_instance import ToolInstance as _ToolInstance
    from proto_tools.utils.tool_pool import ToolPool

    # The worker call is faked, so the GPU slots only have to exist on paper.
    monkeypatch.setattr("proto_tools.utils.tool_pool.number_of_visible_gpus", lambda: 2)

    partition_sizes: list[int] = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        partition_sizes.append(len(payload["items"]))
        return {
            "results": [
                {
                    "gradient": None,
                    "loss": float(item["logits"][0][0]),
                    "metrics": {"model_choice": item["model_choice"]},
                }
                for item in payload["items"]
            ],
            "vocab": list(PROTEIN_AMINO_ACIDS),
        }

    monkeypatch.setattr(
        "proto_tools.tools.masked_models.ablang.ablang_gradient.ToolInstance.dispatch",
        fake_dispatch,
    )
    antibodies = [
        AntibodyLogits(heavy_chain=[[0.0] * 20] * 3),
        AntibodyLogits(light_chain=[[1.0] * 20] * 4),
        AntibodyLogits(heavy_chain=[[2.0] * 20] * 3, light_chain=[[2.0] * 20] * 4),
        AntibodyLogits(heavy_chain=[[3.0] * 20] * 3),
    ]
    expected_losses = [0.0, 1.0, 2.0, 3.0]
    expected_choices = ["ablang1-heavy", "ablang1-light", "ablang2-paired", "ablang1-heavy"]

    def run_with(slots: int) -> tuple[list[float], list[str]]:
        """Run the same mixed batch under a given number of pool slots."""
        DeviceManager.reset_instance()
        _ToolInstance.clear_all()
        try:
            with ToolPool(gpus=slots):
                output = run_ablang_gradient(
                    AbLangGradientInput(inputs=antibodies),
                    AbLangGradientConfig(compute_gradient=False, device="cuda"),
                )
        finally:
            DeviceManager.reset_instance()
            _ToolInstance.clear_all()
        return [result.loss for result in output.results], [result.metrics["model_choice"] for result in output.results]

    partition_sizes.clear()
    one_worker = run_with(1)
    assert partition_sizes == [4], f"one slot should see the whole batch; got {partition_sizes}"

    partition_sizes.clear()
    two_workers = run_with(2)
    assert len(partition_sizes) > 1, f"two slots should split the batch; got {partition_sizes}"
    assert sum(partition_sizes) == len(antibodies)

    assert one_worker == (expected_losses, expected_choices)
    assert two_workers == one_worker, "results must not depend on how many workers ran the batch"


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_gradient_matches_finite_difference() -> None:
    """The returned gradient must be the derivative of the loss the tool reports.

    A consistency test between batched and singleton runs cannot see a uniform scale error,
    because it is present in both. Only a derivative check can: the objective is the *mean*
    negative log-likelihood over masked positions, so differentiating the sum instead would
    inflate every gradient by the number of positions.
    """
    sequence = VH_SEQ[:12]
    base = one_hot_protein_logits(sequence, sharpness=2.0)
    row, column, epsilon = 3, 7, 0.05

    def evaluate(matrix: list[list[float]], *, with_gradient: bool) -> tuple[float, list[list[float]] | None]:
        output = run_ablang_gradient(
            AbLangGradientInput(inputs=AntibodyLogits(heavy_chain=matrix)),
            AbLangGradientConfig(compute_gradient=with_gradient),
        )
        return output.results[0].loss, output.results[0].gradient

    _, gradient = evaluate(base, with_gradient=True)
    plus = [list(matrix_row) for matrix_row in base]
    plus[row][column] += epsilon
    minus = [list(matrix_row) for matrix_row in base]
    minus[row][column] -= epsilon
    loss_plus, _ = evaluate(plus, with_gradient=False)
    loss_minus, _ = evaluate(minus, with_gradient=False)

    central_difference = (loss_plus - loss_minus) / (2 * epsilon)
    assert gradient is not None
    assert gradient[row][column] == pytest.approx(central_difference, rel=0.02, abs=1e-5), (
        f"gradient {gradient[row][column]:.6e} does not match d(loss)/d(logit) "
        f"{central_difference:.6e}; a ratio near {len(sequence)} means the summed loss was "
        "differentiated while the mean was reported"
    )


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_mixed_batch_matches_individual_evaluations() -> None:
    """A mixed batch must give each antibody the same result as evaluating it alone.

    This is what makes grouping by checkpoint safe: the items are independent, so the loss and
    gradient an antibody receives must not depend on which other configurations travelled with
    it, nor on the order the worker loaded the checkpoints in.
    """
    antibodies = [
        AntibodyLogits(heavy_chain=one_hot_protein_logits(VH_SEQ[:30], sharpness=2.0)),
        AntibodyLogits(light_chain=one_hot_protein_logits(VL_SEQ[:28], sharpness=2.0)),
        AntibodyLogits(
            heavy_chain=one_hot_protein_logits(VH_SEQ[:30], sharpness=2.0),
            light_chain=one_hot_protein_logits(VL_SEQ[:28], sharpness=2.0),
        ),
        AntibodyLogits(heavy_chain=one_hot_protein_logits(VH_SEQ[:30], sharpness=1.0)),
    ]
    config = AbLangGradientConfig()

    mixed = run_ablang_gradient(AbLangGradientInput(inputs=antibodies), config)
    singles = [run_ablang_gradient(AbLangGradientInput(inputs=antibody), config) for antibody in antibodies]

    assert len(mixed.results) == len(antibodies)
    for index, (batched, single) in enumerate(zip(mixed.results, singles, strict=True)):
        assert batched.loss == pytest.approx(single.results[0].loss, rel=1e-5, abs=1e-6), (
            f"item {index} scored differently in a mixed batch"
        )
        assert batched.gradient is not None and single.results[0].gradient is not None
        for row, expected_row in zip(batched.gradient, single.results[0].gradient, strict=True):
            assert row == pytest.approx(expected_row, rel=1e-4, abs=1e-6), f"item {index} gradient differs"


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
    assert result.metadata["model_choice"] == "ablang1-heavy"
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
    assert result.metadata["model_choice"] == "ablang2-paired"

    emb = result.results[0]
    assert len(emb.mean_embedding) == 480
    assert all(math.isfinite(v) for v in emb.mean_embedding)
    assert all(v in (0, 1) for v in emb.attention_mask)
    assert sum(emb.attention_mask) >= len(HEAVY_FULL) + len(LIGHT_FULL)


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
        AbLangGradientInput(inputs=AntibodyLogits(heavy_chain=one_hot), temperature=0.6),
        AbLangGradientConfig(hard=1.0, compute_gradient=False),
    )
    grad_pll = grad_result.results[0].metrics["avg_log_likelihood"]

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


def test_ablang_sample_align_requires_paired_antibodies():
    """AbLang's ANARCI spread is paired-only, so an unpaired request is refused up front.

    Without this, the single-chain request reaches ``restore()`` and dies on an ``IndexError``
    from inside the library's light-chain lookup.
    """
    inputs = AbLangSampleInput(antibodies=[Antibody(heavy_chain=VH_SEQ)])
    with pytest.raises(ValueError, match="align=True requires every antibody"):
        run_ablang_sample(inputs, AbLangSampleConfig(align=True))


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_ablang_sample_align_excludes_amino_acids():
    """``align=True`` restores a paired antibody and still honours ``excluded_amino_acids``.

    Alignment rewrites the heavy C-terminus, so positions are not stable across runs; the
    region after the last unmasked framework residues is entirely model-generated, which is
    what gets checked.
    """
    anchor = VH_SEQ[88:98]
    heavy = VH_SEQ[:98] + "_" * 12 + VH_SEQ[110:]
    inputs = AbLangSampleInput(antibodies=[Antibody(heavy_chain=heavy, light_chain=VL_SEQ)])

    def generated_tail(excluded):
        result = run_ablang_sample(inputs, AbLangSampleConfig(align=True, excluded_amino_acids=excluded, seed=5))
        heavy_out = result.results[0].sequence.split("|")[0]
        index = heavy_out.rfind(anchor)
        assert index >= 0, f"framework anchor missing from {heavy_out}"
        return heavy_out[index + len(anchor) :]

    unconstrained = generated_tail(None)
    assert unconstrained, "align produced no generated tail"
    excluded = sorted(set(unconstrained))[:2]

    constrained = generated_tail(excluded)
    assert constrained, "align produced no generated tail under exclusions"
    assert not set(constrained) & set(excluded), f"excluded {excluded} appeared in generated tail {constrained}"


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
    assert all(sample.logits is None for sample in default.results)

    with_logits = run_ablang_sample(inputs, AbLangSampleConfig(return_logits=True))
    validate_output(with_logits)
    assert len(with_logits.results) == len(masked)
    for sample in with_logits.results:
        assert sample.logits is not None
        # Restored output rows include format-time special tokens; AA columns are 20.
        assert all(len(row) == 20 for row in sample.logits)
        assert all(math.isfinite(v) for row in sample.logits for v in row)
        assert len(sample.logits) >= len(sample.sequence)


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
        AbLangGradientInput(inputs=AntibodyLogits(heavy_chain=[[0.0] * 20] * seq_len), temperature=0.6),
    )
    validate_output(result)

    assert result.tool_id == "ablang-gradient"
    assert len(result.results[0].gradient) == seq_len
    assert all(len(row) == 20 for row in result.results[0].gradient)
    assert all(math.isfinite(v) for row in result.results[0].gradient for v in row)
    assert any(v != 0.0 for row in result.results[0].gradient for v in row)
    assert math.isfinite(result.results[0].loss) and result.results[0].loss > 0
    assert result.vocab == _CANONICAL_VOCAB
    assert result.results[0].metrics["sequence_length"] == seq_len
    assert result.results[0].metrics["model_choice"] == "ablang1-heavy"
    assert result.results[0].metrics["objective"] == "masked_pll"
    assert result.results[0].metrics["avg_log_likelihood"] == pytest.approx(-result.results[0].loss, rel=1e-6)


@pytest.mark.uses_gpu
def test_ablang_gradient_paired_chains():
    """Test paired-chain gradient with separate heavy and light chain inputs."""
    heavy_len = 4
    light_len = 5

    result = run_ablang_gradient(
        AbLangGradientInput(
            inputs=AntibodyLogits(heavy_chain=[[0.0] * 20] * heavy_len, light_chain=[[0.0] * 20] * light_len),
            temperature=0.6,
        ),
    )
    validate_output(result)

    assert result.tool_id == "ablang-gradient"
    assert len(result.results[0].gradient) == heavy_len + light_len
    assert all(len(row) == 20 for row in result.results[0].gradient)
    assert all(math.isfinite(v) for row in result.results[0].gradient for v in row)
    assert any(v != 0.0 for row in result.results[0].gradient for v in row)
    assert result.vocab == _CANONICAL_VOCAB
    assert result.results[0].metrics["sequence_length"] == heavy_len + light_len
    assert result.results[0].metrics["model_choice"] == "ablang2-paired"
    assert result.results[0].metrics["objective"] == "masked_pll"


@pytest.mark.uses_gpu
def test_ablang_gradient_forward_mode_matches_backward_loss():
    """Forward and backward use the same masked PLL objective — losses must match."""
    seq_len = 10
    logits = [[0.0] * 20] * seq_len
    ab = AntibodyLogits(heavy_chain=logits)

    backward = run_ablang_gradient(
        AbLangGradientInput(inputs=ab, temperature=0.6),
        AbLangGradientConfig(seed=42),
    )
    forward = run_ablang_gradient(
        AbLangGradientInput(inputs=ab, temperature=0.6),
        AbLangGradientConfig(seed=42, compute_gradient=False),
    )
    validate_output(backward)
    validate_output(forward)

    assert backward.results[0].gradient is not None
    assert forward.results[0].gradient is None
    assert forward.results[0].loss == pytest.approx(backward.results[0].loss, rel=1e-5)
    assert forward.results[0].metrics["log_likelihood"] == pytest.approx(
        backward.results[0].metrics["log_likelihood"], rel=1e-5
    )
    assert forward.results[0].metrics["sequence_length"] == backward.results[0].metrics["sequence_length"]


@pytest.mark.uses_gpu
def test_ablang_gradient_descent_reduces_loss():
    """Taking a step in the negative gradient direction should reduce the loss."""
    seq_len = 10
    initial_logits = [[0.0] * 20] * seq_len

    result_0 = run_ablang_gradient(
        AbLangGradientInput(inputs=AntibodyLogits(heavy_chain=initial_logits), temperature=0.7),
        AbLangGradientConfig(seed=42),
    )

    lr = 10.0
    stepped_logits = [
        [initial_logits[i][j] - lr * result_0.results[0].gradient[i][j] for j in range(20)] for i in range(seq_len)
    ]

    result_1 = run_ablang_gradient(
        AbLangGradientInput(inputs=AntibodyLogits(heavy_chain=stepped_logits), temperature=0.7),
        AbLangGradientConfig(seed=42),
    )

    assert result_1.results[0].loss < result_0.results[0].loss


@pytest.mark.uses_gpu
@pytest.mark.parametrize("batch_size", [1, 3], ids=["chunk1", "chunk3"])
def test_ablang_gradient_batch_size_equivalence(batch_size: int) -> None:
    """Explicit chunk sizes must match the auto-chunked loss and gradient."""
    logits = [[0.1 * (i + j) for j in range(20)] for i in range(5)]
    ab = AntibodyLogits(heavy_chain=logits)

    ref = run_ablang_gradient(
        AbLangGradientInput(inputs=ab, temperature=0.6),
        AbLangGradientConfig(seed=42, batch_size=None),
    )
    result = run_ablang_gradient(
        AbLangGradientInput(inputs=ab, temperature=0.6),
        AbLangGradientConfig(seed=42, batch_size=batch_size),
    )

    assert result.results[0].loss == pytest.approx(ref.results[0].loss, rel=1e-4, abs=1e-6)
    for row_r, row_ref in zip(result.results[0].gradient, ref.results[0].gradient, strict=True):
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
    inputs = AbLangGradientInput(inputs=AntibodyLogits(heavy_chain=heavy_logits, light_chain=light_logits))
    config = AbLangGradientConfig()

    result = benchmark_twice(request, "ablang", lambda: run_ablang_gradient(inputs=inputs, config=config))

    assert result.tool_id == "ablang-gradient"
    (item,) = result.results
    # Each item's gradient is a flat (Lh + Ll, 20) matrix — chains concatenated.
    assert item.gradient is not None
    assert len(item.gradient) == total_len
    assert all(len(row) == 20 for row in item.gradient)
    assert all(math.isfinite(v) for row in item.gradient for v in row)
    assert item.metrics["sequence_length"] == total_len
    assert item.metrics["model_choice"] == "ablang2-paired"
