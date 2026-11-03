"""Tests for the ESMFold2 inversion structure-objective gradient tool."""

import importlib.util
import logging
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from proto_tools.tools.structure_prediction import (
    ESMFold2EpitopeTerm,
    ESMFold2GlobularityTerm,
    ESMFold2InterContactTerm,
    ESMFold2IntraContactTerm,
    ESMFold2InversionCandidate,
    ESMFold2InversionGradientConfig,
    ESMFold2InversionGradientInput,
    ESMFold2InversionGradientOutput,
    ESMFold2PseudoPerplexityTerm,
    run_esmfold2_inversion_gradient,
)
from proto_tools.utils import PROTEIN_AMINO_ACIDS, one_hot_protein_logits
from tests.conftest import benchmark_twice

TARGET = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
BINDER = "MKTAYIAKQRQISFVKSHFSRQLE"

_OBJECTIVE_PATH = (
    Path(__file__).resolve().parents[2]
    / "proto_tools/tools/structure_prediction/esmfold2_inversion/standalone/objective.py"
)

# Values captured from the pinned upstream recipe at the revision the shared env tracks. They
# are literals on purpose: they must agree with something outside this repo, which is the one
# case where pinning a number earns its place.
_REFERENCE_SEED = 0
_REFERENCE_INTRA_CONTACT = 4.852100372314453
_REFERENCE_INTER_CONTACT = 4.852100372314453


def _objective() -> Any:
    """Load the worker's objective module, which needs only torch and a logger."""
    pytest.importorskip("torch")
    sys.modules.setdefault("standalone_helpers", types.SimpleNamespace(get_logger=logging.getLogger))
    spec = importlib.util.spec_from_file_location("esmfold2_inversion_objective", _OBJECTIVE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(**overrides: Any) -> ESMFold2InversionCandidate:
    """Build a valid candidate, overriding whichever fields a test is about."""
    fields: dict[str, Any] = {
        "logits": one_hot_protein_logits(BINDER, sharpness=2.0),
        "target_sequence": TARGET,
    }
    return ESMFold2InversionCandidate(**(fields | overrides))


# ── Input validation ──────────────────────────────────────────────────────────


def test_candidate_rejects_a_non_canonical_target() -> None:
    """A target with a non-standard residue cannot be folded, so it is refused up front."""
    with pytest.raises(ValidationError, match="canonical amino acids"):
        _candidate(target_sequence="MKTZ")


def test_candidate_requires_the_alphabet_to_be_a_permutation() -> None:
    """A column order that drops or repeats a residue would silently mis-map the gradient."""
    with pytest.raises(ValidationError, match="permutation"):
        _candidate(alphabet="ACDEFGHIKLMNPQRSTVWX")


@pytest.mark.parametrize(
    ("hotspot", "expected"),
    [
        ("L999", "outside the target"),
        ("A2", "does not match the target"),
        ("15", "residue letter"),
    ],
)
def test_candidate_rejects_hotspots_the_target_contradicts(hotspot: str, expected: str) -> None:
    """A hotspot names the residue it expects, so a stale selection fails rather than drifting."""
    with pytest.raises(ValidationError, match=expected):
        _candidate(target_hotspot_ids=[hotspot])


def test_candidate_accepts_a_hotspot_the_target_confirms() -> None:
    """The positive case: a hotspot whose residue is where it claims to be."""
    position = 15
    candidate = _candidate(target_hotspot_ids=[f"{TARGET[position - 1]}{position}"])
    assert candidate.target_hotspot_ids == [f"{TARGET[position - 1]}{position}"]


def test_input_accepts_a_single_candidate_without_a_list() -> None:
    """Evaluating one binder should not require wrapping it."""
    single = ESMFold2InversionGradientInput(inputs=_candidate())
    assert len(single.inputs) == 1


# ── Catalog and configuration ─────────────────────────────────────────────────


def test_catalog_is_discoverable_without_a_config() -> None:
    """``supported_terms()`` is how a caller finds out what this tool can differentiate."""
    assert ESMFold2InversionGradientConfig.supported_terms() == {
        "intra_contact": ESMFold2IntraContactTerm,
        "inter_contact": ESMFold2InterContactTerm,
        "globularity": ESMFold2GlobularityTerm,
        "epitope": ESMFold2EpitopeTerm,
        "plm_nll": ESMFold2PseudoPerplexityTerm,
    }


def test_config_resolves_catalog_names_to_terms() -> None:
    """Naming a quantity is the same as instantiating it."""
    by_name = ESMFold2InversionGradientConfig(gradient_terms=["globularity", "epitope"])
    by_class = ESMFold2InversionGradientConfig(gradient_terms=[ESMFold2GlobularityTerm(), ESMFold2EpitopeTerm()])
    assert by_name.gradient_terms == by_class.gradient_terms


def test_config_rejects_the_same_quantity_twice() -> None:
    """Asking for one quantity twice is one question, not two."""
    with pytest.raises(ValidationError, match="more than once"):
        ESMFold2InversionGradientConfig(gradient_terms=["globularity", "globularity"])


def test_epitope_quantity_requires_hotspots() -> None:
    """Pydantic cannot see across input and config, so the pairing is checked in the run function."""
    with pytest.raises(ValueError, match="needs target_hotspot_ids"):
        run_esmfold2_inversion_gradient(
            ESMFold2InversionGradientInput(inputs=_candidate()),
            ESMFold2InversionGradientConfig(gradient_terms=["epitope"]),
        )


# ── Objective numerics, on CPU with no weights ────────────────────────────────


def test_bin_midpoints_span_the_distogram() -> None:
    """The outer bins are open-ended, so their midpoints sit beyond the interior boundaries."""
    torch = pytest.importorskip("torch")
    midpoints = _objective().bin_midpoints()
    assert midpoints.shape == (128,)
    assert midpoints[0] < 2.0 < midpoints[1]
    assert midpoints[-1] > 52.0
    assert torch.all(midpoints.diff() > 0), "midpoints must increase with distance"


@pytest.mark.parametrize("quantity", ["intra_contact_loss", "inter_contact_loss", "globularity_loss"])
def test_structure_objective_gradient_matches_a_finite_difference(quantity: str) -> None:
    """The analytic gradient is what an optimizer steps along, so it is checked numerically.

    In float64 against a synthetic distogram, so this runs with no GPU, no weights and no
    network. The real checkpoint runs in bf16, where a difference this small does not converge;
    that path is checked for direction instead.
    """
    torch = pytest.importorskip("torch")
    objective = _objective()
    binder_length = 24
    torch.manual_seed(0)
    logits = torch.randn(1, 40, 40, 128, dtype=torch.float64, requires_grad=True)
    midpoints = objective.bin_midpoints().double()

    value = getattr(objective, quantity)(logits, binder_length, midpoints)
    value.sum().backward()
    analytic = logits.grad[0, 3, 7, 11].item()

    index = (0, 3, 7, 11)
    epsilon = 1e-5
    nudged = logits.detach().clone()
    nudged[index] += epsilon
    high = getattr(objective, quantity)(nudged, binder_length, midpoints).sum().item()
    nudged[index] -= 2 * epsilon
    low = getattr(objective, quantity)(nudged, binder_length, midpoints).sum().item()

    assert analytic == pytest.approx((high - low) / (2 * epsilon), rel=1e-5, abs=1e-9)


def test_a_descent_step_lowers_every_structure_objective() -> None:
    """Stepping against the gradient must improve the objective, or its sign is wrong."""
    torch = pytest.importorskip("torch")
    objective = _objective()
    binder_length = 24
    for quantity in ("intra_contact_loss", "inter_contact_loss", "globularity_loss"):
        torch.manual_seed(1)
        logits = torch.randn(1, 40, 40, 128, dtype=torch.float64, requires_grad=True)
        midpoints = objective.bin_midpoints().double()
        before = getattr(objective, quantity)(logits, binder_length, midpoints)
        before.sum().backward()
        stepped = logits.detach() - 0.1 * logits.grad
        after = getattr(objective, quantity)(stepped, binder_length, midpoints)
        assert after.item() < before.item(), f"{quantity} did not improve along its own gradient"


def test_epitope_objective_only_scores_the_named_hotspots() -> None:
    """Moving a non-hotspot target residue must not change the epitope objective."""
    torch = pytest.importorskip("torch")
    objective = _objective()
    binder_length, target_length = 24, 16
    torch.manual_seed(2)
    logits = torch.randn(1, binder_length + target_length, binder_length + target_length, 128, dtype=torch.float64)
    midpoints = objective.bin_midpoints().double()
    mask = torch.zeros(1, binder_length + target_length, dtype=torch.bool)
    mask[0, [2, 5]] = True

    baseline = objective.epitope_loss(logits, mask, binder_length, midpoints, 12.0)
    moved = logits.clone()
    moved[0, 9, :, :] += 5.0  # row 9 is a target residue that is not a hotspot
    assert objective.epitope_loss(moved, mask, binder_length, midpoints, 12.0).item() == pytest.approx(baseline.item())


def test_intra_contact_is_inert_when_no_pair_clears_the_separation_floor() -> None:
    """A binder shorter than the separation floor has no eligible pair, so the objective is zero.

    Worth pinning because a silent zero here once made an equivalence check pass vacuously.
    """
    torch = pytest.importorskip("torch")
    objective = _objective()
    torch.manual_seed(3)
    logits = torch.randn(1, 20, 20, 128, dtype=torch.float64)
    midpoints = objective.bin_midpoints().double()
    assert objective.intra_contact_loss(logits, 8, midpoints).item() == 0.0
    assert objective.intra_contact_loss(logits, 12, midpoints).item() != 0.0


def test_straight_through_is_one_hot_forward_and_relaxed_backward() -> None:
    """The language model scores discrete tokens; the gradient still has to reach the logits."""
    torch = pytest.importorskip("torch")
    objective = _objective()
    relaxed = torch.tensor([[0.1, 0.7, 0.2]], dtype=torch.float64, requires_grad=True)
    passed = objective.straight_through(relaxed)
    assert passed.tolist() == [[0.0, 1.0, 0.0]], "forward pass is the argmax"
    passed.sum().backward()
    assert relaxed.grad.tolist() == [[1.0, 1.0, 1.0]], "backward pass reaches every column"


# ── Dispatch contract ─────────────────────────────────────────────────────────


def test_dispatch_sends_the_requested_quantities_and_keeps_the_column_order() -> None:
    """The caller's alphabet is what the gradient comes back in, whatever order the model uses."""
    reversed_alphabet = PROTEIN_AMINO_ACIDS[::-1]
    reply = {
        "terms": [
            {"id": "globularity", "value": 1.5, "gradient": [[0.1] * 20] * len(BINDER)},
            {"id": "inter_contact", "value": 2.5, "gradient": [[0.2] * 20] * len(BINDER)},
        ],
        "metrics": {"binder_length": len(BINDER), "target_length": len(TARGET)},
    }
    with patch(
        "proto_tools.tools.structure_prediction.esmfold2_inversion.esmfold2_inversion_gradient.ToolInstance.dispatch"
    ) as dispatch:
        dispatch.return_value = reply
        output = run_esmfold2_inversion_gradient(
            ESMFold2InversionGradientInput(inputs=_candidate(alphabet=reversed_alphabet), temperature=0.7),
            ESMFold2InversionGradientConfig(gradient_terms=["globularity", "inter_contact"]),
        )

    toolkit, payload = dispatch.call_args.args
    assert toolkit == "esmfold2_inversion"
    assert payload["operation"] == "structure_gradient"
    assert payload["gradient_terms"] == ["globularity", "inter_contact"]
    assert payload["alphabet"] == reversed_alphabet
    assert payload["temperature"] == 0.7

    globularity, inter = output.results[0].terms
    assert [term.term.id for term in output.results[0].terms] == ["globularity", "inter_contact"]
    assert (globularity.term_value, inter.term_value) == (1.5, 2.5)
    assert output.vocab == list(reversed_alphabet), "the gradient is columned the way the caller asked"
    assert globularity.metadata is not None
    assert globularity.metadata.units == "angstroms", "units come from the term class"
    assert inter.metadata.units is None
    assert globularity.metadata.input_fingerprint == inter.metadata.input_fingerprint, "one fold, one point"


def test_every_quantity_is_lower_is_better_so_composing_needs_no_goals() -> None:
    """These are losses, so the advisory direction resolves on its own."""
    reply = {
        "terms": [
            {"id": "intra_contact", "value": 2.0, "gradient": [[0.1] * 20] * len(BINDER)},
            {"id": "globularity", "value": 3.0, "gradient": [[0.2] * 20] * len(BINDER)},
        ],
        "metrics": {"binder_length": len(BINDER), "target_length": len(TARGET)},
    }
    with patch(
        "proto_tools.tools.structure_prediction.esmfold2_inversion.esmfold2_inversion_gradient.ToolInstance.dispatch"
    ) as dispatch:
        dispatch.return_value = reply
        output = run_esmfold2_inversion_gradient(
            ESMFold2InversionGradientInput(inputs=_candidate()),
            ESMFold2InversionGradientConfig(gradient_terms=["intra_contact", "globularity"]),
        )

    composed = output.results[0].compose()
    assert [c.dloss_dvalue for c in composed.contributions] == [1.0, 1.0], "minimizing, by default"
    assert composed.objective_loss == pytest.approx(5.0)

    # The fixed weighting the upstream recipe hard-codes is now just an argument.
    weighted = output.results[0].compose(weights={output.results[0].terms[0].term: 0.5})
    assert weighted.objective_loss == pytest.approx(0.5 * 2.0 + 3.0)


# ---------------------------------------------------------------------------
# Integration tests


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_inversion_gradient_single_candidate() -> None:
    """A real checkpoint returns a finite gradient over the binder's own rows."""
    output = run_esmfold2_inversion_gradient(
        ESMFold2InversionGradientInput(inputs=_candidate()),
        ESMFold2InversionGradientConfig(gradient_terms=["intra_contact", "inter_contact", "globularity"], seed=0),
    )
    (result,) = output.results
    assert [term.term.id for term in result.terms] == ["intra_contact", "inter_contact", "globularity"]
    for term in result.terms:
        assert term.gradient is not None
        assert len(term.gradient) == len(BINDER), "one gradient row per designed residue"
        assert all(len(row) == 20 for row in term.gradient)
        assert all(value == value for row in term.gradient for value in row), "no NaNs"
    assert result.metrics["binder_length"] == len(BINDER)


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_inversion_gradient_points_downhill_on_a_real_checkpoint() -> None:
    """Direction only: the deployed trunk runs in bf16, where a tiny difference does not converge.

    The step size matches the scale the reference optimizer actually resolves.
    """
    torch = pytest.importorskip("torch")
    inputs = ESMFold2InversionGradientInput(inputs=_candidate())
    config = ESMFold2InversionGradientConfig(gradient_terms=["inter_contact"], seed=0)
    baseline = run_esmfold2_inversion_gradient(inputs, config)
    (term,) = baseline.results[0].terms

    direction = torch.tensor(term.gradient)
    direction = direction / direction.norm()
    start = torch.tensor(inputs.inputs[0].logits.root)

    def value_at(step: float) -> float:
        moved = (start - step * direction).tolist()
        moved_output = run_esmfold2_inversion_gradient(
            ESMFold2InversionGradientInput(inputs=_candidate(logits=moved)),
            ESMFold2InversionGradientConfig(gradient_terms=["inter_contact"], seed=0, compute_gradient=False),
        )
        return moved_output.results[0].terms[0].term_value

    assert value_at(0.2) < value_at(-0.2), "stepping against the gradient must improve the objective"


@pytest.mark.benchmark("esmfold2-inversion-gradient")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_esmfold2_inversion_gradient_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark: one fold of a binder/target complex plus a backward per term (cold + warm)."""
    inputs = ESMFold2InversionGradientInput(inputs=_candidate())
    config = ESMFold2InversionGradientConfig(gradient_terms=["intra_contact", "inter_contact", "globularity"], seed=0)
    result = benchmark_twice(request, "esmfold2_inversion", lambda: run_esmfold2_inversion_gradient(inputs, config))

    assert isinstance(result, ESMFold2InversionGradientOutput)
    assert result.tool_id == "esmfold2-inversion-gradient"
    (candidate,) = result.results
    assert len(candidate.terms) == 3
    assert all(term.gradient is not None for term in candidate.terms)
