"""Tests for the ESMFold2 inversion gradient tools, general and binder-specific."""

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
    ESMFold2BinderCandidate,
    ESMFold2BinderGradientConfig,
    ESMFold2BinderGradientInput,
    ESMFold2Chain,
    ESMFold2ComplexCandidate,
    ESMFold2EpitopeTerm,
    ESMFold2GlobularityTerm,
    ESMFold2InterContactTerm,
    ESMFold2IntraContactTerm,
    ESMFold2InversionGradientConfig,
    ESMFold2InversionGradientInput,
    ESMFold2InversionGradientOutput,
    ESMFold2PseudoPerplexityTerm,
    run_esmfold2_inversion_binder_gradient,
    run_esmfold2_inversion_gradient,
)
from proto_tools.utils import PROTEIN_AMINO_ACIDS, one_hot_protein_logits
from tests.conftest import benchmark_twice

TARGET = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
BINDER = "MKTAYIAKQRQISFVKSHFSRQLE"
CHAIN_B = "MKTAYIAKQRQISFVKS"
CHAIN_C = "LEERLGLIEVQAPILSR"

_OBJECTIVE_PATH = (
    Path(__file__).resolve().parents[2] / "proto_tools/tools/structure_prediction/esmfold2/standalone/objective.py"
)

_GENERAL_DISPATCH = "proto_tools.tools.structure_prediction.esmfold2.esmfold2_inversion_gradient.ToolInstance.dispatch"
_BINDER_DISPATCH = (
    "proto_tools.tools.structure_prediction.esmfold2.esmfold2_inversion_binder_gradient.ToolInstance.dispatch"
)


def _objective() -> Any:
    """Load the worker objective module, which needs only torch and a logger."""
    pytest.importorskip("torch")
    sys.modules.setdefault("standalone_helpers", types.SimpleNamespace(get_logger=logging.getLogger))
    spec = importlib.util.spec_from_file_location("esmfold2_objective", _OBJECTIVE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _binder(**overrides: Any) -> ESMFold2BinderCandidate:
    """Build a valid binder candidate, overriding whichever fields a test is about."""
    fields: dict[str, Any] = {
        "logits": one_hot_protein_logits(BINDER, sharpness=2.0),
        "target_sequence": TARGET,
    }
    return ESMFold2BinderCandidate(**(fields | overrides))


def _three_chains() -> ESMFold2ComplexCandidate:
    """One fixed chain followed by two independently designable chains."""
    return ESMFold2ComplexCandidate(
        chains=[
            ESMFold2Chain(chain_id="A", sequence=TARGET),
            ESMFold2Chain(chain_id="B", logits=one_hot_protein_logits(CHAIN_B, sharpness=2.0)),
            ESMFold2Chain(chain_id="C", logits=one_hot_protein_logits(CHAIN_C, sharpness=2.0)),
        ]
    )


# ── Chain and complex validation ───────────────────────────────────────


def test_chain_is_either_fixed_or_designable_never_both() -> None:
    """A chain carrying both a sequence and logits leaves its role undefined."""
    with pytest.raises(ValidationError, match="exactly one of sequence"):
        ESMFold2Chain(chain_id="A", sequence="MKTA", logits=[[0.0] * 20] * 4)


def test_chain_must_be_one_or_the_other() -> None:
    """A chain carrying neither is not foldable."""
    with pytest.raises(ValidationError, match="exactly one of sequence"):
        ESMFold2Chain(chain_id="A")


def test_complex_requires_something_to_design() -> None:
    """A complex of only fixed chains has no gradient to take."""
    with pytest.raises(ValidationError, match="at least one chain must be designable"):
        ESMFold2ComplexCandidate(chains=[ESMFold2Chain(chain_id="A", sequence="MKTA")])


def test_complex_rejects_repeated_chain_ids() -> None:
    """Terms select chains by id, so duplicates would make the selection ambiguous."""
    with pytest.raises(ValidationError, match="unique"):
        ESMFold2ComplexCandidate(
            chains=[
                ESMFold2Chain(chain_id="A", sequence="MKTA"),
                ESMFold2Chain(chain_id="A", logits=[[0.0] * 20] * 4),
            ]
        )


def test_complex_rejects_designable_chains_with_different_alphabets() -> None:
    """One gradient matrix spans every designable chain, so its columns mean one thing."""
    with pytest.raises(ValidationError, match="share one alphabet"):
        ESMFold2ComplexCandidate(
            chains=[
                ESMFold2Chain(chain_id="B", logits=[[0.0] * 20] * 4),
                ESMFold2Chain(chain_id="C", logits=[[0.0] * 20] * 4, alphabet=PROTEIN_AMINO_ACIDS[::-1]),
            ]
        )


def test_designable_chains_are_exposed_in_fold_order() -> None:
    """The shared framework discovers relaxed state here, and row order follows fold order."""
    candidate = _three_chains()
    assert [chain.chain_id for chain in candidate.designable_chains] == ["B", "C"]
    assert [name for name, _ in candidate.iter_sequence_logits()] == [
        "chains[1].logits",
        "chains[2].logits",
    ], "matrices are declared by the path they live at, which is what the shared walker reports"


# ── Term selection ────────────────────────────────────────────────────


def test_general_catalog_excludes_the_interface_quantities() -> None:
    """inter_contact and epitope are asymmetric binder quantities and belong to that tool."""
    assert ESMFold2InversionGradientConfig.supported_terms() == {
        "intra_contact": ESMFold2IntraContactTerm,
        "globularity": ESMFold2GlobularityTerm,
        "plm_nll": ESMFold2PseudoPerplexityTerm,
    }


def test_binder_catalog_offers_all_five() -> None:
    """The binder tool adds the two interface quantities to the three single-chain ones."""
    assert set(ESMFold2BinderGradientConfig.supported_terms()) == {
        "intra_contact",
        "globularity",
        "plm_nll",
        "inter_contact",
        "epitope",
    }


@pytest.mark.parametrize("requested", ["inter_contact", "epitope"])
def test_general_tool_rejects_binder_quantities_by_name(requested: str) -> None:
    """Naming a binder quantity in the general tool fails at config validation."""
    with pytest.raises(ValidationError):
        ESMFold2InversionGradientConfig(gradient_terms=[requested])


@pytest.mark.parametrize("instance", [ESMFold2InterContactTerm(), ESMFold2EpitopeTerm()])
def test_general_tool_rejects_binder_quantities_as_instances(instance: Any) -> None:
    """Supplying the term object rather than its name must fail the same way."""
    with pytest.raises(ValidationError):
        ESMFold2InversionGradientConfig(gradient_terms=[instance])


def test_general_tool_rejects_the_same_quantity_on_one_chain_twice() -> None:
    """One quantity on one chain is one question."""
    with pytest.raises(ValidationError, match="more than once"):
        ESMFold2InversionGradientConfig(
            gradient_terms=[ESMFold2IntraContactTerm(chain_id="B"), ESMFold2IntraContactTerm(chain_id="B")]
        )


def test_general_tool_allows_one_quantity_on_several_chains() -> None:
    """Per-chain terms coexist; that is the point of selecting a chain."""
    config = ESMFold2InversionGradientConfig(
        gradient_terms=[ESMFold2IntraContactTerm(chain_id="B"), ESMFold2IntraContactTerm(chain_id="C")]
    )
    assert [term.chain_id for term in config.gradient_terms] == ["B", "C"]


def test_binder_tool_rejects_a_chain_selection_that_is_not_the_binder() -> None:
    """Chain roles are fixed here; a general complex belongs in the general tool."""
    with pytest.raises(ValidationError, match="esmfold2-inversion-gradient"):
        ESMFold2BinderGradientConfig(gradient_terms=[ESMFold2IntraContactTerm(chain_id="target")])


# ── Chain-aware objective numerics, on CPU with no weights ────────────────────


def test_bin_midpoints_span_the_distogram() -> None:
    """The outer bins are open-ended, so their midpoints sit beyond the interior boundaries."""
    torch = pytest.importorskip("torch")
    midpoints = _objective().bin_midpoints(128)
    assert midpoints.shape == (128,)
    assert midpoints[0] < 2.0 < midpoints[1]
    assert torch.all(midpoints.diff() > 0), "midpoints must increase with distance"


def test_span_mask_rejects_a_span_outside_the_complex() -> None:
    """A chain span that does not fit would silently score the wrong rows."""
    torch = pytest.importorskip("torch")
    objective = _objective()
    with pytest.raises(ValueError, match="does not fit"):
        objective.span_mask(10, (5, 20), torch.device("cpu"))


def test_intra_contact_ignores_contacts_across_a_chain_boundary() -> None:
    """A residue must not satisfy its own chain's objective by touching a neighbouring chain."""
    torch = pytest.importorskip("torch")
    objective = _objective()
    midpoints = objective.bin_midpoints(128).double()
    torch.manual_seed(0)
    logits = torch.randn(1, 44, 44, 128, dtype=torch.float64)
    span = (20, 44)

    baseline = objective.intra_contact_loss(logits, span, midpoints)
    crossed = logits.clone()
    # Make every cross-boundary pair look like a tight contact.
    crossed[0, 20:44, 0:20, :40] += 8.0
    crossed[0, 0:20, 20:44, :40] += 8.0
    assert objective.intra_contact_loss(crossed, span, midpoints).item() == pytest.approx(baseline.item()), (
        "contacts outside the chain must not count toward its intra-chain objective"
    )


def test_globularity_reads_only_its_own_chain() -> None:
    """Compactness of one chain cannot be improved by rearranging another."""
    torch = pytest.importorskip("torch")
    objective = _objective()
    midpoints = objective.bin_midpoints(128).double()
    torch.manual_seed(1)
    logits = torch.randn(1, 44, 44, 128, dtype=torch.float64)
    span = (20, 44)

    baseline = objective.globularity_loss(logits, span, midpoints)
    elsewhere = logits.clone()
    elsewhere[0, 0:20, 0:20, :40] += 8.0
    assert objective.globularity_loss(elsewhere, span, midpoints).item() == pytest.approx(baseline.item())


def test_globularity_differs_between_chains_of_the_same_complex() -> None:
    """Each chain gets its own radius of gyration, not the complex's."""
    torch = pytest.importorskip("torch")
    objective = _objective()
    midpoints = objective.bin_midpoints(128).double()
    torch.manual_seed(2)
    logits = torch.randn(1, 44, 44, 128, dtype=torch.float64)
    first = objective.globularity_loss(logits, (0, 20), midpoints).item()
    second = objective.globularity_loss(logits, (20, 44), midpoints).item()
    assert first != second, "two different chains should not score identically by construction"


@pytest.mark.parametrize("quantity", ["intra_contact_loss", "globularity_loss"])
def test_chain_objective_gradient_matches_a_finite_difference(quantity: str) -> None:
    """The analytic gradient is what an optimizer steps along, so it is checked numerically.

    In float64 against a synthetic distogram, so this runs with no GPU, no weights and no
    network. The real checkpoint runs in bf16, where a difference this small does not converge;
    that path is checked for direction instead.
    """
    torch = pytest.importorskip("torch")
    objective = _objective()
    span = (16, 40)
    torch.manual_seed(0)
    logits = torch.randn(1, 40, 40, 128, dtype=torch.float64, requires_grad=True)
    midpoints = objective.bin_midpoints(128).double()

    getattr(objective, quantity)(logits, span, midpoints).sum().backward()
    index = (0, 20, 27, 11)
    analytic = logits.grad[index].item()

    epsilon = 1e-5
    nudged = logits.detach().clone()
    nudged[index] += epsilon
    high = getattr(objective, quantity)(nudged, span, midpoints).sum().item()
    nudged[index] -= 2 * epsilon
    low = getattr(objective, quantity)(nudged, span, midpoints).sum().item()

    assert analytic == pytest.approx((high - low) / (2 * epsilon), rel=1e-5, abs=1e-9)


def test_a_descent_step_lowers_each_chain_objective() -> None:
    """Stepping against the gradient must improve the objective, or its sign is wrong."""
    torch = pytest.importorskip("torch")
    objective = _objective()
    span = (16, 40)
    for quantity in ("intra_contact_loss", "globularity_loss"):
        torch.manual_seed(1)
        logits = torch.randn(1, 40, 40, 128, dtype=torch.float64, requires_grad=True)
        midpoints = objective.bin_midpoints(128).double()
        before = getattr(objective, quantity)(logits, span, midpoints)
        before.sum().backward()
        after = getattr(objective, quantity)(logits.detach() - 0.1 * logits.grad, span, midpoints)
        assert after.item() < before.item(), f"{quantity} did not improve along its own gradient"


def test_inter_contact_stays_asymmetric() -> None:
    """Swapping the two spans must change the value; it scores the target, not the binder."""
    torch = pytest.importorskip("torch")
    objective = _objective()
    midpoints = objective.bin_midpoints(128).double()
    torch.manual_seed(3)
    logits = torch.randn(1, 44, 44, 128, dtype=torch.float64)
    forward = objective.inter_contact_loss(logits, (0, 20), (20, 44), midpoints).item()
    swapped = objective.inter_contact_loss(logits, (20, 44), (0, 20), midpoints).item()
    assert forward != swapped, "inter_contact is deliberately not a symmetric inter-chain objective"


def test_intra_contact_is_inert_when_no_pair_clears_the_separation_floor() -> None:
    """A chain shorter than the separation floor has no eligible pair, so the objective is zero.

    Worth pinning because a silent zero here once made an equivalence check pass vacuously.
    """
    torch = pytest.importorskip("torch")
    objective = _objective()
    torch.manual_seed(4)
    logits = torch.randn(1, 20, 20, 128, dtype=torch.float64)
    midpoints = objective.bin_midpoints(128).double()
    assert objective.intra_contact_loss(logits, (12, 20), midpoints).item() == 0.0
    assert objective.intra_contact_loss(logits, (0, 20), midpoints).item() != 0.0


def test_straight_through_is_one_hot_forward_and_relaxed_backward() -> None:
    """The language model scores discrete tokens; the gradient still has to reach the logits."""
    torch = pytest.importorskip("torch")
    objective = _objective()
    relaxed = torch.tensor([[0.1, 0.7, 0.2]], dtype=torch.float64, requires_grad=True)
    passed = objective.straight_through(relaxed)
    assert passed.tolist() == [[0.0, 1.0, 0.0]], "forward pass is the argmax"
    passed.sum().backward()
    assert relaxed.grad.tolist() == [[1.0, 1.0, 1.0]], "backward pass reaches every column"


# ── Dispatch contracts ────────────────────────────────────────────────────────


def _general_reply(rows: int, chain_rows: dict[str, list[int]]) -> dict[str, Any]:
    """A worker reply for the general tool with one gradient row block per designable chain."""
    return {
        "terms": [
            {"id": "intra_contact", "chain_id": "B", "value": 1.5, "gradient": [[0.1] * 20] * rows},
            {"id": "globularity", "chain_id": "C", "value": 2.5, "gradient": [[0.2] * 20] * rows},
        ],
        "chain_rows": chain_rows,
        "metrics": {"complex_length": len(TARGET) + rows, "designable_length": rows},
    }


def test_general_dispatch_sends_chains_and_resolved_chain_ids() -> None:
    """Every chain reaches the worker with its role, and each term names the chain it scores."""
    candidate = _three_chains()
    rows = len(CHAIN_B) + len(CHAIN_C)
    chain_rows = {"B": [0, len(CHAIN_B)], "C": [len(CHAIN_B), rows]}
    with patch(_GENERAL_DISPATCH) as dispatch:
        dispatch.return_value = _general_reply(rows, chain_rows)
        output = run_esmfold2_inversion_gradient(
            ESMFold2InversionGradientInput(inputs=candidate, temperature=0.7),
            ESMFold2InversionGradientConfig(
                gradient_terms=[ESMFold2IntraContactTerm(chain_id="B"), ESMFold2GlobularityTerm(chain_id="C")]
            ),
        )

    toolkit, payload = dispatch.call_args.args
    assert toolkit == "esmfold2"
    assert payload["operation"] == "complex_gradient"
    assert [chain["chain_id"] for chain in payload["chains"]] == ["A", "B", "C"]
    assert payload["chains"][0]["logits"] is None, "the fixed chain sends no relaxed state"
    assert payload["chains"][1]["sequence"] is None, "a designable chain sends no discrete sequence"
    assert payload["gradient_terms"] == [
        {"id": "intra_contact", "chain_id": "B"},
        {"id": "globularity", "chain_id": "C"},
    ]
    assert payload["temperature"] == 0.7

    result = output.results[0]
    assert result.chain_rows == chain_rows, "the caller is told where each chain's rows are"
    assert [term.term.chain_id for term in result.terms] == ["B", "C"]
    for term in result.terms:
        assert len(term.gradient) == rows, "every gradient covers all designable rows"
    assert result.terms[0].metadata.input_fingerprint == result.terms[1].metadata.input_fingerprint


def test_general_dispatch_omits_fixed_chains_from_the_gradient_rows() -> None:
    """Fixed chains are context: they are folded, but no gradient is returned for them."""
    candidate = _three_chains()
    rows = len(CHAIN_B) + len(CHAIN_C)
    chain_rows = {"B": [0, len(CHAIN_B)], "C": [len(CHAIN_B), rows]}
    with patch(_GENERAL_DISPATCH) as dispatch:
        dispatch.return_value = _general_reply(rows, chain_rows)
        output = run_esmfold2_inversion_gradient(
            ESMFold2InversionGradientInput(inputs=candidate),
            ESMFold2InversionGradientConfig(
                gradient_terms=[ESMFold2IntraContactTerm(chain_id="B"), ESMFold2GlobularityTerm(chain_id="C")]
            ),
        )
    result = output.results[0]
    assert "A" not in result.chain_rows, "the fixed chain owns no gradient rows"
    assert sum(stop - start for start, stop in result.chain_rows.values()) == rows
    assert rows == len(CHAIN_B) + len(CHAIN_C), "rows cover exactly the designable chains"


def test_a_term_without_a_chain_id_is_ambiguous_when_several_chains_are_designed() -> None:
    """The convenience default only works when there is one designable chain."""
    with patch(_GENERAL_DISPATCH):
        with pytest.raises(ValueError, match="must name a chain_id"):
            run_esmfold2_inversion_gradient(
                ESMFold2InversionGradientInput(inputs=_three_chains()),
                ESMFold2InversionGradientConfig(gradient_terms=[ESMFold2IntraContactTerm()]),
            )


def test_a_term_naming_an_absent_chain_is_rejected() -> None:
    """A stale chain id should fail loudly rather than score something else."""
    with patch(_GENERAL_DISPATCH):
        with pytest.raises(ValueError, match="which this complex does not have"):
            run_esmfold2_inversion_gradient(
                ESMFold2InversionGradientInput(inputs=_three_chains()),
                ESMFold2InversionGradientConfig(gradient_terms=[ESMFold2IntraContactTerm(chain_id="Z")]),
            )


def test_a_single_designable_chain_needs_no_chain_id() -> None:
    """The common case stays simple: one designable chain resolves on its own."""
    candidate = ESMFold2ComplexCandidate(
        chains=[
            ESMFold2Chain(chain_id="A", sequence=TARGET),
            ESMFold2Chain(chain_id="B", logits=one_hot_protein_logits(CHAIN_B, sharpness=2.0)),
        ]
    )
    with patch(_GENERAL_DISPATCH) as dispatch:
        dispatch.return_value = {
            "terms": [{"id": "intra_contact", "chain_id": "B", "value": 1.0, "gradient": [[0.1] * 20] * len(CHAIN_B)}],
            "chain_rows": {"B": [0, len(CHAIN_B)]},
            "metrics": {"complex_length": len(TARGET) + len(CHAIN_B), "designable_length": len(CHAIN_B)},
        }
        run_esmfold2_inversion_gradient(
            ESMFold2InversionGradientInput(inputs=candidate),
            ESMFold2InversionGradientConfig(gradient_terms=[ESMFold2IntraContactTerm()]),
        )
    assert dispatch.call_args.args[1]["gradient_terms"] == [{"id": "intra_contact", "chain_id": "B"}]


def test_binder_dispatch_folds_target_then_binder() -> None:
    """The binder tool is the two-chain case of the shared path, with fixed roles."""
    reversed_alphabet = PROTEIN_AMINO_ACIDS[::-1]
    with patch(_BINDER_DISPATCH) as dispatch:
        dispatch.return_value = {
            "terms": [
                {"id": "intra_contact", "chain_id": "binder", "value": 1.5, "gradient": [[0.1] * 20] * len(BINDER)},
                {"id": "inter_contact", "chain_id": "binder", "value": 2.5, "gradient": [[0.2] * 20] * len(BINDER)},
            ],
            "chain_rows": {"binder": [0, len(BINDER)]},
            "metrics": {},
        }
        output = run_esmfold2_inversion_binder_gradient(
            ESMFold2BinderGradientInput(inputs=_binder(alphabet=reversed_alphabet), temperature=0.7),
            ESMFold2BinderGradientConfig(gradient_terms=["intra_contact", "inter_contact"]),
        )

    payload = dispatch.call_args.args[1]
    assert [chain["chain_id"] for chain in payload["chains"]] == ["target", "binder"]
    assert payload["target_chain_id"] == "target"
    assert payload["alphabet"] == reversed_alphabet
    assert output.vocab == list(reversed_alphabet), "the gradient is columned the way the caller asked"

    intra, inter = output.results[0].terms
    assert (intra.term.id, inter.term.id) == ("intra_contact", "inter_contact")
    assert output.results[0].metrics["binder_length"] == len(BINDER)
    assert intra.metadata.input_fingerprint == inter.metadata.input_fingerprint, "one fold, one point"


def test_binder_epitope_quantity_requires_hotspots() -> None:
    """Pydantic cannot see across input and config, so the pairing is checked in the run function."""
    with pytest.raises(ValueError, match="needs target_hotspot_ids"):
        run_esmfold2_inversion_binder_gradient(
            ESMFold2BinderGradientInput(inputs=_binder()),
            ESMFold2BinderGradientConfig(gradient_terms=["epitope"]),
        )


@pytest.mark.parametrize(
    ("hotspot", "expected"),
    [("L999", "outside the target"), ("A2", "does not match the target"), ("15", "residue letter")],
)
def test_binder_rejects_hotspots_the_target_contradicts(hotspot: str, expected: str) -> None:
    """A hotspot names the residue it expects, so a stale selection fails rather than drifting."""
    with pytest.raises(ValidationError, match=expected):
        _binder(target_hotspot_ids=[hotspot])


def test_terms_compose_without_goals_because_every_quantity_is_a_loss() -> None:
    """These are losses, so the advisory direction resolves on its own and weights are the knob."""
    rows = len(CHAIN_B) + len(CHAIN_C)
    with patch(_GENERAL_DISPATCH) as dispatch:
        dispatch.return_value = _general_reply(rows, {"B": [0, len(CHAIN_B)], "C": [len(CHAIN_B), rows]})
        output = run_esmfold2_inversion_gradient(
            ESMFold2InversionGradientInput(inputs=_three_chains()),
            ESMFold2InversionGradientConfig(
                gradient_terms=[ESMFold2IntraContactTerm(chain_id="B"), ESMFold2GlobularityTerm(chain_id="C")]
            ),
        )
    result = output.results[0]
    composed = result.compose()
    assert [c.dloss_dvalue for c in composed.contributions] == [1.0, 1.0], "minimizing, by default"
    assert composed.objective_loss == pytest.approx(4.0)
    weighted = result.compose(weights={result.terms[0].term: 0.5})
    assert weighted.objective_loss == pytest.approx(0.5 * 1.5 + 2.5)


def test_forward_only_keeps_the_values_and_drops_the_gradients() -> None:
    """compute_gradient=False still reports every quantity, just without derivatives."""
    rows = len(CHAIN_B) + len(CHAIN_C)
    reply = _general_reply(rows, {"B": [0, len(CHAIN_B)], "C": [len(CHAIN_B), rows]})
    for term in reply["terms"]:
        term["gradient"] = None
    with patch(_GENERAL_DISPATCH) as dispatch:
        dispatch.return_value = reply
        output = run_esmfold2_inversion_gradient(
            ESMFold2InversionGradientInput(inputs=_three_chains()),
            ESMFold2InversionGradientConfig(
                gradient_terms=[ESMFold2IntraContactTerm(chain_id="B"), ESMFold2GlobularityTerm(chain_id="C")],
                compute_gradient=False,
            ),
        )
    assert dispatch.call_args.args[1]["compute_gradient"] is False
    for term in output.results[0].terms:
        assert term.gradient is None and term.grad_l2_norm is None
        assert term.term_value > 0.0


# ---------------------------------------------------------------------------
# Integration tests


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_binder_inversion_on_a_real_checkpoint() -> None:
    """A real checkpoint returns a finite gradient over the binder's own rows."""
    output = run_esmfold2_inversion_binder_gradient(
        ESMFold2BinderGradientInput(inputs=_binder()),
        ESMFold2BinderGradientConfig(gradient_terms=["intra_contact", "inter_contact", "globularity"], seed=0),
    )
    (result,) = output.results
    assert [term.term.id for term in result.terms] == ["intra_contact", "inter_contact", "globularity"]
    for term in result.terms:
        assert term.gradient is not None
        assert len(term.gradient) == len(BINDER), "one gradient row per designed residue"
        assert all(len(row) == 20 for row in term.gradient)


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_three_chain_complex_differentiates_both_designable_chains() -> None:
    """The whole point of the general tool: two chains designed at once, in one fold.

    Each gradient must cover both chains' rows, and each designable chain must actually
    influence the computation — a chain that is folded but disconnected would show all zeros.
    """
    output = run_esmfold2_inversion_gradient(
        ESMFold2InversionGradientInput(inputs=_three_chains()),
        ESMFold2InversionGradientConfig(
            gradient_terms=[ESMFold2IntraContactTerm(chain_id="B"), ESMFold2IntraContactTerm(chain_id="C")],
            seed=0,
        ),
    )
    (result,) = output.results
    rows = len(CHAIN_B) + len(CHAIN_C)
    assert result.chain_rows == {"B": [0, len(CHAIN_B)], "C": [len(CHAIN_B), rows]}
    assert "A" not in result.chain_rows, "no gradient rows for the fixed chain"

    for term in result.terms:
        assert term.gradient is not None
        assert len(term.gradient) == rows, "a gradient spans every designable row"

    # Each chain's own objective must be non-zero over that chain's rows.
    for term in result.terms:
        start, stop = result.chain_rows[term.term.chain_id]
        own = term.gradient[start:stop]
        assert any(value != 0.0 for row in own for value in row), (
            f"chain {term.term.chain_id} received no gradient from its own objective"
        )


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_changing_a_designable_chain_changes_its_own_objective() -> None:
    """Both designable chains must really reach the fold, not just the first.

    Each chain is checked against its *own* objective. A chain-local quantity is only weakly
    coupled to the rest of the complex, and at the deployed bf16 precision that coupling is
    often below the representable step, so asserting one chain moves another's value would be
    asserting more than the design promises.
    """

    def globularity_of(chain_id: str, chains: list[ESMFold2Chain]) -> float:
        output = run_esmfold2_inversion_gradient(
            ESMFold2InversionGradientInput(inputs=ESMFold2ComplexCandidate(chains=chains)),
            ESMFold2InversionGradientConfig(gradient_terms=[ESMFold2GlobularityTerm(chain_id=chain_id)], seed=0),
        )
        return output.results[0].terms[0].term_value

    def complex_with(changed: str | None) -> list[ESMFold2Chain]:
        sequences = {"B": CHAIN_B, "C": CHAIN_C}
        if changed is not None:
            sequences[changed] = sequences[changed][::-1]
        return [
            ESMFold2Chain(chain_id="A", sequence=TARGET),
            ESMFold2Chain(chain_id="B", logits=one_hot_protein_logits(sequences["B"], sharpness=2.0)),
            ESMFold2Chain(chain_id="C", logits=one_hot_protein_logits(sequences["C"], sharpness=2.0)),
        ]

    baseline = complex_with(None)
    for chain_id in ("B", "C"):
        before = globularity_of(chain_id, baseline)
        after = globularity_of(chain_id, complex_with(chain_id))
        assert after != before, f"changing chain {chain_id} did not reach the fold"


@pytest.mark.benchmark("esmfold2-inversion-gradient")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_esmfold2_inversion_gradient_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark: one fold of a three-chain complex plus a backward per term (cold + warm)."""
    inputs = ESMFold2InversionGradientInput(inputs=_three_chains())
    config = ESMFold2InversionGradientConfig(
        gradient_terms=[ESMFold2IntraContactTerm(chain_id="B"), ESMFold2GlobularityTerm(chain_id="C")], seed=0
    )
    result = benchmark_twice(request, "esmfold2", lambda: run_esmfold2_inversion_gradient(inputs, config))
    assert isinstance(result, ESMFold2InversionGradientOutput)
    assert all(term.gradient is not None for term in result.results[0].terms)


@pytest.mark.benchmark("esmfold2-inversion-binder-gradient")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_esmfold2_inversion_binder_gradient_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark: one fold of a binder/target complex plus a backward per term (cold + warm)."""
    inputs = ESMFold2BinderGradientInput(inputs=_binder())
    config = ESMFold2BinderGradientConfig(gradient_terms=["intra_contact", "inter_contact", "globularity"], seed=0)
    result = benchmark_twice(request, "esmfold2", lambda: run_esmfold2_inversion_binder_gradient(inputs, config))
    assert result.tool_id == "esmfold2-inversion-binder-gradient"
    assert all(term.gradient is not None for term in result.results[0].terms)


# ── Nested relaxed state: discovery, reconstruction, independence ─────────────


def test_nested_matrices_are_discovered_by_their_paths() -> None:
    """The shared framework finds relaxed state by walking paths, and a complex nests its own."""
    gradient_input = ESMFold2InversionGradientInput(inputs=_three_chains())
    found = gradient_input.sequence_logits()
    assert sorted(found) == ["inputs[0].chains[1].logits", "inputs[0].chains[2].logits"]
    assert found["inputs[0].chains[1].logits"].root == _three_chains().chains[1].logits.root
    # The fixed chain owns no relaxed state, so it contributes no path.
    assert not any("chains[0]" in path for path in found)


def test_nested_matrices_round_trip_through_the_dumped_model() -> None:
    """Each reported path must address the same matrix in the JSON shape a request carries."""
    gradient_input = ESMFold2InversionGradientInput(inputs=_three_chains())
    dumped = gradient_input.model_dump(exclude_none=True)
    for path, matrix in gradient_input.sequence_logits().items():
        cursor: Any = dumped
        for part in path.split("."):
            if part.endswith("]"):
                name, _, index = part[:-1].partition("[")
                cursor = cursor[name][int(index)]
            else:
                cursor = cursor[part]
        assert cursor == matrix.root


def test_several_designable_chains_rebuild_from_their_own_distributions() -> None:
    """from_probabilities must place each distribution on the chain it was reported under."""
    candidate = _three_chains()
    base = ESMFold2InversionGradientInput(inputs=candidate)
    paths = list(base.sequence_logits())
    # Two deliberately different distributions, so a factory that reused one would fail here.
    first = [[0.9 if column == 0 else 0.1 / 19 for column in range(20)] for _ in range(len(CHAIN_B))]
    second = [[0.9 if column == 5 else 0.1 / 19 for column in range(20)] for _ in range(len(CHAIN_C))]

    built = ESMFold2InversionGradientInput.from_probabilities(
        [
            {
                paths[0].removeprefix("inputs[0]."): first,
                paths[1].removeprefix("inputs[0]."): second,
                "chains": candidate.model_dump(exclude_none=True)["chains"],
            }
        ],
        temperature=1.0,
    )
    rebuilt = built.inputs[0]
    assert [chain.chain_id for chain in rebuilt.chains] == ["A", "B", "C"]
    assert rebuilt.chains[0].sequence == TARGET, "the fixed chain is carried through untouched"
    assert rebuilt.chains[1].logits.root != rebuilt.chains[2].logits.root, "each chain kept its own"
    # Each chain's logits must reproduce the distribution it was built from.
    for chain, target in ((rebuilt.chains[1], first), (rebuilt.chains[2], second)):
        for row, expected in zip(chain.logits.root, target, strict=True):
            peak = max(row)
            weights = [pow(2.718281828459045, value - peak) for value in row]
            total = sum(weights)
            assert [weight / total for weight in weights] == pytest.approx(expected, rel=1e-6)


def test_each_designable_chain_is_differentiated_independently() -> None:
    """A term on one chain must not be credited with another chain's rows by construction."""
    rows = len(CHAIN_B) + len(CHAIN_C)
    chain_rows = {"B": [0, len(CHAIN_B)], "C": [len(CHAIN_B), rows]}
    # The worker reports a gradient that is non-zero only over the chain the term scored.
    reply = {
        "terms": [
            {
                "id": "intra_contact",
                "chain_id": "B",
                "value": 1.0,
                "gradient": [[0.5] * 20] * len(CHAIN_B) + [[0.0] * 20] * len(CHAIN_C),
            },
            {
                "id": "intra_contact",
                "chain_id": "C",
                "value": 2.0,
                "gradient": [[0.0] * 20] * len(CHAIN_B) + [[0.7] * 20] * len(CHAIN_C),
            },
        ],
        "chain_rows": chain_rows,
        "metrics": {"complex_length": len(TARGET) + rows, "designable_length": rows},
    }
    with patch(_GENERAL_DISPATCH) as dispatch:
        dispatch.return_value = reply
        output = run_esmfold2_inversion_gradient(
            ESMFold2InversionGradientInput(inputs=_three_chains()),
            ESMFold2InversionGradientConfig(
                gradient_terms=[ESMFold2IntraContactTerm(chain_id="B"), ESMFold2IntraContactTerm(chain_id="C")]
            ),
        )
    for term in output.results[0].terms:
        own_start, own_stop = chain_rows[term.term.chain_id]
        own = term.gradient[own_start:own_stop]
        other = term.gradient[:own_start] + term.gradient[own_stop:]
        assert any(value != 0.0 for row in own for value in row), "its own chain carries the gradient"
        assert all(value == 0.0 for row in other for value in row), "the other chain's rows stay zero"
    # Both terms describe the same point, so a caller can weigh the two chains against each other.
    composed = output.results[0].compose()
    assert composed.objective_loss == pytest.approx(3.0)
    assert len(composed.gradient) == rows


def test_distogram_iptm_proxy_spans_zero_to_one_and_reads_only_the_interface_block() -> None:
    """Certain of one close bin scores 1 and a uniform distribution 0, both from binder rows only."""
    torch = pytest.importorskip("torch")
    objective = _objective()
    bins = objective.DISTOGRAM_BINS
    midpoints = objective.bin_midpoints(bins).double()
    target, binder = (0, 12), (12, 20)
    sharp = torch.full((1, 20, 20, bins), -50.0, dtype=torch.float64)
    sharp[..., 5] = 50.0
    flat = torch.zeros((1, 20, 20, bins), dtype=torch.float64)
    assert objective.distogram_iptm_proxy(sharp, target, binder, midpoints).item() == pytest.approx(1.0, abs=1e-6)
    assert objective.distogram_iptm_proxy(flat, target, binder, midpoints).item() == pytest.approx(0.0, abs=1e-6)

    torch.manual_seed(5)
    logits = torch.randn(1, 20, 20, bins, dtype=torch.float64)
    base = objective.distogram_iptm_proxy(logits, target, binder, midpoints).item()
    edited = logits.clone()
    edited[:, :12, :12, :] = 0.0  # target-target
    edited[:, 12:, 12:, :] = 0.0  # binder-binder
    edited[:, :12, 12:, :] = 0.0  # target rows against binder columns, the other direction
    assert objective.distogram_iptm_proxy(edited, target, binder, midpoints).item() == pytest.approx(base)


def test_binder_tool_reports_the_distogram_iptm_proxy_when_the_worker_returns_it() -> None:
    """The proxy travels from the worker's metrics into the result's metrics."""
    with patch(_BINDER_DISPATCH) as dispatch:
        dispatch.return_value = {
            "terms": [
                {"id": "inter_contact", "chain_id": "binder", "value": 2.5, "gradient": None},
            ],
            "chain_rows": {"binder": [0, len(BINDER)]},
            "metrics": {"distogram_iptm_proxy": 0.42},
        }
        output = run_esmfold2_inversion_binder_gradient(
            ESMFold2BinderGradientInput(inputs=_binder(), temperature=1.0),
            ESMFold2BinderGradientConfig(gradient_terms=["inter_contact"], compute_gradient=False),
        )
    assert output.results[0].metrics["distogram_iptm_proxy"] == pytest.approx(0.42)


def _inference() -> Any:
    """Load the worker inference module, stubbing the helpers it expects from the tool env."""
    pytest.importorskip("torch")
    sys.modules.setdefault(
        "standalone_helpers",
        types.SimpleNamespace(
            get_logger=logging.getLogger,
            get_random_int=lambda *a, **k: 0,
            is_cuda_oom=lambda *a, **k: False,
            release_cuda_memory=lambda *a, **k: None,
            report_msa_rows_loaded=lambda *a, **k: None,
            serialize_output=lambda value: value,
            set_torch_seed=lambda *a, **k: None,
        ),
    )
    sys.modules.setdefault("objective", _objective())
    path = _OBJECTIVE_PATH.parent / "inference.py"
    spec = importlib.util.spec_from_file_location("esmfold2_inference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_each_checkpoint_family_gets_its_own_distance_grid() -> None:
    """Range, edge convention and what a contact cutoff leaves unmasked, per head."""
    objective = _objective()
    production, experimental = objective.bin_midpoints(64), objective.bin_midpoints(128)
    # Production: linspace(2, 22, 65) with both ends opened out. Experimental: 127 interior
    # boundaries over [2, 52], plus the two open-ended outer bins.
    assert (float(production[0]), float(production[-1])) == pytest.approx((1.65625, 24.34375))
    assert (float(experimental[0]), float(experimental[-1])) == pytest.approx((1.5, 54.5))
    # The 22 A cutoff leaves the production head nothing to discriminate with; the 14 A one is
    # closer, but still masks far less than on the experimental head.
    assert (int((production < 22.0).sum()), int((experimental < 22.0).sum())) == (63, 51)
    assert (int((production < 14.0).sum()), int((experimental < 14.0).sum())) == (38, 31)
    with pytest.raises(ValueError, match="silently read the wrong distances"):
        objective.bin_midpoints(100)
    # A bfloat16 head is still scored at float32: bfloat16 carries 7 mantissa bits, so storing
    # the grid in it would quantize the angstrom values the cutoffs compare against.
    torch = pytest.importorskip("torch")
    assert production.dtype == torch.float32
    distogram = torch.zeros(1, 8, 8, 128, dtype=torch.bfloat16)
    assert objective.intra_contact_loss(distogram, (4, 8), experimental).dtype == torch.float32


def test_the_production_grid_is_unchanged_by_the_grid_table() -> None:
    """Existing inversion users must get exactly the distances they got before."""
    torch = pytest.importorskip("torch")
    objective = _objective()
    edges = torch.linspace(2.0, 22.0, 65)
    edges[0], edges[-1] = 1.0, 27.0
    assert torch.equal(((edges[:-1] + edges[1:]) / 2).double(), objective.bin_midpoints(64).double())


def test_lm_dropout_applies_only_to_the_experimental_checkpoints() -> None:
    """Dropout is an experimental-head setting, and is baked in at load."""
    inference = _inference()
    assert len(inference.EXPERIMENTAL_CHECKPOINTS) == 4
    assert all(
        inference._CHECKPOINT_REPOS[k].startswith("biohub/ESMFold2-Experimental")
        for k in inference.EXPERIMENTAL_CHECKPOINTS
    )
    with pytest.raises(ValueError, match="lm_dropout is only configurable"):
        inference.ESMFold2Model(model_checkpoint="esmfold2-fast", lm_dropout=0.5)
    # A different dropout must not reuse the cached model, since it cannot be changed after load.
    first = inference._wrapper("esmfold2-experimental-fast", 0.0)
    assert inference._wrapper("esmfold2-experimental-fast", 0.5) is not first
