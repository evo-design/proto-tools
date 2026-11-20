"""In-env numerical checks for the ESMFold2 inversion objectives.

These exercise the standalone ``objective`` and ``inference`` modules, which import
``torch`` and so are not importable in the base test environment. They run **inside**
the built esmfold2 standalone env via :func:`proto_tools.utils.run_in_env`, invoked by
``test_objective_numerics_run_in_env`` in ``test_esmfold2_inversion.py``.

Running them here rather than in the base process is what makes them run at all. Guarded
with ``pytest.importorskip("torch")`` they skipped silently everywhere -- in CI, which is
``pytest --cpu-only --no-envs``, and in an ordinary local run too -- which is how eight of
them came to call ``bin_midpoints()`` with no argument, long after the signature gained
``n_bins``, without anyone noticing.

``pytest`` is not installed in the standalone env, so these are plain-``assert`` checks.
``main()`` runs every check, prints a per-check status line, and exits non-zero if any
fail (which ``run_in_env`` surfaces to the caller as ``RuntimeError``). The leading ``_``
keeps pytest from collecting this file in the base process.

Everything here is float64 maths on synthetic distograms: no GPU, no weights, no network.
The real-checkpoint paths are covered by the ``uses_gpu`` tests in the base env.
"""

from __future__ import annotations

import traceback
from typing import Any

import inference  # standalone module; on PYTHONPATH inside the esmfold2 env
import objective  # standalone module; on PYTHONPATH inside the esmfold2 env
import torch


def _approx(actual: float, expected: float, rel: float = 1e-6, abs_: float = 1e-12) -> bool:
    """Whether two floats agree within a relative or absolute tolerance."""
    return abs(actual - expected) <= max(abs_, rel * abs(expected))


def check_bin_midpoints_span_the_distogram() -> None:
    """The outer bins are open-ended, so their midpoints sit beyond the interior boundaries."""
    midpoints = objective.bin_midpoints(128)
    assert midpoints.shape == (128,)
    assert midpoints[0] < 2.0 < midpoints[1]
    assert torch.all(midpoints.diff() > 0), "midpoints must increase with distance"


def check_span_mask_rejects_a_span_outside_the_complex() -> None:
    """A chain span that does not fit would silently score the wrong rows."""
    try:
        objective.span_mask(10, (5, 20), torch.device("cpu"))
    except ValueError as error:
        assert "does not fit" in str(error), error
    else:
        raise AssertionError("span_mask accepted a span that does not fit the complex")


def check_intra_contact_ignores_contacts_across_a_chain_boundary() -> None:
    """A residue must not satisfy its own chain's objective by touching a neighbouring chain."""
    midpoints = objective.bin_midpoints(128).double()
    torch.manual_seed(0)
    logits = torch.randn(1, 44, 44, 128, dtype=torch.float64)
    span = (20, 44)

    baseline = objective.intra_contact_loss(logits, span, midpoints)
    crossed = logits.clone()
    # Make every cross-boundary pair look like a tight contact.
    crossed[0, 20:44, 0:20, :40] += 8.0
    crossed[0, 0:20, 20:44, :40] += 8.0
    assert _approx(objective.intra_contact_loss(crossed, span, midpoints).item(), baseline.item()), (
        "contacts outside the chain must not count toward its intra-chain objective"
    )


def check_globularity_reads_only_its_own_chain() -> None:
    """Compactness of one chain cannot be improved by rearranging another."""
    midpoints = objective.bin_midpoints(128).double()
    torch.manual_seed(1)
    logits = torch.randn(1, 44, 44, 128, dtype=torch.float64)
    span = (20, 44)

    baseline = objective.globularity_loss(logits, span, midpoints)
    elsewhere = logits.clone()
    elsewhere[0, 0:20, 0:20, :40] += 8.0
    assert _approx(objective.globularity_loss(elsewhere, span, midpoints).item(), baseline.item())


def check_globularity_differs_between_chains_of_the_same_complex() -> None:
    """Each chain gets its own radius of gyration, not the complex's."""
    midpoints = objective.bin_midpoints(128).double()
    torch.manual_seed(2)
    logits = torch.randn(1, 44, 44, 128, dtype=torch.float64)
    first = objective.globularity_loss(logits, (0, 20), midpoints).item()
    second = objective.globularity_loss(logits, (20, 44), midpoints).item()
    assert first != second, "two different chains should not score identically by construction"


def check_chain_objective_gradients_match_a_finite_difference() -> None:
    """The analytic gradient is what an optimizer steps along, so it is checked numerically.

    In float64 against a synthetic distogram. The real checkpoint runs in bf16, where a
    difference this small does not converge; that path is checked for direction instead.
    """
    for quantity in ("intra_contact_loss", "globularity_loss"):
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

        numeric = (high - low) / (2 * epsilon)
        assert _approx(analytic, numeric, rel=1e-5, abs_=1e-9), f"{quantity}: {analytic} != {numeric}"


def check_a_descent_step_lowers_each_chain_objective() -> None:
    """Stepping against the gradient must improve the objective, or its sign is wrong."""
    span = (16, 40)
    for quantity in ("intra_contact_loss", "globularity_loss"):
        torch.manual_seed(1)
        logits = torch.randn(1, 40, 40, 128, dtype=torch.float64, requires_grad=True)
        midpoints = objective.bin_midpoints(128).double()
        before = getattr(objective, quantity)(logits, span, midpoints)
        before.sum().backward()
        after = getattr(objective, quantity)(logits.detach() - 0.1 * logits.grad, span, midpoints)
        assert after.item() < before.item(), f"{quantity} did not improve along its own gradient"


def check_inter_contact_stays_asymmetric() -> None:
    """Swapping the two spans must change the value; it scores the target, not the binder."""
    midpoints = objective.bin_midpoints(128).double()
    torch.manual_seed(3)
    logits = torch.randn(1, 44, 44, 128, dtype=torch.float64)
    forward = objective.inter_contact_loss(logits, (0, 20), (20, 44), midpoints).item()
    swapped = objective.inter_contact_loss(logits, (20, 44), (0, 20), midpoints).item()
    assert forward != swapped, "inter_contact is deliberately not a symmetric inter-chain objective"


def check_intra_contact_is_inert_when_no_pair_clears_the_separation_floor() -> None:
    """A chain shorter than the separation floor has no eligible pair, so the objective is zero.

    Worth pinning because a silent zero here once made an equivalence check pass vacuously.
    """
    torch.manual_seed(4)
    logits = torch.randn(1, 20, 20, 128, dtype=torch.float64)
    midpoints = objective.bin_midpoints(128).double()
    assert objective.intra_contact_loss(logits, (12, 20), midpoints).item() == 0.0
    assert objective.intra_contact_loss(logits, (0, 20), midpoints).item() != 0.0


def check_straight_through_is_one_hot_forward_and_relaxed_backward() -> None:
    """The language model scores discrete tokens; the gradient still has to reach the logits."""
    relaxed = torch.tensor([[0.1, 0.7, 0.2]], dtype=torch.float64, requires_grad=True)
    passed = objective.straight_through(relaxed)
    assert passed.tolist() == [[0.0, 1.0, 0.0]], "forward pass is the argmax"
    passed.sum().backward()
    assert relaxed.grad.tolist() == [[1.0, 1.0, 1.0]], "backward pass reaches every column"


def check_distogram_iptm_proxy_spans_zero_to_one_and_reads_only_the_interface_block() -> None:
    """Certain of one close bin scores 1 and a uniform distribution 0, both from binder rows only."""
    bins = objective.DISTOGRAM_BINS
    midpoints = objective.bin_midpoints(bins).double()
    target, binder = (0, 12), (12, 20)
    sharp = torch.full((1, 20, 20, bins), -50.0, dtype=torch.float64)
    sharp[..., 5] = 50.0
    flat = torch.zeros((1, 20, 20, bins), dtype=torch.float64)
    assert _approx(objective.distogram_iptm_proxy(sharp, target, binder, midpoints).item(), 1.0, abs_=1e-6)
    assert _approx(objective.distogram_iptm_proxy(flat, target, binder, midpoints).item(), 0.0, abs_=1e-6)

    torch.manual_seed(5)
    logits = torch.randn(1, 20, 20, bins, dtype=torch.float64)
    base = objective.distogram_iptm_proxy(logits, target, binder, midpoints).item()
    edited = logits.clone()
    edited[:, :12, :12, :] = 0.0  # target-target
    edited[:, 12:, 12:, :] = 0.0  # binder-binder
    edited[:, :12, 12:, :] = 0.0  # target rows against binder columns, the other direction
    assert _approx(objective.distogram_iptm_proxy(edited, target, binder, midpoints).item(), base)


def check_each_checkpoint_family_gets_its_own_distance_grid() -> None:
    """Range, edge convention and what a contact cutoff leaves unmasked, per head."""
    production, experimental = objective.bin_midpoints(64), objective.bin_midpoints(128)
    # Production: linspace(2, 22, 65) with both ends opened out. Experimental: 127 interior
    # boundaries over [2, 52], plus the two open-ended outer bins.
    assert _approx(float(production[0]), 1.65625) and _approx(float(production[-1]), 24.34375)
    assert _approx(float(experimental[0]), 1.5) and _approx(float(experimental[-1]), 54.5)
    # The 22 A cutoff leaves the production head nothing to discriminate with; the 14 A one is
    # closer, but still masks far less than on the experimental head.
    assert (int((production < 22.0).sum()), int((experimental < 22.0).sum())) == (63, 51)
    assert (int((production < 14.0).sum()), int((experimental < 14.0).sum())) == (38, 31)
    try:
        objective.bin_midpoints(100)
    except ValueError as error:
        assert "silently read the wrong distances" in str(error), error
    else:
        raise AssertionError("an unknown head was scored on a guessed grid")
    # A bfloat16 head is still scored at float32: bfloat16 carries 7 mantissa bits, so storing
    # the grid in it would quantize the angstrom values the cutoffs compare against.
    assert production.dtype == torch.float32
    distogram = torch.zeros(1, 8, 8, 128, dtype=torch.bfloat16)
    assert objective.intra_contact_loss(distogram, (4, 8), experimental).dtype == torch.float32


def check_the_production_grid_is_unchanged_by_the_grid_table() -> None:
    """Existing inversion users must get exactly the distances they got before."""
    edges = torch.linspace(2.0, 22.0, 65)
    edges[0], edges[-1] = 1.0, 27.0
    assert torch.equal(((edges[:-1] + edges[1:]) / 2).double(), objective.bin_midpoints(64).double())


def check_lm_dropout_applies_only_to_the_experimental_checkpoints() -> None:
    """Dropout is an experimental-head setting, and is baked in at load."""
    assert len(inference.EXPERIMENTAL_CHECKPOINTS) == 4
    assert all(
        inference._CHECKPOINT_REPOS[k].startswith("biohub/ESMFold2-Experimental")
        for k in inference.EXPERIMENTAL_CHECKPOINTS
    )
    try:
        inference.ESMFold2Model(model_checkpoint="esmfold2-fast", lm_dropout=0.5)
    except ValueError as error:
        assert "lm_dropout is only configurable" in str(error), error
    else:
        raise AssertionError("lm_dropout was accepted on a production checkpoint")
    # A different dropout must not reuse the cached model, since it cannot be changed after load.
    first = inference._wrapper("esmfold2-experimental-fast", 0.0)
    assert inference._wrapper("esmfold2-experimental-fast", 0.5) is not first


CHECKS = [
    check_bin_midpoints_span_the_distogram,
    check_span_mask_rejects_a_span_outside_the_complex,
    check_intra_contact_ignores_contacts_across_a_chain_boundary,
    check_globularity_reads_only_its_own_chain,
    check_globularity_differs_between_chains_of_the_same_complex,
    check_chain_objective_gradients_match_a_finite_difference,
    check_a_descent_step_lowers_each_chain_objective,
    check_inter_contact_stays_asymmetric,
    check_intra_contact_is_inert_when_no_pair_clears_the_separation_floor,
    check_straight_through_is_one_hot_forward_and_relaxed_backward,
    check_distogram_iptm_proxy_spans_zero_to_one_and_reads_only_the_interface_block,
    check_each_checkpoint_family_gets_its_own_distance_grid,
    check_the_production_grid_is_unchanged_by_the_grid_table,
    check_lm_dropout_applies_only_to_the_experimental_checkpoints,
]


def _run_check(check: Any) -> str | None:
    """Run one check; print PASS/FAIL and return the check's name on failure, else None."""
    try:
        check()
    except Exception as error:
        print(f"FAIL {check.__name__}: {type(error).__name__}: {error}")
        traceback.print_exc()
        return str(check.__name__)
    print(f"PASS {check.__name__}")
    return None


def main() -> int:
    """Run every check, report per-check status, and return a non-zero code on any failure."""
    failures = [name for name in (_run_check(check) for check in CHECKS) if name is not None]
    if failures:
        print(f"FAILED {len(failures)}/{len(CHECKS)}: {failures}")
        return 1
    print(f"ALL CHECKS PASSED ({len(CHECKS)}/{len(CHECKS)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
