"""tests/tool_infra_tests/_relaxation_env_checks.py.

Numerical checks for ``standalone_helpers.relaxation.relaxed_sequence``, run inside a built
tool environment by ``test_relaxation_formula_runs_in_env`` because the base process has no
torch.

These need no model weights and no GPU — the formula is pure arithmetic, so the checks cover
the forward values of every documented blend plus the backward route through autograd. Plain
asserts, one line of status per check, non-zero exit on any failure (which ``run_in_env``
surfaces as ``RuntimeError``).
"""

from __future__ import annotations

import traceback
from collections.abc import Callable

import torch
import torch.nn.functional as F
from standalone_helpers.relaxation import relaxed_sequence

#  A deliberately asymmetric state so a wrong branch cannot coincide with a right one.
LOGITS = [[2.0, 0.5, -1.0, 0.25], [-0.75, 1.5, 0.0, 3.0], [0.1, 0.1, 0.1, 0.1]]
TEMPERATURE = 0.7


def _state(requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(LOGITS, dtype=torch.float64, requires_grad=requires_grad)


def check_default_is_a_plain_softmax() -> None:
    """soft=1, hard=0 must equal softmax(logits / temperature) exactly."""
    state = _state()
    expected = F.softmax(state / TEMPERATURE, dim=-1)

    actual = relaxed_sequence(state, temperature=TEMPERATURE)

    assert torch.allclose(actual, expected, atol=1e-12), f"{actual} != {expected}"


def check_soft_zero_passes_the_raw_matrix_through() -> None:
    """soft=0, hard=0 must forward the caller's own matrix, not an argmax one-hot.

    This is the branch that differs from the retired proto convention, where soft=0 selected
    one_hot(argmax(probs)).
    """
    state = _state()

    actual = relaxed_sequence(state, temperature=TEMPERATURE, soft=0.0, hard=0.0)

    assert torch.allclose(actual, state, atol=1e-12), f"{actual} != {state}"
    one_hot = F.one_hot(F.softmax(state / TEMPERATURE, dim=-1).argmax(dim=-1), num_classes=4).double()
    assert not torch.allclose(actual, one_hot, atol=1e-6), "soft=0 returned a one-hot, the old convention"


def check_hard_one_forwards_a_one_hot() -> None:
    """soft=1, hard=1 must forward exactly one_hot(argmax(P))."""
    state = _state()
    probs = F.softmax(state / TEMPERATURE, dim=-1)
    expected = F.one_hot(probs.argmax(dim=-1), num_classes=4).double()

    actual = relaxed_sequence(state, temperature=TEMPERATURE, soft=1.0, hard=1.0)

    assert torch.allclose(actual, expected, atol=1e-12), f"{actual} != {expected}"


def check_hard_one_supersedes_soft() -> None:
    """At hard=1 the forward value must not depend on soft, for a fixed P."""
    values = [relaxed_sequence(_state(), temperature=TEMPERATURE, soft=soft, hard=1.0) for soft in (0.0, 0.5, 1.0)]

    for value in values[1:]:
        assert torch.allclose(value, values[0], atol=1e-12), "hard=1 forward depends on soft"


def check_intermediate_blends_match_the_formula() -> None:
    """Every (soft, hard) pair must equal the documented two-step blend."""
    for soft in (0.0, 0.25, 0.5, 1.0):
        for hard in (0.0, 0.5, 1.0):
            state = _state()
            probs = F.softmax(state / TEMPERATURE, dim=-1)
            discrete = F.one_hot(probs.argmax(dim=-1), num_classes=4).double()
            blended = (1.0 - soft) * state + soft * probs
            expected = (1.0 - hard) * blended + hard * discrete

            actual = relaxed_sequence(state, temperature=TEMPERATURE, soft=soft, hard=hard)

            assert torch.allclose(actual, expected, atol=1e-12), f"soft={soft} hard={hard}: {actual} != {expected}"


def check_probability_mode_skips_the_softmax() -> None:
    """temperature=None must consume the supplied matrix unchanged at the default controls."""
    probabilities = torch.tensor([[0.7, 0.2, 0.05, 0.05], [0.1, 0.1, 0.1, 0.7]], dtype=torch.float64)

    actual = relaxed_sequence(probabilities, temperature=None)

    assert torch.allclose(actual, probabilities, atol=1e-12), f"{actual} != {probabilities}"


def check_default_backward_is_the_softmax_jacobian() -> None:
    """The default path's gradient must match the analytic softmax Jacobian-vector product."""
    state = _state(requires_grad=True)
    upstream = torch.tensor([[1.0, -2.0, 0.5, 0.25], [0.0, 1.0, 1.0, -1.0], [2.0, 0.0, -1.0, 0.5]], dtype=torch.float64)

    relaxed_sequence(state, temperature=TEMPERATURE).backward(upstream)

    probs = F.softmax(_state() / TEMPERATURE, dim=-1)
    weighted = (probs * upstream).sum(dim=-1, keepdim=True)
    expected = probs * (upstream - weighted) / TEMPERATURE
    assert torch.allclose(state.grad, expected, atol=1e-10), f"{state.grad} != {expected}"


def check_straight_through_routes_gradient_through_probabilities() -> None:
    """hard=1 forwards a one-hot but must backpropagate the softmax derivative, not zero."""
    state = _state(requires_grad=True)
    upstream = torch.ones(3, 4, dtype=torch.float64)

    relaxed_sequence(state, temperature=TEMPERATURE, hard=1.0).backward(upstream)

    assert state.grad is not None and torch.isfinite(state.grad).all(), "STE produced no finite gradient"
    probs = F.softmax(_state() / TEMPERATURE, dim=-1)
    weighted = (probs * upstream).sum(dim=-1, keepdim=True)
    expected = probs * (upstream - weighted) / TEMPERATURE
    assert torch.allclose(state.grad, expected, atol=1e-10), f"{state.grad} != {expected}"


def check_soft_zero_backward_is_the_identity() -> None:
    """soft=0, hard=0 feeds the raw matrix, so the derivative is the upstream gradient itself."""
    state = _state(requires_grad=True)
    upstream = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0], [0.5, 0.5, 0.5, 0.5]], dtype=torch.float64)

    relaxed_sequence(state, temperature=TEMPERATURE, soft=0.0, hard=0.0).backward(upstream)

    assert torch.allclose(state.grad, upstream, atol=1e-12), f"{state.grad} != {upstream}"


CHECKS: tuple[Callable[[], None], ...] = (
    check_default_is_a_plain_softmax,
    check_soft_zero_passes_the_raw_matrix_through,
    check_hard_one_forwards_a_one_hot,
    check_hard_one_supersedes_soft,
    check_intermediate_blends_match_the_formula,
    check_probability_mode_skips_the_softmax,
    check_default_backward_is_the_softmax_jacobian,
    check_straight_through_routes_gradient_through_probabilities,
    check_soft_zero_backward_is_the_identity,
)


def _run_check(check: Callable[[], None]) -> str | None:
    """Run one check, printing its status; return its name on failure."""
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
