"""tests/tool_infra_tests/_relaxation_jax_env_checks.py.

Numerical checks for ``standalone_helpers.relaxation.relaxed_sequence_jax``, run inside a built
tool environment by ``test_relaxation_formula_runs_in_jax_env`` because the base process has no
JAX.

The JAX definition serves backends whose environment has no torch, so it is checked both against
the formula and against the shared ``REFERENCE``
table, which is what keeps it from drifting away from the torch definition. Plain asserts, one line of
status per check, non-zero exit on any failure (which ``run_in_env`` surfaces as ``RuntimeError``).
"""

from __future__ import annotations

import traceback
from collections.abc import Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from _relaxation_reference import LOGITS, REFERENCE, TEMPERATURE, UPSTREAM  # noqa: E402
from standalone_helpers.relaxation import relaxed_sequence_jax  # noqa: E402

BLEND_GRID = tuple((soft, hard) for soft in (0.0, 0.25, 0.5, 1.0) for hard in (0.0, 0.5, 1.0))


def _relax(state, *, temperature, soft=1.0, hard=0.0):
    """Forward value and the vjp of the upstream cotangent, torch's ``.backward(upstream)``."""
    value, vjp = jax.vjp(
        lambda x: relaxed_sequence_jax(x, temperature=temperature, soft=soft, hard=hard),
        state,
    )
    return value, vjp(jnp.asarray(UPSTREAM))[0]


def _softmax_route(state, *, temperature):
    """Analytic softmax Jacobian-vector product for the shared upstream cotangent."""
    probs = jax.nn.softmax(state / temperature, axis=-1)
    upstream = jnp.asarray(UPSTREAM)
    return probs * (upstream - (probs * upstream).sum(axis=-1, keepdims=True)) / temperature


def check_matches_the_shared_reference_table() -> None:
    """Forward values and gradients must equal the table the torch definition is pinned to."""
    state = jnp.asarray(LOGITS)

    for (soft, hard), expected in REFERENCE.items():
        value, gradient = _relax(state, temperature=TEMPERATURE, soft=soft, hard=hard)

        assert jnp.allclose(value, jnp.asarray(expected["forward"]), atol=1e-12), (
            f"soft={soft} hard={hard} forward: {value} != {expected['forward']}"
        )
        assert jnp.allclose(gradient, jnp.asarray(expected["gradient"]), atol=1e-12), (
            f"soft={soft} hard={hard} gradient: {gradient} != {expected['gradient']}"
        )


def check_blend_grid_matches_the_formula() -> None:
    """Every (soft, hard) pair must equal the documented two-step blend, forward and backward."""
    state = jnp.asarray(LOGITS)
    probs = jax.nn.softmax(state / TEMPERATURE, axis=-1)
    discrete = jax.nn.one_hot(jnp.argmax(probs, axis=-1), 4, dtype=probs.dtype)
    upstream = jnp.asarray(UPSTREAM)
    softmax_route = _softmax_route(state, temperature=TEMPERATURE)

    for soft, hard in BLEND_GRID:
        value, gradient = _relax(state, temperature=TEMPERATURE, soft=soft, hard=hard)

        blended = (1.0 - soft) * state + soft * probs
        expected_value = (1.0 - hard) * blended + hard * discrete
        expected_gradient = (1.0 - hard) * (1.0 - soft) * upstream + ((1.0 - hard) * soft + hard) * softmax_route
        assert jnp.allclose(value, expected_value, atol=1e-12), f"soft={soft} hard={hard}: {value} != {expected_value}"
        assert jnp.allclose(gradient, expected_gradient, atol=1e-10), (
            f"soft={soft} hard={hard}: {gradient} != {expected_gradient}"
        )


def check_hard_one_supersedes_soft() -> None:
    """At hard=1 neither the forward value nor the gradient may depend on soft."""
    state = jnp.asarray(LOGITS)
    results = [_relax(state, temperature=TEMPERATURE, soft=soft, hard=1.0) for soft in (0.0, 0.5, 1.0)]

    for value, gradient in results[1:]:
        assert jnp.allclose(value, results[0][0], atol=1e-12), "hard=1 forward depends on soft"
        assert jnp.allclose(gradient, results[0][1], atol=1e-12), "hard=1 gradient depends on soft"


def check_no_control_setting_silently_zeroes_the_gradient() -> None:
    """No (soft, hard) pair may return a representation the input cannot be optimized through.

    A relaxation that forwards a constant — an argmax one-hot with no straight-through route —
    backpropagates all zeros, so an optimizer runs without moving and nothing raises.
    """
    state = jnp.asarray(LOGITS)

    for soft, hard in BLEND_GRID:
        _, gradient = _relax(state, temperature=TEMPERATURE, soft=soft, hard=hard)

        assert jnp.count_nonzero(gradient) > 0, f"soft={soft} hard={hard} produced an all-zero gradient"


def check_probability_mode_passes_the_distribution_through() -> None:
    """With temperature=None the input is already P, so the blend reduces to the input itself.

    The forward value is the supplied distribution wherever ``hard < 1``, and the gradient is the
    upstream cotangent at every setting: probability-space gradients survive the controls.
    """
    probabilities = jnp.asarray([[0.7, 0.2, 0.05, 0.05], [0.1, 0.1, 0.1, 0.7], [0.25, 0.25, 0.25, 0.25]])
    upstream = jnp.asarray(UPSTREAM)

    for soft, hard in BLEND_GRID:
        value, gradient = _relax(probabilities, temperature=None, soft=soft, hard=hard)

        if hard < 1.0:
            expected = (1.0 - hard) * probabilities + hard * jax.nn.one_hot(
                jnp.argmax(probabilities, axis=-1), 4, dtype=probabilities.dtype
            )
            assert jnp.allclose(value, expected, atol=1e-12), f"soft={soft} hard={hard}: {value} != {expected}"
        assert jnp.allclose(gradient, upstream, atol=1e-12), f"soft={soft} hard={hard}: {gradient} != {upstream}"


CHECKS: tuple[Callable[[], None], ...] = (
    check_matches_the_shared_reference_table,
    check_blend_grid_matches_the_formula,
    check_hard_one_supersedes_soft,
    check_no_control_setting_silently_zeroes_the_gradient,
    check_probability_mode_passes_the_distribution_through,
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
