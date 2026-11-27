"""tests/tool_infra_tests/_alphafold2_relaxation_env_checks.py.

Checks that ColabDesign's own ``soft_seq`` implements the shared relaxation contract.

AlphaFold2 relaxes its sequence inside ColabDesign rather than through
``standalone_helpers.relaxation``, so this suite pins ColabDesign's blend to the same formula
the shared helpers implement. It needs no model weights and no GPU —
``soft_seq`` is pure arithmetic over a sequence state. Plain asserts, one line of status per
check, non-zero exit on any failure (which ``run_in_env`` surfaces as ``RuntimeError``).
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from functools import partial
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from _relaxation_reference import LOGITS, TEMPERATURE, UPSTREAM  # noqa: E402
from inference import _relaxation_opt  # noqa: E402  # the tool's own opt builder
from standalone_helpers.relaxation import relaxed_sequence_jax  # noqa: E402

BLEND_GRID = tuple((soft, hard) for soft in (0.0, 0.25, 0.5, 1.0) for hard in (0.0, 0.5, 1.0))
BIAS = [[0.3, -0.2, 0.1, 0.05], [-0.1, 0.4, 0.0, 0.2], [0.2, 0.2, -0.3, 0.1]]


def _load_soft_seq() -> Any:
    """Import ColabDesign's ``soft_seq``, the blend AlphaFold2 actually runs."""
    from colabdesign.shared.model import soft_seq

    return soft_seq


def _contract(state: Any, *, alpha: float, bias: Any, temperature: float, soft: float, hard: float) -> Any:
    """The shared contract written out, with ColabDesign's preprocessing feeding P.

    ``preprocess_logits`` is ``alpha * X + bias``; the raw-input branch stays ``X`` itself, as
    the contract specifies, so alpha and bias move only the softmax branch.
    """
    preprocessed = alpha * state if bias is None else alpha * state + bias
    probs = jax.nn.softmax(preprocessed / temperature, axis=-1)
    discrete = jax.nn.one_hot(jnp.argmax(probs, axis=-1), probs.shape[-1], dtype=probs.dtype)
    straight_through = jax.lax.stop_gradient(discrete - probs) + probs
    blended = (1.0 - soft) * state + soft * probs
    return (1.0 - hard) * blended + hard * straight_through


def _value_and_gradient(fn: Callable[[Any], Any], state: Any) -> Any:
    """Forward value and the vjp of the shared upstream cotangent, torch's ``.backward(upstream)``."""
    value, vjp = jax.vjp(fn, state)
    return value, vjp(jnp.asarray(UPSTREAM))[0]


def _pseudo_with_opt(soft_seq: Any, state: Any, *, opt: dict[str, float], bias: Any = None) -> Any:
    """``soft_seq``'s pseudo sequence and gradient under an ``opt`` the tool itself built."""
    return _value_and_gradient(partial(_apply_soft_seq, soft_seq=soft_seq, bias=bias, opt=opt), state)


def _apply_soft_seq(state: Any, *, soft_seq: Any, bias: Any, opt: dict[str, float]) -> Any:
    """Call ``soft_seq`` with its arguments bound, so no closure captures a loop variable."""
    return soft_seq(state, bias, opt)["pseudo"]


def _pseudo_and_gradient(soft_seq: Any, state: Any, *, alpha: float, bias: Any, soft: float, hard: float) -> Any:
    """``soft_seq``'s pseudo sequence and the vjp of the shared upstream cotangent."""
    opt = {"alpha": alpha, "temp": TEMPERATURE, "soft": soft, "hard": hard}
    return _value_and_gradient(lambda x: soft_seq(x, bias, opt)["pseudo"], state)


def _check_blend() -> None:
    """Assert ColabDesign's blend matches the contract over the grid, forward and backward."""
    soft_seq = _load_soft_seq()
    state = jnp.asarray(LOGITS)
    bias = jnp.asarray(BIAS)

    for alpha, bias_term in ((1.0, None), (1.0, bias), (2.0, bias)):
        for soft, hard in BLEND_GRID:
            value, gradient = _pseudo_and_gradient(soft_seq, state, alpha=alpha, bias=bias_term, soft=soft, hard=hard)

            contract = partial(_contract, alpha=alpha, bias=bias_term, temperature=TEMPERATURE, soft=soft, hard=hard)
            expected_value, expected_gradient = _value_and_gradient(contract, state)
            label = f"alpha={alpha} bias={bias_term is not None} soft={soft} hard={hard}"
            assert jnp.allclose(value, expected_value, atol=1e-12), f"{label} forward: {value} != {expected_value}"
            assert jnp.allclose(gradient, expected_gradient, atol=1e-10), (
                f"{label} gradient: {gradient} != {expected_gradient}"
            )


def check_colabdesign_matches_the_contract() -> None:
    """ColabDesign's pseudo sequence must equal the contract at every blend."""
    _check_blend()


def check_the_tools_own_opt_matches_the_shared_helper() -> None:
    """The opt the tool builds must drive soft_seq to the same values as relaxed_sequence_jax.

    This is the tie between AlphaFold2 and the helper every other backend calls, and it runs
    through ``_relaxation_opt`` rather than a hand-built dict: ColabDesign defaults ``alpha`` to
    2.0, which would halve the effective temperature and silently break ``from_probabilities``.
    Checking the formula alone would not notice.
    """
    soft_seq = _load_soft_seq()
    state = jnp.asarray(LOGITS)

    for soft, hard in BLEND_GRID:
        opt = _relaxation_opt(temperature=TEMPERATURE, soft=soft, hard=hard)
        assert opt["alpha"] == 1.0, f"the tool left alpha at {opt['alpha']}"
        value, gradient = _pseudo_with_opt(soft_seq, state, opt=opt)

        shared = partial(relaxed_sequence_jax, temperature=TEMPERATURE, soft=soft, hard=hard)
        expected_value, expected_gradient = _value_and_gradient(shared, state)
        label = f"soft={soft} hard={hard}"
        assert jnp.allclose(value, expected_value, atol=1e-12), f"{label} forward: {value} != {expected_value}"
        assert jnp.allclose(gradient, expected_gradient, atol=1e-10), (
            f"{label} gradient: {gradient} != {expected_gradient}"
        )


def check_probability_mode_opt_selects_the_raw_input_branch() -> None:
    """The opt the tool builds for temperature=None must forward the distribution unchanged."""
    soft_seq = _load_soft_seq()
    probabilities = jnp.asarray([[0.7, 0.2, 0.05, 0.05], [0.1, 0.1, 0.1, 0.7], [0.25, 0.25, 0.25, 0.25]])

    opt = _relaxation_opt(temperature=None, soft=1.0, hard=1.0)
    value, gradient = _pseudo_with_opt(soft_seq, probabilities, opt=opt)

    assert opt["soft"] == 0.0 and opt["hard"] == 0.0, f"probability mode left blending on: {opt}"
    assert jnp.allclose(value, probabilities, atol=1e-12), f"{value} != {probabilities}"
    assert jnp.allclose(gradient, jnp.asarray(UPSTREAM), atol=1e-12), f"{gradient} != {UPSTREAM}"


def check_from_probabilities_round_trips_through_soft_seq() -> None:
    """Logits built by from_probabilities must come back out of soft_seq as the same distribution.

    The factory sets ``logits = temperature * log(p)``, so ``softmax(logits / temperature)`` is
    ``p`` again — but only when ColabDesign scales the logits by 1. At its default ``alpha=2``
    the model would see ``softmax(2 log p)``, which is ``p**2`` renormalized: a silent distortion
    that grows with how peaked ``p`` is.
    """
    soft_seq = _load_soft_seq()
    probabilities = jnp.asarray([[0.7, 0.2, 0.05, 0.05], [0.1, 0.1, 0.1, 0.7], [0.25, 0.25, 0.25, 0.25]])
    logits = TEMPERATURE * jnp.log(probabilities)

    opt = _relaxation_opt(temperature=TEMPERATURE, soft=1.0, hard=0.0)
    value, _ = _pseudo_with_opt(soft_seq, logits, opt=opt)

    assert jnp.allclose(value, probabilities, atol=1e-6), f"{value} != {probabilities}"
    squared = probabilities**2 / (probabilities**2).sum(axis=-1, keepdims=True)
    assert not jnp.allclose(value, squared, atol=1e-6), "soft_seq squared the distribution; alpha is not 1"


def check_a_bias_shifts_the_reconstructed_distribution_predictably() -> None:
    """With a bias active, from_probabilities reconstructs p weighted by exp(bias / temperature).

    A bias is meant to move the distribution — that is what ``excluded_amino_acids`` and
    ``bias_redesign`` are for — so reconstruction is exact only without one. Pinning the exact
    relationship keeps that documented rather than assumed.
    """
    soft_seq = _load_soft_seq()
    probabilities = jnp.asarray([[0.7, 0.2, 0.05, 0.05], [0.1, 0.1, 0.1, 0.7], [0.25, 0.25, 0.25, 0.25]])
    logits = TEMPERATURE * jnp.log(probabilities)
    bias = jnp.asarray(BIAS)

    opt = _relaxation_opt(temperature=TEMPERATURE, soft=1.0, hard=0.0)
    value, _ = _pseudo_with_opt(soft_seq, logits, opt=opt, bias=bias)

    weighted = probabilities * jnp.exp(bias / TEMPERATURE)
    expected = weighted / weighted.sum(axis=-1, keepdims=True)
    assert jnp.allclose(value, expected, atol=1e-6), f"{value} != {expected}"


def check_probability_mode_passes_the_distribution_through() -> None:
    """The soft=0 setting the tool uses for temperature=None must forward the input verbatim.

    ``soft_seq`` always softmaxes, so the tool expresses probability mode as ``soft=0``, which
    selects the raw input branch. The gradient must then be probability-space: the upstream
    cotangent, unchanged.
    """
    soft_seq = _load_soft_seq()
    probabilities = jnp.asarray([[0.7, 0.2, 0.05, 0.05], [0.1, 0.1, 0.1, 0.7], [0.25, 0.25, 0.25, 0.25]])
    upstream = jnp.asarray(UPSTREAM)

    value, gradient = _pseudo_and_gradient(soft_seq, probabilities, alpha=1.0, bias=None, soft=0.0, hard=0.0)

    assert jnp.allclose(value, probabilities, atol=1e-12), f"{value} != {probabilities}"
    assert jnp.allclose(gradient, upstream, atol=1e-12), f"{gradient} != {upstream}"


CHECKS: tuple[Callable[[], None], ...] = (
    check_colabdesign_matches_the_contract,
    check_the_tools_own_opt_matches_the_shared_helper,
    check_probability_mode_opt_selects_the_raw_input_branch,
    check_from_probabilities_round_trips_through_soft_seq,
    check_a_bias_shifts_the_reconstructed_distribution_predictably,
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
