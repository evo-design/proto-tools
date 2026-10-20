"""tests/tool_infra_tests/_alphafold2_relaxation_env_checks.py.

Checks that ColabDesign's own ``soft_seq`` implements the shared relaxation contract.

AlphaFold2 relaxes its sequence inside ColabDesign rather than through
``standalone_helpers.relaxation``, so this suite pins ColabDesign's blend to the same formula
the shared helpers implement, on both backends the tool can load: upstream ColabDesign and the
Germinal fork (alpha scaling plus a persistent bias). It needs no model weights and no GPU —
``soft_seq`` is pure arithmetic over a sequence state. Plain asserts, one line of status per
check, non-zero exit on any failure (which ``run_in_env`` surfaces as ``RuntimeError``).
"""

from __future__ import annotations

import os
import sys
import traceback
from collections.abc import Callable
from functools import partial
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from _relaxation_reference import LOGITS, TEMPERATURE, UPSTREAM  # noqa: E402
from standalone_helpers.relaxation import relaxed_sequence_jax  # noqa: E402

BLEND_GRID = tuple((soft, hard) for soft in (0.0, 0.25, 0.5, 1.0) for hard in (0.0, 0.5, 1.0))
BIAS = [[0.3, -0.2, 0.1, 0.05], [-0.1, 0.4, 0.0, 0.2], [0.2, 0.2, -0.3, 0.1]]


def _germinal_dir() -> str:
    """Where the tool's setup.sh installs the Germinal ColabDesign fork."""
    return os.path.join(
        os.environ.get("TOOL_VENV_PATH", os.environ.get("VIRTUAL_ENV", "")), "data", "colabdesign_germinal"
    )


def _load_soft_seq(backend: str) -> Any:
    """Import ``soft_seq`` from one backend, swapping sys.path the way the tool does."""
    germinal_dir = _germinal_dir()
    if backend == "germinal":
        if not os.path.isdir(germinal_dir):
            raise RuntimeError(f"Germinal ColabDesign not found at {germinal_dir}; run setup.sh")
        sys.path.insert(0, germinal_dir)
    elif germinal_dir in sys.path:
        sys.path.remove(germinal_dir)
    for name in [name for name in sys.modules if name.startswith("colabdesign")]:
        del sys.modules[name]

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


def _pseudo_and_gradient(soft_seq: Any, state: Any, *, alpha: float, bias: Any, soft: float, hard: float) -> Any:
    """``soft_seq``'s pseudo sequence and the vjp of the shared upstream cotangent."""
    opt = {"alpha": alpha, "temp": TEMPERATURE, "soft": soft, "hard": hard}
    return _value_and_gradient(lambda x: soft_seq(x, bias, opt)["pseudo"], state)


def _check_backend(backend: str) -> None:
    """Assert one backend's blend matches the contract over the grid, forward and backward."""
    soft_seq = _load_soft_seq(backend)
    state = jnp.asarray(LOGITS)
    bias = jnp.asarray(BIAS)

    for alpha, bias_term in ((1.0, None), (1.0, bias), (2.0, bias)):
        for soft, hard in BLEND_GRID:
            value, gradient = _pseudo_and_gradient(soft_seq, state, alpha=alpha, bias=bias_term, soft=soft, hard=hard)

            contract = partial(_contract, alpha=alpha, bias=bias_term, temperature=TEMPERATURE, soft=soft, hard=hard)
            expected_value, expected_gradient = _value_and_gradient(contract, state)
            label = f"{backend} alpha={alpha} bias={bias_term is not None} soft={soft} hard={hard}"
            assert jnp.allclose(value, expected_value, atol=1e-12), f"{label} forward: {value} != {expected_value}"
            assert jnp.allclose(gradient, expected_gradient, atol=1e-10), (
                f"{label} gradient: {gradient} != {expected_gradient}"
            )


def check_base_backend_matches_the_contract() -> None:
    """Upstream ColabDesign's pseudo sequence must equal the contract at every blend."""
    _check_backend("base")


def check_germinal_backend_matches_the_contract() -> None:
    """The Germinal fork must equal the contract too, alpha scaling and bias included."""
    _check_backend("germinal")


def check_matches_the_shared_helper_without_preprocessing() -> None:
    """With alpha=1 and no bias, ColabDesign must agree with relaxed_sequence_jax exactly.

    This is the tie between AlphaFold2 and the helper every other backend calls: the two
    implementations are independent, so agreement here is what makes the contract one contract.
    """
    soft_seq = _load_soft_seq("base")
    state = jnp.asarray(LOGITS)

    for soft, hard in BLEND_GRID:
        value, gradient = _pseudo_and_gradient(soft_seq, state, alpha=1.0, bias=None, soft=soft, hard=hard)

        shared = partial(relaxed_sequence_jax, temperature=TEMPERATURE, soft=soft, hard=hard)
        expected_value, expected_gradient = _value_and_gradient(shared, state)
        label = f"soft={soft} hard={hard}"
        assert jnp.allclose(value, expected_value, atol=1e-12), f"{label} forward: {value} != {expected_value}"
        assert jnp.allclose(gradient, expected_gradient, atol=1e-10), (
            f"{label} gradient: {gradient} != {expected_gradient}"
        )


def check_probability_mode_passes_the_distribution_through() -> None:
    """The soft=0 setting the tool uses for temperature=None must forward the input verbatim.

    ``soft_seq`` always softmaxes, so the tool expresses probability mode as ``soft=0``, which
    selects the raw input branch. The gradient must then be probability-space: the upstream
    cotangent, unchanged.
    """
    soft_seq = _load_soft_seq("base")
    probabilities = jnp.asarray([[0.7, 0.2, 0.05, 0.05], [0.1, 0.1, 0.1, 0.7], [0.25, 0.25, 0.25, 0.25]])
    upstream = jnp.asarray(UPSTREAM)

    value, gradient = _pseudo_and_gradient(soft_seq, probabilities, alpha=1.0, bias=None, soft=0.0, hard=0.0)

    assert jnp.allclose(value, probabilities, atol=1e-12), f"{value} != {probabilities}"
    assert jnp.allclose(gradient, upstream, atol=1e-12), f"{gradient} != {upstream}"


CHECKS: tuple[Callable[[], None], ...] = (
    check_base_backend_matches_the_contract,
    check_germinal_backend_matches_the_contract,
    check_matches_the_shared_helper_without_preprocessing,
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
