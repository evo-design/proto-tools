"""Random seed and JAX compilation cache helpers for standalone scripts.

The seeding helpers set reproducible RNG state for PyTorch/JAX/NumPy/stdlib
RNGs and, as a companion, expose the JAX persistent compilation cache so
subsequent runs on the same shape skip the (sometimes minutes-long)
XLA/Triton autotuning step.
"""

import os
import random
from collections.abc import Callable
from typing import Any, cast

from .proto_logging import get_logger

logger = get_logger(__name__)

RANDOM_SEED_UPPER_BOUND = 2**31


def get_random_int() -> int:
    """Return a fresh random int in ``[0, 2**31)`` for seeding RNGs.

    Use as a fallback when downstream code requires a concrete int seed:
    ``seed if seed is not None else get_random_int()``.
    """
    return random.randint(0, RANDOM_SEED_UPPER_BOUND - 1)  # noqa: S311 -- not for cryptographic use


def set_torch_seed(seed: int | None) -> None:
    """Seed PyTorch and related RNG sources for reproducibility. No-op when seed is None.

    Sets:
    - ``random.seed`` (Python stdlib)
    - ``numpy.random.seed`` (if numpy is installed)
    - ``torch.manual_seed`` (CPU + all CUDA devices)
    - ``torch.backends.cudnn.deterministic = True``
    - ``torch.backends.cudnn.benchmark = False``
    """
    if seed is not None:
        # Seed Python stdlib RNG
        random.seed(seed)

        # Seed NumPy RNG (optional, not all tools use NumPy)
        try:
            import numpy

            numpy.random.seed(seed)
        except ImportError:
            pass

        import torch

        # Seed PyTorch RNG on CPU and all CUDA devices
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Use deterministic cuDNN algorithms instead of faster non-deterministic ones
        torch.backends.cudnn.deterministic = True
        # Disable cuDNN auto-tuner that selects different kernels between runs
        torch.backends.cudnn.benchmark = False


def set_jax_seed(seed: int | None) -> Any:
    """Seed Python/NumPy RNGs and return a JAX ``PRNGKey``. No-op returning None when seed is None.

    The JAX equivalent of :func:`set_torch_seed`, but with an important
    difference: JAX RNG is **functional** (stateless). There is no global
    JAX RNG state that can be seeded — every JAX random op takes a
    ``PRNGKey`` as an explicit argument. This helper therefore:

    1. Seeds Python stdlib ``random`` and NumPy (for any pre-JAX ops that
       use them, e.g. tokenization, shuffling).
    2. Returns a fresh ``jax.random.PRNGKey(seed)`` that the caller must
       thread explicitly into JAX random ops.

    Implications for persistent-worker reproducibility:

    - For JAX tools whose forward pass already consumes a PRNGKey (e.g.
      ProteinMPNN's ``model.sample_parallel(key=...)``), passing the
      returned key is all you need — consecutive dispatches with the same
      seed produce identical output.
    - For JAX models with a purely deterministic forward pass (e.g. a
      plain MLP), the seed only affects output if the params themselves
      depend on it. In that case the caller must re-initialize the params
      with the new seed on every dispatch, otherwise the first dispatch's
      seed is silently locked in for the lifetime of the worker.

    The concrete runtime type is ``jax.Array | None``, but the signature
    uses ``Any`` because ``jax`` cannot be imported at module load time
    (it lives only in tool-specific standalone envs, not in the proto-tools
    core env where this helper is parsed).

    Args:
        seed (int | None): Seed value. ``None`` is a no-op and returns ``None``.

    Returns:
        Any: A ``jax.random.PRNGKey(seed)`` (i.e. ``jax.Array``) when
            ``seed`` is given, or ``None`` when ``seed`` is ``None``.
    """
    if seed is None:
        return None

    random.seed(seed)

    try:
        import numpy

        numpy.random.seed(seed)
    except ImportError:
        pass

    import jax

    return jax.random.PRNGKey(seed)


def enable_jax_compilation_cache(toolkit: str) -> str | None:
    """Enable JAX disk compilation cache for faster cold starts.

    Persists compiled XLA kernels (and, transitively via
    ``jax_persistent_cache_enable_xla_caches``, the Triton autotuner's
    per-fusion cache) across process restarts. For large JAX models where
    the first compile spends minutes on XLA/Triton autotuning, cached
    shapes on subsequent runs drop from minutes to seconds — a dramatic
    speedup for tests, CI, and repeat usage.

    Must be called BEFORE any JAX computation happens (ideally right after
    ``import jax``). Safe to call multiple times (idempotent per-process).
    No-op if ``jax`` is not importable from the current environment.

    The cache lives inside the tool's micromamba environment at
    ``{CONDA_PREFIX}/jax_cache/``, so it is automatically cleaned up when
    the environment is rebuilt or deleted. An explicit
    ``JAX_COMPILATION_CACHE_DIR`` env var overrides this default.

    Args:
        toolkit (str): Tool name for logging (e.g., ``"alphagenome"``).

    Returns:
        str | None: Resolved cache directory path, or ``None`` if ``jax``
            is not importable or the tool env path cannot be determined.
    """
    from pathlib import Path

    try:
        import jax
    except ImportError as e:
        logger.warning("enable_jax_compilation_cache(%s): jax not importable in this env: %s", toolkit, e)
        return None

    override = os.environ.get("JAX_COMPILATION_CACHE_DIR", "").strip()
    if override:
        cache_dir = Path(override)
    else:
        venv_path = os.environ.get("CONDA_PREFIX", "").strip()
        if not venv_path:
            logger.warning(
                "Cannot enable JAX compilation cache for %s: CONDA_PREFIX not set (tool env path unknown).",
                toolkit,
            )
            return None
        cache_dir = Path(venv_path) / "jax_cache"

    cache_dir.mkdir(parents=True, exist_ok=True)
    config_update = cast(Callable[[str, str], None], jax.config.update)
    config_update("jax_compilation_cache_dir", str(cache_dir))
    logger.info("JAX compilation cache for %s enabled at %s", toolkit, cache_dir)
    return str(cache_dir)
