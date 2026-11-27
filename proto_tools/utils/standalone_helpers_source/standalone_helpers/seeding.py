"""Random seed helpers for standalone scripts.

Set reproducible RNG state for PyTorch/JAX/NumPy/stdlib RNGs. The JAX
persistent compilation cache is not configured here: the parent sets
``JAX_COMPILATION_CACHE_DIR`` on every tool subprocess (see
``persistent_worker._jax_compilation_cache_dir``).
"""

import random
from typing import Any

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
