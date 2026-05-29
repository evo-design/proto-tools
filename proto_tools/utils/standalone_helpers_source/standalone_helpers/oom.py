"""CUDA out-of-memory detection, cleanup, and actionable errors for standalone scripts.

GPU OOM is hardware/config-dependent (sequence length x batch x precision x VRAM), so
proto-tools does not predict it with fixed caps. Instead each tool reacts to a real OOM:
free cached memory and surface a clear, actionable error instead of a deep CUDA trace.

Torch/JAX are imported lazily so this module is importable in torch-free environments.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import NoReturn

from .proto_logging import get_logger

logger = get_logger(__name__)

# Substrings that mark a GPU OOM across frameworks: torch ("CUDA out of memory"),
# raw CUDA ("cuda error: out of memory"), and JAX/XLA ("RESOURCE_EXHAUSTED: ...").
_OOM_MARKERS = ("out of memory", "resource_exhausted")


class GpuOutOfMemoryError(RuntimeError):
    """A GPU ran out of memory; carries an actionable, tool-specific message."""


def is_cuda_oom(exc: BaseException | str) -> bool:
    """Whether ``exc`` (an exception or a stderr/text blob) signals a CUDA or JAX OOM."""
    text = str(exc).lower()
    return any(marker in text for marker in _OOM_MARKERS)


def release_cuda_memory() -> None:
    """Best-effort free of cached GPU memory (``gc.collect`` + ``torch.cuda.empty_cache``).

    Safe to call after an OOM and in torch-free (e.g. JAX-only) environments.
    """
    import gc

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Generic remedy used when a caller does not pass a tool-specific ``hint``.
_DEFAULT_HINT = "Reduce the input size, lower sampling/recycle settings, or use a GPU with more free memory."


def raise_oom(tool: str, exc: BaseException | None = None, *, hint: str = "") -> NoReturn:
    """Re-raise a CUDA OOM as an actionable :class:`GpuOutOfMemoryError`, chained to ``exc``.

    ``hint`` is a tool-specific remedy; the generic ``_DEFAULT_HINT`` is used when it is omitted.
    """
    raise GpuOutOfMemoryError(f"{tool}: ran out of GPU memory at this input size. {hint or _DEFAULT_HINT}") from exc


@contextmanager
def oom_guard(tool: str, *, hint: str = "") -> Iterator[None]:
    """Turn a CUDA/JAX OOM raised inside the block into an actionable :class:`GpuOutOfMemoryError`.

    Frees cached GPU memory on OOM; non-OOM exceptions (and already-actionable
    ``GpuOutOfMemoryError``) propagate unchanged. Wrap a tool's model call with this.
    """
    try:
        yield
    except GpuOutOfMemoryError:
        raise
    except Exception as exc:
        if is_cuda_oom(exc):
            logger.debug("%s: caught CUDA OOM inside oom_guard; freeing memory and re-raising", tool)
            release_cuda_memory()
            raise_oom(tool, exc, hint=hint)
        raise
