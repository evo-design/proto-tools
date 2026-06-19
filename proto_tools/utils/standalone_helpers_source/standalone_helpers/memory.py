"""GPU memory statistics helpers for standalone scripts.

Provides a unified shape for memory reporting across PyTorch and JAX so
DeviceManager can query ``get_memory_stats()`` on any tool without
framework-specific glue.
"""

from typing import Any

from .proto_logging import get_logger

logger = get_logger(__name__)


def get_pytorch_memory_stats(device: int | str = 0) -> dict[str, Any]:
    """Helper for PyTorch tools to report GPU memory stats.

    Args:
        device (int | str): CUDA device index (int) or device string (e.g. "cuda:0")

    Returns:
        dict[str, Any]: Dict with memory statistics or {"available": False} if not available.
            Keys when available:
            - available (bool): Whether memory stats are available
            - framework (str): "pytorch"
            - allocated_bytes (int): Currently allocated GPU memory in bytes
            - reserved_bytes (int): Reserved GPU memory in bytes (PyTorch cache)
            - max_allocated_bytes (int): Peak allocated memory since program start

    Example:
        >>> stats = get_pytorch_memory_stats(0)
        >>> if stats["available"]:
        ...     print(f"GPU using {stats['allocated_bytes'] / 1e9:.2f} GB")
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False, "framework": "pytorch", "reason": "CUDA not available"}

        return {
            "available": True,
            "framework": "pytorch",
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
        }
    except Exception as e:
        return {"available": False, "framework": "pytorch", "reason": str(e)}


def get_jax_memory_stats(device_index: int = 0) -> dict[str, Any]:
    """Helper for JAX tools to report GPU memory stats.

    Args:
        device_index (int): JAX device index (default 0)

    Returns:
        dict[str, Any]: Dict with memory statistics or {"available": False} if not available.
            Keys when available (standardized across frameworks):
            - available (bool): Whether memory stats are available
            - framework (str): "jax"
            - allocated_bytes (int): Currently allocated memory (standardized key)
            - max_allocated_bytes (int): Peak memory usage (standardized key)
            - device_kind (str): Device type (e.g., "gpu", "cpu")

            Legacy JAX-specific keys (for backwards compatibility):
            - bytes_in_use (int): Same as allocated_bytes
            - peak_bytes_in_use (int): Same as max_allocated_bytes

    Example:
        >>> stats = get_jax_memory_stats(0)
        >>> if stats["available"]:
        ...     print(f"GPU using {stats['allocated_bytes'] / 1e9:.2f} GB")
    """
    try:
        import jax

        devices = jax.devices()
        if device_index >= len(devices):
            return {
                "available": False,
                "framework": "jax",
                "reason": f"Device index {device_index} exceeds available devices ({len(devices)})",
            }

        device = devices[device_index]
        stats = device.memory_stats()

        bytes_in_use = stats.get("bytes_in_use", 0)
        peak_bytes = stats.get("peak_bytes_in_use", 0)

        return {
            "available": True,
            "framework": "jax",
            # Standardized keys (same as PyTorch)
            "allocated_bytes": bytes_in_use,
            "max_allocated_bytes": peak_bytes,
            # JAX-specific keys (backwards compatibility)
            "bytes_in_use": bytes_in_use,
            "peak_bytes_in_use": peak_bytes,
            "device_kind": device.device_kind,
        }
    except Exception as e:
        return {"available": False, "framework": "jax", "reason": str(e)}
