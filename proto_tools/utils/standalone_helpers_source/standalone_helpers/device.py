"""Device environment, resolution, and movement helpers for standalone scripts.

Three related concerns live together here:

1. **Device Environment** — translate logical device strings into the
   ``CUDA_VISIBLE_DEVICES`` and JAX environment variables needed by an
   ephemeral CLI subprocess spawned from a persistent worker.
2. **Device Resolution** — map a device string (``"cuda:0"``) to the
   framework-specific device object (``jax.Device``).
3. **Device Movement** — move a model or params pytree between devices
   with proper GPU memory cleanup, for both PyTorch and JAX.
"""

import os
from collections.abc import Callable
from typing import Any, cast

from .proto_logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Device Environment (subprocess CUDA_VISIBLE_DEVICES & JAX env vars)
# ============================================================================


def _parse_cuda_indices(device: str) -> list[int] | None:
    """Parse a CUDA device string into a list of integer device indices.

    Supports formats:
        "cuda:0"         → [0]
        "cuda:2"         → [2]
        "cuda:0,1"       → [0, 1]
        "cuda:2,3"       → [2, 3]
        "cuda:0,cuda:1"  → [0, 1]

    Args:
        device (str): Device string starting with "cuda:".

    Returns:
        list[int] | None: List of integer indices, or None if the device string is malformed
            (missing "cuda:" prefix or non-integer index components).
    """
    if not device.startswith("cuda:"):
        return None

    indices_str = device[5:]  # Remove "cuda:" prefix
    indices = []

    for raw_part in indices_str.split(","):
        part = raw_part.strip()
        # Remove nested "cuda:" prefix (handles "cuda:0,cuda:1" format)
        part = part.removeprefix("cuda:")
        try:
            indices.append(int(part))
        except ValueError:
            return None

    return indices


def _apply_jax_subprocess_env(env: dict[str, str]) -> dict[str, str]:
    """Apply JAX environment settings based on subprocess GPU visibility.

    CLI subprocesses are ephemeral (run one job and exit), so JAX preallocation
    is fine and even beneficial for performance. This undoes the preallocation
    restrictions set by persistent_worker.py for the long-lived worker process.

    Args:
        env (dict[str, str]): Environment variables dict to modify in-place.

    - If CUDA_VISIBLE_DEVICES grants GPU access: remove preallocation restrictions
      so JAX can fully utilize the allocated GPU(s).
    - If CUDA_VISIBLE_DEVICES is empty (CPU-only): set JAX_PLATFORMS=cpu to
      prevent JAX from probing for GPUs.
    """
    cvd = env.get("CUDA_VISIBLE_DEVICES", "")
    if cvd and cvd.strip():
        # Subprocess has GPU access; allow JAX preallocation (ephemeral process)
        env.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
        env.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
        env.pop("JAX_PLATFORMS", None)
    else:
        # No GPUs; force CPU-only so JAX doesn't probe for devices
        env["JAX_PLATFORMS"] = "cpu"
    return env


def get_subprocess_device_env(device: str) -> dict[str, str]:
    """Get environment for CLI subprocess with correct device visibility.

    When the tool process has CUDA_VISIBLE_DEVICES set, this function maps
    logical device indices to physical GPU indices for CLI subprocesses.
    Also configures JAX environment variables: CLI subprocesses are ephemeral,
    so JAX preallocation is re-enabled for GPU subprocesses (undoing the
    restriction set by persistent_worker.py for the long-lived worker).

    Args:
        device (str): Logical device string (e.g., "cuda:0", "cuda:2", "cuda:0,1", "cuda:2,3")

    Returns:
        dict[str, str]: Environment dict with CUDA_VISIBLE_DEVICES set to physical GPU indices
            and JAX environment variables configured appropriately.

    Examples:
        Parent environment: CUDA_VISIBLE_DEVICES=0,1,5,7

        get_subprocess_device_env("cuda:0")
        # Returns: {..., "CUDA_VISIBLE_DEVICES": "0"}

        get_subprocess_device_env("cuda:2")
        # Returns: {..., "CUDA_VISIBLE_DEVICES": "5"}

        get_subprocess_device_env("cuda:2,3")
        # Returns: {..., "CUDA_VISIBLE_DEVICES": "5,7"}

    How it works:
        1. Worker subprocess inherits parent's CUDA_VISIBLE_DEVICES (e.g., "0,1,5,7")
        2. Tool requests logical device (e.g., "cuda:2")
        3. This function maps logical index 2 → physical GPU 5
        4. CLI subprocess gets CUDA_VISIBLE_DEVICES=5
        5. CLI sees physical GPU 5 as its only device (cuda:0)
    """
    env = os.environ.copy()

    # Normalize bare "cuda" to "cuda:0" for single-GPU environments
    # where there's no DeviceManager
    if device == "cuda":
        device = "cuda:0"

    # Handle CPU device
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        return _apply_jax_subprocess_env(env)

    # Parse logical device indices from device string
    indices = _parse_cuda_indices(device)
    if indices is None:
        raise ValueError(f"get_subprocess_device_env: unrecognized device {device!r}; expected 'cpu' or 'cuda[:N]'")

    # Get parent's CUDA_VISIBLE_DEVICES to map logical → physical indices
    parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    if not parent_visible or not parent_visible.strip():
        # No parent CUDA_VISIBLE_DEVICES; use indices directly
        logger.warning(
            f"CUDA_VISIBLE_DEVICES not set in worker environment for device '{device}'. "
            f"CLI subprocess may access unintended GPUs."
        )
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(idx) for idx in indices)
        return _apply_jax_subprocess_env(env)

    # Parent has CUDA_VISIBLE_DEVICES; map logical to physical
    parent_devices = [d.strip() for d in parent_visible.split(",")]

    for idx in indices:
        if idx >= len(parent_devices):
            raise RuntimeError(
                f"get_subprocess_device_env: device index {idx} out of range "
                f"(parent CUDA_VISIBLE_DEVICES={parent_visible!r}, len={len(parent_devices)})"
            )

    physical_devices = [parent_devices[idx] for idx in indices]
    env["CUDA_VISIBLE_DEVICES"] = ",".join(physical_devices)

    logger.debug(f"Mapped device '{device}' to physical GPUs: {env['CUDA_VISIBLE_DEVICES']} (parent: {parent_visible})")

    return _apply_jax_subprocess_env(env)


# ============================================================================
# Device Resolution (device string → framework device object)
# ============================================================================


def resolve_jax_device(device: str) -> Any:
    """Resolve a device string like 'cuda:0' to a JAX device object.

    Handles the cuda→gpu backend conversion that JAX requires.

    Args:
        device (str): Device string (e.g., "cpu", "cuda", "cuda:0", "cuda:1")

    Returns:
        Any: jax.Device object for the specified device.

    Raises:
        ValueError: If the device index is out of range.
        ImportError: If JAX is not installed.
    """
    import jax

    if device == "cpu":
        return jax.devices("cpu")[0]

    device_idx: int
    if ":" in device:
        backend, device_idx_str = device.split(":", 1)
        device_idx = int(device_idx_str)
    else:
        backend = device
        device_idx = 0

    # JAX uses "gpu" not "cuda"
    if backend == "cuda":
        backend = "gpu"

    devices = jax.devices(backend)
    if device_idx >= len(devices):
        raise ValueError(f"Device {device} not available. Only {len(devices)} {backend} device(s) found.")
    return devices[device_idx]


# ============================================================================
# Device Movement (moving models/params between devices)
# ============================================================================


def move_model_to_device(
    model_or_params: Any,
    old_device: str,
    new_device: str,
    custom_move_fn: Callable[[Any, str, str], Any] | None = None,
) -> Any:
    """Move a model or params to a different device with proper GPU memory cleanup.

    This helper standardizes device movement across all tools and ensures GPU
    memory is properly freed when moving off CUDA devices. It handles both
    PyTorch models and JAX params pytrees, with support for custom movement
    logic for opaque third-party models.

    For PyTorch models (nn.Module or objects with .to()):
        - Calls model.to(new_device)
        - Calls torch.cuda.empty_cache() when moving off CUDA

    For JAX params (dict/list/tuple pytrees):
        - Calls jax.device_put(pytree, device) to move all arrays
        - Works natively with Flax/Haiku param dicts

    For JAX objects with jax.Array attributes:
        - Walks object attributes and device_put()s each jax.Array

    For non-ML models (CLI tools, etc.):
        - Returns as-is (no device movement needed)

    For opaque third-party models (e.g., AlphaGenome):
        - Pass custom_move_fn to override default behavior (e.g., full reload)

    Args:
        model_or_params (Any): PyTorch model, JAX params pytree (dict), or any object.
        old_device (str): Current device string (e.g., "cuda:0", "cpu")
        new_device (str): Target device string (e.g., "cuda:1", "cpu")
        custom_move_fn (Callable[[Any, str, str], Any] | None): Optional custom function for specialized device movement.
                       If provided, this function is called instead of default behavior.
                       Should have signature: (model_or_params, old_device, new_device) -> model_or_params.
                       Use for opaque models that can't be moved via .to() or device_put().

    Returns:
        Any: Model or params on the new device.

    Example:
        >>> self.model = move_model_to_device(self.model, self.device, device)
        >>> self.device = device

    Example:
        >>> self.params = move_model_to_device(self.params, self.device, device)
        >>> self.device = device

    Example:
        >>> def _reload(model, old, new):
        ...     return None  # Model will be recreated by self.load()
        >>> self.model = move_model_to_device(self.model, self.device, device, custom_move_fn=_reload)
        >>> self.load(device)

    Note:
        This helper is automatically copied to each tool's standalone/ directory.
        It must work in isolated environments with only tool-specific dependencies.
    """
    # If custom function provided, use it but still ensure GPU cleanup
    if custom_move_fn is not None:
        model_or_params = custom_move_fn(model_or_params, old_device, new_device)
        # Ensure GPU memory cleanup after custom function (safety net)
        # Even if custom function called empty_cache(), calling twice is harmless
        if old_device != "cpu" and old_device.startswith("cuda"):
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        return model_or_params

    # Try PyTorch first (most common case for GPU tools)
    try:
        import torch

        # Check if model has PyTorch's .to() method
        if hasattr(model_or_params, "to") and callable(model_or_params.to):
            model_or_params = model_or_params.to(new_device)

            # Free GPU memory when moving off CUDA device
            if old_device != "cpu" and old_device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

            return model_or_params
    except ImportError:
        # PyTorch not available - not a PyTorch tool
        pass
    except Exception:
        raise

    # Try JAX: move params/arrays via device_put
    try:
        import gc

        import jax

        jax_device = resolve_jax_device(new_device)

        # Pytree-compatible types (dict, list, tuple): device_put recurses natively
        # This is the standard path for Flax/Haiku params dicts
        if isinstance(model_or_params, (dict, list, tuple)):
            model_or_params = jax.device_put(model_or_params, jax_device)
        else:
            # Objects with jax.Array attributes: walk and move each one
            if model_or_params is not None:
                for attr_name in list(vars(model_or_params)):
                    val = getattr(model_or_params, attr_name)
                    if isinstance(val, jax.Array):
                        setattr(model_or_params, attr_name, jax.device_put(val, jax_device))

        # Free GPU memory when moving off CUDA device (GC + clear JIT caches)
        if old_device != "cpu" and old_device.startswith("cuda"):
            clear_caches = cast(Callable[[], None], jax.clear_caches)
            clear_caches()
            gc.collect()

        return model_or_params
    except ImportError:
        # JAX not available - not a JAX tool
        pass

    # Not a PyTorch or JAX model - return as-is
    # This handles CLI-only tools or tools with custom device management
    return model_or_params
