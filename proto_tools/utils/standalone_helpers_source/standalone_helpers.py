"""
proto_tools/utils/standalone_helpers_source/standalone_helpers.py

copied to each tool's standalone/ directory at runtime. It provides common utilities
that standalone scripts need but cannot import from the main package
(due to environment isolation).

DO NOT MODIFY THIS FILE INSIDE STANDALONE FOLDERS. CHANGES WILL BE OVERWRITTEN
If you need to make changes to this file, modify the source file which is located
at proto-tools/proto_tools/utils/standalone_helpers_source/standalone_helpers.py

Inside of tool standalone directories, the functions in this file can be invoked
via: from standalone_helpers import get_subprocess_device_env

This file is copied by the worker bootstrap. The source file is tracked by git,
but the copies in the standalone folders are not tracked.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Device Environment (subprocess CUDA_VISIBLE_DEVICES & JAX env vars)
# ============================================================================


def _parse_cuda_indices(device: str) -> Optional[List[int]]:
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

    for part in indices_str.split(","):
        part = part.strip()
        # Remove nested "cuda:" prefix (handles "cuda:0,cuda:1" format)
        if part.startswith("cuda:"):
            part = part[5:]
        try:
            indices.append(int(part))
        except ValueError:
            return None

    return indices


def _apply_jax_subprocess_env(env: Dict[str, str]) -> Dict[str, str]:
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


def get_subprocess_device_env(device: str) -> Dict[str, str]:
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
        logger.warning(
            f"Unexpected device format: '{device}'. Expected 'cuda:N' or 'cuda:N,M'. "
            f"Setting CUDA_VISIBLE_DEVICES to empty string."
        )
        env["CUDA_VISIBLE_DEVICES"] = ""
        return _apply_jax_subprocess_env(env)

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
            logger.error(
                f"Device index {idx} exceeds parent CUDA_VISIBLE_DEVICES length "
                f"({len(parent_devices)}). Parent devices: {parent_visible}"
            )
            # Fall back to parent's devices unchanged
            return _apply_jax_subprocess_env(env)

    physical_devices = [parent_devices[idx] for idx in indices]
    env["CUDA_VISIBLE_DEVICES"] = ",".join(physical_devices)

    logger.debug(
        f"Mapped device '{device}' to physical GPUs: {env['CUDA_VISIBLE_DEVICES']} "
        f"(parent: {parent_visible})"
    )

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

    if ":" in device:
        backend, device_idx = device.split(":", 1)
        device_idx = int(device_idx)
    else:
        backend = device
        device_idx = 0

    # JAX uses "gpu" not "cuda"
    if backend == "cuda":
        backend = "gpu"

    devices = jax.devices(backend)
    if device_idx >= len(devices):
        raise ValueError(
            f"Device {device} not available. Only {len(devices)} {backend} device(s) found."
        )
    return devices[device_idx]


# ============================================================================
# Device Movement (moving models/params between devices)
# ============================================================================


def move_model_to_device(
    model_or_params: Any,
    old_device: str,
    new_device: str,
    custom_move_fn: Optional[Callable[[Any, str, str], Any]] = None,
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
        >>> self.model = move_model_to_device(
        ...     self.model, self.device, device, custom_move_fn=_reload
        ... )
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
        if hasattr(model_or_params, "to") and callable(getattr(model_or_params, "to")):
            model_or_params = model_or_params.to(new_device)

            # Free GPU memory when moving off CUDA device
            if old_device != "cpu" and old_device.startswith("cuda"):
                if torch.cuda.is_available():
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
            jax.clear_caches()
            gc.collect()

        return model_or_params
    except ImportError:
        # JAX not available - not a JAX tool
        pass

    # Not a PyTorch or JAX model - return as-is
    # This handles CLI-only tools or tools with custom device management
    return model_or_params


# ============================================================================
# Memory Statistics (GPU memory reporting for PyTorch and JAX)
# ============================================================================


def get_pytorch_memory_stats(device: int | str = 0) -> Dict[str, Any]:
    """Helper for PyTorch tools to report GPU memory stats.

    Args:
        device (int | str): CUDA device index (int) or torch.device object

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


def get_jax_memory_stats(device_index: int = 0) -> Dict[str, Any]:
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


# ============================================================================
# Weights Directory Resolution
# ============================================================================


def resolve_weights_dir(tool_name: str) -> Optional[str]:
    """Resolve the weights directory for a tool based on PROTO_MODEL_CACHE.

    Precedence:
        1. PROTO_{TOOL}_WEIGHTS_DIR (per-tool override, always wins)
        2. PROTO_MODEL_CACHE:
           - (default): {PROTO_HOME}/proto_model_cache/{tool_name}/ (survives env rebuilds)
           - "/absolute/path": /absolute/path/{tool_name}/  (shared directory)
           - "IN_ENV": {TOOL_VENV_PATH}/model_weight_cache/ (legacy, per-venv)
           - "NONE": {VENV_PATH}/weights/ (pass-through, matches shell helper)

    Args:
        tool_name (str): The tool's directory name (e.g., "fampnn", "protenix").

    Returns:
        str | None: Absolute path string to the weights directory, or None (NONE mode
            with no per-tool override). Creates the directory if it doesn't exist.
    """
    # 1. Per-tool override always wins
    override_var = f"PROTO_{tool_name.upper()}_WEIGHTS_DIR"
    override = os.environ.get(override_var)
    if override:
        os.makedirs(override, exist_ok=True)
        return override

    # 2. PROTO_MODEL_CACHE
    mode = os.environ.get("PROTO_MODEL_CACHE", "")

    if mode == "NONE":
        # Pass-through: no managed cache, but match the shell helper's fallback
        # (setup.sh downloads to ${VENV_PATH}/weights in NONE mode)
        venv_path = os.environ.get("TOOL_VENV_PATH") or os.environ.get("VENV_PATH")
        if venv_path:
            path = os.path.join(venv_path, "weights")
            os.makedirs(path, exist_ok=True)
            return path
        return None

    if mode == "IN_ENV":
        # Legacy: weights inside the tool's venv
        venv_path = os.environ.get("TOOL_VENV_PATH") or os.environ.get("VENV_PATH")
        if venv_path:
            path = os.path.join(venv_path, "model_weight_cache")
            os.makedirs(path, exist_ok=True)
            return path
        logger.warning(
            "PROTO_MODEL_CACHE=IN_ENV but no TOOL_VENV_PATH or VENV_PATH set. "
            "Returning None; tool will use its own default."
        )
        return None

    if mode:
        # Explicit path (absolute or relative)
        cache_dir = mode
    else:
        # Default: PROTO_HOME/proto_model_cache/ directory
        proto_home = os.environ.get("PROTO_HOME", "")
        if not proto_home:
            # Same default as get_proto_home(): ~/.proto/
            proto_home = os.path.join(os.path.expanduser("~"), ".proto")
            logger.warning(
                "PROTO_HOME not set in subprocess environment. "
                "Falling back to %s. Set PROTO_HOME to customize.",
                proto_home,
            )
        cache_dir = os.path.join(proto_home, "proto_model_cache")

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, tool_name)
    os.makedirs(path, exist_ok=True)
    return path
