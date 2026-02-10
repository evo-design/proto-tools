"""
Infrastructure utilities for bio_programming.bio_tools.tools.

GPU selection (local/the cloud runtime) and device visibility.
"""
from __future__ import annotations

import os


def number_of_available_gpus() -> int:
    """Returns the number of available GPUs."""
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0


def use_cloud_gpu() -> bool:
    """
    Smart GPU selection: try local GPU first, fall back to the cloud runtime.

    Returns:
        bool: True if should use the cloud runtime, False if should use local GPU.

    Environment Variables:
        USE_CLOUD: Set to "true" to force the cloud runtime, "false" to force local
                   If not set, automatically chooses based on GPU availability
    """
    # Check if user explicitly set preference
    use_cloud_env = os.getenv("USE_CLOUD")
    if use_cloud_env is not None:
        return use_cloud_env.lower() == "true"

    # Auto-detect: try local GPU first, fall back to the cloud runtime
    if _is_local_gpu_available():
        return False
    elif _is_cloud_available():
        print("Local GPU not available, falling back to the cloud runtime")
        return True
    else:
        raise RuntimeError(
            "Neither local GPU nor the cloud runtime is available. "
            "Please either:\n"
            "1. Ensure you have CUDA available locally\n"
            "2. Set up the cloud runtime (cloud token new)\n"
            "3. Set USE_CLOUD=true to force the cloud runtime execution"
        )


def _is_local_gpu_available() -> bool:
    """Check if local GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _is_cloud_available() -> bool:
    """Check if the cloud runtime is available and configured."""
    try:
        import _gpu_runtime
        # Try creating a simple app to test authentication
        _gpu_runtime.App("test-auth")
        return True
    except (ImportError, Exception) as e:
        print(f"the cloud runtime not available: {e}")
        return False


def determine_visible_devices(device: int | str) -> str:
    """
    Returns a string corresponding to the CUDA_VISIBLE_DEVICES environment variable
    for a given device.
    """
    # If we are using the CPU, set no devices to be visible
    if device == "cpu":
        return ""

    # If CUDA is specified, but no number is provided, set the first device to be visible
    elif device == "cuda":
        return "0"

    # If CUDA is specified with a number, set the specified device to be visible
    elif hasattr(device, "startswith") and device.startswith("cuda:"):
        return device.replace("cuda:", "")

    else:
        try:
            device_int = int(device)
            if device_int >= number_of_available_gpus():
                raise ValueError(
                    f"Device index {device_int} is greater than the number of available GPUs ({number_of_available_gpus()})"
                )
            return str(device)
        except ValueError:
            raise ValueError(f"Invalid device: {device}")
