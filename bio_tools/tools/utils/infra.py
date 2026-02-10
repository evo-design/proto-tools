"""
Infrastructure utilities for bio_tools.tools.

GPU selection (local/Modal) and device visibility.
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


def use_modal_gpu() -> bool:
    """
    Smart GPU selection: try local GPU first, fall back to Modal.

    Returns:
        bool: True if should use Modal, False if should use local GPU.

    Environment Variables:
        USE_MODAL: Set to "true" to force Modal, "false" to force local
                   If not set, automatically chooses based on GPU availability
    """
    # Check if user explicitly set preference
    use_modal_env = os.getenv("USE_MODAL")
    if use_modal_env is not None:
        return use_modal_env.lower() == "true"

    # Auto-detect: try local GPU first, fall back to Modal
    if _is_local_gpu_available():
        return False
    elif _is_modal_available():
        print("Local GPU not available, falling back to Modal")
        return True
    else:
        raise RuntimeError(
            "Neither local GPU nor Modal is available. "
            "Please either:\n"
            "1. Ensure you have CUDA available locally\n"
            "2. Set up Modal (modal token new)\n"
            "3. Set USE_MODAL=true to force Modal execution"
        )


def _is_local_gpu_available() -> bool:
    """Check if local GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _is_modal_available() -> bool:
    """Check if Modal is available and configured."""
    try:
        import modal

        # Try creating a simple app to test authentication
        modal.App("test-auth")
        return True
    except (ImportError, Exception) as e:
        print(f"Modal not available: {e}")
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
