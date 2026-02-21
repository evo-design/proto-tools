"""
Infrastructure utilities for bio_programming_tools.tools.

GPU detection and device visibility.

GPU detection uses nvidia-smi rather than torch.cuda so that the
orchestrator package works with a CPU-only PyTorch install (or no
PyTorch at all).  Actual GPU workloads run inside isolated venvs
that have their own CUDA-enabled PyTorch.
"""
from __future__ import annotations

import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)


def number_of_available_gpus() -> int:
    """Returns the number of available NVIDIA GPUs via nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            # nvidia-smi returns one line per GPU, each containing total count;
            # the number of lines equals the number of GPUs.
            return len(out.stdout.strip().splitlines())
        return 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0


def _is_local_gpu_available() -> bool:
    """Check if a local NVIDIA GPU is available via nvidia-smi."""
    return shutil.which("nvidia-smi") is not None and number_of_available_gpus() > 0


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
    elif isinstance(device, str) and device.startswith("cuda:"):
        device_index = int(device.split(":", 1)[1])
        num_gpus = number_of_available_gpus()
        if device_index >= num_gpus:
            raise ValueError(
                f"Device index {device_index} is greater than the number of "
                f"available GPUs ({num_gpus})"
            )
        return str(device_index)

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
