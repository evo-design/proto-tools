"""
Centralized compute dependency detection for PyTorch and JAX.

Detects GPU hardware (driver, CUDA version) and recommends compatible
PyTorch/JAX version constraints. These are injected as environment variables
into tool subprocess environments, allowing standalone scripts to consume them
without code coupling.

Environment variables exported:
    DETECTED_COMPUTE_PLATFORM: "cpu" or "cuda"
    DETECTED_DRIVER_VERSION: NVIDIA driver major version (e.g., "550")
    DETECTED_CUDA_VERSION: CUDA toolkit major version (e.g., "12")
    RECOMMENDED_TORCH_SPEC: PyTorch version constraint (e.g., "torch>=2.7,<2.10")
    RECOMMENDED_TORCH_INDEX: PyTorch wheel index URL for the detected CUDA variant
    RECOMMENDED_JAX_SPEC: JAX version constraint with CUDA variant
    RECOMMENDED_JAX_VARIANT: JAX CUDA variant (e.g., "cuda12")

References:
    PyTorch CUDA compatibility:
        https://github.com/pytorch/pytorch/blob/main/RELEASE.md
        https://pytorch.org/get-started/locally/
    JAX CUDA compatibility:
        https://docs.jax.dev/en/latest/installation.html
    NVIDIA driver-CUDA mapping:
        https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
        https://docs.nvidia.com/deploy/cuda-compatibility/
"""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# Compatibility Matrices
# ============================================================================
# Based on official PyTorch Release Matrix (https://github.com/pytorch/pytorch/blob/main/RELEASE.md)
# and NVIDIA driver-CUDA mapping (https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
#
# NVIDIA Driver → CUDA Native Support:
#   Driver 570+: CUDA 12.8
#   Driver 550-569: CUDA 12.4 (cu126 via forward compat; cu124 lacks torch 2.7+)
#   Driver 535-549: CUDA 12.2
#   Driver 525-534: CUDA 12.0-12.1
#
# PyTorch CUDA Support (stable):
#   PyTorch 2.8+: CUDA 12.8 (driver 570+)
#   PyTorch 2.5+: CUDA 12.4 (driver 550+)
#   PyTorch 2.4-2.6.x: CUDA 12.1 (driver 525-549, capped because 2.7+ ships CUDA 12.8 runtime libs)

# PyTorch version + wheel index compatibility by NVIDIA driver major version
# Format: {min_driver_major: (min_torch, max_torch_exclusive, cuda_variant)}
# The cuda_variant maps to https://download.pytorch.org/whl/{cuda_variant}
# Sources:
#   PyTorch releases: https://github.com/pytorch/pytorch/blob/main/RELEASE.md
#   Driver → CUDA mapping: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
#   Available wheel indices: https://download.pytorch.org/whl/
_TORCH_COMPATIBILITY = {
    570: ("2.8", "3", "cu128"),      # Driver 570+ (CUDA 12.8): torch 2.8+
    550: ("2.5", "3", "cu126"),      # Driver 550-569 (CUDA 12.4+): torch 2.5+ (cu126 via CUDA forward compat)
    535: ("2.4", "2.7", "cu121"),    # Driver 535-549 (CUDA 12.2): torch 2.4-2.6.x (2.7+ ships CUDA 12.8 runtime)
    525: ("2.4", "2.7", "cu121"),    # Driver 525-534 (CUDA 12.0-12.1): torch 2.4-2.6.x
    0: ("2.1", "2.4", "cu118"),      # Fallback for older drivers (CUDA 11.x era)
}

_TORCH_INDEX_BASE = "https://download.pytorch.org/whl"

# JAX version compatibility by NVIDIA driver major version
# Based on official JAX docs (https://docs.jax.dev/en/latest/installation.html)
# JAX cuda12 requires driver >= 525, JAX cuda13 requires driver >= 580
# Format: {min_driver_major: (min_jax_version, max_jax_version_exclusive)}
_JAX_COMPATIBILITY = {
    525: ("0.4.20", "1"),    # Driver 525+ supports jax[cuda12] (all modern versions)
    0: ("0.4.20", "1"),      # Fallback (same as above, JAX requires modern drivers)
}


# ============================================================================
# Detection Functions
# ============================================================================
def _get_torch_entry(driver_major: int) -> tuple[str, str, str]:
    """Look up the _TORCH_COMPATIBILITY entry for a driver version.

    Returns
    -------
    tuple[str, str, str]
        (min_torch, max_torch_exclusive, cuda_variant).
    """
    for min_driver in sorted(_TORCH_COMPATIBILITY.keys(), reverse=True):
        if driver_major >= min_driver:
            return _TORCH_COMPATIBILITY[min_driver]
    return _TORCH_COMPATIBILITY[0]


def _get_torch_spec(driver_major: int) -> str:
    """Get PyTorch version constraint for a given driver version.

    Parameters
    ----------
    driver_major
        NVIDIA driver major version (e.g., 550).

    Returns
    -------
    str
        PyTorch version constraint (e.g., "torch>=2.5,<3").
    """
    min_ver, max_ver, _ = _get_torch_entry(driver_major)
    return f"torch>={min_ver},<{max_ver}"


def _get_torch_index(driver_major: int) -> str:
    """Get PyTorch wheel index URL for a given driver version.

    Parameters
    ----------
    driver_major
        NVIDIA driver major version (e.g., 550).

    Returns
    -------
    str
        PyTorch wheel index URL (e.g., "https://download.pytorch.org/whl/cu124").
    """
    _, _, variant = _get_torch_entry(driver_major)
    return f"{_TORCH_INDEX_BASE}/{variant}"


def _get_jax_spec(driver_major: int, cuda_major: int) -> tuple[str, str]:
    """Get JAX version constraint and CUDA variant for given hardware.

    Parameters
    ----------
    driver_major
        NVIDIA driver major version (e.g., 550).
    cuda_major
        CUDA toolkit major version (e.g., 12).

    Returns
    -------
    tuple[str, str]
        (jax_spec, jax_variant) - e.g., ("jax[cuda12]>=0.6,<0.9", "cuda12").
    """
    # Determine JAX CUDA variant (cuda12 or cuda13)
    # JAX cuda13 requires driver >= 580, fall back to cuda12 if driver too old
    if cuda_major >= 13 and driver_major < 580:
        jax_variant = "cuda12"
    elif cuda_major < 12:
        jax_variant = "cuda11"
    else:
        jax_variant = f"cuda{cuda_major}"

    # Find the highest matching driver threshold
    for min_driver in sorted(_JAX_COMPATIBILITY.keys(), reverse=True):
        if driver_major >= min_driver:
            min_ver, max_ver = _JAX_COMPATIBILITY[min_driver]
            jax_spec = f"jax[{jax_variant}]>={min_ver},<{max_ver}"
            return jax_spec, jax_variant

    # Fallback to oldest supported range
    min_ver, max_ver = _JAX_COMPATIBILITY[0]
    jax_spec = f"jax[{jax_variant}]>={min_ver},<{max_ver}"
    return jax_spec, jax_variant


@functools.lru_cache(maxsize=1)
def detect_compute_environment() -> dict[str, str]:
    """Detect compute hardware and return recommended dependency specs.

    Cached for the lifetime of the process

    Queries GPU hardware via system_info.get_gpu_info() and returns a dict
    of environment variables containing detected hardware info and recommended
    PyTorch/JAX version constraints.

    Returns
    -------
    dict[str, str]
        Environment variables to inject into tool subprocess environments:
        - DETECTED_COMPUTE_PLATFORM: "cpu" or "cuda"
        - DETECTED_DRIVER_VERSION: NVIDIA driver major version
        - DETECTED_CUDA_VERSION: CUDA toolkit major version
        - RECOMMENDED_TORCH_SPEC: PyTorch version constraint
        - RECOMMENDED_TORCH_INDEX: PyTorch wheel index URL
        - RECOMMENDED_JAX_SPEC: JAX version constraint with CUDA variant
        - RECOMMENDED_JAX_VARIANT: JAX CUDA variant (e.g., "cuda12")

    Examples
    --------
    >>> env = detect_compute_environment()
    >>> # On GPU system with driver 550, CUDA 12.4:
    >>> env["DETECTED_COMPUTE_PLATFORM"]
    'cuda'
    >>> env["RECOMMENDED_TORCH_SPEC"]
    'torch>=2.7,<2.10'
    """
    from .system_info import get_gpu_info

    gpu_info = get_gpu_info()

    # CPU-only fallback
    if not gpu_info.available or gpu_info.count == 0:
        logger.info("No GPU detected, recommending CPU-only PyTorch/JAX")
        return {
            "DETECTED_COMPUTE_PLATFORM": "cpu",
            "RECOMMENDED_TORCH_SPEC": "torch",
            "RECOMMENDED_TORCH_INDEX": "https://download.pytorch.org/whl/cpu",
            "RECOMMENDED_JAX_SPEC": "jax",
        }

    # Parse driver and CUDA versions
    driver_version = gpu_info.driver_version or "0"
    cuda_version = gpu_info.cuda_version or "12"  # Default to CUDA 12

    try:
        driver_major = int(driver_version.split(".")[0])
    except (ValueError, IndexError):
        logger.warning(
            f"Could not parse driver version '{driver_version}', "
            "using fallback compatibility range"
        )
        driver_major = 0

    try:
        cuda_major = int(cuda_version.split(".")[0])
    except (ValueError, IndexError):
        logger.warning(
            f"Could not parse CUDA version '{cuda_version}', "
            "defaulting to CUDA 12"
        )
        cuda_major = 12

    # Get recommended specs
    torch_spec = _get_torch_spec(driver_major)
    torch_index = _get_torch_index(driver_major)
    jax_spec, jax_variant = _get_jax_spec(driver_major, cuda_major)

    logger.info(
        f"Compute environment detected: platform=cuda, "
        f"driver={driver_version} (major={driver_major}), "
        f"cuda={cuda_version} (major={cuda_major})"
    )
    logger.info(
        f"Recommended: PyTorch {torch_spec} (index: {torch_index}), JAX {jax_spec}"
    )

    env = {
        "DETECTED_COMPUTE_PLATFORM": "cuda",
        "DETECTED_DRIVER_VERSION": str(driver_major),
        "DETECTED_CUDA_VERSION": str(cuda_major),
        "RECOMMENDED_TORCH_SPEC": torch_spec,
        "RECOMMENDED_TORCH_INDEX": torch_index,
        "RECOMMENDED_JAX_SPEC": jax_spec,
        "RECOMMENDED_JAX_VARIANT": jax_variant,
    }

    # Export GPU compute capability for CUDA JIT compilation.
    # Use major.0 since nvcc/PyTorch only recognize major versions (e.g., "12.0" not "12.1").
    if gpu_info.devices:
        cc = gpu_info.devices[0].compute_capability
        if cc:
            cc_major = cc.split(".")[0]
            env["TORCH_CUDA_ARCH_LIST"] = f"{cc_major}.0"

    return env
