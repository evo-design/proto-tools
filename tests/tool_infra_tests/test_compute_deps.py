"""Tests for centralized compute dependency detection."""

from unittest.mock import patch

import pytest

from bio_programming_tools.utils.compute_deps import (
    _get_jax_spec,
    _get_torch_spec,
    detect_compute_environment,
)
from bio_programming_tools.utils.system_info import GPUDevice, GPUInfo, get_gpu_info


# ── Compatibility matrix validation ──────────────────────────────────────────


def test_torch_compatibility_entries_valid():
    """All PyTorch compatibility entries should be valid tuples."""
    from bio_programming_tools.utils.compute_deps import _TORCH_COMPATIBILITY

    for driver_ver, (min_ver, max_ver) in _TORCH_COMPATIBILITY.items():
        assert isinstance(driver_ver, int)
        assert isinstance(min_ver, str)
        assert isinstance(max_ver, str)
        spec = f"torch>={min_ver},<{max_ver}"
        assert "torch>=" in spec
        assert ",<" in spec


def test_jax_compatibility_entries_valid():
    """All JAX compatibility entries should be valid tuples."""
    from bio_programming_tools.utils.compute_deps import _JAX_COMPATIBILITY

    for driver_ver, (min_ver, max_ver) in _JAX_COMPATIBILITY.items():
        assert isinstance(driver_ver, int)
        assert isinstance(min_ver, str)
        assert isinstance(max_ver, str)
        spec = f"jax[cuda12]>={min_ver},<{max_ver}"
        assert "jax[" in spec
        assert "]>=" in spec
        assert ",<" in spec


def test_get_torch_spec_returns_valid_format():
    """_get_torch_spec should return properly formatted specs."""
    for driver in [525, 550, 570, 999]:
        spec = _get_torch_spec(driver)
        assert spec.startswith("torch>=")
        assert ",<" in spec


def test_get_jax_spec_returns_valid_format():
    """_get_jax_spec should return properly formatted specs."""
    for driver, cuda in [(525, 12), (550, 12), (570, 13)]:
        spec, variant = _get_jax_spec(driver, cuda)
        assert spec.startswith("jax[")
        assert "]>=" in spec
        assert ",<" in spec
        assert variant in ["cuda11", "cuda12", "cuda13"]


def test_cuda13_requires_driver_580():
    """JAX cuda13 variant should only be used with driver >= 580."""
    # Driver 570 with CUDA 13 should fall back to cuda12
    spec, variant = _get_jax_spec(570, 13)
    assert variant == "cuda12"
    assert "cuda12" in spec

    # Driver 580+ with CUDA 13 should use cuda13
    spec, variant = _get_jax_spec(580, 13)
    assert variant == "cuda13"
    assert "cuda13" in spec


# ── detect_compute_environment ───────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_caches():
    """Clear LRU caches before each test to ensure mocks work correctly."""
    get_gpu_info.cache_clear()
    detect_compute_environment.cache_clear()
    yield
    get_gpu_info.cache_clear()
    detect_compute_environment.cache_clear()


def test_cpu_only():
    """CPU-only systems should get CPU recommendations."""
    fake_gpu_info = GPUInfo(
        available=False,
        count=0,
        driver_version=None,
        cuda_version=None,
        devices=[],
    )

    with patch(
        "bio_programming_tools.utils.system_info.get_gpu_info",
        return_value=fake_gpu_info,
    ):
        env = detect_compute_environment()

    assert env["DETECTED_COMPUTE_PLATFORM"] == "cpu"
    assert env["RECOMMENDED_TORCH_SPEC"] == "torch"
    assert env["RECOMMENDED_JAX_SPEC"] == "jax"
    assert "DETECTED_DRIVER_VERSION" not in env
    assert "DETECTED_CUDA_VERSION" not in env
    assert "RECOMMENDED_JAX_VARIANT" not in env


def test_h100_driver_570():
    """H100 with driver 570 (CUDA 12.8) should get torch 2.8+, jax[cuda12]."""
    fake_gpu_info = GPUInfo(
        available=True,
        count=1,
        driver_version="570.127",
        cuda_version="13.1",
        devices=[
            GPUDevice(
                index=0,
                name="NVIDIA H100",
                compute_capability="9.0",
                vram_gb=80.0,
            )
        ],
    )

    with patch(
        "bio_programming_tools.utils.system_info.get_gpu_info",
        return_value=fake_gpu_info,
    ):
        env = detect_compute_environment()

    assert env["DETECTED_COMPUTE_PLATFORM"] == "cuda"
    assert env["DETECTED_DRIVER_VERSION"] == "570"
    assert env["DETECTED_CUDA_VERSION"] == "13"
    assert env["RECOMMENDED_TORCH_SPEC"] == "torch>=2.8,<3"
    assert env["RECOMMENDED_JAX_SPEC"] == "jax[cuda12]>=0.4.20,<1"
    assert env["RECOMMENDED_JAX_VARIANT"] == "cuda12"


def test_a100_driver_550():
    """A100 with driver 550 (CUDA 12.4) should get torch 2.5+, jax[cuda12]."""
    fake_gpu_info = GPUInfo(
        available=True,
        count=1,
        driver_version="550.127",
        cuda_version="12.4",
        devices=[
            GPUDevice(
                index=0,
                name="NVIDIA A100",
                compute_capability="8.0",
                vram_gb=40.0,
            )
        ],
    )

    with patch(
        "bio_programming_tools.utils.system_info.get_gpu_info",
        return_value=fake_gpu_info,
    ):
        env = detect_compute_environment()

    assert env["DETECTED_COMPUTE_PLATFORM"] == "cuda"
    assert env["DETECTED_DRIVER_VERSION"] == "550"
    assert env["DETECTED_CUDA_VERSION"] == "12"
    assert env["RECOMMENDED_TORCH_SPEC"] == "torch>=2.5,<3"
    assert env["RECOMMENDED_JAX_SPEC"] == "jax[cuda12]>=0.4.20,<1"
    assert env["RECOMMENDED_JAX_VARIANT"] == "cuda12"


def test_old_gpu_driver_535():
    """Older GPU with driver 535 (CUDA 12.2) should get torch 2.4-2.6.x, jax[cuda12]."""
    fake_gpu_info = GPUInfo(
        available=True,
        count=1,
        driver_version="535.104",
        cuda_version="12.1",
        devices=[
            GPUDevice(
                index=0,
                name="NVIDIA V100",
                compute_capability="7.0",
                vram_gb=16.0,
            )
        ],
    )

    with patch(
        "bio_programming_tools.utils.system_info.get_gpu_info",
        return_value=fake_gpu_info,
    ):
        env = detect_compute_environment()

    assert env["DETECTED_COMPUTE_PLATFORM"] == "cuda"
    assert env["DETECTED_DRIVER_VERSION"] == "535"
    assert env["DETECTED_CUDA_VERSION"] == "12"
    assert env["RECOMMENDED_TORCH_SPEC"] == "torch>=2.4,<2.7"
    assert env["RECOMMENDED_JAX_SPEC"] == "jax[cuda12]>=0.4.20,<1"
    assert env["RECOMMENDED_JAX_VARIANT"] == "cuda12"


def test_malformed_driver_version():
    """Malformed driver version should use fallback range."""
    fake_gpu_info = GPUInfo(
        available=True,
        count=1,
        driver_version="unknown",
        cuda_version="12.4",
        devices=[
            GPUDevice(
                index=0,
                name="NVIDIA GPU",
                compute_capability="8.0",
                vram_gb=16.0,
            )
        ],
    )

    with patch(
        "bio_programming_tools.utils.system_info.get_gpu_info",
        return_value=fake_gpu_info,
    ):
        env = detect_compute_environment()

    assert env["RECOMMENDED_TORCH_SPEC"] == "torch>=2.1,<2.4"
    assert env["RECOMMENDED_JAX_SPEC"] == "jax[cuda12]>=0.4.20,<1"


def test_malformed_cuda_version():
    """Malformed CUDA version should default to CUDA 12."""
    fake_gpu_info = GPUInfo(
        available=True,
        count=1,
        driver_version="550.127",
        cuda_version="unknown",
        devices=[
            GPUDevice(
                index=0,
                name="NVIDIA GPU",
                compute_capability="8.0",
                vram_gb=16.0,
            )
        ],
    )

    with patch(
        "bio_programming_tools.utils.system_info.get_gpu_info",
        return_value=fake_gpu_info,
    ):
        env = detect_compute_environment()

    assert env["DETECTED_CUDA_VERSION"] == "12"
    assert "cuda12" in env["RECOMMENDED_JAX_SPEC"]
    assert env["RECOMMENDED_JAX_VARIANT"] == "cuda12"


def test_missing_driver_version():
    """Missing driver version (None) should use fallback range."""
    fake_gpu_info = GPUInfo(
        available=True,
        count=1,
        driver_version=None,
        cuda_version="12.4",
        devices=[
            GPUDevice(
                index=0,
                name="NVIDIA GPU",
                compute_capability="8.0",
                vram_gb=16.0,
            )
        ],
    )

    with patch(
        "bio_programming_tools.utils.system_info.get_gpu_info",
        return_value=fake_gpu_info,
    ):
        env = detect_compute_environment()

    assert env["RECOMMENDED_TORCH_SPEC"] == "torch>=2.1,<2.4"


def test_missing_cuda_version():
    """Missing CUDA version (None) should default to CUDA 12."""
    fake_gpu_info = GPUInfo(
        available=True,
        count=1,
        driver_version="550.127",
        cuda_version=None,
        devices=[
            GPUDevice(
                index=0,
                name="NVIDIA GPU",
                compute_capability="8.0",
                vram_gb=16.0,
            )
        ],
    )

    with patch(
        "bio_programming_tools.utils.system_info.get_gpu_info",
        return_value=fake_gpu_info,
    ):
        env = detect_compute_environment()

    assert env["DETECTED_CUDA_VERSION"] == "12"
    assert "cuda12" in env["RECOMMENDED_JAX_SPEC"]
