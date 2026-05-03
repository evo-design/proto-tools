"""tests/tool_infra_tests/test_device_manager/test_configuration.py.

Tests for DeviceManager configuration and device pool discovery.
"""

import os
from unittest.mock import patch

import pytest

from proto_tools.utils.device_manager import DeviceManager, OffloadStrategy

# ── Configuration tests ────────────────────────────────────────────────────


def test_env_var_overrides():
    """Test environment variable configuration overrides."""
    DeviceManager.reset_instance()

    with (
        patch(
            "proto_tools.utils.device_manager.number_of_visible_gpus",
            return_value=3,
        ),
        patch.dict(
            os.environ,
            {
                "BIO_TOOLS_MANAGED_DEVICES": "cuda:0,cuda:1,cuda:2",
                "BIO_TOOLS_OFFLOAD_STRATEGY": "restart",
                "BIO_TOOLS_ALLOW_MULTI_DEVICE": "true",
            },
        ),
    ):
        dm = DeviceManager.get_instance()

    assert dm._managed_devices == [
        "cuda:0",
        "cuda:1",
        "cuda:2",
    ], "Should parse managed devices from env"
    assert dm._offload_strategy == OffloadStrategy.RESTART, "Should parse offload strategy from env"
    assert dm._allow_multiple_per_device is True, "Should parse allow_multiple from env"

    DeviceManager.reset_instance()


def test_env_var_managed_devices_number_format():
    """Test that BIO_TOOLS_MANAGED_DEVICES accepts number format (same as CUDA_VISIBLE_DEVICES)."""
    DeviceManager.reset_instance()

    with (
        patch(
            "proto_tools.utils.device_manager.number_of_visible_gpus",
            return_value=3,
        ),
        patch.dict(os.environ, {"BIO_TOOLS_MANAGED_DEVICES": "0,1,2"}),
    ):
        dm = DeviceManager.get_instance()

    # Number format should be normalized to cuda:N format internally
    assert dm._managed_devices == [
        "cuda:0",
        "cuda:1",
        "cuda:2",
    ], "Should normalize number format to cuda:N"

    DeviceManager.reset_instance()


def test_env_var_managed_devices_mixed_format():
    """Test that BIO_TOOLS_MANAGED_DEVICES accepts mixed formats."""
    DeviceManager.reset_instance()

    with (
        patch(
            "proto_tools.utils.device_manager.number_of_visible_gpus",
            return_value=3,
        ),
        patch.dict(os.environ, {"BIO_TOOLS_MANAGED_DEVICES": "0,cuda:1,2"}),
    ):
        dm = DeviceManager.get_instance()

    # All should be normalized to cuda:N format
    assert dm._managed_devices == [
        "cuda:0",
        "cuda:1",
        "cuda:2",
    ], "Should normalize mixed formats to cuda:N"

    DeviceManager.reset_instance()


# ── Device pool discovery tests ────────────────────────────────────────────


def test_auto_detect_devices(device_manager):
    """Test automatic GPU detection (2 GPUs mocked)."""
    devices = device_manager._get_available_devices()
    assert devices == ["cuda:0", "cuda:1"], "Should auto-detect both mocked GPUs"


def test_cuda_visible_devices_respected(mock_2_gpus):
    """Test that CUDA_VISIBLE_DEVICES filtering works correctly.

    DeviceManager should return the logical indices, not the physical ones.
    """
    DeviceManager.reset_instance()

    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,2"}):
        dm = DeviceManager.get_instance()
        devices = dm._get_available_devices()
        # CUDA renumbers physical GPUs 0,2 as logical cuda:0,cuda:1
        assert devices == [
            "cuda:0",
            "cuda:1",
        ], "Should return logical indices, not physical"

    DeviceManager.reset_instance()


def test_managed_devices_override(device_manager):
    """Test explicit managed_devices configuration."""
    device_manager.configure(managed_devices=["cuda:1"])
    devices = device_manager._get_available_devices()
    assert devices == ["cuda:1"], "Should only manage explicitly configured devices"


def test_managed_devices_validation_invalid_device(mock_2_gpus):
    """Test that specifying invalid devices in BIO_TOOLS_MANAGED_DEVICES raises ValueError."""
    DeviceManager.reset_instance()

    # Try to configure more devices than available (only 2 GPUs exist: cuda:0, cuda:1)
    with patch.dict(os.environ, {"BIO_TOOLS_MANAGED_DEVICES": "cuda:0,cuda:1,cuda:2"}):
        dm = DeviceManager.get_instance()
        with pytest.raises(ValueError, match=r"BIO_TOOLS_MANAGED_DEVICES has invalid device\(s\)"):
            dm._get_available_devices()

    DeviceManager.reset_instance()


def test_managed_devices_validation_invalid_device_number_format(mock_2_gpus):
    """Test that validation works with number format too."""
    DeviceManager.reset_instance()

    # Try to configure device index 2 with only 2 GPUs (0 and 1 valid)
    with patch.dict(os.environ, {"BIO_TOOLS_MANAGED_DEVICES": "0,1,2"}):
        dm = DeviceManager.get_instance()
        with pytest.raises(ValueError, match=r"BIO_TOOLS_MANAGED_DEVICES has invalid device\(s\)"):
            dm._get_available_devices()

    DeviceManager.reset_instance()


def test_managed_devices_validation_no_gpus(mock_0_gpus):
    """Test that specifying devices when no GPUs exist raises ValueError."""
    DeviceManager.reset_instance()

    with patch.dict(os.environ, {"BIO_TOOLS_MANAGED_DEVICES": "cuda:0"}):
        dm = DeviceManager.get_instance()
        with pytest.raises(ValueError, match="no GPUs are available"):
            dm._get_available_devices()

    DeviceManager.reset_instance()


def test_managed_devices_defaults_to_all_visible(mock_2_gpus):
    """Test that when BIO_TOOLS_MANAGED_DEVICES is not set, all visible GPUs are used."""
    DeviceManager.reset_instance()

    # No BIO_TOOLS_MANAGED_DEVICES set - should auto-detect all 2 GPUs
    dm = DeviceManager.get_instance()
    devices = dm._get_available_devices()
    assert devices == ["cuda:0", "cuda:1"], "Should default to all visible GPUs"

    DeviceManager.reset_instance()


def test_cuda_visible_devices_one_visible():
    """Test that CUDA_VISIBLE_DEVICES=5 maps to logical cuda:0."""
    DeviceManager.reset_instance()

    # Simulate parent process with CUDA_VISIBLE_DEVICES=5
    # This means only physical GPU 5 is visible, appearing as logical cuda:0
    with (
        patch(
            "proto_tools.utils.device_manager.number_of_visible_gpus",
            return_value=1,
        ),
        patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "5"}),
    ):
        dm = DeviceManager.get_instance()
        devices = dm._get_available_devices()
        # Should return cuda:0 (the logical device, not physical GPU 5)
        assert devices == ["cuda:0"], "Should map physical GPU 5 to logical cuda:0"

    DeviceManager.reset_instance()


def test_no_gpus_available(no_gpus_manager):
    """Test behavior when no GPUs are available."""
    devices = no_gpus_manager._get_available_devices()
    assert devices == [], "Should return empty list when no GPUs"
