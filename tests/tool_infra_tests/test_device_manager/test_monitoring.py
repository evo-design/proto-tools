"""tests/tool_infra_tests/test_device_manager/test_monitoring.py

Tests for device status monitoring and memory tracking."""

import pytest

from bio_programming_tools.utils.device_manager import DeviceManager


# ── Status monitoring tests ──────────────────────────────────────────────

@pytest.mark.uses_gpu
@pytest.mark.slow
def test_get_device_status_with_tools():
    """Test get_device_status with tool instances on actual GPUs."""
    from bio_programming_tools.tools.testing.mock_pytorch_tool import (
        run_mock_pytorch_tool,
        MockPyTorchToolInput,
        MockPyTorchToolConfig,
    )
    from bio_programming_tools.utils.tool_instance import ToolInstance

    DeviceManager.reset_instance()
    ToolInstance.clear_all()

    dm = DeviceManager.get_instance()

    try:
        with ToolInstance.persist_tool("mock_pytorch_tool", instance_name="mock_status_1"):
            input1 = MockPyTorchToolInput()
            config1 = MockPyTorchToolConfig()
            result1 = run_mock_pytorch_tool(input1, config1, instance="mock_status_1")
            assert result1.success

            status = dm.get_device_status()

            # Verify status structure
            assert "available_devices" in status
            assert "offload_strategy" in status
            assert "allow_multiple_per_device" in status
            assert "allocations" in status

            # Verify allocation
            assert len(status["allocations"]) >= 1
            assert "mock_status_1" in status["allocations"]

            alloc1 = status["allocations"]["mock_status_1"]
            assert alloc1["tool_name"] == "mock_pytorch_tool"
            assert alloc1["device_id"].startswith("cuda:")
            assert "allocated_at" in alloc1
            assert "last_used" in alloc1

            # Verify available_devices is a list of cuda devices
            assert isinstance(status["available_devices"], list)
            assert all(d.startswith("cuda:") for d in status["available_devices"])

    finally:
        ToolInstance.clear_all()
        DeviceManager.reset_instance()


# ── Memory tracking tests ────────────────────────────────────────────────

def test_get_gpu_memory_used_cpu_device(device_manager):
    """Test get_gpu_memory_used returns 0 for CPU devices."""
    mem = device_manager.get_gpu_memory_used("cpu")
    assert mem == 0, "CPU devices should return 0 memory"


def test_get_gpu_memory_used_invalid_device(device_manager):
    """Test get_gpu_memory_used returns 0 for invalid device strings."""
    assert device_manager.get_gpu_memory_used("invalid") == 0
    assert device_manager.get_gpu_memory_used("cuda:abc") == 0
    assert device_manager.get_gpu_memory_used("") == 0


@pytest.mark.uses_gpu
def test_get_gpu_memory_used_valid_device(device_manager):
    """Test get_gpu_memory_used returns non-negative value for valid GPU."""
    mem = device_manager.get_gpu_memory_used("cuda:0")
    assert mem >= 0, "GPU memory should be non-negative"


def test_get_instance_memory_stats_nonexistent_instance(device_manager):
    """Test get_instance_memory_stats for non-existent instance."""
    stats = device_manager.get_instance_memory_stats("nonexistent")
    assert stats["available"] is False
    assert "not found" in stats["error"].lower()


def test_tool_instance_get_memory_stats_no_worker():
    """Test ToolInstance.get_memory_stats() with no worker."""
    from bio_programming_tools.utils.tool_instance import ToolInstance

    tool = ToolInstance.get("mock_pytorch_tool", instance_name="test_no_worker")
    stats = tool.get_memory_stats()

    assert stats["available"] is False
    assert "No worker running" in stats["error"]

    ToolInstance.clear_all()


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_memory_stats():
    """Test memory stats via both ToolInstance and DeviceManager access paths."""
    from bio_programming_tools.tools.testing.mock_pytorch_tool import (
        run_mock_pytorch_tool,
        MockPyTorchToolInput,
        MockPyTorchToolConfig,
    )
    from bio_programming_tools.utils.tool_instance import ToolInstance

    DeviceManager.reset_instance()
    ToolInstance.clear_all()

    dm = DeviceManager.get_instance()

    try:
        with ToolInstance.persist_tool("mock_pytorch_tool", instance_name="mock_mem_test"):
            input1 = MockPyTorchToolInput()
            config1 = MockPyTorchToolConfig()
            result = run_mock_pytorch_tool(input1, config1, instance="mock_mem_test")
            assert result.success

            # Access via ToolInstance
            tool = ToolInstance.get("mock_pytorch_tool", instance_name="mock_mem_test")
            ti_stats = tool.get_memory_stats()

            assert ti_stats["available"] is True, "Memory stats should be available"
            assert ti_stats["framework"] == "pytorch"

            # Verify standardized keys
            required_keys = {"available", "framework", "allocated_bytes", "max_allocated_bytes"}
            assert required_keys.issubset(ti_stats.keys()), "Must have standardized keys"
            assert ti_stats["allocated_bytes"] > 0, "Model should use memory"
            assert ti_stats["max_allocated_bytes"] >= ti_stats["allocated_bytes"]

            # PyTorch-specific key
            assert "reserved_bytes" in ti_stats, "PyTorch should have reserved_bytes"

            # Access via DeviceManager
            dm_stats = dm.get_instance_memory_stats("mock_mem_test")

            assert dm_stats["available"] is True
            assert dm_stats["framework"] == "pytorch"
            assert "allocated_bytes" in dm_stats
            assert "max_allocated_bytes" in dm_stats
            assert dm_stats["allocated_bytes"] > 0

    finally:
        ToolInstance.clear_all()
        DeviceManager.reset_instance()


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_instance_memory_vs_total_gpu_memory():
    """Test relationship between instance memory and total GPU memory."""
    from bio_programming_tools.tools.testing.mock_pytorch_tool import (
        run_mock_pytorch_tool,
        MockPyTorchToolInput,
        MockPyTorchToolConfig,
    )
    from bio_programming_tools.utils.tool_instance import ToolInstance

    DeviceManager.reset_instance()
    ToolInstance.clear_all()

    dm = DeviceManager.get_instance()

    try:
        with ToolInstance.persist_tool("mock_pytorch_tool", instance_name="mock_compare"):
            input1 = MockPyTorchToolInput()
            config1 = MockPyTorchToolConfig()
            result = run_mock_pytorch_tool(input1, config1, instance="mock_compare")
            assert result.success

            # Get per-instance memory
            instance_stats = dm.get_instance_memory_stats("mock_compare")
            instance_mem = instance_stats["allocated_bytes"]

            # Get total GPU memory
            total_mem = dm.get_gpu_memory_used("cuda:0")

            # Instance memory should be <= total GPU memory
            assert instance_mem <= total_mem, (
                f"Instance memory ({instance_mem / 1e9:.2f} GB) should be <= "
                f"total GPU memory ({total_mem / 1e9:.2f} GB)"
            )

            assert instance_mem > 0, "Instance should use memory"
            assert total_mem > 0, "Total GPU should show memory usage"

    finally:
        ToolInstance.clear_all()
        DeviceManager.reset_instance()
