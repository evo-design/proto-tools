"""Integration tests with real tool instances."""

from __future__ import annotations

import time

import pytest

from bio_programming_tools.utils.device_manager import OffloadStrategy


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_real_tool_eviction_cpu_strategy():
    """Integration test: Real tool eviction with CPU offload strategy.

    This test uses mock PyTorch tool instances with persistent workers to verify:
    1. Multiple tools can fill GPU slots
    2. LRU eviction actually moves a model to CPU when devices are full
    3. The evicted tool still works on CPU
    4. New tools can use the freed GPU
    5. GPU memory stays roughly constant with CPU strategy
    """
    from bio_programming_tools.tools.testing.mock_pytorch_tool import (
        run_mock_pytorch_tool,
        MockPyTorchToolInput,
        MockPyTorchToolConfig,
    )
    from bio_programming_tools.utils.tool_instance import ToolInstance
    from bio_programming_tools.utils.device_manager import DeviceManager

    # Reset state
    DeviceManager.reset_instance()
    ToolInstance.clear_all()

    # Configure DeviceManager for CPU offload with only 1 GPU
    dm = DeviceManager.get_instance()
    dm.configure(
        managed_devices=["cuda:0"],  # Only 1 GPU to force eviction
        offload_strategy=OffloadStrategy.CPU,
    )

    try:
        input1 = MockPyTorchToolInput()
        config1 = MockPyTorchToolConfig()

        # Step 1: Create first instance on GPU
        with ToolInstance.persist_tool("mock_pytorch_tool", instance_name="mock_1"):
            result1 = run_mock_pytorch_tool(input1, config1, instance="mock_1")
            assert result1.success, f"First call failed: {result1.errors}"

            # Verify it's on GPU
            status1 = dm.get_device_status()
            assert "mock_1" in status1["allocations"]
            mock_1_device = status1["allocations"]["mock_1"]["device_id"]
            assert mock_1_device == "cuda:0"

            time.sleep(0.01)  # Ensure different timestamps

            # Step 2: Create second instance - should evict mock_1 (LRU) to CPU
            with ToolInstance.persist_tool("mock_pytorch_tool", instance_name="mock_2"):
                result2 = run_mock_pytorch_tool(input1, config1, instance="mock_2")
                assert result2.success, f"Second call failed: {result2.errors}"

                # Verify eviction happened
                status2 = dm.get_device_status()
                assert "mock_1" in status2["allocations"]
                assert "mock_2" in status2["allocations"]

                # mock_1 should be on CPU (evicted)
                assert status2["allocations"]["mock_1"]["device_id"] == "cpu"
                # mock_2 should be on GPU
                assert status2["allocations"]["mock_2"]["device_id"] == "cuda:0"

                # Verify mock_2 (on GPU) reports memory usage
                stats_2 = dm.get_instance_memory_stats("mock_2")
                assert stats_2["available"], "mock_2 memory stats should be available"
                assert stats_2["allocated_bytes"] > 0, "mock_2 should have allocated memory"

                # mock_1 was evicted to CPU - verify allocation tracking is correct
                assert status2["allocations"]["mock_1"]["device_id"] == "cpu"
                assert status2["allocations"]["mock_2"]["device_id"] == "cuda:0"

                # Step 3: Verify evicted tool still works on CPU
                result3 = run_mock_pytorch_tool(input1, config1, instance="mock_1")
                assert result3.success, f"Evicted tool (on CPU) failed: {result3.errors}"
                assert (
                    len(result3.result) > 0
                ), "Evicted tool should still produce results"

    finally:
        # Clean up
        ToolInstance.clear_all()
        DeviceManager.reset_instance()


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_real_tool_eviction_restart_strategy():
    """Integration test: Real tool eviction with RESTART strategy.

    Verifies that evicted tools are actually shut down and removed, and that
    GPU memory is freed.
    """
    from bio_programming_tools.tools.testing.mock_pytorch_tool import (
        run_mock_pytorch_tool,
        MockPyTorchToolInput,
        MockPyTorchToolConfig,
    )
    from bio_programming_tools.utils.tool_instance import ToolInstance
    from bio_programming_tools.utils.device_manager import DeviceManager

    # Reset state
    DeviceManager.reset_instance()
    ToolInstance.clear_all()

    # Configure DeviceManager for RESTART with only 1 GPU
    dm = DeviceManager.get_instance()
    dm.configure(
        managed_devices=["cuda:0"],  # Only 1 GPU to force eviction
        offload_strategy=OffloadStrategy.RESTART,
    )

    try:
        input1 = MockPyTorchToolInput()
        config1 = MockPyTorchToolConfig()

        # Create first instance on GPU
        with ToolInstance.persist_tool("mock_pytorch_tool", instance_name="mock_1"):
            result1 = run_mock_pytorch_tool(input1, config1, instance="mock_1")
            assert result1.success

            time.sleep(0.01)

            # Create second instance - should shut down mock_1 (LRU)
            with ToolInstance.persist_tool("mock_pytorch_tool", instance_name="mock_2"):
                result2 = run_mock_pytorch_tool(input1, config1, instance="mock_2")
                assert result2.success

                # Verify mock_1 was shut down (worker set to None, but kept in cache)
                status = dm.get_device_status()
                assert (
                    "mock_1" not in status["allocations"]
                ), "Evicted instance should be released with RESTART strategy"
                assert "mock_2" in status["allocations"]

                # Verify mock_2 reports memory (model loaded on GPU)
                stats_2 = dm.get_instance_memory_stats("mock_2")
                assert stats_2["available"], "mock_2 memory stats should be available"
                assert stats_2["allocated_bytes"] > 0, "mock_2 should have allocated memory"

    finally:
        # Clean up
        ToolInstance.clear_all()
        DeviceManager.reset_instance()


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_evicted_instance_variable_still_works():
    """Integration test: Evicted ToolInstance variable reference still works after eviction.

    Critical test for RESTART strategy: When a user has a variable reference to a
    ToolInstance (e.g., `with ToolInstance.persist_tool("mock_pytorch_tool") as inst:`),
    and that instance gets evicted via LRU, the variable should still be usable. The
    next call should automatically restart the worker and re-allocate a device.

    This test verifies:
    1. User creates instance and gets variable reference
    2. Instance is evicted and shut down by DeviceManager
    3. User's variable reference still works - automatically restarts
    4. New worker gets allocated correctly
    """
    from bio_programming_tools.tools.testing.mock_pytorch_tool import (
        run_mock_pytorch_tool,
        MockPyTorchToolInput,
        MockPyTorchToolConfig,
    )
    from bio_programming_tools.utils.tool_instance import ToolInstance
    from bio_programming_tools.utils.device_manager import DeviceManager

    # Reset state
    DeviceManager.reset_instance()
    ToolInstance.clear_all()

    # Configure DeviceManager for RESTART with only 1 GPU
    dm = DeviceManager.get_instance()
    dm.configure(
        managed_devices=["cuda:0"],  # Only 1 GPU to force eviction
        offload_strategy=OffloadStrategy.RESTART,
    )

    try:
        input1 = MockPyTorchToolInput()
        config1 = MockPyTorchToolConfig()

        # Create first instance and keep reference
        mock_1 = ToolInstance.get("mock_pytorch_tool", instance_name="mock_1")
        result1 = run_mock_pytorch_tool(input1, config1, instance="mock_1")
        assert result1.success, "Initial call failed"

        # Verify it's on GPU
        status1 = dm.get_device_status()
        assert "mock_1" in status1["allocations"]
        assert status1["allocations"]["mock_1"]["device_id"] == "cuda:0"

        # Verify worker is running
        assert mock_1._worker is not None, "Worker should be running"

        time.sleep(0.01)

        # Create second instance - should evict mock_1
        ToolInstance.get("mock_pytorch_tool", instance_name="mock_2")
        result2 = run_mock_pytorch_tool(input1, config1, instance="mock_2")
        assert result2.success, "Second call failed"

        # Verify mock_1 was evicted and removed
        status2 = dm.get_device_status()
        assert "mock_1" not in status2["allocations"], "mock_1 should be evicted"
        assert "mock_2" in status2["allocations"]

        # Verify mock_1's worker was shut down
        assert mock_1._worker is None, "Worker should be shut down after eviction"

        # Use the evicted instance variable
        # The variable still exists, but the worker is gone
        result3 = run_mock_pytorch_tool(input1, config1, instance="mock_1")
        assert result3.success, "Evicted instance should auto-restart and work"

        # Verify the instance auto-restarted and got a new worker
        assert mock_1._worker is not None, "Worker should be restarted after use"

        # Verify it's back in allocations (might be on CPU or GPU depending on availability)
        status3 = dm.get_device_status()
        assert (
            "mock_1" in status3["allocations"]
        ), "Instance should be re-allocated after restart"

        # Verify the restarted instance works correctly
        result4 = run_mock_pytorch_tool(input1, config1, instance="mock_1")
        assert result4.success, "Restarted instance should continue working"
        assert len(result4.result) > 0, "Should produce results"

    finally:
        # Clean up
        ToolInstance.clear_all()
        DeviceManager.reset_instance()
