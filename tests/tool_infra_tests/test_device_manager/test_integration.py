"""tests/tool_infra_tests/test_device_manager/test_integration.py.

Tests for DeviceManager integration with real tool instances.
"""

import time

import pytest

from proto_tools.utils.device_manager import OffloadStrategy

# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_evicted_instance_variable_still_works():
    """An instance reference remains usable after RESTART eviction.

    Keep a reference to a ToolInstance, then let DeviceManager evict and shut down
    its worker. The next call through that reference should restart the worker
    and allocate a device again.
    """
    from proto_tools.tools.testing.mock_pytorch_tool import (
        MockPyTorchToolConfig,
        MockPyTorchToolInput,
        run_mock_pytorch_tool,
    )
    from proto_tools.utils.device_manager import DeviceManager
    from proto_tools.utils.tool_instance import ToolInstance

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
        assert "mock_1" in status3["allocations"], "Instance should be re-allocated after restart"

        # Verify the restarted instance works correctly
        result4 = run_mock_pytorch_tool(input1, config1, instance="mock_1")
        assert result4.success, "Restarted instance should continue working"
        assert len(result4.results) > 0, "Should produce results"

    finally:
        # Clean up
        ToolInstance.clear_all()
        DeviceManager.reset_instance()
