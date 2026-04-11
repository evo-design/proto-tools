"""tests/tool_infra_tests/test_device_manager/test_eviction_round_trip.py.

Tests for GPU tool eviction round-trip across all registered GPU tools.
"""

import time
from unittest.mock import patch

import pytest

from proto_tools.tools.tool_registry import ToolRegistry
from proto_tools.utils.device_manager import DeviceManager, OffloadStrategy
from proto_tools.utils.tool_instance import ToolInstance
from tests.tool_infra_tests.pytest_helpers import CHIMERA_ONLY_KEYS, parse_min_gpu_count

# Collect GPU tools (excluding mock/testing tools which are covered by stress tests)
# Tools requiring specific clusters get per-item markers via pytest.param.
_GPU_TOOLS = []
for _spec in ToolRegistry.list_all():
    if not _spec.uses_gpu or _spec.category == "testing":
        continue
    if _spec.key in CHIMERA_ONLY_KEYS:
        _GPU_TOOLS.append(pytest.param(_spec, id=_spec.key, marks=pytest.mark.only_chimera))
    else:
        _GPU_TOOLS.append(pytest.param(_spec, id=_spec.key))


@pytest.mark.extensive
@pytest.mark.uses_gpu
@pytest.mark.slow
@pytest.mark.parametrize("tool_spec", _GPU_TOOLS)
def test_gpu_tool_eviction_round_trip(tool_spec):
    """Run a GPU tool, evict it with a mock tool, then run it again.

    This exercises the full device lifecycle for every GPU tool:
    1. Tool loads onto GPU via DeviceManager
    2. Mock tool evicts it to CPU (LRU eviction)
    3. Tool moves back to GPU on next run (device mismatch detection)

    Verifies that every tool's standalone worker correctly handles
    to_device() calls for both GPU->CPU and CPU->GPU transitions.
    """
    n_gpus = parse_min_gpu_count(tool_spec.device_count)

    # Build managed device list: tool's GPUs + 1 extra for the evictor
    # Actually, for eviction we want all GPUs occupied so the mock tool
    # forces eviction. Use exactly n_gpus (tool fills them all, then
    # mock tool evicts it).
    managed_devices = [f"cuda:{i}" for i in range(n_gpus)]

    with patch(
        "proto_tools.utils.device_manager.number_of_visible_gpus",
        return_value=n_gpus,
    ):
        DeviceManager.reset_instance()
        ToolInstance.clear_all()
        dm = DeviceManager.get_instance()
        dm.configure(
            managed_devices=managed_devices,
            offload_strategy=OffloadStrategy.CPU,
        )
        # DeviceManager auto-escalates CPU -> RESTART under Exclusive_Process
        active_strategy = dm._offload_strategy

        try:
            # Extract the tool's registry key to derive a safe instance name
            tool_key = tool_spec.key
            instance_name = f"test_{tool_key.replace('-', '_')}"
            # Derive the tool_name for ToolInstance (directory name, snake_case)
            # The source_file path tells us: .../tools/{category}/{tool_dir}/{file}.py
            tool_dir = tool_spec.source_file.parent.name

            # --- Step 1: Run the tool on GPU ---
            example = tool_spec.example_input()
            with ToolInstance.persist_tool(tool_dir, instance_name=instance_name):
                result1 = tool_spec.function(example, instance=instance_name)
                assert result1.success, f"{tool_key} failed on first run: {result1.errors}"

                status = dm.get_device_status()
                alloc = status["allocations"].get(instance_name)
                assert alloc is not None, f"{tool_key} not found in DeviceManager allocations"
                initial_device = alloc["device_id"]
                assert initial_device != "cpu", f"{tool_key} should be on GPU, got {initial_device}"

                time.sleep(0.01)  # Ensure LRU ordering

                # --- Step 2: Mock tool evicts the real tool ---
                from proto_tools.tools.testing.mock_pytorch_tool import (
                    MockPyTorchToolConfig,
                    MockPyTorchToolInput,
                    run_mock_pytorch_tool,
                )

                mock_instance = f"evictor_{instance_name}"
                with ToolInstance.persist_tool("mock_pytorch_tool", instance_name=mock_instance):
                    mock_result = run_mock_pytorch_tool(
                        MockPyTorchToolInput(),
                        MockPyTorchToolConfig(memory_mb=64),
                        instance=mock_instance,
                    )
                    assert mock_result.success, f"Mock tool failed while evicting {tool_key}: {mock_result.errors}"

                    # Verify eviction happened
                    status = dm.get_device_status()
                    alloc = status["allocations"].get(instance_name)
                    if active_strategy == OffloadStrategy.CPU:
                        assert alloc is not None, f"{tool_key} allocation disappeared after eviction"
                        assert alloc["device_id"] == "cpu", (
                            f"{tool_key} should be evicted to CPU, got {alloc['device_id']}"
                        )
                    else:
                        # RESTART strategy removes the allocation entirely
                        assert alloc is None, (
                            f"{tool_key} allocation should be removed under RESTART strategy, but found {alloc}"
                        )

                    time.sleep(0.01)

                    # --- Step 3: Re-run the tool (should move back to GPU) ---
                    example2 = tool_spec.example_input()
                    result2 = tool_spec.function(example2, instance=instance_name)
                    assert result2.success, f"{tool_key} failed after eviction round-trip: {result2.errors}"

                    status = dm.get_device_status()
                    alloc = status["allocations"].get(instance_name)
                    assert alloc is not None, f"{tool_key} allocation missing after round-trip"
                    assert alloc["device_id"] != "cpu", (
                        f"{tool_key} should be back on GPU after round-trip, got {alloc['device_id']}"
                    )
        finally:
            ToolInstance.clear_all()
            DeviceManager.reset_instance()
