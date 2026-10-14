"""tests/tool_infra_tests/test_device_manager/test_eviction_round_trip.py.

Tests for GPU eviction round-trip and memory reporting, one case per GPU toolkit.
"""

from __future__ import annotations

import time
from collections import defaultdict
from unittest.mock import patch

import pytest

from proto_tools.tools.tool_registry import ToolRegistry, ToolSpec
from proto_tools.utils.device_manager import DeviceManager, OffloadStrategy
from proto_tools.utils.standalone_helpers_source.standalone_helpers.memory import JAX_UNTRACKED_MEMORY_REASON
from proto_tools.utils.tool_instance import ToolInstance
from tests.tool_infra_tests.pytest_helpers import parse_min_gpu_count

# Tools that take a GPU only when explicitly asked, so their minimal() config runs on CPU and
# never occupies a device to evict. Their to_device() is a passthrough — the CLI subprocess
# releases the GPU when it exits — so there is no residency for this round-trip to exercise.
_NO_EVICTION_TOOLS = frozenset({"mmseqs2-homology-search", "mmseqs2-search-proteins"})

# Toolkits whose worker holds no framework-tracked GPU state, so get_memory_stats() reports
# available=False: fresh CLI subprocess per call, or TensorFlow (whose stats are not tracked).
_NO_MEMORY_STATS_TOOLKITS = frozenset({"alphafold3", "deeppbs_specificity", "germinal", "opendde", "spliceai"})

# Key tokens in order of preference when choosing a toolkit's representative key: the cheapest
# call that still loads the toolkit's model.
_KEY_TOKEN_PREFERENCE = ("score", "embedding", "prediction", "predict")


def _representative_key_rank(spec: ToolSpec) -> tuple[int, str]:
    """Rank a key by its best preferred token, breaking ties alphabetically."""
    tokens = spec.key.split("-")
    rank = min(
        (i for i, token in enumerate(_KEY_TOKEN_PREFERENCE) if token in tokens), default=len(_KEY_TOKEN_PREFERENCE)
    )
    return rank, spec.key


def _one_spec_per_toolkit() -> list[ToolSpec]:
    """Return one representative non-mock GPU tool per toolkit.

    Device movement lives in the toolkit's standalone worker, which every key in the toolkit
    shares, so one key per toolkit exercises the same ``to_device()``/``get_memory_stats()``.
    """
    by_toolkit: dict[str, list[ToolSpec]] = defaultdict(list)
    for spec in ToolRegistry.list_all():
        if spec.uses_gpu and spec.category != "testing":
            by_toolkit[spec.source_file.parent.name].append(spec)
    return [min(specs, key=_representative_key_rank) for _, specs in sorted(by_toolkit.items())]


_GPU_TOOLKITS = [pytest.param(spec, id=spec.key) for spec in _one_spec_per_toolkit()]


def _assert_memory_stats(tool_key: str, toolkit: str, stats: dict) -> None:
    """Assert a loaded worker reports well-formed memory stats, or opts out by a documented rule.

    Byte counts may be 0: workers that load per call or free GPU memory after each call hold
    nothing between calls. JAX workers run XLA's platform allocator, which tracks no stats, so
    they may report unavailable, but only with ``JAX_UNTRACKED_MEMORY_REASON``.
    """
    if toolkit in _NO_MEMORY_STATS_TOOLKITS:
        assert stats["available"] is False, (
            f"{tool_key}: listed in _NO_MEMORY_STATS_TOOLKITS but reports memory stats; drop it from the set"
        )
        return
    assert stats["framework"] in ("pytorch", "jax"), f"{tool_key}: unexpected framework in {stats}"
    if stats["framework"] == "jax" and stats["available"] is False:
        assert stats["reason"] == JAX_UNTRACKED_MEMORY_REASON, f"{tool_key}: JAX memory stats failed: {stats}"
        return
    assert stats["available"] is True, f"{tool_key}: get_memory_stats() unavailable with the model loaded: {stats}"
    for key in ("allocated_bytes", "max_allocated_bytes"):
        assert isinstance(stats[key], int) and stats[key] >= 0, f"{tool_key}: malformed {key} in {stats}"


@pytest.mark.extensive
@pytest.mark.uses_gpu
@pytest.mark.slow
@pytest.mark.parametrize("tool_spec", _GPU_TOOLKITS)
def test_gpu_tool_eviction_round_trip(tool_spec):
    """Run a GPU toolkit, check its memory stats, evict it with a mock tool, then run it again.

    This exercises the full device lifecycle for every GPU toolkit's standalone worker:
    1. Tool loads onto GPU via DeviceManager and reports its memory via get_memory_stats()
    2. Mock tool evicts it to CPU (LRU eviction)
    3. Tool moves back to GPU on next run (device mismatch detection)

    Each case runs with ``config_model.minimal()``, so a broken ``minimal()`` for the
    representative key surfaces here, not just a device-lifecycle regression.
    """
    if tool_spec.key in _NO_EVICTION_TOOLS:
        assert tool_spec.config_model.minimal().gpus_per_instance == 0, (
            f"{tool_spec.key}: listed in _NO_EVICTION_TOOLS but its minimal() config claims a GPU; "
            "drop it from the set so the round-trip runs"
        )
        pytest.skip(f"{tool_spec.key}: minimal() config runs on CPU, so there is no eviction to exercise")

    n_gpus = parse_min_gpu_count(tool_spec.device_count)

    # Manage exactly the tool's GPUs so the mock tool must evict it.
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
            tool_key = tool_spec.key
            instance_name = f"test_{tool_key.replace('-', '_')}"
            # ToolInstance toolkit name is the tool's directory: .../tools/{category}/{toolkit}/{file}.py
            toolkit = tool_spec.source_file.parent.name

            # --- Step 1: Run the tool on GPU ---
            # Minimal config — this test only exercises the device round-trip, not a
            # production-scale run (germinal would otherwise launch a full campaign).
            example = tool_spec.example_input()
            config = tool_spec.config_model.minimal()
            with ToolInstance.persist_tool(toolkit, instance_name=instance_name):
                result1 = tool_spec.function(example, config, instance=instance_name)
                assert result1.success, f"{tool_key} failed on first run: {result1.errors}"

                status = dm.get_device_status()
                alloc = status["allocations"].get(instance_name)
                assert alloc is not None, f"{tool_key} not found in DeviceManager allocations"
                initial_device = alloc["device_id"]
                assert initial_device != "cpu", f"{tool_key} should be on GPU, got {initial_device}"

                stats = ToolInstance.get(toolkit, instance_name=instance_name).get_memory_stats()
                _assert_memory_stats(tool_key, toolkit, stats)

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
                    result2 = tool_spec.function(example2, config, instance=instance_name)
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
