"""Tests for the DeviceManager allocation name each ToolInstance carries."""

from unittest.mock import patch

import pytest

from proto_tools.utils.device_manager import DeviceManager
from proto_tools.utils.tool_instance import ToolInstance

_TOOLKIT = "mock_pytorch_tool"


@pytest.fixture
def exclusive_1gpu_manager():
    """A fresh DeviceManager on one GPU in Exclusive_Process mode, as on Sherlock."""
    with (
        patch("proto_tools.utils.device_manager.number_of_visible_gpus", return_value=1),
        patch("proto_tools.utils.device_manager.is_exclusive_process_mode", return_value=True),
    ):
        DeviceManager.reset_instance()
        yield DeviceManager.get_instance()
        DeviceManager.reset_instance()


@pytest.fixture(autouse=True)
def _clear_instance_cache():
    """Start and end each test with no cached ToolInstances (none here start a worker)."""
    ToolInstance.clear_all()
    yield
    ToolInstance.clear_all()


# ── Naming ────────────────────────────────────────────────────────────────


def test_cached_instance_allocates_under_its_cache_key():
    """A named instance keeps its cache key as its allocation name, as callers and logs expect."""
    inst = ToolInstance.get(_TOOLKIT, instance_name="named")
    assert inst._device_manager_name() == "named"


def test_uncached_instances_get_distinct_allocation_names():
    """Two uncached instances are two processes, so they never share one allocation record."""
    first, second = ToolInstance(_TOOLKIT), ToolInstance(_TOOLKIT)
    assert first._device_manager_name() != second._device_manager_name()
    assert first._device_manager_name().startswith(f"{_TOOLKIT}#")


def test_nested_persist_tool_does_not_reuse_the_outer_allocation_name():
    """The inner block's instance holds its own worker, so it allocates apart from the outer one."""
    with ToolInstance.persist_tool(_TOOLKIT) as outer, ToolInstance.persist_tool(_TOOLKIT) as inner:
        assert inner is not outer
        assert outer._device_manager_name() == _TOOLKIT
        assert inner._device_manager_name() != _TOOLKIT


def test_allocation_name_is_fixed_once_set():
    """Gaining a cache key later does not rename the allocation it will be released under."""
    inst = ToolInstance(_TOOLKIT)
    name = inst._device_manager_name()
    inst._cache_keys.add("registered_later")
    assert inst._device_manager_name() == name


# ── DeviceManager behavior ────────────────────────────────────────────────


def test_uncached_instance_evicts_the_cached_worker_on_an_exclusive_gpu(exclusive_1gpu_manager):
    """A second same-toolkit worker evicts the first instead of failing to create a CUDA context."""
    with ToolInstance.persist_tool(_TOOLKIT) as outer, ToolInstance.persist_tool(_TOOLKIT) as inner:
        outer_calls: list[str] = []
        exclusive_1gpu_manager.request_device(
            _TOOLKIT, outer._device_manager_name(), device="cuda", eviction_callback=outer_calls.append
        )
        device = exclusive_1gpu_manager.request_device(
            _TOOLKIT, inner._device_manager_name(), device="cuda", eviction_callback=lambda _action: None
        )

        assert device == "cuda:0"
        assert outer_calls == ["shutdown"]  # Exclusive_Process escalates eviction to RESTART
        assert outer._device_manager_name() not in exclusive_1gpu_manager._allocations


def test_shutdown_releases_an_uncached_instance_allocation(exclusive_1gpu_manager):
    """An uncached instance releases what it allocated, so its GPU is not held after it exits."""
    inst = ToolInstance(_TOOLKIT)
    name = inst._device_manager_name()
    exclusive_1gpu_manager.request_device(_TOOLKIT, name, device="cuda", eviction_callback=lambda _action: None)
    assert name in exclusive_1gpu_manager._allocations

    inst.shutdown()

    assert name not in exclusive_1gpu_manager._allocations
