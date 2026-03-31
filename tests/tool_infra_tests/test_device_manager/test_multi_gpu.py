"""tests/tool_infra_tests/test_device_manager/test_multi_gpu.py

Tests for multi-GPU allocation."""

import os
import time
from unittest.mock import patch

import pytest

from proto_tools.utils.device_manager import DeviceManager, OffloadStrategy


@pytest.fixture(autouse=True)
def _mock_exclusive_process():
    """Disable Exclusive_Process auto-escalation so tests control the strategy."""
    with patch(
        "proto_tools.utils.device_manager.is_exclusive_process_mode",
        return_value=False,
    ):
        yield


# ── Auto allocation ─────────────────────────────────────────────────────

def test_allocate_2_gpus_auto(device_manager, mock_callback):
    """Test auto-allocating 2 GPUs (cudax2)."""
    devices = device_manager.request_device(
        "tool1", "instance1", device="cudax2", eviction_callback=mock_callback
    )

    assert devices == "cuda:0,cuda:1", "Should allocate both GPUs"
    assert "instance1" in device_manager._allocations, "Instance should be tracked"
    assert device_manager._allocations["instance1"].device_ids == [
        "cuda:0",
        "cuda:1",
    ], "Should store both device IDs"


def test_allocate_3_gpus_auto(mock_callback):
    """Test auto-allocating 3 GPUs (cudax3) with mocked 3 GPUs."""
    DeviceManager.reset_instance()

    with patch(
        "proto_tools.utils.device_manager.number_of_visible_gpus",
        return_value=3,
    ):
        dm = DeviceManager.get_instance()

        devices = dm.request_device(
            "tool1", "instance1", device="cudax3", eviction_callback=mock_callback
        )

        assert devices == "cuda:0,cuda:1,cuda:2"
        assert dm._allocations["instance1"].device_ids == [
            "cuda:0",
            "cuda:1",
            "cuda:2",
        ]

    DeviceManager.reset_instance()


# ── Explicit allocation ─────────────────────────────────────────────────

def test_allocate_2_gpus_explicit_shorthand():
    """Test explicit 2-GPU allocation with shorthand (cuda:0,2) with 3 visible GPUs."""
    DeviceManager.reset_instance()

    with patch(
        "proto_tools.utils.device_manager.number_of_visible_gpus",
        return_value=3,
    ):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,5,6"}):
            dm = DeviceManager.get_instance()
            mock_callback = lambda action: None

            devices = dm.request_device(
                "tool1",
                "instance1",
                device="cuda:0,2",
                eviction_callback=mock_callback,
            )

            assert devices == "cuda:0,cuda:2", "Should allocate cuda:0 and cuda:2"
            assert dm._allocations["instance1"].device_ids == [
                "cuda:0",
                "cuda:2",
            ], "Should store both device IDs"

    DeviceManager.reset_instance()


def test_allocate_2_gpus_explicit_verbose(device_manager, mock_callback):
    """Test explicit 2-GPU allocation with verbose syntax (cuda:0,cuda:1)."""
    devices = device_manager.request_device(
        "tool1",
        "instance1",
        device="cuda:0,cuda:1",
        eviction_callback=mock_callback,
    )

    assert devices == "cuda:0,cuda:1"
    assert device_manager._allocations["instance1"].device_ids == ["cuda:0", "cuda:1"]


def test_insufficient_gpus_for_multi_gpu_request(device_manager, mock_callback):
    """Test requesting more GPUs than available raises error."""
    with pytest.raises(RuntimeError, match="Cannot allocate"):
        device_manager.request_device(
            "tool1",
            "instance1",
            device="cudax7",
            eviction_callback=mock_callback,
        )


# ── Multi-GPU eviction ──────────────────────────────────────────────────

def test_multi_gpu_lru_eviction(device_manager, mock_callback):
    """Test LRU eviction with multi-GPU allocations."""
    device_manager.configure(offload_strategy=OffloadStrategy.CPU)

    device_manager.request_device(
        "tool1", "instance1", device="cudax2", eviction_callback=mock_callback
    )
    time.sleep(0.01)
    device_manager.request_device(
        "tool2", "instance2", device="cuda", eviction_callback=mock_callback
    )

    device_manager.request_device(
        "tool3", "instance3", device="cuda", eviction_callback=mock_callback
    )

    assert device_manager._allocations["instance1"].device_ids == [
        "cpu"
    ], "instance1 should be evicted to CPU"
    assert device_manager._allocations["instance2"].device_ids == [
        "cuda:0"
    ], "instance2 should be on cuda:0"
    assert device_manager._allocations["instance3"].device_ids == [
        "cuda:1"
    ], "instance3 should be on cuda:1"


def test_multi_gpu_request_evicts_two_single_instances(device_manager, mock_callback):
    """Test cudax2 request evicting two single-GPU instances."""
    device_manager.configure(offload_strategy=OffloadStrategy.CPU)

    device_manager.request_device(
        "tool1", "instance1", device="cuda", eviction_callback=mock_callback
    )
    time.sleep(0.01)
    device_manager.request_device(
        "tool2", "instance2", device="cuda", eviction_callback=mock_callback
    )

    assert device_manager._allocations["instance1"].device_ids == [
        "cuda:0"
    ], "instance1 should be on cuda:0"
    assert device_manager._allocations["instance2"].device_ids == [
        "cuda:1"
    ], "instance2 should be on cuda:1"

    devices = device_manager.request_device(
        "tool3", "instance3", device="cudax2", eviction_callback=mock_callback
    )

    assert device_manager._allocations["instance1"].device_ids == [
        "cpu"
    ], "instance1 should be evicted to CPU"
    assert device_manager._allocations["instance2"].device_ids == [
        "cpu"
    ], "instance2 should be evicted to CPU"
    assert device_manager._allocations["instance3"].device_ids == [
        "cuda:0",
        "cuda:1",
    ], "instance3 should get cuda:0 and cuda:1"
    assert devices == "cuda:0,cuda:1", "Should allocate both GPUs"


def test_lru_eviction_multi_gpu_request():
    """Test LRU eviction when requesting 2 GPUs with 3 single-GPU allocations (mocked 3 GPUs)."""
    DeviceManager.reset_instance()

    with patch(
        "proto_tools.utils.device_manager.number_of_visible_gpus",
        return_value=3,
    ):
        dm = DeviceManager.get_instance()
        dm.configure(offload_strategy=OffloadStrategy.CPU)

        calls1, calls2, calls3 = [], [], []
        cb1 = lambda action: calls1.append(action)
        cb2 = lambda action: calls2.append(action)
        cb3 = lambda action: calls3.append(action)

        dm.request_device(
            "tool1", "instance1", device="cuda:0", eviction_callback=cb1
        )
        time.sleep(0.01)
        dm.request_device(
            "tool2", "instance2", device="cuda:1", eviction_callback=cb2
        )
        time.sleep(0.01)
        dm.request_device(
            "tool3", "instance3", device="cuda:2", eviction_callback=cb3
        )

        assert dm._allocations["instance1"].device_ids == ["cuda:0"]
        assert dm._allocations["instance2"].device_ids == ["cuda:1"]
        assert dm._allocations["instance3"].device_ids == ["cuda:2"]

        cb4 = lambda action: None
        dm.request_device(
            "tool4", "instance4", device="cudax2", eviction_callback=cb4
        )

        assert dm._allocations["instance1"].device_ids == [
            "cpu"
        ], "instance1 should be evicted to CPU"
        assert dm._allocations["instance2"].device_ids == [
            "cpu"
        ], "instance2 should be evicted to CPU"
        assert dm._allocations["instance3"].device_ids == [
            "cuda:2"
        ], "instance3 should still be on cuda:2"
        assert dm._allocations["instance4"].device_ids == [
            "cuda:0",
            "cuda:1",
        ], "instance4 should get cuda:0 and cuda:1"

        assert len(calls1) == 1, "instance1 callback should be called"
        assert len(calls2) == 1, "instance2 callback should be called"
        assert len(calls3) == 0, "instance3 callback should not be called"

    DeviceManager.reset_instance()


# ── Multi-GPU with allow_multiple ────────────────────────────────────────

def test_multi_gpu_with_allow_multiple(device_manager, mock_callback):
    """Test multi-GPU allocation with allow_multiple_per_device=True."""
    device_manager.configure(allow_multiple_per_device=True)

    device_manager.request_device(
        "tool1", "instance1", device="cudax2", eviction_callback=mock_callback
    )
    device_manager.request_device(
        "tool2", "instance2", device="cuda", eviction_callback=mock_callback
    )

    assert "instance1" in device_manager._allocations
    assert "instance2" in device_manager._allocations


# ── Multi-GPU move and release ──────────────────────────────────────────

def test_multi_gpu_move():
    """Test moving a multi-GPU allocation to different devices with mocked 4 GPUs."""
    DeviceManager.reset_instance()

    with patch(
        "proto_tools.utils.device_manager.number_of_visible_gpus",
        return_value=4,
    ):
        dm = DeviceManager.get_instance()
        cb = lambda action: None

        dm.request_device(
            "tool1", "instance1", device="cuda:0,1", eviction_callback=cb
        )

        move_cb = lambda dev: {"success": True}
        dm.move_to_device("instance1", "cuda:2,3", move_cb)

        assert dm._allocations["instance1"].device_ids == ["cuda:2", "cuda:3"]

    DeviceManager.reset_instance()


def test_multi_gpu_release(device_manager, mock_callback):
    """Test releasing a multi-GPU allocation frees all devices."""
    device_manager.request_device(
        "tool1", "instance1", device="cudax2", eviction_callback=mock_callback
    )

    assert device_manager._allocations["instance1"].device_ids == ["cuda:0", "cuda:1"]

    device_manager.release_device("instance1")
    assert "instance1" not in device_manager._allocations

    devices = device_manager.request_device(
        "tool2", "instance2", device="cudax2", eviction_callback=mock_callback
    )
    assert devices == "cuda:0,cuda:1"


# ── Mixed single and multi-GPU ──────────────────────────────────────────

def test_mixed_single_and_multi_gpu():
    """Test mixing single-GPU and multi-GPU allocations with mocked 3 GPUs."""
    DeviceManager.reset_instance()

    with patch(
        "proto_tools.utils.device_manager.number_of_visible_gpus",
        return_value=3,
    ):
        dm = DeviceManager.get_instance()
        cb = lambda action: None

        result1 = dm.request_device(
            "tool1", "instance1", device="cuda", eviction_callback=cb
        )
        assert result1 == "cuda:0"

        result2 = dm.request_device(
            "tool2", "instance2", device="cudax2", eviction_callback=cb
        )
        assert result2 == "cuda:1,cuda:2"

        assert dm._allocations["instance1"].device_ids == ["cuda:0"]
        assert dm._allocations["instance2"].device_ids == ["cuda:1", "cuda:2"]

    DeviceManager.reset_instance()


# ── Multi-GPU status ────────────────────────────────────────────────────

def test_multi_gpu_device_status(device_manager, mock_callback):
    """Test get_device_status() correctly reports multi-GPU allocations."""
    device_manager.request_device(
        "tool1", "instance1", device="cudax2", eviction_callback=mock_callback
    )

    status = device_manager.get_device_status()

    assert "instance1" in status["allocations"]
    alloc_info = status["allocations"]["instance1"]
    assert alloc_info["tool_name"] == "tool1"

    assert "," in alloc_info["device_id"] or len(alloc_info["device_id"].split(":")) == 2
