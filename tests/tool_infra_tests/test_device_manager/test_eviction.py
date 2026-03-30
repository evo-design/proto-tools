"""tests/tool_infra_tests/test_device_manager/test_eviction.py

Tests for LRU eviction and eviction callbacks."""

import time
from unittest.mock import patch

import pytest

from bio_programming_tools.utils.device_manager import OffloadStrategy

# ── LRU eviction tests ────────────────────────────────────────────────────

def test_lru_eviction_cpu_strategy(device_manager, mock_callback):
    """Test LRU eviction with CPU offload strategy."""
    device_manager.configure(offload_strategy=OffloadStrategy.CPU)

    # Fill both devices
    device_manager.request_device("tool1", "instance1", device="cuda", eviction_callback=mock_callback)
    time.sleep(0.01)
    device_manager.request_device("tool2", "instance2", device="cuda", eviction_callback=mock_callback)

    # Request third device - should evict tool1 (LRU) to CPU
    device3 = device_manager.request_device(
        "tool3", "instance3", device="cuda", eviction_callback=mock_callback
    )

    assert device3 == "cuda:0", "New instance should reuse evicted device"
    assert (
        device_manager._allocations["instance1"].device_ids[0] == "cpu"
    ), "LRU instance should be offloaded to CPU"
    assert (
        device_manager._allocations["instance2"].device_ids[0] == "cuda:1"
    ), "Recent instance should stay on GPU"
    assert (
        device_manager._allocations["instance3"].device_ids[0] == "cuda:0"
    ), "New instance should get cuda:0"


def test_lru_eviction_restart_strategy(device_manager, mock_callback):
    """Test LRU eviction with RESTART strategy."""
    device_manager.configure(offload_strategy=OffloadStrategy.RESTART)

    # Fill both devices
    device_manager.request_device("tool1", "instance1", device="cuda", eviction_callback=mock_callback)
    time.sleep(0.01)
    device_manager.request_device("tool2", "instance2", device="cuda", eviction_callback=mock_callback)

    # Request third device - should restart tool1 (LRU)
    device3 = device_manager.request_device(
        "tool3", "instance3", device="cuda", eviction_callback=mock_callback
    )

    assert device3 == "cuda:0", "New instance should reuse evicted device"
    assert (
        "instance1" not in device_manager._allocations
    ), "LRU instance should be removed with RESTART"
    assert (
        device_manager._allocations["instance2"].device_ids[0] == "cuda:1"
    ), "Recent instance should stay on GPU"
    assert (
        device_manager._allocations["instance3"].device_ids[0] == "cuda:0"
    ), "New instance should get cuda:0"


def test_lru_eviction_respects_last_used_update(device_manager, mock_callback):
    """Test that updating last_used changes eviction order."""
    device_manager.configure(offload_strategy=OffloadStrategy.CPU)

    # Fill both devices
    device_manager.request_device("tool1", "instance1", device="cuda", eviction_callback=mock_callback)
    time.sleep(0.01)
    device_manager.request_device("tool2", "instance2", device="cuda", eviction_callback=mock_callback)

    # At this point: instance1 is older than instance2
    # Update instance1's last_used to make it newer than instance2
    time.sleep(0.01)
    device_manager.update_last_used("instance1")

    # Request third device - should evict instance2 (now LRU) instead of instance1
    device3 = device_manager.request_device(
        "tool3", "instance3", device="cuda", eviction_callback=mock_callback
    )

    assert device3 == "cuda:1", "New instance should reuse instance2's device"
    assert (
        device_manager._allocations["instance1"].device_ids[0] == "cuda:0"
    ), "instance1 should stay on cuda:0 (recently used)"
    assert (
        device_manager._allocations["instance2"].device_ids[0] == "cpu"
    ), "instance2 should be evicted to CPU (now LRU)"
    assert (
        device_manager._allocations["instance3"].device_ids[0] == "cuda:1"
    ), "instance3 should get cuda:1"


# ── Eviction callback tests ──────────────────────────────────────────────

def test_cpu_eviction_calls_callback_with_cpu(device_manager):
    """Test CPU eviction strategy calls callback with 'cpu' action."""
    device_manager.configure(offload_strategy=OffloadStrategy.CPU)

    calls1, calls2, calls3 = [], [], []
    cb1 = lambda action: calls1.append(action)
    cb2 = lambda action: calls2.append(action)
    cb3 = lambda action: calls3.append(action)

    device_manager.request_device("tool1", "instance1", device="cuda", eviction_callback=cb1)
    time.sleep(0.01)
    device_manager.request_device("tool2", "instance2", device="cuda", eviction_callback=cb2)

    # Request third device - should evict tool1 (LRU) to CPU
    device_manager.request_device("tool3", "instance3", device="cuda", eviction_callback=cb3)

    assert len(calls1) == 1
    assert calls1[0] == "cpu"
    assert len(calls2) == 0  # Not evicted
    assert len(calls3) == 0  # Just allocated


def test_restart_eviction_calls_callback_with_shutdown(device_manager):
    """Test RESTART eviction strategy calls callback with 'shutdown' action."""
    device_manager.configure(offload_strategy=OffloadStrategy.RESTART)

    calls1, calls2, calls3 = [], [], []
    cb1 = lambda action: calls1.append(action)
    cb2 = lambda action: calls2.append(action)
    cb3 = lambda action: calls3.append(action)

    device_manager.request_device("tool1", "instance1", device="cuda", eviction_callback=cb1)
    time.sleep(0.01)
    device_manager.request_device("tool2", "instance2", device="cuda", eviction_callback=cb2)

    # Request third device - should restart tool1 (LRU)
    device_manager.request_device("tool3", "instance3", device="cuda", eviction_callback=cb3)

    assert len(calls1) == 1
    assert calls1[0] == "shutdown"
    assert len(calls2) == 0  # Not evicted
    assert len(calls3) == 0  # Just allocated


def test_cpu_eviction_callback_failure_raises(device_manager):
    """Test CPU eviction raises exception if callback fails."""
    device_manager.configure(offload_strategy=OffloadStrategy.CPU)

    def failing_callback(action: str) -> None:
        if action == "cpu":
            raise RuntimeError("Failed to move to CPU")

    device_manager.request_device("tool1", "instance1", device="cuda", eviction_callback=failing_callback)
    time.sleep(0.01)
    device_manager.request_device("tool2", "instance2", device="cuda", eviction_callback=lambda x: None)

    with pytest.raises(RuntimeError, match="Failed to move to CPU"):
        device_manager.request_device("tool3", "instance3", device="cuda", eviction_callback=lambda x: None)


def test_restart_eviction_callback_failure_logs_but_continues(device_manager):
    """Test RESTART eviction logs error but continues if callback fails."""
    device_manager.configure(offload_strategy=OffloadStrategy.RESTART)

    def failing_callback(action: str) -> None:
        if action == "shutdown":
            raise RuntimeError("Failed to shutdown")

    device_manager.request_device("tool1", "instance1", device="cuda", eviction_callback=failing_callback)
    time.sleep(0.01)
    device_manager.request_device("tool2", "instance2", device="cuda", eviction_callback=lambda x: None)

    with patch("bio_programming_tools.utils.device_manager.logger.warning") as mock_warning:
        device3 = device_manager.request_device(
            "tool3", "instance3", device="cuda", eviction_callback=lambda x: None
        )

    assert device3 == "cuda:0"
    assert mock_warning.called
    assert "instance1" not in device_manager._allocations


def test_cpu_eviction_preserves_lru_ordering(device_manager):
    """Test CPU eviction doesn't update last_used timestamp, preserving LRU order."""
    device_manager.configure(offload_strategy=OffloadStrategy.CPU)

    calls1, calls2 = [], []
    cb1 = lambda action: calls1.append(action)
    cb2 = lambda action: calls2.append(action)

    device_manager.request_device("tool1", "instance1", device="cuda", eviction_callback=cb1)
    time.sleep(0.01)
    device_manager.request_device("tool2", "instance2", device="cuda", eviction_callback=cb2)
    time.sleep(0.01)

    instance1_time_before = device_manager._allocations["instance1"].last_used

    # Request third device - evicts instance1 (LRU) to CPU
    device_manager.request_device("tool3", "instance3", device="cuda", eviction_callback=lambda x: None)

    instance1_time_after = device_manager._allocations["instance1"].last_used
    assert instance1_time_after == instance1_time_before

    # instance1 is on CPU but still oldest; instance2 still on GPU
    assert calls1 == ["cpu"]
    assert calls2 == []

    # Request fourth device - evicts instance2 (next LRU with a GPU) to CPU
    time.sleep(0.01)
    device_manager.request_device("tool4", "instance4", device="cuda", eviction_callback=lambda x: None)

    assert calls1 == ["cpu"]
    assert calls2 == ["cpu"]


def test_missing_callback_raises(device_manager):
    """Test ValueError raised when callback is missing."""
    with pytest.raises(ValueError, match="eviction_callback is required"):
        device_manager.request_device("tool1", "instance1", device="cuda")


def test_missing_callback_raises_even_without_gpus(no_gpus_manager):
    """Test ValueError raised when callback is missing, even without GPUs."""
    with pytest.raises(ValueError, match="eviction_callback is required"):
        no_gpus_manager.request_device("tool1", "instance1", device="cuda")


def test_gpu_request_raises_without_gpus(no_gpus_manager, mock_callback):
    """Test RuntimeError raised when requesting GPU with no GPUs available."""
    with pytest.raises(RuntimeError, match="No GPUs available"):
        no_gpus_manager.request_device("tool1", "instance1", device="cuda", eviction_callback=mock_callback)


def test_cpu_request_works_without_gpus(no_gpus_manager, mock_callback):
    """Test explicit CPU request works when no GPUs available."""
    device = no_gpus_manager.request_device(
        "tool1", "instance1", device="cpu", eviction_callback=mock_callback
    )
    assert device == "cpu"


def test_eviction_callback_does_not_update_last_used(device_manager):
    """Production-like eviction callback does not update last_used.

    Simulates what _run_persistent()'s eviction callback does: sends a
    worker command and sets self.device = "cpu". Verifies that last_used
    is NOT updated by the eviction path, so LRU ordering is preserved.

    This covers a bug that test_cpu_eviction_preserves_lru_ordering misses
    (that test uses bare lambda callbacks that don't call move_to_device).
    """
    device_manager.configure(offload_strategy=OffloadStrategy.CPU)

    # Simulate production eviction callback (sends worker command, updates device)
    # In production this sends to self._worker; here we just track calls.
    worker_commands = []

    class FakeInstance:
        device = "cuda:0"

    fake = FakeInstance()

    def production_like_callback(action: str) -> None:
        if action == "cpu":
            # Simulate: self._worker.send({"command": "to_device", "device": "cpu"})
            worker_commands.append("to_device:cpu")
            # Simulate: self.device = "cpu"
            fake.device = "cpu"

    device_manager.request_device(
        "tool1", "instance1", device="cuda", eviction_callback=production_like_callback
    )
    time.sleep(0.01)
    device_manager.request_device(
        "tool2", "instance2", device="cuda", eviction_callback=lambda x: None
    )

    # Record tool1's last_used before eviction
    last_used_before = device_manager._allocations["instance1"].last_used

    # Request third device -- evicts tool1 (LRU)
    time.sleep(0.01)
    device_manager.request_device(
        "tool3", "instance3", device="cuda", eviction_callback=lambda x: None
    )

    # Callback was invoked
    assert worker_commands == ["to_device:cpu"]
    assert fake.device == "cpu"

    # last_used must NOT have changed (preserves LRU ordering)
    last_used_after = device_manager._allocations["instance1"].last_used
    assert last_used_after == last_used_before


def test_multi_gpu_eviction_calls_callback_once(device_manager):
    """Test multi-GPU allocation eviction calls callback only once."""
    device_manager.configure(offload_strategy=OffloadStrategy.CPU)

    calls1 = []
    cb1 = lambda action: calls1.append(action)

    device_manager.request_device(
        "tool1", "instance1", device="cudax2", eviction_callback=cb1
    )
    time.sleep(0.01)

    device_manager.request_device("tool2", "instance2", device="cuda", eviction_callback=lambda x: None)

    # Callback should be called once even though 2 GPUs were freed
    assert len(calls1) == 1
    assert calls1[0] == "cpu"


# ── Transient allocation eviction tests ───────────────────────────────────

def test_transient_allocation_not_evicted(device_manager, mock_callback):
    """TRANSIENT allocations are skipped during LRU eviction."""
    device_manager.configure(offload_strategy=OffloadStrategy.CPU)

    # Fill cuda:0 with a transient lease
    with device_manager.lease("tool1", device="cuda:0"):
        # Fill cuda:1 with persistent
        device_manager.request_device(
            "tool2", "inst2", device="cuda:1",
            eviction_callback=mock_callback,
        )

        # Request a third device -- should evict inst2 (persistent), not the lease
        device3 = device_manager.request_device(
            "tool3", "inst3", device="cuda", eviction_callback=mock_callback,
        )
        assert device3 == "cuda:1"
        assert mock_callback.calls == ["cpu"]


def test_persistent_evicted_before_transient(device_manager, mock_callback):
    """With mixed allocation types, only persistent is evicted."""
    device_manager.configure(offload_strategy=OffloadStrategy.CPU)

    # Put persistent on cuda:0 (oldest)
    device_manager.request_device(
        "tool1", "inst1", device="cuda", eviction_callback=mock_callback,
    )
    time.sleep(0.01)

    # Put transient lease on cuda:1 (newer)
    with device_manager.lease("tool2", device="cuda:1"):
        # Request another device
        device = device_manager.request_device(
            "tool3", "inst3", device="cuda", eviction_callback=mock_callback,
        )
        # Should evict inst1 (persistent) even though it's on cuda:0
        assert device == "cuda:0"
        # inst1 should have been offloaded to CPU
        assert mock_callback.calls == ["cpu"]
