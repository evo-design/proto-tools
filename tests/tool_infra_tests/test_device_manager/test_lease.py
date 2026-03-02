"""Tests for lease-based transient device allocations."""

from __future__ import annotations

import threading
import time

import pytest

from bio_programming_tools.utils.device_manager import AllocationType, OffloadStrategy


# ============================================================================
# Lease Basics
# ============================================================================


class TestLeaseBasics:
    def test_lease_acquires_and_releases_device(self, device_manager):
        """Lease creates allocation during block and releases on exit."""
        with device_manager.lease("esmfold", device="cuda") as device:
            assert device.startswith("cuda:")
            # Allocation exists during lease
            status = device_manager.get_device_status()
            lease_allocs = [
                v for v in status["allocations"].values()
                if v["allocation_type"] == "transient"
            ]
            assert len(lease_allocs) == 1

        # Allocation released after exit
        status = device_manager.get_device_status()
        assert len(status["allocations"]) == 0

    def test_lease_returns_specific_cuda_device(self, device_manager):
        """Generic 'cuda' resolves to 'cuda:N'."""
        with device_manager.lease("esmfold", device="cuda") as device:
            assert device in ("cuda:0", "cuda:1")

    def test_lease_with_explicit_device(self, device_manager):
        """Explicit 'cuda:1' allocates cuda:1."""
        with device_manager.lease("esmfold", device="cuda:1") as device:
            assert device == "cuda:1"

    def test_lease_cpu_passthrough(self, device_manager):
        """CPU devices yield immediately with no allocation tracking."""
        with device_manager.lease("esmfold", device="cpu") as device:
            assert device == "cpu"
            # No allocation created
            status = device_manager.get_device_status()
            assert len(status["allocations"]) == 0

    def test_lease_allocation_is_transient(self, device_manager):
        """Lease allocations are marked TRANSIENT."""
        with device_manager.lease("esmfold", device="cuda"):
            # Find the lease allocation
            for alloc in device_manager._allocations.values():
                if alloc.allocation_type == AllocationType.TRANSIENT:
                    break
            else:
                pytest.fail("No TRANSIENT allocation found")

    def test_lease_cleanup_on_exception(self, device_manager):
        """Lease is released even when exception occurs in block."""
        with pytest.raises(ValueError, match="test error"):
            with device_manager.lease("esmfold", device="cuda") as device:
                assert device.startswith("cuda:")
                raise ValueError("test error")

        # Allocation released after exception
        status = device_manager.get_device_status()
        assert len(status["allocations"]) == 0

    def test_lease_no_gpus_raises(self, no_gpus_manager):
        """RuntimeError raised when no GPUs available."""
        with pytest.raises(RuntimeError, match="No GPUs available"):
            with no_gpus_manager.lease("esmfold", device="cuda"):
                pass


# ============================================================================
# Lease + Eviction Interaction
# ============================================================================


class TestLeaseEvictionInteraction:
    def test_lease_evicts_persistent_lru(self, device_manager, mock_callback):
        """Transient lease evicts persistent LRU when GPUs are full."""
        device_manager.configure(offload_strategy=OffloadStrategy.CPU)
        # Fill both GPUs with persistent allocations
        device_manager.request_device("tool1", "inst1", device="cuda", eviction_callback=mock_callback)
        time.sleep(0.01)
        device_manager.request_device("tool2", "inst2", device="cuda", eviction_callback=mock_callback)

        # Lease should evict inst1 (LRU)
        with device_manager.lease("esmfold", device="cuda") as device:
            assert device == "cuda:0"
            assert mock_callback.calls == ["cpu"]

    def test_lease_cannot_evict_transient(self, device_manager_1gpu):
        """Lease waits instead of evicting another transient lease."""
        dm = device_manager_1gpu
        acquired = threading.Event()
        released = threading.Event()
        second_device = []

        def hold_lease():
            with dm.lease("tool1", device="cuda"):
                acquired.set()
                released.wait(timeout=5)

        def wait_lease():
            acquired.wait(timeout=5)
            with dm.lease("tool2", device="cuda", timeout=5) as device:
                second_device.append(device)

        t1 = threading.Thread(target=hold_lease)
        t2 = threading.Thread(target=wait_lease)
        t1.start()
        t2.start()

        # Let t2 start waiting, then release t1
        acquired.wait(timeout=5)
        time.sleep(0.1)
        released.set()

        t1.join(timeout=5)
        t2.join(timeout=5)

        assert len(second_device) == 1
        assert second_device[0] == "cuda:0"

    def test_persistent_request_skips_transient(self, device_manager, mock_callback):
        """request_device evicts persistent, not transient allocations."""
        device_manager.configure(offload_strategy=OffloadStrategy.CPU)
        # Put a persistent on cuda:0
        device_manager.request_device("tool1", "inst1", device="cuda", eviction_callback=mock_callback)
        time.sleep(0.01)

        # Put a transient lease on cuda:1
        with device_manager.lease("tool2", device="cuda:1"):
            # Request a new persistent — should evict inst1 (persistent), not lease
            device = device_manager.request_device(
                "tool3", "inst3", device="cuda", eviction_callback=mock_callback
            )
            assert device == "cuda:0"
            assert mock_callback.calls == ["cpu"]

    def test_mixed_persistent_and_transient(self, device_manager, mock_callback):
        """In mixed scenario, only persistent allocations are evicted."""
        device_manager.configure(offload_strategy=OffloadStrategy.CPU)
        # Fill cuda:0 with persistent
        device_manager.request_device("tool1", "inst1", device="cuda", eviction_callback=mock_callback)
        time.sleep(0.01)

        # Fill cuda:1 with transient lease
        with device_manager.lease("tool2", device="cuda:1"):
            # New lease request — should evict inst1 (persistent), not the other lease
            with device_manager.lease("tool3", device="cuda") as device:
                assert device == "cuda:0"
                assert mock_callback.calls == ["cpu"]


# ============================================================================
# Lease Wait Mechanism
# ============================================================================


class TestLeaseWaitMechanism:
    def test_lease_waits_for_transient_release(self, device_manager_1gpu):
        """Thread waits for lease release, then acquires GPU."""
        dm = device_manager_1gpu
        acquired = threading.Event()
        second_device = []

        def hold_lease():
            with dm.lease("tool1", device="cuda"):
                acquired.set()
                # Hold for a short time
                time.sleep(0.3)

        def wait_lease():
            acquired.wait(timeout=5)
            with dm.lease("tool2", device="cuda", timeout=5) as device:
                second_device.append(device)

        t1 = threading.Thread(target=hold_lease)
        t2 = threading.Thread(target=wait_lease)
        t1.start()
        t2.start()

        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not t1.is_alive()
        assert not t2.is_alive()
        assert len(second_device) == 1
        assert second_device[0] == "cuda:0"

    def test_lease_timeout_raises(self, device_manager_1gpu):
        """TimeoutError when GPUs held beyond timeout."""
        dm = device_manager_1gpu
        acquired = threading.Event()
        error = []

        def hold_lease():
            with dm.lease("tool1", device="cuda"):
                acquired.set()
                time.sleep(2)  # Hold longer than timeout

        def timeout_lease():
            acquired.wait(timeout=5)
            try:
                with dm.lease("tool2", device="cuda", timeout=0.2):
                    pass
            except TimeoutError as e:
                error.append(e)

        t1 = threading.Thread(target=hold_lease)
        t2 = threading.Thread(target=timeout_lease)
        t1.start()
        t2.start()

        t2.join(timeout=5)
        t1.join(timeout=5)

        assert len(error) == 1
        assert "Timed out" in str(error[0])

    def test_lease_wakes_on_release(self, device_manager_1gpu):
        """notify_all wakes blocked waiters promptly."""
        dm = device_manager_1gpu
        acquired = threading.Event()
        wait_start = []
        wait_end = []

        def hold_lease():
            with dm.lease("tool1", device="cuda"):
                acquired.set()
                time.sleep(0.2)

        def wait_lease():
            acquired.wait(timeout=5)
            wait_start.append(time.monotonic())
            with dm.lease("tool2", device="cuda", timeout=5):
                wait_end.append(time.monotonic())

        t1 = threading.Thread(target=hold_lease)
        t2 = threading.Thread(target=wait_lease)
        t1.start()
        t2.start()

        t1.join(timeout=5)
        t2.join(timeout=5)

        assert len(wait_end) == 1
        # Should wake up promptly after release (well under timeout)
        wait_duration = wait_end[0] - wait_start[0]
        assert wait_duration < 2.0, f"Wait took {wait_duration:.2f}s, expected < 2s"


# ============================================================================
# Lease Thread Safety
# ============================================================================


class TestLeaseThreadSafety:
    def test_concurrent_leases_on_2_gpus(self, device_manager):
        """2 threads get 2 GPUs, 3rd waits and gets GPU after release."""
        devices = []
        barrier = threading.Barrier(2, timeout=5)
        all_acquired = threading.Event()

        def acquire_lease(name, hold_time):
            with device_manager.lease(name, device="cuda", timeout=10) as device:
                devices.append(device)
                barrier.wait()
                all_acquired.set()
                time.sleep(hold_time)

        third_device = []

        def third_lease():
            all_acquired.wait(timeout=5)
            time.sleep(0.05)  # Ensure both are held
            with device_manager.lease("tool3", device="cuda", timeout=10) as device:
                third_device.append(device)

        t1 = threading.Thread(target=acquire_lease, args=("tool1", 0.5))
        t2 = threading.Thread(target=acquire_lease, args=("tool2", 0.5))
        t3 = threading.Thread(target=third_lease)

        t1.start()
        t2.start()
        t3.start()

        t1.join(timeout=10)
        t2.join(timeout=10)
        t3.join(timeout=10)

        assert not t1.is_alive()
        assert not t2.is_alive()
        assert not t3.is_alive()

        # First two got different GPUs
        assert set(devices) == {"cuda:0", "cuda:1"}
        # Third got one of them after release
        assert len(third_device) == 1
        assert third_device[0] in ("cuda:0", "cuda:1")

    def test_concurrent_lease_and_persistent(self, device_manager, mock_callback):
        """No deadlock between lease and request_device."""
        # Put persistent on cuda:0
        device_manager.request_device("tool1", "inst1", device="cuda", eviction_callback=mock_callback)

        results = []

        def lease_thread():
            with device_manager.lease("tool2", device="cuda", timeout=5) as device:
                results.append(("lease", device))
                time.sleep(0.1)

        def persistent_thread():
            time.sleep(0.05)  # Start slightly after lease
            device = device_manager.request_device(
                "tool3", "inst3", device="cuda", eviction_callback=mock_callback
            )
            results.append(("persistent", device))

        t1 = threading.Thread(target=lease_thread)
        t2 = threading.Thread(target=persistent_thread)
        t1.start()
        t2.start()

        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not t1.is_alive()
        assert not t2.is_alive()
        assert len(results) == 2
