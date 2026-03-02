"""Tests for preferred device allocation, re-allocation, and multi-model per device."""

from __future__ import annotations

from unittest.mock import patch


# ============================================================================
# Existing Allocation Compatibility Tests
# ============================================================================


class TestExistingAllocationCompatibility:
    """Verify that request_device returns existing allocations when compatible
    and re-allocates when not."""

    def test_cuda_allocation_stays_on_general_cuda_request(self, device_manager, mock_callback):
        """Tool on cuda:0 called with general 'cuda' should stay on cuda:0."""
        device = device_manager.request_device(
            "tool1", "inst1", device="cuda:0", eviction_callback=mock_callback
        )
        assert device == "cuda:0"

        device2 = device_manager.request_device(
            "tool1", "inst1", device="cuda", eviction_callback=mock_callback
        )
        assert device2 == "cuda:0", "Should keep existing cuda:0 allocation"
        assert len(device_manager._allocations) == 1

    def test_cuda_allocation_stays_on_same_specific_request(self, device_manager, mock_callback):
        """Tool on cuda:0 called with 'cuda:0' should stay on cuda:0."""
        device_manager.request_device(
            "tool1", "inst1", device="cuda:0", eviction_callback=mock_callback
        )
        device2 = device_manager.request_device(
            "tool1", "inst1", device="cuda:0", eviction_callback=mock_callback
        )
        assert device2 == "cuda:0", "Should keep existing allocation"
        assert len(device_manager._allocations) == 1

    def test_cpu_allocation_stays_on_cpu_request(self, device_manager, mock_callback):
        """Tool on cpu called with 'cpu' should stay on cpu."""
        device_manager.request_device(
            "tool1", "inst1", device="cpu", eviction_callback=mock_callback
        )
        device2 = device_manager.request_device(
            "tool1", "inst1", device="cpu", eviction_callback=mock_callback
        )
        assert device2 == "cpu", "Should keep existing CPU allocation"
        assert len(device_manager._allocations) == 1

    def test_cpu_allocation_re_promotes_on_cuda_request(self, device_manager, mock_callback):
        """Tool on cpu called with 'cuda' should be re-allocated to GPU."""
        device_manager.request_device(
            "tool1", "inst1", device="cpu", eviction_callback=mock_callback
        )
        device2 = device_manager.request_device(
            "tool1", "inst1", device="cuda", eviction_callback=mock_callback
        )
        assert device2.startswith("cuda:"), "Should re-allocate to a GPU"
        assert len(device_manager._allocations) == 1

    def test_cuda_allocation_moves_on_different_specific_request(self, device_manager, mock_callback):
        """Tool on cuda:0 called with 'cuda:1' should move to cuda:1."""
        device_manager.request_device(
            "tool1", "inst1", device="cuda:0", eviction_callback=mock_callback
        )
        device2 = device_manager.request_device(
            "tool1", "inst1", device="cuda:1", eviction_callback=mock_callback
        )
        assert device2 == "cuda:1", "Should re-allocate to cuda:1"
        assert len(device_manager._allocations) == 1

    def test_single_gpu_allocation_re_allocates_on_multi_gpu_request(self, device_manager, mock_callback):
        """Tool on cuda:0 called with 'cudax2' should re-allocate to 2 GPUs."""
        device_manager.request_device(
            "tool1", "inst1", device="cuda:0", eviction_callback=mock_callback
        )
        device2 = device_manager.request_device(
            "tool1", "inst1", device="cudax2", eviction_callback=mock_callback
        )
        assert "," in device2, f"Should get 2 GPUs, got {device2}"
        assert len(device_manager._allocations) == 1


# ============================================================================
# Preferred Device Tests
# ============================================================================


class TestPreferredDevice:
    def test_preferred_device_free(self, device_manager, mock_callback):
        """Test allocating preferred device when it's free."""
        device = device_manager.request_device(
            "tool1", "instance1", device="cuda:1", eviction_callback=mock_callback
        )
        assert device == "cuda:1", "Should get preferred device when free"

    def test_preferred_device_busy_with_warning(self, device_manager, mock_callback):
        """Test allocating preferred device when busy, with free alternatives."""
        # Occupy cuda:1
        device_manager.request_device(
            "tool1", "instance1", device="cuda:1", eviction_callback=mock_callback
        )

        # Request cuda:1 again - should evict instance1 and allocate to instance2
        with patch(
            "bio_programming_tools.utils.device_manager.logger.warning"
        ) as mock_warning:
            device = device_manager.request_device(
                "tool2",
                "instance2",
                device="cuda:1",
                eviction_callback=mock_callback,
            )

        assert device == "cuda:1", "Should evict and allocate preferred device"
        assert mock_warning.called, "Should warn when evicting with free alternatives"
        # instance1 should be evicted (either to CPU or removed depending on strategy)

    def test_preferred_device_all_busy_no_warning(self, device_manager, mock_callback):
        """Test preferred device when all devices are busy."""
        # Fill all devices
        device_manager.request_device("tool1", "instance1", device="cuda", eviction_callback=mock_callback)
        device_manager.request_device("tool2", "instance2", device="cuda", eviction_callback=mock_callback)

        # Request cuda:0 specifically - no warning since no alternatives
        with patch(
            "bio_programming_tools.utils.device_manager.logger.warning"
        ) as mock_warning:
            device = device_manager.request_device(
                "tool3",
                "instance3",
                device="cuda:0",
                eviction_callback=mock_callback,
            )

        assert device == "cuda:0", "Should get preferred device"
        # No warning about alternatives since there are none
        assert not any(
            "but" in str(call) and "are free" in str(call)
            for call in mock_warning.call_args_list
        ), "Should not warn when no free alternatives"


# ============================================================================
# Multi-model per Device Tests
# ============================================================================


class TestMultiModelPerDevice:
    def test_allow_multiple_per_device(self, device_manager, mock_callback):
        """Test multiple allocations on same device when enabled."""
        device_manager.configure(allow_multiple_per_device=True)

        device1 = device_manager.request_device(
            "tool1", "instance1", device="cuda", eviction_callback=mock_callback
        )
        device2 = device_manager.request_device(
            "tool2", "instance2", device="cuda", eviction_callback=mock_callback
        )

        device3 = device_manager.request_device(
            "tool3", "instance3", device="cuda", eviction_callback=mock_callback
        )

        # All should be allocated (no LRU eviction)
        assert len(device_manager._allocations) == 3, "All instances should be allocated"
        # First two get different devices, third routes to least-loaded GPU
        assert device1 == "cuda:0", "First instance should get cuda:0"
        assert device2 == "cuda:1", "Second instance should get cuda:1"
        assert device3 in ("cuda:0", "cuda:1"), "Third instance should share a GPU"

    def test_routes_to_least_allocated_gpu(self, device_manager, mock_callback):
        """With allow_multiple_per_device, new allocations go to GPU with fewest allocations."""
        device_manager.configure(allow_multiple_per_device=True)

        # Put 2 allocations on cuda:0, 1 on cuda:1
        device_manager.request_device("tool1", "inst1", device="cuda:0", eviction_callback=mock_callback)
        device_manager.request_device("tool2", "inst2", device="cuda:0", eviction_callback=mock_callback)
        device_manager.request_device("tool3", "inst3", device="cuda:1", eviction_callback=mock_callback)

        # Next allocation should go to cuda:1 (fewer allocations)
        device4 = device_manager.request_device(
            "tool4", "inst4", device="cuda", eviction_callback=mock_callback
        )

        assert device4 == "cuda:1", \
            f"Should route to GPU with fewest allocations (cuda:1), got {device4}"
