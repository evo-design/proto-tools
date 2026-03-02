"""Tests for multi-GPU allocation."""

from __future__ import annotations

import os
import time
from unittest.mock import patch

import pytest

from bio_programming_tools.utils.device_manager import DeviceManager, OffloadStrategy


# ============================================================================
# Multi-GPU Allocation Tests
# ============================================================================


class TestMultiGPUAllocation:

    def test_allocate_2_gpus_auto(self, device_manager, mock_callback):
        """Test auto-allocating 2 GPUs (cudax2)."""
        # Request 2 GPUs
        devices = device_manager.request_device(
            "tool1", "instance1", device="cudax2", eviction_callback=mock_callback
        )

        assert devices == "cuda:0,cuda:1", "Should allocate both GPUs"
        assert "instance1" in device_manager._allocations, "Instance should be tracked"
        assert device_manager._allocations["instance1"].device_ids == [
            "cuda:0",
            "cuda:1",
        ], "Should store both device IDs"

    def test_allocate_3_gpus_auto(self, mock_callback):
        """Test auto-allocating 3 GPUs (cudax3) with mocked 3 GPUs."""
        DeviceManager.reset_instance()

        with patch(
            "bio_programming_tools.utils.device_manager.number_of_visible_gpus",
            return_value=3,
        ):
            device_manager = DeviceManager.get_instance()

            # Request 3 GPUs
            devices = device_manager.request_device(
                "tool1", "instance1", device="cudax3", eviction_callback=mock_callback
            )

            assert devices == "cuda:0,cuda:1,cuda:2"
            assert device_manager._allocations["instance1"].device_ids == [
                "cuda:0",
                "cuda:1",
                "cuda:2",
            ]

        DeviceManager.reset_instance()

    def test_allocate_2_gpus_explicit_shorthand(self):
        """Test explicit 2-GPU allocation with shorthand (cuda:0,2) with 3 visible GPUs."""
        # Reset and configure with 3 GPUs (physical 4,5,6)
        DeviceManager.reset_instance()

        with patch(
            "bio_programming_tools.utils.device_manager.number_of_visible_gpus",
            return_value=3,
        ):
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,5,6"}):
                device_manager = DeviceManager.get_instance()
                mock_callback = lambda action: None

                # Request specific GPUs using shorthand (skip cuda:1)
                devices = device_manager.request_device(
                    "tool1",
                    "instance1",
                    device="cuda:0,2",
                    eviction_callback=mock_callback,
                )

                assert devices == "cuda:0,cuda:2", "Should allocate cuda:0 and cuda:2"
                assert device_manager._allocations["instance1"].device_ids == [
                    "cuda:0",
                    "cuda:2",
                ], "Should store both device IDs"

        DeviceManager.reset_instance()

    def test_allocate_2_gpus_explicit_verbose(self, device_manager, mock_callback):
        """Test explicit 2-GPU allocation with verbose syntax (cuda:0,cuda:1)."""
        # Request specific GPUs using verbose syntax
        devices = device_manager.request_device(
            "tool1",
            "instance1",
            device="cuda:0,cuda:1",
            eviction_callback=mock_callback,
        )

        assert devices == "cuda:0,cuda:1"
        assert device_manager._allocations["instance1"].device_ids == ["cuda:0", "cuda:1"]

    def test_multi_gpu_lru_eviction(self, device_manager, mock_callback):
        """Test LRU eviction with multi-GPU allocations."""
        device_manager.configure(offload_strategy=OffloadStrategy.CPU)

        # Allocate 2 tools with 2 GPUs each (fills all GPUs if only 2 available)
        device_manager.request_device(
            "tool1", "instance1", device="cudax2", eviction_callback=mock_callback
        )
        time.sleep(0.01)  # Ensure different timestamps
        device_manager.request_device(
            "tool2", "instance2", device="cuda", eviction_callback=mock_callback
        )

        # Request another GPU - should evict tool1 (LRU)
        device_manager.request_device(
            "tool3", "instance3", device="cuda", eviction_callback=mock_callback
        )

        # instance1 should be moved to CPU (evicted)
        assert device_manager._allocations["instance1"].device_ids == [
            "cpu"
        ], "instance1 should be evicted to CPU"
        # instance2 should be on cuda:0
        assert device_manager._allocations["instance2"].device_ids == [
            "cuda:0"
        ], "instance2 should be on cuda:0"
        # instance3 should be on cuda:1
        assert device_manager._allocations["instance3"].device_ids == [
            "cuda:1"
        ], "instance3 should be on cuda:1"

    def test_multi_gpu_request_evicts_two_single_instances(
        self, device_manager, mock_callback
    ):
        """Test cudax2 request evicting two single-GPU instances."""
        device_manager.configure(offload_strategy=OffloadStrategy.CPU)

        # Fill both GPUs with single-GPU instances
        device_manager.request_device(
            "tool1", "instance1", device="cuda", eviction_callback=mock_callback
        )
        time.sleep(0.01)  # Ensure different timestamps
        device_manager.request_device(
            "tool2", "instance2", device="cuda", eviction_callback=mock_callback
        )

        # Verify both GPUs are filled
        assert device_manager._allocations["instance1"].device_ids == [
            "cuda:0"
        ], "instance1 should be on cuda:0"
        assert device_manager._allocations["instance2"].device_ids == [
            "cuda:1"
        ], "instance2 should be on cuda:1"

        # Request 2 GPUs - should evict both single-GPU instances
        devices = device_manager.request_device(
            "tool3", "instance3", device="cudax2", eviction_callback=mock_callback
        )

        # Both instance1 and instance2 should be evicted to CPU
        assert device_manager._allocations["instance1"].device_ids == [
            "cpu"
        ], "instance1 should be evicted to CPU"
        assert device_manager._allocations["instance2"].device_ids == [
            "cpu"
        ], "instance2 should be evicted to CPU"

        # instance3 should get both GPUs
        assert device_manager._allocations["instance3"].device_ids == [
            "cuda:0",
            "cuda:1",
        ], "instance3 should get cuda:0 and cuda:1"
        assert devices == "cuda:0,cuda:1", "Should allocate both GPUs"

    def test_multi_gpu_with_allow_multiple(self, device_manager, mock_callback):
        """Test multi-GPU allocation with allow_multiple_per_device=True."""
        device_manager.configure(allow_multiple_per_device=True)

        # Allocate 2 GPUs for tool1
        device_manager.request_device(
            "tool1", "instance1", device="cudax2", eviction_callback=mock_callback
        )

        # Should still be able to allocate more tools (sharing devices)
        # Since allow_multiple=True, eviction only happens when truly needed
        device_manager.request_device(
            "tool2", "instance2", device="cuda", eviction_callback=mock_callback
        )

        # Both allocations should exist
        assert "instance1" in device_manager._allocations
        assert "instance2" in device_manager._allocations

    def test_multi_gpu_move(self):
        """Test moving a multi-GPU allocation to different devices with mocked 4 GPUs."""
        DeviceManager.reset_instance()

        with patch(
            "bio_programming_tools.utils.device_manager.number_of_visible_gpus",
            return_value=4,
        ):
            device_manager = DeviceManager.get_instance()
            cb = lambda action: None

            # Allocate 2 GPUs
            device_manager.request_device(
                "tool1", "instance1", device="cuda:0,1", eviction_callback=cb
            )

            # Move to different 2 GPUs
            move_cb = lambda dev: {"success": True}
            device_manager.move_to_device("instance1", "cuda:2,3", move_cb)

            # Should now be on cuda:2,3
            assert device_manager._allocations["instance1"].device_ids == ["cuda:2", "cuda:3"]

        DeviceManager.reset_instance()

    def test_multi_gpu_release(self, device_manager, mock_callback):
        """Test releasing a multi-GPU allocation frees all devices."""
        # Allocate 2 GPUs
        device_manager.request_device(
            "tool1", "instance1", device="cudax2", eviction_callback=mock_callback
        )

        # Verify allocated
        assert device_manager._allocations["instance1"].device_ids == ["cuda:0", "cuda:1"]

        # Release
        device_manager.release_device("instance1")

        # Should be freed
        assert "instance1" not in device_manager._allocations

        # Both devices should now be available
        devices = device_manager.request_device(
            "tool2", "instance2", device="cudax2", eviction_callback=mock_callback
        )
        assert devices == "cuda:0,cuda:1"

    def test_mixed_single_and_multi_gpu(self):
        """Test mixing single-GPU and multi-GPU allocations with mocked 3 GPUs."""
        DeviceManager.reset_instance()

        with patch(
            "bio_programming_tools.utils.device_manager.number_of_visible_gpus",
            return_value=3,
        ):
            device_manager = DeviceManager.get_instance()
            cb = lambda action: None

            # Allocate 1 GPU for tool1
            result1 = device_manager.request_device(
                "tool1", "instance1", device="cuda", eviction_callback=cb
            )
            assert result1 == "cuda:0"

            # Allocate 2 GPUs for tool2
            result2 = device_manager.request_device(
                "tool2", "instance2", device="cudax2", eviction_callback=cb
            )
            assert result2 == "cuda:1,cuda:2"

            # Both should coexist
            assert device_manager._allocations["instance1"].device_ids == ["cuda:0"]
            assert device_manager._allocations["instance2"].device_ids == ["cuda:1", "cuda:2"]

        DeviceManager.reset_instance()

    def test_insufficient_gpus_for_multi_gpu_request(self, device_manager, mock_callback):
        """Test requesting more GPUs than available raises error."""
        # Request more GPUs than exist in system (mock has 2 GPUs)
        with pytest.raises(RuntimeError, match="Cannot allocate"):
            device_manager.request_device(
                "tool1",
                "instance1",
                device="cudax7",
                eviction_callback=mock_callback,
            )

    def test_multi_gpu_device_status(self, device_manager, mock_callback):
        """Test get_device_status() correctly reports multi-GPU allocations."""
        # Allocate 2 GPUs
        device_manager.request_device(
            "tool1", "instance1", device="cudax2", eviction_callback=mock_callback
        )

        # Get status
        status = device_manager.get_device_status()

        # Should show instance1 with both devices
        assert "instance1" in status["allocations"]
        alloc_info = status["allocations"]["instance1"]
        assert alloc_info["tool_name"] == "tool1"

        # device_id should be comma-joined for multi-GPU
        assert "," in alloc_info["device_id"] or len(alloc_info["device_id"].split(":")) == 2

    def test_lru_eviction_multi_gpu_request(self):
        """Test LRU eviction when requesting 2 GPUs with 3 single-GPU allocations (mocked 3 GPUs)."""
        DeviceManager.reset_instance()

        with patch(
            "bio_programming_tools.utils.device_manager.number_of_visible_gpus",
            return_value=3,
        ):
            device_manager = DeviceManager.get_instance()
            device_manager.configure(offload_strategy=OffloadStrategy.CPU)

            # Create three separate callbacks to track evictions
            calls1, calls2, calls3 = [], [], []
            cb1 = lambda action: calls1.append(action)
            cb2 = lambda action: calls2.append(action)
            cb3 = lambda action: calls3.append(action)

            # Allocate 3 GPUs (one instance on each)
            device_manager.request_device(
                "tool1", "instance1", device="cuda:0", eviction_callback=cb1
            )
            time.sleep(0.01)  # Ensure different timestamps
            device_manager.request_device(
                "tool2", "instance2", device="cuda:1", eviction_callback=cb2
            )
            time.sleep(0.01)
            device_manager.request_device(
                "tool3", "instance3", device="cuda:2", eviction_callback=cb3
            )

            # Verify all 3 are allocated
            assert device_manager._allocations["instance1"].device_ids == [
                "cuda:0"
            ], "instance1 should be on cuda:0"
            assert device_manager._allocations["instance2"].device_ids == [
                "cuda:1"
            ], "instance2 should be on cuda:1"
            assert device_manager._allocations["instance3"].device_ids == [
                "cuda:2"
            ], "instance3 should be on cuda:2"

            # Request 2 GPUs - should evict the 2 LRU instances (instance1 and instance2)
            cb4 = lambda action: None
            device_manager.request_device(
                "tool4", "instance4", device="cudax2", eviction_callback=cb4
            )

            # instance1 and instance2 should be evicted to CPU (they were allocated first)
            assert device_manager._allocations["instance1"].device_ids == [
                "cpu"
            ], "instance1 should be evicted to CPU"
            assert device_manager._allocations["instance2"].device_ids == [
                "cpu"
            ], "instance2 should be evicted to CPU"

            # instance3 should still be on cuda:2
            assert device_manager._allocations["instance3"].device_ids == [
                "cuda:2"
            ], "instance3 should still be on cuda:2"

            # instance4 should get cuda:0 and cuda:1
            assert device_manager._allocations["instance4"].device_ids == [
                "cuda:0",
                "cuda:1",
            ], "instance4 should get cuda:0 and cuda:1"

            # Callbacks for instance1 and instance2 should be called
            assert len(calls1) == 1, "instance1 callback should be called"
            assert len(calls2) == 1, "instance2 callback should be called"
            assert len(calls3) == 0, "instance3 callback should not be called"

        DeviceManager.reset_instance()
