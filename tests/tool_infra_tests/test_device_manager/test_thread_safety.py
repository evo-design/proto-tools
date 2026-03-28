"""tests/tool_infra_tests/test_device_manager/test_thread_safety.py

Tests for thread safety of DeviceManager."""

import threading

from bio_programming_tools.utils.device_manager import OffloadStrategy


# ── Thread safety ───────────────────────────────────────────────────────

def test_concurrent_requests(device_manager, mock_callback):
    """Test thread-safe concurrent device requests."""
    device_manager.configure(offload_strategy=OffloadStrategy.CPU)
    results = {}

    def request_device(tool_name, instance_name):
        device = device_manager.request_device(
            tool_name, instance_name, device="cuda", eviction_callback=mock_callback
        )
        results[instance_name] = device

    threads = [
        threading.Thread(target=request_device, args=(f"tool{i}", f"instance{i}"))
        for i in range(5)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 5, "All concurrent requests should complete"
    assert len(device_manager._allocations) == 5, "All instances should be allocated"
