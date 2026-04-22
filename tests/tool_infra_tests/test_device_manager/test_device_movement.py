"""tests/tool_infra_tests/test_device_manager/test_device_movement.py.

Tests for moving tool instances between devices.
"""

import time
from unittest.mock import MagicMock, patch

from proto_tools.utils.device_manager import OffloadStrategy
from proto_tools.utils.tool_instance import ToolInstance

# ── Move to device tests ──────────────────────────────────────────────────


def test_move_to_device(device_manager, mock_callback):
    """Test moving an instance to a different device."""
    device_manager.request_device("tool1", "instance1", device="cuda", eviction_callback=mock_callback)
    assert device_manager._allocations["instance1"].device_ids[0] == "cuda:0", "Should start on cuda:0"

    callback = MagicMock()
    device_manager.move_to_device("instance1", "cuda:1", worker_callback=callback)

    assert device_manager._allocations["instance1"].device_ids[0] == "cuda:1", "Should move to cuda:1"
    callback.assert_called_once_with("cuda:1")


def test_move_to_same_device(device_manager, mock_callback):
    """Test moving to same device (should be no-op)."""
    device_manager.request_device("tool1", "instance1", device="cuda", eviction_callback=mock_callback)

    callback = MagicMock()
    device_manager.move_to_device("instance1", "cuda:0", worker_callback=callback)

    # Should skip move
    callback.assert_not_called()


def test_move_nonexistent_instance(device_manager, mock_callback):
    """Test moving non-existent instance (should log warning)."""
    with patch("proto_tools.utils.device_manager.logger.warning") as mock_warning:
        device_manager.move_to_device("nonexistent", "cuda:1")

    assert mock_warning.called, "Should warn when moving nonexistent instance"


def test_move_to_generic_cuda(device_manager, mock_callback):
    """Test moving to generic 'cuda' resolves to specific device."""
    device_manager.request_device("tool1", "instance1", device="cuda:0", eviction_callback=mock_callback)
    assert device_manager._allocations["instance1"].device_ids[0] == "cuda:0", "Should start on cuda:0"

    callback = MagicMock()
    device_manager.move_to_device("instance1", "cuda", worker_callback=callback)

    # Should resolve "cuda" to a specific device (cuda:1 since cuda:0 is occupied by self)
    alloc = device_manager._allocations["instance1"]
    assert alloc.device_ids[0] in [
        "cuda:0",
        "cuda:1",
    ], "Should resolve to specific cuda device"
    callback.assert_called_once()


def test_move_to_generic_cuda_triggers_lru_eviction(device_manager, mock_callback):
    """Test moving to generic 'cuda' triggers LRU eviction when all GPUs occupied."""
    # Use CPU strategy so evicted allocations remain trackable
    device_manager.configure(managed_devices=["cuda:0"], offload_strategy=OffloadStrategy.CPU)

    # Occupy the single GPU
    device_manager.request_device("tool1", "instance1", device="cuda", eviction_callback=mock_callback)  # cuda:0
    time.sleep(0.01)

    # Allocate instance2 - should evict instance1 to CPU
    device_manager.request_device(
        "tool2", "instance2", device="cuda", eviction_callback=mock_callback
    )  # cuda:0 (evicts instance1)

    assert device_manager._allocations["instance2"].device_ids[0] == "cuda:0", "Instance2 should get cuda:0"
    assert device_manager._allocations["instance1"].device_ids[0] == "cpu", "Instance1 should be on CPU"

    time.sleep(0.01)

    # Now move instance1 back to generic "cuda"
    # Should evict instance2 (LRU on GPU) and take cuda:0
    callback = MagicMock()
    device_manager.move_to_device("instance1", "cuda", worker_callback=callback)

    # instance1 should be on GPU (evicted instance2)
    assert device_manager._allocations["instance1"].device_ids[0] == "cuda:0", "Instance1 should move back to cuda:0"
    # instance2 should be evicted to CPU (CPU strategy)
    assert device_manager._allocations["instance2"].device_ids[0] == "cpu", "Instance2 should be evicted to CPU"


def test_move_to_generic_cuda_excludes_self_from_eviction(device_manager, mock_callback):
    """Test that moving to generic 'cuda' doesn't evict itself."""
    # Occupy both GPUs
    device_manager.request_device("tool1", "instance1", device="cuda", eviction_callback=mock_callback)  # cuda:0
    time.sleep(0.01)
    device_manager.request_device("tool2", "instance2", device="cuda", eviction_callback=mock_callback)  # cuda:1

    # Move instance1 to generic "cuda" (it's already on cuda:0)
    # Should NOT evict itself, should evict instance2 instead
    callback = MagicMock()
    device_manager.move_to_device("instance1", "cuda", worker_callback=callback)

    # instance1 should still be on a GPU (might be same or different)
    assert device_manager._allocations["instance1"].device_ids[0].startswith("cuda"), "Instance should stay on GPU"


# ── Device mismatch detection in _run_persistent ──────────────────────────


def _make_instance(device_manager, toolkit="mock_pytorch_tool", instance_name="inst"):
    """Create a ToolInstance with a mock worker (no real subprocess).

    Also stubs _ensure_env to skip the real micromamba/PyTorch install;
    these tests only exercise device-movement logic.
    """
    ToolInstance.clear_all()
    inst = ToolInstance.get(toolkit, instance_name=instance_name)
    # Mock the worker so _run_persistent takes the `else` branch
    mock_worker = MagicMock()
    mock_worker.script_path = inst.script_path
    mock_worker.send.return_value = {"success": True, "result": []}
    inst._worker = mock_worker
    inst._reload_params = {}
    inst._ensure_env = lambda: None
    return inst, mock_worker


def test_cpu_to_generic_cuda(device_manager, mock_callback):
    """Tool on CPU + config says 'cuda' -> should move to GPU."""
    inst, _mock_worker = _make_instance(device_manager)

    device_manager.request_device("mock_pytorch_tool", "inst", device="cpu", eviction_callback=mock_callback)
    inst.device = "cpu"

    # Call with config requesting "cuda"
    input_dict = {"device": "cuda"}
    with patch.object(inst, "_to", wraps=inst._to) as mock_to:
        inst.run(input_dict, reload_on=set())

    mock_to.assert_called_once_with("cuda")
    assert inst.device.startswith("cuda"), f"Should be on GPU, got {inst.device}"


def test_cpu_to_specific_cuda(device_manager, mock_callback):
    """Tool on CPU + config says 'cuda:1' -> should move to cuda:1."""
    inst, _mock_worker = _make_instance(device_manager)

    input_dict = {"device": "cuda:1"}
    with patch.object(inst, "_to", wraps=inst._to) as mock_to:
        inst.run(input_dict, reload_on=set())

    mock_to.assert_called_once_with("cuda:1")


def test_wrong_gpu_to_specific_cuda(device_manager, mock_callback):
    """Tool on cuda:0 + config says 'cuda:1' -> should move to cuda:1."""
    inst, _mock_worker = _make_instance(device_manager)

    device_manager.request_device("mock_pytorch_tool", "inst", device="cuda:0", eviction_callback=mock_callback)
    inst.device = "cuda:0"

    input_dict = {"device": "cuda:1"}
    with patch.object(inst, "_to", wraps=inst._to) as mock_to:
        inst.run(input_dict, reload_on=set())

    mock_to.assert_called_once_with("cuda:1")


def test_already_on_correct_gpu_no_move(device_manager, mock_callback):
    """Tool on cuda:0 + config says 'cuda:0' -> no move needed."""
    inst, _mock_worker = _make_instance(device_manager)

    device_manager.request_device("mock_pytorch_tool", "inst", device="cuda:0", eviction_callback=mock_callback)
    inst.device = "cuda:0"

    input_dict = {"device": "cuda:0"}
    with patch.object(inst, "_to", wraps=inst._to) as mock_to:
        inst.run(input_dict, reload_on=set())

    mock_to.assert_not_called()


def test_generic_cuda_already_on_gpu_no_move(device_manager, mock_callback):
    """Tool on cuda:0 + config says 'cuda' -> no move (already on a GPU)."""
    inst, _mock_worker = _make_instance(device_manager)

    device_manager.request_device("mock_pytorch_tool", "inst", device="cuda:0", eviction_callback=mock_callback)
    inst.device = "cuda:0"

    input_dict = {"device": "cuda"}
    with patch.object(inst, "_to", wraps=inst._to) as mock_to:
        inst.run(input_dict, reload_on=set())

    mock_to.assert_not_called()


def test_gpu_to_cpu(device_manager, mock_callback):
    """Tool on cuda:0 + config says 'cpu' -> should move to CPU."""
    inst, _mock_worker = _make_instance(device_manager)

    device_manager.request_device("mock_pytorch_tool", "inst", device="cuda:0", eviction_callback=mock_callback)
    inst.device = "cuda:0"

    input_dict = {"device": "cpu"}
    with patch.object(inst, "_to", wraps=inst._to) as mock_to:
        inst.run(input_dict, reload_on=set())

    mock_to.assert_called_once_with("cpu")


def test_device_override_after_move(device_manager, mock_callback):
    """After move, input_dict['device'] should reflect the actual device."""
    inst, mock_worker = _make_instance(device_manager)

    device_manager.request_device("mock_pytorch_tool", "inst", device="cpu", eviction_callback=mock_callback)
    inst.device = "cpu"

    input_dict = {"device": "cuda"}
    inst.run(input_dict, reload_on=set())

    # The worker.send should have been called with the resolved device
    sent_input = mock_worker.send.call_args[0][0]
    assert sent_input["device"].startswith("cuda"), (
        f"Worker should receive resolved cuda device, got {sent_input['device']}"
    )


# ── cudax auto-allocate resolution ──────────────────────────────────────────


def test_move_to_cudax_resolves_to_specific_devices(device_manager, mock_callback):
    """move_to_device('cudax2') should resolve to specific device indices, not pass 'cudax2' through."""
    # Start with a multi-GPU allocation (the normal path via request_device)
    device_manager.request_device("tool1", "instance1", device="cudax2", eviction_callback=mock_callback)
    assert device_manager._allocations["instance1"].device_ids == ["cuda:0", "cuda:1"]

    # Simulate a subsequent move_to_device with the same cudax2 spec.
    # This happens in persistent mode when a second dispatch sends the
    # default device string again.
    callback = MagicMock()
    result = device_manager.move_to_device("instance1", "cudax2", worker_callback=callback)

    alloc = device_manager._allocations["instance1"]
    assert all(d.startswith("cuda:") for d in alloc.device_ids), (
        f"All devices should be specific indices, got {alloc.device_ids}"
    )
    assert "cudax" not in result, f"Should return resolved devices, got {result!r}"
