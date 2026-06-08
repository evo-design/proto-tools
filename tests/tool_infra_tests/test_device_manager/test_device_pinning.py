"""Tests for pin_visible_devices worker pinning."""

import time

import pytest

from proto_tools.utils.device_manager import DeviceManager, OffloadStrategy
from proto_tools.utils.tool_instance import ToolInstance, _device_change_needed, _local_visible_device

# ── Device-string comparison / local mapping ────────────────────────────────


def test_device_change_needed():
    """A move is needed only when the requested device differs; generic 'cuda' and '' don't."""
    assert _device_change_needed("cuda:0", "cuda:1")
    assert _device_change_needed("cuda:0", "cpu")
    assert _device_change_needed("cpu", "cuda:0")
    assert not _device_change_needed("cuda:1", "cuda:1")
    assert not _device_change_needed("cuda:0", "cuda")  # generic cuda satisfied by any GPU
    assert not _device_change_needed("cuda:0", "")  # no preference


def test_device_change_needed_equivalent_forms():
    """Equivalent device strings don't force a (respawn-triggering) move; differing arities do."""
    # cuda:0,1 shorthand == cuda:0,cuda:1 (canonical allocation form), via parse_device_string
    assert not _device_change_needed("cuda:0,cuda:1", "cuda:0,1")
    # auto multi "cudaxN" satisfied by any N-GPU allocation, but arity must match
    assert not _device_change_needed("cuda:0,cuda:1", "cudax2")
    assert _device_change_needed("cuda:0,cuda:1", "cudax3")
    assert _device_change_needed("cuda:0,cuda:1", "cuda:0")  # 2 GPUs -> 1 GPU is a real change


def test_local_visible_device():
    """A pinned worker addresses its N assigned GPUs as local cuda:0..N-1 positionally."""
    assert _local_visible_device("cuda:3") == "cuda:0"
    assert _local_visible_device("cuda") == "cuda:0"
    assert _local_visible_device("cuda:5,cuda:7") == "cuda:0,cuda:1"
    assert _local_visible_device("cpu") == "cpu"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def _gpu_mb(dm, device):
    return dm.get_gpu_memory_used(device) / (1024 * 1024)


def _fresh_dm(devices):
    DeviceManager.reset_instance()
    ToolInstance.clear_all()
    dm = DeviceManager.get_instance()
    dm.configure(managed_devices=devices, offload_strategy=OffloadStrategy.CPU)
    return dm


def _teardown():
    ToolInstance.clear_all()
    DeviceManager.reset_instance()
    time.sleep(2)  # let GPU memory reclaim before the next test


@pytest.mark.uses_gpu(2)
@pytest.mark.slow
def test_pinned_worker_does_not_touch_other_gpus():
    """A pinned JAX worker on cuda:1 lands there and leaves cuda:0 untouched.

    Covers both the no-cross-GPU-context fix and that the user-facing
    ``device="cuda:1"`` selection maps to the right physical GPU.
    """
    from proto_tools.tools.testing.mock_jax_tool import MockJAXToolConfig, MockJAXToolInput, run_mock_jax_tool

    dm = _fresh_dm(["cuda:0", "cuda:1"])
    try:
        base0 = _gpu_mb(dm, "cuda:0")
        with ToolInstance.persist_tool("mock_jax_tool", instance_name="pinned"):
            result = run_mock_jax_tool(
                MockJAXToolInput(), MockJAXToolConfig(memory_mb=4096, device="cuda:1"), instance="pinned"
            )
            assert result.success, result.errors
            assert _gpu_mb(dm, "cuda:1") > 3072, "model should be resident on the requested GPU"
            assert _gpu_mb(dm, "cuda:0") - base0 < 512, "pinned worker must not initialize a context on cuda:0"
    finally:
        _teardown()


@pytest.mark.uses_gpu(2)
@pytest.mark.slow
def test_pinned_respawns_on_move_while_pytorch_moves_in_process():
    """Pinned workers respawn on a device change (freeing the source); non-pinned don't."""
    from proto_tools.tools.testing.mock_jax_tool import MockJAXToolConfig, MockJAXToolInput, run_mock_jax_tool
    from proto_tools.tools.testing.mock_pytorch_tool import (
        MockPyTorchToolConfig,
        MockPyTorchToolInput,
        run_mock_pytorch_tool,
    )

    # Pinned (JAX): physical-device change ⇒ new worker process, source GPU freed.
    dm = _fresh_dm(["cuda:0", "cuda:1"])
    try:
        inst = ToolInstance.get("mock_jax_tool", instance_name="pin")
        with ToolInstance.persist_tool("mock_jax_tool", instance_name="pin"):
            run_mock_jax_tool(MockJAXToolInput(), MockJAXToolConfig(memory_mb=4096, device="cuda:0"), instance="pin")
            pid_before = inst._worker._process.pid
            run_mock_jax_tool(MockJAXToolInput(), MockJAXToolConfig(memory_mb=4096, device="cuda:1"), instance="pin")
            pid_after = inst._worker._process.pid
            assert pid_after != pid_before, "pinned worker should respawn on a device change"
            assert _gpu_mb(dm, "cuda:0") < 512, "source GPU should be freed after respawn"
    finally:
        _teardown()

    # Non-pinned (PyTorch): same worker process moves the model in-process.
    dm = _fresh_dm(["cuda:0", "cuda:1"])
    try:
        inst = ToolInstance.get("mock_pytorch_tool", instance_name="mov")
        with ToolInstance.persist_tool("mock_pytorch_tool", instance_name="mov"):
            run_mock_pytorch_tool(
                MockPyTorchToolInput(), MockPyTorchToolConfig(memory_mb=4096, device="cuda:0"), instance="mov"
            )
            pid_before = inst._worker._process.pid
            run_mock_pytorch_tool(
                MockPyTorchToolInput(), MockPyTorchToolConfig(memory_mb=4096, device="cuda:1"), instance="mov"
            )
            pid_after = inst._worker._process.pid
            assert pid_after == pid_before, "non-pinned worker should move in-process, not respawn"
    finally:
        _teardown()
