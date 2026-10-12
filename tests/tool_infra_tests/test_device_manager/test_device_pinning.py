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
