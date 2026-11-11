"""tests/tool_infra_tests/test_device_manager/test_auto_pin.py.

Which workers see only their leased GPU(s), and that every dispatch path honors it.

A tool's worker is pinned when the tool declares ``pin_visible_devices``, or when it is a GPU tool
on an Exclusive_Process host (``DeviceManager.pins_gpu_workers``). There an in-process move between
GPUs cannot release the old GPU, so every GPU worker respawns on a device change instead.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from proto_tools.utils.device_manager import DeviceManager
from proto_tools.utils.standalone_helpers_source.standalone_helpers.device import set_current_cuda_device
from proto_tools.utils.tool_instance import ToolInstance, _current_tool_invocation, _dispatch_pins_worker
from tests.tool_infra_tests.test_tool_instance import _make_fake_instance


@contextlib.contextmanager
def _host(exclusive: bool) -> Iterator[DeviceManager]:
    """A fresh DeviceManager on a host whose GPUs are (or are not) in Exclusive_Process mode."""
    with (
        patch("proto_tools.utils.device_manager.is_exclusive_process_mode", return_value=exclusive),
        patch("proto_tools.utils.device_manager.number_of_visible_gpus", return_value=2),
    ):
        DeviceManager.reset_instance()
        try:
            yield DeviceManager.get_instance()
        finally:
            DeviceManager.reset_instance()


@contextlib.contextmanager
def _invocation(**flags: bool) -> Iterator[None]:
    token = _current_tool_invocation.set({"key": "fake-tool", **flags})
    try:
        yield
    finally:
        _current_tool_invocation.reset(token)


# ── The decision ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("exclusive", "flags", "pinned"),
    [
        (False, {"uses_gpu": True}, False),  # normal host: GPU tools route and move in-process
        (True, {"uses_gpu": True}, True),  # Exclusive_Process host: every GPU tool is pinned
        (True, {"uses_gpu": False}, False),  # a CPU tool is never given a GPU-less view
        (False, {"uses_gpu": True, "pin_visible_devices": True}, True),  # a declared pin always holds
    ],
    ids=["normal", "exclusive", "cpu_tool", "declared"],
)
def test_dispatch_pins_worker(exclusive: bool, flags: dict, pinned: bool) -> None:
    with _host(exclusive), _invocation(**flags):
        assert _dispatch_pins_worker() is pinned


def test_tests_can_turn_auto_pin_off() -> None:
    """The internal knob lets a 2-GPU test exercise unpinned routing on an Exclusive_Process node."""
    with _host(exclusive=True) as dm, _invocation(uses_gpu=True):
        dm._auto_pin_exclusive = False
        assert not dm.pins_gpu_workers
        assert not _dispatch_pins_worker()


def test_no_invocation_keeps_the_instance_setting() -> None:
    """Direct ToolInstance.run calls carry no tool flags, so nothing is auto-pinned."""
    with _host(exclusive=True):
        assert not _dispatch_pins_worker()


def test_exclusive_mode_is_read_once() -> None:
    """nvidia-smi is queried when the DeviceManager is built, not on every dispatch."""
    with patch("proto_tools.utils.device_manager.is_exclusive_process_mode", return_value=True) as query:
        DeviceManager.reset_instance()
        try:
            dm = DeviceManager.get_instance()
            calls = query.call_count
            for _ in range(5):
                assert dm.pins_gpu_workers
            assert query.call_count == calls
        finally:
            DeviceManager.reset_instance()


# ── One-shot runs ─────────────────────────────────────────────────────────────


def _oneshot_env(pinned: bool, device: str) -> dict:
    """Run ``_run_oneshot`` up to its subprocess; return the env and the input the script would see."""
    inst = _make_fake_instance(device=device)
    captured: dict = {}

    def fake_build_env(dev, **kwargs):
        captured["device"], captured["kwargs"] = dev, kwargs
        return {}

    def fake_popen(cmd, **kwargs):
        captured["input"] = Path(cmd[-2]).read_text()
        raise RuntimeError("stop before running the tool")

    with (
        patch("proto_tools.utils.tool_instance._build_subprocess_env", side_effect=fake_build_env),
        patch("proto_tools.utils.tool_instance.subprocess.Popen", side_effect=fake_popen),
        contextlib.suppress(Exception),
    ):
        inst._run_oneshot({"device": device}, script_path=Path("/fake/inference.py"), pin_visible_devices=pinned)
    return captured


def test_pinned_oneshot_sees_only_its_gpu_as_local_cuda0() -> None:
    captured = _oneshot_env(pinned=True, device="cuda:1")
    assert captured["device"] == "cuda:1"  # the env is built from the physical lease
    assert captured["kwargs"]["pin_visible_devices"] is True
    assert '"device": "cuda:0"' in captured["input"]  # the script addresses it locally


def test_unpinned_oneshot_keeps_the_leased_index() -> None:
    captured = _oneshot_env(pinned=False, device="cuda:1")
    assert captured["kwargs"]["pin_visible_devices"] is False
    assert '"device": "cuda:1"' in captured["input"]


def test_oneshot_dispatch_passes_the_pin_decision() -> None:
    """A plain call (no persistent worker) is pinned exactly when a persistent one would be."""
    with _host(exclusive=True), _invocation(uses_gpu=True):
        with (
            patch.object(ToolInstance, "__init__", return_value=None),
            patch.object(ToolInstance, "_run_oneshot", return_value={"ok": True}) as run,
        ):
            ToolInstance.script_path = Path("/fake/inference.py")
            try:
                ToolInstance._oneshot("fake", {"device": "cpu"})
            finally:
                del ToolInstance.script_path
        assert run.call_args.kwargs["pin_visible_devices"] is True


# ── Persistent workers ────────────────────────────────────────────────────────


def test_live_worker_with_other_visibility_is_respawned() -> None:
    """A worker spawned unpinned is replaced once the tool must be pinned (and the reverse)."""
    inst = _make_fake_instance(device="cuda:1")
    old_worker = MagicMock()
    inst._worker = old_worker
    inst._worker_pinned = False
    new_worker = MagicMock()
    new_worker.send.return_value = {"ok": True}
    with (
        _host(exclusive=False) as dm,
        _invocation(pin_visible_devices=True, uses_gpu=True),
        patch("proto_tools.utils.tool_instance.PersistentWorker", return_value=new_worker) as spawn,
        patch.object(dm, "request_device", return_value="cuda:1"),
    ):
        inst._run_persistent({"device": "cuda:1"})
    old_worker.stop.assert_called_once()
    assert spawn.call_args.kwargs["pin_visible_devices"] is True
    assert inst._worker_pinned is True
    assert new_worker.send.call_args.args[0]["device"] == "cuda:0"


# ── ToolPool ──────────────────────────────────────────────────────────────────


def test_pool_partitions_see_the_tool_flags() -> None:
    """ToolPool runs partitions in copies of the dispatching context, so they carry the tool's flags."""
    from proto_tools.tools.testing.mock_pytorch_tool import (
        MockPyTorchToolConfig,
        MockPyTorchToolInput,
        run_mock_pytorch_tool,
    )
    from proto_tools.utils.tool_pool import ToolPool

    seen: dict = {}

    def capture(self, key, func, inputs, config):
        seen.update(_current_tool_invocation.get() or {})
        raise RuntimeError("stop before dispatching partitions")

    with patch.object(ToolPool, "_parallel_dispatch", capture), ToolPool(gpus=0, cpus=2):
        with pytest.raises(RuntimeError, match="stop before dispatching partitions"):
            run_mock_pytorch_tool(MockPyTorchToolInput(data_items=[[1.0], [2.0]]), MockPyTorchToolConfig())
    assert seen.get("key") == "mock-pytorch-tool-run"
    assert seen.get("uses_gpu") is True


# ── set_current_cuda_device ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("device", "expected"),
    [("cuda:1", 1), ("cuda:2,3", 2), ("cuda:0,cuda:1", 0), ("cpu", None), ("cuda", None), ("cuda:x", None)],
)
def test_set_current_cuda_device(device: str, expected: int | None) -> None:
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    with patch.dict("sys.modules", {"torch": fake_torch}):
        set_current_cuda_device(device)
    if expected is None:
        fake_torch.cuda.set_device.assert_not_called()
    else:
        fake_torch.cuda.set_device.assert_called_once_with(expected)
