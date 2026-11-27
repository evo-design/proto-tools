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


# ── Moving a model between GPUs in-process ────────────────────────────────────

_MOVE_CHECK = """
import json, torch
from standalone_helpers import move_model_to_device

class Cached(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        self.owner = torch.nn.LayerNorm(4)
        self.norm_weight = self.owner.weight  # a parameter used here after its module leaves
        with torch.inference_mode():
            self.t = torch.arange(8)  # a cached inference tensor, as stripedhyena keeps
        self.host_table = torch.arange(4)  # kept on the CPU on purpose

def devices(model):
    return {
        "lin": str(model.lin.weight.device), "cached": str(model.t.device),
        "param_ref": str(model.norm_weight.device), "host": str(model.host_table.device),
    }

model = Cached().to("cuda:1")
model.t = model.t.to("cuda:1")
ref = model.norm_weight
del model.owner  # norm_weight is now an attribute only, which .to() does not move
steps = {}
for old, new in [("cuda:1", "cuda:0"), ("cuda:0", "cpu"), ("cpu", "cuda:1")]:
    model = move_model_to_device(model, old, new)
    steps[new] = devices(model)
print(json.dumps({**steps, "same_param": ref is model.norm_weight, "current": torch.cuda.current_device()}))
"""


@pytest.mark.uses_env
@pytest.mark.uses_gpu(2)
def test_move_model_to_device_takes_cached_tensors_along() -> None:
    """``.to()`` skips tensors held as plain attributes; the move helper must not leave them behind.

    Covers a move between GPUs and an offload round trip (GPU, CPU, then another GPU): cached
    tensors follow the model back from the CPU, and a tensor the model keeps on the CPU stays there.
    """
    import json

    from proto_tools.utils import run_in_env

    out = json.loads(run_in_env("esm2", code=_MOVE_CHECK, device="cudax2").strip().splitlines()[-1])

    def on(device: str) -> dict[str, str]:
        return {"lin": device, "cached": device, "param_ref": device, "host": "cpu"}

    assert out == {"cuda:0": on("cuda:0"), "cpu": on("cpu"), "cuda:1": on("cuda:1"), "same_param": True, "current": 1}


_GPU_TO_GPU_MOVE = """
import json, os, subprocess, time, torch
from standalone_helpers import move_model_to_device

def gpus_held():
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout
    rows = (line.partition(",") for line in out.splitlines())
    return len({bus.strip() for pid, _, bus in rows if pid.strip() == str(os.getpid())})

def timed(fn):
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)
    return result, time.perf_counter() - start

# About 1 GB of fp32 weights, enough for a copy over the CPU to be far slower than a direct one
model = torch.nn.Sequential(*[torch.nn.Linear(4096, 4096) for _ in range(16)])
model = move_model_to_device(model, "cpu", "cuda:1")
x = torch.randn(2, 4096)
expected = model(x.to("cuda:1")).cpu()
model, _ = timed(lambda: move_model_to_device(model, "cuda:1", "cuda:0"))  # warms cuda:0's allocator
model, _ = timed(lambda: move_model_to_device(model, "cuda:0", "cuda:1"))
model, direct = timed(lambda: move_model_to_device(model, "cuda:1", "cuda:0"))
staged = timed(lambda: [p.detach().cpu().to("cuda:1") for p in model.parameters()])[1]  # what a CPU round trip costs
print(json.dumps({
    "gpus_held": gpus_held(),
    "on": str(next(model.parameters()).device),
    "same_output": bool(torch.allclose(model(x.to("cuda:0")).cpu(), expected)),
    "direct_faster_than_staging": direct * 3 < staged,
    "seconds": {"direct": round(direct, 4), "staged": round(staged, 4)},
}))
"""


@pytest.mark.uses_env
@pytest.mark.uses_gpu(3)
def test_move_between_gpus_is_direct_and_opens_no_context_on_a_third_gpu() -> None:
    """A model moving from one GPU to another goes straight across, touching only those two GPUs.

    PyTorch's own GPU-to-GPU copy opens a CUDA context (~500 MB) on every visible GPU, so a worker
    that followed one move held memory on GPUs leased to other tools. With three GPUs visible, the
    moved worker may hold contexts on only the two it used, its outputs must be unchanged, and the
    move must stay a direct copy: at least 3x faster than a round trip through the CPU.
    """
    import json

    from proto_tools.utils import run_in_env

    out = json.loads(run_in_env("esm2", code=_GPU_TO_GPU_MOVE, device="cudax3").strip().splitlines()[-1])
    seconds = out.pop("seconds")

    assert out == {"gpus_held": 2, "on": "cuda:0", "same_output": True, "direct_faster_than_staging": True}, seconds


# ── The worker follows each request's device ─────────────────────────────────


@pytest.mark.parametrize(
    ("torch_state", "device", "called_with"),
    [
        ("cuda_up", "cuda:1", "cuda:1"),
        ("imported_only", "cuda:1", None),
        ("absent", "cuda:1", None),
        ("cuda_up", "cpu", None),
        ("cuda_up", None, None),
    ],
    ids=["torch_cuda_worker", "torch_without_cuda", "no_torch", "cpu", "no_device"],
)
def test_worker_makes_the_request_device_current(monkeypatch, torch_state, device, called_with) -> None:
    """Code using PyTorch's current device must run on the dispatched GPU.

    Only in a worker that already uses CUDA through torch. One that merely imported torch (a
    per-call CLI tool reporting memory stats, say) must not open a context: on an
    Exclusive_Process GPU it would lock out the subprocess doing the work.
    """
    import sys
    import types

    from proto_tools.utils import _worker_bootstrap

    calls: list[str] = []
    helpers = types.SimpleNamespace(set_current_cuda_device=calls.append)
    monkeypatch.setitem(sys.modules, "standalone_helpers", helpers)
    if torch_state == "absent":
        monkeypatch.delitem(sys.modules, "torch", raising=False)
    else:
        initialized = torch_state == "cuda_up"
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda: initialized))
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
    _worker_bootstrap._follow_request_device({} if device is None else {"device": device})
    assert calls == ([] if called_with is None else [called_with])


# ── Auto-pinning on real GPUs ─────────────────────────────────────────────────


@pytest.mark.uses_env
@pytest.mark.uses_gpu(2)
def test_auto_pinned_worker_sees_only_its_gpu() -> None:
    """On an Exclusive_Process host a GPU worker lands on its lease alone, and a move respawns it."""
    from proto_tools.tools.testing.mock_pytorch_tool import (
        MockPyTorchToolConfig,
        MockPyTorchToolInput,
        run_mock_pytorch_tool,
    )
    from tests.tool_infra_tests.test_device_manager._gpu_watch import GpuProcessWatcher, physical_index

    def run(device: str) -> str:
        result = run_mock_pytorch_tool(
            MockPyTorchToolInput(), MockPyTorchToolConfig(device=device, memory_mb=64), instance="auto_pinned"
        )
        assert result.success, result.errors
        return result.results[0].device_used

    with _host(exclusive=True) as dm:
        dm.configure(managed_devices=["cuda:0", "cuda:1"])
        try:
            with ToolInstance.persist_tool("mock_pytorch_tool", instance_name="auto_pinned") as instance:
                with GpuProcessWatcher() as landing:
                    assert run("cuda:1") == "cuda:0"  # the pinned worker addresses its lease locally
                    landing.sample()
                assert landing.used == {physical_index("cuda:1")}, landing.seen
                pid = instance._worker._process.pid

                with GpuProcessWatcher() as moved:
                    assert run("cuda:0") == "cuda:0"
                    moved.sample()
                new_pid = instance._worker._process.pid
                assert new_pid != pid, "a pinned worker must respawn on a device change"
                assert moved.seen.get(new_pid) == {physical_index("cuda:0")}, moved.seen
        finally:
            ToolInstance.clear_all()
