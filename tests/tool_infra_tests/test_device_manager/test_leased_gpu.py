"""tests/tool_infra_tests/test_device_manager/test_leased_gpu.py.

Every GPU toolkit computes on the GPU DeviceManager leased it, not on whichever GPU is first.

A worker that is not pinned (``pin_visible_devices``) sees every GPU, so code that resolves "the
GPU" without the leased index (a bare ``"cuda"``, Lightning's ``devices=1``, a library default)
lands on GPU 0 whatever the lease. Nothing fails while GPU 0 is free: the model just runs on a
device another worker may hold, which on an ``Exclusive_Process`` GPU breaks that worker. So this
leases ``cuda:1`` and watches, through ``nvidia-smi``, which GPUs the tool's processes actually use.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import threading
from collections.abc import Iterator
from unittest.mock import patch

import psutil
import pytest

from proto_tools.utils.device_manager import DeviceManager
from proto_tools.utils.tool_instance import ToolInstance
from tests.tool_infra_tests.test_device_manager.test_eviction_round_trip import _GPU_TOOLKITS, _NO_EVICTION_TOOLS

_LEASED = "cuda:1"
_POLL_S = 1.0


def _gpu_index_by_bus_id() -> dict[str, int]:
    """Map each GPU's PCI bus id to its nvidia-smi index (the physical index CUDA_VISIBLE_DEVICES uses)."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {bus.strip().lower(): int(index) for index, bus in (line.split(",") for line in out.splitlines() if line)}


def _physical_index(device: str) -> int:
    """The physical GPU index a parent-side ``cuda:N`` names, through the parent's CUDA_VISIBLE_DEVICES."""
    logical = int(device.split(":")[1])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return int(visible.split(",")[logical]) if visible else logical


class _GpuProcessWatcher:
    """Record, while running, which GPUs this test's descendant processes hold contexts on."""

    def __init__(self) -> None:
        self._bus_to_index = _gpu_index_by_bus_id()
        self._root = psutil.Process()
        self._stop = threading.Event()
        self.seen: dict[int, set[int]] = {}  # pid -> physical GPU indices
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _descendants(self) -> set[int]:
        try:
            return {p.pid for p in self._root.children(recursive=True)}
        except psutil.Error:
            return set()

    def sample(self) -> None:
        """Record the GPUs this test's descendant processes hold right now."""
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        ours = self._descendants()
        for line in out.splitlines():
            pid_text, _, bus = line.partition(",")
            if not pid_text.strip().isdigit():
                continue
            pid = int(pid_text)
            index = self._bus_to_index.get(bus.strip().lower())
            if pid in ours and index is not None:
                self.seen.setdefault(pid, set()).add(index)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(_POLL_S)

    def __enter__(self) -> _GpuProcessWatcher:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.sample()
        self._stop.set()
        self._thread.join()


@contextlib.contextmanager
def _two_gpu_manager(*, auto_pin: bool) -> Iterator[DeviceManager]:
    """A fresh DeviceManager over cuda:0 and cuda:1, with Exclusive_Process auto-pinning on or off.

    Off, every tool runs as on a normal-mode host: it sees both GPUs, so this checks that its own
    code lands on the lease. On (an Exclusive_Process host), every GPU worker sees only its lease.
    """
    with patch("proto_tools.utils.device_manager.number_of_visible_gpus", return_value=2):
        DeviceManager.reset_instance()
        ToolInstance.clear_all()
        dm = DeviceManager.get_instance()
        dm._auto_pin_exclusive = auto_pin
        dm.configure(managed_devices=["cuda:0", "cuda:1"])
        try:
            yield dm
        finally:
            ToolInstance.clear_all()
            DeviceManager.reset_instance()


def _run_on(tool_spec, device: str, instance_name: str):
    config = tool_spec.config_model.minimal().model_copy(update={"device": device})
    result = tool_spec.function(tool_spec.example_input(), config, instance=instance_name)
    assert result.success, f"{tool_spec.key} failed on {device}: {result.errors}"
    return result


def _skip_cpu_only(tool_spec) -> None:
    if tool_spec.key in _NO_EVICTION_TOOLS:
        pytest.skip(f"{tool_spec.key}: minimal() config runs on CPU")


@pytest.mark.extensive
@pytest.mark.uses_gpu(2)
@pytest.mark.slow
@pytest.mark.parametrize("auto_pin", [False, True], ids=["routed", "auto_pinned"])
@pytest.mark.parametrize("tool_spec", _GPU_TOOLKITS)
def test_gpu_tool_computes_on_its_leased_gpu(tool_spec, auto_pin):
    """Lease cuda:1 to the tool; every GPU context its processes open must be on that GPU."""
    _skip_cpu_only(tool_spec)
    toolkit = tool_spec.source_file.parent.name
    instance_name = f"leased_{tool_spec.key.replace('-', '_')}"

    with _two_gpu_manager(auto_pin=auto_pin) as dm:
        if auto_pin and not dm.pins_gpu_workers:
            pytest.skip("not an Exclusive_Process host: auto-pinning is off, as in the routed case")
        with _GpuProcessWatcher() as watcher, ToolInstance.persist_tool(toolkit, instance_name=instance_name):
            _run_on(tool_spec, _LEASED, instance_name)
            # A fast tool can finish between polls; its idle persistent worker still holds its context.
            watcher.sample()
            lease = dm.get_device_status()["allocations"].get(instance_name)
            assert lease is not None and lease["device_id"] == _LEASED, f"{tool_spec.key} lease: {lease}"

    expected = _physical_index(_LEASED)
    stray = {pid: sorted(gpus) for pid, gpus in watcher.seen.items() if gpus - {expected}}
    assert watcher.seen, f"{tool_spec.key}: no GPU process was observed; the poll may have missed a short run"
    assert not stray, (
        f"{tool_spec.key} was leased {_LEASED} (GPU {expected}) but its processes used other GPUs: {stray}"
    )


@pytest.mark.extensive
@pytest.mark.uses_gpu(2)
@pytest.mark.slow
@pytest.mark.parametrize("tool_spec", _GPU_TOOLKITS)
def test_gpu_tool_follows_a_move_to_another_gpu(tool_spec):
    """Move a loaded tool from cuda:1 to cuda:0; its next run must use cuda:0.

    Routed tools (all GPUs visible) move in-process, keeping their worker; the context they opened on
    cuda:1 stays, as CUDA keeps it until the process exits. Pinned tools respawn on cuda:0 alone.
    """
    _skip_cpu_only(tool_spec)
    toolkit = tool_spec.source_file.parent.name
    instance_name = f"mover_{tool_spec.key.replace('-', '_')}"
    old, new = _physical_index(_LEASED), _physical_index("cuda:0")

    with (
        _two_gpu_manager(auto_pin=False),
        ToolInstance.persist_tool(toolkit, instance_name=instance_name) as instance,
    ):
        _run_on(tool_spec, _LEASED, instance_name)
        worker = instance._worker
        pid_before = worker._process.pid if worker is not None and worker._process is not None else None
        with _GpuProcessWatcher() as watcher:
            _run_on(tool_spec, "cuda:0", instance_name)
            watcher.sample()
            worker = instance._worker
            pid_after = worker._process.pid if worker is not None and worker._process is not None else None

    used = set().union(*watcher.seen.values()) if watcher.seen else set()
    assert new in used, f"{tool_spec.key}: after moving to cuda:0, nothing ran on GPU {new}: {watcher.seen}"
    assert used <= {old, new}, f"{tool_spec.key}: a GPU other than its old and new lease was used: {watcher.seen}"
    if tool_spec.pin_visible_devices:
        assert pid_after != pid_before, f"{tool_spec.key} is pinned, so the move should respawn its worker"
        assert used == {new}, f"{tool_spec.key}: the respawned pinned worker touched GPU {old}: {watcher.seen}"
    else:
        assert pid_after == pid_before, f"{tool_spec.key} should move in-process, not respawn"
