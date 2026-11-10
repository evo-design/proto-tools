"""tests/tool_infra_tests/test_device_manager/test_leased_gpu.py.

Every GPU toolkit computes on the GPU DeviceManager leased it, not on whichever GPU is first.

A worker that is not pinned (``pin_visible_devices``) sees every GPU, so code that resolves "the
GPU" without the leased index (a bare ``"cuda"``, Lightning's ``devices=1``, a library default)
lands on GPU 0 whatever the lease. Nothing fails while GPU 0 is free: the model just runs on a
device another worker may hold, which on an ``Exclusive_Process`` GPU breaks that worker. So this
leases ``cuda:1`` and watches, through ``nvidia-smi``, which GPUs the tool's processes actually use.
"""

from __future__ import annotations

import os
import subprocess
import threading
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


@pytest.mark.extensive
@pytest.mark.uses_gpu(2)
@pytest.mark.slow
@pytest.mark.parametrize("tool_spec", _GPU_TOOLKITS)
def test_gpu_tool_computes_on_its_leased_gpu(tool_spec):
    """Lease cuda:1 to the tool; every GPU context its processes open must be on that GPU."""
    if tool_spec.key in _NO_EVICTION_TOOLS:
        pytest.skip(f"{tool_spec.key}: minimal() config runs on CPU")
    toolkit = tool_spec.source_file.parent.name
    instance_name = f"leased_{tool_spec.key.replace('-', '_')}"
    config = tool_spec.config_model.minimal().model_copy(update={"device": _LEASED})

    with patch("proto_tools.utils.device_manager.number_of_visible_gpus", return_value=2):
        DeviceManager.reset_instance()
        ToolInstance.clear_all()
        dm = DeviceManager.get_instance()
        dm.configure(managed_devices=["cuda:0", "cuda:1"])
        try:
            with _GpuProcessWatcher() as watcher, ToolInstance.persist_tool(toolkit, instance_name=instance_name):
                result = tool_spec.function(tool_spec.example_input(), config, instance=instance_name)
                assert result.success, f"{tool_spec.key} failed: {result.errors}"
                # A fast tool can finish between polls; its idle persistent worker still holds its context.
                watcher.sample()
                lease = dm.get_device_status()["allocations"].get(instance_name)
                assert lease is not None and lease["device_id"] == _LEASED, f"{tool_spec.key} lease: {lease}"
        finally:
            ToolInstance.clear_all()
            DeviceManager.reset_instance()

    expected = _physical_index(_LEASED)
    stray = {pid: sorted(gpus) for pid, gpus in watcher.seen.items() if gpus - {expected}}
    assert watcher.seen, f"{tool_spec.key}: no GPU process was observed; the poll may have missed a short run"
    assert not stray, (
        f"{tool_spec.key} was leased {_LEASED} (GPU {expected}) but its processes used other GPUs: {stray}"
    )
