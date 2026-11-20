"""tests/tool_infra_tests/test_device_manager/_gpu_watch.py.

Watch which GPUs a test's tool processes open CUDA contexts on, through ``nvidia-smi``.

A worker that picks "the GPU" itself instead of its leased device fails nothing while the other
GPU is idle, so the only reliable signal is where its processes actually hold contexts.
"""

from __future__ import annotations

import os
import subprocess
import threading

import psutil

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


def physical_index(device: str) -> int:
    """The physical GPU index a parent-side ``cuda:N`` names, through the parent's CUDA_VISIBLE_DEVICES."""
    logical = int(device.split(":")[1])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return int(visible.split(",")[logical]) if visible else logical


class GpuProcessWatcher:
    """Record, while running, which GPUs this test's descendant processes hold contexts on.

    Polls every second; call ``sample()`` after a fast run while its worker is still alive, since
    an idle persistent worker keeps its context.
    """

    def __init__(self) -> None:
        self._bus_to_index = _gpu_index_by_bus_id()
        self._root = psutil.Process()
        self._stop = threading.Event()
        self.seen: dict[int, set[int]] = {}  # pid -> physical GPU indices
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def used(self) -> set[int]:
        """Every physical GPU any watched process opened a context on."""
        return set().union(*self.seen.values()) if self.seen else set()

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

    def __enter__(self) -> GpuProcessWatcher:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.sample()
        self._stop.set()
        self._thread.join()
