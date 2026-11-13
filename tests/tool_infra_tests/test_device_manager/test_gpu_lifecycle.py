"""tests/tool_infra_tests/test_device_manager/test_gpu_lifecycle.py.

One device lifecycle per GPU toolkit: land on the leased GPU, move to another, survive eviction.

Each GPU toolkit's representative key runs through, in one persistent worker:

1. **Land**: lease ``cuda:1`` and run. Every GPU context the tool's processes open must be on
   that GPU, and the loaded worker must report well-formed memory stats. A worker that is not
   pinned sees every GPU, so code that picks "the GPU" itself (a bare ``"cuda"``, Lightning's
   ``devices=1``, upstream ``.cuda()``) would land on GPU 0 and fail nothing while it is idle.
2. **Move**: rerun on ``cuda:0``. The next run must use GPU 0. Routed tools move in-process and
   keep their worker (their old context stays until it exits); pinned tools respawn on GPU 0 alone.
3. **Evict and return**: a mock tool claims the tool's GPU, DeviceManager evicts it (to CPU, or
   by restart on Exclusive_Process hosts), and the tool must run again on a GPU.

Pipeline tools that start a fresh process for every call (BindCraft, AlphaFold3, RF3, ...) stop
after phase 1. Their worker holds no GPU state, so a move or an eviction only reruns the pipeline;
they say so by reporting ``framework="cli"`` memory stats.

Exclusive_Process auto-pinning is turned off (``_auto_pin_exclusive``), so on such a host each tool
still runs as on a normal-mode one and its own device routing is what is tested. With one visible
GPU, the tool lands on ``cuda:0`` and phase 2 is skipped. Each case runs ``config_model.minimal()``,
so a broken ``minimal()`` for the representative key surfaces here too.
"""

from __future__ import annotations

import contextlib
import time
from collections import defaultdict
from collections.abc import Iterator
from unittest.mock import patch

import pytest

from proto_tools.tools.tool_registry import ToolRegistry, ToolSpec
from proto_tools.utils.device import number_of_visible_gpus
from proto_tools.utils.device_manager import DeviceManager, OffloadStrategy
from proto_tools.utils.standalone_helpers_source.standalone_helpers.memory import JAX_UNTRACKED_MEMORY_REASON
from proto_tools.utils.tool_instance import ToolInstance
from tests.tool_infra_tests.test_device_manager._gpu_watch import GpuProcessWatcher, physical_index

# Tools that take a GPU only when explicitly asked, so their minimal() config runs on CPU and
# never occupies a device to evict. Their to_device() is a passthrough — the CLI subprocess
# releases the GPU when it exits — so there is no residency for this lifecycle to exercise.
_NO_EVICTION_TOOLS = frozenset({"mmseqs2-homology-search", "mmseqs2-search-proteins"})

# Toolkits whose worker holds GPU state its framework does not track (TensorFlow), so
# get_memory_stats() reports available=False. Per-call CLI tools report framework="cli" instead.
_NO_MEMORY_STATS_TOOLKITS = frozenset({"spliceai"})

# The framework a worker reports when every call runs in a fresh subprocess.
_PER_CALL_FRAMEWORK = "cli"

# Key tokens in order of preference when choosing a toolkit's representative key: the cheapest
# call that still loads the toolkit's model.
_KEY_TOKEN_PREFERENCE = ("score", "embedding", "prediction", "predict")


def _representative_key_rank(spec: ToolSpec) -> tuple[int, str]:
    """Rank a key by its best preferred token, breaking ties alphabetically."""
    tokens = spec.key.split("-")
    rank = min(
        (i for i, token in enumerate(_KEY_TOKEN_PREFERENCE) if token in tokens), default=len(_KEY_TOKEN_PREFERENCE)
    )
    return rank, spec.key


def _one_spec_per_toolkit() -> list[ToolSpec]:
    """Return one representative non-mock GPU tool per toolkit.

    Device movement lives in the toolkit's standalone worker, which every key in the toolkit
    shares, so one key per toolkit exercises the same ``to_device()``/``get_memory_stats()``.
    """
    by_toolkit: dict[str, list[ToolSpec]] = defaultdict(list)
    for spec in ToolRegistry.list_all():
        if spec.uses_gpu and spec.category != "testing":
            by_toolkit[spec.source_file.parent.name].append(spec)
    return [min(specs, key=_representative_key_rank) for _, specs in sorted(by_toolkit.items())]


_GPU_TOOLKITS = [pytest.param(spec, id=spec.key) for spec in _one_spec_per_toolkit()]


def _assert_memory_stats(tool_key: str, toolkit: str, stats: dict) -> None:
    """Assert a loaded worker reports well-formed memory stats, or opts out by a documented rule.

    Byte counts may be 0: workers that load per call or free GPU memory after each call hold
    nothing between calls. JAX workers run XLA's platform allocator, which tracks no stats, so
    they may report unavailable, but only with ``JAX_UNTRACKED_MEMORY_REASON``.
    """
    if stats.get("framework") == _PER_CALL_FRAMEWORK:
        assert stats["available"] is False, f"{tool_key}: a per-call CLI worker holds no GPU state: {stats}"
        return
    if toolkit in _NO_MEMORY_STATS_TOOLKITS:
        assert stats["available"] is False, (
            f"{tool_key}: listed in _NO_MEMORY_STATS_TOOLKITS but reports memory stats; drop it from the set"
        )
        return
    assert stats["framework"] in ("pytorch", "jax"), f"{tool_key}: unexpected framework in {stats}"
    if stats["framework"] == "jax" and stats["available"] is False:
        assert stats["reason"] == JAX_UNTRACKED_MEMORY_REASON, f"{tool_key}: JAX memory stats failed: {stats}"
        return
    assert stats["available"] is True, f"{tool_key}: get_memory_stats() unavailable with the model loaded: {stats}"
    for key in ("allocated_bytes", "max_allocated_bytes"):
        assert isinstance(stats[key], int) and stats[key] >= 0, f"{tool_key}: malformed {key} in {stats}"


@contextlib.contextmanager
def _routing_manager(devices: list[str]) -> Iterator[DeviceManager]:
    """A fresh DeviceManager over ``devices``, with CPU offload and Exclusive_Process auto-pinning off.

    On an Exclusive_Process host DeviceManager still escalates CPU offload to RESTART.
    """
    with patch("proto_tools.utils.device_manager.number_of_visible_gpus", return_value=len(devices)):
        DeviceManager.reset_instance()
        ToolInstance.clear_all()
        dm = DeviceManager.get_instance()
        dm._auto_pin_exclusive = False
        dm.configure(managed_devices=devices, offload_strategy=OffloadStrategy.CPU)
        try:
            yield dm
        finally:
            ToolInstance.clear_all()
            DeviceManager.reset_instance()


def _run_on(tool_spec: ToolSpec, device: str, instance_name: str) -> None:
    config = tool_spec.config_model.minimal().model_copy(update={"device": device})
    result = tool_spec.function(tool_spec.example_input(), config, instance=instance_name)
    assert result.success, f"{tool_spec.key} failed on {device}: {result.errors}"


def _worker_pid(instance: ToolInstance) -> int | None:
    worker = instance._worker
    return worker._process.pid if worker is not None and worker._process is not None else None


def _lease(dm: DeviceManager, instance_name: str) -> str | None:
    allocation = dm.get_device_status()["allocations"].get(instance_name)
    return None if allocation is None else allocation["device_id"]


@pytest.mark.extensive
@pytest.mark.uses_gpu
@pytest.mark.slow
@pytest.mark.parametrize("tool_spec", _GPU_TOOLKITS)
def test_gpu_tool_device_lifecycle(tool_spec: ToolSpec) -> None:
    """Land on the leased GPU, follow a move to another GPU, and run again after eviction."""
    key = tool_spec.key
    if key in _NO_EVICTION_TOOLS:
        assert tool_spec.config_model.minimal().gpus_per_instance == 0, (
            f"{key}: listed in _NO_EVICTION_TOOLS but its minimal() config claims a GPU; "
            "drop it from the set so the lifecycle runs"
        )
        pytest.skip(f"{key}: minimal() config runs on CPU, so there is no GPU lifecycle to exercise")

    two_gpus = number_of_visible_gpus() >= 2
    devices = ["cuda:0", "cuda:1"] if two_gpus else ["cuda:0"]
    leased = devices[-1]
    toolkit = tool_spec.source_file.parent.name
    name = f"lifecycle_{key.replace('-', '_')}"

    with _routing_manager(devices) as dm, ToolInstance.persist_tool(toolkit, instance_name=name) as instance:
        # 1. Land on the leased GPU, and only there.
        with GpuProcessWatcher() as landing:
            _run_on(tool_spec, leased, name)
            landing.sample()  # a fast tool can finish between polls; its idle worker keeps its context
        assert _lease(dm, name) == leased, f"{key} lease: {_lease(dm, name)}"
        stray = {pid: sorted(gpus) for pid, gpus in landing.seen.items() if gpus - {physical_index(leased)}}
        assert landing.seen, f"{key}: no GPU process was observed; the poll may have missed a short run"
        assert not stray, f"{key} was leased {leased} but its processes used other GPUs: {stray}"
        stats = instance.get_memory_stats()
        _assert_memory_stats(key, toolkit, stats)
        if stats.get("framework") == _PER_CALL_FRAMEWORK:
            return  # each call is a fresh process: a move or an eviction only reruns the pipeline

        # 2. Follow a move to the other GPU.
        if two_gpus:
            old, new = physical_index(leased), physical_index("cuda:0")
            pid_before = _worker_pid(instance)
            with GpuProcessWatcher() as moved:
                _run_on(tool_spec, "cuda:0", name)
                moved.sample()
            assert new in moved.used, f"{key}: after moving to cuda:0, nothing ran on GPU {new}: {moved.seen}"
            assert moved.used <= {old, new}, f"{key}: a GPU other than its old and new lease was used: {moved.seen}"
            if tool_spec.pin_visible_devices:
                pid_after = _worker_pid(instance)
                assert pid_after != pid_before, f"{key} is pinned, so the move should respawn its worker"
                # The old worker holds its GPU until it is stopped, so check the new one alone.
                respawned = moved.seen.get(pid_after, set()) if pid_after is not None else set()
                assert respawned <= {new}, f"{key}: the respawned pinned worker touched GPU {old}: {moved.seen}"
            else:
                assert _worker_pid(instance) == pid_before, f"{key} should move in-process, not respawn"

        # 3. A mock tool claims this tool's GPU; the tool is evicted, then runs on a GPU again.
        from proto_tools.tools.testing.mock_pytorch_tool import (
            MockPyTorchToolConfig,
            MockPyTorchToolInput,
            run_mock_pytorch_tool,
        )

        held = _lease(dm, name)
        time.sleep(0.01)  # LRU ordering
        evictor = f"evictor_{name}"
        with ToolInstance.persist_tool("mock_pytorch_tool", instance_name=evictor):
            mock = run_mock_pytorch_tool(
                MockPyTorchToolInput(), MockPyTorchToolConfig(device=held, memory_mb=64), instance=evictor
            )
            assert mock.success, f"Mock tool failed while evicting {key}: {mock.errors}"
            if dm._offload_strategy == OffloadStrategy.CPU:
                assert _lease(dm, name) == "cpu", f"{key} should be offloaded to CPU, got {_lease(dm, name)}"
            else:  # RESTART removes the allocation
                assert _lease(dm, name) is None, f"{key} should have been restarted off the GPU, got {_lease(dm, name)}"
            time.sleep(0.01)
            _run_on(tool_spec, leased, name)
            assert _lease(dm, name) not in (None, "cpu"), f"{key} should be back on a GPU, got {_lease(dm, name)}"
