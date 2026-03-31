"""tests/tool_infra_tests/test_device_manager/test_stress.py

Tests for device movement stress with real tool instances."""

import time
from unittest.mock import patch

import pytest

from proto_tools.utils.device_manager import DeviceManager, OffloadStrategy
from proto_tools.utils.tool_instance import ToolInstance


@pytest.fixture(autouse=True)
def _mock_exclusive_process():
    """Disable Exclusive_Process auto-escalation so tests control the strategy."""
    with patch(
        "proto_tools.utils.device_manager.is_exclusive_process_mode",
        return_value=False,
    ):
        yield


# ── Constants ───────────────────────────────────────────────────────────

# Memory allocated by PyTorch/JAX mock tools via buffer tensor/array (MB).
# Set large enough that CUDA/JAX runtime context overhead (~500-700 MB)
# falls well within tolerance, avoiding the need for warm-up baselines.
_TOOL_MEMORY_MB = 4096

# Tolerance for GPU memory assertions (MB).  Must absorb CUDA/JAX runtime
# context (~500-700 MB per framework), caching allocator overhead, nvidia-smi
# granularity, and residual context from cross-framework eviction — but less
# than _TOOL_MEMORY_MB so "loaded" vs "freed" states remain distinguishable.
_GPU_MEMORY_TOLERANCE_MB = 3072


# ── Helpers ─────────────────────────────────────────────────────────────

# Each entry: (tool_name, InputClass, ConfigClass, run_fn)
# Populated lazily so imports don't fail on environments without JAX/torch.

def _pytorch_tool():
    from proto_tools.tools.testing.mock_pytorch_tool import (
        MockPyTorchToolConfig,
        MockPyTorchToolInput,
        run_mock_pytorch_tool,
    )
    return "mock_pytorch_tool", MockPyTorchToolInput, MockPyTorchToolConfig, run_mock_pytorch_tool


def _jax_tool():
    from proto_tools.tools.testing.mock_jax_tool import (
        MockJAXToolConfig,
        MockJAXToolInput,
        run_mock_jax_tool,
    )
    return "mock_jax_tool", MockJAXToolInput, MockJAXToolConfig, run_mock_jax_tool


def _cli_tool():
    from proto_tools.tools.testing.mock_cli_tool import (
        MockCLIToolConfig,
        MockCLIToolInput,
        run_mock_cli_tool,
    )
    return "mock_cli_tool", MockCLIToolInput, MockCLIToolConfig, run_mock_cli_tool


def _pytorch_multi_gpu_tool():
    from proto_tools.tools.testing.mock_pytorch_multi_gpu_tool import (
        MockPyTorchMultiGPUToolConfig,
        MockPyTorchMultiGPUToolInput,
        run_mock_pytorch_multi_gpu_tool,
    )
    return "mock_pytorch_multi_gpu_tool", MockPyTorchMultiGPUToolInput, MockPyTorchMultiGPUToolConfig, run_mock_pytorch_multi_gpu_tool


def _jax_multi_gpu_tool():
    from proto_tools.tools.testing.mock_jax_multi_gpu_tool import (
        MockJAXMultiGPUToolConfig,
        MockJAXMultiGPUToolInput,
        run_mock_jax_multi_gpu_tool,
    )
    return "mock_jax_multi_gpu_tool", MockJAXMultiGPUToolInput, MockJAXMultiGPUToolConfig, run_mock_jax_multi_gpu_tool


_SINGLE_GPU_TOOLS = [
    pytest.param(_pytorch_tool, id="pytorch"),
    pytest.param(_jax_tool, id="jax"),
    pytest.param(_cli_tool, id="cli"),
]

# Tools that hold GPU memory in persistent workers (PyTorch buffers, JAX arrays).
# CLI tools auto-unload after each call and don't maintain GPU state.
_PERSISTENT_GPU_TOOLS = [
    pytest.param(_pytorch_tool, id="pytorch"),
    pytest.param(_jax_tool, id="jax"),
]

_MULTI_GPU_PERSISTENT_TOOLS = [
    pytest.param(_pytorch_multi_gpu_tool, id="pytorch-multi"),
    pytest.param(_jax_multi_gpu_tool, id="jax-multi"),
]


def _run_tool(tool_factory, instance_name, **config_kwargs):
    """Helper: dispatch a single call through a mock tool.

    Extra *config_kwargs* (e.g. ``memory_mb=2048``) are forwarded to the
    tool's Config constructor, allowing tests to control the memory footprint.
    Defaults ``memory_mb`` to _TOOL_MEMORY_MB so all stress tests use a
    consistent, large-enough buffer for reliable memory assertions.
    """
    if "memory_mb" not in config_kwargs:
        config_kwargs["memory_mb"] = _TOOL_MEMORY_MB
    tool_name, InputCls, ConfigCls, run_fn = tool_factory()
    return run_fn(InputCls(), ConfigCls(**config_kwargs), instance=instance_name)


def _setup_dm(managed_devices=None, strategy=OffloadStrategy.CPU):
    """Reset DeviceManager and ToolInstance state, configure fresh."""
    DeviceManager.reset_instance()
    ToolInstance.clear_all()
    dm = DeviceManager.get_instance()
    kwargs = {"offload_strategy": strategy}
    if managed_devices is not None:
        kwargs["managed_devices"] = managed_devices
    dm.configure(**kwargs)
    return dm


def _teardown():
    ToolInstance.clear_all()
    DeviceManager.reset_instance()
    # Delay to allow GPU memory to fully reclaim before the next test snapshots baseline.
    time.sleep(2)


# ── GPU memory snapshot helpers ─────────────────────────────────────────

def _snapshot_gpu_memory(dm: DeviceManager) -> dict[str, float]:
    """Capture memory usage (MB) for managed CUDA devices only.

    Returns a dict like ``{"cuda:0": 1234.5, "cuda:1": 567.8}``.
    Only snapshots devices that DeviceManager is managing, since we cannot
    control what other processes do on non-managed GPUs.
    """
    status = dm.get_device_status()
    managed = status["available_devices"]

    snapshot = {}
    for device in managed:
        if device.startswith("cuda:"):
            snapshot[device] = dm.get_gpu_memory_used(device) / (1024 * 1024)
    return snapshot


def _assert_gpu_memory(
    dm: DeviceManager,
    baseline: dict[str, float],
    loaded: list[str] | None = None,
    freed: list[str] | None = None,
    tolerance_mb: float = _GPU_MEMORY_TOLERANCE_MB,
    label: str = "",
) -> dict[str, float]:
    """Snapshot GPU memory and assert expected state relative to baseline.

    The baseline should be a "warm" snapshot taken after CUDA/JAX context
    is already initialized on the relevant devices (e.g., after a tool has
    been loaded and unloaded once). This avoids counting one-time CUDA
    context overhead (~500-700 MB) in the delta.

    Args:
        dm: DeviceManager instance.
        baseline: Warm memory snapshot (post-context-init, pre-model-load).
        loaded: Devices expected to have ~_TOOL_MEMORY_MB above baseline
                (a persistent GPU tool is resident). CLI tools auto-unload
                after each call and should NOT be listed here.
        freed: Devices that previously had a model but should now be near
               baseline (model moved away / evicted). Functionally identical
               to unmentioned devices — included for test readability.
        tolerance_mb: Acceptable deviation in MB for all comparisons.
        label: Context string for assertion messages.

    Returns:
        Current memory snapshot (for optional further comparisons).

    Note:
        If other processes are actively using the GPUs, memory assertions
        may fail due to external memory fluctuations.
    """
    current = _snapshot_gpu_memory(dm)
    loaded_set = set(loaded or [])
    suffix = f" ({label})" if label else ""
    noise_note = " (if other processes are using the GPU, this assertion may be unreliable)"

    for device, baseline_mb in baseline.items():
        current_mb = current.get(device, 0.0)
        delta = current_mb - baseline_mb

        if device in loaded_set:
            assert abs(delta - _TOOL_MEMORY_MB) < tolerance_mb, (
                f"{device}: expected ~{_TOOL_MEMORY_MB} MB above baseline{suffix}, "
                f"got delta={delta:.0f} MB (tolerance=+/-{tolerance_mb} MB)"
                f"{noise_note}"
            )
        else:
            # freed or unmentioned — should be near baseline
            assert abs(delta) < tolerance_mb, (
                f"{device}: expected near baseline{suffix}, "
                f"got delta={delta:.0f} MB (tolerance=+/-{tolerance_mb} MB)"
                f"{noise_note}"
            )

    return current


# ── Mixed-framework eviction ────────────────────────────────────────────

@pytest.mark.uses_gpu
@pytest.mark.slow
@pytest.mark.parametrize("tool_a_factory,tool_b_factory", [
    pytest.param(_pytorch_tool, _jax_tool, id="pytorch-evicts-jax"),
    pytest.param(_pytorch_tool, _cli_tool, id="pytorch-evicts-cli"),
    pytest.param(_jax_tool, _pytorch_tool, id="jax-evicts-pytorch"),
    pytest.param(_jax_tool, _cli_tool, id="jax-evicts-cli"),
    pytest.param(_cli_tool, _pytorch_tool, id="cli-evicts-pytorch"),
    pytest.param(_cli_tool, _jax_tool, id="cli-evicts-jax"),
])
def test_cross_framework_eviction_cpu(tool_a_factory, tool_b_factory):
    """Tool B should evict tool A to CPU when only 1 GPU is available."""
    dm = _setup_dm(["cuda:0"], strategy=OffloadStrategy.CPU)
    tool_a_name = tool_a_factory()[0]
    tool_b_name = tool_b_factory()[0]
    is_a_persistent = "cli" not in tool_a_name
    is_b_persistent = "cli" not in tool_b_name

    try:
        baseline = _snapshot_gpu_memory(dm)

        with ToolInstance.persist_tool(tool_a_name, instance_name="inst_a"):
            result_a = _run_tool(tool_a_factory, "inst_a")
            assert result_a.success, f"Tool A failed: {result_a.errors}"

            status = dm.get_device_status()
            assert status["allocations"]["inst_a"]["device_id"] == "cuda:0"

            if is_a_persistent:
                _assert_gpu_memory(dm, baseline, loaded=["cuda:0"],
                                   label="after loading tool A")

            time.sleep(0.01)

            with ToolInstance.persist_tool(tool_b_name, instance_name="inst_b"):
                result_b = _run_tool(tool_b_factory, "inst_b")
                assert result_b.success, f"Tool B failed: {result_b.errors}"

                status = dm.get_device_status()
                assert status["allocations"]["inst_a"]["device_id"] == "cpu", \
                    "Tool A should be evicted to CPU"
                assert status["allocations"]["inst_b"]["device_id"] == "cuda:0", \
                    "Tool B should be on GPU"

                if is_b_persistent:
                    _assert_gpu_memory(dm, baseline, loaded=["cuda:0"],
                                       label="after eviction, tool B on GPU")

                time.sleep(0.01)

                # 3) Re-run A — config says "cuda", so _run_persistent
                #    detects the mismatch (worker on CPU, config wants GPU),
                #    moves A back to GPU, evicting B to CPU.
                #    This covers the GPU->CPU->GPU round-trip.
                result_a2 = _run_tool(tool_a_factory, "inst_a")
                assert result_a2.success, "Tool A should reload after eviction"

                status = dm.get_device_status()
                assert status["allocations"]["inst_a"]["device_id"] == "cuda:0", \
                    "Tool A should be back on GPU after reload"
                assert status["allocations"]["inst_b"]["device_id"] == "cpu", \
                    "Tool B should be evicted to CPU"

                if is_a_persistent:
                    _assert_gpu_memory(dm, baseline, loaded=["cuda:0"],
                                       label="after tool A round-trip reload")
    finally:
        _teardown()


@pytest.mark.uses_gpu
@pytest.mark.slow
@pytest.mark.parametrize("tool_a_factory,tool_b_factory", [
    pytest.param(_pytorch_tool, _jax_tool, id="pytorch-evicts-jax"),
    pytest.param(_jax_tool, _pytorch_tool, id="jax-evicts-pytorch"),
    pytest.param(_cli_tool, _pytorch_tool, id="cli-evicts-pytorch"),
])
def test_cross_framework_eviction_restart(tool_a_factory, tool_b_factory):
    """Tool B should shut down tool A when using RESTART strategy."""
    dm = _setup_dm(["cuda:0"], strategy=OffloadStrategy.RESTART)
    tool_a_name = tool_a_factory()[0]
    tool_b_name = tool_b_factory()[0]
    is_a_persistent = "cli" not in tool_a_name
    is_b_persistent = "cli" not in tool_b_name

    try:
        baseline = _snapshot_gpu_memory(dm)

        with ToolInstance.persist_tool(tool_a_name, instance_name="inst_a"):
            result_a = _run_tool(tool_a_factory, "inst_a")
            assert result_a.success

            time.sleep(0.01)

            with ToolInstance.persist_tool(tool_b_name, instance_name="inst_b"):
                result_b = _run_tool(tool_b_factory, "inst_b")
                assert result_b.success

                # RESTART should fully remove tool A from allocations
                status = dm.get_device_status()
                assert "inst_a" not in status["allocations"], \
                    "Tool A should be removed with RESTART strategy"
                assert status["allocations"]["inst_b"]["device_id"] == "cuda:0"

                # Tool B on GPU (if persistent), tool A fully removed
                if is_b_persistent:
                    _assert_gpu_memory(dm, baseline, loaded=["cuda:0"],
                                       label="after RESTART eviction")

                time.sleep(0.01)

                # 3) Re-run A — auto-restarts, gets fresh allocation on GPU,
                #    evicts B via RESTART (B's worker is killed).
                #    This covers the GPU->killed->GPU round-trip.
                result_a2 = _run_tool(tool_a_factory, "inst_a")
                assert result_a2.success, "Evicted tool should auto-restart"

                status = dm.get_device_status()
                assert status["allocations"]["inst_a"]["device_id"] == "cuda:0", \
                    "Tool A should be back on GPU after restart"
                assert "inst_b" not in status["allocations"], \
                    "Tool B should be removed with RESTART strategy"

                if is_a_persistent:
                    _assert_gpu_memory(dm, baseline, loaded=["cuda:0"],
                                       label="after tool A round-trip restart")
    finally:
        _teardown()


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_three_tool_eviction_chain():
    """Three different frameworks compete for 1 GPU: each evicts the previous."""
    dm = _setup_dm(["cuda:0"], strategy=OffloadStrategy.CPU)

    try:
        baseline = _snapshot_gpu_memory(dm)

        with ToolInstance.persist_tool("mock_pytorch_tool", instance_name="pt"):
            result = _run_tool(_pytorch_tool, "pt")
            assert result.success
            assert dm.get_device_status()["allocations"]["pt"]["device_id"] == "cuda:0"
            _assert_gpu_memory(dm, baseline, loaded=["cuda:0"],
                               label="PyTorch loaded")

            time.sleep(0.01)

            with ToolInstance.persist_tool("mock_jax_tool", instance_name="jx"):
                result = _run_tool(_jax_tool, "jx")
                assert result.success

                status = dm.get_device_status()
                assert status["allocations"]["pt"]["device_id"] == "cpu"
                assert status["allocations"]["jx"]["device_id"] == "cuda:0"
                _assert_gpu_memory(dm, baseline, loaded=["cuda:0"],
                                   label="JAX replaced PyTorch")

                time.sleep(0.01)

                with ToolInstance.persist_tool("mock_cli_tool", instance_name="cl"):
                    result = _run_tool(_cli_tool, "cl")
                    assert result.success

                    status = dm.get_device_status()
                    assert status["allocations"]["pt"]["device_id"] == "cpu"
                    assert status["allocations"]["jx"]["device_id"] == "cpu"
                    assert status["allocations"]["cl"]["device_id"] == "cuda:0"

                    # All three should still produce results
                    for name, factory in [("pt", _pytorch_tool), ("jx", _jax_tool), ("cl", _cli_tool)]:
                        r = _run_tool(factory, name)
                        assert r.success, f"{name} should still work after eviction chain"
    finally:
        _teardown()


# ── Config-driven device movement ──────────────────────────────────────

@pytest.mark.uses_gpu(2)
@pytest.mark.slow
@pytest.mark.parametrize("tool_factory", _PERSISTENT_GPU_TOOLS)
def test_config_moves_between_gpus(tool_factory):
    """Changing config device between calls should move the tool."""
    dm = _setup_dm(["cuda:0", "cuda:1"], strategy=OffloadStrategy.CPU)
    tool_name = tool_factory()[0]

    try:
        baseline = _snapshot_gpu_memory(dm)

        with ToolInstance.persist_tool(tool_name, instance_name="mover"):
            result = _run_tool(tool_factory, "mover", device="cuda:0")
            assert result.success

            assert dm.get_device_status()["allocations"]["mover"]["device_id"] == "cuda:0"
            _assert_gpu_memory(dm, baseline, loaded=["cuda:0"],
                               label="after loading on cuda:0")

            # Re-run with device=cuda:1 — mismatch detection should move it
            result2 = _run_tool(tool_factory, "mover", device="cuda:1")
            assert result2.success, "Should work after config-driven move"
            assert dm.get_device_status()["allocations"]["mover"]["device_id"] == "cuda:1"
            _assert_gpu_memory(dm, baseline, loaded=["cuda:1"], freed=["cuda:0"],
                               label="after move to cuda:1")
    finally:
        _teardown()


@pytest.mark.uses_gpu(2)
@pytest.mark.slow
@pytest.mark.parametrize("tool_factory", _MULTI_GPU_PERSISTENT_TOOLS)
def test_config_gpu_cpu_gpu_multi_gpu(tool_factory):
    """Round-trip: GPU,GPU -> CPU -> GPU,GPU via config changes."""
    dm = _setup_dm(["cuda:0", "cuda:1"])
    tool_name = tool_factory()[0]

    try:
        baseline = _snapshot_gpu_memory(dm)

        with ToolInstance.persist_tool(tool_name, instance_name="bouncer"):
            result = _run_tool(tool_factory, "bouncer")
            assert result.success

            status = dm.get_device_status()
            assert "cuda:0" in status["allocations"]["bouncer"]["device_id"]
            assert "cuda:1" in status["allocations"]["bouncer"]["device_id"]
            _assert_gpu_memory(dm, baseline, loaded=["cuda:0", "cuda:1"],
                               label="after loading on both GPUs")

            # Run with device=cpu — mismatch detection should move to CPU
            result2 = _run_tool(tool_factory, "bouncer", device="cpu")
            assert result2.success, "Should work on CPU"
            assert dm.get_device_status()["allocations"]["bouncer"]["device_id"] == "cpu"
            _assert_gpu_memory(dm, baseline, freed=["cuda:0", "cuda:1"],
                               label="after move to CPU")

            # Run with device back on GPUs — should move back
            result3 = _run_tool(tool_factory, "bouncer", device="cuda:0,cuda:1")
            assert result3.success, "Should work back on GPUs"
            status = dm.get_device_status()
            assert "cuda:0" in status["allocations"]["bouncer"]["device_id"]
            assert "cuda:1" in status["allocations"]["bouncer"]["device_id"]
            _assert_gpu_memory(dm, baseline, loaded=["cuda:0", "cuda:1"],
                               label="after move back to GPUs")
    finally:
        _teardown()


# ── Worker lifecycle ────────────────────────────────────────────────────

@pytest.mark.uses_gpu
@pytest.mark.slow
@pytest.mark.parametrize("tool_factory", _PERSISTENT_GPU_TOOLS)
def test_shutdown_and_auto_restart(tool_factory):
    """Manually shutting down a worker should allow auto-restart on next call."""
    dm = _setup_dm(["cuda:0"])
    tool_name = tool_factory()[0]

    try:
        baseline = _snapshot_gpu_memory(dm)

        inst = ToolInstance.get(tool_name, instance_name="restarter")
        result = _run_tool(tool_factory, "restarter")
        assert result.success
        assert inst._worker is not None
        _assert_gpu_memory(dm, baseline, loaded=["cuda:0"],
                           label="after loading")

        # Kill the worker — GPU memory should return to baseline
        inst.shutdown(remove_from_cache=False)
        assert inst._worker is None
        _assert_gpu_memory(dm, baseline, freed=["cuda:0"],
                           label="after shutdown")

        # Next call should auto-restart — GPU memory should rise again
        result2 = _run_tool(tool_factory, "restarter")
        assert result2.success, "Should auto-restart after shutdown"
        assert inst._worker is not None, "Worker should be back"
        assert "restarter" in dm.get_device_status()["allocations"]
        _assert_gpu_memory(dm, baseline, loaded=["cuda:0"],
                           label="after auto-restart")
    finally:
        _teardown()


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_evict_then_restart_preserves_correctness():
    """After RESTART eviction, re-running the tool gives correct results."""
    dm = _setup_dm(["cuda:0"], strategy=OffloadStrategy.RESTART)

    try:
        baseline = _snapshot_gpu_memory(dm)

        inst_a = ToolInstance.get("mock_pytorch_tool", instance_name="a")
        result1 = _run_tool(_pytorch_tool, "a")
        assert result1.success
        first_output = result1.results  # Save for comparison
        _assert_gpu_memory(dm, baseline, loaded=["cuda:0"],
                           label="after loading tool A")

        time.sleep(0.01)

        # Evict A by loading B (RESTART kills A's worker entirely)
        with ToolInstance.persist_tool("mock_jax_tool", instance_name="b"):
            _run_tool(_jax_tool, "b")

            assert inst_a._worker is None, "Worker A should be shut down"
            assert "a" not in dm.get_device_status()["allocations"]

            # Tool B now occupies cuda:0 (tool A fully gone)
            _assert_gpu_memory(dm, baseline, loaded=["cuda:0"],
                               label="after RESTART eviction, tool B loaded")

        # Re-run A — should restart and produce same-shape output
        result2 = _run_tool(_pytorch_tool, "a")
        assert result2.success
        assert len(result2.results) == len(first_output), \
            "Restarted tool should produce same-shape output"
        _assert_gpu_memory(dm, baseline, loaded=["cuda:0"],
                           label="after tool A restart")
    finally:
        _teardown()


# ── Multi-GPU stress ────────────────────────────────────────────────────

@pytest.mark.uses_gpu(2)
@pytest.mark.slow
def test_multi_gpu_tool_evicts_two_single_gpu_tools():
    """A 2-GPU tool should evict two single-GPU tools occupying both GPUs."""
    dm = _setup_dm(managed_devices=["cuda:0", "cuda:1"], strategy=OffloadStrategy.CPU)

    try:
        baseline = _snapshot_gpu_memory(dm)

        # Fill both GPUs with different frameworks
        with ToolInstance.persist_tool("mock_pytorch_tool", instance_name="pt"):
            result_pt = _run_tool(_pytorch_tool, "pt")
            assert result_pt.success
            time.sleep(0.01)

            with ToolInstance.persist_tool("mock_jax_tool", instance_name="jx"):
                result_jx = _run_tool(_jax_tool, "jx")
                assert result_jx.success

                status = dm.get_device_status()
                assert status["allocations"]["pt"]["device_id"] == "cuda:0"
                assert status["allocations"]["jx"]["device_id"] == "cuda:1"
                _assert_gpu_memory(dm, baseline, loaded=["cuda:0", "cuda:1"],
                                   label="both GPUs occupied")

                time.sleep(0.01)

                # Multi-GPU tool needs both GPUs — should evict both
                with ToolInstance.persist_tool("mock_pytorch_multi_gpu_tool", instance_name="mg"):
                    result_mg = _run_tool(_pytorch_multi_gpu_tool, "mg")
                    assert result_mg.success

                    status = dm.get_device_status()
                    assert status["allocations"]["pt"]["device_id"] == "cpu"
                    assert status["allocations"]["jx"]["device_id"] == "cpu"
                    assert "cuda:0" in status["allocations"]["mg"]["device_id"]
                    assert "cuda:1" in status["allocations"]["mg"]["device_id"]
                    _assert_gpu_memory(dm, baseline, loaded=["cuda:0", "cuda:1"],
                                       label="multi-GPU tool replaced both")

                    # Evicted tools still work on CPU
                    assert _run_tool(_pytorch_tool, "pt").success
                    assert _run_tool(_jax_tool, "jx").success
    finally:
        _teardown()


@pytest.mark.uses_gpu(2)
@pytest.mark.slow
def test_single_gpu_tool_evicts_multi_gpu_tool():
    """A single-GPU request should evict a multi-GPU tool (freeing 2 GPUs)."""
    dm = _setup_dm(managed_devices=["cuda:0", "cuda:1"], strategy=OffloadStrategy.CPU)

    try:
        baseline = _snapshot_gpu_memory(dm)

        # Load multi-GPU tool on both GPUs
        with ToolInstance.persist_tool("mock_pytorch_multi_gpu_tool", instance_name="mg"):
            result_mg = _run_tool(_pytorch_multi_gpu_tool, "mg")
            assert result_mg.success

            status = dm.get_device_status()
            assert "cuda:0" in status["allocations"]["mg"]["device_id"]
            assert "cuda:1" in status["allocations"]["mg"]["device_id"]
            _assert_gpu_memory(dm, baseline, loaded=["cuda:0", "cuda:1"],
                               label="multi-GPU tool on both GPUs")

            time.sleep(0.01)

            # Two single-GPU tools force eviction of multi-GPU tool
            with ToolInstance.persist_tool("mock_pytorch_tool", instance_name="pt1"):
                result_pt = _run_tool(_pytorch_tool, "pt1")
                assert result_pt.success
                time.sleep(0.01)

                with ToolInstance.persist_tool("mock_jax_tool", instance_name="jx1"):
                    result_jx = _run_tool(_jax_tool, "jx1")
                    assert result_jx.success

                    status = dm.get_device_status()
                    assert status["allocations"]["mg"]["device_id"] == "cpu"
                    single_devices = {
                        status["allocations"]["pt1"]["device_id"],
                        status["allocations"]["jx1"]["device_id"],
                    }
                    assert single_devices == {"cuda:0", "cuda:1"}
                    _assert_gpu_memory(dm, baseline, loaded=["cuda:0", "cuda:1"],
                                       label="single-GPU tools replaced multi-GPU")
    finally:
        _teardown()


@pytest.mark.uses_gpu(2)
@pytest.mark.slow
def test_rapid_eviction_cycle():
    """Rapidly cycle tools on and off GPUs to stress eviction bookkeeping."""
    dm = _setup_dm(managed_devices=["cuda:0", "cuda:1"], strategy=OffloadStrategy.CPU)

    # Map instance names to tool names for CLI detection
    cli_instances = set()

    try:
        baseline = _snapshot_gpu_memory(dm)

        tools = [
            ("mock_pytorch_tool", _pytorch_tool, "cyc_pt"),
            ("mock_jax_tool", _jax_tool, "cyc_jx"),
            ("mock_cli_tool", _cli_tool, "cyc_cl"),
        ]

        # Track which instances are CLI tools
        for tool_name, _, inst_name in tools:
            if "cli" in tool_name:
                cli_instances.add(inst_name)

        # Persist all tools (only 2 GPUs, so one will always be evicted)
        for tool_name, _, inst_name in tools:
            ToolInstance.get(tool_name, instance_name=inst_name)

        # Run each tool twice — forces eviction/restore cycles
        for cycle in range(2):
            for _, factory, inst_name in tools:
                result = _run_tool(factory, inst_name)
                assert result.success, \
                    f"Cycle {cycle}, {inst_name} failed: {result.errors}"
                time.sleep(0.01)

        # All tools should still be tracked
        status = dm.get_device_status()
        for _, _, inst_name in tools:
            assert inst_name in status["allocations"], \
                f"{inst_name} should still be tracked after rapid cycling"

        # Exactly 2 tools on GPUs, 1 on CPU
        gpu_allocs = {
            name: alloc["device_id"]
            for name, alloc in status["allocations"].items()
            if alloc["device_id"].startswith("cuda:")
        }
        cpu_allocs = [
            name for name, alloc in status["allocations"].items()
            if alloc["device_id"] == "cpu"
        ]
        assert len(gpu_allocs) == 2, (
            f"Exactly 2 tools should be on GPUs (got {len(gpu_allocs)}: {gpu_allocs})"
        )
        assert len(cpu_allocs) == 1, (
            f"Exactly 1 tool should be on CPU (got {len(cpu_allocs)}: {cpu_allocs})"
        )

        # Assert memory on devices with persistent (non-CLI) tools.
        # CLI tools auto-unload so their device won't show model memory.
        persistent_gpu_devices = [
            alloc["device_id"] for name, alloc in status["allocations"].items()
            if alloc["device_id"].startswith("cuda:") and name not in cli_instances
        ]
        _assert_gpu_memory(dm, baseline, loaded=persistent_gpu_devices,
                           label="after rapid cycling")
    finally:
        _teardown()


# ── Multiple tools per device ───────────────────────────────────────────

@pytest.mark.uses_gpu(1)
@pytest.mark.slow
def test_two_tools_share_one_gpu():
    """Two tools on the same GPU should both have memory allocated."""
    dm = _setup_dm(managed_devices=["cuda:0"], strategy=OffloadStrategy.CPU)
    dm.configure(allow_multiple_per_device=True)

    try:
        baseline = _snapshot_gpu_memory(dm)

        with ToolInstance.persist_tool("mock_pytorch_tool", instance_name="pt"):
            result_pt = _run_tool(_pytorch_tool, "pt")
            assert result_pt.success

            status = dm.get_device_status()
            assert status["allocations"]["pt"]["device_id"] == "cuda:0"
            _assert_gpu_memory(dm, baseline, loaded=["cuda:0"],
                               label="first tool on cuda:0")

            # Snapshot after first tool is loaded
            after_one = _snapshot_gpu_memory(dm)

            with ToolInstance.persist_tool("mock_jax_tool", instance_name="jx"):
                result_jx = _run_tool(_jax_tool, "jx")
                assert result_jx.success

                status = dm.get_device_status()
                assert status["allocations"]["pt"]["device_id"] == "cuda:0"
                assert status["allocations"]["jx"]["device_id"] == "cuda:0"

                # Memory should be roughly double — second tool added more
                _assert_gpu_memory(dm, after_one, loaded=["cuda:0"],
                                   label="second tool also on cuda:0")
    finally:
        _teardown()
