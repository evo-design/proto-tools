"""tests/tool_infra_tests/test_tool_pool.py.

Tests for ToolPool parallel fan-out across devices.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest
from pydantic import Field

from proto_tools.tools.tool_registry import ToolRegistry
from proto_tools.utils import BaseConfig, ConfigField
from proto_tools.utils.tool_io import BaseToolInput
from proto_tools.utils.tool_pool import (
    DeviceCapability,
    PartialFailureError,
    ToolPool,
    WorkItem,
    _active_pool,
    _compute_worker_layout,
    _pool_executing,
    get_active_pool,
    is_pool_executing,
    lpt_schedule,
)
from tests.tool_infra_tests.test_export_functionality import MockToolOutputBase

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def clean_registry():
    """Provide a clean registry for each test."""
    original_registry = ToolRegistry._registry.copy()
    ToolRegistry._registry.clear()
    yield ToolRegistry
    ToolRegistry._registry = original_registry


# ── Mock models ─────────────────────────────────────────────────────────────


class MockInput(BaseToolInput):
    items: list[str] = Field(description="Items to process")


class MockConfig(BaseConfig):
    device: str = ConfigField(default="cuda", title="Device", description="Device to use")


class MockOutput(MockToolOutputBase):
    results: list[str] = Field(description="Processed results")


class MockNonIterableInput(BaseToolInput):
    query: str = Field(description="A single query")


class MockNonIterableOutput(MockToolOutputBase):
    answer: str = Field(description="The answer")


# ── Helpers ─────────────────────────────────────────────────────────────────


def _register_mock_tool(
    registry, key="mock-process", iterable_input_fields=("items",), iterable_output_field="results"
):
    """Register a mock tool and return (wrapper_function, call_log)."""
    call_log = []

    @registry.register(
        key=key,
        label="Mock Process",
        category="testing",
        input_class=MockInput,
        config_class=MockConfig,
        output_class=MockOutput,
        description="Mock tool for testing",
        iterable_input_fields=list(iterable_input_fields),
        iterable_output_field=iterable_output_field,
    )
    def run_mock_process(inputs, config=None, instance=None):
        call_log.append(
            {
                "items": list(inputs.items),
                "device": config.device if config else None,
                "instance": instance,
                "thread": threading.current_thread().name,
            }
        )
        return MockOutput(
            results=[f"processed_{item}" for item in inputs.items],
            tool_id=key,
            execution_time=0.01,
            success=True,
        )

    return run_mock_process, call_log


# ── LPT scheduling tests ───────────────────────────────────────────────────


def test_lpt_uniform_costs_round_robin():
    """Uniform costs should distribute items roughly evenly."""
    items = [WorkItem(i, f"item_{i}", cost=1.0) for i in range(6)]
    devices = [
        DeviceCapability("cuda:0"),
        DeviceCapability("cuda:1"),
    ]
    assignments = lpt_schedule(items, devices)
    assert len(assignments) == 2
    assert len(assignments[0].items) == 3
    assert len(assignments[1].items) == 3


def test_lpt_variable_costs_balances_load():
    """Large items should be placed on least-loaded devices."""
    items = [
        WorkItem(0, "big", cost=100.0),
        WorkItem(1, "medium", cost=50.0),
        WorkItem(2, "small1", cost=10.0),
        WorkItem(3, "small2", cost=10.0),
    ]
    devices = [
        DeviceCapability("cuda:0"),
        DeviceCapability("cuda:1"),
    ]
    assignments = lpt_schedule(items, devices)

    # big(100) should be on one device, medium(50)+small1(10)+small2(10) on the other
    costs = sorted([a.total_cost for a in assignments])
    assert costs == [70.0, 100.0]


def test_lpt_heterogeneous_throughput_weights():
    """Higher throughput_weight devices should get more work."""
    items = [WorkItem(i, f"item_{i}", cost=10.0) for i in range(6)]
    devices = [
        DeviceCapability("cuda:0", throughput_weight=2.0),  # 2x faster
        DeviceCapability("cuda:1", throughput_weight=1.0),
    ]
    assignments = lpt_schedule(items, devices)

    # Faster device (weight=2.0) gets more items
    fast = next(a for a in assignments if a.device.device_id == "cuda:0")
    slow = next(a for a in assignments if a.device.device_id == "cuda:1")
    assert len(fast.items) >= len(slow.items)


def test_lpt_max_item_cost_filtering():
    """Items exceeding max_item_cost should be routed to capable devices."""
    items = [
        WorkItem(0, "huge", cost=5000.0),
        WorkItem(1, "small", cost=10.0),
    ]
    devices = [
        DeviceCapability("cuda:0", max_item_cost=100.0),  # Can't handle huge
        DeviceCapability("cuda:1", max_item_cost=None),  # No limit
    ]
    assignments = lpt_schedule(items, devices)

    # huge should go to cuda:1 (no limit)
    d1 = next(a for a in assignments if a.device.device_id == "cuda:1")
    assert any(wi.item == "huge" for wi in d1.items)


def test_lpt_single_device():
    """All items go to the only device."""
    items = [WorkItem(i, f"item_{i}", cost=1.0) for i in range(5)]
    devices = [DeviceCapability("cuda:0")]
    assignments = lpt_schedule(items, devices)
    assert len(assignments) == 1
    assert len(assignments[0].items) == 5


def test_lpt_empty_items():
    """No items produces empty assignments."""
    devices = [DeviceCapability("cuda:0"), DeviceCapability("cuda:1")]
    assignments = lpt_schedule([], devices)
    assert all(len(a.items) == 0 for a in assignments)


def test_lpt_preserves_original_index():
    """Original indices are preserved through scheduling."""
    items = [WorkItem(42, "x", cost=1.0), WorkItem(7, "y", cost=2.0)]
    devices = [DeviceCapability("cuda:0")]
    assignments = lpt_schedule(items, devices)
    indices = {wi.original_index for wi in assignments[0].items}
    assert indices == {42, 7}


def test_lpt_no_eligible_device_fallback():
    """When no device can handle an item, falls back to least-loaded."""
    items = [WorkItem(0, "huge", cost=5000.0)]
    devices = [
        DeviceCapability("cuda:0", max_item_cost=100.0),
        DeviceCapability("cuda:1", max_item_cost=100.0),
    ]
    assignments = lpt_schedule(items, devices)
    total_items = sum(len(a.items) for a in assignments)
    assert total_items == 1  # Item still gets assigned


# ── ContextVar tests ────────────────────────────────────────────────────────


def test_contextvar_no_pool_by_default():
    assert get_active_pool() is None
    assert is_pool_executing() is False


def test_contextvar_pool_set_and_cleared():
    """Pool should be visible after set, gone after reset."""
    pool = ToolPool(gpus=["cuda:0"])
    token = _active_pool.set(pool)
    assert get_active_pool() is pool
    _active_pool.reset(token)
    assert get_active_pool() is None


def test_contextvar_pool_executing_prevents_recursion():
    """_pool_executing should be set inside worker threads."""
    token = _pool_executing.set(True)
    assert is_pool_executing() is True
    _pool_executing.reset(token)
    assert is_pool_executing() is False


def test_contextvar_propagates_to_worker_threads():
    """ContextVars from the main thread must be visible in pool worker threads."""
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    test_var = contextvars.ContextVar("test_var", default=False)
    test_var.set(True)

    # Without copy_context, ThreadPoolExecutor threads see the default
    with ThreadPoolExecutor(max_workers=1) as ex:
        assert ex.submit(test_var.get).result() is False

    # With copy_context (as ToolPool does), threads see the parent's value
    ctx = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as ex:
        assert ex.submit(ctx.run, test_var.get).result() is True


def test_contextvar_nesting_raises_error():
    """Nested ToolPool contexts should raise RuntimeError."""
    pool = ToolPool(gpus=["cuda:0"])
    token = _active_pool.set(pool)
    try:
        pool2 = ToolPool(gpus=["cuda:1"])
        with pytest.raises(RuntimeError, match="cannot be nested"):
            pool2.__enter__()
    finally:
        _active_pool.reset(token)


# ── gpus / cpus argument resolution ──────────────────────────────────────────


def test_gpus_int_takes_first_n_visible():
    """gpus=N resolves to the first N visible GPU device strings."""
    with patch("proto_tools.utils.tool_pool.number_of_available_gpus", return_value=4):
        pool = ToolPool(gpus=2, cpus=1)
        assert pool._resolve_gpus() == ["cuda:0", "cuda:1"]


def test_gpus_int_zero_skips_detection():
    """gpus=0 returns an empty list and never queries available GPUs."""
    with patch("proto_tools.utils.tool_pool.number_of_available_gpus") as mock_n:
        pool = ToolPool(gpus=0, cpus=1)
        assert pool._resolve_gpus() == []
        mock_n.assert_not_called()


def test_gpus_int_too_many_raises():
    """gpus=N where N > visible raises RuntimeError."""
    with patch("proto_tools.utils.tool_pool.number_of_available_gpus", return_value=2):
        pool = ToolPool(gpus=4, cpus=1)
        with pytest.raises(RuntimeError, match="requested gpus=4 but only 2 are visible"):
            pool._resolve_gpus()


def test_gpus_all_resolves_to_visible_set():
    """gpus='all' (default) returns every visible GPU."""
    with patch("proto_tools.utils.tool_pool.number_of_available_gpus", return_value=3):
        pool = ToolPool(cpus=1)  # gpus defaults to "all"
        assert pool._resolve_gpus() == ["cuda:0", "cuda:1", "cuda:2"]


def test_gpus_negative_raises():
    """gpus=-1 raises ValueError."""
    pool = ToolPool(gpus=-1, cpus=1)
    with pytest.raises(ValueError, match="gpus must be >= 0"):
        pool._resolve_gpus()


def test_cpus_negative_raises_at_init():
    """Passing cpus < 1 raises ValueError immediately at construction."""
    with pytest.raises(ValueError, match="cpus must be >= 1"):
        ToolPool(gpus=0, cpus=0)


def test_cpus_default_capped_at_four():
    """Default cpus resolves to min(_detect_cpus(), 4)."""
    with patch("proto_tools.utils.tool_pool._detect_cpus", return_value=64):
        pool = ToolPool(gpus=0)  # cpus=None → default
        assert pool.cpus == 4


def test_cpu_only_pool_enters_without_gpus():
    """gpus=0 enters cleanly on a 0-GPU host (no GPU validation runs)."""
    with patch("proto_tools.utils.tool_pool.number_of_available_gpus") as mock_n:
        pool = ToolPool(gpus=0, cpus=4)
        try:
            pool.__enter__()
            assert pool._gpu_devices == []
            assert pool.cpus == 4
            mock_n.assert_not_called()
        finally:
            pool.__exit__(None, None, None)


def test_no_parallelism_warning(caplog):
    """gpus=0 + cpus=1 logs a warning that the pool will short-circuit everything."""
    import logging

    with (
        patch("proto_tools.utils.tool_pool.number_of_available_gpus", return_value=0),
        caplog.at_level(logging.WARNING, logger="proto_tools.utils.tool_pool"),
    ):
        pool = ToolPool(gpus=0, cpus=1)
        try:
            pool.__enter__()
            assert any("no parallelism" in rec.message for rec in caplog.records)
        finally:
            pool.__exit__(None, None, None)


# ── Parallel dispatch tests ─────────────────────────────────────────────────


def test_dispatch_items_split_across_devices(clean_registry):
    """Items should be split across available devices."""
    func, _call_log = _register_mock_tool(clean_registry)

    pool = ToolPool(gpus=["cuda:0", "cuda:1"])
    pool._gpu_devices = ["cuda:0", "cuda:1"]

    inputs = MockInput(items=["a", "b", "c", "d"])
    config = MockConfig(device="cuda")

    result = pool._parallel_dispatch("mock-process", func, inputs, config)

    assert len(result.results) == 4
    assert result.results == ["processed_a", "processed_b", "processed_c", "processed_d"]


def test_dispatch_results_reassembled_in_order(clean_registry):
    """Output items must match original input order."""
    func, _call_log = _register_mock_tool(clean_registry)

    pool = ToolPool(gpus=["cuda:0", "cuda:1", "cuda:2"])
    pool._gpu_devices = ["cuda:0", "cuda:1", "cuda:2"]

    items = [f"item_{i}" for i in range(10)]
    inputs = MockInput(items=items)
    config = MockConfig(device="cuda")

    result = pool._parallel_dispatch("mock-process", func, inputs, config)

    expected = [f"processed_item_{i}" for i in range(10)]
    assert result.results == expected


def test_dispatch_config_device_overridden_per_worker(clean_registry):
    """Each worker should get config.device set to its assigned device."""
    func, call_log = _register_mock_tool(clean_registry)

    pool = ToolPool(gpus=["cuda:0", "cuda:1"])
    pool._gpu_devices = ["cuda:0", "cuda:1"]

    inputs = MockInput(items=["a", "b", "c", "d"])
    config = MockConfig(device="cuda")

    pool._parallel_dispatch("mock-process", func, inputs, config)

    devices_used = {call["device"] for call in call_log}
    assert devices_used == {"cuda:0", "cuda:1"}


def test_dispatch_worker_instance_names(clean_registry):
    """Worker instance names should encode the device."""
    func, call_log = _register_mock_tool(clean_registry)

    pool = ToolPool(gpus=["cuda:0", "cuda:1"])
    pool._gpu_devices = ["cuda:0", "cuda:1"]

    inputs = MockInput(items=["a", "b"])
    config = MockConfig(device="cuda")

    pool._parallel_dispatch("mock-process", func, inputs, config)

    instances = {call["instance"] for call in call_log}
    assert "mock-process-pool-cuda_0" in instances
    assert "mock-process-pool-cuda_1" in instances


def test_dispatch_single_item_skips_pool(clean_registry):
    """Single-item inputs should bypass pool overhead."""
    func, call_log = _register_mock_tool(clean_registry)

    pool = ToolPool(gpus=["cuda:0", "cuda:1"])
    pool._gpu_devices = ["cuda:0", "cuda:1"]

    inputs = MockInput(items=["only_one"])
    config = MockConfig(device="cuda")

    result = pool._parallel_dispatch("mock-process", func, inputs, config)

    assert result.results == ["processed_only_one"]
    assert len(call_log) == 1
    assert call_log[0]["instance"] is None  # Direct call, no worker name


def test_dispatch_gpus_per_instance_grouping(clean_registry):
    """Multi-GPU tools should group devices."""
    _, call_log = _register_mock_tool(clean_registry)

    class MultiGPUConfig(BaseConfig):
        device: str = ConfigField(default="cuda", title="Device", description="Device to use")

        @property
        def gpus_per_instance(self) -> int:
            return 2

    # Re-register with MultiGPUConfig
    clean_registry._registry.clear()

    @clean_registry.register(
        key="multi-gpu-process",
        label="Multi GPU",
        category="testing",
        input_class=MockInput,
        config_class=MultiGPUConfig,
        output_class=MockOutput,
        description="Multi-GPU test",
        iterable_input_fields=["items"],
        iterable_output_field="results",
    )
    def run_multi(inputs, config=None, instance=None):
        call_log.append({"device": config.device, "instance": instance})
        return MockOutput(
            results=[f"processed_{item}" for item in inputs.items],
            tool_id="multi-gpu-process",
            execution_time=0.01,
            success=True,
        )

    pool = ToolPool(gpus=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    pool._gpu_devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]

    inputs = MockInput(items=["a", "b", "c", "d"])
    config = MultiGPUConfig(device="cuda")

    result = pool._parallel_dispatch("multi-gpu-process", run_multi, inputs, config)

    # 4 GPUs / 2 per instance = 2 workers
    assert len(call_log) == 2
    devices_used = {call["device"] for call in call_log}
    assert "cuda:0,cuda:1" in devices_used
    assert "cuda:2,cuda:3" in devices_used
    assert len(result.results) == 4


def test_dispatch_gpus_per_instance_zero_bypasses_pool(clean_registry):
    """CPU-opt-out tools (gpus_per_instance=0 + cpus_per_instance=None) bypass partitioning.

    Regression: ToolPool._parallel_dispatch previously crashed with
    `ValueError: range() arg 3 must not be zero` when a tool's config returned
    gpus_per_instance=0 (e.g. mmseqs2-homology-search with use_gpu=False) and the
    pool was dispatched with >=2 input items. The fix short-circuits to a
    single direct call mirroring the n_items<=1 path. With the cpus_per_instance
    default of 1, the explicit None opt-out (mirroring mmseqs2-* in production)
    is required to reach the short-circuit branch.
    """
    _, call_log = _register_mock_tool(clean_registry)

    class CPUOnlyConfig(BaseConfig):
        device: str = ConfigField(default="cuda", title="Device", description="Device to use")

        @property
        def gpus_per_instance(self) -> int:
            return 0

        @property
        def cpus_per_instance(self) -> int | None:
            """Opt out of CPU fan-out — internally threaded (same as mmseqs2-homology-search)."""
            return None

    clean_registry._registry.clear()

    @clean_registry.register(
        key="cpu-only-process",
        label="CPU Only",
        category="testing",
        input_class=MockInput,
        config_class=CPUOnlyConfig,
        output_class=MockOutput,
        description="CPU-only test",
        iterable_input_fields=["items"],
        iterable_output_field="results",
    )
    def run_cpu_only(inputs, config=None, instance=None):
        call_log.append({"device": config.device, "instance": instance, "items": list(inputs.items)})
        return MockOutput(
            results=[f"processed_{item}" for item in inputs.items],
            tool_id="cpu-only-process",
            execution_time=0.01,
            success=True,
        )

    pool = ToolPool(gpus=["cuda:0", "cuda:1"])
    pool._gpu_devices = ["cuda:0", "cuda:1"]

    inputs = MockInput(items=["a", "b", "c"])
    config = CPUOnlyConfig()

    # Must not raise ValueError("range() arg 3 must not be zero")
    result = pool._parallel_dispatch("cpu-only-process", run_cpu_only, inputs, config)

    # Single direct call with all items; no GPU partitioning, no per-device worker name
    assert len(call_log) == 1
    assert call_log[0]["instance"] is None
    assert call_log[0]["items"] == ["a", "b", "c"]
    assert result.results == ["processed_a", "processed_b", "processed_c"]


def test_dispatch_pool_receives_pre_deduped_items(clean_registry):
    """Pool should receive already-deduped items from @tool wrapper."""
    func, call_log = _register_mock_tool(clean_registry)

    pool = ToolPool(gpus=["cuda:0", "cuda:1"])
    pool._gpu_devices = ["cuda:0", "cuda:1"]

    inputs = MockInput(items=["a", "b", "c"])
    config = MockConfig(device="cuda")

    result = pool._parallel_dispatch("mock-process", func, inputs, config)

    assert len(result.results) == 3
    all_items = []
    for call in call_log:
        all_items.extend(call["items"])
    assert sorted(all_items) == ["a", "b", "c"]


def test_dispatch_collects_warnings_and_errors(clean_registry):
    """Merged output should collect warnings/errors from all partitions."""

    @clean_registry.register(
        key="warn-process",
        label="Warn Process",
        category="testing",
        input_class=MockInput,
        config_class=MockConfig,
        output_class=MockOutput,
        description="Mock tool that produces warnings and errors",
        iterable_input_fields=["items"],
        iterable_output_field="results",
    )
    def run_with_warnings(inputs, config=None, instance=None):
        items = inputs.items
        results = [f"processed_{item}" for item in items]
        warnings = [f"warn_{item}" for item in items]
        errors = [f"err_{item}" for item in items if item > "c"]
        return MockOutput(
            results=results,
            warnings=warnings,
            errors=errors,
            execution_time=0.01,
            success=True,
        )

    pool = ToolPool(gpus=["cuda:0", "cuda:1"])
    pool._gpu_devices = ["cuda:0", "cuda:1"]

    inputs = MockInput(items=["a", "b", "c", "d"])
    config = MockConfig(device="cuda")

    result = pool._parallel_dispatch("warn-process", run_with_warnings, inputs, config)

    assert len(result.results) == 4
    assert len(result.warnings) == 4
    assert set(result.warnings) == {"warn_a", "warn_b", "warn_c", "warn_d"}
    assert len(result.errors) == 1
    assert result.errors == ["err_d"]


def test_dispatch_persistence_entered():
    """Pool still enters ToolInstance.persist()."""
    with patch("proto_tools.utils.tool_instance.ToolInstance.persist") as mock_persist:
        mock_ctx = MagicMock()
        mock_persist.return_value = mock_ctx
        pool = ToolPool(gpus=["cuda:0"])
        # Patch GPU validation
        with patch("proto_tools.utils.tool_pool.determine_visible_devices"):
            pool.__enter__()
        try:
            assert pool._persist_ctx is mock_ctx
            mock_ctx.__enter__.assert_called_once()
        finally:
            pool.__exit__(None, None, None)


# ── Tool interception tests ─────────────────────────────────────────────────


def test_interception_non_iterable_tool_not_intercepted(clean_registry):
    """Tools without iterable fields should not be intercepted by pool."""

    @clean_registry.register(
        key="scalar-tool",
        label="Scalar Tool",
        category="testing",
        input_class=MockNonIterableInput,
        config_class=MockConfig,
        output_class=MockNonIterableOutput,
        description="Non-iterable tool",
    )
    def run_scalar(inputs, config=None, instance=None):
        return MockNonIterableOutput(
            answer=f"answer_{inputs.query}",
            tool_id="scalar-tool",
            execution_time=0.01,
            success=True,
        )

    pool = MagicMock(spec=ToolPool)
    token = _active_pool.set(pool)
    try:
        inputs = MockNonIterableInput(query="hello")
        result = run_scalar(inputs, MockConfig())

        pool._parallel_dispatch.assert_not_called()
        assert result.answer == "answer_hello"
    finally:
        _active_pool.reset(token)


def test_interception_iterable_tool_intercepted_by_pool(clean_registry):
    """Tools with iterable fields should be routed through pool."""

    @clean_registry.register(
        key="iterable-tool",
        label="Iterable Tool",
        category="testing",
        input_class=MockInput,
        config_class=MockConfig,
        output_class=MockOutput,
        description="Iterable tool",
        iterable_input_fields=["items"],
        iterable_output_field="results",
    )
    def run_iterable(inputs, config=None, instance=None):
        return MockOutput(
            results=[f"r_{x}" for x in inputs.items],
            tool_id="iterable-tool",
            execution_time=0.01,
            success=True,
        )

    mock_output = MockOutput.model_construct(
        results=["pooled_a", "pooled_b"],
        tool_id="iterable-tool",
        success=True,
    )
    pool = MagicMock(spec=ToolPool)
    pool._parallel_dispatch.return_value = mock_output

    token = _active_pool.set(pool)
    try:
        inputs = MockInput(items=["a", "b"])
        result = run_iterable(inputs, MockConfig())

        pool._parallel_dispatch.assert_called_once()
        assert result.results == ["pooled_a", "pooled_b"]
    finally:
        _active_pool.reset(token)


def test_interception_pool_executing_prevents_re_interception(clean_registry):
    """When _pool_executing is True, tool calls should not re-route."""

    @clean_registry.register(
        key="inner-tool",
        label="Inner Tool",
        category="testing",
        input_class=MockInput,
        config_class=MockConfig,
        output_class=MockOutput,
        description="Inner tool call",
        iterable_input_fields=["items"],
        iterable_output_field="results",
    )
    def run_inner(inputs, config=None, instance=None):
        return MockOutput(
            results=[f"direct_{x}" for x in inputs.items],
            tool_id="inner-tool",
            execution_time=0.01,
            success=True,
        )

    pool = MagicMock(spec=ToolPool)
    pool_token = _active_pool.set(pool)
    exec_token = _pool_executing.set(True)
    try:
        inputs = MockInput(items=["a"])
        result = run_inner(inputs, MockConfig())

        pool._parallel_dispatch.assert_not_called()
        assert result.results == ["direct_a"]
    finally:
        _pool_executing.reset(exec_token)
        _active_pool.reset(pool_token)


# ── ToolSpec integration tests ──────────────────────────────────────────────


def test_toolspec_iterable_fields_stored(clean_registry):
    """iterable_input_fields and iterable_output_field should be on ToolSpec."""

    @clean_registry.register(
        key="spec-test",
        label="Spec Test",
        category="testing",
        input_class=MockInput,
        config_class=MockConfig,
        output_class=MockOutput,
        description="Test",
        iterable_input_fields=["items"],
        iterable_output_field="results",
    )
    def run_spec_test(inputs, config=None, instance=None):
        return MockOutput(results=[], tool_id="spec-test", success=True)

    spec = clean_registry.get("spec-test")
    assert spec.iterable_input_fields == ["items"]
    assert spec.iterable_output_field == "results"


def test_toolspec_iterable_input_fields_parallel_group(clean_registry):
    """iterable_input_fields stores the parallel group (primary first)."""

    @clean_registry.register(
        key="parallel-spec-test",
        label="Parallel",
        category="testing",
        input_class=MockInput,
        config_class=MockConfig,
        output_class=MockOutput,
        description="Test",
        iterable_input_fields=["items", "tags"],
        iterable_output_field="results",
    )
    def run_parallel(inputs, config=None, instance=None):
        return MockOutput(results=[], tool_id="parallel-spec-test", success=True)

    spec = clean_registry.get("parallel-spec-test")
    assert spec.iterable_input_fields == ["items", "tags"]


def test_parallel_item_cache_key_folds_in_siblings():
    """A parallel-item bundle's key reflects all fields; a 1-tuple matches the bare item (no cache churn)."""
    from proto_tools.tools.tool_registry import _ParallelItem
    from proto_tools.utils.tool_cache import _serialize_for_cache_key

    # Single-field bundle keys identically to the bare item → existing cache entries still hit.
    assert _ParallelItem(("seqA",)).cache_key() == _serialize_for_cache_key("seqA")
    # Same primary, different sibling → distinct key (the dedup/cache-collision fix).
    assert _ParallelItem(("seqA", "msaX")).cache_key() != _ParallelItem(("seqA", "msaY")).cache_key()
    # Identical primary + sibling → identical key.
    assert _ParallelItem(("seqA", "msaX")).cache_key() == _ParallelItem(("seqA", "msaX")).cache_key()


def test_zip_apply_iter_items_keeps_parallel_fields_aligned():
    """Slicing/reordering the parallel group moves every field together (the alignment fix)."""
    from proto_tools.tools.tool_registry import _apply_iter_items, _zip_iter_items
    from proto_tools.utils import BaseToolInput, InputField

    class _PInput(BaseToolInput):
        items: list[str] = InputField(default_factory=list, title="Items", description="primary list")
        tags: list[str] | None = InputField(default=None, title="Tags", description="parallel sibling list")

    inp = _PInput(items=["a", "b", "c"], tags=["1", "2", "3"])
    fields = ["items", "tags"]
    bundles = _zip_iter_items(inp, fields)
    # Reorder + subset to original indices [2, 0]; both fields must follow in lockstep.
    out = _apply_iter_items(inp, fields, [bundles[2], bundles[0]])
    assert out.items == ["c", "a"]
    assert out.tags == ["3", "1"]


def test_zip_iter_items_rejects_misaligned_parallel_fields():
    """A sibling whose length differs from the primary raises a clear ValueError."""
    from proto_tools.tools.tool_registry import _zip_iter_items
    from proto_tools.utils import BaseToolInput, InputField

    class _PInput(BaseToolInput):
        items: list[str] = InputField(default_factory=list, title="Items", description="primary list")
        tags: list[str] = InputField(default_factory=list, title="Tags", description="parallel sibling list")

    with pytest.raises(ValueError, match="not aligned"):
        _zip_iter_items(_PInput(items=["a", "b", "c"], tags=["1"]), ["items", "tags"])  # too short
    with pytest.raises(ValueError, match="not aligned"):
        _zip_iter_items(_PInput(items=["a"], tags=["1", "2"]), ["items", "tags"])  # too long


def test_parallel_group_dedup_keys_and_aligns_through_wrapper(clean_registry):
    """A cacheable parallel-group iterable tool dedups on the full group through the @tool wrapper.

    The sibling participates in the dedup key, stays aligned to the primary, and results expand
    back to original positions.
    """

    class _MPInput(BaseToolInput):
        items: list[str] = Field(description="primary list")
        tags: list[str] = Field(description="index-parallel sibling list")

    class _MPOutput(MockToolOutputBase):
        results: list[str] = Field(description="one result per row")

    seen: list[tuple[list[str], list[str]]] = []

    @clean_registry.register(
        key="parallel-dedup",
        label="Parallel Dedup",
        category="testing",
        input_class=_MPInput,
        config_class=MockConfig,
        output_class=_MPOutput,
        description="Records the (items, tags) rows it actually receives post-dedup",
        iterable_input_fields=["items", "tags"],
        iterable_output_field="results",
        cacheable=True,
    )
    def run_parallel_dedup(inputs, config=None, instance=None):
        seen.append((list(inputs.items), list(inputs.tags)))
        return _MPOutput(
            results=[f"{i}-{t}" for i, t in zip(inputs.items, inputs.tags, strict=True)],
            tool_id="parallel-dedup",
            success=True,
        )

    spec = clean_registry.get("parallel-dedup")
    # Rows (a,x), (a,y), (a,x): the two (a,x) are true duplicates; (a,y) shares the primary
    # but differs in the sibling, so it must NOT be collapsed.
    out = spec.function(_MPInput(items=["a", "a", "a"], tags=["x", "y", "x"]), MockConfig(device="cpu"))

    # The tool saw exactly the 2 unique rows, sibling aligned to primary (items-only dedup would
    # have collapsed all three to one row and lost the (a,y) distinction).
    assert seen == [(["a", "a"], ["x", "y"])]
    # Output expanded back to all 3 original positions, in order.
    assert out.results == ["a-x", "a-y", "a-x"]


def test_toolspec_iterable_fields_default_to_none(clean_registry):
    """Tools without iterable fields should have None."""

    @clean_registry.register(
        key="no-iter-test",
        label="No Iter",
        category="testing",
        input_class=MockNonIterableInput,
        config_class=MockConfig,
        output_class=MockNonIterableOutput,
        description="Test",
    )
    def run_no_iter(inputs, config=None, instance=None):
        return MockNonIterableOutput(answer="x", tool_id="no-iter-test", success=True)

    spec = clean_registry.get("no-iter-test")
    assert spec.iterable_input_fields is None
    assert spec.iterable_output_field is None


def test_toolspec_iterable_fields_excluded_from_serialization(clean_registry):
    """Iterable fields should be excluded from API serialization."""

    @clean_registry.register(
        key="serial-test",
        label="Serial Test",
        category="testing",
        input_class=MockInput,
        config_class=MockConfig,
        output_class=MockOutput,
        description="Test",
        iterable_input_fields=["items"],
        iterable_output_field="results",
    )
    def run_serial(inputs, config=None, instance=None):
        return MockOutput(results=[], tool_id="serial-test", success=True)

    spec = clean_registry.get("serial-test")
    serialized = spec.model_dump()
    assert "iterable_input_fields" not in serialized
    assert "iterable_output_field" not in serialized


# ── CPU fan-out tests ───────────────────────────────────────────────────────


def test_compute_worker_layout_gpu_path():
    """GPU path: groups devices into gpus_per_instance-sized slots."""

    class GpuConfig(BaseConfig):
        device: str = ConfigField(default="cuda", title="Device", description="Device to use")

        @property
        def gpus_per_instance(self) -> int:
            return 2

    slots = _compute_worker_layout(
        "tool",
        GpuConfig(),
        gpu_devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        cpus_budget=4,
    )
    assert len(slots) == 2
    assert slots[0].device_id == "cuda:0,cuda:1"
    assert slots[0].device_override == "cuda:0,cuda:1"
    assert slots[0].worker_name == "tool-pool-cuda_0_cuda_1"
    assert slots[0].env_overrides is None
    assert slots[1].device_id == "cuda:2,cuda:3"


def test_compute_worker_layout_cpu_short_circuit():
    """Default cpus_per_instance is None → empty slots (single direct call)."""
    cfg = BaseConfig(device="cpu")  # default cpus_per_instance is None
    slots = _compute_worker_layout("tool", cfg, gpu_devices=[], cpus_budget=8)
    assert slots == []


def test_compute_worker_layout_cpu_fanout():
    """gpus_per_instance==0 + cpus_per_instance==N → N CPU worker slots."""

    class CpuFanoutConfig(BaseConfig):
        device: str = ConfigField(default="cpu", title="Device", description="Device to use")

        @property
        def cpus_per_instance(self) -> int | None:
            return 1

    slots = _compute_worker_layout("pyrosetta-relax", CpuFanoutConfig(), gpu_devices=[], cpus_budget=4)
    assert len(slots) == 4
    for i, slot in enumerate(slots):
        assert slot.device_id == f"cpu#{i}"
        assert slot.device_override == "cpu"
        assert slot.worker_name == f"pyrosetta-relax-pool-cpu-{i}"
        assert slot.env_overrides == {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }


def test_compute_worker_layout_cpu_fanout_with_thread_budget():
    """cpus_per_instance=2 with cpus=8 → 4 workers, each gets OMP=2."""

    class CpuMultiThreadConfig(BaseConfig):
        device: str = ConfigField(default="cpu", title="Device", description="Device to use")

        @property
        def cpus_per_instance(self) -> int | None:
            return 2

    slots = _compute_worker_layout("tool", CpuMultiThreadConfig(), gpu_devices=[], cpus_budget=8)
    assert len(slots) == 4
    assert all(
        slot.env_overrides
        == {  # type: ignore[comparison-overlap]
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
        }
        for slot in slots
    )


def test_dispatch_cpu_fanout_partitions_items(clean_registry):
    """CPU fan-out: cpus_per_instance=1 + cpus=4 → 4 partitions."""
    _, call_log = _register_mock_tool(clean_registry)

    class CpuFanoutConfig(BaseConfig):
        device: str = ConfigField(default="cpu", title="Device", description="Device to use")

        @property
        def cpus_per_instance(self) -> int | None:
            return 1

    clean_registry._registry.clear()

    @clean_registry.register(
        key="cpu-fanout-process",
        label="CPU Fanout",
        category="testing",
        input_class=MockInput,
        config_class=CpuFanoutConfig,
        output_class=MockOutput,
        description="CPU fan-out test",
        iterable_input_fields=["items"],
        iterable_output_field="results",
    )
    def run_cpu_fanout(inputs, config=None, instance=None):
        call_log.append(
            {
                "items": list(inputs.items),
                "device": config.device,
                "instance": instance,
                "thread": threading.current_thread().name,
            }
        )
        return MockOutput(
            results=[f"processed_{item}" for item in inputs.items],
            tool_id="cpu-fanout-process",
            execution_time=0.01,
            success=True,
        )

    pool = ToolPool(gpus=0, cpus=4)
    pool._gpu_devices = []  # __enter__ would set this; we're calling _parallel_dispatch directly

    inputs = MockInput(items=["a", "b", "c", "d", "e", "f", "g", "h"])
    config = CpuFanoutConfig()

    with patch("proto_tools.utils.tool_pool.ToolInstance.get") as mock_get:
        result = pool._parallel_dispatch("cpu-fanout-process", run_cpu_fanout, inputs, config)

    # Four parallel calls, each with 2 items, each on device="cpu"
    assert len(call_log) == 4
    assert all(c["device"] == "cpu" for c in call_log)
    assert all(c["instance"] is not None and c["instance"].startswith("cpu-fanout-process-pool-cpu-") for c in call_log)
    # 8 items distributed evenly
    items_per_call = sorted(len(c["items"]) for c in call_log)
    assert items_per_call == [2, 2, 2, 2]
    # Pre-create-with-env path: ToolInstance.get was called once per partition with env_overrides
    assert mock_get.call_count == 4
    for call in mock_get.call_args_list:
        kwargs = call.kwargs
        assert kwargs["env_overrides"]["OMP_NUM_THREADS"] == "1"
    # All 8 results returned in original order
    assert result.results == [f"processed_{item}" for item in ["a", "b", "c", "d", "e", "f", "g", "h"]]


def test_dispatch_cpu_short_circuit_preserved(clean_registry):
    """Tools that explicitly opt out via cpus_per_instance=None short-circuit to a single call."""
    _, call_log = _register_mock_tool(clean_registry)

    class CpuOptOutConfig(BaseConfig):
        device: str = ConfigField(default="cpu", title="Device", description="Device to use")

        @property
        def cpus_per_instance(self) -> int | None:
            """Opt out — internal threading."""
            return None

    clean_registry._registry.clear()

    @clean_registry.register(
        key="mmseqs-style",
        label="Mmseqs Style",
        category="testing",
        input_class=MockInput,
        config_class=CpuOptOutConfig,
        output_class=MockOutput,
        description="Internal-threading CPU tool",
        iterable_input_fields=["items"],
        iterable_output_field="results",
    )
    def run_mmseqs_style(inputs, config=None, instance=None):
        call_log.append({"items": list(inputs.items), "instance": instance})
        return MockOutput(
            results=[f"processed_{item}" for item in inputs.items],
            tool_id="mmseqs-style",
            execution_time=0.01,
            success=True,
        )

    pool = ToolPool(gpus=0, cpus=8)
    pool._gpu_devices = []

    inputs = MockInput(items=["a", "b", "c", "d"])
    config = CpuOptOutConfig()
    result = pool._parallel_dispatch("mmseqs-style", run_mmseqs_style, inputs, config)

    # Single direct call with all items — no fan-out, no per-worker instance
    assert len(call_log) == 1
    assert call_log[0]["instance"] is None
    assert call_log[0]["items"] == ["a", "b", "c", "d"]
    assert result.results == ["processed_a", "processed_b", "processed_c", "processed_d"]


# ── gpus_per_instance tests ──────────────────────────────────────────────


def test_gpus_per_instance_derived_from_device_string():
    """Default gpus_per_instance is derived from the device field via parse_device_string.

    - cpu → 0 (no pool partitioning, single direct call)
    - cuda / cuda:N → 1
    - cudaxN / cuda:0,cuda:1 → N
    - cloud → 1 (cloud dispatch is handled before pool partitioning)
    """
    assert BaseConfig().gpus_per_instance == 0  # default device='cpu'
    assert BaseConfig(device="cpu").gpus_per_instance == 0
    assert BaseConfig(device="cuda").gpus_per_instance == 1
    assert BaseConfig(device="cuda:0").gpus_per_instance == 1
    assert BaseConfig(device="cudax2").gpus_per_instance == 2
    assert BaseConfig(device="cudax4").gpus_per_instance == 4
    assert BaseConfig(device="cuda:0,cuda:1").gpus_per_instance == 2
    assert BaseConfig(device="cloud").gpus_per_instance == 1


def test_gpus_per_instance_override():
    """Subclasses can override gpus_per_instance based on config values."""

    class MultiGPU(BaseConfig):
        model_name: str = ConfigField(default="small", title="Model Name", description="Model name")

        @property
        def gpus_per_instance(self) -> int:
            return 2 if self.model_name == "large" else 1

    assert MultiGPU(model_name="small").gpus_per_instance == 1
    assert MultiGPU(model_name="large").gpus_per_instance == 2


# ── Error propagation tests ─────────────────────────────────────────────────


def test_local_partition_failure_preserves_other_results(clean_registry):
    """When one device fails, succeeded results and failure info are on the exception."""
    call_count = 0

    @clean_registry.register(
        key="fail-tool",
        label="Fail Tool",
        category="testing",
        input_class=MockInput,
        config_class=MockConfig,
        output_class=MockOutput,
        description="Tool that fails on second device",
        iterable_input_fields=["items"],
        iterable_output_field="results",
    )
    def run_fail_tool(inputs, config=None, instance=None):
        nonlocal call_count
        call_count += 1
        # Fail on cuda:1
        if config and "cuda_1" in (instance or ""):
            raise RuntimeError("OOM on cuda:1")
        return MockOutput(
            results=[f"ok_{item}" for item in inputs.items],
            tool_id="fail-tool",
            execution_time=0.01,
            success=True,
        )

    pool = ToolPool(gpus=["cuda:0", "cuda:1"])
    pool._gpu_devices = ["cuda:0", "cuda:1"]

    inputs = MockInput(items=["a", "b", "c", "d"])
    config = MockConfig(device="cuda")

    with pytest.raises(PartialFailureError, match=r"1/2 partition\(s\) failed") as exc_info:
        pool._parallel_dispatch("fail-tool", run_fail_tool, inputs, config)

    err = exc_info.value
    # Some items succeeded
    assert len(err.succeeded) > 0
    # One partition failed
    assert len(err.failed) == 1
    assert err.failed[0]["device_id"] == "cuda:1"
    assert isinstance(err.failed[0]["exception"], Exception)


def test_all_partitions_fail(clean_registry):
    """When all partitions fail, succeeded is empty and all failures are captured."""

    @clean_registry.register(
        key="all-fail-tool",
        label="All Fail Tool",
        category="testing",
        input_class=MockInput,
        config_class=MockConfig,
        output_class=MockOutput,
        description="Tool that always fails",
        iterable_input_fields=["items"],
        iterable_output_field="results",
    )
    def run_all_fail(inputs, config=None, instance=None):
        raise RuntimeError("everything is broken")

    pool = ToolPool(gpus=["cuda:0", "cuda:1"])
    pool._gpu_devices = ["cuda:0", "cuda:1"]

    inputs = MockInput(items=["a", "b", "c", "d"])
    config = MockConfig(device="cuda")

    with pytest.raises(PartialFailureError, match=r"2 partition.*failed") as exc_info:
        pool._parallel_dispatch("all-fail-tool", run_all_fail, inputs, config)

    err = exc_info.value
    assert err.succeeded == []
    assert len(err.failed) == 2
    failed_devices = {f["device_id"] for f in err.failed}
    assert failed_devices == {"cuda:0", "cuda:1"}


def test_all_succeed_unchanged(clean_registry):
    """Regression: happy path is unaffected by error propagation changes."""
    func, call_log = _register_mock_tool(clean_registry)

    pool = ToolPool(gpus=["cuda:0", "cuda:1"])
    pool._gpu_devices = ["cuda:0", "cuda:1"]

    inputs = MockInput(items=["a", "b", "c", "d"])
    config = MockConfig(device="cuda")

    result = pool._parallel_dispatch("mock-process", func, inputs, config)

    assert result.results == ["processed_a", "processed_b", "processed_c", "processed_d"]
    assert len(call_log) == 2  # Two partitions


# ---------------------------------------------------------------------------
# Integration tests (require GPU)
# ---------------------------------------------------------------------------


@pytest.mark.uses_gpu(2)
@pytest.mark.slow
def test_gpu_fanout_items_land_on_different_gpus():
    """Items should be dispatched to different physical GPUs."""
    from proto_tools.tools.testing.mock_pytorch_tool import (
        MockPyTorchToolConfig,
        MockPyTorchToolInput,
        run_mock_pytorch_tool,
    )
    from proto_tools.utils.device_manager import DeviceManager
    from proto_tools.utils.tool_instance import ToolInstance

    DeviceManager.reset_instance()
    ToolInstance.clear_all()

    try:
        data_items = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        with ToolPool(gpus=["cuda:0", "cuda:1"]):
            result = run_mock_pytorch_tool(
                MockPyTorchToolInput(data_items=data_items),
                MockPyTorchToolConfig(memory_mb=64),
            )

        assert result.success, f"ToolPool call failed: {result.errors}"
        assert len(result.results) == 4

        devices_used = {r.device_used for r in result.results}
        assert "cuda:0" in devices_used, "cuda:0 should have been used"
        assert "cuda:1" in devices_used, "cuda:1 should have been used"
    finally:
        ToolInstance.clear_all()
        DeviceManager.reset_instance()


@pytest.mark.uses_gpu(2)
@pytest.mark.slow
def test_gpu_fanout_results_in_original_order():
    """Results must be reassembled in original input order."""
    from proto_tools.tools.testing.mock_pytorch_tool import (
        MockPyTorchToolConfig,
        MockPyTorchToolInput,
        run_mock_pytorch_tool,
    )
    from proto_tools.utils.device_manager import DeviceManager
    from proto_tools.utils.tool_instance import ToolInstance

    DeviceManager.reset_instance()
    ToolInstance.clear_all()

    try:
        data_items = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 6.0, 0.0, 0.0],
        ]

        with ToolPool(gpus=["cuda:0", "cuda:1"]):
            result = run_mock_pytorch_tool(
                MockPyTorchToolInput(data_items=data_items),
                MockPyTorchToolConfig(memory_mb=64),
            )

        assert result.success
        assert len(result.results) == 6

        # With 6 items and 2 GPUs (LPT round-robin on uniform costs),
        # items [0,2,4] go to one GPU and [1,3,5] go to the other.
        devices = [r.device_used for r in result.results]
        assert devices[0] == devices[2] == devices[4], f"Items 0,2,4 should share a GPU: {devices}"
        assert devices[1] == devices[3] == devices[5], f"Items 1,3,5 should share a GPU: {devices}"
        assert devices[0] != devices[1], f"The two groups should be on different GPUs: {devices}"
    finally:
        ToolInstance.clear_all()
        DeviceManager.reset_instance()


@pytest.mark.uses_gpu(2)
@pytest.mark.slow
def test_gpu_fanout_persistence_across_pool_calls():
    """Workers should persist across calls within the same pool context.

    The first call pays model loading cost; the second call should be
    faster because workers are already warm.
    """
    import time

    from proto_tools.tools.testing.mock_pytorch_tool import (
        MockPyTorchToolConfig,
        MockPyTorchToolInput,
        run_mock_pytorch_tool,
    )
    from proto_tools.utils.device_manager import DeviceManager
    from proto_tools.utils.tool_instance import ToolInstance

    DeviceManager.reset_instance()
    ToolInstance.clear_all()

    try:
        data_items = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]

        with ToolPool(gpus=["cuda:0", "cuda:1"]):
            # Cold call: pays model loading on both GPUs
            t0 = time.time()
            result1 = run_mock_pytorch_tool(
                MockPyTorchToolInput(data_items=data_items),
                MockPyTorchToolConfig(memory_mb=64),
            )
            cold_time = time.time() - t0

            # Warm call: workers already loaded
            t0 = time.time()
            result2 = run_mock_pytorch_tool(
                MockPyTorchToolInput(data_items=data_items),
                MockPyTorchToolConfig(memory_mb=64),
            )
            warm_time = time.time() - t0

        assert result1.success
        assert result2.success
        assert warm_time < cold_time, (
            f"Warm call ({warm_time:.2f}s) should be faster than cold call ({cold_time:.2f}s), workers should persist"
        )
    finally:
        ToolInstance.clear_all()
        DeviceManager.reset_instance()


@pytest.mark.uses_gpu(2)
@pytest.mark.slow
def test_gpu_fanout_single_item_bypasses_pool():
    """A single-item input should bypass pool overhead."""
    from proto_tools.tools.testing.mock_pytorch_tool import (
        MockPyTorchToolConfig,
        MockPyTorchToolInput,
        run_mock_pytorch_tool,
    )
    from proto_tools.utils.device_manager import DeviceManager
    from proto_tools.utils.tool_instance import ToolInstance

    DeviceManager.reset_instance()
    ToolInstance.clear_all()

    try:
        with ToolPool(gpus=["cuda:0", "cuda:1"]):
            result = run_mock_pytorch_tool(
                MockPyTorchToolInput(data_items=[[1.0, 2.0, 3.0, 4.0]]),
                MockPyTorchToolConfig(memory_mb=64),
            )

        assert result.success
        assert len(result.results) == 1
    finally:
        ToolInstance.clear_all()
        DeviceManager.reset_instance()


# ---------------------------------------------------------------------------
# Integration tests (CPU fan-out — real subprocesses, no GPU required)
# ---------------------------------------------------------------------------


@pytest.mark.uses_cpu(4)
@pytest.mark.slow
def test_cpu_fanout_items_land_on_different_workers():
    """cpus=4 with 8 items should run across 4 distinct subprocesses."""
    from proto_tools.tools.testing.mock_cpu_tool import (
        MockCPUToolConfig,
        MockCPUToolInput,
        run_mock_cpu_tool,
    )
    from proto_tools.utils.device_manager import DeviceManager
    from proto_tools.utils.tool_instance import ToolInstance

    DeviceManager.reset_instance()
    ToolInstance.clear_all()

    try:
        with ToolPool(gpus=0, cpus=4):
            result = run_mock_cpu_tool(
                MockCPUToolInput(items=list(range(8))),
                MockCPUToolConfig(),
            )

        assert result.success, f"ToolPool call failed: {result.errors}"
        assert len(result.results) == 8

        worker_ids = {r.process_unique_id for r in result.results}
        assert len(worker_ids) == 4, (
            f"Expected 4 distinct worker subprocesses for cpus=4, got {len(worker_ids)}: {worker_ids}"
        )
    finally:
        ToolInstance.clear_all()
        DeviceManager.reset_instance()


@pytest.mark.uses_cpu(4)
@pytest.mark.slow
def test_cpu_fanout_omp_pinning():
    """Each CPU worker subprocess should observe OMP_NUM_THREADS == cpus_per_instance."""
    from proto_tools.tools.testing.mock_cpu_tool import (
        MockCPUToolConfig,
        MockCPUToolInput,
        run_mock_cpu_tool,
    )
    from proto_tools.utils.device_manager import DeviceManager
    from proto_tools.utils.tool_instance import ToolInstance

    DeviceManager.reset_instance()
    ToolInstance.clear_all()

    try:
        with ToolPool(gpus=0, cpus=4):
            result = run_mock_cpu_tool(
                MockCPUToolInput(items=list(range(8))),
                MockCPUToolConfig(),  # default cpus_per_instance=1
            )

        assert result.success
        observed = {r.omp_num_threads for r in result.results}
        assert observed == {"1"}, (
            f"Every worker should see OMP_NUM_THREADS=1 (cpus_per_instance default); got {observed}"
        )
    finally:
        ToolInstance.clear_all()
        DeviceManager.reset_instance()


@pytest.mark.uses_cpu(2)
@pytest.mark.slow
def test_cpu_fanout_persistence_across_pool_calls():
    """Second dispatch in the same pool should reuse the warm worker subprocesses."""
    from proto_tools.tools.testing.mock_cpu_tool import (
        MockCPUToolConfig,
        MockCPUToolInput,
        run_mock_cpu_tool,
    )
    from proto_tools.utils.device_manager import DeviceManager
    from proto_tools.utils.tool_instance import ToolInstance

    DeviceManager.reset_instance()
    ToolInstance.clear_all()

    try:
        with ToolPool(gpus=0, cpus=2):
            result1 = run_mock_cpu_tool(
                MockCPUToolInput(items=[1, 2, 3, 4]),
                MockCPUToolConfig(),
            )
            result2 = run_mock_cpu_tool(
                MockCPUToolInput(items=[5, 6, 7, 8]),
                MockCPUToolConfig(),
            )

        assert result1.success and result2.success
        workers_call_1 = {r.process_unique_id for r in result1.results}
        workers_call_2 = {r.process_unique_id for r in result2.results}
        assert workers_call_1 == workers_call_2, (
            "Warm workers should be reused across dispatches; "
            f"call 1 used {workers_call_1}, call 2 used {workers_call_2}"
        )
    finally:
        ToolInstance.clear_all()
        DeviceManager.reset_instance()


@pytest.mark.uses_cpu(4)
@pytest.mark.slow
def test_cpu_fanout_explicit_opt_out_short_circuits():
    """A subclass with cpus_per_instance=None should run a single direct call."""
    from proto_tools.tools.testing.mock_cpu_tool import (
        MockCPUToolConfig,
        MockCPUToolInput,
        run_mock_cpu_tool,
    )
    from proto_tools.utils.device_manager import DeviceManager
    from proto_tools.utils.tool_instance import ToolInstance

    class OptOutConfig(MockCPUToolConfig):
        @property
        def cpus_per_instance(self) -> int | None:
            """Opt out of fan-out — mirrors mmseqs2 in production."""
            return None

    DeviceManager.reset_instance()
    ToolInstance.clear_all()

    try:
        with ToolPool(gpus=0, cpus=4):
            result = run_mock_cpu_tool(
                MockCPUToolInput(items=[1, 2, 3, 4]),
                OptOutConfig(),
            )

        assert result.success
        worker_ids = {r.process_unique_id for r in result.results}
        assert len(worker_ids) == 1, (
            f"Opt-out should short-circuit to a single worker; got {len(worker_ids)}: {worker_ids}"
        )
    finally:
        ToolInstance.clear_all()
        DeviceManager.reset_instance()
