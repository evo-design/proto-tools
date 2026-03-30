"""tests/tool_infra_tests/test_tool_pool.py

Tests for ToolPool parallel fan-out across devices."""
from __future__ import annotations

import threading
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from pydantic import Field

from bio_programming_tools.tools.tool_registry import ToolRegistry
from bio_programming_tools.utils import BaseConfig, ConfigField
from bio_programming_tools.utils.tool_io import BaseToolInput
from bio_programming_tools.utils.tool_pool import (
    DeviceCapability,
    PartialFailureError,
    ToolPool,
    WorkItem,
    _active_pool,
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
    items: List[str] = Field(description="Items to process")


class MockConfig(BaseConfig):
    device: str = ConfigField(default="cuda", hidden=True)


class MockOutput(MockToolOutputBase):
    results: List[str] = Field(description="Processed results")


class MockNonIterableInput(BaseToolInput):
    query: str = Field(description="A single query")


class MockNonIterableOutput(MockToolOutputBase):
    answer: str = Field(description="The answer")


# ── Helpers ─────────────────────────────────────────────────────────────────

def _register_mock_tool(registry, key="mock-process",
                        iterable_input_field="items",
                        iterable_output_field="results"):
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
        iterable_input_field=iterable_input_field,
        iterable_output_field=iterable_output_field,
    )
    def run_mock_process(inputs, config=None, instance=None):
        call_log.append({
            "items": list(inputs.items),
            "device": config.device if config else None,
            "instance": instance,
            "thread": threading.current_thread().name,
        })
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
        DeviceCapability("cuda:1", max_item_cost=None),   # No limit
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
    pool = ToolPool(devices=["cuda:0"])
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
    pool = ToolPool(devices=["cuda:0"])
    token = _active_pool.set(pool)
    try:
        pool2 = ToolPool(devices=["cuda:1"])
        with pytest.raises(RuntimeError, match="cannot be nested"):
            pool2.__enter__()
    finally:
        _active_pool.reset(token)


# ── Parallel dispatch tests ─────────────────────────────────────────────────

def test_dispatch_items_split_across_devices(clean_registry):
    """Items should be split across available devices."""
    func, call_log = _register_mock_tool(clean_registry)

    pool = ToolPool(devices=["cuda:0", "cuda:1"])
    pool._devices = ["cuda:0", "cuda:1"]

    inputs = MockInput(items=["a", "b", "c", "d"])
    config = MockConfig(device="cuda")

    result = pool._parallel_dispatch("mock-process", func, inputs, config)

    assert len(result.results) == 4
    assert result.results == [
        "processed_a", "processed_b", "processed_c", "processed_d"
    ]


def test_dispatch_results_reassembled_in_order(clean_registry):
    """Output items must match original input order."""
    func, call_log = _register_mock_tool(clean_registry)

    pool = ToolPool(devices=["cuda:0", "cuda:1", "cuda:2"])
    pool._devices = ["cuda:0", "cuda:1", "cuda:2"]

    items = [f"item_{i}" for i in range(10)]
    inputs = MockInput(items=items)
    config = MockConfig(device="cuda")

    result = pool._parallel_dispatch("mock-process", func, inputs, config)

    expected = [f"processed_item_{i}" for i in range(10)]
    assert result.results == expected


def test_dispatch_config_device_overridden_per_worker(clean_registry):
    """Each worker should get config.device set to its assigned device."""
    func, call_log = _register_mock_tool(clean_registry)

    pool = ToolPool(devices=["cuda:0", "cuda:1"])
    pool._devices = ["cuda:0", "cuda:1"]

    inputs = MockInput(items=["a", "b", "c", "d"])
    config = MockConfig(device="cuda")

    pool._parallel_dispatch("mock-process", func, inputs, config)

    devices_used = {call["device"] for call in call_log}
    assert devices_used == {"cuda:0", "cuda:1"}


def test_dispatch_worker_instance_names(clean_registry):
    """Worker instance names should encode the device."""
    func, call_log = _register_mock_tool(clean_registry)

    pool = ToolPool(devices=["cuda:0", "cuda:1"])
    pool._devices = ["cuda:0", "cuda:1"]

    inputs = MockInput(items=["a", "b"])
    config = MockConfig(device="cuda")

    pool._parallel_dispatch("mock-process", func, inputs, config)

    instances = {call["instance"] for call in call_log}
    assert "mock-process-pool-cuda_0" in instances
    assert "mock-process-pool-cuda_1" in instances


def test_dispatch_single_item_skips_pool(clean_registry):
    """Single-item inputs should bypass pool overhead."""
    func, call_log = _register_mock_tool(clean_registry)

    pool = ToolPool(devices=["cuda:0", "cuda:1"])
    pool._devices = ["cuda:0", "cuda:1"]

    inputs = MockInput(items=["only_one"])
    config = MockConfig(device="cuda")

    result = pool._parallel_dispatch("mock-process", func, inputs, config)

    assert result.results == ["processed_only_one"]
    assert len(call_log) == 1
    assert call_log[0]["instance"] is None  # Direct call, no worker name


def test_dispatch_devices_per_instance_grouping(clean_registry):
    """Multi-GPU tools should group devices."""
    _, call_log = _register_mock_tool(clean_registry)

    class MultiGPUConfig(BaseConfig):
        device: str = ConfigField(default="cuda", hidden=True)

        @property
        def devices_per_instance(self) -> int:
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
        iterable_input_field="items",
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

    pool = ToolPool(devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    pool._devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]

    inputs = MockInput(items=["a", "b", "c", "d"])
    config = MultiGPUConfig(device="cuda")

    result = pool._parallel_dispatch("multi-gpu-process", run_multi, inputs, config)

    # 4 GPUs / 2 per instance = 2 workers
    assert len(call_log) == 2
    devices_used = {call["device"] for call in call_log}
    assert "cuda:0,cuda:1" in devices_used
    assert "cuda:2,cuda:3" in devices_used
    assert len(result.results) == 4


def test_dispatch_pool_receives_pre_deduped_items(clean_registry):
    """Pool should receive already-deduped items from @tool wrapper."""
    func, call_log = _register_mock_tool(clean_registry)

    pool = ToolPool(devices=["cuda:0", "cuda:1"])
    pool._devices = ["cuda:0", "cuda:1"]

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
        iterable_input_field="items",
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

    pool = ToolPool(devices=["cuda:0", "cuda:1"])
    pool._devices = ["cuda:0", "cuda:1"]

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
    with patch("bio_programming_tools.utils.tool_instance.ToolInstance.persist") as mock_persist:
        mock_ctx = MagicMock()
        mock_persist.return_value = mock_ctx
        pool = ToolPool(devices=["cuda:0"])
        # Patch GPU validation
        with patch("bio_programming_tools.utils.tool_pool.determine_visible_devices"):
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
        iterable_input_field="items",
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
        iterable_input_field="items",
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
    """iterable_input_field and iterable_output_field should be on ToolSpec."""

    @clean_registry.register(
        key="spec-test",
        label="Spec Test",
        category="testing",
        input_class=MockInput,
        config_class=MockConfig,
        output_class=MockOutput,
        description="Test",
        iterable_input_field="items",
        iterable_output_field="results",
    )
    def run_spec_test(inputs, config=None, instance=None):
        return MockOutput(results=[], tool_id="spec-test", success=True)

    spec = clean_registry.get("spec-test")
    assert spec.iterable_input_field == "items"
    assert spec.iterable_output_field == "results"


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
    assert spec.iterable_input_field is None
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
        iterable_input_field="items",
        iterable_output_field="results",
    )
    def run_serial(inputs, config=None, instance=None):
        return MockOutput(results=[], tool_id="serial-test", success=True)

    spec = clean_registry.get("serial-test")
    serialized = spec.model_dump()
    assert "iterable_input_field" not in serialized
    assert "iterable_output_field" not in serialized




# ── devices_per_instance tests ──────────────────────────────────────────────

def test_devices_per_instance_default_is_one():
    """Default devices_per_instance should be 1."""
    config = BaseConfig()
    assert config.devices_per_instance == 1


def test_devices_per_instance_override():
    """Subclasses can override devices_per_instance based on config values."""

    class MultiGPU(BaseConfig):
        model_name: str = ConfigField(default="small")

        @property
        def devices_per_instance(self) -> int:
            return 2 if self.model_name == "large" else 1

    assert MultiGPU(model_name="small").devices_per_instance == 1
    assert MultiGPU(model_name="large").devices_per_instance == 2



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
        iterable_input_field="items",
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

    pool = ToolPool(devices=["cuda:0", "cuda:1"])
    pool._devices = ["cuda:0", "cuda:1"]

    inputs = MockInput(items=["a", "b", "c", "d"])
    config = MockConfig(device="cuda")

    with pytest.raises(PartialFailureError, match="1 partition.*failed") as exc_info:
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
        iterable_input_field="items",
        iterable_output_field="results",
    )
    def run_all_fail(inputs, config=None, instance=None):
        raise RuntimeError("everything is broken")

    pool = ToolPool(devices=["cuda:0", "cuda:1"])
    pool._devices = ["cuda:0", "cuda:1"]

    inputs = MockInput(items=["a", "b", "c", "d"])
    config = MockConfig(device="cuda")

    with pytest.raises(PartialFailureError, match="2 partition.*failed") as exc_info:
        pool._parallel_dispatch("all-fail-tool", run_all_fail, inputs, config)

    err = exc_info.value
    assert err.succeeded == []
    assert len(err.failed) == 2
    failed_devices = {f["device_id"] for f in err.failed}
    assert failed_devices == {"cuda:0", "cuda:1"}


def test_all_succeed_unchanged(clean_registry):
    """Regression: happy path is unaffected by error propagation changes."""
    func, call_log = _register_mock_tool(clean_registry)

    pool = ToolPool(devices=["cuda:0", "cuda:1"])
    pool._devices = ["cuda:0", "cuda:1"]

    inputs = MockInput(items=["a", "b", "c", "d"])
    config = MockConfig(device="cuda")

    result = pool._parallel_dispatch("mock-process", func, inputs, config)

    assert result.results == [
        "processed_a", "processed_b", "processed_c", "processed_d"
    ]
    assert len(call_log) == 2  # Two partitions


# ---------------------------------------------------------------------------
# Integration tests (require GPU)
# ---------------------------------------------------------------------------

@pytest.mark.uses_gpu(2)
@pytest.mark.slow
def test_gpu_fanout_items_land_on_different_gpus():
    """Items should be dispatched to different physical GPUs."""
    from bio_programming_tools.tools.testing.mock_pytorch_tool import (
        MockPyTorchToolConfig,
        MockPyTorchToolInput,
        run_mock_pytorch_tool,
    )
    from bio_programming_tools.utils.device_manager import DeviceManager
    from bio_programming_tools.utils.tool_instance import ToolInstance

    DeviceManager.reset_instance()
    ToolInstance.clear_all()

    try:
        data_items = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        with ToolPool(devices=["cuda:0", "cuda:1"]):
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
    from bio_programming_tools.tools.testing.mock_pytorch_tool import (
        MockPyTorchToolConfig,
        MockPyTorchToolInput,
        run_mock_pytorch_tool,
    )
    from bio_programming_tools.utils.device_manager import DeviceManager
    from bio_programming_tools.utils.tool_instance import ToolInstance

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

        with ToolPool(devices=["cuda:0", "cuda:1"]):
            result = run_mock_pytorch_tool(
                MockPyTorchToolInput(data_items=data_items),
                MockPyTorchToolConfig(memory_mb=64),
            )

        assert result.success
        assert len(result.results) == 6

        # With 6 items and 2 GPUs (LPT round-robin on uniform costs),
        # items [0,2,4] go to one GPU and [1,3,5] go to the other.
        devices = [r.device_used for r in result.results]
        assert devices[0] == devices[2] == devices[4], (
            f"Items 0,2,4 should share a GPU: {devices}"
        )
        assert devices[1] == devices[3] == devices[5], (
            f"Items 1,3,5 should share a GPU: {devices}"
        )
        assert devices[0] != devices[1], (
            f"The two groups should be on different GPUs: {devices}"
        )
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

    from bio_programming_tools.tools.testing.mock_pytorch_tool import (
        MockPyTorchToolConfig,
        MockPyTorchToolInput,
        run_mock_pytorch_tool,
    )
    from bio_programming_tools.utils.device_manager import DeviceManager
    from bio_programming_tools.utils.tool_instance import ToolInstance

    DeviceManager.reset_instance()
    ToolInstance.clear_all()

    try:
        data_items = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]

        with ToolPool(devices=["cuda:0", "cuda:1"]):
            # Cold call — pays model loading on both GPUs
            t0 = time.time()
            result1 = run_mock_pytorch_tool(
                MockPyTorchToolInput(data_items=data_items),
                MockPyTorchToolConfig(memory_mb=64),
            )
            cold_time = time.time() - t0

            # Warm call — workers already loaded
            t0 = time.time()
            result2 = run_mock_pytorch_tool(
                MockPyTorchToolInput(data_items=data_items),
                MockPyTorchToolConfig(memory_mb=64),
            )
            warm_time = time.time() - t0

        assert result1.success
        assert result2.success
        assert warm_time < cold_time, (
            f"Warm call ({warm_time:.2f}s) should be faster than "
            f"cold call ({cold_time:.2f}s) — workers should persist"
        )
    finally:
        ToolInstance.clear_all()
        DeviceManager.reset_instance()


@pytest.mark.uses_gpu(2)
@pytest.mark.slow
def test_gpu_fanout_single_item_bypasses_pool():
    """A single-item input should bypass pool overhead."""
    from bio_programming_tools.tools.testing.mock_pytorch_tool import (
        MockPyTorchToolConfig,
        MockPyTorchToolInput,
        run_mock_pytorch_tool,
    )
    from bio_programming_tools.utils.device_manager import DeviceManager
    from bio_programming_tools.utils.tool_instance import ToolInstance

    DeviceManager.reset_instance()
    ToolInstance.clear_all()

    try:
        with ToolPool(devices=["cuda:0", "cuda:1"]):
            result = run_mock_pytorch_tool(
                MockPyTorchToolInput(data_items=[[1.0, 2.0, 3.0, 4.0]]),
                MockPyTorchToolConfig(memory_mb=64),
            )

        assert result.success
        assert len(result.results) == 1
    finally:
        ToolInstance.clear_all()
        DeviceManager.reset_instance()
