"""tests/tool_infra_tests/test_tool_registry.py

Tests for ToolRegistry."""

import time

import pytest
from pydantic import Field

from proto_tools.tools.tool_registry import (
    MAX_RETRIES,
    ToolRegistry,
    ToolSpec,
)
from proto_tools.utils import BaseConfig, ConfigField
from proto_tools.utils.tool_io import BaseToolInput
from tests.tool_infra_tests.test_export_functionality import MockToolOutputBase

# ── example_input completeness ───────────────────────────────────────────────

@pytest.mark.parametrize(
    "tool_spec",
    ToolRegistry.list_all(),
    ids=lambda spec: spec.key,
)
def test_all_tools_have_example_input(tool_spec):
    """Every tool must define example_input for parametrized testing."""
    assert tool_spec.example_input is not None, (
        f"Tool {tool_spec.key} missing example_input= in @tool() decorator"
    )
    example = tool_spec.example_input()
    assert isinstance(example, tool_spec.input_model), (
        f"example_input() returned {type(example).__name__}, "
        f"expected {tool_spec.input_model.__name__}"
    )


# ── Mock data models ─────────────────────────────────────────────────────────
class MockToolInput(BaseToolInput):
    """Mock input for testing"""

    input_data: str = Field(description="Input data to process")


class MockToolConfig(BaseConfig):
    """Mock configuration for testing"""
    param1: str = ConfigField(description="Parameter 1")
    param2: int = ConfigField(default=10, ge=0, description="Parameter 2")


class MockToolOutput(MockToolOutputBase):
    """Mock output for testing"""
    result: str = Field(description="Result string")


class AnotherMockToolInput(BaseToolInput):
    """Another mock input for testing"""

    sequences: list[str] = Field(description="Input sequences")


class AnotherMockToolConfig(BaseConfig):
    """Another mock configuration for testing"""
    threshold: float = ConfigField(default=0.5, ge=0.0, le=1.0, description="Threshold")


class AnotherMockToolOutput(MockToolOutputBase):
    """Another mock output for testing"""
    processed_data: list[str] = Field(description="Processed data")
    count: int = Field(description="Data count")


@pytest.fixture
def clean_registry():
    """Fixture to provide a clean registry for each test"""
    # Save original registry
    original_registry = ToolRegistry._registry.copy()

    # Clear registry
    ToolRegistry._registry.clear()

    yield ToolRegistry

    # Restore original registry
    ToolRegistry._registry = original_registry


def test_tool_registry_register_decorator(clean_registry):
    """Test basic tool registration using decorator"""

    @clean_registry.register(
        key="mock-tool",
        label="Mock Tool",
        category="test",
        input_class=MockToolInput,
        config_class=MockToolConfig,
        output_class=MockToolOutput,
        description="Mock tool for testing",
    )
    def mock_tool(inputs: MockToolInput, config: MockToolConfig) -> MockToolOutput:
        return MockToolOutput(
            tool_id="mock-tool",
            execution_time=0.1,
            success=True,
            result=f"Processed {inputs.input_data} with {config.param1} and {config.param2}",
        )

    # Check that tool is registered
    assert "mock-tool" in clean_registry._registry

    # Check spec details
    spec = clean_registry.get("mock-tool")
    assert isinstance(spec, ToolSpec)
    assert spec.input_model == MockToolInput
    assert spec.config_model == MockToolConfig
    assert spec.description == "Mock tool for testing"
    assert spec.output_model == MockToolOutput


def test_tool_registry_prevent_duplicate_registration(clean_registry):
    """Test that duplicate registration raises error"""

    @clean_registry.register(
        key="duplicate-tool",
        label="Duplicate Tool",
        category="test",
        input_class=MockToolInput,
        config_class=MockToolConfig,
        output_class=MockToolOutput,
        description="First registration",
    )
    def first_tool(inputs: MockToolInput, config: MockToolConfig) -> MockToolOutput:
        return MockToolOutput(
            tool_id="duplicate-tool",
            execution_time=0.1,
            success=True,
            result="first"
        )

    # Attempt to register with same key should fail
    with pytest.raises(ValueError, match="(?i)already registered"):

        @clean_registry.register(
            key="duplicate-tool",  # Same key
            label="Duplicate Tool 2",
            category="test",
            input_class=AnotherMockToolInput,
            config_class=AnotherMockToolConfig,
            output_class=AnotherMockToolOutput,
            description="Second registration",
        )
        def second_tool(
            inputs: AnotherMockToolInput, config: AnotherMockToolConfig
        ) -> AnotherMockToolOutput:
            return AnotherMockToolOutput(
                tool_id="duplicate-tool",
                execution_time=0.1,
                success=True,
                processed_data=inputs.sequences,
                count=len(inputs.sequences),
            )


def test_tool_registry_list_all(clean_registry):
    """Test listing all registered tools"""

    @clean_registry.register(
        key="tool-1",
        label="Tool 1",
        category="test",
        input_class=MockToolInput,
        config_class=MockToolConfig,
        output_class=MockToolOutput,
        description="Tool 1",
    )
    def tool_1(inputs: MockToolInput, config: MockToolConfig) -> MockToolOutput:
        return MockToolOutput(
            tool_id="tool-1",
            execution_time=0.1,
            success=True,
            result=f"tool1: {inputs.input_data}",
        )

    @clean_registry.register(
        key="tool-2",
        label="Tool 2",
        category="test",
        input_class=AnotherMockToolInput,
        config_class=AnotherMockToolConfig,
        output_class=AnotherMockToolOutput,
        description="Tool 2",
    )
    def tool_2(
        inputs: AnotherMockToolInput, config: AnotherMockToolConfig
    ) -> AnotherMockToolOutput:
        return AnotherMockToolOutput(
            tool_id="tool-2",
            execution_time=10.0,
            success=True,
            processed_data=inputs.sequences,
            count=len(inputs.sequences),
        )

    all_tools = clean_registry.list_all()

    assert len(all_tools) == 2
    assert "tool-1" in {spec.key for spec in all_tools}
    assert "tool-2" in {spec.key for spec in all_tools}

    # Convert to dict for easy access
    tools_dict = {spec.key: spec for spec in all_tools}

    # Check tool-1 metadata
    tool1_spec = tools_dict["tool-1"]
    assert tool1_spec.description == "Tool 1"
    # Verify config_model can generate JSON schema
    assert tool1_spec.config_model is not None
    schema = tool1_spec.config_model.model_json_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema

    # Check tool-2 metadata
    tool2_spec = tools_dict["tool-2"]
    assert tool2_spec.description == "Tool 2"


def test_tool_registry_schema_methods(clean_registry):
    """All schema query methods (input, config, output, combined) return valid JSON schemas."""

    @clean_registry.register(
        key="schema-tool",
        label="Schema Tool",
        category="test",
        input_class=MockToolInput,
        config_class=MockToolConfig,
        output_class=MockToolOutput,
        description="Schema tool",
    )
    def schema_tool(inputs: MockToolInput, config: MockToolConfig) -> MockToolOutput:
        return MockToolOutput(
            tool_id="schema-tool",
            execution_time=0.1,
            success=True,
            result=f"ok: {inputs.input_data}",
        )

    # get_config_schema
    config_schema = clean_registry.get_config_schema("schema-tool")
    assert "properties" in config_schema
    assert config_schema["properties"]["param1"]["type"] == "string"
    assert config_schema["properties"]["param2"]["type"] == "integer"
    assert config_schema["properties"]["param2"]["default"] == 10
    assert config_schema["properties"]["param2"]["minimum"] == 0

    # get_input_schema
    input_schema = clean_registry.get_input_schema("schema-tool")
    assert "properties" in input_schema
    assert input_schema["properties"]["input_data"]["type"] == "string"

    # get_schemas (combined)
    schemas = clean_registry.get_schemas("schema-tool")
    assert set(schemas.keys()) == {"inputs", "config", "output"}
    assert "properties" in schemas["inputs"]
    assert "properties" in schemas["config"]
    assert "properties" in schemas["output"]


def test_tool_registry_get_unknown_tool(clean_registry):
    """Test that getting unknown tool raises error"""
    with pytest.raises(ValueError, match="non-existent-tool"):
        clean_registry.get("non-existent-tool")


def test_tool_registry_decorator_populates_metadata(clean_registry):
    """Test that decorator automatically populates metadata fields"""

    @clean_registry.register(
        key="metadata-tool",
        label="Metadata Tool",
        category="test",
        input_class=MockToolInput,
        config_class=MockToolConfig,
        output_class=MockToolOutput,
        description="Tool that tests metadata population",
    )
    def metadata_tool(inputs: MockToolInput, config: MockToolConfig, instance=None) -> MockToolOutput:
        # Tool function doesn't need to set metadata fields
        time.sleep(0.1)
        return MockToolOutput(result=f"Processed {inputs.input_data}")

    # Get the registered function and call it
    spec = clean_registry.get("metadata-tool")
    inputs = MockToolInput(input_data="test")
    config = MockToolConfig(param1="value1", param2=5)

    result = spec.function(inputs, config)

    # Verify metadata was populated by decorator
    assert result.tool_id == "metadata-tool", "Tool ID should be populated by decorator"
    assert (
        result.execution_time is not None
    ), "Execution time should be populated by decorator"
    assert (
        result.execution_time > 0.0
    ), "Execution time should be positive for successful execution"
    assert result.success is True, "Success should be True"
    assert result.timestamp is not None, "Timestamp should be populated by decorator"
    assert result.result == "Processed test"


def test_tool_registry_decorator_handles_exceptions(clean_registry):
    """Test that decorator handles exceptions and returns error output"""

    @clean_registry.register(
        key="failing-tool",
        label="Failing Tool",
        category="test",
        input_class=MockToolInput,
        config_class=MockToolConfig,
        output_class=MockToolOutput,
        description="Tool that raises an exception",
    )
    def failing_tool(inputs: MockToolInput, config: MockToolConfig, instance=None) -> MockToolOutput:
        raise ValueError("Something went wrong!")

    # Get the registered function and call it
    spec = clean_registry.get("failing-tool")
    inputs = MockToolInput(input_data="test")
    config = MockToolConfig(param1="value1", param2=5)

    result = spec.function(inputs, config)

    # Verify error output was created
    assert result.tool_id == "failing-tool"
    assert result.execution_time is not None
    assert result.success is False
    assert result.timestamp is not None
    assert len(result.errors) == 2
    assert "Something went wrong!" in result.errors[0]


def test_tool_registry_decorator_captures_warnings(clean_registry):
    """Test that decorator captures warnings during execution"""
    import warnings

    @clean_registry.register(
        key="warning-tool",
        label="Warning Tool",
        category="test",
        input_class=MockToolInput,
        config_class=MockToolConfig,
        output_class=MockToolOutput,
        description="Tool that generates warnings",
    )
    def warning_tool(inputs: MockToolInput, config: MockToolConfig, instance=None) -> MockToolOutput:
        warnings.warn("This is a warning!")
        return MockToolOutput(result="Done")

    # Get the registered function and call it
    spec = clean_registry.get("warning-tool")
    inputs = MockToolInput(input_data="test")
    config = MockToolConfig(param1="value1", param2=5)

    result = spec.function(inputs, config)

    # Verify warning was captured
    assert result.success is True
    assert len(result.warnings) >= 1
    assert any("This is a warning!" in w for w in result.warnings)


def test_tool_output_error_access_raises_exception(clean_registry):
    """Test that accessing unset fields on failed output raises ToolExecutionError"""
    from proto_tools.utils.tool_io import ToolExecutionError

    @clean_registry.register(
        key="error-access-tool",
        label="Error Access Tool",
        category="test",
        input_class=MockToolInput,
        config_class=MockToolConfig,
        output_class=MockToolOutput,
        description="Tool that fails and tests error access",
    )
    def error_access_tool(
        inputs: MockToolInput, config: MockToolConfig, instance=None
    ) -> MockToolOutput:
        raise RuntimeError("Tool execution failed")

    # Get the registered function and call it
    spec = clean_registry.get("error-access-tool")
    inputs = MockToolInput(input_data="test")
    config = MockToolConfig(param1="value1", param2=5)

    result = spec.function(inputs, config)

    # Verify output indicates failure
    assert result.success is False, "Success should be False for failed execution"

    # Trying to access 'result' field should raise ToolExecutionError
    with pytest.raises(ToolExecutionError, match="Tool execution failed"):
        _ = result.result

    # Attempting to access standard fields should not raise ToolExecutionError
    assert result.tool_id is not None
    assert result.execution_time is not None
    assert result.timestamp is not None
    assert len(result.warnings) == 0
    assert len(result.errors) == 2
    assert result.errors[0] == "Tool execution failed"


def test_tool_output_successful_access_works(clean_registry):
    """Test that accessing fields on successful output works normally"""

    @clean_registry.register(
        key="success-access-tool",
        label="Success Access Tool",
        category="test",
        input_class=MockToolInput,
        config_class=MockToolConfig,
        output_class=MockToolOutput,
        description="Tool that succeeds and tests field access",
    )
    def success_access_tool(
        inputs: MockToolInput, config: MockToolConfig, instance=None
    ) -> MockToolOutput:
        return MockToolOutput(result="Success!")

    # Get the registered function and call it
    spec = clean_registry.get("success-access-tool")
    inputs = MockToolInput(input_data="test")
    config = MockToolConfig(param1="value1", param2=5)

    result = spec.function(inputs, config)

    # Verify output indicates success
    assert result.success is True

    # Accessing 'result' field should work normally
    assert result.result == "Success!"
    assert result.tool_id == "success-access-tool"
    assert result.execution_time is not None


def test_tool_registry_list_gpu_tools(clean_registry):
    """Test list_gpu_tools returns only GPU tools."""

    @clean_registry.register(
        key="gpu-1",
        label="GPU 1",
        category="test",
        input_class=MockToolInput,
        config_class=MockToolConfig,
        output_class=MockToolOutput,
        description="GPU 1",
        uses_gpu=True,
    )
    def gpu_1(inputs, config):
        return MockToolOutput(result="ok")

    @clean_registry.register(
        key="cpu-1",
        label="CPU 1",
        category="test",
        input_class=MockToolInput,
        config_class=MockToolConfig,
        output_class=MockToolOutput,
        description="CPU 1",
    )
    def cpu_1(inputs, config):
        return MockToolOutput(result="ok")

    @clean_registry.register(
        key="gpu-2",
        label="GPU 2",
        category="test",
        input_class=MockToolInput,
        config_class=MockToolConfig,
        output_class=MockToolOutput,
        description="GPU 2",
        uses_gpu=True,
    )
    def gpu_2(inputs, config):
        return MockToolOutput(result="ok")

    gpu_tools = clean_registry.list_gpu_tools()
    cpu_tools = clean_registry.list_cpu_tools()

    assert len(gpu_tools) == 2
    assert len(cpu_tools) == 1
    assert {s.key for s in gpu_tools} == {"gpu-1", "gpu-2"}
    assert {s.key for s in cpu_tools} == {"cpu-1"}


def test_gpu_tools_are_marked():
    """Verify that known GPU tools are marked with uses_gpu=True in the real registry."""
    gpu_tools = ToolRegistry.list_gpu_tools()
    gpu_keys = {spec.key for spec in gpu_tools}

    # At least 20 GPU tool registrations should exist
    assert len(gpu_keys) >= 20

    # Spot check known GPU tools
    assert "esm2-embedding" in gpu_keys
    assert "evo2-sample" in gpu_keys
    assert "boltz2-prediction" in gpu_keys
    assert "proteinmpnn-sample" in gpu_keys


def test_cpu_tools_are_not_marked_gpu():
    """Verify that known CPU tools are NOT marked uses_gpu in the real registry."""
    cpu_tools = ToolRegistry.list_cpu_tools()
    cpu_keys = {spec.key for spec in cpu_tools}

    # Spot check known CPU tools
    assert "blast-search" in cpu_keys
    assert "mafft-align" in cpu_keys


def _register_and_run(registry, key, func):
    """Register a tool with mock types and run it with default inputs."""
    registry.register(
        key=key, label=key, category="test",
        input_class=MockToolInput, config_class=MockToolConfig, output_class=MockToolOutput,
        description=key,
    )(func)
    spec = registry.get(key)
    return spec.function(MockToolInput(input_data="test"), MockToolConfig(param1="v"))


@pytest.fixture
def fast_retry(monkeypatch):
    """Zero out retry delay for tests."""
    import proto_tools.tools.tool_registry as reg_module
    monkeypatch.setattr(reg_module, "RETRY_DELAY", 0.01)


def test_retryable_error_succeeds_after_retries(clean_registry, fast_retry):
    call_count = 0

    def tool(inputs, config, instance=None):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Connection refused")
        return MockToolOutput(result="success")

    result = _register_and_run(clean_registry, "retry-succeed", tool)
    assert result.success is True
    assert result.result == "success"
    assert call_count == 3


def test_non_retryable_error_not_retried(clean_registry, fast_retry):
    call_count = 0

    def tool(inputs, config, instance=None):
        nonlocal call_count
        call_count += 1
        raise ValueError("Bad input")

    result = _register_and_run(clean_registry, "no-retry", tool)
    assert result.success is False
    assert call_count == 1
    assert "Bad input" in result.errors[0]


def test_try_dispatch_intercepts_tool_call(clean_registry):
    """_try_dispatch intercepts tool calls before local execution."""
    original = ToolRegistry._try_dispatch

    ToolRegistry._try_dispatch = classmethod(
        lambda cls, key, inputs, config: MockToolOutput(result="dispatched", success=True)
    )
    try:
        result = _register_and_run(
            clean_registry, "dispatch-test",
            lambda inputs, config, instance=None: MockToolOutput(result="local"),
        )
        assert result.success is True
        assert result.result == "dispatched"
    finally:
        ToolRegistry._try_dispatch = original


def test_try_dispatch_none_falls_through(clean_registry):
    """_try_dispatch returning None falls through to local execution."""
    original = ToolRegistry._try_dispatch

    ToolRegistry._try_dispatch = classmethod(
        lambda cls, key, inputs, config: None
    )
    try:
        result = _register_and_run(
            clean_registry, "dispatch-fallthrough",
            lambda inputs, config, instance=None: MockToolOutput(result="local ok"),
        )
        assert result.success is True
        assert result.result == "local ok"
    finally:
        ToolRegistry._try_dispatch = original


def test_try_dispatch_exception_returns_error_output(clean_registry):
    """_try_dispatch raising returns a structured error output."""
    original = ToolRegistry._try_dispatch

    ToolRegistry._try_dispatch = classmethod(
        lambda cls, key, inputs, config: (_ for _ in ()).throw(RuntimeError("remote down"))
    )
    try:
        result = _register_and_run(
            clean_registry, "dispatch-error",
            lambda inputs, config, instance=None: MockToolOutput(result="local"),
        )
        assert result.success is False
        assert "remote down" in result.errors[0]
    finally:
        ToolRegistry._try_dispatch = original


def test_retries_exhaust_with_meaningful_traceback(clean_registry, fast_retry):
    call_count = 0

    def tool(inputs, config, instance=None):
        nonlocal call_count
        call_count += 1
        raise ConnectionError("connection refused")

    result = _register_and_run(clean_registry, "retry-exhaust", tool)
    assert result.success is False
    assert call_count == 1 + MAX_RETRIES
    assert "connection refused" in result.errors[0]
    assert "ConnectionError" in result.errors[1]
    assert "NoneType: None" not in result.errors[1]


def test_timeout_error_not_retried(clean_registry, fast_retry):
    """TimeoutError from ToolInstance/PersistentWorker means the tool exceeded its
    configured timeout — retrying with the same limit would just time out again."""
    call_count = 0

    def tool(inputs, config, instance=None):
        nonlocal call_count
        call_count += 1
        raise TimeoutError("worker timed out after 300s")

    result = _register_and_run(clean_registry, "timeout-no-retry", tool)
    assert result.success is False
    assert call_count == 1
    assert "worker timed out" in result.errors[0]


# ── Device count validation ──────────────────────────────────────────────────


class MockConfigWithDevice(BaseConfig):
    """Mock config with device field for testing device validation."""
    device: str = ConfigField(default="cpu", description="Device to use")


def test_tool_registry_default_device_count(clean_registry):
    """Test that device_count defaults to '1' when not specified."""

    @clean_registry.register(
        key="default-device-count",
        label="Default Device Count Tool",
        category="test",
        input_class=MockToolInput,
        config_class=MockToolConfig,
        output_class=MockToolOutput,
        description="Tool with default device_count",
    )
    def default_device_count_tool(inputs: MockToolInput, config: MockToolConfig, instance=None) -> MockToolOutput:
        return MockToolOutput(result="ok")

    spec = clean_registry.get("default-device-count")
    assert spec.device_count == "1"


def test_tool_registry_exact_device_count_validation_passes(clean_registry):
    """Test that exact device count validation passes when allocation matches."""

    @clean_registry.register(
        key="exact-count-tool",
        label="Exact Count Tool",
        category="test",
        input_class=MockToolInput,
        config_class=MockConfigWithDevice,
        output_class=MockToolOutput,
        description="Tool requiring exactly 1 device",
        device_count="1",
    )
    def exact_count_tool(inputs: MockToolInput, config: MockConfigWithDevice, instance=None) -> MockToolOutput:
        return MockToolOutput(result="ok")

    spec = clean_registry.get("exact-count-tool")
    inputs = MockToolInput(input_data="test")
    config = MockConfigWithDevice(device="cuda:0")

    # Should not raise any errors
    result = spec.function(inputs, config)
    assert result.success is True


def test_tool_registry_under_allocation_raises_error(clean_registry):
    """Test that under-allocation raises ValueError."""

    @clean_registry.register(
        key="under-alloc-tool",
        label="Under Allocation Tool",
        category="test",
        input_class=MockToolInput,
        config_class=MockConfigWithDevice,
        output_class=MockToolOutput,
        description="Tool requiring 2 devices",
        uses_gpu=True,
        device_count="2",
    )
    def under_alloc_tool(inputs: MockToolInput, config: MockConfigWithDevice, instance=None) -> MockToolOutput:
        return MockToolOutput(result="ok")

    spec = clean_registry.get("under-alloc-tool")
    inputs = MockToolInput(input_data="test")
    config = MockConfigWithDevice(device="cuda:0")  # Only 1 device

    # Should raise ValueError for under-allocation
    with pytest.raises(ValueError, match="requires at least 2"):
        spec.function(inputs, config)


def test_tool_registry_over_allocation_logs_warning(clean_registry, caplog):
    """Test that over-allocation logs a warning but succeeds."""
    import logging

    @clean_registry.register(
        key="over-alloc-tool",
        label="Over Allocation Tool",
        category="test",
        input_class=MockToolInput,
        config_class=MockConfigWithDevice,
        output_class=MockToolOutput,
        description="Tool requiring 1 device",
        uses_gpu=True,
        device_count="1",
    )
    def over_alloc_tool(inputs: MockToolInput, config: MockConfigWithDevice, instance=None) -> MockToolOutput:
        return MockToolOutput(result="ok")

    spec = clean_registry.get("over-alloc-tool")
    inputs = MockToolInput(input_data="test")
    config = MockConfigWithDevice(device="cuda:0,1")  # 2 devices for tool that needs 1

    with caplog.at_level(logging.WARNING):
        result = spec.function(inputs, config)

    # Should succeed despite over-allocation
    assert result.success is True

    # Should log a warning
    assert "requires at most 1 device(s), but 2 requested" in caplog.text


def test_tool_registry_range_device_count_validation_passes(clean_registry):
    """Test that range device count validation passes when within range."""

    @clean_registry.register(
        key="range-count-tool",
        label="Range Count Tool",
        category="test",
        input_class=MockToolInput,
        config_class=MockConfigWithDevice,
        output_class=MockToolOutput,
        description="Tool supporting 1-2 devices",
        device_count="1-2",
    )
    def range_count_tool(inputs: MockToolInput, config: MockConfigWithDevice, instance=None) -> MockToolOutput:
        return MockToolOutput(result="ok")

    spec = clean_registry.get("range-count-tool")
    inputs = MockToolInput(input_data="test")

    # Test with 1 device (min)
    config1 = MockConfigWithDevice(device="cuda:0")
    result1 = spec.function(inputs, config1)
    assert result1.success is True

    # Test with 2 devices (max)
    config2 = MockConfigWithDevice(device="cuda:0,1")
    result2 = spec.function(inputs, config2)
    assert result2.success is True


def test_tool_registry_open_ended_device_count_validation_passes(clean_registry):
    """Test that open-ended device count validation passes."""

    @clean_registry.register(
        key="open-ended-tool",
        label="Open Ended Tool",
        category="test",
        input_class=MockToolInput,
        config_class=MockConfigWithDevice,
        output_class=MockToolOutput,
        description="Tool supporting >=1 devices",
        device_count=">=1",
    )
    def open_ended_tool(inputs: MockToolInput, config: MockConfigWithDevice, instance=None) -> MockToolOutput:
        return MockToolOutput(result="ok")

    spec = clean_registry.get("open-ended-tool")
    inputs = MockToolInput(input_data="test")

    # Test with 1 device
    config1 = MockConfigWithDevice(device="cuda:0")
    result1 = spec.function(inputs, config1)
    assert result1.success is True

    # Test with 3 devices (auto-allocate)
    config3 = MockConfigWithDevice(device="cudax3")
    result3 = spec.function(inputs, config3)
    assert result3.success is True


# ── Iterable dedup in @tool wrapper ──────────────────────────────────────────


class MockIterableInput(BaseToolInput):
    """Input with an iterable field for dedup tests."""
    items: list[str] = Field(description="Items to process")


class MockIterableOutput(MockToolOutputBase):
    """Output with an iterable field for dedup tests."""
    results: list[str] = Field(description="Processed results")


def _register_cacheable_iterable(registry, key, func):
    """Helper to register a cacheable iterable tool."""
    registry.register(
        key=key,
        label=key,
        category="test",
        input_class=MockIterableInput,
        config_class=MockToolConfig,
        output_class=MockIterableOutput,
        description=key,
        iterable_input_field="items",
        iterable_output_field="results",
        cacheable=True,
    )(func)
    return registry.get(key)


@pytest.mark.parametrize(
    "items, expected_dispatched, expected_results",
    [
        # Mixed duplicates: dedup to 3 unique, expand back to 5
        (["a", "b", "a", "c", "b"], ["a", "b", "c"], ["out_a", "out_b", "out_a", "out_c", "out_b"]),
        # All unique: pass through unchanged
        (["a", "b", "c"], ["a", "b", "c"], ["out_a", "out_b", "out_c"]),
        # Single item: skip dedup entirely
        (["only"], ["only"], ["out_only"]),
        # All duplicates: reduce to 1, expand back to 4
        (["x", "x", "x", "x"], ["x"], ["out_x", "out_x", "out_x", "out_x"]),
    ],
    ids=["mixed-dupes", "all-unique", "single-item", "all-dupes"],
)
def test_tool_wrapper_dedup_iterable_items(clean_registry, items, expected_dispatched, expected_results):
    """@tool should dedup cacheable iterable items and expand results back."""
    received_items = []

    def run_tool(inputs, config=None, instance=None):
        received_items.extend(inputs.items)
        return MockIterableOutput(results=[f"out_{x}" for x in inputs.items])

    # Each parametrize case needs a unique key to avoid duplicate registration
    key = f"dedup-{'-'.join(items[:2])}-{len(items)}"
    spec = _register_cacheable_iterable(clean_registry, key, run_tool)
    result = spec.function(MockIterableInput(items=items), MockToolConfig(param1="v"))

    assert received_items == expected_dispatched
    assert result.results == expected_results
    assert result.success is True


def test_tool_wrapper_dedup_skipped_without_cacheable(clean_registry):
    """@tool should NOT dedup when cacheable=False (generative tools)."""
    received_items = []

    @clean_registry.register(
        key="no-cache-test",
        label="No Cache Test",
        category="test",
        input_class=MockIterableInput,
        config_class=MockToolConfig,
        output_class=MockIterableOutput,
        description="Generative tool without cache",
        iterable_input_field="items",
        iterable_output_field="results",
        cacheable=False,
    )
    def run_no_cache(inputs, config=None, instance=None):
        received_items.extend(inputs.items)
        return MockIterableOutput(
            results=[f"out_{x}" for x in inputs.items],
        )

    spec = clean_registry.get("no-cache-test")
    inputs = MockIterableInput(items=["a", "a", "a"])
    config = MockToolConfig(param1="v")

    result = spec.function(inputs, config)

    # All 3 identical items should be passed through (no dedup)
    assert len(received_items) == 3
    assert received_items == ["a", "a", "a"]
    assert result.results == ["out_a", "out_a", "out_a"]


# ── Cacheable flow through @tool wrapper ──────────────────────────────────────


@pytest.fixture
def _setup_cache():
    """Set up cache in contextvar before each test, clear after."""
    from proto_tools.utils.tool_cache import ToolCache, _program_tool_cache
    cache = ToolCache()
    _program_tool_cache.set(cache)
    yield cache
    _program_tool_cache.set(None)


def test_cacheable_iterable_full_hit_skips_dispatch(clean_registry, _setup_cache):
    """Cacheable iterable tool returns cached results without calling func."""
    call_count = 0

    def run_tool(inputs, config=None, instance=None):
        nonlocal call_count
        call_count += len(inputs.items)
        return MockIterableOutput(
            results=[f"out_{x}" for x in inputs.items],
        )

    spec = _register_cacheable_iterable(clean_registry, "cache-hit-iter", run_tool)
    config = MockToolConfig(param1="v")

    # First call: all miss
    result1 = spec.function(MockIterableInput(items=["a", "b"]), config)
    assert call_count == 2
    assert result1.results == ["out_a", "out_b"]

    # Second call: all cached, func should NOT be called again
    result2 = spec.function(MockIterableInput(items=["a", "b"]), config)
    assert call_count == 2  # unchanged
    assert result2.results == ["out_a", "out_b"]
    assert result2.execution_time == 0.0  # full cache hit


def test_cacheable_iterable_partial_hit(clean_registry, _setup_cache):
    """Cacheable iterable tool dispatches only uncached items."""
    call_count = 0

    def run_tool(inputs, config=None, instance=None):
        nonlocal call_count
        call_count += len(inputs.items)
        return MockIterableOutput(
            results=[f"out_{x}" for x in inputs.items],
        )

    spec = _register_cacheable_iterable(clean_registry, "cache-partial", run_tool)
    config = MockToolConfig(param1="v")

    # First call: cache "a" and "b"
    spec.function(MockIterableInput(items=["a", "b"]), config)
    assert call_count == 2

    # Second call: "a" cached, "c" new
    result = spec.function(MockIterableInput(items=["a", "c"]), config)
    assert call_count == 3  # only "c" dispatched
    assert result.results == ["out_a", "out_c"]


def test_cacheable_whole_output_hit_skips_dispatch(clean_registry, _setup_cache):
    """Cacheable non-iterable tool returns cached output on second call."""
    call_count = 0

    @clean_registry.register(
        key="cache-hit-whole",
        label="Whole Cache Hit",
        category="test",
        input_class=MockToolInput,
        config_class=MockToolConfig,
        output_class=MockToolOutput,
        description="Whole-output cache test",
        cacheable=True,
    )
    def run_tool(inputs, config=None, instance=None):
        nonlocal call_count
        call_count += 1
        return MockToolOutput(result=f"processed_{inputs.input_data}")

    spec = clean_registry.get("cache-hit-whole")
    inputs = MockToolInput(input_data="test")
    config = MockToolConfig(param1="v")

    result1 = spec.function(inputs, config)
    assert call_count == 1
    assert result1.result == "processed_test"

    # Second call: cache hit
    result2 = spec.function(inputs, config)
    assert call_count == 1  # unchanged
    assert result2.result == "processed_test"


def test_cacheable_whole_output_different_inputs(clean_registry, _setup_cache):
    """Different inputs produce different cache entries for whole-output tools."""
    call_count = 0

    @clean_registry.register(
        key="cache-diff-inputs",
        label="Cache Diff Inputs",
        category="test",
        input_class=MockToolInput,
        config_class=MockToolConfig,
        output_class=MockToolOutput,
        description="Different inputs test",
        cacheable=True,
    )
    def run_tool(inputs, config=None, instance=None):
        nonlocal call_count
        call_count += 1
        return MockToolOutput(result=f"processed_{inputs.input_data}")

    spec = clean_registry.get("cache-diff-inputs")
    config = MockToolConfig(param1="v")

    spec.function(MockToolInput(input_data="a"), config)
    assert call_count == 1

    spec.function(MockToolInput(input_data="b"), config)
    assert call_count == 2  # different input, cache miss

    # Original input still cached
    spec.function(MockToolInput(input_data="a"), config)
    assert call_count == 2  # cache hit


def test_no_cache_when_contextvar_none(clean_registry):
    """Non-cached tools skip all cache logic when no cache is set."""
    call_count = 0

    def run_tool(inputs, config=None, instance=None):
        nonlocal call_count
        call_count += len(inputs.items)
        return MockIterableOutput(
            results=[f"out_{x}" for x in inputs.items],
        )

    spec = _register_cacheable_iterable(clean_registry, "no-ctx-cache", run_tool)
    config = MockToolConfig(param1="v")

    # No cache set (contextvar is None) — every call dispatches
    spec.function(MockIterableInput(items=["a"]), config)
    assert call_count == 1

    spec.function(MockIterableInput(items=["a"]), config)
    assert call_count == 2  # no caching, called again


def test_non_cacheable_tool_skips_cache_logic(clean_registry, _setup_cache):
    """Non-cacheable tool runs func every time even with active cache."""
    call_count = 0

    @clean_registry.register(
        key="non-cacheable",
        label="Non Cacheable",
        category="test",
        input_class=MockToolInput,
        config_class=MockToolConfig,
        output_class=MockToolOutput,
        description="Non-cacheable tool",
        cacheable=False,
    )
    def run_tool(inputs, config=None, instance=None):
        nonlocal call_count
        call_count += 1
        return MockToolOutput(result=f"processed_{inputs.input_data}")

    spec = clean_registry.get("non-cacheable")
    inputs = MockToolInput(input_data="test")
    config = MockToolConfig(param1="v")

    spec.function(inputs, config)
    assert call_count == 1

    spec.function(inputs, config)
    assert call_count == 2  # no caching


def test_cacheable_on_toolspec(clean_registry):
    """cacheable flag is stored on ToolSpec."""

    @clean_registry.register(
        key="cacheable-spec",
        label="Cacheable Spec",
        category="test",
        input_class=MockToolInput,
        config_class=MockToolConfig,
        output_class=MockToolOutput,
        description="Test cacheable on spec",
        cacheable=True,
    )
    def run_tool(inputs, config=None, instance=None):
        return MockToolOutput(result="ok")

    spec = clean_registry.get("cacheable-spec")
    assert spec.cacheable is True

    @clean_registry.register(
        key="non-cacheable-spec",
        label="Non Cacheable Spec",
        category="test",
        input_class=MockToolInput,
        config_class=MockToolConfig,
        output_class=MockToolOutput,
        description="Test non-cacheable on spec",
    )
    def run_tool2(inputs, config=None, instance=None):
        return MockToolOutput(result="ok")

    spec2 = clean_registry.get("non-cacheable-spec")
    assert spec2.cacheable is False


# ── preprocess hook ──────────────────────────────────────────────────────────


class PreprocessConfig(BaseConfig):
    """Config that transforms inputs via preprocess."""

    prefix: str = ConfigField(default="PRE", description="Prefix to add")

    def preprocess(self, inputs):
        return inputs.model_copy(
            update={"input_data": f"{self.prefix}_{inputs.input_data}"}
        )


def test_preprocess_hook_called_by_wrapper(clean_registry):
    """The @tool wrapper must call config.preprocess(inputs) before execution."""

    @clean_registry.register(
        key="preprocess-test",
        label="Preprocess Test",
        category="test",
        input_class=MockToolInput,
        config_class=PreprocessConfig,
        output_class=MockToolOutput,
        description="Test preprocess hook",
    )
    def run_tool(inputs, config=None, instance=None):
        return MockToolOutput(result=inputs.input_data)

    inputs = MockToolInput(input_data="hello")
    config = PreprocessConfig(prefix="PRE")
    result = run_tool(inputs, config)
    assert result.result == "PRE_hello"


def test_preprocess_default_noop(clean_registry):
    """BaseConfig.preprocess() returns inputs unchanged."""

    @clean_registry.register(
        key="preprocess-noop-test",
        label="Preprocess Noop Test",
        category="test",
        input_class=MockToolInput,
        config_class=MockToolConfig,
        output_class=MockToolOutput,
        description="Test default preprocess is noop",
    )
    def run_tool(inputs, config=None, instance=None):
        return MockToolOutput(result=inputs.input_data)

    inputs = MockToolInput(input_data="unchanged")
    config = MockToolConfig(param1="x")
    result = run_tool(inputs, config)
    assert result.result == "unchanged"
