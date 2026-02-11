"""
test_tool_registry.py

Tests for ToolRegistry class.

Tests the tool registry system for decorator-based tool registration,
discovery, and schema generation.
"""

import time

import pytest
from pydantic import Field

from bio_programming_tools.utils.tool_io import BaseToolInput
from bio_programming_tools.tools.tool_registry import ToolRegistry, ToolSpec
from bio_programming_tools.utils import BaseConfig, ConfigField
from tests.tool_infra_tests.test_export_functionality import MockToolOutputBase


# Test fixtures: Mock tool inputs, configs and outputs
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
        input=MockToolInput,
        config=MockToolConfig,
        output=MockToolOutput,
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
        input=MockToolInput,
        config=MockToolConfig,
        output=MockToolOutput,
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
    with pytest.raises(ValueError) as exc_info:

        @clean_registry.register(
            key="duplicate-tool",  # Same key
            label="Duplicate Tool 2",
            input=AnotherMockToolInput,
            config=AnotherMockToolConfig,
            output=AnotherMockToolOutput,
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

    assert "already registered" in str(exc_info.value).lower()


def test_tool_registry_list_all(clean_registry):
    """Test listing all registered tools"""

    @clean_registry.register(
        key="tool-1",
        label="Tool 1",
        input=MockToolInput,
        config=MockToolConfig,
        output=MockToolOutput,
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
        input=AnotherMockToolInput,
        config=AnotherMockToolConfig,
        output=AnotherMockToolOutput,
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


def test_tool_registry_get_schema(clean_registry):
    """Test getting JSON schema for a tool"""

    @clean_registry.register(
        key="schema-tool",
        label="Schema Tool",
        input=MockToolInput,
        config=MockToolConfig,
        output=MockToolOutput,
        description="Schema tool",
    )
    def schema_tool(inputs: MockToolInput, config: MockToolConfig) -> MockToolOutput:
        return MockToolOutput(
            tool_id="schema-tool",
            execution_time=0.1,
            success=True,
            result=f"ok: {inputs.input_data}",
        )

    schema = clean_registry.get_config_schema("schema-tool")

    # Check schema structure
    assert "properties" in schema
    assert "param1" in schema["properties"]
    assert "param2" in schema["properties"]

    # Check param1 details
    assert schema["properties"]["param1"]["type"] == "string"
    assert "description" in schema["properties"]["param1"]

    # Check param2 details (has default and constraints)
    assert schema["properties"]["param2"]["type"] == "integer"
    assert schema["properties"]["param2"]["default"] == 10
    assert schema["properties"]["param2"]["minimum"] == 0


def test_tool_registry_get_unknown_tool(clean_registry):
    """Test that getting unknown tool raises error"""
    with pytest.raises(ValueError) as exc_info:
        clean_registry.get("non-existent-tool")

    assert "unknown" in str(exc_info.value).lower()
    assert "non-existent-tool" in str(exc_info.value)


def test_tool_registry_get_input_schema(clean_registry):
    """Test getting JSON schema for tool inputs"""

    @clean_registry.register(
        key="input-schema-tool",
        label="Input Schema Tool",
        input=MockToolInput,
        config=MockToolConfig,
        output=MockToolOutput,
        description="Input schema tool",
    )
    def input_schema_tool(inputs: MockToolInput, config: MockToolConfig) -> MockToolOutput:
        return MockToolOutput(
            tool_id="input-schema-tool",
            execution_time=0.1,
            success=True,
            result=f"ok: {inputs.input_data}"
        )

    input_schema = clean_registry.get_input_schema("input-schema-tool")

    # Check input schema structure
    assert "properties" in input_schema
    assert "input_data" in input_schema["properties"]

    # Check input_data details
    assert input_schema["properties"]["input_data"]["type"] == "string"
    assert "description" in input_schema["properties"]["input_data"]


def test_tool_registry_get_schemas(clean_registry):
    """Test getting both input and config schemas"""

    @clean_registry.register(
        key="both-schemas-tool",
        label="Both Schemas Tool",
        input=MockToolInput,
        config=MockToolConfig,
        output=MockToolOutput,
        description="Both schemas tool",
    )
    def both_schemas_tool(inputs: MockToolInput, config: MockToolConfig) -> MockToolOutput:
        return MockToolOutput(
            tool_id="both-schemas-tool",
            execution_time=0.1,
            success=True,
            result=f"ok: {inputs.input_data}"
        )

    schemas = clean_registry.get_schemas("both-schemas-tool")

    # Check structure
    assert "inputs" in schemas
    assert "config" in schemas
    assert "output" in schemas

    # Check input schema
    input_schema = schemas["inputs"]
    assert "properties" in input_schema
    assert "input_data" in input_schema["properties"]

    # Check config schema
    config_schema = schemas["config"]
    assert "properties" in config_schema
    assert "param1" in config_schema["properties"]
    assert "param2" in config_schema["properties"]


def test_tool_registry_decorator_populates_metadata(clean_registry):
    """Test that decorator automatically populates metadata fields"""

    @clean_registry.register(
        key="metadata-tool",
        label="Metadata Tool",
        input=MockToolInput,
        config=MockToolConfig,
        output=MockToolOutput,
        description="Tool that tests metadata population",
    )
    def metadata_tool(inputs: MockToolInput, config: MockToolConfig) -> MockToolOutput:
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
        input=MockToolInput,
        config=MockToolConfig,
        output=MockToolOutput,
        description="Tool that raises an exception",
    )
    def failing_tool(inputs: MockToolInput, config: MockToolConfig) -> MockToolOutput:
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
        input=MockToolInput,
        config=MockToolConfig,
        output=MockToolOutput,
        description="Tool that generates warnings",
    )
    def warning_tool(inputs: MockToolInput, config: MockToolConfig) -> MockToolOutput:
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
    from bio_programming_tools.utils.tool_io import ToolExecutionError

    @clean_registry.register(
        key="error-access-tool",
        label="Error Access Tool",
        input=MockToolInput,
        config=MockToolConfig,
        output=MockToolOutput,
        description="Tool that fails and tests error access",
    )
    def error_access_tool(
        inputs: MockToolInput, config: MockToolConfig
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
    with pytest.raises(ToolExecutionError) as exc_info:
        _ = result.result

    assert "Tool execution failed" in str(
        exc_info.value
    ), "ToolExecutionError should be raised when accessing unset fields on failed output"

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
        input=MockToolInput,
        config=MockToolConfig,
        output=MockToolOutput,
        description="Tool that succeeds and tests field access",
    )
    def success_access_tool(
        inputs: MockToolInput, config: MockToolConfig
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
