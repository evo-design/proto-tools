"""
Tests for BaseToolOutput class.

Tests the base output class that all tool outputs should extend.
"""

from datetime import datetime

import pytest
from pydantic import Field, ValidationError

from tests.tool_tests.tool_infra_tests.test_export_functionality import (
    MockToolOutputBase,
)


class SimpleToolOutput(MockToolOutputBase):
    """Example tool output for testing"""
    result: str = Field(description="Simple result string")


class ComplexToolOutput(MockToolOutputBase):
    """Example complex tool output for testing"""
    sequences: list[str] = Field(description="Output sequences")
    scores: list[float] = Field(description="Quality scores")
    count: int = Field(description="Number of results")


def test_tool_io_creation():
    """Test basic creation of BaseToolOutput subclass"""
    output = SimpleToolOutput(
        tool_id="test-tool",
        execution_time=1.5,
        success=True,
        result="test result"
    )

    assert output.tool_id == "test-tool"
    assert output.execution_time == 1.5
    assert output.success is True
    assert output.result == "test result"
    assert isinstance(output.timestamp, datetime)
    assert output.warnings == []
    assert output.metadata == {}


def test_tool_io_with_optional_fields():
    """Test creation with all optional fields"""
    timestamp = datetime.now()
    output = SimpleToolOutput(
        tool_id="test-tool",
        execution_time=2.5,
        timestamp=timestamp,
        success=True,
        warnings=["Warning 1", "Warning 2"],
        metadata={"version": "1.0", "backend": "local"},
        result="test result"
    )

    assert output.timestamp == timestamp
    assert len(output.warnings) == 2
    assert output.warnings[0] == "Warning 1"
    assert output.metadata["version"] == "1.0"
    assert output.metadata["backend"] == "local"


def test_tool_io_execution_time_validation():
    """Test that execution_time must be non-negative"""
    # Negative execution time should fail
    with pytest.raises(ValidationError) as exc_info:
        SimpleToolOutput(
            tool_id="test",
            execution_time=-1.0,
            success=True,
            result="test"
        )
    assert "execution_time" in str(exc_info.value)

    # Zero execution time is valid
    output = SimpleToolOutput(
        tool_id="test",
        execution_time=0.0,
        success=True,
        result="test"
    )
    assert output.execution_time == 0.0


def test_tool_io_complex_subclass():
    """Test complex tool output with multiple fields"""
    output = ComplexToolOutput(
        tool_id="complex-tool",
        execution_time=10.5,
        success=True,
        sequences=["ATCG", "GCTA"],
        scores=[0.95, 0.87],
        count=2
    )

    assert output.sequences == ["ATCG", "GCTA"]
    assert output.scores == [0.95, 0.87]
    assert output.count == 2


def test_tool_io_json_serialization():
    """Test JSON serialization and deserialization"""
    output = SimpleToolOutput(
        tool_id="test-tool",
        execution_time=1.5,
        success=True,
        warnings=["test warning"],
        metadata={"key": "value"},
        result="test result"
    )

    # Serialize to JSON
    json_str = output.model_dump_json()
    assert isinstance(json_str, str)
    assert "test-tool" in json_str
    assert "test result" in json_str

    # Deserialize from JSON
    json_dict = output.model_dump()
    reconstructed = SimpleToolOutput(**json_dict)
    assert reconstructed.tool_id == output.tool_id
    assert reconstructed.result == output.result
    assert reconstructed.warnings == output.warnings


def test_tool_io_extra_fields_forbidden():
    """Test that extra fields are forbidden in outputs (unlike configs)"""
    output_dict = {
        "tool_id": "test-tool",
        "execution_time": 1.5,
        "success": True,
        "result": "test result",
        "extra_field": "extra value",  # Extra field which is forbidden
    }
    with pytest.raises(ValidationError) as exc_info:
        SimpleToolOutput(**output_dict)
    assert "extra_field" in str(exc_info.value)


def test_tool_io_failed_execution():
    """Test output for failed execution"""
    output = SimpleToolOutput(
        tool_id="failing-tool",
        execution_time=0.1,
        success=False,
        warnings=["Tool failed", "Check input parameters"],
        metadata={"error_code": 1, "error_message": "Invalid input"},
        result=""
    )

    assert output.success is False
    assert len(output.warnings) == 2
    assert output.metadata["error_code"] == 1


def test_tool_io_model_schema():
    """Test that JSON schema is generated correctly"""
    schema = SimpleToolOutput.model_json_schema()

    # Check required fields
    assert "properties" in schema
    assert "tool_id" in schema["properties"]
    assert "execution_time" in schema["properties"]
    assert "success" in schema["properties"]
    assert "result" in schema["properties"]

    # Check field descriptions
    assert "description" in schema["properties"]["tool_id"]
    assert "description" in schema["properties"]["execution_time"]

    # Check required list only contains result field
    assert "required" in schema
    assert "result" in schema["required"]


def test_tool_io_timestamp_auto_generation():
    """Test that timestamp is auto-generated if not provided"""
    before = datetime.now()
    output = SimpleToolOutput(
        tool_id="test-tool",
        execution_time=1.0,
        success=True,
        result="test"
    )
    after = datetime.now()

    # Timestamp should be between before and after
    assert before <= output.timestamp <= after


def test_tool_io_defaults():
    """Test default values for optional fields"""
    output = SimpleToolOutput(
        tool_id="test-tool",
        execution_time=1.0,
        success=True,
        result="test"
    )

    # Check defaults
    assert output.warnings == []
    assert output.metadata == {}
    assert isinstance(output.timestamp, datetime)


def test_tool_io_immutability_with_validation():
    """Test that validation occurs on assignment"""
    output = SimpleToolOutput(
        tool_id="test-tool",
        execution_time=1.0,
        success=True,
        result="test"
    )

    # Should be able to modify (validate_assignment=True)
    output.execution_time = 2.0
    assert output.execution_time == 2.0

    # But validation should still apply
    with pytest.raises(ValidationError):
        output.execution_time = -1.0  # Negative not allowed
