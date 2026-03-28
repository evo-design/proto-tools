"""tests/style_consistency_tests/test_tool_input_output_consistency.py

Tests for tool input and output consistency."""

import inspect

import pytest

from bio_programming_tools.tools.tool_registry import ToolRegistry
from bio_programming_tools.utils.tool_io import BaseToolInput, BaseToolOutput

from .helpers import field_description_is_valid, find_missing_fields_in_docstring

_MAX_FIELD_DESCRIPTION_LENGTH = 100


def _list_tool_inputs_and_outputs():
    """List of all tool inputs and outputs."""
    full_list = []
    for tool in ToolRegistry.list_all():
        full_list.append((tool.input_model, tool.output_model))
    return full_list


def _list_tool_input_models():
    """List of all tool input models."""
    return [spec.input_model for spec in ToolRegistry.list_all()]


# ── InputField consistency ─────────────────────────────────────────────────

@pytest.mark.parametrize("tool_input", _list_tool_input_models())
def test_tool_input_uses_input_field(tool_input):
    """Ensure every Input field uses InputField() instead of plain Field()."""
    for field_name, field_info in tool_input.model_fields.items():
        json_schema_extra = field_info.json_schema_extra or {}
        assert json_schema_extra.get("_field_type") == "InputField", (
            f"{tool_input.__name__}.{field_name} must use InputField() instead of Field()."
        )


# ── Input and output consistency ────────────────────────────────────────────

@pytest.mark.parametrize("tool_input, tool_output", _list_tool_inputs_and_outputs())
def test_tool_input_and_output_consistency(tool_input, tool_output):
    """Test if tool inputs and outputs are defined consistently."""
    # Ensure tool input inherits from BaseToolInput
    assert issubclass(
        tool_input, BaseToolInput
    ), f"Tool input {tool_input} is not a subclass of BaseToolInput"
    # Ensure tool output inherits from BaseToolOutput
    assert issubclass(
        tool_output, BaseToolOutput
    ), f"Tool output {tool_output} is not a subclass of BaseToolOutput"

    # Ensure docstring exists and is not empty for both tool input and output
    input_docstring = tool_input.__doc__
    assert input_docstring is not None, f"Tool input {tool_input.__name__} is missing docstring. "
    assert len(input_docstring) > 0, f"Tool input {tool_input.__name__} docstring is empty. "
    output_docstring = tool_output.__doc__
    assert (
        output_docstring is not None
    ), f"Tool output {tool_output.__name__} is missing docstring. "
    assert len(output_docstring) > 0, f"Tool output {tool_output.__name__} docstring is empty. "

    # Iterate through input fields and ensure they are defined consistently
    for field_name, field_info in tool_input.model_fields.items():
        description_error = field_description_is_valid(field_info.description, _MAX_FIELD_DESCRIPTION_LENGTH)
        assert description_error == "", (
            f"Tool input {tool_input.__name__} has field {field_name} {description_error}. "
            "Ensure: Field(..., description='Brief explanation for tooltip')"
        )

    # Iterate through output fields and ensure they are defined consistently
    for field_name, field_info in tool_output.model_fields.items():
        description_error = field_description_is_valid(field_info.description, _MAX_FIELD_DESCRIPTION_LENGTH)
        assert description_error == "", (
            f"Tool output {tool_output.__name__} has field {field_name} {description_error}. "
            "Ensure: Field(..., description='Brief explanation for tooltip')"
        )

    # DOCUMENTATION CHECK: Ensure that all fields are mentioned in the docstring
    missing_fields = find_missing_fields_in_docstring(input_docstring, tool_input.model_fields.keys())
    assert len(missing_fields) == 0, (
        f"Tool input {tool_input.__name__} is missing the following fields in the docstring: {missing_fields}. "
        "Ensure: Field(..., description='Brief explanation for tooltip')"
    )
    missing_fields = find_missing_fields_in_docstring(output_docstring, tool_output.model_fields.keys())
    # Remove standardized output fields
    standard_tool_output_fields = (
        "tool_id",
        "execution_time",
        "timestamp",
        "success",
        "warnings",
        "errors",
        "metadata",
    )
    missing_fields = [field for field in missing_fields if field not in standard_tool_output_fields]
    assert len(missing_fields) == 0, (
        f"Tool output {tool_output.__name__} is missing the following fields in the docstring: {missing_fields}. "
        "Ensure: Field(..., description='Brief explanation for tooltip')"
    )

    # Ensure tool output is concrete (all abstract methods implemented)
    assert not inspect.isabstract(tool_output), (
        f"Tool output {tool_output.__name__} is abstract. "
        f"Missing implementations for abstract methods: "
        f"{sorted(tool_output.__abstractmethods__)}"
    )


