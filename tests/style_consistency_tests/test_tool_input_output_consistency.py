"""Tests for tool input and output consistency."""

import inspect

import pytest

from bio_programming_tools.tools.tool_registry import ToolRegistry
from bio_programming_tools.utils.tool_io import BaseToolInput, BaseToolOutput

_MAX_FIELD_DESCRIPTION_LENGTH = 100


def _list_tool_inputs_and_outputs():
    """List of all tool inputs and outputs."""
    full_list = []
    for tool in ToolRegistry.list_all():
        full_list.append((tool.input_model, tool.output_model))
    return full_list


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
        description_error = _field_description_is_valid(field_info.description)
        assert description_error == "", (
            f"Tool input {tool_input.__name__} has field {field_name} {description_error}. "
            "Ensure: Field(..., description='Brief explanation for tooltip')"
        )

    # Iterate through output fields and ensure they are defined consistently
    for field_name, field_info in tool_output.model_fields.items():
        description_error = _field_description_is_valid(field_info.description)
        assert description_error == "", (
            f"Tool output {tool_output.__name__} has field {field_name} {description_error}. "
            "Ensure: Field(..., description='Brief explanation for tooltip')"
        )

    # DOCUMENTATION CHECK: Ensure that all fields are mentioned in the docstring
    missing_fields = _find_missing_fields_in_docstring(input_docstring, tool_input.model_fields.keys())
    assert len(missing_fields) == 0, (
        f"Tool input {tool_input.__name__} is missing the following fields in the docstring: {missing_fields}. "
        "Ensure: Field(..., description='Brief explanation for tooltip')"
    )
    missing_fields = _find_missing_fields_in_docstring(output_docstring, tool_output.model_fields.keys())
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


# ── Helpers ─────────────────────────────────────────────────────────────────

def _field_description_is_valid(description):
    """Check if the description is under _MAX_FIELD_DESCRIPTION_LENGTH characters."""
    if description is None:
        return "is None"
    if len(description) > _MAX_FIELD_DESCRIPTION_LENGTH:
        return f"is too long (currently {len(description)} characters, must be under {_MAX_FIELD_DESCRIPTION_LENGTH} characters)"
    if not description.strip():
        return "description is empty or just whitespace"
    if "\n" in description:
        return "description contains newline characters. Please use single line descriptions."
    return ""


def _find_missing_fields_in_docstring(docstring, field_names):
    """Find missing fields in the docstring."""
    missing_fields = []
    for field_name in field_names:
        if field_name not in docstring:
            missing_fields.append(field_name)
    return missing_fields
