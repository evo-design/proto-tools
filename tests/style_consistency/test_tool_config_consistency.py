"""
Pulls all tool config models from registered tools and checks for field definition
consistency.
"""
from __future__ import annotations

from typing import List, Type, Union, get_args, get_origin

import pytest

from bio_programming_tools.tools.tool_registry import ToolRegistry
from bio_programming_tools.tools.utils import BaseConfig as ToolsBaseConfig

# Defines the maximum length of a field title in characters
MAX_FIELD_TITLE_LENGTH = 31

# Defines the maximum length of a field description in characters
MAX_FIELD_DESCRIPTION_LENGTH = 100


def list_of_all_tool_config_models() -> List[Type]:
    """
    List of all config models of registered tools.
    """
    return [spec.config_model for spec in ToolRegistry.list_all()]


@pytest.mark.parametrize(
    "config_model", [config_model for config_model in list_of_all_tool_config_models()]
)
def test_tool_config_consistency(config_model: Type):
    """
    Determines if tool config models are defined consistently throughout the codebase
    for consistency of the API and client.
    """
    # Check if config_model is subclass of ToolsBaseConfig
    assert issubclass(config_model, ToolsBaseConfig), (
        f"Config model {config_model} is not a subclass of ToolsBaseConfig"
    )

    # Pull the model schema and ensure fields are defined consistently
    schema = config_model.model_json_schema()
    required_fields = set(schema.get("required", []))

    # Pull the docstring for the config model
    docstring = config_model.__doc__
    assert docstring is not None, f"{config_model.__name__} is missing docstring. "
    assert len(docstring) > 0, f"{config_model.__name__} docstring is empty. "

    # Ensure all fields are defined consistently
    for field_name, field_info in config_model.model_fields.items():

        # TITLE: Ensure title is explicitly provided and is under 45 characters
        title = field_info.title
        assert title is not None, f"{config_model.__name__}.{field_name} is missing title. "
        assert (
            len(title) <= MAX_FIELD_TITLE_LENGTH
        ), f"{config_model.__name__}.{field_name} title is too long (currently {len(title)} characters, must be under {MAX_FIELD_TITLE_LENGTH} characters). "

        # DESCRIPTION: Must exist and be concise (~15 words / ~90 chars for tooltip)
        description_error = _field_description_is_valid(field_info.description)
        assert description_error == "", (
            f"{config_model.__name__}.{field_name} {description_error}. "
            "Ensure: Field(..., description='Brief explanation for tooltip')"
        )

        # OPTIONALITY: Check for Optional types (should be rare)
        # If the default value is None, the field must have annotation Optional[type]
        if field_info.default is None:
            annotation = field_info.annotation
            origin = get_origin(annotation)
            ann_args = get_args(annotation)

            # Optional[...] is Union[..., None]
            is_optional = origin is Union and type(None) in ann_args

            if not is_optional:
                raise TypeError(
                    f"{config_model.__name__}.{field_name} default value is None but annotation is not Optional[...]"
                )

        # ADVANCED FLAG: Must exist and be a boolean
        json_schema_extra = field_info.json_schema_extra or {}
        assert "advanced" in json_schema_extra, (
            f"{config_model.__name__}.{field_name} missing 'advanced' flag. "
            "Add: Field(..., json_schema_extra={{'advanced': False}})"
        )
        assert isinstance(json_schema_extra["advanced"], bool), (
            f"{config_model.__name__}.{field_name} 'advanced' flag is not a boolean. "
            "Add: Field(..., json_schema_extra={{'advanced': False}})"
        )

        assert "hidden" in json_schema_extra, (
            f"{config_model.__name__}.{field_name} missing 'hidden' flag. "
            "Add: Field(..., json_schema_extra={{'hidden': False}})"
        )
        assert isinstance(json_schema_extra["hidden"], bool), (
            f"{config_model.__name__}.{field_name} 'hidden' flag is not a boolean. "
            "Add: Field(..., json_schema_extra={{'hidden': False}})"
        )

        # Pull advanced and hidden flags
        advanced = json_schema_extra.get("advanced", False)
        hidden = json_schema_extra.get("hidden", False)

        # Advanced and hidden flags must be false if the field is required
        if field_name in required_fields:
            assert not advanced, (
                f"{config_model.__name__}.{field_name} 'advanced' flag cannot be True if the field is required. "
                "Remove the 'advanced' flag."
            )
            assert not hidden, (
                f"{config_model.__name__}.{field_name} 'hidden' flag cannot be True if the field is required. "
                "Remove the 'hidden' flag."
            )

    # DOCUMENTATION CHECK: Ensure that all fields are mentioned in the docstring
    missing_fields = _find_missing_fields_in_docstring(
        docstring, config_model.model_fields.keys()
    )
    assert len(missing_fields) == 0, (
        f"{config_model.__name__} is missing the following fields in the docstring: {missing_fields}. "
        "Add: Field(..., description='Brief explanation for tooltip')"
    )


def _field_description_is_valid(description: str) -> str:
    """
    Check if the description is under MAX_FIELD_DESCRIPTION_LENGTH characters.
    """
    if description is None:
        return "is None"
    if len(description) > MAX_FIELD_DESCRIPTION_LENGTH:
        return f"is too long (currently {len(description)} characters, must be under {MAX_FIELD_DESCRIPTION_LENGTH} characters)"
    if not description.strip():
        return "description is empty or just whitespace"
    if "\n" in description:
        return "description contains newline characters. Please use single line descriptions."
    return ""


def _find_missing_fields_in_docstring(docstring: str, field_names: List[str]) -> List[str]:
    """
    Find missing fields in the docstring.
    """
    missing_fields = []
    for field_name in field_names:
        if field_name not in docstring:
            missing_fields.append(field_name)
    return missing_fields
