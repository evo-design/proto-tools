"""Tests for tool config consistency."""

from typing import Union, get_args, get_origin

import pytest
from pydantic.fields import PydanticUndefined

from bio_programming_tools.tools.tool_registry import ToolRegistry
from bio_programming_tools.utils import BaseConfig as ToolsBaseConfig

_MAX_FIELD_TITLE_LENGTH = 31
_MAX_FIELD_DESCRIPTION_LENGTH = 100


def _list_of_all_tool_config_models():
    """List of all config models of registered tools."""
    return [spec.config_model for spec in ToolRegistry.list_all()]


# ── Config field consistency ────────────────────────────────────────────────

@pytest.mark.parametrize("config_model", _list_of_all_tool_config_models())
def test_tool_config_consistency(config_model):
    """Test if tool config models are defined consistently."""
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

        # TITLE: Ensure title is explicitly provided and is under limit
        title = field_info.title
        assert title is not None, f"{config_model.__name__}.{field_name} is missing title. "
        assert (
            len(title) <= _MAX_FIELD_TITLE_LENGTH
        ), f"{config_model.__name__}.{field_name} title is too long (currently {len(title)} characters, must be under {_MAX_FIELD_TITLE_LENGTH} characters). "

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

        # CONFIG FIELD TYPE: Must be created via ConfigField()
        json_schema_extra = field_info.json_schema_extra or {}
        assert json_schema_extra.get("_field_type") == "ConfigField", (
            f"{config_model.__name__}.{field_name} must use ConfigField() instead of Field()."
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

    # DOCUMENTATION CHECK: Every field must appear in at least one docstring
    # in the MRO (the class itself or a superclass). This mirrors how
    # documentation generation would collect field descriptions.
    missing_fields = _find_undocumented_fields(config_model)
    assert len(missing_fields) == 0, (
        f"{config_model.__name__} is missing the following fields in the docstring "
        f"(not found in any superclass docstring either): {missing_fields}. "
        "Add: Field(..., description='Brief explanation for tooltip')"
    )


# ── Default config instantiation ────────────────────────────────────────────

@pytest.mark.parametrize("config_model", _list_of_all_tool_config_models())
def test_tool_config_accepts_none(config_model):
    """Test that all tool configs can be instantiated with default values.

    This enables the @tool decorator to accept config=None and automatically
    instantiate a default config, allowing simpler API usage:
        run_tool(inputs)  # instead of run_tool(inputs, ToolConfig())

    All config fields must have defaults for this to work.
    """
    # Verify every field explicitly has a default value or default_factory
    fields_missing_defaults = []
    for field_name, field_info in config_model.model_fields.items():
        has_default = (
            field_info.default is not PydanticUndefined
            or field_info.default_factory is not None
        )
        if not has_default:
            fields_missing_defaults.append(field_name)

    assert not fields_missing_defaults, (
        f"Config {config_model.__name__} has fields without defaults: "
        f"{fields_missing_defaults}. All config fields must have defaults "
        f"to support config=None auto-instantiation."
    )

    # Verify the config model can actually be instantiated with no args
    default_config = config_model()
    assert isinstance(default_config, config_model)


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


def _find_undocumented_fields(config_model):
    """Return field names that don't appear in any docstring across the MRO."""
    # Collect all docstrings from the class and its superclasses
    all_docs = ""
    for cls in config_model.__mro__:
        if cls.__doc__:
            all_docs += cls.__doc__

    missing = []
    for field_name in config_model.model_fields:
        if field_name not in all_docs:
            missing.append(field_name)
    return missing
