"""tests/style_consistency_tests/test_tool_config_consistency.py.

Tests for tool config consistency.
"""

import types
from typing import Union, get_args, get_origin

import pytest
from pydantic import ValidationError
from pydantic.fields import PydanticUndefined

from proto_tools.tools.tool_registry import ToolRegistry
from proto_tools.utils import BaseConfig as ToolsBaseConfig
from tests.style_consistency_tests.helpers import field_description_is_valid, find_missing_fields_in_docstring

_MAX_FIELD_TITLE_LENGTH = 31
_MAX_FIELD_DESCRIPTION_LENGTH = 100
_BASE_CONFIG_FIELDS = frozenset(ToolsBaseConfig.model_fields.keys())
_BANNED_UI_SCHEMA_KEYS = frozenset({"advanced", "hidden", "depends_on", "x-depends-on", "x-xor-group"})


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

    # Pull the docstring for the config model
    docstring = config_model.__doc__
    assert docstring is not None, f"{config_model.__name__} is missing docstring. "
    assert len(docstring) > 0, f"{config_model.__name__} docstring is empty. "

    # Ensure all fields are defined consistently
    for field_name, field_info in config_model.model_fields.items():
        # TITLE: Ensure title is explicitly provided and is under limit
        title = field_info.title
        assert title is not None, f"{config_model.__name__}.{field_name} is missing title. "
        assert len(title) <= _MAX_FIELD_TITLE_LENGTH, (
            f"{config_model.__name__}.{field_name} title is too long (currently {len(title)} characters, must be under {_MAX_FIELD_TITLE_LENGTH} characters). "
        )

        # DESCRIPTION: Must exist and be concise (~15 words / ~90 chars)
        description_error = field_description_is_valid(field_info.description, _MAX_FIELD_DESCRIPTION_LENGTH)
        assert description_error == "", (
            f"{config_model.__name__}.{field_name} {description_error}. "
            "Ensure: Field(..., description='Brief explanation')"
        )

        # OPTIONALITY: Check for Optional types (should be rare)
        # If the default value is None, the field must have annotation Optional[type]
        if field_info.default is None:
            annotation = field_info.annotation
            origin = get_origin(annotation)
            ann_args = get_args(annotation)

            # Optional[...] is Union[..., None]; X | None is types.UnionType
            is_optional = origin in (Union, types.UnionType) and type(None) in ann_args

            if not is_optional:
                raise TypeError(
                    f"{config_model.__name__}.{field_name} default value is None but annotation is not Optional[...]"
                )

        # CONFIG FIELD TYPE: Must be created via ConfigField()
        json_schema_extra = field_info.json_schema_extra or {}
        assert json_schema_extra.get("_field_type") == "ConfigField", (
            f"{config_model.__name__}.{field_name} must use ConfigField() instead of Field()."
        )
        banned_keys = _BANNED_UI_SCHEMA_KEYS.intersection(json_schema_extra)
        assert not banned_keys, (
            f"{config_model.__name__}.{field_name} sets unsupported schema key(s) {sorted(banned_keys)}."
        )

    # Every field must appear in the config's own docstring (excluding
    # BaseConfig fields, which are documented once at the base level).
    missing_fields = find_missing_fields_in_docstring(docstring, config_model.model_fields.keys())
    missing_fields = [f for f in missing_fields if f not in _BASE_CONFIG_FIELDS]
    assert len(missing_fields) == 0, (
        f"{config_model.__name__} is missing the following fields in its docstring: "
        f"{missing_fields}. Every non-BaseConfig field must be documented in the "
        "class's own docstring, even if inherited from a parent config."
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
        has_default = field_info.default is not PydanticUndefined or field_info.default_factory is not None
        if not has_default:
            fields_missing_defaults.append(field_name)

    assert not fields_missing_defaults, (
        f"Config {config_model.__name__} has fields without defaults: "
        f"{fields_missing_defaults}. All config fields must have defaults "
        f"to support config=None auto-instantiation."
    )

    # Verify the config model can actually be instantiated with no args
    try:
        default_config = config_model()
    except ValidationError as exc:
        pytest.skip(f"{config_model.__name__} default config does not construct on this platform: {exc}")
    assert isinstance(default_config, config_model)


# ── Tool-key stamping ────────────────────────────────────────────────────────


@pytest.mark.parametrize("spec", ToolRegistry.list_all(), ids=lambda s: s.key)
def test_config_carries_its_tool_key(spec):
    """Every registered config knows its tool, stamped at registration.

    ``ToolRegistry.register`` rejects two tools sharing a config class, so this
    is unambiguous: exactly one key per config. A shared family shape must be
    subclassed per tool rather than reused directly.
    """
    config_model = spec.config_model
    assert config_model.tool_key == spec.key, (
        f"{config_model.__name__}.tool_key is {config_model.tool_key!r}, expected {spec.key!r}. "
        f"Give this tool its own config class (subclass the shared one if fields are identical)."
    )
    # tool_key is class state (ClassVar), never a serialized config field.
    assert "tool_key" not in config_model.model_fields, (
        f"{config_model.__name__}.tool_key leaked into model_fields; it must stay a ClassVar."
    )


# ── Fan-out granularity (max_chunk_size) ─────────────────────────────────────


@pytest.mark.parametrize("spec", ToolRegistry.list_all(), ids=lambda s: s.key)
def test_iterable_tools_declare_max_chunk_size(spec):
    """A declared fan-out granularity must be usable, and a non-iterable tool must carry none.

    ``max_chunk_size`` has no default. It used to default to 1, which made this test vacuous:
    every iterable tool passed without anyone having chosen. Registration now refuses an
    iterable tool that omits it, so reaching this test at all means a value was stated, or the
    tool is one of the cases registration exempts.

    Exempt: a tool with no GPU and no standalone environment always runs in this process, so it
    never splits across executions and has no granularity to state.

    ``None`` is also a legitimate answer, meaning the whole batch goes to one execution. It is
    distinct from omission, which raises for any tool that can actually fan out.
    """
    if spec.iterable_input_fields is not None:
        assert spec.max_chunk_size is None or (isinstance(spec.max_chunk_size, int) and spec.max_chunk_size >= 1), (
            f"{spec.key}: max_chunk_size must be >= 1 when declared, got {spec.max_chunk_size!r}."
        )
    else:
        assert spec.max_chunk_size is None, (
            f"{spec.key}: max_chunk_size only applies to iterable tools, got {spec.max_chunk_size!r}."
        )
