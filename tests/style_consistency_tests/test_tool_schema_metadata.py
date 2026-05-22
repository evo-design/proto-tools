"""tests/style_consistency_tests/test_tool_schema_metadata.py.

For every registered tool, walk the JSON Schema of its Input, Config, and
Output models — including every nested ``$defs`` model — and assert each
property has a non-empty ``title`` and ``description``.

Complements:
    * ``test_tool_config_consistency.py``: per-class field checks on Config.
    * ``test_tool_input_output_consistency.py``: per-class field checks on Input/Output.

Whereas the above walk Pydantic ``model_fields`` of the top-level class,
this test walks the resolved JSON Schema and follows ``$ref`` into shared
data submodels — so nested model fields get the same enforcement without
forcing every submodel to use a tool-shaped helper.
"""

from collections.abc import Iterator
from typing import Any

import pytest

from proto_tools.tools.tool_registry import ToolRegistry
from tests.style_consistency_tests.helpers import field_description_is_valid

_MAX_FIELD_TITLE_LENGTH = 31
_MAX_FIELD_DESCRIPTION_LENGTH = 100


def _walk_schema(schema: dict[str, Any], path: str) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield ``(path, prop_name, prop_dict)`` for every property in a JSON schema and its ``$defs``."""
    for name, prop in schema.get("properties", {}).items():
        yield path, name, prop
    for defname, defschema in schema.get("$defs", {}).items():
        for name, prop in defschema.get("properties", {}).items():
            yield f"{path}::$defs::{defname}", name, prop


def _check_property(model_label: str, surface: str, path: str, name: str, prop: dict[str, Any]) -> list[str]:
    """Return a list of error messages describing missing/invalid metadata on ``prop``."""
    errors: list[str] = []
    title = prop.get("title")
    if not title:
        errors.append(f"{model_label} ({surface}) {path}.{name}: missing title")
    elif len(title) > _MAX_FIELD_TITLE_LENGTH:
        errors.append(
            f"{model_label} ({surface}) {path}.{name}: title is too long "
            f"({len(title)} chars, must be ≤ {_MAX_FIELD_TITLE_LENGTH})"
        )
    description_error = field_description_is_valid(prop.get("description"), _MAX_FIELD_DESCRIPTION_LENGTH)
    if description_error:
        errors.append(f"{model_label} ({surface}) {path}.{name}: description {description_error}")
    return errors


def _list_tool_specs():
    """Return all registered tool specs (one parameter per tool)."""
    return list(ToolRegistry.list_all())


def _ids(specs):
    return [spec.key for spec in specs]


_TOOL_SPECS = _list_tool_specs()


@pytest.mark.parametrize("spec", _TOOL_SPECS, ids=_ids(_TOOL_SPECS))
def test_tool_schema_has_title_and_description(spec):
    """Every property in Input/Config/Output schemas (including $defs) must have title + description."""
    errors: list[str] = []
    for surface, model in (
        ("input", spec.input_model),
        ("config", spec.config_model),
        ("output", spec.output_model),
    ):
        schema = model.model_json_schema()
        for path, name, prop in _walk_schema(schema, surface):
            errors.extend(_check_property(model.__name__, surface, path, name, prop))

    if errors:
        message = f"Tool {spec.key!r} has {len(errors)} field-metadata violation(s):\n  " + "\n  ".join(errors)
        message += (
            "\n\nFix: add title= and description= on every Pydantic field exposed via the JSON Schema. "
            "Use InputField/ConfigField for direct Input/Config classes; bare "
            "pydantic.Field(title=..., description=...) is acceptable on shared nested submodels."
        )
        pytest.fail(message)
