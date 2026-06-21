"""tests/style_consistency_tests/test_config_field_docs.py.

Tests that every config field resolves to per-field documentation extracted
from class docstrings. This text is the source for the config schema's
``x-proto-doc`` annotation, so the coverage check guards that surface.
"""

import pytest

from proto_tools.tools.tool_registry import ToolRegistry
from proto_tools.utils.tool_docs import field_docs_from_docstrings


def _list_of_all_tool_config_models():
    """List of all config models of registered tools."""
    return [spec.config_model for spec in ToolRegistry.list_all()]


@pytest.mark.parametrize("config_model", _list_of_all_tool_config_models())
def test_every_config_field_has_docstring_doc(config_model):
    """Every config field must resolve to non-empty docstring documentation.

    ``field_docs_from_docstrings`` walks the MRO and parses each class's own
    Google-style ``Attributes:`` section. Every field, including those inherited
    from a base config, must produce extractable help text so the schema's
    per-field ``x-proto-doc`` annotation is always populated.
    """
    docs = field_docs_from_docstrings(config_model)
    missing = [name for name in config_model.model_fields if not docs.get(name)]
    assert not missing, (
        f"{config_model.__name__} has no docstring documentation for fields: {missing}. "
        "Document each field in its class's Google-style 'Attributes:' section."
    )
