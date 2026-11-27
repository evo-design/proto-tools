"""tests/style_consistency_tests/test_field_docs.py.

Tests that every tool input/config/output field resolves to per-field
documentation extracted from class docstrings. This text is the source for the
schema's ``x-proto-doc`` annotation, so the coverage check guards that surface.
"""

import pytest

from proto_tools.tools.tool_registry import ToolRegistry
from proto_tools.utils.tool_docs import field_docs_from_docstrings
from proto_tools.utils.tool_io import _OUTPUT_METADATA_FIELDS


def _input_models():
    """Input models of all registered tools."""
    return [spec.input_model for spec in ToolRegistry.list_all()]


def _config_models():
    """Config models of all registered tools."""
    return [spec.config_model for spec in ToolRegistry.list_all()]


def _output_models():
    """Output models of all registered tools."""
    return [spec.output_model for spec in ToolRegistry.list_all()]


def _missing_docs(model, exclude=frozenset()):
    """Return fields of ``model`` that resolve to no docstring documentation."""
    docs = field_docs_from_docstrings(model)
    return [name for name in model.model_fields if name not in exclude and not docs.get(name)]


@pytest.mark.parametrize("model", _input_models())
def test_every_input_field_has_docstring_doc(model):
    """Every input field must resolve to non-empty docstring documentation."""
    missing = _missing_docs(model)
    assert not missing, (
        f"{model.__name__} has no docstring documentation for fields: {missing}. "
        "Document each field in its class's Google-style 'Attributes:' section."
    )


@pytest.mark.parametrize("model", _config_models())
def test_every_config_field_has_docstring_doc(model):
    """Every config field must resolve to non-empty docstring documentation.

    ``field_docs_from_docstrings`` walks the MRO and parses each class's own
    Google-style ``Attributes:`` section. Every field, including those inherited
    from a base config, must produce extractable help text so the schema's
    per-field ``x-proto-doc`` annotation is always populated.
    """
    missing = _missing_docs(model)
    assert not missing, (
        f"{model.__name__} has no docstring documentation for fields: {missing}. "
        "Document each field in its class's Google-style 'Attributes:' section."
    )


@pytest.mark.parametrize("model", _output_models())
def test_every_output_field_has_docstring_doc(model):
    """Every output field must resolve to non-empty docstring documentation.

    Framework-managed metadata fields (``tool_id``, ``execution_time``, etc.) are
    excluded — they are not part of a tool's authored output docstring.
    """
    missing = _missing_docs(model, exclude=_OUTPUT_METADATA_FIELDS)
    assert not missing, (
        f"{model.__name__} has no docstring documentation for fields: {missing}. "
        "Document each field in its class's Google-style 'Attributes:' section."
    )
