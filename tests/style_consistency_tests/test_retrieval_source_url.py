"""tests/style_consistency_tests/test_retrieval_source_url.py.

Consistency test: every database-retrieval output names its source.

Each tool in the ``database_retrieval`` category declares a required ``source_url: str`` on its
output, so a result can always be cited back to the request that produced it.
"""

import pytest

from proto_tools.tools.tool_registry import ToolRegistry

# Tools that carry their source somewhere other than a top-level output field.
_EXEMPT_TOOLS = {
    "ccd-lookup": "reads a bundled copy of the CCD offline",
    "sequence-fetch": "records source_url on each FetchedSequence and FetchedStructure",
}

_RETRIEVAL_SPECS = sorted(
    (spec for spec in ToolRegistry.list_all() if spec.category == "database_retrieval"),
    key=lambda spec: spec.key,
)


@pytest.mark.parametrize("spec", _RETRIEVAL_SPECS, ids=lambda spec: spec.key)
def test_retrieval_output_declares_required_source_url(spec) -> None:
    """Retrieval outputs carry a required ``source_url: str`` unless explicitly exempt."""
    if spec.key in _EXEMPT_TOOLS:
        pytest.skip(_EXEMPT_TOOLS[spec.key])
    field = spec.output_model.model_fields.get("source_url")
    assert field is not None, f"{spec.output_model.__name__} has no source_url field"
    assert field.annotation is str and field.is_required(), (
        f"{spec.output_model.__name__}.source_url must be a required str, got {field.annotation}"
    )


def test_exemptions_are_registered_retrieval_tools() -> None:
    """Keep the exemption list exact: every entry is a registered retrieval tool."""
    assert set(_EXEMPT_TOOLS) <= {spec.key for spec in _RETRIEVAL_SPECS}
