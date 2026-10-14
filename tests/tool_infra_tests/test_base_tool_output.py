"""tests/tool_infra_tests/test_base_tool_output.py.

Tests for BaseToolInput and BaseToolOutput.
"""

import pytest
from pydantic import BaseModel, Field, computed_field

from proto_tools.utils.tool_io import BaseToolInput, InputField, ToolExecutionError, _approx_equal_values
from tests.tool_infra_tests.test_export_functionality import MockToolOutputBase


class _SimpleToolOutput(MockToolOutputBase):
    """Example tool output for testing."""

    result: str = Field(description="Simple result string")


class _ComputedToolOutput(MockToolOutputBase):
    """Example output with derived serialized state."""

    items: list[str] = Field(default_factory=list, description="Output items")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def count(self) -> int:
        return len(self.items)


# ── Validation ───────────────────────────────────────────────────────────────


def test_extra_fields_warn_and_are_ignored():
    """Output hydration warns on unknown fields but still ignores them."""
    with pytest.warns(UserWarning, match="extra_field"):
        output = _SimpleToolOutput(
            tool_id="test-tool",
            execution_time=1.5,
            success=True,
            result="test result",
            extra_field="extra value",
        )

    assert output.result == "test result"
    assert "extra_field" not in output.model_dump()


# ── Serialization ────────────────────────────────────────────────────────────


def test_computed_field_json_round_trip(recwarn):
    """Computed fields emitted in JSON dumps do not block revalidation."""
    output = _ComputedToolOutput(items=["a", "b"])

    dumped = output.model_dump(mode="json")
    assert dumped["count"] == 2

    reconstructed = _ComputedToolOutput.model_validate(dumped)
    assert reconstructed.items == ["a", "b"]
    assert reconstructed.count == 2
    assert not recwarn


# ── Cache key (BaseToolInput) ───────────────────────────────────────────────


class _TestInput(BaseToolInput):
    sequences: list[str] = InputField(title="Sequences", description="Input sequences")
    device: str = InputField(default="cpu", title="Device", description="Device", include_in_key=False)


def test_cache_key_changes_with_key_fields():
    a = _TestInput(sequences=["MVLSP"])
    b = _TestInput(sequences=["MKTAY"])
    assert a.cache_key() != b.cache_key()


def test_cache_key_excludes_non_key_fields():
    a = _TestInput(sequences=["MVLSP"], device="cpu")
    b = _TestInput(sequences=["MVLSP"], device="cuda")
    assert a.cache_key() == b.cache_key()


# ── __getattr__ on failed output ────────────────────────────────────────────


@pytest.mark.parametrize(
    "errors,match",
    [
        (["something broke"], "something broke"),
        ([], "no errors recorded"),
    ],
    ids=["with-errors", "empty-errors"],
)
def test_getattr_failed_output(errors, match):
    output = _SimpleToolOutput.model_construct(success=False, errors=errors)
    with pytest.raises(ToolExecutionError, match=match):
        _ = output.result


# ── approx_equal ────────────────────────────────────────────────────────────


class _FloatOutput(MockToolOutputBase):
    score: float = Field(description="A score")


class _NestedModel(BaseModel):
    value: float = Field(description="Nested value")


class _NestedOutput(MockToolOutputBase):
    inner: _NestedModel = Field(description="Nested model")


def test_approx_equal_skips_metadata():
    """Different metadata fields don't cause mismatch."""
    a = _FloatOutput(tool_id="a", execution_time=1.0, score=1.0)
    b = _FloatOutput(tool_id="b", execution_time=99.0, score=1.0)
    a.approx_equal(b)


@pytest.mark.parametrize(
    "a_val,b_val,should_pass",
    [(1.0, 1.0 + 1e-6, True), (1.0, 2.0, False)],
    ids=["within-tolerance", "beyond-tolerance"],
)
def test_approx_equal_float(a_val, b_val, should_pass):
    if should_pass:
        _approx_equal_values(a_val, b_val, rtol=1e-4, atol=1e-5, path="test")
    else:
        with pytest.raises(AssertionError, match="Float mismatch"):
            _approx_equal_values(a_val, b_val, rtol=1e-4, atol=1e-5, path="test")


def test_approx_equal_nested_basemodel():
    a = _NestedOutput(inner=_NestedModel(value=1.0))
    b = _NestedOutput(inner=_NestedModel(value=1.0 + 1e-6))
    a.approx_equal(b)


@pytest.mark.parametrize(
    "a,b,match",
    [
        (1, "1", "Type mismatch"),
        ({"a": 1}, {"b": 1}, "Dict keys differ"),
        ([1, 2], [1], "Length mismatch"),
        ("a", "b", "Value mismatch"),
    ],
    ids=["type", "dict-keys", "list-length", "value"],
)
def test_approx_equal_values_error(a, b, match):
    with pytest.raises(AssertionError, match=match):
        _approx_equal_values(a, b, 1e-4, 1e-6, "test")


@pytest.mark.parametrize(
    "a,b",
    [
        ({"x": 1.0}, {"x": 1.0 + 1e-7}),
        ([1.0, 2.0], [1.0, 2.0 + 1e-7]),
    ],
    ids=["dict", "list"],
)
def test_approx_equal_values_recursive(a, b):
    _approx_equal_values(a, b, 1e-4, 1e-6, "test")
