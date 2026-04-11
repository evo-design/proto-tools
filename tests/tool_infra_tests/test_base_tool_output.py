"""tests/tool_infra_tests/test_base_tool_output.py.

Tests for BaseToolInput and BaseToolOutput.
"""

from datetime import datetime

import pytest
from pydantic import BaseModel, Field, ValidationError

from proto_tools.utils.tool_io import BaseToolInput, InputField, ToolExecutionError, _approx_equal_values
from tests.tool_infra_tests.test_export_functionality import MockToolOutputBase


class _SimpleToolOutput(MockToolOutputBase):
    """Example tool output for testing."""

    result: str = Field(description="Simple result string")


class _ComplexToolOutput(MockToolOutputBase):
    """Example complex tool output for testing."""

    sequences: list[str] = Field(description="Output sequences")
    scores: list[float] = Field(description="Quality scores")
    count: int = Field(description="Number of results")


# ── Validation ───────────────────────────────────────────────────────────────


def test_execution_time_rejects_negative():
    """execution_time must be non-negative."""
    with pytest.raises(ValidationError, match="execution_time"):
        _SimpleToolOutput(tool_id="test", execution_time=-1.0, success=True, result="test")


def test_execution_time_allows_zero():
    """Zero execution time is valid."""
    output = _SimpleToolOutput(tool_id="test", execution_time=0.0, success=True, result="test")
    assert output.execution_time == 0.0


def test_extra_fields_forbidden():
    """Extra fields are forbidden in outputs (unlike configs)."""
    with pytest.raises(ValidationError, match="extra_field"):
        _SimpleToolOutput(
            tool_id="test-tool",
            execution_time=1.5,
            success=True,
            result="test result",
            extra_field="extra value",
        )


def test_validate_assignment_rejects_negative_execution_time():
    """Validation occurs on field assignment too."""
    output = _SimpleToolOutput(tool_id="test-tool", execution_time=1.0, success=True, result="test")
    output.execution_time = 2.0
    assert output.execution_time == 2.0

    with pytest.raises(ValidationError, match="execution_time"):
        output.execution_time = -1.0


# ── Timestamp ────────────────────────────────────────────────────────────────


def test_timestamp_auto_generated():
    """Timestamp is auto-generated when not provided."""
    before = datetime.now()
    output = _SimpleToolOutput(tool_id="test-tool", execution_time=1.0, success=True, result="test")
    after = datetime.now()

    assert before <= output.timestamp <= after


# ── Serialization ────────────────────────────────────────────────────────────


def test_json_round_trip():
    """JSON serialization and deserialization preserve data."""
    output = _SimpleToolOutput(
        tool_id="test-tool",
        execution_time=1.5,
        success=True,
        warnings=["test warning"],
        metadata={"key": "value"},
        result="test result",
    )

    json_str = output.model_dump_json()
    assert "test-tool" in json_str
    assert "test result" in json_str

    json_dict = output.model_dump()
    reconstructed = _SimpleToolOutput(**json_dict)
    assert reconstructed.tool_id == output.tool_id
    assert reconstructed.result == output.result
    assert reconstructed.warnings == output.warnings


# ── Schema ───────────────────────────────────────────────────────────────────


def test_json_schema_includes_tool_specific_fields():
    """JSON schema contains tool-specific fields with descriptions."""
    schema = _SimpleToolOutput.model_json_schema()

    assert "result" in schema["properties"]
    assert "description" in schema["properties"]["tool_id"]
    assert "description" in schema["properties"]["execution_time"]
    assert "result" in schema["required"]


# ── Subclass fields ──────────────────────────────────────────────────────────


def test_complex_subclass_preserves_typed_fields():
    """Complex tool output with multiple typed fields works correctly."""
    output = _ComplexToolOutput(
        tool_id="complex-tool",
        execution_time=10.5,
        success=True,
        sequences=["ATCG", "GCTA"],
        scores=[0.95, 0.87],
        count=2,
    )

    assert output.sequences == ["ATCG", "GCTA"]
    assert output.scores == [0.95, 0.87]
    assert output.count == 2


# ── Cache key (BaseToolInput) ───────────────────────────────────────────────


class _TestInput(BaseToolInput):
    sequences: list[str] = InputField(description="Input sequences")
    device: str = InputField(default="cpu", description="Device", include_in_key=False)


def test_cache_key_deterministic():
    a = _TestInput(sequences=["MVLSP"])
    b = _TestInput(sequences=["MVLSP"])
    assert a.cache_key() == b.cache_key()


def test_cache_key_excludes_non_key_fields():
    a = _TestInput(sequences=["MVLSP"], device="cpu")
    b = _TestInput(sequences=["MVLSP"], device="cuda")
    assert a.cache_key() == b.cache_key()


# ── __getattr__ on failed output ────────────────────────────────────────────


def test_getattr_failed_output_raises_tool_execution_error():
    output = _SimpleToolOutput.model_construct(success=False, errors=["something broke"])
    with pytest.raises(ToolExecutionError, match="something broke"):
        _ = output.result


def test_getattr_failed_output_no_errors():
    output = _SimpleToolOutput.model_construct(success=False, errors=[])
    with pytest.raises(ToolExecutionError, match="no error messages"):
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
    [(1.0, 1.0 + 1e-6, True), (1.0, 2.0, False), (float("nan"), float("nan"), True)],
    ids=["within-tolerance", "beyond-tolerance", "nan-equal"],
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
