"""tests/tool_infra_tests/test_gradient_contract.py.

Cross-backend contract tests for every relaxed-sequence gradient tool.

These run against each registered ``*-gradient`` tool that inherits the shared
``GradientInput``, so a new gradient backend is covered the moment it is registered.
Coverage:

- Every gradient tool inherits the shared input (with a shrinking allowlist of
  tools still on their pre-standardization classes).
- ``temperature`` defaults to ``1.0`` and rejects non-positive / non-finite values.
- The relaxed state is discoverable, round-trips through nested lists, and keeps
  its vocabulary aligned with the output.
- ``temperature=None`` accepts only valid probability distributions, honoring
  fixed-context rows and per-position vocabulary restrictions.
- ``SequenceLogits.from_probabilities`` reconstructs the supplied distribution.

Pure validation, no model execution: CPU-only and environment-free.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import ValidationError

from proto_tools.tools.tool_registry import ToolRegistry
from proto_tools.utils.gradient_models import GradientInput, GradientOutput
from proto_tools.utils.sequence_logits import (
    ProteinSequenceLogits,
    SequenceLogits,
)

# Gradient tools still carrying their own pre-standardization input classes.
# This set must be empty before the gradient-standardization work merges; each
# migration removes one entry, and the completeness test below fails loudly if a
# tool is migrated (or added) without updating it.
_UNMIGRATED_GRADIENT_TOOLS: frozenset[str] = frozenset()


def _gradient_specs() -> list[Any]:
    """Return the registered tool specs whose input inherits the shared ``GradientInput``."""
    return sorted(
        (spec for spec in ToolRegistry.list_all() if issubclass(spec.input_model, GradientInput)),
        key=lambda spec: spec.key,
    )


_GRADIENT_SPECS = _gradient_specs()


def _uniform_probabilities(matrix: SequenceLogits, allowed: list[list[bool]] | None) -> list[list[float]]:
    """Build a valid probability-mode matrix: uniform over permitted columns, zero elsewhere.

    Fixed-context rows (no permitted column) become all-zero, which is what both temperature
    modes require of them.

    Args:
        matrix (SequenceLogits): Matrix whose shape to mirror.
        allowed (list[list[bool]] | None): Per-position permitted columns, or ``None``.

    Returns:
        list[list[float]]: Rows satisfying the probability contract for this matrix.
    """
    _, width = matrix.shape
    rows: list[list[float]] = []
    for index in range(len(matrix)):
        mask = allowed[index] if allowed is not None else [True] * width
        permitted = [column for column, ok in enumerate(mask) if ok]
        row = [0.0] * width
        for column in permitted:
            row[column] = 1.0 / len(permitted)
        rows.append(row)
    return rows


def _set_path(payload: dict[str, Any], path: str, value: Any) -> None:
    """Assign ``value`` at a dotted/indexed field path inside a ``model_dump`` payload.

    Args:
        payload (dict[str, Any]): Dumped input payload to mutate in place.
        path (str): Path as keyed by ``GradientInput.sequence_logits``, e.g. ``"logits"``,
            ``"antibody.heavy_chain"``, or ``"logits[0]"``.
        value (Any): Replacement value.
    """
    *parents, leaf = path.split(".")
    cursor: Any = payload
    for part in parents:
        cursor = cursor[part]
    if leaf.endswith("]"):
        name, _, index = leaf[:-1].partition("[")
        cursor[name][int(index)] = value
    else:
        cursor[leaf] = value


def _probability_payload(example: GradientInput) -> dict[str, Any]:
    """Return ``example``'s payload with every relaxed matrix replaced by a valid distribution."""
    payload = example.model_dump(exclude_none=True)
    for path, matrix in example.sequence_logits().items():
        _set_path(payload, path, _uniform_probabilities(matrix, example.allowed_tokens(path, matrix)))
    payload["temperature"] = None
    return payload


def _example(spec: Any) -> GradientInput:
    """Return a tool's registered example input, failing the test if it has none."""
    example = ToolRegistry.get_example_input(spec.key)
    assert example is not None, f"Tool {spec.key!r} has no example_input registered."
    return example


# ── Completeness ─────────────────────────────────────────────────────────────


def test_every_gradient_tool_uses_the_shared_input() -> None:
    """Every registered ``*-gradient`` tool inherits ``GradientInput``, bar the known holdouts."""
    gradient_keys = {spec.key for spec in ToolRegistry.list_all() if spec.key.endswith("-gradient")}
    migrated = {spec.key for spec in _GRADIENT_SPECS}
    unmigrated = gradient_keys - migrated

    assert unmigrated == set(_UNMIGRATED_GRADIENT_TOOLS), (
        "Gradient tools not inheriting the shared GradientInput changed.\n"
        f"  expected holdouts: {sorted(_UNMIGRATED_GRADIENT_TOOLS)}\n"
        f"  actual holdouts:   {sorted(unmigrated)}\n"
        "Update _UNMIGRATED_GRADIENT_TOOLS when migrating a tool, and empty it before merge."
    )


# ── Shared input contract ────────────────────────────────────────────────────


@pytest.mark.parametrize("spec", _GRADIENT_SPECS, ids=lambda spec: spec.key)
def test_gradient_tool_surfaces_match_the_contract(spec: Any) -> None:
    """Input inherits ``GradientInput``, output inherits ``GradientOutput``."""
    assert issubclass(spec.input_model, GradientInput)
    assert issubclass(spec.output_model, GradientOutput), (
        f"{spec.key!r} output {spec.output_model.__name__} must inherit GradientOutput "
        "so gradient/loss/metrics/vocab stay consistent across backends."
    )


@pytest.mark.parametrize("spec", _GRADIENT_SPECS, ids=lambda spec: spec.key)
def test_temperature_defaults_to_one(spec: Any) -> None:
    """``temperature`` is an optional float defaulting to 1.0 (logit-space gradients)."""
    field = spec.input_model.model_fields["temperature"]
    assert field.default == 1.0, (
        f"{spec.key!r} must default temperature to 1.0 so omitting it means "
        f"softmax(logits / 1.0); got {field.default!r}."
    )
    assert _example(spec).temperature == 1.0


@pytest.mark.parametrize("spec", _GRADIENT_SPECS, ids=lambda spec: spec.key)
@pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf"), float("nan")])
def test_temperature_rejects_nonpositive_and_nonfinite(spec: Any, temperature: float) -> None:
    """Only positive finite temperatures (or None) are accepted."""
    payload = _example(spec).model_dump(exclude_none=True) | {"temperature": temperature}
    with pytest.raises(ValidationError):
        spec.input_model(**payload)


@pytest.mark.parametrize("spec", _GRADIENT_SPECS, ids=lambda spec: spec.key)
def test_relaxed_state_is_discoverable_and_vocab_aligned(spec: Any) -> None:
    """Each tool exposes at least one relaxed matrix, rectangular over its own vocabulary."""
    matrices = _example(spec).sequence_logits()
    assert matrices, f"{spec.key!r} exposes no SequenceLogits via sequence_logits()."
    for path, matrix in matrices.items():
        length, width = matrix.shape
        assert length > 0 and width == len(matrix.vocab), f"{spec.key!r}.{path} has shape {matrix.shape}."
        assert all(len(row) == width for row in matrix), f"{spec.key!r}.{path} is not rectangular."


@pytest.mark.parametrize("spec", _GRADIENT_SPECS, ids=lambda spec: spec.key)
def test_nested_lists_parse_and_serialize_unchanged(spec: Any) -> None:
    """The relaxed state round-trips as plain nested lists — the JSON/MCP request shape."""
    example = _example(spec)
    dumped = example.model_dump(exclude_none=True)
    for path, matrix in example.sequence_logits().items():
        cursor: Any = dumped
        for part in path.split("."):
            if part.endswith("]"):
                name, _, index = part[:-1].partition("[")
                cursor = cursor[name][int(index)]
            else:
                cursor = cursor[part]
        assert cursor == matrix.root, f"{spec.key!r}.{path} did not serialize to its nested list."
        assert isinstance(cursor, list) and all(isinstance(row, list) for row in cursor)

    assert spec.input_model(**dumped).model_dump(exclude_none=True) == dumped


# ── Probability mode (temperature=None) ──────────────────────────────────────


@pytest.mark.parametrize("spec", _GRADIENT_SPECS, ids=lambda spec: spec.key)
def test_probability_mode_accepts_valid_distributions(spec: Any) -> None:
    """``temperature=None`` accepts rows that are uniform over their permitted columns."""
    parsed = spec.input_model(**_probability_payload(_example(spec)))
    assert parsed.temperature is None


@pytest.mark.parametrize("spec", _GRADIENT_SPECS, ids=lambda spec: spec.key)
@pytest.mark.parametrize("corruption", ["unnormalized", "negative"])
def test_probability_mode_rejects_invalid_distributions(spec: Any, corruption: str) -> None:
    """``temperature=None`` rejects rows that are unnormalized or negative."""
    example = _example(spec)
    payload = _probability_payload(example)
    path, matrix = next(iter(example.sequence_logits().items()))
    rows = _uniform_probabilities(matrix, example.allowed_tokens(path, matrix))
    target = next((index for index, row in enumerate(rows) if any(row)), None)
    assert target is not None, f"{spec.key!r}.{path} has no active row to corrupt."
    column = next(index for index, value in enumerate(rows[target]) if value > 0.0)

    if corruption == "unnormalized":
        # 1e-3 is three orders of magnitude outside the 1e-6 tolerance.
        rows[target][column] += 1e-3
    else:
        rows[target][column] = -rows[target][column]
    _set_path(payload, path, rows)

    with pytest.raises(ValidationError):
        spec.input_model(**payload)


@pytest.mark.parametrize("spec", _GRADIENT_SPECS, ids=lambda spec: spec.key)
def test_logit_mode_does_not_apply_probability_rules(spec: Any) -> None:
    """A positive temperature accepts unnormalized, negative logits."""
    example = _example(spec)
    payload = example.model_dump(exclude_none=True) | {"temperature": 1.0}
    for path, matrix in example.sequence_logits().items():
        allowed = example.allowed_tokens(path, matrix)
        rows = _uniform_probabilities(matrix, allowed)
        # Shift only the active rows; fixed-context rows must stay zero in both modes.
        _set_path(payload, path, [[value - 3.0 for value in row] if any(row) else row for row in rows])
    assert spec.input_model(**payload).temperature == 1.0


# ── SequenceLogits ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("temperature", [0.5, 1.0, 3.0])
def test_from_probabilities_reconstructs_the_distribution(temperature: float) -> None:
    """``softmax(from_probabilities(P) / T)`` reproduces ``P``."""
    probabilities = [
        [0.5, 0.25, 0.125] + [0.125 / 17] * 17,
        [1.0 / 20] * 20,
    ]
    logits = ProteinSequenceLogits.from_probabilities(probabilities, temperature=temperature)

    for row, expected in zip(logits.root, probabilities, strict=True):
        scaled = [value / temperature for value in row]
        peak = max(scaled)
        exponentiated = [math.exp(value - peak) for value in scaled]
        total = sum(exponentiated)
        for actual, target in zip((value / total for value in exponentiated), expected, strict=True):
            assert actual == pytest.approx(target, abs=1e-9)


def test_from_probabilities_rejects_exact_zeros_with_guidance() -> None:
    """Exact zeros map to infinite logits, so they are refused with the two intents named."""
    probabilities = [[0.5, 0.5] + [0.0] * 18]
    with pytest.raises(ValueError, match="temperature=None"):
        ProteinSequenceLogits.from_probabilities(probabilities)


def test_from_probabilities_rejects_invalid_temperature_and_rows() -> None:
    """Non-finite temperatures and non-distribution rows are refused."""
    uniform = [[1.0 / 20] * 20]
    for temperature in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="positive and finite"):
            ProteinSequenceLogits.from_probabilities(uniform, temperature=temperature)
    with pytest.raises(ValueError, match="probability distribution"):
        ProteinSequenceLogits.from_probabilities([[0.9] * 20])


def test_sequence_logits_enforces_vocabulary_width_and_finiteness() -> None:
    """Rows must match the vocabulary width and contain only finite numbers."""
    with pytest.raises(ValidationError, match="20 columns"):
        ProteinSequenceLogits([[0.0] * 19])
    with pytest.raises(ValidationError, match="finite"):
        ProteinSequenceLogits([[float("nan")] * 20])
    with pytest.raises(ValidationError, match="at least one position"):
        ProteinSequenceLogits([])


def test_sequence_logits_base_class_requires_a_vocabulary() -> None:
    """The base class is abstract: it has no vocabulary to validate against."""
    with pytest.raises(ValidationError, match="no VOCAB"):
        SequenceLogits([[0.0]])


def test_validate_probabilities_honors_an_allowed_mask() -> None:
    """Mass outside the permitted columns is refused; fixed-context rows are skipped."""
    permitted = [[True] * 10 + [False] * 10, [False] * 20]
    on_domain = ProteinSequenceLogits([[0.1] * 10 + [0.0] * 10, [0.0] * 20])
    on_domain.validate_probabilities(allowed=permitted)

    off_domain = ProteinSequenceLogits([[0.05] * 20, [0.0] * 20])
    with pytest.raises(ValueError, match="non-permitted columns"):
        off_domain.validate_probabilities(allowed=permitted)


def test_validate_fixed_rows_requires_zeros_in_both_modes() -> None:
    """A row with no permitted column must be exactly zero."""
    permitted = [[False] * 20]
    ProteinSequenceLogits([[0.0] * 20]).validate_fixed_rows(allowed=permitted)
    with pytest.raises(ValueError, match="Fixed context row 1"):
        ProteinSequenceLogits([[0.0] * 19 + [0.5]]).validate_fixed_rows(allowed=permitted)
