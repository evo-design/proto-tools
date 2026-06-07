"""Tests that ``stochastic=True`` iterable tools diversify duplicate inputs.

The framework passes duplicate iterable items through to the
tool (skipping dedup). The contract is that the tool's per-item sampling
primitives — ``torch.multinomial``, JAX ``random.split``, autoregressive
decode loops, etc. — consume and advance RNG between batch elements so
identical inputs produce distinct outputs from one seed. This file enforces
that contract.

Three tests per stochastic iterable tool:

- ``test_duplicate_inputs_produce_diverse_outputs``: ``[i1] * 3 + seed=42``
  → 3 pairwise-distinct outputs.
- ``test_duplicate_inputs_reproducible_across_calls``: same call twice →
  identical triples.
- ``test_unseeded_calls_produce_different_outputs``: ``seed=None`` repeat
  → cache-skip yields different results.

Marked ``@pytest.mark.extensive`` — opt-in via ``pytest --ext``.
"""

import json
from typing import Any

import pytest

from proto_tools.tools.tool_registry import ToolRegistry, ToolSpec
from tests.seed_tests.test_seed_reproducibility import (
    _SEED_EXCLUDED_KEYS,
    _SEED_NON_PERSISTENT_EXCLUDED_KEYS,
)
from tests.tool_infra_tests.pytest_helpers import (
    EXCLUDED_CATEGORIES,
    SKIP_CI_TOOLKITS,
    build_inputs_and_config,
    parse_min_gpu_count,
)

# Tool keys whose outputs are non-deterministic enough across runs that the
# cross-call reproducibility check (test 2) cannot pass reliably. Reuses the
# exclusion sets from ``test_seed_reproducibility`` since the failure modes
# are the same (GPU kernel autotune drift, bfloat16 ULP boundaries, etc.).
_REPRODUCIBILITY_EXCLUDED_KEYS: frozenset[str] = _SEED_EXCLUDED_KEYS | _SEED_NON_PERSISTENT_EXCLUDED_KEYS


def _build_diversification_test_params() -> list:
    """Build pytest parametrize params for diversification tests.

    Filters to ``stochastic=True`` tools with an ``iterable_input_field`` —
    the diversification contract is iterable-specific. Skips categories
    excluded from broad tool tests (``database_retrieval``) and tools
    without a valid ``example_input()`` factory.
    """
    params = []

    for spec in sorted(ToolRegistry.list_all(), key=lambda s: s.key):
        if not spec.stochastic:
            continue
        if spec.iterable_input_field is None:
            continue
        if spec.category in EXCLUDED_CATEGORIES:
            continue
        if spec.example_input is None:
            continue

        marks: list = []

        if spec.uses_gpu:
            gpu_count = parse_min_gpu_count(spec.device_count)
            marks.append(pytest.mark.uses_gpu(gpu_count))

        if spec.source_file.parent.name in SKIP_CI_TOOLKITS:
            marks.append(pytest.mark.skip_ci)

        params.append(pytest.param(spec, id=spec.key, marks=marks))

    return params


def _replicate_first_item(spec: ToolSpec, inputs: Any, n: int = 3) -> Any:
    """Return a copy of ``inputs`` with ``iterable_input_field`` replaced by N copies of its first item.

    Sibling list fields that are parallel-aligned to the iterable field (same
    original length) are replicated too, so per-item alignment contracts hold for
    the fabricated batch — e.g. a structure-prediction input's pre-supplied
    ``msas`` must stay parallel to ``complexes``.

    Raises ``pytest.skip`` if the example input has an empty iterable.
    """
    iterable_field = spec.iterable_input_field
    items = list(getattr(inputs, iterable_field))
    if not items:
        pytest.skip(f"{spec.key}: example_input() has no items in {iterable_field!r}")
    update: dict[str, Any] = {iterable_field: [items[0]] * n}
    for field_name in type(inputs).model_fields:
        if field_name == iterable_field:
            continue
        value = getattr(inputs, field_name)
        if isinstance(value, list) and len(value) == len(items):
            update[field_name] = [value[0]] * n
    return inputs.model_copy(update=update)


def _item_fingerprint(item: Any) -> str:
    """Canonical string fingerprint of an output iterable item, for set-based distinctness checks.

    Recurses into list/tuple/dict containers and dumps nested Pydantic models so the
    fingerprint reflects model *content*. Without this, an item that is a *list* of
    models (e.g. fampnn-pack's ``[Structure]``) would stringify via ``repr`` — and a
    content-free ``Structure(...)`` repr makes genuinely-distinct outputs collapse to
    one fingerprint.
    """

    def _canonical(obj: Any) -> Any:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if isinstance(obj, (list, tuple)):
            return [_canonical(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _canonical(v) for k, v in obj.items()}
        return obj

    return json.dumps(_canonical(item), sort_keys=True, default=str)


@pytest.mark.extensive
@pytest.mark.parametrize("spec", _build_diversification_test_params())
def test_duplicate_inputs_produce_diverse_outputs(spec: ToolSpec, tmp_path):
    """[i1] * 3 + seed=42 produces 3 pairwise-distinct outputs."""
    inputs, config = build_inputs_and_config(spec, tmp_path, {"seed": 42})
    duplicated = _replicate_first_item(spec, inputs, n=3)

    result = spec.function(duplicated, config)
    assert result.success, f"{spec.key} dispatch failed: {result.errors}"

    outputs = getattr(result, spec.iterable_output_field)
    assert len(outputs) == 3, f"{spec.key}: expected 3 outputs, got {len(outputs)}"

    fingerprints = [_item_fingerprint(o) for o in outputs]
    distinct = len(set(fingerprints))
    assert distinct == 3, (
        f"{spec.key}: expected 3 distinct outputs from duplicate inputs, got {distinct} ({fingerprints!r})"
    )


@pytest.mark.extensive
@pytest.mark.parametrize("spec", _build_diversification_test_params())
def test_duplicate_inputs_reproducible_across_calls(spec: ToolSpec, tmp_path):
    """Calling the tool twice with same duplicated inputs + same seed yields the same triple."""
    if spec.key in _REPRODUCIBILITY_EXCLUDED_KEYS:
        pytest.skip(
            f"{spec.key} excluded from reproducibility check: "
            f"GPU/kernel non-determinism unrelated to seed handling "
            f"(see _SEED_EXCLUDED_KEYS / _SEED_NON_PERSISTENT_EXCLUDED_KEYS in test_seed_reproducibility.py)."
        )

    inputs, config = build_inputs_and_config(spec, tmp_path, {"seed": 42})
    duplicated = _replicate_first_item(spec, inputs, n=3)

    r1 = spec.function(duplicated, config)
    assert r1.success, f"{spec.key} first call failed: {r1.errors}"
    r2 = spec.function(duplicated, config)
    assert r2.success, f"{spec.key} second call failed: {r2.errors}"

    r1.approx_equal(r2)


@pytest.mark.extensive
@pytest.mark.parametrize("spec", _build_diversification_test_params())
def test_unseeded_calls_produce_different_outputs(spec: ToolSpec, tmp_path):
    """seed=None repeat-call produces different results — cache must skip."""
    inputs, config = build_inputs_and_config(spec, tmp_path)  # seed defaults to None
    duplicated = _replicate_first_item(spec, inputs, n=3)

    r1 = spec.function(duplicated, config)
    assert r1.success, f"{spec.key} first call failed: {r1.errors}"
    r2 = spec.function(duplicated, config)
    assert r2.success, f"{spec.key} second call failed: {r2.errors}"

    out_field = spec.iterable_output_field
    fp1 = [_item_fingerprint(o) for o in getattr(r1, out_field)]
    fp2 = [_item_fingerprint(o) for o in getattr(r2, out_field)]
    assert fp1 != fp2, f"{spec.key}: unseeded repeat-call produced identical results (cache not skipped?)"
