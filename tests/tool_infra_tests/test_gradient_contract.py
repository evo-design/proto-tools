"""tests/tool_infra_tests/test_gradient_contract.py.

Cross-backend contract tests for every relaxed-sequence gradient tool.

These run against each registered ``*-gradient`` tool that inherits the shared
``GradientInput``, so a new gradient backend is covered the moment it is registered.
Coverage:

- Every gradient tool inherits the shared input.
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
from proto_tools.utils.gradient_models import (
    BatchedNucleotideGradientInput,
    GradientInput,
    GradientOutput,
    RelaxationConfig,
)
from proto_tools.utils.sequence_logits import (
    ProteinSequenceLogits,
    SequenceLogits,
)


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
            ``"antibody.heavy_chain"``, ``"logits[0]"``, or ``"inputs[2].logits"``.
        value (Any): Replacement value.
    """

    def resolve(cursor: Any, segment: str) -> Any:
        """Step into one path segment, which may carry a list index."""
        if segment.endswith("]"):
            name, _, index = segment[:-1].partition("[")
            return cursor[name][int(index)]
        return cursor[segment]

    *parents, leaf = path.split(".")
    cursor: Any = payload
    for part in parents:
        cursor = resolve(cursor, part)
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
    """Every registered ``*-gradient`` tool must inherit ``GradientInput``.

    A tool that declares its own input class keeps the field names while opting out of the
    shared temperature and probability-mode validation, so the contract tests below silently
    skip it.
    """
    gradient_keys = {spec.key for spec in ToolRegistry.list_all() if spec.key.endswith("-gradient")}
    sharing_the_input = {spec.key for spec in _GRADIENT_SPECS}

    assert gradient_keys == sharing_the_input, (
        f"these gradient tools do not inherit the shared GradientInput: {sorted(gradient_keys - sharing_the_input)}"
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
def test_relaxed_state_is_discoverable(spec: Any) -> None:
    """Each tool exposes at least one relaxed matrix through the shared discovery interface.

    Shape and rectangularity are guaranteed by ``SequenceLogits.validate_matrix`` at
    construction, so asserting them here would only restate the type.
    """
    matrices = _example(spec).sequence_logits()

    assert matrices, f"{spec.key!r} exposes no SequenceLogits via sequence_logits()."


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
        # Keep the row summing to 1.0, so the case tests nonnegativity rather than the simplex rule.
        other = next(index for index, value in enumerate(rows[target]) if value > 0.0 and index != column)
        rows[target][other] += 2 * rows[target][column]
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


# ── from_probabilities on the tool input ─────────────────────────────────────


def _positive_probabilities(
    matrix: SequenceLogits, allowed: list[list[bool]] | None, favor: int = 0
) -> list[list[float]]:
    """Build rows biased toward one permitted column, strictly positive where permitted.

    ``from_probabilities`` refuses exact zeros, so every permitted column gets some mass. A row
    with no permitted column is fixed context and carries no state, so it stays all-zero — the
    only valid value for it in either temperature mode.

    Args:
        matrix (SequenceLogits): Matrix whose shape to mirror.
        allowed (list[list[bool]] | None): Per-position permitted columns, or ``None``.
        favor (int): Which permitted column carries the extra mass, so callers can build several
            distributions that differ from each other.

    Returns:
        list[list[float]]: ``(L, V)`` rows, each a distribution over its permitted columns.
    """
    _, width = matrix.shape
    rows: list[list[float]] = []
    for index in range(len(matrix)):
        mask = allowed[index] if allowed is not None else [True] * width
        permitted = [column for column, ok in enumerate(mask) if ok]
        row = [0.0] * width
        for rank, column in enumerate(permitted):
            row[column] = 2.0 if rank == favor % len(permitted) else 1.0
        total = sum(row)
        rows.append(row if total == 0.0 else [value / total for value in row])
    return rows


def _build_from_probabilities(spec: Any, temperature: float) -> tuple[Any, dict[str, list[list[float]]]]:
    """Construct a tool input through its ``from_probabilities`` factory.

    Returns the input together with the distributions it was built from, keyed by the same
    paths ``sequence_logits()`` reports.
    """
    example = _example(spec)
    fields = example.model_dump(exclude_none=True)
    fields.pop("temperature", None)
    targets = {
        path: _positive_probabilities(matrix, example.allowed_tokens(path, matrix))
        for path, matrix in example.sequence_logits().items()
    }

    if issubclass(spec.input_model, BatchedNucleotideGradientInput):
        fields.pop("inputs")
        built = spec.input_model.from_probabilities(list(targets.values()), temperature=temperature, **fields)
    elif all(path.endswith(("heavy_chain", "light_chain")) for path in targets):
        # A chain-named bundle: the factory takes the chains by keyword, not one matrix.
        fields.pop(spec.input_model.relaxed_field())
        chains = {path.rsplit(".", 1)[1]: rows for path, rows in targets.items()}
        built = spec.input_model.from_probabilities(temperature=temperature, **chains, **fields)
    else:
        field = spec.input_model.relaxed_field()
        fields.pop(field)
        # One example item, whether the field holds a matrix or a one-item list of them.
        target = next(iter(targets.values()))
        # A bundled item carries its own context — a template, a backbone — which the factory
        # still takes by keyword, so pass everything on the item that is not a relaxed matrix.
        items = getattr(example, "inputs", None)
        context: dict[str, Any] = {}
        if items and callable(getattr(items[0], "iter_sequence_logits", None)):
            first = items[0]
            matrices = {name for name, _ in first.iter_sequence_logits()}
            context = {name: getattr(first, name) for name in type(first).model_fields if name not in matrices}
        built = spec.input_model.from_probabilities(target, temperature=temperature, **context, **fields)
    return built, targets


@pytest.mark.parametrize("spec", _GRADIENT_SPECS, ids=lambda spec: spec.key)
@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.5])
def test_from_probabilities_sets_a_consistent_temperature(spec: Any, temperature: float) -> None:
    """The factory stores the temperature it converted at, so softmax reproduces the input.

    Going through ``SequenceLogits.from_probabilities`` and then passing a different
    ``temperature`` to the constructor silently reshapes the distribution; this is the contract
    that makes that impossible when the factory is used.
    """
    built, targets = _build_from_probabilities(spec, temperature)

    assert built.temperature == temperature
    rebuilt = built.sequence_logits()
    assert set(rebuilt) == set(targets)
    for path, matrix in rebuilt.items():
        for row, expected in zip(matrix.root, targets[path], strict=True):
            if not any(expected):
                # Fixed context: no state to reproduce, and the logits stay zero.
                assert not any(row)
                continue
            scaled = [value / temperature for value in row]
            peak = max(scaled)
            exponentiated = [math.exp(value - peak) for value in scaled]
            total = sum(exponentiated)
            actual = [value / total for value in exponentiated]
            active = [column for column, value in enumerate(expected) if value > 0.0]
            renormalized = sum(actual[column] for column in active)
            assert [actual[column] / renormalized for column in active] == pytest.approx(
                [expected[column] for column in active], abs=1e-9
            ), f"{spec.key} {path} did not round-trip"


@pytest.mark.parametrize("spec", _GRADIENT_SPECS, ids=lambda spec: spec.key)
def test_config_preprocess_understands_the_input_shape(spec: Any) -> None:
    """Every gradient tool's config hook must accept its own input.

    ``BaseConfig.preprocess`` runs on each dispatch before any model loads, and the hooks reach
    into the input to validate it — context limits, strand markers, ignored MSAs. A hook that
    reads a field the input no longer carries raises there, which no test driving the worker
    directly can see, because none of those paths cross the hook.
    """
    example = _example(spec)

    result = spec.config_model().preprocess(example)

    returned = result[0] if isinstance(result, tuple) else result
    assert isinstance(returned, type(example)), f"{spec.key}: preprocess returned {type(returned).__name__}"


# ── Iterable fanout ──────────────────────────────────────────────────────────

_ITERABLE_GRADIENT_SPECS = [spec for spec in _GRADIENT_SPECS if spec.iterable_input_fields]


@pytest.mark.parametrize("spec", _ITERABLE_GRADIENT_SPECS, ids=lambda spec: spec.key)
def test_iterable_gradient_tool_declares_the_whole_trio(spec: Any) -> None:
    """Fanout needs all three declarations, and ToolPool asserts the output field at run time."""
    assert spec.iterable_input_fields == ["inputs"], f"{spec.key}: {spec.iterable_input_fields}"
    assert spec.iterable_output_field == "results", f"{spec.key}: {spec.iterable_output_field}"
    assert isinstance(spec.max_chunk_size, int) and spec.max_chunk_size >= 1, (
        f"{spec.key}: max_chunk_size must be an explicit positive cap, got {spec.max_chunk_size!r}"
    )


@pytest.mark.parametrize("spec", _ITERABLE_GRADIENT_SPECS, ids=lambda spec: spec.key)
def test_single_item_normalizes_and_empty_is_rejected(spec: Any) -> None:
    """One item may be passed bare; a list keeps its length; an empty list is refused.

    Built from the tool's own example item, so it covers a bare matrix and a bundle carrying its
    own context alike.
    """
    example = _example(spec)
    item = example.inputs[0]
    other = example.model_dump()["inputs"][0]

    assert len(spec.input_model(inputs=item).inputs) == 1, "a single item must normalize to a one-item list"
    assert len(spec.input_model(inputs=[item, other]).inputs) == 2
    with pytest.raises(ValidationError, match="at least one item"):
        spec.input_model(inputs=[])


def test_gradient_batch_fans_out_across_pool_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    """A batch split across two pool workers returns one result per item, in input order.

    Goes through the ``@tool`` wrapper so the real partitioning, dispatch and stitching run; only
    the worker call is faked. Each item carries a distinguishing value that comes back as its
    loss, so a reordered or duplicated reassembly fails rather than merely looking plausible.
    """
    from proto_tools.tools.masked_models.esm2 import ESM2GradientConfig, ESM2GradientInput, run_esm2_gradient
    from proto_tools.utils.device_manager import DeviceManager
    from proto_tools.utils.tool_instance import ToolInstance
    from proto_tools.utils.tool_pool import ToolPool

    # esm2-gradient is a GPU tool, so the pool needs GPU slots to have anything to fan out over.
    # The worker call is faked, so the slots only have to exist on paper and this runs anywhere.
    monkeypatch.setattr("proto_tools.utils.tool_pool.number_of_visible_gpus", lambda: 2)
    DeviceManager.reset_instance()
    ToolInstance.clear_all()

    item_count = 6
    items = [[[float(index)] * 20] * 4 for index in range(item_count)]
    partitions: list[int] = []

    def fake_dispatch(toolkit: str, payload: dict[str, Any], *, instance: Any = None, config: Any = None) -> Any:
        chunk = payload["items"]
        partitions.append(len(chunk))
        return {
            "results": [
                {"gradient": item, "loss": item[0][0], "metrics": {"sequence_length": len(item)}} for item in chunk
            ],
            "vocab": list(ProteinSequenceLogits.VOCAB),
        }

    monkeypatch.setattr(
        "proto_tools.tools.masked_models.esm2.esm2_gradient.ToolInstance.dispatch",
        fake_dispatch,
    )

    try:
        with ToolPool(gpus=2):
            output = run_esm2_gradient(ESM2GradientInput(inputs=items), ESM2GradientConfig(device="cuda"))
    finally:
        DeviceManager.reset_instance()
        ToolInstance.clear_all()

    assert len(output.results) == item_count, "one result per item"
    assert [result.loss for result in output.results] == [float(index) for index in range(item_count)], (
        "results must come back in input order"
    )
    assert len(partitions) > 1, f"the batch was not split across workers: {partitions}"
    assert sum(partitions) == item_count, f"items were duplicated or dropped: {partitions}"
    assert output.vocab == list(ProteinSequenceLogits.VOCAB)


# ── Relaxation controls ──────────────────────────────────────────────────────


@pytest.mark.integration
def test_relaxation_formula_runs_in_env() -> None:
    """The relaxed_sequence blend and its gradients are checked inside a built tool env.

    ``standalone_helpers.relaxation`` imports torch, which the base process does not have, so
    ``_relaxation_env_checks.py`` runs the numerical suite in esm2's environment. It needs no
    model weights and no GPU — the formula is pure arithmetic. A non-zero exit surfaces here as
    ``RuntimeError``.
    """
    from pathlib import Path as _Path

    from proto_tools.utils import run_in_env

    checks = _Path(__file__).resolve().parent / "_relaxation_env_checks.py"
    output = run_in_env("esm2", script=str(checks), timeout=900)

    assert "ALL CHECKS PASSED" in output, output


@pytest.mark.integration
def test_relaxation_formula_runs_in_jax_env() -> None:
    """The JAX twin of the blend is checked inside proteinmpnn's environment, which has no torch.

    ``_relaxation_jax_env_checks.py`` asserts the same formula and the same ``REFERENCE`` table
    as the torch suite above, so the two definitions cannot drift apart. It needs no model
    weights and no GPU. A non-zero exit surfaces here as ``RuntimeError``.
    """
    from pathlib import Path as _Path

    from proto_tools.utils import run_in_env

    checks = _Path(__file__).resolve().parent / "_relaxation_jax_env_checks.py"
    output = run_in_env("proteinmpnn", script=str(checks), timeout=900)

    assert "ALL CHECKS PASSED" in output, output


@pytest.mark.integration
def test_alphafold2_relaxation_matches_the_contract_in_env() -> None:
    """AlphaFold2 relaxes inside ColabDesign, so its own blend is checked against the contract.

    ``_alphafold2_relaxation_env_checks.py`` compares ``soft_seq`` on both backends — upstream
    ColabDesign and the Germinal fork, whose alpha scaling and bias feed the softmax branch —
    against the shared formula, and ties the no-preprocessing case to ``relaxed_sequence_jax``.
    It needs no model weights and no GPU. A non-zero exit surfaces here as ``RuntimeError``.
    """
    from pathlib import Path as _Path

    from proto_tools.utils import run_in_env

    checks = _Path(__file__).resolve().parent / "_alphafold2_relaxation_env_checks.py"
    output = run_in_env("alphafold2", script=str(checks), timeout=900)

    assert "ALL CHECKS PASSED" in output, output


@pytest.mark.parametrize("spec", _GRADIENT_SPECS, ids=lambda spec: spec.key)
def test_gradient_config_inherits_the_shared_relaxation_controls(spec: Any) -> None:
    """Every gradient tool must take ``soft`` and ``hard`` from one place.

    Re-declaring the fields keeps the names while letting the bounds, defaults and documented
    meaning drift per tool, which is the failure this whole contract exists to prevent.
    """
    assert issubclass(spec.config_model, RelaxationConfig), (
        f"{spec.key}: {spec.config_model.__name__} must inherit RelaxationConfig"
    )

    for field in ("soft", "hard"):
        own = spec.config_model.model_fields[field]
        shared = RelaxationConfig.model_fields[field]
        assert own.default == shared.default, f"{spec.key}: {field} default differs from the shared config"
        assert own.description == shared.description, f"{spec.key}: {field} re-declares its own description"


@pytest.mark.parametrize(
    "overrides",
    [
        {"soft": 1.5},
        {"soft": -0.1},
        {"hard": 1.5},
        {"hard": -0.1},
        {"soft": float("nan")},
        {"hard": float("inf")},
    ],
)
def test_relaxation_config_rejects_out_of_range_blends(overrides: dict[str, float]) -> None:
    """``soft`` and ``hard`` are blend weights, so only finite values in [0, 1] are meaningful."""
    with pytest.raises(ValidationError):
        RelaxationConfig(**overrides)


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


@pytest.mark.parametrize(
    "allowed, expected",
    [
        ([[False] * 3, [True] * 20], r"mask row 1 must have 20 columns, got 3"),
        ([[True] * 3, [True] * 20], r"mask row 1 must have 20 columns, got 3"),
        ([[], [True] * 20], r"mask row 1 must have 20 columns, got 0"),
        ([[False] * 99, [True] * 20], r"mask row 1 must have 20 columns, got 99"),
        ([[True] * 20], r"mask must have 2 rows, got 1"),
    ],
)
@pytest.mark.parametrize("method", ["validate_fixed_rows", "validate_probabilities"])
def test_malformed_allowed_mask_is_rejected_by_both_validators(
    allowed: list[list[bool]], expected: str, method: str
) -> None:
    """A mask that is not ``(L, V)`` is refused, naming the mask rather than blaming the data.

    Without the width check, a short or empty mask row reads as all-``False`` and both
    validators would take it for a deliberate fixed-context marker: the probability checks get
    skipped and the position goes unvalidated.
    """
    matrix = ProteinSequenceLogits([[1.0 / 20] * 20, [1.0 / 20] * 20])

    with pytest.raises(ValueError, match=expected):
        getattr(matrix, method)(allowed=allowed)


def test_validate_fixed_rows_requires_zeros_in_both_modes() -> None:
    """A row with no permitted column must be exactly zero."""
    permitted = [[False] * 20]
    ProteinSequenceLogits([[0.0] * 20]).validate_fixed_rows(allowed=permitted)
    with pytest.raises(ValueError, match="Fixed context row 1"):
        ProteinSequenceLogits([[0.0] * 19 + [0.5]]).validate_fixed_rows(allowed=permitted)


# ── from_probabilities with several candidates ───────────────────────────────


def _candidate_matrices(candidate: Any, names: set[str]) -> dict[str, SequenceLogits]:
    """Return a built candidate's relaxed matrices, keyed the way its targets are.

    Args:
        candidate (Any): One entry of a built input's ``inputs``.
        names (set[str]): Target keys; ``{""}`` for a candidate that is itself a bare matrix.

    Returns:
        dict[str, SequenceLogits]: The candidate's matrices under those keys.
    """
    if isinstance(candidate, SequenceLogits):
        return dict.fromkeys(names, candidate)
    return {name: getattr(candidate, name) for name in names}


def _build_many_from_probabilities(
    spec: Any, temperature: float, count: int
) -> tuple[Any, list[dict[str, list[list[float]]]]]:
    """Build one input holding ``count`` distinct candidates through ``from_probabilities``.

    Each candidate is described by one object carrying its own matrix together with its own
    context, never by parallel arrays, and every matrix is converted at the one shared
    temperature.

    Args:
        spec (Any): The registered tool spec to build for.
        temperature (float): The single temperature every candidate converts at.
        count (int): How many candidates to build.

    Returns:
        tuple[Any, list[dict[str, list[list[float]]]]]: The built input, and the distribution each
            candidate was built from, in input order.
    """
    example = _example(spec)
    fields = example.model_dump(exclude_none=True)
    fields.pop("temperature", None)
    fields.pop("inputs")
    first = example.inputs[0]

    specs: list[Any] = []
    targets: list[dict[str, list[list[float]]]] = []
    for index in range(count):
        if isinstance(first, SequenceLogits):
            # The candidate is the matrix itself, so the list holds bare matrices.
            rows = _positive_probabilities(first, example.allowed_tokens("inputs[0]", first), favor=index)
            specs.append(rows)
            targets.append({"": rows})
            continue
        matrices = dict(first.iter_sequence_logits())
        chains = {
            name: _positive_probabilities(matrix, example.allowed_tokens(f"inputs[0].{name}", matrix), favor=index)
            for name, matrix in matrices.items()
        }
        context = {name: getattr(first, name) for name in type(first).model_fields if name not in matrices}
        if len(chains) > 1 or "logits" not in chains:
            # Chains are named rather than stacked, so the candidate names each one.
            specs.append({**chains, **context})
        else:
            specs.append({"probabilities": chains["logits"], **context})
        targets.append(chains)

    return spec.input_model.from_probabilities(specs, temperature=temperature, **fields), targets


@pytest.mark.parametrize("spec", _ITERABLE_GRADIENT_SPECS, ids=lambda spec: spec.key)
@pytest.mark.parametrize("temperature", [0.5, 2.5])
def test_from_probabilities_converts_every_candidate_at_the_shared_temperature(spec: Any, temperature: float) -> None:
    """Every candidate reconstructs its own distribution under the one temperature given.

    The candidates are deliberately different from one another, so a factory that converted only
    the first and reused it, or that reassembled them out of order, fails here rather than
    looking plausible.
    """
    built, targets = _build_many_from_probabilities(spec, temperature, count=3)

    assert len(built.inputs) == len(targets) == 3, f"{spec.key}: {len(built.inputs)} candidates"
    assert built.temperature == temperature
    for index, (candidate, expected) in enumerate(zip(built.inputs, targets, strict=True)):
        for name, matrix in _candidate_matrices(candidate, set(expected)).items():
            rows = expected[name]
            for position, (logits, target) in enumerate(zip(matrix.root, rows, strict=True)):
                permitted = [column for column, value in enumerate(target) if value > 0.0]
                if not permitted:
                    # Fixed context, and outside every modality block: no state to reproduce.
                    assert not any(logits), f"{spec.key} candidate {index} row {position} must stay zero"
                    continue
                scaled = [logits[column] / temperature for column in permitted]
                peak = max(scaled)
                weights = [math.exp(value - peak) for value in scaled]
                total = sum(weights)
                assert [weight / total for weight in weights] == pytest.approx(
                    [target[column] for column in permitted], abs=1e-9
                ), f"{spec.key} candidate {index} {name} row {position} did not round-trip"
                assert all(logits[column] == 0.0 for column in range(len(target)) if column not in permitted), (
                    f"{spec.key} candidate {index} {name} row {position} put mass outside its permitted columns"
                )


@pytest.mark.parametrize("spec", _ITERABLE_GRADIENT_SPECS, ids=lambda spec: spec.key)
def test_from_probabilities_still_builds_one_candidate_from_one_distribution(spec: Any) -> None:
    """The single-candidate call is unchanged: one distribution still yields exactly one item."""
    built, targets = _build_from_probabilities(spec, 0.75)

    assert len(built.inputs) == 1, f"{spec.key}: one distribution built {len(built.inputs)} candidates"
    assert built.temperature == 0.75
    assert set(built.sequence_logits()) == set(targets)


def test_from_probabilities_rejects_malformed_candidate_lists() -> None:
    """A list of candidates must be non-empty, and each entry must carry its own distribution."""
    esm2 = ToolRegistry.get("esm2-gradient").input_model
    rows = [[1.0 / 20] * 20]

    with pytest.raises(ValueError, match="at least one candidate"):
        esm2.from_probabilities([])
    with pytest.raises(ValueError, match="must supply its own 'probabilities'"):
        esm2.from_probabilities([{"logits": rows}])
    with pytest.raises(TypeError, match="take no context fields"):
        esm2.from_probabilities([{"probabilities": rows, "structure": "irrelevant"}])


def test_from_probabilities_requires_each_candidate_to_bring_its_context() -> None:
    """A bundled candidate without its context is refused, however many candidates there are."""
    proteinmpnn = ToolRegistry.get("proteinmpnn-gradient").input_model
    rows = [[1.0 / 20] * 20]

    with pytest.raises(ValueError, match="requires the backbone structure"):
        proteinmpnn.from_probabilities([{"probabilities": rows}, {"probabilities": rows}])


def test_ablang_from_probabilities_refuses_mixing_the_two_candidate_forms() -> None:
    """Chains come either as candidate mappings or as keywords, never as both at once."""
    ablang = ToolRegistry.get("ablang-gradient").input_model
    rows = [[1.0 / 20] * 20]

    with pytest.raises(ValueError, match="either as candidates or by keyword"):
        ablang.from_probabilities([{"heavy_chain": rows}], heavy_chain=rows)
    with pytest.raises(ValueError, match="at least one antibody"):
        ablang.from_probabilities([])
