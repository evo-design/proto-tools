# Test Conventions

All tests use **flat functions** (no test classes). Follow these patterns when writing new tests.

## Structure

- **One-liner module docstring**: `"""Tests for {tool/entity name}."""`
- **No `from __future__ import annotations`** in test files
- **Flat functions only**: No `class Test*` — use descriptive function names (e.g., `test_blast_search_exact_match`)
- **Section separators**: Use `# ── Section name ──...` for groups. Use `# ---------------------------------------------------------------------------` + `# Integration tests` for the integration boundary
- **File ordering**: Unit tests first, then the integration boundary separator, then integration/GPU tests
- **Module-level fixtures**: Use `@pytest.fixture` at module level, not inside classes. Use `scope="module"` for expensive setup
- **Module-level constants**: Deduplicate repeated values as `_PREFIXED` constants (e.g., `_SETUP_SH`, `_CRISPR_SEQUENCE`)
- **Test directory naming**: `tests/{category}_tests/` matching `tools/{category}/`

## Assertions

- **Specific exception matching**: Always use `pytest.raises(ExceptionType, match="...")` — never bare `pytest.raises(Exception)`. For Pydantic `ge=N` constraints, match `"greater than or equal to N"`
- **No trivial tests**: Don't test that Pydantic stores default values. Test computed properties, validators, normalization, and error cases
- **`tmp_path` over `tempfile`**: Use pytest's built-in `tmp_path` fixture

## Markers

- **`@pytest.mark.integration`**: Tests calling `ToolInstance.dispatch()` for CPU tools. Skipped by default; run with `--integration`
- **`@pytest.mark.uses_gpu`**: Tests calling `ToolInstance.dispatch()` for GPU tools. Auto-skipped when no GPU. Implies environment requirement — do **not** also add `@pytest.mark.integration`
- **`@pytest.mark.include_in_env_report(category="...")`**: Add to the primary integration/GPU test for each tool. Category must match the tool's category
- **No `@pytest.mark.skip_ci`** for core dependencies: If a package is in `pyproject.toml`, its tests should run without special markers
- **`@pytest.mark.skip_ci`**: Only for tests requiring optional/external dependencies not in `pyproject.toml`

## Naming

- **Validation tests**: `test_{model}_rejects_{what}`
- **Property tests**: `test_{model}_{property}`
- **Export tests**: `test_export_{format}`
- **Integration tests**: `test_{tool}_{scenario}`

## Parallel Execution

Integration tests can be run in parallel using pytest-xdist. This is useful on machines with 4+ cores where tool environment builds (the main bottleneck) can run concurrently.

```bash
# Run integration tests in parallel, grouped by file so each worker handles one tool
pytest --integration --cpu --dist loadfile -n auto

# Specify worker count explicitly
pytest --integration --cpu --dist loadfile -n 4
```

`--dist loadfile` ensures all tests from the same file (i.e. the same tool) go to the same worker, avoiding redundant env builds.
