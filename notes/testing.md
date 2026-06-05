# Test Conventions

This note covers the conventions for writing tests in proto-tools, including structure, assertions, markers, and naming.

All tests use **flat functions** (no test classes). Follow these patterns when writing new tests.

## Structure

- **One-liner module docstring**: `"""Tests for {tool/entity name}."""`
- **No `from __future__ import annotations`** anywhere in the codebase
- **Flat functions only**: No `class Test*`. Use descriptive function names (e.g., `test_blast_search_exact_match`)
- **Section separators**: Use `# ── Section name ──...` for groups. Use `# ---------------------------------------------------------------------------` + `# Integration tests` for the integration boundary
- **File ordering**: Unit tests first, then the integration boundary separator, then integration/GPU tests
- **Module-level fixtures**: Use `@pytest.fixture` at module level, not inside classes. Use `scope="module"` for expensive setup
- **Module-level constants**: Deduplicate repeated values as `_PREFIXED` constants (e.g., `_SETUP_SH`, `_CRISPR_SEQUENCE`)
- **Test directory naming**: `tests/{category}_tests/` matching `tools/{category}/`

## Assertions

- **Specific exception matching**: Always use `pytest.raises(ExceptionType, match="...")`. Never use bare `pytest.raises(Exception)`. For Pydantic `ge=N` constraints, match `"greater than or equal to N"`
- **No trivial tests**: Don't test that Pydantic stores default values. Test computed properties, validators, normalization, and error cases
- **`tmp_path` over `tempfile`**: Use pytest's built-in `tmp_path` fixture

## Markers

Canonical reference is `pyproject.toml [tool.pytest.ini_options].markers`. Tool-author cheat sheet:

- **`@pytest.mark.integration`**: Tests calling `ToolInstance.dispatch()` for CPU tools. Skipped by default. Run with `--integration`
- **`@pytest.mark.uses_gpu`**: Tests calling `ToolInstance.dispatch()` for GPU tools. Auto-skipped when no GPU. Implies environment requirement. Do **not** also add `@pytest.mark.integration`. Optional arg `@pytest.mark.uses_gpu(n)` requires `n` visible GPUs
- **`@pytest.mark.uses_cpu`**: CPU-only test. Optional arg `@pytest.mark.uses_cpu(n)` requires `n` CPUs (matches what `ToolPool._detect_cpus` would see)
- **`@pytest.mark.slow`**: Tests that may take several minutes. Skipped by default. Run with `--slow` or `--all`
- **`@pytest.mark.extensive`**: Combinatorial tests (e.g. every tool × device transition). Skipped unless `--ext` (or `--extensive`) is passed
- **`@pytest.mark.benchmark('<tool-key>')`**: E2E benchmark for one tool. Required arg. Skipped by default. Opt in via `--benchmark` (or `--benchmark-report` / `--benchmark-tool` / `--benchmark-toolkit`). Not enabled by `--all` or `--slow`
- **`@pytest.mark.skip_ci`**: Only for tests requiring optional/external dependencies not in `pyproject.toml`. Do not add for core deps
- **`@pytest.mark.include_in_env_report`**: Applied automatically by `test_env_report.py` parametrization. Do not add manually
- **`@pytest.mark.test_on_platforms('x86_64', ...)`**: Restrict to specific architectures

## Naming

- **Validation tests**: `test_{model}_rejects_{what}`
- **Property tests**: `test_{model}_{property}`
- **Export tests**: `test_export_{format}`
- **Integration tests**: `test_{tool}_{scenario}`

## Test order randomization

`pytest-randomly` shuffles test order each run to catch hidden inter-test dependencies. The seed is printed in the test header. Reproduce a specific order with `--randomly-seed=N`.
