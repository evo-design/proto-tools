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
- **Registry-wide contracts**: some tests run every registered tool of a kind, so a new tool is
  checked from its first commit. `tests/structure_prediction_tests/test_msa_contract.py` runs each
  tool whose input takes `msas` against a set of MSA bundle shapes with its worker stubbed; a new
  MSA-taking tool needs an entry in its `_STUBS`.

## Assertions

- **Specific exception matching**: Always use `pytest.raises(ExceptionType, match="...")`. Never use bare `pytest.raises(Exception)`. For Pydantic `ge=N` constraints, match `"greater than or equal to N"`
- **No trivial tests**: Don't test that Pydantic stores default values. Test computed properties, validators, normalization, and error cases
- **`tmp_path` over `tempfile`**: Use pytest's built-in `tmp_path` fixture

## Markers

Canonical reference is `pyproject.toml [tool.pytest.ini_options].markers`. Tool-author cheat sheet:

- **`@pytest.mark.integration`**: Tests calling `ToolInstance.dispatch()` for CPU tools. Skipped by default. Run with `--integration`
- **`@pytest.mark.uses_gpu`**: Tests calling `ToolInstance.dispatch()` for GPU tools. Auto-skipped when no GPU. Implies environment requirement. Do **not** also add `@pytest.mark.integration`. Optional arg `@pytest.mark.uses_gpu(n)` requires `n` visible GPUs. Mark every test that needs more than one GPU with its count: `--multi-gpu` runs exactly the tests with `n >= 2`, slow ones included, since `-m` cannot select on the count
- **`@pytest.mark.uses_cpu`**: CPU-only test. Optional arg `@pytest.mark.uses_cpu(n)` requires `n` CPUs (matches what `ToolPool._detect_cpus` would see)
- **`@pytest.mark.requires_credential("ENV_VAR", ...)`**: The test needs these credentials, by the environment variable a `ConfigField(credential=...)` falls back to (`GI_API_KEY`, `NCBI_API_KEY`, `SYNBIOHUB_PASSWORD`, ...). It is skipped at collection unless every one is set, and a blank value counts as unset. Each name must be a credential some tool config declares, so a typo fails the run instead of skipping the test forever. Use it rather than an in-body `os.environ` check. A credential being declared does not make a tool need it (NCBI's key only raises rate limits), so mark only the tests that cannot run without it.
- **`@pytest.mark.network`**: Tests that call a live external service (NCBI, UniProt, RCSB, EBI, Ensembl, remote Foldseek, ...) rather than a mocked transport. Skipped unless `--network` is passed; not enabled by `--all` or `--ext`, so an upstream queue or outage never decides how long a run takes. Mark the test (or the module with `pytestmark`) whenever it would fail with the network blocked
- **`@pytest.mark.slow`**: Tests that may take several minutes. Skipped by default. Run with `--slow` or `--all`
- **`@pytest.mark.extensive`**: Combinatorial tests (e.g. every tool × device transition). Skipped unless `--ext` (or `--extensive`) is passed
- **`@pytest.mark.benchmark('<tool-key>')`**: E2E benchmark for one tool. Required arg. Skipped by default. Opt in via `--benchmark` (or `--benchmark-report` / `--benchmark-tool` / `--benchmark-toolkit`). Not enabled by `--all` or `--slow`
- **`@pytest.mark.skip_ci`**: Only for tests requiring optional/external dependencies not in `pyproject.toml`. Do not add for core deps, or for tests that submit jobs to shared public compute services (e.g. EBI Job Dispatcher), whose fair-use policies block automated submitters
- **`@pytest.mark.include_in_env_report`**: Applied automatically by `test_env_report.py` parametrization. Do not add manually
- **`@pytest.mark.test_on_platforms('x86_64', ...)`**: Restrict to specific architectures

## Upstream outages

A declared `Outage` (see `notes/error-handling.md`) gets a recovery probe: the integration test for
the affected mode, run with `monkeypatch.setenv("PROTO_IGNORE_OUTAGES", "1")` under
`@pytest.mark.xfail(strict=True, raises=<the upstream's error>)`. While the outage lasts it XFAILs;
once the upstream recovers it passes, strict xfail fails the nightly integration run, and the fix is to
delete the `Outage` and the marker. Example: `test_alphafold_db_msa_outage_recovery_probe`.
`tests/style_consistency_tests/test_outage_declarations.py` evaluates every scoped outage's `when` on the
tool's example input and default config, and requires `False`: a scoped outage the defaults reach refuses
every default call, the example, and the hosted smoke run, so declare it with `affects=None` or change the
default or example instead.

## Naming

- **Validation tests**: `test_{model}_rejects_{what}`
- **Property tests**: `test_{model}_{property}`
- **Export tests**: `test_export_{format}`
- **Integration tests**: `test_{tool}_{scenario}`

## Test order randomization

`pytest-randomly` is installed, but this suite disables its reordering by default. Pass `--random-order` to shuffle test order and catch hidden inter-test dependencies. The seed is printed in the test header. Reproduce a specific order with `--randomly-seed=N`.
