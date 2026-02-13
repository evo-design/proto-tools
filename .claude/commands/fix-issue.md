# Fix GitHub Issue $ARGUMENTS

## Step 1: Read the Issue

```bash
gh issue view $ARGUMENTS
gh issue view $ARGUMENTS --comments
```

Extract from the issue:
- **What's broken or requested** — the core problem or feature
- **Reproduction steps** — if it's a bug
- **Affected tool(s)** — which tool category and tool name
- **Labels/assignees** — for priority and area context

## Step 2: Explore the Codebase

Use sub-agents in parallel to investigate all relevant areas simultaneously:

- **Search for keywords** from the issue (error messages, function names, tool keys) across the codebase
- **Read the affected tool's source** — implementation, standalone scripts, data models
- **Read existing tests** for the affected tool
- **Check recent commits** touching the affected files: `git log --oneline -20 -- <file>`

Parallelize exploration aggressively — launch multiple sub-agents to search different areas at once rather than searching sequentially.

### Where to Look by Area

| Area | Source | Tests |
|------|--------|-------|
| Tool implementation | `bio_programming_tools/tools/{category}/{tool}/{tool}.py` | `tests/` (see below) |
| Tool data models | `bio_programming_tools/tools/{category}/{tool}/__init__.py` | `tests/tool_infra_tests/` |
| Standalone scripts | `bio_programming_tools/tools/{category}/{tool}/standalone/` | — |
| Shared data models | `bio_programming_tools/tools/{category}/shared_data_models.py` | — |
| Tool citations | `bio_programming_tools/tools/{category}/{tool}/cite.bib` | `tests/tool_infra_tests/test_citations.py` |
| Tool registry | `bio_programming_tools/tools/tool_registry.py` | `tests/tool_infra_tests/` |
| Caching | `bio_programming_tools/utils/tool_cache.py` | `tests/tool_infra_tests/` |
| EnvManager | `bio_programming_tools/utils/env_manager.py` | `tests/tool_infra_tests/` |
| Entities (Structure, Ligands) | `bio_programming_tools/entities/` | `tests/structure_tests/`, `tests/ligand_tests/` |

### Test File Locations

Tests are organized by tool category:
```
tests/
├── test_blast.py, test_mmseqs.py, test_pyhmmer.py
├── inverse_folding_tests/
├── language_model_tests/
├── ligand_tests/
├── orf_prediction_tests/
├── sequence_alignment_tests/
├── sequence_scoring_tests/
├── structure_design_tests/
├── structure_dynamics_tests/
├── structure_prediction_tests/
├── structure_tests/
├── tool_infra_tests/
└── conftest.py
```

## Step 3: Write a Failing Test

**Always write a test that reproduces the bug before attempting a fix.**

Follow the existing test patterns for the affected tool category. Use the same markers and fixtures as neighboring test files.

```bash
# Verify the test fails as expected
pytest -xvs -k "test_name" tests/path/to/test_file.py
```

For feature requests (not bugs), skip the failing-test step — but still plan the tests you'll write alongside the implementation.

## Step 4: Implement the Fix

Follow the coding conventions:
- `from __future__ import annotations` at top of every file
- `logging.getLogger(__name__)` — never `print()`
- Black (line length 88), isort (black-compatible profile)
- Pydantic v2: `ConfigField()` for Config, `Field()` for Input/Output
- Config: `extra="ignore"` | Input: `extra="forbid"` | Output: `extra="forbid"`
- Never catch exceptions inside tool functions — `@tool` decorator handles error wrapping
- All biological coordinates are 1-indexed, inclusive

Keep the fix minimal and focused. Don't refactor surrounding code unless the issue specifically asks for it.

If the fix touches the tool's public API (Input/Config/Output classes), update the `__init__.py` export chain:
1. Tool `__init__.py` → exports Input, Config, Output, run_*
2. Category `__init__.py` → re-exports from all tools
3. `tools/__init__.py` → master re-export
4. `bio_programming_tools/__init__.py` → package-level re-export

## Step 5: Verify

Run these checks in parallel using sub-agents where possible:

```bash
# 1. Verify the new test passes
pytest -xvs -k "test_name" tests/path/to/test_file.py

# 2. Run the broader test suite for the affected area
pytest tests/{category}_tests/ -m "not slow"

# 3. Run the full fast test suite to check for regressions
pytest -m "not slow"

# 4. Lint
flake8 bio_programming_tools tests
```

If any test fails, fix it before proceeding. Don't ask — just fix regressions.

## Step 6: Summary

After all checks pass, provide a concise summary:
- **Issue**: one-line restatement of the problem
- **Root cause**: what was wrong
- **Fix**: what changed (files + brief description)
- **Tests**: what tests were added/modified
- **Verification**: confirmation that all tests and lint pass

## Tips

- For issues involving standalone scripts, test the standalone `run.py` separately: `python bio_programming_tools/tools/{category}/{tool}/standalone/run.py`
- If the issue involves GPU tools, mark new tests with `@pytest.mark.uses_gpu`
- When fixing tool data model issues, always verify the JSON schema output: `ToolRegistry.get_schemas("tool-key")`
- Missing exports are a common source of "tool not found" issues — check the 4-level `__init__.py` export chain
- If the issue references a tool README being wrong, update the README and run `pre-commit run --all-files` to regenerate docs
