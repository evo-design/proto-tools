# Fix GitHub Issue $ARGUMENTS

## Step 0: Set Up Worktree

Create an isolated worktree so this fix doesn't block or conflict with other in-progress work:

```bash
git fetch origin main
USER=$(git config user.name | tr ' ' '-' | tr '[:upper:]' '[:lower:]')
git worktree add .claude/worktrees/issue-$ARGUMENTS -B "$USER/fix-issue-$ARGUMENTS" origin/main
cd .claude/worktrees/issue-$ARGUMENTS
git submodule update --init --recursive
```

`-B` (not `-b`) ensures this works even if the branch exists from a previous attempt — it resets it to `origin/main`. The submodule init is required because worktrees don't auto-initialize submodules, and most tests import from `proto_tools`.

Work inside this worktree for all subsequent steps. If the branch name doesn't capture the intent (e.g., it's a feature, not a fix), rename it after reading the issue with `git branch -m $USER/better-name`.

If a worktree already exists for this issue, `cd` into it and `git pull origin main && git submodule update --init --recursive` to stay current.

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
| Tool implementation | `proto_tools/tools/{category}/{tool}/{tool}.py` | `tests/` (see below) |
| Tool data models | `proto_tools/tools/{category}/{tool}/__init__.py` | `tests/tool_infra_tests/` |
| Standalone scripts | `proto_tools/tools/{category}/{tool}/standalone/` | — |
| Shared data models | `proto_tools/tools/{category}/shared_data_models.py` | — |
| Tool citations | `proto_tools/tools/{category}/{tool}/cite.bib` | `tests/tool_infra_tests/test_citations.py` |
| Tool registry | `proto_tools/tools/tool_registry.py` | `tests/tool_infra_tests/` |
| Caching | `proto_tools/utils/tool_cache.py` | `tests/tool_infra_tests/` |
| ToolInstance | `proto_tools/utils/tool_instance.py` | `tests/tool_infra_tests/` |
| Entities (Structure, Ligands) | `proto_tools/entities/` | `tests/structure_tests/`, `tests/ligand_tests/` |

### Test File Locations

Tests are organized by tool category:
```
tests/
├── conftest.py
├── causal_models_tests/        # evo1, evo2, progen2
├── masked_models_tests/        # esm2, esm3
├── database_retrieval_tests/
├── gene_annotation_tests/
├── inverse_folding_tests/
├── ligand_tests/
├── orf_prediction_tests/
├── rna_splicing_tests/
├── sequence_alignment_tests/
├── sequence_scoring_tests/
├── structure_alignment_tests/
├── structure_design_tests/
├── structure_dynamics_tests/
├── structure_prediction_tests/
├── structure_tests/
├── style_consistency_tests/
└── tool_infra_tests/
```

## Step 3: Present Findings — STOP and wait for user

**Do not write code yet.** Present your interpretation of the problem and proposed approach:

- **My read on the issue**: one paragraph — what's actually broken/needed and why
- **Root cause hypothesis**: what you think is wrong, with evidence from the code you read
- **Proposed approach**: which files you'll touch and what changes you'll make
- **Scope check**: is this a clean fix, or does it touch the public API / export chain / multiple tools?
- **Branch name**: confirm or suggest renaming the branch to something more descriptive

Wait for the user to confirm, redirect, or add context before proceeding.

## Step 4: Write a Failing Test

**Always write a test that reproduces the bug before attempting a fix.**

Follow the existing test patterns for the affected tool category. Use the same markers and fixtures as neighboring test files.

```bash
# Verify the test fails as expected
pytest -xvs -k "test_name" tests/path/to/test_file.py
```

For feature requests (not bugs), skip the failing-test step — but still plan the tests you'll write alongside the implementation.

## Step 5: Implement the Fix

Follow the coding conventions:
- `from __future__ import annotations` at top of every file
- `logging.getLogger(__name__)` — never `print()`
- Ruff (line length 88, import sorting)
- Pydantic v2: `ConfigField()` for Config, `Field()` for Input/Output
- Config: `extra="ignore"` | Input: `extra="forbid"` | Output: `extra="forbid"`
- Never catch exceptions inside tool functions — `@tool` decorator handles error wrapping
- All biological coordinates are 1-indexed, inclusive

Keep the fix minimal and focused. Don't refactor surrounding code unless the issue specifically asks for it.

If the fix touches the tool's public API (Input/Config/Output classes), update the `__init__.py` export chain:
1. Tool `__init__.py` -> exports Input, Config, Output, run_*
2. Category `__init__.py` -> re-exports from all tools
3. `tools/__init__.py` -> master re-export
4. `proto_tools/__init__.py` -> package-level re-export

## Step 6: Verify

Run these checks in parallel using sub-agents where possible:

```bash
# 1. Verify the new test passes
pytest -xvs -k "test_name" tests/path/to/test_file.py

# 2. Run the broader test suite for the affected area
pytest tests/{category}_tests/ -m "not slow"

# 3. Run the full fast test suite to check for regressions
pytest -m "not slow"

# 4. Lint
ruff check proto_tools tests
```

If any test fails, fix it before proceeding. Don't ask — just fix regressions.

## Step 7: Push & PR

After all checks pass, push the branch and create a PR:

```bash
git push -u origin HEAD
gh pr create --title "Fix #$ARGUMENTS: <concise description>" --body "Closes #$ARGUMENTS

## Summary
<1-3 bullets>

## Test plan
- [ ] New test reproduces the bug and passes with fix
- [ ] Area tests pass
- [ ] Full fast suite passes
- [ ] Lint clean"
```

Then offer to clean up the worktree:
```bash
REPO_ROOT=$(git worktree list --porcelain | head -1 | sed 's/worktree //')
cd "$REPO_ROOT"
git worktree remove --force .claude/worktrees/issue-$ARGUMENTS
```

## Step 8: Summary

Provide a concise summary:
- **Issue**: one-line restatement of the problem
- **Root cause**: what was wrong
- **Fix**: what changed (files + brief description)
- **Tests**: what tests were added/modified
- **PR**: link to the created PR

## Tips

- For issues involving standalone scripts, test the standalone `run.py` separately: `python proto_tools/tools/{category}/{tool}/standalone/run.py`
- If the issue involves GPU tools, mark new tests with `@pytest.mark.uses_gpu`
- When fixing tool data model issues, always verify the JSON schema output: `ToolRegistry.get_schemas("tool-key")`
- Missing exports are a common source of "tool not found" issues — check the 4-level `__init__.py` export chain
- If the issue references a tool README being wrong, update the README and run `pre-commit run --all-files` to regenerate docs

## Validation Checklist

- [ ] Issue read and requirements understood (`gh issue view`)
- [ ] Affected source files identified and read
- [ ] Findings presented and user confirmed approach
- [ ] Failing test reproduces the bug (or test plan for features)
- [ ] Fix is minimal and focused
- [ ] `__init__.py` export chain updated (if public API changed)
- [ ] New test passes: `pytest -xvs -k "test_name"`
- [ ] Area tests pass: `pytest tests/{category}_tests/ -m "not slow"`
- [ ] Full fast suite passes: `pytest -m "not slow"`
- [ ] Lint passes: `ruff check proto_tools tests`
