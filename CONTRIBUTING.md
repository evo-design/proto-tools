# Contributing to bio-programming-tools

Thank you for your interest in contributing! The codebase is in a mature state with well-established patterns, but it is very much still in active development. Our goal is to make this a hub for the open source computational biology community to collaborate: whether that's wrapping new models as they come out, improving existing tools, or building infrastructure that makes everything easier to use. Contributions of all kinds are welcome! Please try to adhere to the existing patterns and conventions as much as possible. (Coding agents are very helpful for this!)

This guide covers the conventions and workflows used in this project.

## Development Setup

We recommend using the provided `environment.yml`, which sets up Python, compilers, build tools, and system libraries that some tool environments need when compiling from source:

```bash
conda env create -f environment.yml
conda activate bio-tools
pip install -e ".[dev]"
pre-commit install
```

Most newer systems will already have compilers and build tools available at a system level, which should allow you to just `pip install -e ".[dev]"` in any Python 3.10+ environment.

### Storage for developers

All persistent data (tool environments, model weights, micromamba) lives under `PROTO_HOME` (defaults to `~/.proto/`). See the [README](README.md) setup instructions.

```bash
# Add to ~/.bashrc
export PROTO_HOME=/path/to/your/proto_home
```

See the [README](README.md) for HuggingFace authentication setup.

## Branch Naming

Use descriptive branch names with a category prefix:

- `feat/description` — new features or tools
- `fix/description` — bug fixes
- `refactor/description` — code restructuring without behavior change
- `docs/description` — documentation-only changes
- `test/description` — test additions or fixes

## Pull Requests

### Title

Keep the title under 70 characters. Use imperative mood ("Add blast retry logic", not "Added" or "Adds").

### Body Format

Every PR must include **Summary** and **Test plan** sections. Include **Problem** for bug fixes.

```markdown
## Summary

- Bullet points describing what changed and why

## Problem

<!-- For bug fixes only. Delete this section for features. -->
What's broken and why.

## Test plan

- [x] Completed verification step
- [ ] Remaining verification step
```

### Linking Issues

Reference related issues in the PR body:

- `Closes #123` — automatically closes the issue when merged
- `Fixes #123` — same behavior, preferred for bug fixes
- `Related to #123` — links without auto-closing

## Issues

We use three issue templates:

- **Bug report** — something is broken (include reproduction steps, error output, environment)
- **Feature request** — propose new functionality (include use case, proposed solution)
- **Tool request** — request a new bioinformatics tool wrapper (include tool name, operations to wrap, motivation)

When filing a bug, include the full traceback and your environment details (OS, Python version, GPU if relevant).

## Code Style

### Formatting

- **No auto-formatting** — formatting is handled manually
- **ruff** — enforced. Checks F401 (unused imports), F841 (unused variables), and import sorting (line length 88). Run `ruff check bio_programming_tools` before committing

### Conventions

- `from __future__ import annotations` at the top of every file
- Use `logging.getLogger(__name__)` — never `print()`
- All biological coordinates are **1-indexed, inclusive**
- Pydantic v2 models: Config uses `extra="ignore"`, Input uses `extra="forbid"`, Output uses `extra="forbid"`

## Testing

Tests use flat functions only — no test classes.

### Running Tests

```bash
pytest                          # CPU unit tests (skips GPU, slow, integration)
pytest --gpu                    # GPU tests only
pytest --integration            # Integration tests (external APIs) + GPU if available
pytest --all                    # Everything: GPU + slow + integration
pytest --all --cpu              # Slow + integration, but skip GPU
```

### Markers

- `@pytest.mark.uses_gpu` — requires GPU (auto-skipped without one)
- `@pytest.mark.integration` — tests external APIs or dispatches to real environments (skipped by default)
- `@pytest.mark.slow` — long-running tests
- `@pytest.mark.skip_ci` — skip in CI (e.g., tests that exceed CI runner memory limits)

### CI Integration Tests

Integration tests do not run on every push. To run them on a PR, add the `run-integration` label. Once the label is present, integration tests will re-run automatically on each subsequent push. Remove the label to stop running them.

See [notes/testing.md](notes/testing.md) for full conventions including naming, assertions, and file structure.

## Implementing a New Tool

Every tool follows the universal `Input` / `Config` / `run_{tool}()` / `Output` pattern. The `/implement-tool` Claude Code skill provides the complete step-by-step guide with templates and examples.

Key requirements for new tools:

- Tool directory at `tools/{category}/{tool_name}/`
- `@tool()` decorator registration with all required metadata
- `example_input()` factory function
- `cite.bib` citation file
- `examples/example.ipynb` notebook
- Follow the `__init__.py` export chain: tool -> category -> `tools/__init__.py`

See [CLAUDE.md](CLAUDE.md) for the full architecture reference.

## Commit Messages

- Use imperative mood: "Fix blast timeout" not "Fixed" or "Fixes"
- Keep the first line under 72 characters
- Reference issues where relevant: "Fix blast timeout on large queries (#42)"
- For multi-line messages, leave a blank line after the summary

## Questions?

Open an issue or start a discussion if you're unsure about anything. We're happy to help!
