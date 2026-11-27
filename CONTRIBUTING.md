# Contributing to proto-tools

Contributions are welcome, whether you want to wrap a new model, improve an existing tool, or make the library easier to use. The project is actively developed and has established patterns. Please follow those patterns when making changes. Coding agents can help with this.

This guide covers the conventions and workflows used in this project.

## Development Setup

All you need is Python 3.10+ and pip:

```bash
pip install -e ".[dev]"
```

A shared foundation environment installs system tools such as git, curl, and gcc on first use. You do not need to install them manually.


### Agent skills

Task-specific guides live in [`.claude/skills/`](.claude/skills/). Claude Code surfaces them as slash commands (e.g. `/implement-tool`); other agents read the same `SKILL.md` files directly.

- **`implement-tool`**: step-by-step guide for implementing a new tool wrapper (architecture, templates, export chain, examples, tests)
- **`fix-env`**: troubleshooting recipes for tool environment setup failures


### Storage for developers

All persistent data (tool environments, model weights, micromamba) lives under `PROTO_HOME` (defaults to `~/.proto/`). See the [README](README.md) setup instructions.

```bash
# Add to ~/.bashrc
export PROTO_HOME=/path/to/your/proto_home
```

See the [README](README.md) for HuggingFace authentication setup.

## Branch Naming

Use descriptive branch names with a category prefix:

- `feat/description`: new features or tools
- `fix/description`: bug fixes
- `refactor/description`: code restructuring without behavior change
- `docs/description`: documentation-only changes
- `test/description`: test additions or fixes

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

- `Closes #123`: automatically closes the issue when merged
- `Fixes #123`: same behavior, preferred for bug fixes
- `Related to #123`: links without auto-closing

## Issues

We use three issue templates:

- **Bug report**: something is broken (include reproduction steps, error output, environment)
- **Feature request**: propose new functionality (include use case, proposed solution)
- **Tool request**: request a new bioinformatics tool wrapper (include tool name, operations to wrap, motivation)

When filing a bug, include the full traceback and your environment details (OS, Python version, GPU if relevant).

## Code Style

### Formatting

- **ruff**: enforced. Linting covers 22 rule groups including Pyflakes, pycodestyle, isort, pyupgrade, bugbear, bandit, pydocstyle (Google convention), and more (line length 120). Formatting is enforced in CI via `ruff format --check`. Run `ruff check proto_tools tests` before committing

### Conventions

- Mypy strict mode with Pydantic plugin. Every `# type: ignore` must include the error code. Prefer `assert` guards for type narrowing. Do NOT use `cast()`, `Protocol`, or `TYPE_CHECKING` blocks
- Use `logging.getLogger(__name__)`, never `print()`
- All biological coordinates are **1-indexed, inclusive**
- Pydantic v2 models: Config uses `extra="forbid"`, Input uses `extra="forbid"`, Output uses `extra="forbid"`

## Testing

Tests use flat functions only, no test classes.

### Running Tests

```bash
pytest                          # run everything the host can handle (GPU tests run iff a GPU is visible; slow + network + extensive skipped)
pytest --gpu-only               # filter: only GPU-marked tests
pytest --cpu-only               # filter: only CPU-only tests
pytest --no-envs                # filter: skip tests that build or run a tool env (what CI's unit jobs run)
pytest --network                # add: include tests that call live external services
pytest --ext                    # add: include extensive (combinatorial) tests
pytest --benchmark              # add: include @pytest.mark.benchmark tests (slow gate bypassed)
pytest --all                    # add: include slow (does NOT include network, extensive or benchmark)
pytest --all --cpu-only         # add slow, but skip GPU
```

### Markers

- `@pytest.mark.uses_gpu`: requires GPU (auto-skipped without one)
- `@pytest.mark.uses_env`: builds or runs a tool env on CPU (runs by default; `--no-envs` skips it, along with every GPU test)
- `@pytest.mark.network`: calls a live external service (skipped unless `--network`)
- `@pytest.mark.slow`: long-running tests
- `@pytest.mark.skip_ci`: skip in CI (e.g., tests that exceed CI runner memory limits)

### CI Integration Tests

CI's unit jobs run with `--no-envs`, so tests that build a tool env, and tests that call live services, do not run on every push. To run them on a PR, add the `run-integration` label: the Integration Tests workflow runs `pytest --network --cpu-only`. Once the label is present, it re-runs automatically on each subsequent push. Remove the label to stop running them. It also runs daily.

See [notes/testing.md](notes/testing.md) for full conventions including naming, assertions, and file structure.

## Implementing a New Tool

Every tool follows the `Input` / `Config` / `run_{tool}()` / `Output` pattern. The `/implement-tool` Claude Code skill provides the complete step-by-step guide with templates and examples.

Key requirements for new tools:

- Tool directory at `tools/{category}/{toolkit}/`
- `@tool()` decorator registration with all required metadata
- `example_input()` factory function
- `cite.bib` citation file
- `examples/example.ipynb` notebook
- Follow the `__init__.py` export chain: tool -> category -> `tools/__init__.py`

To create a custom toolkit for your project, either:

- Add proto-tools to your project as a git submodule, install it in editable mode (`pip install -e path/to/proto-tools`), and add the toolkit inside it following the steps above.
- Define the toolkit in your package as an [out-of-tree toolkit](notes/external-toolkits.md).

See [CLAUDE.md](CLAUDE.md) for the full architecture reference.

## Commit Messages

- Use imperative mood: "Fix blast timeout" not "Fixed" or "Fixes"
- Keep the first line under 72 characters
- Reference issues where relevant: "Fix blast timeout on large queries (#42)"
- For multi-line messages, leave a blank line after the summary

## Questions?

Open an issue or start a discussion if you're unsure about anything. We're happy to help!
