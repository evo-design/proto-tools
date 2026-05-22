# CLAUDE.md

This file is the short entrypoint for coding agents contributing to
`proto-tools`. Keep long-form guidance in `notes/`, source docstrings, toolkit
READMEs, and local skills so instructions do not drift.

`SYSTEM_PROMPT.md` is for agents that use the existing library to write
programs and scripts. Use this file when editing the repo itself.

## Project Overview

`proto_tools` is the Proto Bio library of typed bioinformatics tool wrappers.
It provides uniform Python APIs, schemas, docs, examples, citations, licenses,
and isolated execution environments for sequence, structure, ligand, genomic,
and publication/database tools. It is also mounted as the `proto-tools/`
submodule inside `proto-language`.

## Read Before Editing

- `README.md`: user-facing setup, usage examples, gated model access, and tool
  catalog.
- Toolkit docs under `proto_tools/tools/{category}/{toolkit}/`: scientific
  context, metadata, examples, and standalone setup for the touched toolkit.
- Relevant `notes/` files before changing runtime behavior, standalone
  environments, logging, storage, seeding, error handling, testing, or platform
  support.
- Source and tests: the final authority for signatures, schemas, exports, and
  behavior.

## Development Setup

Use the active Python environment. For first-time development setup, install
the package in editable mode with dev dependencies:

```bash
pip install -e ".[dev]"      # first-time setup only
ruff check proto_tools tests
ruff format .
mypy proto_tools/
pytest --all
```

Do not manually create or activate tool-specific environments; proto-tools
builds and manages standalone tool environments automatically.

Integration tests are skipped by default. See `notes/testing.md` for markers,
selection flags, logs, and test layout.

## Repository Map

- `proto_tools/tools/{category}/{toolkit}/`: registered tools grouped by
  toolkit. Toolkits contain implementation files, `README.md`, metadata,
  `examples/example.ipynb`, and optional `standalone/` environment code.
- `proto_tools/entities/`: structures, ligands, and other biological data
  objects.
- `proto_tools/utils/`: registry, IO models, caching, execution, device
  management, docs extraction, storage, and logging infrastructure.
- `proto_tools/shared_envs/`: reusable standalone environment definitions.
- `tests/`: style, registry, infrastructure, and tool-specific tests.
- `tutorials/` and `scripts/`: examples, runtime guides, and repository
  utilities.
- `notes/`: canonical long-form developer references.

## Tool Conventions

Every registered tool follows the same broad shape:

```python
Input -> Config -> run_*() -> Output
```

Use the local field helpers and base classes: `BaseToolInput` with input
fields, `BaseConfig` with `ConfigField()`, `BaseToolOutput` with Pydantic
fields, and `Metrics` / `MetricSpec` for scalar metrics. Config is optional at
the public call site because the decorator supplies defaults.

Naming and file placement:

- Tool files live at `proto_tools/tools/{category}/{toolkit}/{tool_key_snake}.py`.
- Registry keys are `{toolkit}-{suffix}` kebab-case.
- Run functions are `run_{tool_key_snake}`.
- A toolkit is the directory/family sharing code, model, environment, and
  persistent worker; a tool is one registered operation.

Rules that affect behavior:

- Never catch exceptions inside tool functions; the decorator owns error
  policy. See `notes/error-handling.md`.
- Declare `stochastic=True` only when outputs depend on `config.seed`, and
  advance RNG per item inside batched tools. See `notes/seeding.md`.
- Outputs must be JSON-serializable Pydantic-compatible types. Do not expose
  DataFrames, numpy arrays, or arbitrary objects as output fields.
- Biological coordinates are 1-indexed and inclusive.
- Use `logging.getLogger(__name__)`, never `print()`. See `notes/logging.md`.
- Keep heavy ML and optional dependencies lazy unless a Pydantic field type
  requires a module-level import.
- Tools with heavy dependencies use `standalone/` or shared environments. See
  `notes/tool-environments.md`.

## Runtime API

`ToolRegistry` is the discovery and runtime surface for tools, schemas, docs,
citations, links, licenses, example inputs, and access requirements. The CLI
mirrors that surface through `proto-tools ...` commands.

Read `notes/runtime-api.md` for identifier resolution, registry methods,
README/doc extraction, schemas, JSON surfaces, gated weights, and calling
patterns.

## Documentation

Generated reference pages come from Python docstrings, Pydantic field
descriptions, toolkit READMEs, notebooks, and metadata files. Update those
sources rather than generated docs.

When behavior changes, update the matching docs in the same commit. Common
targets are:

- Runtime API, registry, docs extraction: `notes/runtime-api.md`.
- Standalone environments, compute deps, binary install, `to_device()`:
  `notes/tool-environments.md`.
- Error handling: `notes/error-handling.md`.
- Logging: `notes/logging.md`.
- Seeding and cache behavior: `notes/seeding.md`.
- Storage and weights: `notes/storage.md`.
- Test markers, placement, and patterns: `notes/testing.md`.
- Tool implementation details: toolkit README, example notebook, metadata,
  source docstrings, tests, and relevant `.claude/skills/` files.

## Skills

- `implement-tool`: full lifecycle for implementing a new wrapper.
- `fix-env`: debugging and fixing standalone environment setup failures.
