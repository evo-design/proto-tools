## Project Overview

`proto_tools` is the Proto Bio library of typed bioinformatics tool wrappers.
It provides uniform Python APIs, schemas, docs, examples, citations, licenses,
and isolated execution environments for sequence, structure, ligand, genomic,
and publication/database tools.

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
```

Do not manually create or activate tool-specific environments; proto-tools
builds and manages standalone tool environments automatically.

Integration tests are skipped by default. See `notes/testing.md` for markers,
selection flags, logs, and test layout.

Before pushing any PR, run the full `tests/style_consistency_tests/` suite (fast,
CPU-only, no env). These run in CI and enforce consistency checks (docstring style,
config/schema field metadata including the <100-char description limit,
README/license/version consistency, tool I/O round-trip, and more) that `ruff` and
`mypy` do not catch (`ruff` ignores `E501`). Failures propagate through configs
embedded by other tools, e.g. an over-long `Mmseqs2HomologySearchConfig` field description
fails every predictor that nests it.

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

## Canonical Toolkit Layout

Use this default layout for each toolkit directory:

```text
proto_tools/tools/{category}/{toolkit}/
  __init__.py
  {tool_key_snake}.py
  README.md
  cite.bib
  license.yaml
  links.yaml
  examples/
    example.ipynb
  standalone/              # optional, for isolated tool environments
    inference.py           # or run.py, depending on the toolkit
    setup.sh
    requirements.txt
    python_version.txt
    env_vars.txt           # optional environment passthrough/blocklist
```

Treat the toolkit directory as the shared home for docs, citations, licenses,
examples, model/runtime setup, and common code. Add one `{tool_key_snake}.py`
file per registered operation unless the toolkit has an established local
pattern. See `notes/tool-environments.md` for the required `standalone/`
contract and shared-environment options.

## Tool Conventions

Every registered tool follows the same broad shape:

```python
Input -> Config -> run_*() -> Output
```

Field helpers: `InputField` on `BaseToolInput`, `ConfigField` on `BaseConfig`,
bare `pydantic.Field` on `BaseToolOutput` and nested submodels. All fields
require `title=` and `description=`. Config is optional at the call site —
the decorator supplies defaults.

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

Read `notes/finding-tools.md` for identifier resolution, registry methods,
README/doc extraction, schemas, JSON surfaces, gated weights, and calling
patterns. Registry list methods return `ToolSpec` objects; use `spec.key` when comparing registered tool names.

## Documentation

Generated reference pages come from Python docstrings, Pydantic field
descriptions, toolkit READMEs, notebooks, and metadata files. Update those
sources rather than generated docs.

To keep a developer-only region of a published Markdown source (toolkit
`README.md`, `notes/storage.md`) off the docs site, wrap it in
`<!-- docs:ignore start -->` / `<!-- docs:ignore end -->`. See `notes/README.md`.

When behavior changes, update the matching docs in the same commit. Common
targets are:

- Runtime API, registry, docs extraction: `notes/finding-tools.md`.
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
