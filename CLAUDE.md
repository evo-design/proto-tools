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

At the start of a session, and after switching branch or worktree, confirm the
environment imports the checkout you are editing:

```bash
python -c "import proto_tools; print(proto_tools.__file__)"
```

`pip install -e` records an absolute path, so an environment built for one
worktree keeps importing that worktree from every other one, and several
worktrees sharing one environment all import whichever was installed last. The
symptoms do not point at the cause: expect an `ImportError` for a symbol
plainly present in the file you are reading, or a `pydantic` error about a
model you never touched, both of which mean two trees are on one path.

Fix it by installing this worktree, not by prepending `PYTHONPATH` -- a path
prefix leaves the editable `.pth` entries active and both trees importable,
turning a clear failure into a stranger one.

Repos that vendor proto-tools as a submodule, such as proto-train, add a second
way for this to drift, since the pin moves between their branches.

Never treat a triggered environment rebuild as a cost or a risk when weighing a
change. Editing `setup.sh`, `requirements.txt` or anything else in the env
fingerprint rebuilds that tool's environment, and that is intended. Do not
raise it as a drawback, scope a change around avoiding it, or ask whether the
rebuild is acceptable.

Integration tests are skipped by default. See `notes/testing.md` for markers,
selection flags, logs, and test layout.

Do not write frivolous tests. A test that asserts Pydantic stored the default you
just wrote (`test_sample_config_defaults`, `test_*_config_defaults`) proves nothing:
it restates the source in a second place, so changing a default means editing both
and the test never catches a mistake. Test computed properties, validators,
normalization, error cases, and defaults that are *derived* from other fields.
Pinning a value is only worth it when it must agree with something external, such as
an upstream snapshot.

A test that exercises standalone implementation code directly — anything importing
`torch`, `jax`, or another tool-env dependency into the pytest process — must run
inside the toolkit's environment via `run_in_env`. Never write it against a copy of
that package installed alongside proto-tools, and never reach for
`pytest.importorskip("torch")` to paper over its absence.

The orchestrator has no ML dependencies by design; models live in isolated tool
environments. So `importorskip` on a toolkit's own dependency does not make the test
fail when that dependency is missing — it makes it *skip*, silently and green, both
in CI (which runs `pytest --cpu-only --no-envs`) and in an ordinary local run. Tests
written that way can stop running for months with nobody noticing.

Put the assertions in a `_{thing}_env_check.py` beside the test, run it with
`run_in_env`, and assert on a sentinel in its stdout:

```python
check = Path(__file__).resolve().parent / "_esmc_token_layout_env_check.py"
output = run_in_env("esmc", script=str(check), timeout=1800)
assert "ESMC_TOKEN_LAYOUT_OK" in output, output
```

Mark such tests `uses_env` (or `uses_gpu`, which implies it). Reserve `importorskip`
for genuinely optional extras, never for a dependency the toolkit owns.

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
- `guides/` and `scripts/`: examples, runtime guides, and repository
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
  versions.yaml
  examples/
    example.ipynb
  standalone/              # optional, for isolated tool environments
    inference.py           # or run.py, depending on the toolkit
    setup.sh
    requirements.txt
    python_version.txt
    uv_version.txt         # optional uv pin override (default: UV_VERSION)
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
- Any change to a toolkit's hashed source needs a new top entry in its
  `versions.yaml`; the style tests print the line to add. See
  `notes/versioning.md`.
- Use `logging.getLogger(__name__)`, never `print()`. See `notes/logging.md`.
- Keep heavy ML and optional dependencies lazy unless a Pydantic field type
  requires a module-level import.
- Tools with heavy dependencies use `standalone/` or shared environments. See
  `notes/tool-environments.md`.
- Weight URLs, revisions and checksums live in the worker (`standalone/`
  code or `setup.sh`), next to the download that checks them. The host side
  owns only the checkpoint names a config accepts, and sends the name. See
  `notes/storage.md`.
- **Checkpoint selection is one `model_checkpoint` field**, even when a tool
  has a single checkpoint. It takes a short bundled name (the default) or
  the user's own checkpoint as a path (or https link); never add separate
  `model_name` / `checkpoint_path` / `checkpoint` fields or an empty-string
  default. Bundled files are downloaded by `setup.sh`, and a test asserts
  `setup.sh` fetches every one. Reject a local path in
  `remote_unsupported_reason`. See the implement-tool skill's "one
  `model_checkpoint` field" section; `opendde` and `parade` follow it.
- Credentials (API keys, tokens, account emails and passwords) are
  `ConfigField(credential="TOOLKIT_NAME")` fields named `{toolkit}_{name}`,
  such as `paperclip_api_key` and `PAPERCLIP_API_KEY`. Client code takes them
  from the config and never reads credential environment variables itself.
- A service URL override (`PAPERCLIP_BASE_URL`, `GI_BASE_URL`) is an
  environment variable read where the request is built. It is never a
  `ConfigField` and never a credential.

## Runtime API

`ToolRegistry` is the discovery and runtime surface for tools, schemas, docs,
citations, links, licenses, example inputs, and access requirements. The agent
surface (list, search, schema, example, info, run, workspace) lives once in
`proto_tools/mcp/tools.py`; the MCP server, the `proto-tools` CLI, and the
hosted MCP server are thin adapters over it, so change behavior there.

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
- Toolkit versions and the output `tool_version` field: `notes/versioning.md`.
- Storage and weights: `notes/storage.md`.
- Test markers, placement, and patterns: `notes/testing.md`.
- Tool implementation details: toolkit README, example notebook, metadata,
  source docstrings, tests, and relevant `.claude/skills/` files.

## Skills

- `implement-tool`: full lifecycle for implementing a new wrapper.
- `fix-env`: debugging and fixing standalone environment setup failures.
