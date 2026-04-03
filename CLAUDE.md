# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

proto_tools is a modular computational biology and biological AI tool library providing Python wrappers for generative biological AI models, and biological sequence and structure analysis tools/models. It is a git submodule of [proto-language](https://github.com/evo-design/proto-language), mounted at `proto-tools/`. It also works standalone.

## Knowledge Management

Three layers for persistent knowledge. Put information in the right one:

| Layer | Location | Shared? | Best For |
|-------|----------|---------|----------|
| **CLAUDE.md** | Repo root (git) | Team | Conventions, architecture, commands, standards |
| **notes/** | `notes/` (git) | Team | Platform compatibility reports, tool-specific gotchas, architecture decisions |
| **Auto-memory** | `~/.claude/.../memory/` | Personal | Debugging patterns, tool/model quirks, non-obvious discoveries |

### notes/

Team-shared development docs. Read at the start of relevant tasks.

- `environments/`: Machine-generated platform compatibility reports (Chimera H100, DGX Spark, Ashley Lab H100, Sherlock)

Update notes/ when you discover something **every developer needs to know** (platform issues, new setup steps, architecture decisions).

### Auto-memory

Save to auto-memory when you discover something **non-obvious during a session** (debugging that took multiple attempts, undocumented behavior, non-obvious coupling, platform-specific issues). Do NOT save anything already in CLAUDE.md or notes/.

## Documentation

Documentation reference pages are auto-generated from Python docstrings and field descriptions. To update documentation, update the source code. Developer reference docs live in `notes/`. See the Detailed Reference Docs table at the bottom of this file.

## Build & Development Setup

Assume `proto_tools` is already installed in the current Python environment. Do **not** create or activate a virtual environment before running tools; just use `python3` directly.

```bash
# First-time setup only
pip install -e ".[dev]"

# Linting & type checking
ruff check proto_tools tests
ruff format .
mypy proto_tools/
```

## Keeping Docs in Sync

When a code change alters behavior documented in this file, any `SKILL.md`, or `notes/*.md`, update the docs in the same change. Key mappings:

| Code area | Update in |
|---|---|
| `utils/persistent_worker.py` | `notes/tool-environments.md` |
| `utils/compute_deps.py` | `notes/tool-environments.md` (compatibility matrices) |
| `utils/tool_instance.py` | Docstrings (reference pages auto-generated) |
| `utils/device_manager.py` | Docstrings (reference pages auto-generated) |
| `utils/tool_io.py`, `tools/tool_registry.py` | CLAUDE.md (Universal Tool Pattern, Key File Paths) |
| `utils/install_binary.py` | `notes/tool-environments.md` (Binary Installation) |
| `standalone/env_vars.txt` (any tool) | `notes/tool-environments.md` |
| `standalone/setup.sh` patterns | `notes/tool-environments.md`, `fix-env` SKILL.md |
| `standalone/python_version.txt` | `notes/tool-environments.md` |
| New tool added/removed | CLAUDE.md (Package Hierarchy if structure changes), Key File Paths |
| New skills or commands added | CLAUDE.md Skills & Commands section |
| pytest markers, test patterns | `notes/testing.md`, CLAUDE.md Configuration |
| Docstring conventions | CLAUDE.md (Docstring Conventions), `tests/style_consistency_tests/test_docstring_style.py` |

## Architecture

### Package Hierarchy

```
proto_tools/
├── tools/                          # All tool wrappers
│   ├── {category}/                 # e.g., gene_annotation, structure_prediction
│   │   ├── {tool_name}/            # e.g., blast, esmfold
│   │   │   ├── __init__.py         # Exports: Input, Config, Output, run_*
│   │   │   ├── tool_name.py        # Implementation
│   │   │   ├── cite.bib            # BibTeX citation (optional if no published paper)
│   │   │   ├── examples/           # Example notebook
│   │   │   │   └── example.ipynb   # Working example with imports and output
│   │   │   └── standalone/         # [optional] Isolated tool environment
│   │   ├── shared_data_models.py   # [optional] Shared schemas
│   │   └── __init__.py             # Re-exports from all tools in category
│   ├── tool_registry.py            # @tool decorator and ToolRegistry
│   └── __init__.py                 # Master re-export of all tools
├── entities/                       # Data structures: Structure, Ligands
└── utils/                          # Shared utilities
```

### Tool Registry: Quick Navigation

Every tool is registered via `@tool()` and discoverable through `ToolRegistry`. Tools are at `tools/{category}/{tool_name}/`.

```python
from proto_tools.tools import ToolRegistry

ToolRegistry.list_all()                          # All registered tools
ToolRegistry.get_schemas("tool-key")             # Input, config, output JSON schemas
ToolRegistry.get_citation("tool-key")            # BibTeX string
ToolRegistry.get_example_input("esmfold-prediction")  # Minimal valid Input
```

### The Universal Tool Pattern

Every tool follows this exact pattern, no exceptions:

```python
def example_input():
    """Minimal valid input for testing and examples."""
    return ToolInput(sequences=["MKTL"])

@tool(key="tool-key", label="Tool Label", category="category_name", input_class=ToolInput, config_class=ToolConfig, output_class=ToolOutput, description="...", example_input=example_input, iterable_input_field="sequences", iterable_output_field="results", cacheable=True)
def run_tool_name(inputs: ToolInput, config: ToolConfig, instance: Any = None) -> ToolOutput:
```

- **Input** (`BaseToolInput`): primary data (sequences, structures, files). Uses `extra="forbid"`.
- **Config** (`BaseConfig`): parameters (evalue, threads, temperature). Uses `extra="ignore"`. **Optional at call time**; the `@tool` wrapper defaults `None` to `config_class()`. Inner function takes non-optional `config` since the wrapper guarantees it. All config fields must have defaults.
- **Output** (`BaseToolOutput`): results + auto-populated metadata (tool_id, execution_time, success, errors).
- **`@tool()`**: handles error catching, timing, metadata, registry, default config, and device allocation validation.
- **`example_input`**: callable factory returning a minimal valid `Input`. Must be a public named function (not a lambda).
- **`device_count`** (optional): expected device allocation ("1", "1-2", ">=1"). Defaults to "1".
- **`devices_per_instance`**: a `@property` on `BaseConfig` (not a field) that tells `ToolPool` how many GPUs each worker needs. Defaults to 1. Override in tool config subclasses where needed.

### Tool Execution & Persistence

Tools run in **isolated micromamba-based environments** via `ToolInstance`. One-shot by default (ephemeral subprocess per call). Use `ToolInstance.persist()` for batch workloads. See `utils/tool_instance.py` docstrings for the full API.

### Device Management

**DeviceManager** (`utils/device_manager.py`) provides centralized GPU allocation with LRU eviction. Works transparently with `ToolInstance.persist()`. See `utils/device_manager.py` docstrings for full API and configuration. See `notes/tool-environments.md` for the `to_device()` protocol when implementing new tools.

### Standalone Environments

Tools with heavy dependencies run in isolated micromamba environments with centralized hardware detection (`utils/compute_deps.py`). See `notes/tool-environments.md` for setup patterns, env_vars.txt, GCC/nvcc compatibility, cache management, binary installation, and Python version specification.

### Key File Paths

| File | Provides |
|---|---|
| `utils/tool_io.py` | `BaseToolInput`, `BaseToolOutput`, `ToolExecutionError` |
| `utils/base_config.py` | `BaseConfig`, `ConfigField()` |
| `tools/tool_registry.py` | `@tool` decorator, `ToolRegistry`, `ToolSpec` |
| `utils/tool_cache.py` | `ToolCache`, `cache_strip_items`, `cache_store_items`, `cache_stitch_items` |
| `utils/tool_instance.py` | `ToolInstance`: isolated environment execution with opt-in persistence |
| `utils/tool_pool.py` | `ToolPool`: multi-GPU parallel execution with LPT scheduling |
| `utils/device.py` | GPU detection, `DeviceSpec`, `number_of_visible_gpus()`, `determine_visible_devices()` |
| `utils/device_manager.py` | `DeviceManager`: centralized GPU allocation tracking with LRU eviction |
| `utils/compute_deps.py` | `detect_compute_environment()`: hardware detection & version resolution |
| `utils/install_binary.py` | Shared binary downloader for standalone tool environments |
| `utils/standalone_helpers_source/standalone_helpers.py` | `resolve_weights_dir()`, `get_subprocess_device_env()`, `move_model_to_device()` |
| `utils/standalone_helpers_source/standalone_helpers.sh` | `proto_install_pytorch()`, `proto_install_jax()`, `proto_resolve_weights_dir()`, `proto_check_gated_hf_repo()` |
| `utils/sequence.py` | Sequence validation, detection, `resolve_sequence_ids()` |
| `utils/auth.py` | `require_hf_token()`: HuggingFace gated model auth |
| `utils/chemistry.py` | `validate_smiles()`: SMILES string validation |
| `utils/msa.py` | `extract_msa_sequences()`: MSA extraction utilities |
| `tools/__init__.py` | Master export, all tools re-exported here |

## Naming Conventions

- **Tool registry key**: `{tool}-{action}` kebab-case, e.g. `"evo2-sample"`, `"blast-search"`, `"alphafold3-prediction"`. Every key must have an action suffix.
- **Run function**: `run_{tool_name}`, e.g. `run_evo2_sample`, `run_blast_search`
- **Classes**: PascalCase, e.g. `Evo2SampleInput`, `Evo2SampleConfig`, `Evo2SampleOutput`
- **Directories**: snake_case, e.g. `evo2/`, `blast/`
- **Files**: snake_case, e.g. `evo2_sample.py`, `blast_search.py`
- **Code section headers**: `# ============================================================================`

## Docstring Conventions

Google style everywhere. Enforced by ruff D rules (Google convention) and `tests/style_consistency_tests/test_docstring_style.py` (type-matching, return-type, continuation indent).

- **Module docstrings**: A one-line Google-style summary ending with a period, or a summary line + blank line + details for longer descriptions. `__init__.py` files in test directories are exempt (D104 suppressed via per-file-ignores for `tests/**/*.py`). No path-header prefix.
  ```python
  """Centralized GPU allocation tracking with LRU eviction."""
  ```
- **One-liners**: Acceptable for simple functions. No structured sections needed.
- **Multi-line docstrings** (anything with a blank line): Google style. Summary line, blank line, then sections as needed: `Args:`, `Returns:`, `Raises:`, `Attributes:`, `Example:`, `Note:`.
- **Types required in docstrings**: Every `Args:`, `Attributes:`, and `Returns:` entry must include the type annotation matching the function signature or class annotation. Use modern Python syntax (`list[str]`, `X | None`). Consistency tests enforce that docstring types match signatures.
  ```python
  Args:
      sequences (list[str]): Input protein sequences.
      config (GCContentConfig | None): Optional configuration.

  Attributes:
      min_gc (float): Minimum acceptable GC content percentage.

  Returns:
      list[float]: Constraint scores for each sequence.
  ```
- **Pydantic classes**: Always include `Attributes:` section with full descriptions. These intentionally duplicate the short `Field(description=...)` / `ConfigField(description=...)` strings; field descriptions are short tooltips for the client UI, while docstring descriptions are longer developer-facing explanations.

## Rules When Implementing Tools

- Use `ConfigField()` for Config fields, `Field()` for Input and Output fields; never mix them. `ConfigField` supports `reload_on_change=True` (triggers worker restart) and `include_in_key=False` (excludes from cache key). `include_in_key` defaults to `True`; set it to `False` for fields that don't affect computation (device, verbose, timeout, already excluded on `BaseConfig`)
- Never catch exceptions inside tool functions; the `@tool` decorator handles all error wrapping
- All biological coordinates are **1-indexed, inclusive**
- `batch_size` defaults to `1` (prevents OOM). The tool layer owns the batching loop. **Exception**: inverse folding tools default `batch_size` to `num_sequences_per_structure`
- Use `logging.getLogger(__name__)`, never `print()`
- Config: `extra="ignore"` | Input: `extra="forbid"` | Output: `extra="forbid"`
- Follow the `__init__.py` export chain: tool → category → `tools/__init__.py` → package `__init__.py`
- Every tool directory must include an `examples/example.ipynb` notebook. Include a `cite.bib` when wrapping a published model or tool; omit it for simple algorithmic utilities with no paper to cite
- Output model fields must be JSON-serializable Pydantic types (primitives, `list`, nested `BaseModel`). Never use `pd.DataFrame`, numpy arrays, or other non-serializable types as `Field()` or `@computed_field` on output models — they break JSON Schema generation for MCP and downstream consumers. Never set `arbitrary_types_allowed=True` on output models. DataFrames are a presentation layer; construct them lazily inside `_export_output()` or provide a `to_dataframe()` method instead

**The `implement-tool` skill provides the complete tool implementation guide with step-by-step templates and examples.**

## Test Conventions

Flat functions only (no test classes). See `notes/testing.md` for full conventions (structure, assertions, markers, naming).

**Quick reference**: `@pytest.mark.integration` for CPU dispatch tests (skipped by default, run with `--integration`). `@pytest.mark.uses_gpu` for GPU dispatch tests (auto-skipped without GPU).

## Configuration

- Python >=3.10, Pydantic >=2.0
- Ruff for linting and formatting (line length 120, 22 rule groups with Google-convention pydocstyle — see `pyproject.toml [tool.ruff.lint]` for full config). Formatting is enforced in CI via `ruff format --check`.
- Mypy strict mode with Pydantic plugin — all code must pass `mypy proto_tools/` with zero errors. Every `# type: ignore` must include the error code (e.g. `# type: ignore[arg-type]`). Use only for genuinely unfixable external-lib issues. Prefer `assert` guards for type narrowing over `# type: ignore`. Do NOT use `cast()`, arbitrary `Protocol` definitions, or `TYPE_CHECKING` blocks to work around type issues. Third-party deps without stubs are listed in `[[tool.mypy.overrides]]`.
- Pytest markers: `uses_gpu`, `uses_cpu`, `slow`, `integration`, `skip_ci`, `asyncio`, `only_chimera`, `exhaustive`
- Integration tests are **skipped by default**. Run with `pytest --integration` or `pytest --all`
- Other useful flags: `--gpu` (GPU tests only), `--slow` (slow tests only), `--exhaustive` (combinatorial tests)
- **Generally use `--all` when running tests** to include integration and GPU tests
- Before running GPU tests, check GPU availability. No GPU → `pytest --cpu`
- Test logs saved to `logs/`. Every `pytest` run creates a `logs/pytest_*.log` file. To monitor a running test, tail the latest log file (`ls -t logs/ | head -1`) rather than relying on stdout (which buffers). Check logs before re-running tests
- **`PROTO_HOME`** controls where all persistent data lives: model weights, tool envs, and micromamba (defaults to `~/.proto/`). **`PROTO_MODEL_CACHE`** overrides just the model weights location. Per-tool override: `PROTO_{TOOL}_WEIGHTS_DIR`. All configured via environment variables. See `notes/storage.md`.

## Using proto-tools with Claude Code

### MCP Server

The MCP server has been migrated to [the tools backend](https://github.com/evo-design/the tools backend). See that repo for MCP server setup and usage.

### Running Tools Directly

When a user asks to run a bioinformatics tool:
1. **Find the tool**: Browse `proto_tools/tools/` or use `ToolRegistry.list_all()`
2. **Read README + notebook**: `tools/{category}/{tool}/README.md` and `examples/example.ipynb`
3. **Read API**: Tool's `Input`/`Config`/`Output` classes for the Pydantic schema
4. **Call**: `Input` → `Config` → `run_{tool}()` → `Output`

```python
from proto_tools.tools.{category}.{tool} import run_{tool}, {Tool}Input, {Tool}Config

result = run_{tool}({Tool}Input(...), {Tool}Config(...))
# result.success, result.execution_time, result.errors, plus tool-specific fields
# Config is optional; omit to use all defaults: run_{tool}({Tool}Input(...))
```

For script patterns, batch persistence, GPU tools, and citations, see `notes/usage-guide.md`.

## Skills & Commands

- **`implement-tool`**: Full lifecycle for implementing a new tool wrapper
- **`fix-env`**: Debug and fix tool environment setup failures

## Detailed Reference Docs

| File | Contents |
|---|---|
| `notes/storage.md` | `PROTO_HOME`, `PROTO_MODEL_CACHE`, shared weights, per-tool overrides, storage layout |
| `notes/tool-environments.md` | Standalone env setup, compute deps, GCC/nvcc, caches, binaries, `to_device()` protocol |
| `utils/device_manager.py` | DeviceManager API (auto-generated reference pages from docstrings) |
| `utils/tool_instance.py` | ToolInstance API (auto-generated reference pages from docstrings) |
| `notes/testing.md` | Test structure, assertions, markers, naming conventions |
| `notes/usage-guide.md` | Script patterns, batch persistence, GPU tools, citations |
