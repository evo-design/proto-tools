# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

proto_tools is a modular computational biology and biological AI tool library providing Python wrappers for generative biological AI models, and biological sequence and structure analysis tools/models. It is a git submodule of [proto-language](https://github.com/proto-bio/proto-language), mounted at `proto-tools/`. It also works standalone.

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
| `utils/persistent_worker.py` | `notes/tool-environments.md`, `notes/logging.md` (drain thread / tagged-stream demux) |
| `utils/logging_config.py` | `notes/logging.md` (parent-side: ProtoLogger, spinner handler, install entry points) |
| `utils/standalone_helpers_source/standalone_helpers/proto_logging.py` | `notes/logging.md` (subprocess-side bridge: ProtoLogger, _BridgeHandler, install, get_logger) |
| `utils/_worker_bootstrap.py` | `notes/logging.md` (bridge install via `import standalone_helpers`) |
| `utils/compute_deps.py` | `notes/tool-environments.md` (compatibility matrices) |
| `utils/base_config.py` (`seed`) | CLAUDE.md (Rules When Implementing Tools), `BaseConfig` docstring |
| `tools/tool_registry.py` (`cacheable`, `generative`) | CLAUDE.md (Rules When Implementing Tools), `ToolRegistry` docstrings |
| `tools/tool_registry.py` (`_should_capture_errors`, `_make_error_output_or_raise`) | `notes/error-handling.md`, CLAUDE.md (Rules When Implementing Tools, env var section) |
| `utils/tool_instance.py` | Docstrings (reference pages auto-generated) |
| `utils/device_manager.py` | Docstrings (reference pages auto-generated) |
| `utils/tool_io.py`, `tools/tool_registry.py` | CLAUDE.md (Universal Tool Pattern, Key File Paths) |
| `utils/install_binary.py` | `notes/tool-environments.md` (Binary Installation) |
| `standalone/env_vars.txt` (any tool) | `notes/tool-environments.md` |
| `standalone/setup.sh` patterns | `notes/tool-environments.md`, `fix-env` SKILL.md |
| `standalone/python_version.txt` | `notes/tool-environments.md`, `implement-tool` SKILL.md |
| `standalone/shared_env.txt` or `shared_envs/` | `notes/tool-environments.md` (Shared Environments section) |
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
│   │   ├── {toolkit}/            # e.g., blast, esmfold
│   │   │   ├── __init__.py         # Exports: Input, Config, Output, run_*
│   │   │   ├── {tool_key_snake}.py # Implementation (one per registered tool)
│   │   │   ├── cite.bib            # BibTeX citation (optional if no published paper)
│   │   │   ├── license.yaml        # License metadata (code/weights, commercial use, redistribution)
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

Every tool is registered via `@tool()` and discoverable through `ToolRegistry`. Tools are at `tools/{category}/{toolkit}/`.

```python
from proto_tools.tools import ToolRegistry

ToolRegistry.list_all()                               # All registered tools
ToolRegistry.get_schemas("tool-key")                  # Input, config, output JSON schemas
ToolRegistry.get_citation("tool-key")                 # BibTeX string
ToolRegistry.get_doi("tool-key")                      # DOI extracted from cite.bib
ToolRegistry.get_links("tool-key")                    # Parsed links.yaml (github, image, huggingface, …)
ToolRegistry.get_license("tool-key")                  # Parsed license.yaml (code/weights SPDX, commercial use, …)
ToolRegistry.get_docs_url("tool-key")                 # Documentation URL computed from tool directory
ToolRegistry.get_example_input("esmfold-prediction")  # Minimal valid Input
ToolRegistry.get_example_notebook_path("tool-key")    # Path to examples/example.ipynb
```

### The Universal Tool Pattern

Every tool follows this exact pattern, no exceptions:

```python
def example_input():
    """Minimal valid input for testing and examples."""
    return ToolInput(sequences=["MKTL"])

@tool(
    key="{tool_key}",
    label="Tool Label",
    category="category_name",
    input_class=ToolInput,
    config_class=ToolConfig,
    output_class=ToolOutput,
    description="...",
    example_input=example_input,
    iterable_input_field="sequences",
    iterable_output_field="results",
    cacheable=True,
)
def run_{tool_key_snake}(inputs: ToolInput, config: ToolConfig, instance: Any = None) -> ToolOutput:
```

- **Input** (`BaseToolInput`): primary data (sequences, structures, files). Uses `extra="forbid"`.
- **Config** (`BaseConfig`): parameters (evalue, threads, temperature). Uses `extra="forbid"`. **Optional at call time**; the `@tool` wrapper defaults `None` to `config_class()`. Inner function takes non-optional `config` since the wrapper guarantees it. All config fields must have defaults. `BaseConfig.seed` is the only standard random seed field and participates in cache keys when set.
- **Output** (`BaseToolOutput`): results + auto-populated metadata (tool_id, execution_time, success, errors). Uses `extra="ignore"` with warnings for unexpected fields so computed-field JSON round-trips can validate cleanly.
- **Metrics** (`Metrics`): standardized container for scalar metric values emitted by the tool (plDDT, perplexity, SASA, etc.). Subclass `Metrics` (per-tool or per-category) and declare a `metric_spec: ClassVar[dict[str, MetricSpec]]` mapping each metric name to its type/range. Access values via attribute (`m.plddt`) or mapping (`m["plddt"]`). See `Metrics` / `MetricSpec` in `utils/tool_io.py` for the contract and the `implement-tool` SKILL.md for usage.
- **`@tool()`**: handles error catching, timing, metadata, registry, default config, device allocation validation, and cache policy. Set `generative=True` on tools whose repeated unseeded calls should produce diversified outputs; for `cacheable=True` tools this skips cache/dedup until `config.seed` is set.
- **`example_input`**: callable factory returning a minimal valid `Input`. Must be a public named function (not a lambda).
- **`device_count`** (optional): expected device allocation ("1", "1-2", ">=1"). Defaults to "1".
- **`devices_per_instance`**: a `@property` on `BaseConfig` (not a field) that tells `ToolPool` how many GPUs each worker needs. Default is derived from the `device` string (`cpu` → 0, `cuda`/`cuda:N` → 1, `cudaxN`/multi → N, `cloud` → 1). Override only when GPU need is decoupled from the device string (e.g. a separate `use_gpu` toggle, or model-variant-dependent count). `0` short-circuits the pool to a single direct call.

### Tool Execution & Persistence

Tools run in **isolated micromamba-based environments** via `ToolInstance`. One-shot by default (ephemeral subprocess per call). Use `ToolInstance.persist()` for batch workloads. See `utils/tool_instance.py` docstrings for the full API.

### Device Management

**DeviceManager** (`utils/device_manager.py`) provides centralized GPU allocation with LRU eviction. Works transparently with `ToolInstance.persist()`. See `utils/device_manager.py` docstrings for full API and configuration. See `notes/tool-environments.md` for the `to_device()` protocol when implementing new tools.

### Standalone Environments

Tools with heavy dependencies run in isolated micromamba environments with centralized hardware detection (`utils/compute_deps.py`). See `notes/tool-environments.md` for setup patterns, env_vars.txt, GCC/nvcc compatibility, cache management, binary installation, and Python version specification.

### Key File Paths

| File | Provides |
|---|---|
| `utils/tool_io.py` | `BaseToolInput`, `BaseToolOutput`, `Metrics`, `MetricSpec`, `ToolExecutionError` |
| `utils/base_config.py` | `BaseConfig`, `ConfigField()` |
| `tools/tool_registry.py` | `@tool` decorator, `ToolRegistry`, `ToolSpec` |
| `utils/tool_cache.py` | `ToolCache`, `cache_strip_items`, `cache_store_items`, `cache_stitch_items` |
| `utils/tool_instance.py` | `ToolInstance`: isolated environment execution with opt-in persistence |
| `utils/tool_pool.py` | `ToolPool`: multi-GPU parallel execution with LPT scheduling |
| `utils/device.py` | GPU detection, `DeviceSpec`, `number_of_visible_gpus()`, `determine_visible_devices()` |
| `utils/device_manager.py` | `DeviceManager`: centralized GPU allocation tracking with LRU eviction |
| `utils/compute_deps.py` | `detect_compute_environment()`: hardware detection & version resolution |
| `utils/install_binary.py` | Shared binary downloader for standalone tool environments |
| `utils/standalone_helpers_source/standalone_helpers/` | `resolve_weights_dir()`, `get_subprocess_device_env()`, `move_model_to_device()`, `serialize_output()`, `AMINO_ACIDS_LIST` |
| `utils/standalone_helpers_source/standalone_helpers.sh` | `proto_install_pytorch()`, `proto_install_jax()`, `proto_resolve_weights_dir()`, `proto_check_gated_hf_repo()` |
| `utils/sequence.py` | Sequence validation, detection, `resolve_sequence_ids()` |
| `utils/auth.py` | `require_hf_token()`: HuggingFace gated model auth |
| `utils/chemistry.py` | `validate_smiles()`: SMILES string validation |
| `utils/msa.py` | `extract_msa_sequences()`: MSA extraction utilities |
| `tools/__init__.py` | Master export, all tools re-exported here |

## Key Concepts: `tool`, `toolkit`, `tool_key`, `env_name`

Three identifiers show up across the codebase, plus the informal term "tool" itself. Do not conflate them — each names a different level of the hierarchy.

**A toolkit is a group of specific tools that share the same underlying model or codebase.** For example, `pyrosetta` is a toolkit that bundles 5 distinct tools (`pyrosetta-energy`, `pyrosetta-relax`, `pyrosetta-sap`, `pyrosetta-sasa`, `pyrosetta-interface-analyzer`) — all running against the same PyRosetta installation and sharing one persistent subprocess when a worker is warm. The word **tool** always refers to a single registered operation (what `@tool(key=...)` registers); the **toolkit** is the family they belong to.

| Identifier | Example | What it identifies | Where it's used |
|---|---|---|---|
| **tool** (concept) | `pyrosetta-energy`, `esm2-sample` | A single registered operation. One `@tool(key=...)` invocation = one tool. | Everywhere — the primary unit of functionality |
| **`tool_key`** | `"pyrosetta-energy"`, `"esm2-sample"` | The string identifier for a tool (the value passed to `@tool(key=...)`) | `ToolRegistry`, `ToolCache` (per-operation output cache), error messages |
| **`toolkit`** | `"pyrosetta"`, `"esm2"` | The family of tools sharing a codebase / model / standalone script / persistent subprocess. All `pyrosetta-*` tools belong to the `"pyrosetta"` toolkit. | `ToolInstance`, `DeviceManager`, `PersistentWorker`, `persist_tool()`, worker cache key |
| **`env_name`** | `"pyrosetta_env"`, `"evolutionaryscale_esm_env"` | The physical micromamba env directory on disk. Resolved from the toolkit via `standalone/shared_env.txt`. Multiple toolkits may share one env (e.g. `esm3` and `esmc` both resolve to `evolutionaryscale_esm`). | Internal to `ToolInstance.__init__`; not part of the public dispatch / caching API |

The system invariant is **one persistent worker per toolkit** (not per tool, not per env). APIs that take a `toolkit` (e.g. `ToolInstance.persist_tool`, `DeviceManager.lease`) also accept a `tool_key` as a convenience — the registry maps the tool_key back to its toolkit via `ToolInstance._normalize_toolkit`.

## Naming Conventions

- **Tool registry key** (`tool_key`): `{toolkit}-{suffix}` kebab-case, e.g. `"evo2-sample"`, `"blast-search"`, `"alphafold3-prediction"`, `"alphafold2-binder"`. The suffix names the tool's operation or operational domain and must uniquely distinguish it from sibling tools in the same `{toolkit}` family. Verb-like suffixes (`-search`, `-sample`) describe actions; noun-like suffixes (`-prediction`, `-binder`) describe output/mode when that's the cleaner fit. Pick the suffix that reads naturally; the rule is "disambiguate within a `{toolkit}` family," not "require a verb."
- **Run function**: `run_{tool_key_snake}`, e.g. `run_evo2_sample`, `run_blast_search` (snake_case form of the `@tool(key=...)` registration key)
- **Classes**: PascalCase, tool-specific. Typically the PascalCase of the `tool_key` + `Input`/`Config`/`Output` (e.g. `Evo2SampleInput`, `BlastSearchConfig`), but the PascalCase prefix is the developer's choice — pick whatever reads cleanly (e.g. `ESMFold` over `Esmfold`) as long as it's specific to the tool.
- **Directories**: snake_case, one per `{toolkit}` — e.g. `evo2/`, `blast/`, `pyrosetta/`.
- **Files**: snake_case `{tool_key_snake}.py`, one per registered tool — e.g. `evo2_sample.py` inside `evo2/`, `pyrosetta_energy.py` + `pyrosetta_relax.py` inside `pyrosetta/`. Test files follow the same rule: `tests/{category}_tests/test_{tool_key_snake}.py`.
- **Code section headers**: `# ============================================================================`

**Filesystem mapping recap:** directory = `{toolkit}`; one `.py` per `{tool_key}` named `{tool_key_snake}.py`; test file `test_{tool_key_snake}.py`. The `implement-tool` skill's SKILL.md / TEMPLATES.md / PATTERNS.md use these placeholders verbatim — `{toolkit}`, `{tool_key}`, `{tool_key_snake}` are strict; `{ToolName}` (PascalCase class prefix) and `{tool_display_name}` (human label) are developer's choice.

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
- **Pydantic classes**: Always include `Attributes:` section with full descriptions. These intentionally duplicate the short `Field(description=...)` / `ConfigField(description=...)` strings; field descriptions are short tooltips for the frontend UI, while docstring descriptions are longer developer-facing explanations.

## Rules When Implementing Tools

- Use `ConfigField()` for Config fields, `Field()` for Input and Output fields; never mix them. `ConfigField` supports `reload_on_change=True` (triggers worker restart) and `include_in_key=False` (excludes from cache key). `include_in_key` defaults to `True`; set it to `False` for fields that don't affect computation (device, verbose, timeout, already excluded on `BaseConfig`)
- `seed=None` remains cacheable by default. Set `generative=True` on sampling, gradient, or design tools whose repeated unseeded calls should produce diversified outputs; for `cacheable=True` tools this skips cache and iterable dedup until the caller supplies `config.seed`. Do not set `generative=True` merely because a tool accepts or forwards a seed.
- Never catch exceptions inside tool functions. The `@tool` decorator **raises** by default; set `PROTO_CAPTURE_ERRORS=1` to opt in to packing exceptions into `success=False` outputs. `MissingAssetError` always raises regardless of the env var (the pytest skip hook in `tests/conftest.py` depends on it). See `notes/error-handling.md`.
- All biological coordinates are **1-indexed, inclusive**
- `batch_size` defaults to `1` (prevents OOM). The tool layer owns the batching loop. **Exception**: inverse folding tools default `batch_size` to `num_sequences_per_structure`
- Use `logging.getLogger(__name__)`, never `print()`. Loggers under the `proto_tools.*` namespace are auto-routed: from inside a worker subprocess they're serialized as JSON-tagged stderr lines and re-emitted by the parent on `proto_tools.worker.{toolkit}.*`, where standard `logging` filtering applies. Pass `update_status=True` on a log call to make that record take over the spinner subtitle (e.g. `logger.info("Sampling chain A", update_status=True)`). See `notes/logging.md` for the full architecture.
- Config: `extra="forbid"` | Input: `extra="forbid"` | Output: `extra="ignore"`
- Follow the `__init__.py` export chain: tool → category → `tools/__init__.py` → package `__init__.py`
- Every tool directory must include an `examples/example.ipynb` notebook. Include a `cite.bib` when wrapping a published model or tool; omit it for simple algorithmic utilities with no paper to cite
- Output model fields must be JSON-serializable Pydantic types (primitives, `list`, nested `BaseModel`). Never use `pd.DataFrame`, numpy arrays, or other non-serializable types as `Field()` or `@computed_field` on output models — they break JSON Schema generation for MCP and downstream consumers. Never set `arbitrary_types_allowed=True` on output models. DataFrames are a presentation layer; construct them lazily inside `_export_output()` or provide a `to_dataframe()` method instead

**The `implement-tool` skill provides the complete tool implementation guide with step-by-step templates and examples.**

## Import Conventions

| Location | When to use | Examples |
|---|---|---|
| **Module-level** | stdlib, lightweight deps, anything needed for Pydantic field types | `import json`, `from pathlib import Path`, `import csv` |
| **Lazy (in function body)** | Heavy ML libs, truly optional deps, circular dep breaks | `import torch`, `from rdkit import Chem` |
| **Standalone inference files** | Always lazy — these run in isolated subprocess envs | All imports in `standalone/inference.py` |

Never weaken a Pydantic field type annotation (e.g. `Optional[object]`) to avoid an import. If a type is needed for a field definition, import it at module level.

## Test Conventions

Flat functions only (no test classes). See `notes/testing.md` for full conventions (structure, assertions, markers, naming).

**Quick reference**: `@pytest.mark.integration` for CPU dispatch tests (skipped by default, run with `--integration`). `@pytest.mark.uses_gpu` for GPU dispatch tests (auto-skipped without GPU).

## Configuration

- Python >=3.10, Pydantic >=2.0
- Ruff for linting and formatting (line length 120, 22 rule groups with Google-convention pydocstyle — see `pyproject.toml [tool.ruff.lint]` for full config). Formatting is enforced in CI via `ruff format --check`.
- Mypy strict mode with Pydantic plugin — all code must pass `mypy proto_tools/` with zero errors. Every `# type: ignore` must include the error code (e.g. `# type: ignore[arg-type]`). Use only for genuinely unfixable external-lib issues. Prefer `assert` guards for type narrowing over `# type: ignore`. Do NOT use `cast()`, arbitrary `Protocol` definitions, or `TYPE_CHECKING` blocks to work around type issues. Third-party deps without stubs are listed in `[[tool.mypy.overrides]]`.
- Pytest markers: `uses_gpu`, `uses_cpu`, `slow`, `integration`, `skip_ci`, `asyncio`, `extensive`
- `pytest-randomly` randomizes test order; reproduce with `--randomly-seed=N`
- Branch coverage configured via `[tool.coverage]` in `pyproject.toml`; CI runs `--cov`
- Integration tests are **skipped by default**. Run with `pytest --integration` or `pytest --all`
- GPU dispatch is hardware-gated: plain `pytest` runs `uses_gpu` tests iff a GPU is visible, otherwise skips them automatically. `--cpu-only` and `--gpu-only` are *selection filters* (only that flavor of test runs); they do not change hardware availability
- Other useful flags: `--gpu-only` (GPU tests only), `--cpu-only` (CPU tests only), `--slow` (slow tests only), `--ext` / `--extensive` (combinatorial tests), `--benchmark` (additive opt-in for `@pytest.mark.benchmark` tests; gated off by default and not enabled by `--all`/`--slow`)
- **Generally use `--all` when running tests** to include integration and slow tests
- Test logs saved to `logs/`. Every `pytest` run creates a `logs/pytest_*.log` file. To monitor a running test, tail the latest log file (`ls -t logs/ | head -1`) rather than relying on stdout (which buffers). Check logs before re-running tests
- **`PROTO_HOME`** controls where all persistent data lives: model weights, tool envs, and micromamba (defaults to `~/.proto/`). **`PROTO_MODEL_CACHE`** overrides just the model weights location. Per-tool override: `PROTO_{TOOL}_WEIGHTS_DIR`. All configured via environment variables. See `notes/storage.md`.
- **`PROTO_CAPTURE_ERRORS=1`** switches `@tool` to capture mode, where exceptions are packed into `success=False` outputs instead of raising. See `notes/error-handling.md`.

## Using proto-tools with Claude Code

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

## Skills & Commands

- **`implement-tool`**: Full lifecycle for implementing a new tool wrapper
- **`fix-env`**: Debug and fix tool environment setup failures

## Detailed Reference Docs

| File | Contents |
|---|---|
| `notes/storage.md` | `PROTO_HOME`, `PROTO_MODEL_CACHE`, shared weights, per-tool overrides, storage layout |
| `notes/tool-environments.md` | Standalone env setup, compute deps, GCC/nvcc, caches, binaries, `to_device()` protocol |
| `notes/error-handling.md` | `@tool` raise-vs-capture policy, `PROTO_CAPTURE_ERRORS`, `MissingAssetError` carve-out |
| `notes/logging.md` | Worker logging architecture, `update_status=True` spinner takeover, int verbosity levels, third-party progress bar handling |
| `utils/device_manager.py` | DeviceManager API (auto-generated reference pages from docstrings) |
| `utils/tool_instance.py` | ToolInstance API (auto-generated reference pages from docstrings) |
| `notes/testing.md` | Test structure, assertions, markers, naming conventions |
