# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

bio_programming_tools is a modular computational biology and biological AI tool library providing Python wrappers for generative biological AI models, and biological sequence and structure analysis tools/models. It is a git submodule of [bio-programming](https://github.com/evo-design/bio-programming), mounted at `bio-programming-tools/`. It also works standalone.

## Notes (`notes/`)

Dynamic development notes. **Read these at the start of relevant tasks. Actively update them** when you discover new gotchas, resolve issues, or learn something future sessions should know — don't ask, just update and mention what you added.

- `dev.md` — Setup, pre-commit hooks, CI checks, docs generation, code quality tools.
- `environment.md` — Platform compatibility matrix (Chimera H100, DGX Spark GB10) and per-tool venv status.

## Build & Development Setup

Assume `bio_programming_tools` is already installed in the current Python environment. Do **not** create or activate a virtual environment before running tools — just use `python3` directly.

```bash
# First-time setup only
pip install -e ".[dev]"
pre-commit install

# Formatting
black bio_programming_tools
isort bio_programming_tools
```

## Architecture

### Package Hierarchy

```
bio_programming_tools/
├── tools/                          # All tool wrappers
│   ├── {category}/                 # e.g., gene_annotation, structure_prediction
│   │   ├── {tool_name}/            # e.g., blast, esmfold
│   │   │   ├── __init__.py         # Exports: Input, Config, Output, run_*
│   │   │   ├── tool_name.py        # Implementation
│   │   │   ├── cite.bib            # BibTeX citation for the tool
│   │   │   ├── examples/           # Example notebook
│   │   │   │   └── example.ipynb   # Working example with imports and output
│   │   │   └── standalone/         # [optional] Isolated venv
│   │   ├── shared_data_models.py   # [optional] Shared schemas
│   │   └── __init__.py             # Re-exports from all tools in category
│   ├── tool_registry.py            # @tool decorator and ToolRegistry
│   └── __init__.py                 # Master re-export of all tools
├── entities/                       # Data structures: Structure, Ligands
└── utils/                          # Shared utilities
```

### Tool Registry — Quick Navigation

Every tool is registered via the `@tool()` decorator and discoverable through `ToolRegistry`. Tools are organized as `tools/{category}/{tool_name}/`.

**To find a tool's source code:**
1. List category directories: `ls bio_programming_tools/tools/` — each subdirectory is a category
2. List tools in a category: `ls bio_programming_tools/tools/{category}/` — each subdirectory is a tool
3. Read the tool's `__init__.py` to see its exported classes and run function
4. Each tool's main implementation is in `{tool_name}/{tool_name}.py` (or `{tool_name}/{operation}.py` for multi-operation tools)

**To programmatically discover all registered tools:**
```python
from bio_programming_tools.tools import ToolRegistry
for spec in ToolRegistry.list_all():
    print(f"{spec.key}: {spec.label} — {spec.description}")
```

**To get a tool's schemas:**
```python
ToolRegistry.get_schemas("tool-key")  # Returns input, config, output JSON schemas
```

**To get a tool's citation:**
```python
ToolRegistry.get_citation("tool-key")  # Returns BibTeX string
ToolRegistry.list_citations()          # Returns {tool_key: bibtex} for all tools
```

### The Universal Tool Pattern

Every tool follows this exact pattern — no exceptions:

```python
@tool(key="tool-key", label="Tool Label", input=ToolInput, config=ToolConfig, output=ToolOutput, description="...")
def run_tool_name(inputs: ToolInput, config: ToolConfig) -> ToolOutput:
```

- **Input** (`BaseToolInput`) — primary data (sequences, structures, files). Uses `extra="forbid"`.
- **Config** (`BaseConfig`) — parameters (evalue, threads, temperature). Uses `extra="ignore"`.
- **Output** (`BaseToolOutput`) — results + auto-populated metadata (tool_id, execution_time, success, errors).
- **`@tool()`** decorator — handles error catching, timing, metadata population, registry.

### Binary Installation (`install_binary.py`)

Tools that need external binaries (not available via pip) must use the shared `utils/install_binary.py` utility — never raw `curl`/`wget` in `setup.sh`.

1. Create `standalone/binary_config.py` with:
   - `URLS`: dict mapping `(system, machine)` tuples to download URLs (use `"arm64"` not `"aarch64"`)
   - `extract(archive_path: Path, bin_dir: Path)`: extracts/copies binaries into the venv's `bin/`
2. In `setup.sh`, call: `python "$SEARCH_DIR/utils/install_binary.py" <tool_name>` (see blast or mmseqs for the standard pattern)

For platform-independent tools (e.g., Java JARs), use the same URL for all platform keys and generate any wrapper scripts in `extract()`.

### Key File Paths

| File | Provides |
|---|---|
| `utils/tool_io.py` | `BaseToolInput`, `BaseToolOutput`, `ToolExecutionError` |
| `tools/tool_registry.py` | `@tool` decorator, `ToolRegistry`, `ToolSpec` |
| `utils/tool_cache.py` | `@tool_cache`, `@tool_cache_iterable` |
| `utils/env_manager.py` | `EnvManager` for isolated venv execution |
| `utils/install_binary.py` | Shared binary downloader for standalone venvs |
| `utils/helpers.py` | `resolve_sequence_ids()` and shared utilities |
| `tools/__init__.py` | Master export — all tools re-exported here |

## Naming Conventions

- **Tool registry key**: `{tool}-{action}` kebab-case — `"evo2-sample"`, `"blast-search"`, `"alphafold3-prediction"`. Every key must have an action suffix.
- **Run function**: `run_{tool_name}` — `run_evo2_sample`, `run_blast_search`
- **Classes**: PascalCase — `Evo2SampleInput`, `Evo2SampleConfig`, `Evo2SampleOutput`
- **Directories**: snake_case — `evo2/`, `blast/`
- **Files**: snake_case — `evo2_sample.py`, `blast_search.py`
- **Code section headers**: `# ============================================================================`

## Rules When Implementing Tools

- Use `ConfigField()` for Config fields, `Field()` for Input and Output fields — never mix them
- Never catch exceptions inside tool functions — the `@tool` decorator handles all error wrapping
- All biological coordinates are **1-indexed, inclusive**
- Use `from __future__ import annotations` at the top of every file
- Use `logging.getLogger(__name__)` — never `print()`
- Config: `extra="ignore"` | Input: `extra="forbid"` | Output: `extra="forbid"`
- Follow the `__init__.py` export chain: tool → category → `tools/__init__.py` → package `__init__.py`
- Every tool directory must include a `cite.bib` file with the BibTeX citation for the underlying paper/tool
- Every tool directory must include an `examples/example.ipynb` notebook with working code, exact imports, API reference tables, and example output

**Run `/implement-tool` for the complete tool implementation guide with step-by-step templates and examples.**

## Configuration

- Python >=3.12, Pydantic >=2.0
- Black/isort line length: 88
- Flake8 only checks: F401 (unused imports), F841 (unused variables)
- Pytest markers: `uses_gpu`, `uses_cpu`, `slow`, `integration`, `skip_ci`, `asyncio`, `only_chimera`
- Tests auto-mark as `uses_cpu` unless explicitly marked `uses_gpu`

## Using bio_tools with Claude Code

When a user asks to run a bioinformatics tool:
1. Browse `bio_programming_tools/tools/` to find the right category and tool directory
2. Read the tool's `README.md` for biological context, parameters, and examples
3. Read the tool's `Input`/`Config`/`Output` classes for the exact API

```python
from bio_programming_tools.tools.{category}.{tool} import run_{tool}, {Tool}Input, {Tool}Config

inputs = {Tool}Input(...)   # Primary data: sequences, structures, files
config = {Tool}Config(...)  # Parameters: evalue, num_threads, seeds
result = run_{tool}(inputs, config)
# result.success, result.execution_time, result.errors, plus tool-specific fields
```

**Invoke the `bio-tools` skill for full workflow guidance** (script vs direct execution, script structure, GPU handling, output conventions).

## Skills (`.claude/skills/`) & Commands (`.claude/commands/`)

### For users (running tools)

- **bio-tools** — workflow for running, analyzing, and writing scripts for any bioinformatics tool (discovery, script generation, GPU handling, output conventions)

### For developers (extending the tool library)

Commands (invoked with `/command-name [args]`):

- **`/fix-issue <number>`** — full GitHub issue fix lifecycle for bio-programming-tools (read issue, explore, reproduce, fix, test, verify)
- **`/implement-tool`** — step-by-step guide for implementing a new tool wrapper (architecture, templates, export chain, examples, tests)
