---
name: implement-tool
description: >
  Implements a new bioinformatics tool wrapper in proto-tools using a
  parallelized agent pipeline. Orchestrates 6 phases: Research, Contract (core
  tool file), Fan-out (5 parallel subagents), Verify, Self-Audit, and Ship. Use
  when creating tools from GitHub issues, wrapping models, or implementing new
  tool wrappers end-to-end.
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Glob
  - Grep
  - Task
  - WebFetch
  - WebSearch
  - AskUserQuestion
---

# Implement a New Tool — Orchestrator Pipeline

You are implementing a new bioinformatics tool in the proto-tools codebase. This skill orchestrates the full lifecycle using parallel subagents for speed.

**Pipeline overview:**
```
Phase 1: Research → Phase 2: Contract → Phase 3: Fan-out (5 parallel agents) → Phase 4: Verify → Phase 4.5: Self-Audit → Phase 5: Ship
```

**Key principle:** The core tool file (Input/Config/Output + `@tool()` + `run_*()`) is the **contract** everything else depends on. Write it first (sequential), then fan out to subagents (parallel).

**CRITICAL: Standalone scripts run in isolated environments and MUST NOT import from `proto_tools`:**
- `from proto_tools.utils import ...` — will fail at runtime
- `from proto_tools.entities import ...` — will fail at runtime
- `from standalone_helpers import get_subprocess_device_env` — auto-copied by worker bootstrap (OK)
- Standard library imports: `import json`, `import subprocess`, etc. (OK)
- Dependencies from `requirements.txt`: `import torch`, `import numpy`, etc. (OK)
- NEVER install `proto_tools` in standalone environments (creates circular dependency, breaks isolation)

---

## Phase 0: Parse Input

The user provides EITHER:
- A **GitHub issue URL/number** (e.g., `#53` or `https://github.com/proto-bio/proto-tools/issues/53`)
- A **tool name + source repo** (e.g., "ESM-IF from https://github.com/facebookresearch/esm")

**If a GitHub issue is provided:**
1. Fetch the issue with `gh issue view <number> --json title,body,labels`
2. Extract from the issue body: tool name, source repo URLs, paper links, category, operations
3. Present the extracted info to the user for confirmation

**If direct tool info is provided:**
1. Confirm tool name, category, and source repo with the user

**Output of Phase 0:** A clear specification:
- Tool name (e.g., `esmif`)
- Category (e.g., `inverse_folding`)
- Operations (e.g., `sample`, `score`)
- Source repo URL(s)
- Paper DOI (if available)
- Whether it uses shared data models from the category

---

## Phase 1: Research

**Goal:** Understand the tool's API, inputs, outputs, and dependencies.

**Steps:**

1. **Fetch source repo README** — Use WebFetch or `gh` to read the README of the research repo. Extract:
   - Installation instructions (pip packages, dependencies)
   - Python API or CLI interface
   - Input/output formats
   - Model weights location (HuggingFace, GitHub releases, etc.) — determines which weight management pattern to use:
     - **HuggingFace `from_pretrained`** → no weight code needed (HF_HOME set automatically). Example: `tools/masked_models/esm2/`
     - **Direct download in setup.sh** → must implement `PROTO_MODEL_CACHE` pattern. Example: `tools/inverse_folding/fampnn/standalone/setup.sh`
     - **Foundry install** → must implement `PROTO_MODEL_CACHE` pattern. Example: `tools/inverse_folding/ligandmpnn/standalone/setup.sh`
     - **Cannot be auto-downloaded** (the asset is too large to auto-download, or there's no programmatic fetch path at all — e.g. license requires a manual approval flow before the user receives a one-time download link). Distinct from *gated-but-fetchable* assets like HF gated repos, where setup.sh can still download once a token is present (use `proto_check_gated_hf_repo` for those). For not-fetchable assets, the user must place files on disk themselves; setup.sh doesn't fetch. → use `proto_resolve_asset_availability` as a fail-fast precheck at the top of `setup.sh`. Tests skip cleanly on hosts without weights; misconfigured `PROTO_<TOOLKIT>_WEIGHTS_DIR` still fails. Example: `tools/structure_prediction/alphafold3/standalone/setup.sh`. See PATTERNS.md → "standalone/setup.sh for tools whose assets are NOT automatically downloaded".

2. **Read key source files** — Find and read the research repo's inference/prediction scripts to understand:
   - How the model is loaded
   - What inputs it expects (sequences, structures, files)
   - What outputs it produces (scores, sequences, structures)
   - Device handling (CUDA/CPU)

3. **Read existing tools in the same category** — Find the reference implementation:
   ```bash
   ls proto_tools/tools/{category}/
   ```
   Read the reference tool's:
   - Main tool file (the `@tool()` decorated function)
   - `standalone/inference.py` (or `run.py`)
   - `standalone/setup.sh`
   - `standalone/requirements.txt`
   - `standalone/python_version.txt` (required — the Python version pin for the env)
   - `__init__.py` (at tool and category level)
   - `README.md`
   - `cite.bib`
   - Test file in `tests/{category}_tests/`

4. **Check for shared data models** — If the category has a `shared_data_models.py`, read it. The new tool should extend these base classes rather than creating new ones.

**Output of Phase 1:** Mental model of:
- The tool's Python API (function signatures, class names)
- Its dependencies and installation requirements
- How it maps to the existing tool patterns
- Which reference tool to use as the structural template

---

## Phase 2: Contract (Write Core Tool File)

**Goal:** Write the core tool file that defines the API surface. Everything else depends on this.

This phase is **sequential** — no subagents. The orchestrator writes this directly.

**Steps:**

**Placeholder glossary** (used throughout this skill — all distinct concepts, not interchangeable):

- `{toolkit}` — snake_case directory name shared by all tools in the family (e.g., `evo2`, `pyrosetta`). **Strict** — drives the directory path and the dispatch/worker identifier.
- `{tool_key}` — kebab-case registration key, `{toolkit}-{suffix}` (e.g., `evo2-sample`, `pyrosetta-energy`). **Strict** — the value passed to `@tool(key=...)`.
- `{tool_key_snake}` — snake_case form of `{tool_key}` (e.g., `evo2_sample`). **Strict** — the core tool file name, the `run_*` function, and the test file.
- `{ToolName}` — PascalCase class-name prefix for this tool's `Input` / `Config` / `Output` (e.g., `Evo2Sample`, `ESMFoldPrediction`). **Developer's choice** — typically the PascalCase of `{tool_key}`, but pick whatever reads cleanly (e.g., `ESMFold` over `Esmfold`) as long as it's specific to this tool.
- `{tool_display_name}` — human-readable label (e.g., `"Evo 2"`). Used in docstrings, commit messages, and PR titles.

1. Create the tool directory structure:
   ```bash
   mkdir -p proto_tools/tools/{category}/{toolkit}/standalone
   mkdir -p proto_tools/tools/{category}/{toolkit}/examples
   ```

   Target file tree:
   ```
   tools/{category}/
   +-- shared_data_models.py   # Shared Input/Config/Output base classes (if category has 2+ tools)
   +-- {toolkit}/
   |   +-- __init__.py
   |   +-- {tool_key_snake}.py # Core tool file (Input/Config/Output + @tool + run_*); one per registered tool
   |   +-- cite.bib            # BibTeX citation (required)
   |   +-- README.md
   |   +-- examples/
   |   |   +-- example.ipynb   # Working example notebook (required)
   |   +-- helpers.py            # [optional] Plain-type helpers shared with deployment service
   |   +-- standalone/
   |   |   +-- setup.sh
   |   |   +-- run.py OR inference.py  # run.py for CPU tools, inference.py for AI models
   |   |   +-- requirements.txt
   |   |   +-- binary_config.py   # [optional]
   +-- __init__.py
   ```

2. Write the core tool file `proto_tools/tools/{category}/{toolkit}/{tool_key_snake}.py` with:
   - Proper imports
   - Input class extending `BaseToolInput` (or shared base) with `Field()` — `extra="forbid"`
   - Config class extending `BaseConfig` (or shared base) with `ConfigField()` — `extra="forbid"`. Use `reload_on_change=True` on fields that require worker restart (model checkpoint, etc.). Use `include_in_key=False` on fields that don't affect computation results (device, verbose, timeout are already excluded on `BaseConfig`; tool-level overrides of `device` must also set `include_in_key=False`). `include_in_key` defaults to `True`
   - Output class extending `BaseToolOutput` (or shared base) with `Field()` — `extra="forbid"`
   - `@tool()` decorator with all 9 required kwargs: key, label, category, input_class, config_class, output_class, description, uses_gpu, example_input (plus optional `device_count`, `cacheable`, and `generative`)
   - `run_*()` function that calls `ToolInstance.dispatch()`
   - If category has shared data models, use type aliases (e.g., `ToolInput = InverseFoldingInput`)
   - **Metrics**: if the tool emits scalar metric-like values (plDDT, perplexity, scores, etc.), route them through a `Metrics` subclass (from `proto_tools/utils/tool_io.py`). Prefer a shared per-category class (`MaskedModelScoringMetrics`, `InverseFoldingScoringMetrics`, `CausalModelScoringMetrics`, etc.) when the metric set matches siblings. Otherwise declare a per-tool `<Tool>Metrics(Metrics)` subclass in the tool file with a `metric_spec: ClassVar[dict[str, MetricSpec]]` documenting type/range for each metric. Verification lives in each tool's e2e test via `assert_metrics_in_spec(result)` (helper at `tests/tool_infra_tests/_metric_helpers.py`).

**Critical conventions:**
- Tool registry key (`{tool_key}`): `{toolkit}-{suffix}` kebab-case (e.g., `"esmif-sample"`)
- Run function: `run_{tool_key_snake}` (e.g., `run_esmif_sample`)
- Classes: `{ToolName}Input/Config/Output`, PascalCase (e.g., `ESMIFSampleInput`)
- `batch_size` defaults to `1` for GPU tools
- No try/except — `@tool` decorator handles errors
- Use `logging.getLogger(__name__)`, never `print()`
- Output must implement `output_format_options`, `output_format_default`, `_export_output()`

**Inherited field audit:** When reusing a shared base config (e.g., `InverseFoldingConfig`) or base input, enumerate every inherited field and verify the target model can implement it. For each unsupported field, either implement support (e.g., logit masking for `excluded_amino_acids`) or override the field with a validator that raises `ValueError("'{field_name}' is not supported by {tool_display_name}")` when a non-default value is provided. Do not silently inherit fields that the model ignores.

## Code Style Conventions

These ensure consistent formatting across all generated tools:

1. **Module docstrings** — Use one-liner format: `"""Tool description."""`. No multi-line module docstrings.
2. **No unused imports** — Only import what's actually used. Don't speculatively include `Path`, `Optional`, `numpy`, etc.
3. **Type alias labels** — When using type aliases for shared data models, label them with comments:
   ```python
   # Input:
   ToolSampleInput = InverseFoldingInput
   # Output:
   ToolSampleOutput = InverseFoldingOutput
   ```
4. **ToolInstance import at top level** — Import `ToolInstance` at module level, not lazily inside the run function:
   ```python
   from proto_tools.utils.tool_instance import ToolInstance
   ```
5. **logger.debug() before main loop** — Add a status message before the processing loop:
   ```python
   logger.debug("Using local venv for {toolkit} {operation}")
   ```
6. **Don't redefine inherited fields** — Fields like `verbose`, `timeout`, and `device` are inherited from `BaseConfig`. Don't redeclare them in tool-specific Config classes unless overriding the default value.
7. **`__init__.py` files** — Sort `__all__` alphabetically.

For the complete tool file template, refer to `.claude/skills/implement-tool/TEMPLATES.md`.
For implementation patterns (CPU/GPU/compile-from-source), refer to `.claude/skills/implement-tool/PATTERNS.md`.

---

## Phase 3: Fan-Out (5 Parallel Subagents)

**Goal:** Generate all supporting files in parallel. Each subagent gets the contract (tool file) + one reference file + a focused prompt.

**IMPORTANT:** Launch all 5 `Task()` calls in a **single message**. If sent in separate messages, the first blocks and the rest wait.

Before launching subagents:
1. Read the contract tool file you just wrote (Phase 2)
2. Read the matching reference file for each subagent from the reference tool

Then launch all 5 subagents simultaneously:

---

### Subagent 1: Standalone Environment

**What it produces:** `standalone/inference.py` (or `run.py`), `standalone/setup.sh`, `standalone/requirements.txt`, `standalone/python_version.txt`, optionally `standalone/env_vars.txt`

**Prompt template:**
```
You are implementing the standalone execution environment for a bioinformatics tool.

## Your Task
Create the standalone files for the {tool_display_name} tool in:
  proto_tools/tools/{category}/{toolkit}/standalone/

## Contract (the tool file this standalone serves)
<paste full tool file content here>

## Reference Implementation
Here is the standalone from {reference_tool} to use as a structural template:
<paste reference inference.py content>
<paste reference setup.sh content>
<paste reference requirements.txt content>

## Research: Source Repo API
<paste key findings about how the research code works — model loading, inference API, etc.>

## Files to Create

### 1. inference.py (for GPU/AI tools) or run.py (for CPU tools)
Structure:
- Module docstring
- Imports (json, sys, os at top; heavy imports inside functions)
- **Module-level logger** (REQUIRED — see "Logger Convention" below):
  ```python
  from standalone_helpers import get_logger

  logger = get_logger(__name__)
  ```
- Model class with lazy loading pattern:
  - `__init__`: set `self._loaded = False`
  - `load(device)`: import heavy deps, load model weights
  - Core method(s) matching the operations from the contract
  - `serialize_output()` from `standalone_helpers` for tensor → list conversion
- `dispatch(input_dict)` function as entry point:
  - Global model instance (lazy-initialized)
  - Routes by `input_dict["operation"]` to model methods
  - Returns JSON-serializable dict
- `if __name__ == "__main__":` block reading/writing JSON files

#### Logger Convention (enforced by `tests/style_consistency_tests/test_standalone_logger_consistency.py`)

**Every `.py` file under `proto_tools/tools/*/*/standalone/`** — `inference.py`, `run.py`, `binary_config.py`, `model.py`, helper modules, etc. — must declare a module-level logger:

```python
from standalone_helpers import get_logger

logger = get_logger(__name__)
```

Why: standalones run inside isolated micromamba subprocesses where `proto_tools` is NOT importable. `get_logger` is shipped via `standalone_helpers/proto_logging.py` (auto-copied at worker startup). Records emitted through it are bridged back to the parent process for re-emission under `proto_tools.worker.{toolkit}.*`. Plain `logging.getLogger(__name__)` produces a logger outside the bridge namespace and its records are silently dropped.

`__init__.py` is the only exemption (no log calls there). The consistency test enforces this on every other `.py` file in the standalone tree.

For status updates that should drive the parent's spinner subtitle (and never clutter console output), use the dedicated method:

```python
logger.update_status("Loading checkpoint")  # spinner subtitle update; not shown in console
logger.info("...")                           # normal log line
```

**If the tool has file-format conversion helpers** (e.g., writing MSA to Parquet, converting complexes to YAML/FASTA), implement them as plain-type functions in `helpers.py` at the tool directory level. These functions must be self-contained (no `proto_tools` imports) and take only plain types (str, list, dict). The tool layer imports and wraps them with typed signatures. The deployment service mounts them via `add_local_file()`. This ensures a single source of truth across tool and service layers. See esmfold, chai1, boltz2 for examples.

CRITICAL RULES:
- Every `.py` file in `standalone/` (except `__init__.py`) MUST declare `logger = get_logger(__name__)` from `standalone_helpers` — see "Logger Convention" above. Plain `logging.getLogger(__name__)` is forbidden and enforced by a consistency test.
- Heavy imports (torch, model libraries) ONLY inside methods, never at module level
- The dispatch() function is the entry point for both persistent-worker and one-shot execution
- All tensor/array outputs must be converted to Python lists via serialize_output() from standalone_helpers
- Match the operation names used in the contract's ToolInstance.dispatch() calls
- Device handling: accept device from input_dict, pass to model.load()
- Seed handling: pass `config.seed` (raw `Optional[int]`) through the dispatch dict. In inference.py, call `set_torch_seed(seed)` unconditionally — the helper's None-check gates the expensive cuDNN flags. For any downstream sampler that needs a concrete int, do `sampling_seed = seed if seed is not None else get_random_int()` using the helper from `standalone_helpers`.
- `generative=True`: set this on sampling, gradient, or design tools whose repeated unseeded calls should produce diversified outputs. This is metadata for non-cacheable tools; for `cacheable=True` tools it also skips cache/dedup until `config.seed` is set. Do not set it merely because the tool accepts a seed.
- Audit hardcoded values in the reference/research code (chain IDs, model paths, default parameters, assumed dimensions). If a value could vary across valid inputs, parameterize it — accept it from the dispatch input dict rather than hardcoding. For example, `chain_id="A"` should come from the input, not be a constant.

### Required Device Management Protocol Functions

> **Sync note:** These patterns are duplicated in PATTERNS.md (templates + reference section) because subagents can't see PATTERNS.md. If the protocol changes, update both files.

All standalone scripts MUST implement two module-level protocol functions for DeviceManager integration. Add these AFTER the dispatch() function (or after operation functions for CPU tools), BEFORE `if __name__`.

#### 1. `to_device(device: str) -> dict`
Enables DeviceManager to move models between GPUs and CPU.

**Persistent PyTorch tools** (global `_model` kept loaded):
```python
def to_device(device: str) -> dict:
    global _model
    if _model is not None and _model._loaded:
        _model.to_device(device)
        return {"success": True, "device": device}
    else:
        return {"success": True, "device": device, "note": "model not loaded yet"}
```

**Non-persistent / CLI tools** (no persistent model state):
```python
def to_device(device: str) -> dict:
    return {"success": True, "device": device, "note": "CLI tool, auto-unloads"}
```

#### 2. `get_memory_stats() -> dict`
Enables DeviceManager to query GPU memory usage. Import the helper from `standalone_helpers` (auto-copied by worker bootstrap).

**PyTorch tools:**
```python
def get_memory_stats() -> dict:
    from standalone_helpers import get_pytorch_memory_stats
    global _model
    device = _model.device if _model and hasattr(_model, "device") else 0
    return get_pytorch_memory_stats(device)
```

**JAX tools:**
```python
def get_memory_stats() -> dict:
    from standalone_helpers import get_jax_memory_stats
    return get_jax_memory_stats(device_index=0)
```

**CPU-only tools:**
```python
def get_memory_stats() -> dict:
    return {"available": False, "framework": "cpu", "note": "CPU tool"}
```

**GPU CLI tools** (e.g., Boltz2, RFDiffusion3 — spawn subprocesses that use GPU):
```python
def get_memory_stats() -> dict:
    from standalone_helpers import get_pytorch_memory_stats
    return get_pytorch_memory_stats(device=0)
```

**Key rules:**
- Both functions MUST be at module level (not inside classes)
- `to_device()` returns `{"success": bool, "device": str}`
- `get_memory_stats()` returns `{"available": bool, "framework": str, ...}`
- Import `standalone_helpers` (auto-copied by worker bootstrap) — do NOT import from `proto_tools`

### 2. setup.sh

All setup.sh scripts source `standalone_helpers.sh` (auto-copied alongside `standalone_helpers.py`) for shared infrastructure functions. Reference: `tools/inverse_folding/fampnn/standalone/setup.sh`.

```bash
#!/bin/bash
set -euo pipefail
source standalone_helpers.sh

pip install uv

# PyTorch tools (add extras like torchvision if needed):
proto_install_pytorch
# proto_install_pytorch "" torchvision  # torch + version-matched torchvision

# JAX tools (with optional tool-specific override prefix):
# proto_install_jax MYTOOL

# If CUDA toolkit needed (for JIT compilation):
# proto_install_cuda_toolkit

uv pip install -r requirements.txt

# Non-HF tools that download weights:
proto_resolve_weights_dir {toolkit}
if [ ! -f "$WEIGHTS_DIR/model.pt" ]; then
    wget -q -O "$WEIGHTS_DIR/model.pt" "https://example.com/model.pt"
fi

# Gated HF models (validates access before pip install):
# proto_check_gated_hf_repo "org/model-name" "https://huggingface.co/org/model-name"
```

**Available functions** (see `utils/standalone_helpers_source/standalone_helpers.sh`):
- `proto_install_pytorch [spec] [extras...]` — install PyTorch via RECOMMENDED_TORCH_SPEC, with optional version-matched extras (e.g., `torchvision`)
- `proto_install_jax [TOOL_PREFIX]` — install JAX with tool-specific overrides
- `proto_install_cuda_toolkit [constraint] [extras...]` — micromamba CUDA toolkit
- `proto_resolve_weights_dir <toolkit>` — set `$WEIGHTS_DIR` via PROTO_MODEL_CACHE
- `proto_resolve_asset_availability <toolkit> <pattern> [license_url] [asset_kind]` — fail-fast precheck for gated / manually-provisioned weights / databases. Sentinel exit (64) on default-cache-empty signals a *test skip*; misconfigured `PROTO_<TOOLKIT>_WEIGHTS_DIR` is an exit-1 *failure*. Pair with the `MissingAssetError` plumbing in `tool_instance.py`.
- `proto_check_gated_hf_repo <repo_id> <license_url> [probe_file]` — validate HF access

**For the standalone inference.py**, non-HF tools must call `resolve_weights_dir()` to find weights at runtime:
```python
from standalone_helpers import resolve_weights_dir
weights_dir = resolve_weights_dir("{toolkit}")
model_path = os.path.join(weights_dir, "model.pt")
```

### 3. requirements.txt
List all Python dependencies with version pins. Do NOT include torch/jax — those are installed by setup.sh using hardware-aware detection.

### 4. python_version.txt
**Required.** Every tool must pin its Python version — `ToolInstance` raises `FileNotFoundError` at setup time for a tool without this file, and the style consistency tests fail if any tool is missing it. Always use the keyed format with a required `default` key. Comments (`#` to end of line) and blank lines are allowed.

```text
default: 3.12
```

**Per-platform overrides** (rarely needed): use only when an upstream dependency is unavailable for the default Python on a specific platform. Three-tier lookup, most specific wins:

1. `{system}-{machine}` (e.g. `linux-aarch64`)
2. `{system}` (e.g. `linux`)
3. `default` (required catch-all)

```text
default: 3.11
linux-aarch64: 3.10    # only when forced by dep availability — see PyRosetta
```

Pick `default` to match the reference tool unless the upstream library has explicit Python-version constraints. The full spec lives in `notes/tool-environments.md`. **Validation:** every `python_version.txt` is checked by `tests/style_consistency_tests/test_python_version_consistency.py`, which enforces the format and the required `default` key on every shipped file.

### 5. env_vars.txt (only if needed)
[passthrough]
HF_TOKEN
[set]
# Only if the tool needs LD_LIBRARY_PATH or other env vars
```

**10C: Run all tests for the new tool**

Run all tests (functional + infra) filtered to the new tool:

```bash
pytest --all --ext -k "{tool_key}" -v
```

This runs the tool's functional tests AND the parametrized infra tests (`example_input`, device consistency, registry integration). A detailed log file is generated in `logs/` (project root).

---

### Subagent 2: README + cite.bib

**What it produces:** `README.md`, `cite.bib`

**Prompt template:**

```
You are writing the documentation for a bioinformatics tool.

## Your Task
Create README.md and cite.bib for the {tool_display_name} tool in:
  proto_tools/tools/{category}/{toolkit}/

## Contract (the tool's API)
<paste full tool file content here>

## Reference README
Here is the README from {reference_tool} to use as a structural template:
<paste reference README.md content>

## Research Context
<paste tool description, paper info, biological context from Phase 1>

## README.md Structure
Follow this exact structure:
1. # {Tool Display Name}
2. ## Overview — biological context, what the tool does, why it's useful
3. ## When to Use This Tool — primary use cases + when NOT to use
4. ## Biological Background — scientific foundation for non-biologists
5. ## Tool Catalog — table of operations (key, input, output, use case)
6. ## Execution Modes — GPU/CPU requirements, memory, timing
7. ## How It Works — brief description of each operation
8. ## Input Parameters — tables for each operation
9. ## Configuration — parameter tables with types, defaults, descriptions
10. ## Output Specification — output field tables
11. ## Interpreting Results — how to interpret scores/metrics biologically
12. ## Quick Start Examples — 3-5 working code examples with exact imports
13. ## Best Practices & Gotchas — parameter tuning, common mistakes
14. ## References — paper citation, GitHub links
15. ## Related Tools — tools often used together, alternatives

CRITICAL: Use exact import paths from proto_tools.tools.{category}.{toolkit}. Class names and function names must match the contract exactly.

## cite.bib
Look up the paper's BibTeX citation using the DOI. Format:
@article{firstauthor_year_toolname,
  title={...},
  author={...},
  journal={...},
  year={...},
  doi={...}
}
```

---

### Subagent 3: Example Notebook

**What it produces:** `examples/example.ipynb`

**Prompt template:**

```
You are creating an example Jupyter notebook for a bioinformatics tool.

## Your Task
Create examples/example.ipynb for the {tool_display_name} tool in:
  proto_tools/tools/{category}/{toolkit}/examples/

## Contract (the tool's API)
<paste full tool file content here>

## Instructions
Create a Jupyter notebook (.ipynb JSON) with these cells:

1. **Markdown title cell** — Tool name, brief description, paper link
2. **Code: Imports** — Exact imports from proto_tools.tools.{category}.{toolkit}
3. **Markdown: Input API Reference** — Table with Field, Type, Default, Description
4. **Markdown: Config API Reference** — Same table format
5. **Markdown: Output API Reference** — Same table format
6. **Code: Basic Usage** — Minimal working example with realistic biological data
7. **Code: Advanced Usage** — Example with non-default config parameters
8. **Code: Export** — Demonstrate result.export()

Notebook metadata:
- kernel: proto-language, Python 3.12
- Use realistic biological data (real protein sequences, not placeholder text)
- Include comments explaining each step
- Show output inspection (printing key fields)

Write the notebook as valid .ipynb JSON (the raw JSON format with cells array, metadata, etc.).

Note: the notebook will be executed during Phase 4 via `scripts/run_example_notebooks.py`, which strips widget/Plotly/Bokeh mime types that don't render in static viewers.
```

---

### Subagent 4: Tests

**What it produces:** `tests/{category}_tests/test_{tool_key_snake}.py`

**Prompt template:**

```
You are writing tests for a bioinformatics tool.

## Your Task
Create tests/{category}_tests/test_{tool_key_snake}.py

## Contract (the tool's API)
<paste full tool file content here>

## Reference Tests
Here are the tests from {reference_tool} as a structural template:
<paste reference test file content>

## Test Structure

```python
"""Tests for {tool_display_name} tool."""
from pathlib import Path

import pytest

from proto_tools.entities.structures.structure import Structure  # if needed
from proto_tools.tools import run_{tool_key_snake}, {ToolName}Input, {ToolName}Config

TEST_PDB_FILE = Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb"  # if needed

# ── Validation ────────────────────────────────────────────────────────────────

def test_{tool_key_snake}_input_normalizes_single_item():
    """Test that single inputs are normalized to lists (custom validator)."""

# ── Integration ───────────────────────────────────────────────────────────────

@pytest.mark.uses_gpu  # or @pytest.mark.integration for CPU tools
def test_{tool_key_snake}_basic_execution():
    """Test basic tool execution with default config."""
    # Create input with realistic data
    # Run tool
    # Assert result.success
    # Assert result.tool_id == "{tool_key}"
    # Assert specific output fields are populated

@pytest.mark.uses_gpu
def test_{tool_key_snake}_export(tmp_path):
    """Test output export to supported formats."""
```

CRITICAL RULES:
- **Flat functions only** — no `class Test*`, use descriptive function names
- **No trivial tests** — do NOT test Pydantic built-in behavior. Tests that verify required fields exist, `extra="forbid"` rejects unknown fields, or config stores default values are worthless — Pydantic already guarantees these. Only test custom validators, computed properties, normalization logic, and real tool behavior.
- Do NOT write tests like: `test_*_rejects_missing_*`, `test_*_rejects_extra_fields`, `test_*_config_defaults`, `test_*_rejects_invalid_*` (for simple `ge=`/`le=` constraints). These just re-test Pydantic.
- Import from `proto_tools.tools` (top-level), not from deep paths
- Use `@pytest.mark.uses_gpu` for any test that runs the actual tool on GPU
- Use `@pytest.mark.integration` for CPU dispatch tests
- Use realistic biological data (real sequences/structures from dummy_data/)
- Check `result.success` and `result.tool_id`
- Test both default config and non-default parameters
```

---

### Subagent 5: Export Chain (__init__.py files)

**What it produces:** Updated `__init__.py` at 3 levels (tool, category, master tools)

**Prompt template:**

```
You are updating the Python export chain for a new bioinformatics tool.

## Your Task
Update __init__.py files at 3 levels to export the new tool's classes and functions.

## Contract (the tool's API — extract class names and function names from this)
<paste full tool file content here>

## Current Category __init__.py
<paste current category __init__.py content>

## Current Master tools/__init__.py
<paste current tools/__init__.py content>

## Changes Required

### Level 1: Create tool __init__.py
File: proto_tools/tools/{category}/{toolkit}/__init__.py

```python
from proto_tools.tools.{category}.{toolkit}.{tool_key_snake} import (
    {ToolName}Config,
    {ToolName}Input,
    {ToolName}Output,
    run_{tool_key_snake},
)

__all__ = [
    "{ToolName}Config",
    "{ToolName}Input",
    "{ToolName}Output",
    "run_{tool_key_snake}",
]
```

### Level 2: Update category __init__.py
File: proto_tools/tools/{category}/__init__.py
Add import block and __all__ entries for the new tool. Preserve all existing imports.

### Level 3: Update master tools/__init__.py
File: proto_tools/tools/__init__.py
Add import block and __all__ entries. Preserve all existing imports.

CRITICAL RULES:
- Sort `__all__` alphabetically
- Do NOT remove or modify any existing imports
- Match class names and function names EXACTLY from the contract
- If the tool has multiple operations (e.g., sample + score), export ALL of them
- Level 4 (proto_tools/__init__.py) uses `from proto_tools.tools import *` — no changes needed
- Use the Edit tool to modify existing files, Write for new files
```

---

## Phase 4: Verify

**Goal:** Confirm all subagent outputs integrate correctly.

**Steps:**

1. **Check import chain** — Verify the tool imports correctly:
   ```bash
   python3 -c "from proto_tools.tools import run_{tool_key_snake}, {ToolName}Input, {ToolName}Config; print('Import OK')"
   ```

2. **Run tests** — Execute the test file:
   ```bash
   python3 -m pytest tests/{category}_tests/test_{tool_key_snake}.py -v
   ```
   (Plain `pytest` runs everything the host can handle. Add `--cpu-only` to force-skip GPU tests, or `--gpu-only` to filter to GPU-marked tests.)

3. **Validate tool registration** — Check the tool appears in the registry:
   ```bash
   python3 -c "from proto_tools.tools.tool_registry import ToolRegistry; specs = [s for s in ToolRegistry.list_all() if '{tool_key}' in s.key]; print([(s.key, s.label) for s in specs])"
   ```

4. **Execute the example notebook** — Always route notebook execution through the sanitizer script, never `jupyter nbconvert --execute` directly. The script strips widget / Plotly / Bokeh mime types that a kernel emits but static viewers (VS Code, JupyterLab, GitHub, nbviewer) can't render, leaving only the `text/plain` / `text/html` fallbacks that survive the round-trip:
   ```bash
   python3 scripts/run_example_notebooks.py --only {toolkit}
   ```
   Use `--sanitize-only` to clean a pre-executed notebook without re-running (fast; for notebooks that can't re-run on the current host).

5. **Fix any failures** — If imports fail, check the export chain. If tests fail, fix the tool file or test file directly.

6. **Run the validation checklist:**
   - [ ] Uses `logging.getLogger(__name__)`, never `print()`
   - [ ] Input extends `BaseToolInput`, uses `Field()` (not ConfigField)
   - [ ] Config extends `BaseConfig`, uses `ConfigField()` (not bare Field)
   - [ ] Output extends `BaseToolOutput`, implements `output_format_options`, `output_format_default`, `_export_output()`
   - [ ] Run function signature: `def run_*(inputs: *Input, config: *Config) -> *Output`
   - [ ] Run function returns Output with `metadata={}` dict of key parameters
   - [ ] `@tool()` has all 9 kwargs: key, label, category, input_class, config_class, output_class, description, uses_gpu, example_input
   - [ ] `@tool()`: `uses_gpu=True` matches Config `device="cuda"` override
   - [ ] `@tool()`: Optional `device_count` specifies expected device allocation ("1", "1-2", ">=1", etc.)
   - [ ] `@tool()`: `generative=True` is set for diversified unseeded sampling/gradient/design outputs; for `cacheable=True` tools, seeded calls remain cacheable
   - [ ] No try/except wrapping tool logic
   - [ ] Google-style docstrings with Attributes (for classes) and Args/Returns/Examples (for functions)
   - [ ] `__init__.py` exports at all 4 levels
   - [ ] README.md exists with correct import paths
   - [ ] cite.bib exists with valid BibTeX
   - [ ] Example notebook exists
   - [ ] Tests written and importable
   - [ ] Standalone script implements `to_device(device: str) -> dict` at module level
   - [ ] Standalone script implements `get_memory_stats() -> dict` at module level
   - [ ] Biological coordinates are 1-indexed, inclusive (if applicable)
   - [ ] **Weight management**: If the tool downloads model weights:
     - [ ] HF tools: no weight code needed (HF_HOME set automatically)
     - [ ] Non-HF tools: `setup.sh` implements `PROTO_MODEL_CACHE` pattern (see `fampnn/standalone/setup.sh`)
     - [ ] Non-HF tools: `inference.py` calls `resolve_weights_dir()` from `standalone_helpers` (see `fampnn/standalone/inference.py`)

---

## Phase 4.5: Self-Audit

**Goal:** Catch convention violations, dead fields, hardcoded assumptions, and consistency gaps before shipping.

**When:** After Phase 4 passes (imports work, tests pass, registry OK). Before Phase 5 (Ship).

Launch a single audit agent via `Task()`:

```
You are auditing a newly implemented bioinformatics tool for quality issues.

## Your Task
Read all generated files for the {tool_display_name} tool and check for issues.

## Files to Read
- proto_tools/tools/{category}/{toolkit}/{tool_key_snake}.py (core tool file)
- proto_tools/tools/{category}/{toolkit}/standalone/inference.py (or run.py)
- proto_tools/tools/{category}/{toolkit}/standalone/setup.sh
- proto_tools/tools/{category}/{toolkit}/standalone/requirements.txt
- proto_tools/tools/{category}/{toolkit}/standalone/python_version.txt
- proto_tools/tools/{category}/{toolkit}/README.md
- proto_tools/tools/{category}/{toolkit}/examples/example.ipynb
- tests/{category}_tests/test_{tool_key_snake}.py

## Reference Tool (for comparison)
Also read the same files from {reference_tool} to compare patterns.

## Checks
1. **Module-level imports** — Heavy dependencies (torch, model libraries) must only be imported inside functions in standalone scripts, never at module level
2. **Inherited fields** — Every field inherited from base classes (shared_data_models.py) must be either implemented in the standalone code or raise ValueError when non-default values are provided
3. **Hardcoded values** — Flag any values in standalone code that are hardcoded but should vary with input (chain IDs, model paths, assumed dimensions, fixed sequence lengths)
4. **Contract-standalone consistency** — Every Config field with `reload_on_change=True` must be read and used in the standalone code. Every operation dispatched in the tool file must be handled in standalone dispatch()
5. **Notebook accuracy** — Import paths, class names, and field names in the example notebook must exactly match the contract
6. **Seed propagation** — If the tool uses PyTorch/JAX RNG, Config inherits `seed: int | None` from `BaseConfig`. Pass `config.seed` (raw `Optional[int]`) through the dispatch dict as `"seed": config.seed`. In `standalone/inference.py`, call `set_torch_seed(seed)` unconditionally — the helper no-ops when `seed is None`, so the expensive cuDNN determinism flags are only set on the explicit-seed path. If the tool's internal sampling code needs a concrete int (e.g., a model sampler that doesn't accept `None`), use `seed if seed is not None else get_random_int()` to fall back to a fresh random int from `standalone_helpers.get_random_int`
7. **Test coverage** — Every operation (sample, score, etc.) should have at least one test

## Output Format
Print a JSON array of issues:
[
  {"severity": "fix", "file": "path", "line": "~N", "issue": "description"},
  {"severity": "cosmetic", "file": "path", "line": "~N", "issue": "description"}
]

"fix" = must fix before shipping. "cosmetic" = nice to fix, won't cause bugs.
If no issues found, print: []
```

**After the audit agent returns:**
1. Read the agent's output
2. Fix all `"fix"` severity issues directly
3. Fix `"cosmetic"` issues if quick (< 1 minute each)
4. Re-run Phase 4 import/test checks if any fixes touched the tool file or standalone code
5. **Log findings to auto-memory** — If any `"fix"` issues were found, save them to persistent memory so we can track how the pipeline fails over time. Write (or append) to the auto-memory directory at `implement-tool-audits.md`:
   ```
   ## {toolkit} audit — {date}
   - {N} fix issues, {M} cosmetic
   - Fixes: {brief list of fix issues}
   - These indicate systemic gaps in the pipeline.
   ```
   This builds a dataset of pipeline failure modes that informs future skill improvements.

---

## Phase 5: Ship

**Goal:** Commit, push, create PR.

**Steps:**

1. **Stage all new files:**
   ```bash
   git add proto_tools/tools/{category}/{toolkit}/
   git add tests/{category}_tests/test_{tool_key_snake}.py
   # Also stage modified __init__.py files
   git add proto_tools/tools/{category}/__init__.py
   git add proto_tools/tools/__init__.py
   ```

2. **Commit:**
   ```bash
   git commit -m "feat: implement {tool_display_name} tool wrapper

   Implements {tool_display_name} ({operations}) in the {category} category.
   Closes #{issue_number}"
   ```

3. **Push and create PR:**
   ```bash
   git push -u origin HEAD
   gh pr create --title "feat: implement {tool_display_name} tool" --body "$(cat <<'EOF'
   ## Summary
   - Implements {tool_display_name} tool wrapper ({operations})
   - Category: {category}
   - Follows universal tool pattern with standalone execution via ToolInstance

   Closes #{issue_number}

   ## What's Included
   - Core tool file with Input/Config/Output + @tool decorator
   - Standalone environment (inference.py, setup.sh, requirements.txt)
   - README with biological context and usage examples
   - cite.bib with paper citation
   - Example Jupyter notebook
   - Test suite
   - Full export chain (__init__.py at all levels)

   ## Test Plan
   - [ ] `python3 -c "from proto_tools.tools import run_{tool_key_snake}"` imports successfully
   - [ ] `pytest tests/{category}_tests/test_{tool_key_snake}.py` passes
   - [ ] Tool appears in ToolRegistry.list_all()
   - [ ] GPU execution verified (if applicable)
   EOF
   )"
   ```

4. **Report the PR URL to the user.**
