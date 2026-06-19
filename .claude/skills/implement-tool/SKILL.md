---
name: implement-tool
description: >
  Implements a new bioinformatics tool wrapper in proto-tools using a
  parallelized agent pipeline. Orchestrates phases: Research, Contract (core
  tool file), Fan-out (5 parallel subagents), Verify, then the decisive gates —
  Config Field Audit, Temp Integration & Stress, Docs Verification, Self-Audit &
  Full PR Review — and Ship. Use when creating tools from GitHub issues,
  wrapping models, or implementing new tool wrappers end-to-end.
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
Phase 1: Research → Phase 2: Contract → Phase 3: Fan-out (5 parallel agents) → Phase 4: Verify → Phase 4.5: Config Field Audit → Phase 4.6: Temp Integration & Stress → Phase 4.7: Docs Verification → Phase 4.8: Self-Audit & Full PR Review → Phase 5: Ship
```

**The Phase 4.x gates are decisive and non-negotiable** — they catch the failure modes that the fast fan-out reliably misses (invented config parameters, untested real workloads, hallucinated citations/licenses, missing `license.yaml`/`links.yaml`). Do not skip them to ship faster. Ground every decision in upstream source or a credible reference — never assume.

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
- A **GitHub issue URL/number** (e.g., `#<issue-number>` or `https://github.com/evo-design/proto-tools/issues/<issue-number>`)
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
   - **License** — read the repo's actual `LICENSE`/`COPYING` file (not just the README badge) and the weights license if it differs. Record the SPDX id (or note it's non-standard) and the file URL. Feeds `license.yaml` and the README License callout.
   - **Canonical links** — the repo URL, project homepage/docs, paper DOI or preprint URL, HuggingFace model page, affiliated orgs. Feeds `links.yaml`. Verify each resolves.
   - **The full upstream parameter surface** — which parameters the upstream CLI/Python API actually exposes to *end users*, with their defaults and valid ranges. This is the ground truth for Phase 4.5's config audit; capture it now while reading the source.
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
   |   +-- license.yaml        # License metadata (required — enforced by test_license_consistency.py)
   |   +-- links.yaml          # Canonical upstream links (required — github/website/paper/huggingface/...)
   |   +-- README.md
   |   +-- examples/
   |   |   +-- example.ipynb   # Working example notebook (required)
   |   +-- helpers.py            # [optional] Self-contained plain-type file-format conversion helpers
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
   - Config class extending `BaseConfig` (or shared base) with `ConfigField()` — `extra="forbid"`. Use `reload_on_change=True` on fields that require worker restart (model checkpoint, etc.). Use `include_in_key=False` on fields that don't affect computation results (device, verbose, timeout are already excluded on `BaseConfig`; tool-level overrides of `device` must also set `include_in_key=False`). `include_in_key` defaults to `True`. Use `xor_group="<slug>"` to mark mutually exclusive sibling fields (see "Mutual-exclusion fields (XOR groups)" below).
   - Output class extending `BaseToolOutput` (or shared base) with `Field()` — `extra="ignore"` (set by `BaseToolOutput`; the base also emits a warning for unexpected fields so computed-field JSON round-trips can validate cleanly)
   - `@tool()` decorator with the 7 required kwargs (key, label, category, input_class, config_class, output_class, description) plus conventionally set `uses_gpu`. Optional: `example_input`, `device_count`, `cacheable`, `stochastic`, `iterable_input_fields` (a list — `["sequences"]` for an ordinary iterable, or an index-parallel group like `["complexes", "msas"]`), `iterable_output_field`, `metrics_class`, `gpu_only`, `post_process_iterable`
   - `run_*()` function that calls `ToolInstance.dispatch()`
   - If category has shared data models, use type aliases (e.g., `ToolInput = InverseFoldingInput`)
   - **Metrics**: if the tool emits scalar metric-like values (plDDT, perplexity, scores, etc.), route them through a `Metrics` subclass (from `proto_tools/utils/tool_io.py`). Prefer a shared per-category class (`MaskedModelScoringMetrics`, `InverseFoldingScoringMetrics`, `CausalModelScoringMetrics`, etc.) when the metric set matches siblings. Otherwise declare a per-tool `<Tool>Metrics(Metrics)` subclass in the tool file with a `metric_spec: ClassVar[dict[str, MetricSpec]]` documenting type/range and a `better_values_are` direction (`"higher"`, `"lower"`, `"in-range"`, or `"context-dependent"`) for each metric — `test_metric_spec_consistency.py` requires a valid `better_values_are` on every metric. Verification lives in each tool's e2e test via `assert_metrics_in_spec(result)` (helper at `tests/tool_infra_tests/_metric_helpers.py`).

**Critical conventions:**
- Tool registry key (`{tool_key}`): `{toolkit}-{suffix}` kebab-case (e.g., `"esmif-sample"`)
- Run function: `run_{tool_key_snake}` (e.g., `run_esmif_sample`)
- Classes: `{ToolName}Input/Config/Output`, PascalCase (e.g., `ESMIFSampleInput`)
- `batch_size` defaults to `1` for GPU tools
- No try/except — `@tool` decorator handles errors
- Use `logging.getLogger(__name__)`, never `print()`
- Output must implement `output_format_options`, `output_format_default`, `_export_output()`
- **Iterable input/output cardinality must be 1:1**: when `iterable_input_fields` and `iterable_output_field` are both set, `len(output.{iterable_output_field})` must equal the (primary) input list length, by position. The framework's per-item cache stitching, dedup, and diversification testing all rely on this. If the tool produces N samples per input (e.g. a `num_sequences_per_structure` or `num_designs` config parameter), bundle the N samples inside a single per-input wrapper class — do NOT flatten N×inputs into the output list. Canonical patterns to follow: `ProteinMPNNSequences` / `LigandMPNNSequences` (per-structure sequence bundles, `num_sequences_per_structure` config) and `RFdiffusion3Designs` (per-spec design bundles, `n_batches * diffusion_batch_size` per spec). See PATTERNS.md → "Iterable cardinality" for the full rule and code shape.
- **Input = per-item, Config = shared**: model each field by "is there one of these per item, or one for the whole batch?" Shared whole-batch params/collections → Config (which also keeps the per-item cache key correct by construction). Per-item values → Input, either bundled inside the iterable element or as a parallel sibling in `iterable_input_fields`. A per-item value that is *usually uniform* can be a **broadcastable** field — `InputField(..., broadcastable=True)`, declared as `list[...] | None`, added to `iterable_input_fields`, normalized via `broadcast_parallel_fields()` (one value broadcasts to all, N is strict 1:1). Broadcast is opt-in per field and only when a single value applied to every item is genuinely correct (safe for `binder_chain`; **wrong for `msas`**). See PATTERNS.md → "Input = per-item, Config = shared".
- **`cpus_per_instance` opt-in**: `BaseConfig.cpus_per_instance` defaults to `None` — every CPU tool stays off ToolPool's CPU scheduler and runs as a single direct call. **Most CPU tools should leave this alone.** Only opt in (override to a positive int) when per-call work is heavy enough to amortize spinning up N persistent worker subprocesses — each holds its own venv in RAM and pays a startup tax, so cheap tools (short per-item compute, internal threading, network IO) lose more than they gain. The canonical opt-in is PyRosetta (heavy `init`, multi-second per pose, embarrassingly parallel poses → `cpus_per_instance = 1`). GPU tools (`gpus_per_instance > 0`) ignore `cpus_per_instance` entirely.

**Inherited field audit:** When reusing a shared base config (e.g., `InverseFoldingConfig`) or base input, enumerate every inherited field and verify the target model can implement it. For each unsupported field, either implement support (e.g., logit masking for `excluded_amino_acids`) or override the field with a validator that raises `ValueError("'{field_name}' is not supported by {tool_display_name}")` when a non-default value is provided. Do not silently inherit fields that the model ignores.

**Cloud fail-fast for local-resource configs:** If a config setting needs a **local resource that can't be staged to the hosted service** — a local database/index, an on-disk weights/artifact directory, or a local file path — override `BaseConfig.cloud_unsupported_reason(self) -> str | None` so `device='cloud'` fails fast at dispatch with a clear message instead of a late runtime error (a missing-file crash, or silently ignoring the override). Return a user-facing reason when the offending setting is active, `None` otherwise (the default means cloud-compatible). Guard on the setting actually being non-default so ordinary cloud runs are untouched:

```python
def cloud_unsupported_reason(self) -> str | None:
    """A local weights directory (``local_path``) isn't present on a hosted worker."""
    if self.local_path:  # optional override; unset by default → cloud-OK
        return "local_path points to a local weights directory not available on device='cloud'. Unset it, or run locally with device='cpu'."
    return None
```

A tool whose *only* mode needs a local file (e.g. a required local DB/HMM input, like `blast-search` local mode or `pyhmmer-hmmscan`) returns the reason unconditionally. Scope this to **intrinsically** local configs — do **not** encode deployment-specific hosting knowledge here (e.g. which model checkpoints are warmed on the cloud runtime); that stays private to the tools backend. The router invokes this only on the `device='cloud'` path, after the `local_cpu` no-op and the license gate.

## Mutual-exclusion fields (XOR groups)

For "pick one" siblings: make each `Optional` with `default=None`, tag with the same `xor_group="<slug>"`, and add a `@model_validator(mode="after")` to enforce at runtime.

```python
mmseqs_db: str | None = InputField(default=None, xor_group="target",
    description="Target DB (path/slug/AssetRef).")
target_sequences: list[str] | None = InputField(default=None, xor_group="target",
    description="Inline target protein sequences.")

@model_validator(mode="after")
def exactly_one_target(self) -> "Mmseqs2SearchProteinsInput":
    if (self.mmseqs_db is None) == (self.target_sequences is None):
        raise ValueError("provide exactly one of `mmseqs_db` or `target_sequences`")
    return self
```

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

**If the tool has file-format conversion helpers** (e.g., writing MSA to Parquet, converting complexes to YAML/FASTA), implement them as plain-type functions in `helpers.py` at the tool directory level. These functions must be self-contained (no `proto_tools` imports) and take only plain types (str, list, dict). The tool layer imports and wraps them with typed signatures. Keeping them dependency-free lets the hosted execution backend reuse the same files, preserving a single source of truth. See esmfold, chai1, boltz2 for examples.

CRITICAL RULES:
- Every `.py` file in `standalone/` (except `__init__.py`) MUST declare `logger = get_logger(__name__)` from `standalone_helpers` — see "Logger Convention" above. Plain `logging.getLogger(__name__)` is forbidden and enforced by a consistency test.
- Heavy imports (torch, model libraries) ONLY inside methods, never at module level
- The dispatch() function is the entry point for both persistent-worker and one-shot execution
- All tensor/array outputs must be converted to Python lists via serialize_output() from standalone_helpers
- Match the operation names used in the contract's ToolInstance.dispatch() calls
- Device handling: accept device from input_dict, pass to model.load()
- Seed handling: pass `config.seed` (raw `Optional[int]`) through the dispatch dict. In inference.py, call `set_torch_seed(seed)` unconditionally — the helper's None-check gates the expensive cuDNN flags. For any downstream sampler that needs a concrete int, do `sampling_seed = seed if seed is not None else get_random_int()` using the helper from `standalone_helpers`.
- `stochastic=True`: set when outputs depend on `config.seed`. Cacheable unseeded calls skip cache; iterable dispatches skip dedup so duplicate items diverge via per-item RNG advancement. The framework does **not** unroll multi-item dispatches — per-item seed handling is the tool's responsibility. Do not set for tools that accept but ignore the seed.
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

### Subagent 2: README + cite.bib + license.yaml + links.yaml

**What it produces:** `README.md`, `cite.bib`, `license.yaml`, `links.yaml`

`license.yaml` and `links.yaml` are **required** for every toolkit — `tests/style_consistency_tests/test_license_consistency.py` enforces `license.yaml`'s existence and schema, and the registry/docs surface both files. A toolkit missing either fails CI. Every value in all four files must be **grounded in a credible source** (the upstream repo, its `LICENSE`/`COPYING`, the paper, the SPDX registry) — never guess a license, DOI, or URL. The deeper triple-check happens in Phase 4.7; this subagent's job is to get them right the first time.

**Prompt template:**

```
You are writing the documentation and metadata for a bioinformatics tool.

## Your Task
Create README.md, cite.bib, license.yaml, and links.yaml for the {tool_display_name}
tool in:
  proto_tools/tools/{category}/{toolkit}/

## Contract (the tool's API)
<paste full tool file content here>

## Reference files
Here are the same files from {reference_tool} to use as structural templates:
<paste reference README.md, cite.bib, license.yaml, links.yaml content>

## Research Context
<paste tool description, paper info, biological context, AND the verified upstream
 license + canonical links from Phase 1>

## README.md Structure
Follow the structured template in TEMPLATES.md exactly. Four canonical H2 sections in order:
1. `# {Toolkit Display Name}` (plus the badge row above the title and a `> [!NOTE] License: ...` callout)
2. `## Overview` — 2–3 sentences: who built it, what it does, why useful (link the upstream repo on first mention)
3. `## Background` — 1–3 paragraphs of biological/algorithmic context with inline paper citations; optional `### Learning Resources` subsection for user-facing explainers (blogs, talks, courses)
4. `## Tools` — one `### {Tool Display Name} (\`{tool-key}\`)` block per registered tool, each with `#### Applications` and `#### Usage Tips` (critical-parameter tips in bold)
5. `## Toolkit Notes` — Toolkit-wide guide badges row + bulleted notes that apply to every tool in the toolkit

DO NOT add other H2 sections — schemas, configs, and output specs are auto-generated from Pydantic field descriptions. Read `gene_annotation/pyhmmer/README.md` for the canonical example before drafting. The `> [!NOTE] License: ...` callout MUST agree with `license.yaml`.

## cite.bib
Look up the paper's BibTeX citation using the DOI (verify the DOI resolves — do not fabricate). Format:
@article{firstauthor_year_toolname,
  title={...},
  author={...},
  journal={...},
  year={...},
  doi={...}
}

## license.yaml
Capture the upstream license verified from the repo's LICENSE/COPYING (and any
weights/data license, which can differ from the code license). Schema and the
SPDX allowlist are in TEMPLATES.md → "license.yaml Template". SPDX-allowlisted
licenses must NOT inline text (it lives in proto_tools/tools/_licenses/{spdx}.txt);
non-SPDX terms use a "Custom (...)" spdx string with inline `text:`.

## links.yaml
Canonical upstream links, verified to resolve. Common keys: `github`, `website`,
`paper` / `preprint` (DOI or arXiv URL), `huggingface`, `organizations` (list).
See TEMPLATES.md → "links.yaml Template".
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
3. **Code: Input API Reference** — Call `display_api_reference("{tool_key}", "input", "run_{tool_key_snake}")` to auto-render the table. Never hand-write the table.
4. **Code: Config API Reference** — Same pattern with `"config"`.
5. **Code: Output API Reference** — Same pattern with `"output"`.
6. **Code: Basic Usage** — Minimal working example with realistic biological data
7. **Code: Advanced Usage** — Example with non-default config parameters
8. **Code: Export** — Demonstrate result.export()

Notebook metadata:
- kernelspec must be exactly `{"name": "python3", "display_name": "proto-tools"}`. Custom names like `proto-language` raise `NoSuchKernel` outside the original author's machine.
- `language_info.name` is `"python"`.
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
   - [ ] `@tool()` has the 7 required kwargs (key, label, category, input_class, config_class, output_class, description); optional kwargs commonly set: `uses_gpu` (default `False`), `example_input` (factory for parametrized tests)
   - [ ] `@tool()`: `uses_gpu=True` matches Config `device="cuda"` override
   - [ ] `@tool()`: Optional `device_count` specifies expected device allocation ("1", "1-2", ">=1", etc.)
   - [ ] `@tool()`: `stochastic=True` is set when outputs depend on `config.seed`; the tool advances its own RNG per item (framework does not unroll)
   - [ ] `@tool()`: `metrics_class=<Tool>Metrics` is set when the tool emits scalar metric-like values
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

## Phase 4.5: Config Field Audit

**Goal:** Guarantee every Input/Config field is real, correctly typed, correctly defaulted, and earns its place. The fast fan-out tends to invent parameters the upstream tool doesn't have, copy defaults/ranges blindly, and leave types loose (`str` where a `Literal` belongs).

**When:** After Phase 4 passes. Before Phase 4.6 — pruning/retyping a field changes the surface the temp tests exercise.

**How:** Walk **every field, one at a time** — do NOT batch-approve. Check each against the upstream parameter surface captured in Phase 1 (re-read the upstream source if Phase 1 didn't fully capture it). Make no assumptions; ground each decision in the source.

Per-field checklist:

1. **Real upstream parameter?** Is this parameter actually exposed to *end users* upstream? If it's an internal implementation detail (a managed path, a fixed buffer, a constant the upstream API never surfaces), **delete it** — do not manufacture complexity the tool doesn't have. Keep only what a user would meaningfully set.
2. **Default matches upstream.** The default must equal the upstream default, or be a deliberate deviation noted in the description. Never invent a default.
3. **Range/constraints match upstream.** `ge`/`le`/`gt`/`lt` bounds must reflect the real valid range from the source, not a guess.
4. **Type is as tight as the domain allows.** Prefer `Literal[...]` over `str`, and `list[Literal[...]]` over `list[str]`, whenever upstream accepts a fixed enumerated set. Use bounded `int`/`float` over unbounded numerics. Use `X | None` only when `None` is genuinely meaningful.
5. **Description is concise + dual-audience.** One line, accurate, useful to both a human and an agent: what the field does and (when non-obvious) the effect of changing it. No restating the field name; no prose paragraphs.
6. **Flags set deliberately** (decide each, don't default-copy):
   - `reload_on_change=True` only for fields that require a worker restart (model checkpoint / anything that reloads the model).
   - `include_in_key=False` for fields that don't affect results (`device`, `verbose`, `timeout` are already excluded on `BaseConfig`; a tool-level `device` override must re-set it).
   - `xor_group="<slug>"` for mutually exclusive siblings, paired with the `@model_validator`.
7. **Inherited fields** — confirm the Phase 2 inherited-field audit holds: every field from a shared base is implemented or rejected with `ValueError`, never silently ignored.
8. **Local-resource field → cloud hook.** If the field points at a local file/database/index or an on-disk weights/artifact path, confirm the config overrides `cloud_unsupported_reason()` so `device='cloud'` fails fast (see "Cloud fail-fast for local-resource configs" in Phase 2). Skip for AssetRef-stageable uploads and `Structure`-typed inputs (the gateway stages those).

**Output:** edit the contract directly. If you remove/rename/retype a field, propagate to standalone `dispatch()`, README, notebook, tests, and the export chain — leave no dangling references. Re-run Phase 4 import/test checks. Do this deliberately yourself, or delegate to a focused subagent given the Phase 1 upstream param surface + the contract.

---

## Phase 4.6: Temp Integration & Stress Tests

**Goal:** Prove the tool behaves correctly under a real user's workload — realistic data, batch/large inputs, edge sizes — beyond the committed unit tests. These are **throwaway**: write, run, read, delete. The committed suite stays as Subagent 4 wrote it.

**When:** After the config surface is final (4.5), on a host that can actually run the tool (GPU if `uses_gpu=True`).

**How:**

1. **Direct-call driver script** — write a temp script (e.g. `scratch/drive_{toolkit}.py`) that imports `run_{tool_key_snake}` and calls it exactly as a user would: realistic biological inputs, default config, then a couple of non-default configs exercising the parameters you just audited. Print key output fields and `result.success`. Run it; confirm the output is scientifically sensible, not just non-erroring.
2. **Temp integration + stress tests** emulating real usage — run them:
   - A realistic end-to-end call on representative data.
   - Batch / large-input behavior (many sequences, a long sequence, multiple structures) to surface batching or memory bugs.
   - Boundary inputs (minimal, maximal-within-range) and the validation / `xor_group` paths.
   - Stochastic tools: seeded reproducibility + unseeded divergence.
3. **Concise and meaningful only** — every test asserts something real about behavior. No slop, no bloat, no re-testing Pydantic. Delete any test that doesn't earn its place.
4. **Trace the logic against upstream source** — follow the dispatch path end-to-end and confirm it matches what the upstream code does (argument order, units, 1-indexed coordinates, output shape). Don't assume — read the source. Strip overly-defensive code (the `@tool` decorator owns errors; no try/except in tool logic) and keep comments/docstrings brief.
5. **Clean up** — delete the temp script and temp tests. Fold any genuinely valuable case into the committed suite (Subagent 4's file); do not leave scratch artifacts in the PR.

---

## Phase 4.7: Docs Verification (citations, links, licenses, READMEs)

**Goal:** Triple-check every external claim in `cite.bib`, `links.yaml`, `license.yaml`, and `README.md` against credible, verifiable sources. Hallucinated DOIs, wrong licenses, dead links, and invented capabilities are common fan-out failures and are user-facing.

**How — verify, never trust:**

1. **cite.bib** — resolve the DOI; it must point to the actual paper (confirm title/authors/year/venue). No DOI → use the arXiv/bioRxiv id. Don't fabricate volume/pages.
2. **license.yaml** — open the upstream repo's real `LICENSE`/`COPYING` and confirm the SPDX id, `commercial_use`, `redistribution`, and `attribution_required`. Confirm separate weights/data licenses where they differ. Confirm `_licenses/{spdx}.txt` exists for any SPDX id, and that the README `> [!NOTE] License:` callout agrees.
3. **links.yaml** — every URL resolves (github, homepage, paper, huggingface). No placeholders, no links copied from the reference tool.
4. **README.md** — Overview/Background claims are accurate (who built it, what it does, the science); inline paper citations resolve; no invented benchmarks or capabilities.

Use WebFetch / WebSearch to verify and ground each fact in a source. Fix discrepancies in place, then re-run `test_license_consistency.py` / `test_readme_consistency.py` if touched.

---

## Phase 4.8: Self-Audit & Full PR Review

**Goal:** Catch convention violations, dead fields, hardcoded assumptions, and consistency gaps — then do a final file-by-file review of the entire PR diff against the quality established in the rest of the repo. Last gate before Ship.

**When:** After Phases 4.5–4.7. Before Phase 5 (Ship).

### Part A — Convention audit

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

### Part B — Full PR diff review (parallel, file-by-file)

Review the **actual diff**, not just the files in isolation, against the quality bar set by the rest of the repo.

1. Stage everything and capture the diff:
   ```bash
   git add -A && git diff --cached --stat
   git diff --cached -- <each changed file>   # review file-by-file
   ```
2. Launch review subagents **in parallel (single message)**, each owning a slice of the diff. Give each agent the file's diff, the equivalent file from `{reference_tool}`, and instructions to (a) flag any drift from established repo conventions, (b) re-check everything flagged earlier in this session, and (c) confirm the change reads like world-class code already in the repo. Prefer the repo's `pr-review-toolkit` agents where the diff warrants — `code-reviewer` (conventions/correctness), `silent-failure-hunter` (error handling), `type-design-analyzer` (the audited types), `comment-analyzer` (docstring/comment accuracy) — falling back to generic `Task()` review agents otherwise. Each returns the same JSON issue array as Part A.

**After the audit and review agents return:**
1. Read every agent's output and consolidate the issues
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
