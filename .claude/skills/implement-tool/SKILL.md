---
name: implement-tool
description: >
  Implements a new bioinformatics tool wrapper in bio-programming-tools using a
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

You are implementing a new bioinformatics tool in the bio-programming-tools codebase. This skill orchestrates the full lifecycle using parallel subagents for speed.

**Pipeline overview:**
```
Phase 1: Research → Phase 2: Contract → Phase 3: Fan-out (5 parallel agents) → Phase 4: Verify → Phase 4.5: Self-Audit → Phase 5: Ship
```

**Key principle:** The core tool file (Input/Config/Output + `@tool()` + `run_*()`) is the **contract** everything else depends on. Write it first (sequential), then fan out to subagents (parallel).

**CRITICAL: Standalone scripts run in isolated environments and MUST NOT import from `bio_programming_tools`:**
- `from bio_programming_tools.utils import ...` — will fail at runtime
- `from bio_programming_tools.entities import ...` — will fail at runtime
- `from standalone_helpers import get_subprocess_device_env` — auto-copied by worker bootstrap (OK)
- Standard library imports: `import json`, `import subprocess`, etc. (OK)
- Dependencies from `requirements.txt`: `import torch`, `import numpy`, etc. (OK)
- NEVER install `bio_programming_tools` in standalone environments (creates circular dependency, breaks isolation)

---

## Phase 0: Parse Input

The user provides EITHER:
- A **GitHub issue URL/number** (e.g., `#53` or `https://github.com/evo-design/bio-programming-tools/issues/53`)
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
   - Model weights location (HuggingFace, GitHub releases, etc.)

2. **Read key source files** — Find and read the research repo's inference/prediction scripts to understand:
   - How the model is loaded
   - What inputs it expects (sequences, structures, files)
   - What outputs it produces (scores, sequences, structures)
   - Device handling (CUDA/CPU)

3. **Read existing tools in the same category** — Find the reference implementation:
   ```bash
   ls bio_programming_tools/tools/{category}/
   ```
   Read the reference tool's:
   - Main tool file (the `@tool()` decorated function)
   - `standalone/inference.py` (or `run.py`)
   - `standalone/setup.sh`
   - `standalone/requirements.txt`
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

1. Create the tool directory structure:
   ```bash
   mkdir -p bio_programming_tools/tools/{category}/{tool_name}/standalone
   mkdir -p bio_programming_tools/tools/{category}/{tool_name}/examples
   ```

   Target file tree:
   ```
   tools/{category}/
   +-- shared_data_models.py   # Shared Input/Config/Output base classes (if category has 2+ tools)
   +-- {tool_name}/
   |   +-- __init__.py
   |   +-- {tool_name}.py      # Core tool file (Input/Config/Output + @tool + run_*)
   |   +-- cite.bib            # BibTeX citation (required)
   |   +-- README.md
   |   +-- examples/
   |   |   +-- example.ipynb   # Working example notebook (required)
   |   +-- standalone/
   |   |   +-- setup.sh
   |   |   +-- run.py OR inference.py  # run.py for CPU tools, inference.py for AI models
   |   |   +-- requirements.txt
   |   |   +-- binary_config.py   # [optional]
   +-- __init__.py
   ```

2. Write the core tool file `bio_programming_tools/tools/{category}/{tool_name}/{operation}.py` with:
   - Proper imports (including `from __future__ import annotations`)
   - Input class extending `BaseToolInput` (or shared base) with `Field()` — `extra="forbid"`
   - Config class extending `BaseConfig` (or shared base) with `ConfigField()` — `extra="ignore"`
   - Output class extending `BaseToolOutput` (or shared base) with `Field()` — `extra="forbid"`
   - `@tool()` decorator with all 8 kwargs: key, label, category, input, config, output, description, uses_gpu
   - `run_*()` function that calls `ToolInstance.dispatch()`
   - If category has shared data models, use type aliases (e.g., `ToolInput = InverseFoldingInput`)

**Critical conventions:**
- Tool registry key: `{tool}-{action}` kebab-case (e.g., `"esmif-sample"`)
- Run function: `run_{tool_name}_{action}` (e.g., `run_esmif_sample`)
- Classes: PascalCase (e.g., `ESMIFSampleInput`)
- `batch_size` defaults to `1` for GPU tools
- No try/except — `@tool` decorator handles errors
- Use `logging.getLogger(__name__)`, never `print()`
- Output must implement `output_format_options`, `output_format_default`, `_export_output()`

**Inherited field audit:** When reusing a shared base config (e.g., `InverseFoldingConfig`) or base input, enumerate every inherited field and verify the target model can implement it. For each unsupported field, either implement support (e.g., logit masking for `excluded_amino_acids`) or override the field with a validator that raises `ValueError("'{field_name}' is not supported by {tool_name}")` when a non-default value is provided. Do not silently inherit fields that the model ignores.

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
   from bio_programming_tools.utils.tool_instance import ToolInstance
   ```
5. **logger.debug() before main loop** — Add a status message before the processing loop:
   ```python
   logger.debug("Using local venv for {tool_name} {operation}")
   ```
6. **Don't redefine inherited fields** — Fields like `verbose`, `timeout`, and `device` are inherited from `BaseConfig`. Don't redeclare them in tool-specific Config classes unless overriding the default value.
7. **`__init__.py` files** — No `from __future__ import annotations`. Sort `__all__` alphabetically.

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

**What it produces:** `standalone/inference.py` (or `run.py`), `standalone/setup.sh`, `standalone/requirements.txt`, optionally `standalone/env_vars.txt`

**Prompt template:**
```
You are implementing the standalone execution environment for a bioinformatics tool.

## Your Task
Create the standalone files for the {tool_name} tool in:
  bio_programming_tools/tools/{category}/{tool_name}/standalone/

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
- `from __future__ import annotations`
- Imports (json, sys, os at top; heavy imports inside functions)
- Model class with lazy loading pattern:
  - `__init__`: set `self._loaded = False`
  - `load(device)`: import heavy deps, load model weights
  - Core method(s) matching the operations from the contract
  - `_serialize_output()` helper for tensor → list conversion
- `dispatch(input_dict)` function as entry point:
  - Global model instance (lazy-initialized)
  - Routes by `input_dict["operation"]` to model methods
  - Returns JSON-serializable dict
- `if __name__ == "__main__":` block reading/writing JSON files

CRITICAL RULES:
- Heavy imports (torch, model libraries) ONLY inside methods, never at module level
- The dispatch() function is the entry point for both persistent-worker and one-shot execution
- All tensor/array outputs must be converted to Python lists via _serialize_output()
- Match the operation names used in the contract's ToolInstance.dispatch() calls
- Device handling: accept device from input_dict, pass to model.load()
- Audit hardcoded values in the reference/research code (chain IDs, model paths, default parameters, assumed dimensions). If a value could vary across valid inputs, parameterize it — accept it from the dispatch input dict rather than hardcoding. For example, `chain_id="A"` should come from the input, not be a constant.

### 2. setup.sh
```bash
#!/bin/bash
set -euo pipefail
pip install uv
# For PyTorch tools: use ${RECOMMENDED_TORCH_SPEC:-torch} --torch-backend=auto
# For JAX tools: use ${RECOMMENDED_JAX_SPEC:-jax[cuda12]}
uv pip install -r requirements.txt
```

### 3. requirements.txt
List all Python dependencies with version pins. Do NOT include torch/jax — those are installed by setup.sh using hardware-aware detection.

### 4. env_vars.txt (only if needed)
[passthrough]
HF_TOKEN
[set]
# Only if the tool needs LD_LIBRARY_PATH or other env vars
```

**10C: Run all tests for the new tool**

Run all tests (functional + infra) filtered to the new tool:

```bash
pytest --all --exhaustive -k "{tool_name}" -v
```

This runs the tool's functional tests AND the parametrized infra tests (`example_input`, device consistency, registry integration). A detailed log file is generated in `logs/` (project root).

---

### Subagent 2: README + cite.bib

**What it produces:** `README.md`, `cite.bib`

**Prompt template:**

```
You are writing the documentation for a bioinformatics tool.

## Your Task
Create README.md and cite.bib for the {tool_name} tool in:
  bio_programming_tools/tools/{category}/{tool_name}/

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

CRITICAL: Use exact import paths from bio_programming_tools.tools.{category}.{tool_name}. Class names and function names must match the contract exactly.

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
Create examples/example.ipynb for the {tool_name} tool in:
  bio_programming_tools/tools/{category}/{tool_name}/examples/

## Contract (the tool's API)
<paste full tool file content here>

## Instructions
Create a Jupyter notebook (.ipynb JSON) with these cells:

1. **Markdown title cell** — Tool name, brief description, paper link
2. **Code: Imports** — Exact imports from bio_programming_tools.tools.{category}.{tool_name}
3. **Markdown: Input API Reference** — Table with Field, Type, Default, Description
4. **Markdown: Config API Reference** — Same table format
5. **Markdown: Output API Reference** — Same table format
6. **Code: Basic Usage** — Minimal working example with realistic biological data
7. **Code: Advanced Usage** — Example with non-default config parameters
8. **Code: Export** — Demonstrate result.export()

Notebook metadata:
- kernel: bio-programming, Python 3.12
- Use realistic biological data (real protein sequences, not placeholder text)
- Include comments explaining each step
- Show output inspection (printing key fields)

Write the notebook as valid .ipynb JSON (the raw JSON format with cells array, metadata, etc.).
```

---

### Subagent 4: Tests

**What it produces:** `tests/{category}_tests/test_{tool_name}.py`

**Prompt template:**

```
You are writing tests for a bioinformatics tool.

## Your Task
Create tests/{category}_tests/test_{tool_name}.py

## Contract (the tool's API)
<paste full tool file content here>

## Reference Tests
Here are the tests from {reference_tool} as a structural template:
<paste reference test file content>

## Test Structure

```python
"""Tests for {ToolName} tool."""
import pytest
from bio_programming_tools.tools import run_{tool_name}, {ToolName}Input, {ToolName}Config

# If the tool uses structures:
from bio_programming_tools.entities.structures.structure import Structure
from pathlib import Path
TEST_PDB_FILE = Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb"

# If the category has shared data models:
from bio_programming_tools.tools.{category}.shared_data_models import ...

# For export validation:
from tests.tool_infra_tests.test_export_functionality import validate_output


class Test{ToolName}:
    @pytest.mark.uses_gpu  # Add for GPU tools
    def test_basic_execution(self):
        """Test basic tool execution with default config."""
        # Create input with realistic data
        # Run tool
        # Assert result.success
        # Assert result.tool_id == "{tool-key}"
        # Assert specific output fields are populated
        # validate_output(result)

    def test_input_normalization(self):
        """Test that single inputs are normalized to lists."""

    def test_empty_input_raises(self):
        """Test that empty input raises ValidationError."""

    @pytest.mark.uses_gpu
    def test_export(self, tmp_path):
        """Test output export to supported formats."""

    @pytest.mark.uses_gpu
    def test_batched_execution(self):
        """Test with batch_size > 1 if applicable."""
```

CRITICAL RULES:
- Import from `bio_programming_tools.tools` (top-level), not from deep paths
- Use `@pytest.mark.uses_gpu` for any test that runs the actual tool
- Use `validate_output()` from test_export_functionality to check metadata
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
File: bio_programming_tools/tools/{category}/{tool_name}/__init__.py

```python
from bio_programming_tools.tools.{category}.{tool_name}.{operation} import (
    {ToolName}Config,
    {ToolName}Input,
    {ToolName}Output,
    run_{tool_name},
)

__all__ = [
    "{ToolName}Config",
    "{ToolName}Input",
    "{ToolName}Output",
    "run_{tool_name}",
]
```

### Level 2: Update category __init__.py
File: bio_programming_tools/tools/{category}/__init__.py
Add import block and __all__ entries for the new tool. Preserve all existing imports.

### Level 3: Update master tools/__init__.py
File: bio_programming_tools/tools/__init__.py
Add import block and __all__ entries. Preserve all existing imports.

CRITICAL RULES:
- Do NOT use `from __future__ import annotations` in `__init__.py` files
- Sort `__all__` alphabetically
- Do NOT remove or modify any existing imports
- Match class names and function names EXACTLY from the contract
- If the tool has multiple operations (e.g., sample + score), export ALL of them
- Level 4 (bio_programming_tools/__init__.py) uses `from bio_programming_tools.tools import *` — no changes needed
- Use the Edit tool to modify existing files, Write for new files
```

---

## Phase 4: Verify

**Goal:** Confirm all subagent outputs integrate correctly.

**Steps:**

1. **Check import chain** — Verify the tool imports correctly:
   ```bash
   python3 -c "from bio_programming_tools.tools import run_{tool_name}, {ToolName}Input, {ToolName}Config; print('Import OK')"
   ```

2. **Run tests** — Execute the test file:
   ```bash
   python3 -m pytest tests/{category}_tests/test_{tool_name}.py -v --cpu
   ```
   (Use `--cpu` if no GPU available; tests requiring GPU will be skipped)

3. **Validate tool registration** — Check the tool appears in the registry:
   ```bash
   python3 -c "from bio_programming_tools.tools.tool_registry import ToolRegistry; specs = [s for s in ToolRegistry.list_all() if '{tool_name}' in s.key]; print([(s.key, s.label) for s in specs])"
   ```

4. **Fix any failures** — If imports fail, check the export chain. If tests fail, fix the tool file or test file directly.

5. **Run the validation checklist:**
   - [ ] File starts with `from __future__ import annotations`
   - [ ] Uses `logging.getLogger(__name__)`, never `print()`
   - [ ] Input extends `BaseToolInput`, uses `Field()` (not ConfigField)
   - [ ] Config extends `BaseConfig`, uses `ConfigField()` (not bare Field)
   - [ ] Output extends `BaseToolOutput`, implements `output_format_options`, `output_format_default`, `_export_output()`
   - [ ] Run function signature: `def run_*(inputs: *Input, config: *Config) -> *Output`
   - [ ] Run function returns Output with `metadata={}` dict of key parameters
   - [ ] `@tool()` has all 9 kwargs: key, label, category, input_class, config_class, output_class, description, uses_gpu, example_input
   - [ ] `@tool()`: `uses_gpu=True` matches Config `device="cuda"` override
   - [ ] `@tool()`: Optional `device_count` specifies expected device allocation ("1", "1-2", ">=1", etc.)
   - [ ] No try/except wrapping tool logic
   - [ ] Google-style docstrings with Attributes (for classes) and Args/Returns/Examples (for functions)
   - [ ] `__init__.py` exports at all 4 levels
   - [ ] README.md exists with correct import paths
   - [ ] cite.bib exists with valid BibTeX
   - [ ] Example notebook exists
   - [ ] Tests written and importable
   - [ ] Biological coordinates are 1-indexed, inclusive (if applicable)

---

## Phase 4.5: Self-Audit

**Goal:** Catch convention violations, dead fields, hardcoded assumptions, and consistency gaps before shipping.

**When:** After Phase 4 passes (imports work, tests pass, registry OK). Before Phase 5 (Ship).

Launch a single audit agent via `Task()`:

```
You are auditing a newly implemented bioinformatics tool for quality issues.

## Your Task
Read all generated files for the {tool_name} tool and check for issues.

## Files to Read
- bio_programming_tools/tools/{category}/{tool_name}/{operation}.py (core tool file)
- bio_programming_tools/tools/{category}/{tool_name}/standalone/inference.py (or run.py)
- bio_programming_tools/tools/{category}/{tool_name}/standalone/setup.sh
- bio_programming_tools/tools/{category}/{tool_name}/standalone/requirements.txt
- bio_programming_tools/tools/{category}/{tool_name}/README.md
- bio_programming_tools/tools/{category}/{tool_name}/examples/example.ipynb
- tests/{category}_tests/test_{tool_name}.py

## Reference Tool (for comparison)
Also read the same files from {reference_tool} to compare patterns.

## Checks
1. **Module-level imports** — Heavy dependencies (torch, model libraries) must only be imported inside functions in standalone scripts, never at module level
2. **Inherited fields** — Every field inherited from base classes (shared_data_models.py) must be either implemented in the standalone code or raise ValueError when non-default values are provided
3. **Hardcoded values** — Flag any values in standalone code that are hardcoded but should vary with input (chain IDs, model paths, assumed dimensions, fixed sequence lengths)
4. **Contract-standalone consistency** — Every Config field with `reload_on_change=True` must be read and used in the standalone code. Every operation dispatched in the tool file must be handled in standalone dispatch()
5. **Notebook accuracy** — Import paths, class names, and field names in the example notebook must exactly match the contract
6. **Seed propagation** — If Config has a `seed` field, it must be passed through dispatch and used in the standalone code
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
5. **Log findings to Pilot Memory** — If any `"fix"` issues were found, save them to persistent memory so we can track how the pipeline fails over time:
   ```
   mcp__plugin_pilot_mem-search__save_memory(
       text="implement-tool audit for {tool_name}: {N} fix issues, {M} cosmetic. Fixes: {brief list of fix issues}. These indicate systemic gaps in the pipeline.",
       title="implement-tool audit: {tool_name}",
       project="bio-programming-tools"
   )
   ```
   This builds a dataset of pipeline failure modes that informs future skill improvements.

---

## Phase 5: Ship

**Goal:** Commit, push, create PR.

**Steps:**

1. **Stage all new files:**
   ```bash
   git add bio_programming_tools/tools/{category}/{tool_name}/
   git add tests/{category}_tests/test_{tool_name}.py
   # Also stage modified __init__.py files
   git add bio_programming_tools/tools/{category}/__init__.py
   git add bio_programming_tools/tools/__init__.py
   ```

2. **Commit:**
   ```bash
   git commit -m "feat: implement {tool_name} tool wrapper

   Implements {tool_display_name} ({operations}) in the {category} category.
   Closes #{issue_number}"
   ```

3. **Push and create PR:**
   ```bash
   git push -u origin HEAD
   gh pr create --title "feat: implement {tool_name} tool" --body "$(cat <<'EOF'
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
   - [ ] `python3 -c "from bio_programming_tools.tools import run_{tool_name}"` imports successfully
   - [ ] `pytest tests/{category}_tests/test_{tool_name}.py` passes
   - [ ] Tool appears in ToolRegistry.list_all()
   - [ ] GPU execution verified (if applicable)
   EOF
   )"
   ```

4. **Report the PR URL to the user.**
