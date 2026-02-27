---
name: implement-tool
description: >
  Implements a new bioinformatics tool wrapper in bio-programming-tools.
  Covers the full lifecycle: directory structure, Input/Config/Output Pydantic
  data models, @tool decorator registration, ToolInstance standalone execution,
  4-level __init__.py export chain, caching, tests, README, cite.bib, and
  example notebook. Use when creating tools, wrapping models (ESM2, Evo2,
  AlphaFold, BLAST), writing standalone scripts, or debugging tool registration.
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
---

# Implement a New Tool

You are implementing a new bioinformatics tool in the bio-programming-tools codebase. This codebase has extremely strict conventions -- every tool must follow the exact same patterns. Read this entire guide before writing any code.

## Step 0: Gather References

**Before writing ANY code, ask the user for the following:**

1. **Source GitHub repository URL** for the tool being wrapped (e.g., https://github.com/sokrypton/ColabFold)
2. **API documentation or web app reference** (if applicable)
3. **Academic paper link/DOI** (if applicable)

**After receiving the references:**
- Use WebFetch to read the GitHub README and understand the tool's interface, inputs, outputs, and dependencies
- Read any API docs or paper abstracts to understand biological context
- Identify the tool's key parameters, input formats, and output formats

---

## Step 1: Tool Architecture

**Always use the standalone pattern** -- every tool runs in an isolated venv via ToolInstance:

```
tools/{category}/{tool_name}/
+-- __init__.py
+-- {tool_name}.py          # Input, Config, Output, run function (calls ToolInstance)
+-- cite.bib                # BibTeX citation (required)
+-- examples/
|   +-- example.ipynb       # Working example notebook (required)
+-- standalone/
|   +-- setup.sh            # Creates venv, installs deps
|   +-- run.py OR inference.py  # run.py for CPU tools, inference.py for AI models
|   +-- requirements.txt    # Python dependencies
|   +-- binary_config.py    # [optional] For external C/C++ binaries
+-- README.md
```

**Optional: Shared Data Models** -- If the category already has 2+ tools with overlapping schemas (e.g., structure_prediction, inverse_folding, causal_models), use a shared data models file at the category level:
```
tools/{category}/
+-- shared_data_models.py   # Shared Input/Config/Output base classes
+-- {tool_name}/
|   +-- __init__.py
|   +-- {tool_name}.py      # Extends shared models, calls ToolInstance
|   +-- cite.bib            # BibTeX citation (required)
|   +-- examples/
|   |   +-- example.ipynb   # Working example notebook (required)
|   +-- standalone/
|   |   +-- setup.sh
|   |   +-- run.py OR inference.py  # run.py for CPU tools, inference.py for AI models
|   |   +-- requirements.txt
|   |   +-- binary_config.py   # [optional]
|   +-- README.md
+-- __init__.py
```

---

## Step 2: Create the Tool File

Every tool file has exactly 3 sections: **Data Models**, **Tool Implementation**, and uses exactly the standard imports. The complete template with Input/Config/Output classes, `@tool()` decorator (8 kwargs: key, label, category, input, config, output, description, uses_gpu), and run function is in the templates file.

For complete code templates: Read `.claude/skills/implement-tool/TEMPLATES.md`

---

## Step 3: Implementation Patterns

Choose the right pattern based on your tool type:

- **Standalone CPU Tool** -- uses `ToolInstance.dispatch()` with `run.py` for CPU-based tools
- **AI Model Tool (GPU)** -- uses `ToolInstance.dispatch()` with `inference.py`, includes `reload_on` for model config changes
- **Compile-from-Source Tool** -- for C/C++ source distributions, compiles during `setup.sh` (no `binary_config.py` or `requirements.txt`)
- **Batching** -- GPU tools include `batch_size` config field (default 1), standalone script implements the batching loop

For full implementation patterns with code examples: Read `.claude/skills/implement-tool/PATTERNS.md`

---

## Step 4: Caching

Two caching decorators are available:

- **`@tool_cache("tool-key")`** -- whole-result caching (single output). Goes BELOW `@tool`.
- **`@tool_cache_iterable(...)`** -- per-item caching (list/batch). Goes ABOVE `@tool`.

Import from: `from bio_programming_tools.utils.tool_cache import tool_cache, tool_cache_iterable`

For full caching patterns with decorator ordering: Read `.claude/skills/implement-tool/PATTERNS.md`

---

## Step 5: The `__init__.py` Export Chain

**You MUST update ALL 4 levels.** Missing any level breaks imports.

1. **Tool `__init__.py`** -- exports Input, Config, Output, run_* from the tool module
2. **Category `__init__.py`** -- re-exports from the tool's `__init__.py`
3. **Master `tools/__init__.py`** -- re-exports from the category
4. **Package `bio_programming_tools/__init__.py`** -- uses `from bio_programming_tools.tools import *` (no changes needed if `tools/__init__.py` is correct)

For complete export chain code examples at all 4 levels: Read `.claude/skills/implement-tool/PATTERNS.md`

---

## Step 6: Write the README.md

Create `tools/{category}/{tool_name}/README.md` with: Overview (biological context), Key Parameters, Quick Start (with exact imports), Interpreting Results, and References.

For the complete README template: Read `.claude/skills/implement-tool/TEMPLATES.md`

---

## Step 7: Create the cite.bib

Every tool **must** have a `cite.bib` file with the BibTeX citation for the underlying tool/paper. This enables `ToolRegistry.get_citation("tool-key")` to return the citation. Use the paper's DOI to find the correct BibTeX entry.

For the cite.bib template: Read `.claude/skills/implement-tool/TEMPLATES.md`

---

## Step 8: Create the Example Notebook

Create `tools/{category}/{tool_name}/examples/example.ipynb` with title cell, import cell, API reference tables, execution cells with realistic biological data, and export cell.

For example notebook guidance: Read `.claude/skills/implement-tool/TEMPLATES.md`

---

## Step 9: Write Tests

Create `tests/{category}_tests/test_{tool_name}.py` with test classes covering basic execution, input normalization, empty input validation, and CSV export. GPU tools use the `@pytest.mark.uses_gpu` marker.

For the complete test template: Read `.claude/skills/implement-tool/TEMPLATES.md`

---

## Step 10: Verify by Running the Tool

After implementing, do BOTH of the following:

**10A: Run the tests**
```bash
pytest tests/{category}_tests/test_{tool_name}.py -v
```

**10B: Run the tool directly** -- write and execute a short verification script that imports the tool, runs it with realistic data, and confirms all output fields are populated.

For the complete verification script template: Read `.claude/skills/implement-tool/TEMPLATES.md`

---

## Documentation

Documentation `.mdx` files in `docs/` are auto-generated by `generate_docs.py` (run by pre-commit hooks). Never manually edit `.mdx` files -- update the Python config docstrings/field descriptions instead.

## Reference Implementations

When in doubt, read these canonical examples:

| Pattern | Example | File |
|---|---|---|
| Standalone + binary | BLAST | `tools/gene_annotation/blast/` |
| CPU standalone (run.py) | BLAST | `tools/gene_annotation/blast/standalone/run.py` |
| AI model standalone (inference.py) | ESMFold | `tools/structure_prediction/esmfold/standalone/inference.py` |
| GPU standalone | Evo2 | `tools/causal_models/evo2/evo2_sample.py` |
| Compile-from-source | TMalign | `tools/structure_alignment/tmalign/` |
| Shared data models | Inverse Folding | `tools/inverse_folding/shared_data_models.py` |
| Per-item caching | Orfipy | `tools/orf_prediction/orfipy/orfipy.py` |
| Scoring tool | Evo2 Score | `tools/causal_models/evo2/evo2_score.py` |
| Structure prediction | ESMFold | `tools/structure_prediction/esmfold/` |

Read the actual source files of these tools when implementing -- they are the ground truth for conventions.

## Validation Checklist

- [ ] File starts with `from __future__ import annotations`
- [ ] Uses `logging.getLogger(__name__)`, never `print()`
- [ ] Input extends `BaseToolInput`, uses `Field()` (not ConfigField)
- [ ] Config extends `BaseConfig`, uses `ConfigField()` (not bare Field)
- [ ] Output extends `BaseToolOutput`, does NOT redeclare inherited metadata fields
- [ ] Output implements `output_format_options`, `output_format_default`, `_export_output()`
- [ ] `@tool()` decorator has all 8 kwargs: key, label, category, input, config, output, description, uses_gpu
- [ ] Run function signature: `def run_*(inputs: *Input, config: *Config) -> *Output`
- [ ] Run function returns Output with `metadata={}` dict of key parameters
- [ ] No try/except wrapping tool logic -- `@tool` decorator handles errors
- [ ] Google-style docstrings with Attributes (for classes) and Args/Returns/Examples (for functions)
- [ ] `__init__.py` exports at all 4 levels
- [ ] README.md in tool directory
- [ ] cite.bib with BibTeX citation in tool directory
- [ ] `examples/example.ipynb` with working code, API reference tables, and example output
- [ ] Tests written and passing
- [ ] Tool runs end-to-end (verified via verification script)
- [ ] Biological coordinates are 1-indexed, inclusive (if applicable)
