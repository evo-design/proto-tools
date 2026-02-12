---
name: bio-tools
description: >
  ALWAYS use this skill when the user asks to run, analyze, or write scripts
  for ANY bioinformatics tool or sequence/structure analysis. Invoke this skill
  FIRST before reading tool source code or writing any analysis script.
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
---

# bio-tools skill

## Workflow

1. **Find the tool**: Browse `bio_programming_tools/tools/` to find the right category and tool directory, or use `ToolRegistry.list_all()` from `bio_programming_tools.tools` to discover available tools
2. **Read README**: `bio_programming_tools/tools/{category}/{tool}/README.md` — has parameters, thresholds, biological context, and examples
3. **Read notebook**: `bio_programming_tools/tools/{category}/{tool}/examples/example.ipynb` — has working code with exact imports and real output
4. **Read API**: Read the tool's `Input`/`Config`/`Output` classes for the exact Pydantic schema
5. **Call**: `Input` → `Config` → `run_{tool}()` → `Output`

Start with READMEs and notebooks for API details. Read `.py` source only if you need deeper implementation context.

## Universal Call Pattern

```python
from bio_programming_tools.tools.{category}.{tool} import run_{tool}, {Tool}Input, {Tool}Config

inputs = {Tool}Input(...)   # Primary data: sequences, structures, files
config = {Tool}Config(...)  # Parameters: evalue, num_threads, seeds
result = run_{tool}(inputs, config)
# result.success, result.execution_time, result.errors, plus tool-specific fields
```

## Script vs Direct Execution

Infer from context whether to **write a script** or **execute directly**:

**Write a script** to `./analyses/{descriptive_name}_{YYYY-MM-DD}.py` when:
- The user says "write", "create", "set up", "notebook", or similar authoring language
- The task is a multi-step pipeline or expensive GPU job
- The user will likely iterate on parameters or review before running
- When unclear — default to writing a script (safer, reproducible)

**Execute directly** (run inline via Bash, include code in the response) when:
- The user says "what is", "how many", "show me", "quick", "find", or similar query language
- The task is a simple one-off lookup or quick check
- The answer is more important than the script

In either case, always show the equivalent Python code so the user can reproduce the result. See `analyses/examples/` for reference scripts.

## Script Structure

Generated scripts should follow this structure:

```python
"""
Brief description of what this analysis does.
Generated: {date}
"""
from bio_programming_tools.tools.{category}.{tool} import ...

# --- Configuration (review these) ---
# All parameters in one place with comments explaining choices

# --- Run ---
# Tool execution

# --- Results ---
# Parse and display output
```

For multi-step pipelines, use one script with `# === Step N: Description ===` section headers.

## Citations

Every tool has a BibTeX citation accessible via `ToolRegistry.get_citation("tool-key")`. When writing analysis scripts or reports, include citations for the tools used:

```python
from bio_programming_tools.tools import ToolRegistry

# Get citation for a specific tool
bibtex = ToolRegistry.get_citation("alphafold3-prediction")

# Get all citations
all_citations = ToolRegistry.list_citations()  # {tool_key: bibtex_string}
```

## GPU Tools

Some tools require GPU access. To determine if a tool needs GPU, check its Config class for a `device` field (defaulting to `"cuda"`). When writing scripts for GPU tools, note the GPU requirement in a comment at the top of the script.
