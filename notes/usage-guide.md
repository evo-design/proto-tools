# Using proto-tools with Claude Code

## Script vs Direct Execution

Infer from context whether to **write a script** or **execute directly**:

**Write a script** to `./analyses/{descriptive_name}_{YYYY-MM-DD}.py` when:
- The user says "write", "create", "set up", "notebook", or similar authoring language
- The task is a multi-step pipeline or expensive GPU job
- The user will likely iterate on parameters or review before running
- When unclear, default to writing a script (safer, reproducible)

**Execute directly** when:
- The user says "what is", "how many", "show me", "quick", "find", or similar query language
- The task is a simple one-off lookup or quick check
- The answer is more important than the script

In either case, always show the equivalent Python code so the user can reproduce the result.

## Script Structure

```python
"""
Brief description of what this analysis does.
Generated: {date}
"""
from proto_tools.tools.{category}.{tool} import ...

# --- Configuration (review these) ---
# All parameters in one place with comments explaining choices

# --- Run ---
# Tool execution

# --- Results ---
# Parse and display output
```

For multi-step pipelines, use one script with `# === Step N: Description ===` section headers.

## Batch Persistence

For batch workloads or loops calling the same tool repeatedly, use `ToolInstance.persist()` to avoid reloading the model on every call:

```python
from proto_tools.utils.tool_instance import ToolInstance

with ToolInstance.persist():
    for seq in sequences:
        result = run_esmfold(ESMFoldInput(complexes=[seq]), ESMFoldConfig())
```

## GPU Tools

Some tools require GPU access. Check a tool's Config class for a `device` field (defaulting to `"cuda"`). When writing scripts for GPU tools, note the GPU requirement in a comment at the top.

## Citations

Every tool has a BibTeX citation accessible via `ToolRegistry.get_citation("tool-key")`. When writing analysis scripts or reports, include citations for the tools used. Use `ToolRegistry.list_citations()` for all citations.
