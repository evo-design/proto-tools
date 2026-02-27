# implement-tool: Implementation Patterns

Reference file for the `implement-tool` skill. Contains implementation patterns for different tool types, caching, and the export chain.

## Standalone CPU Tool (ToolInstance)

**Main tool file** -- calls ToolInstance with `run.py`:
```python
def run_tool_name(inputs: ToolInput, config: ToolConfig) -> ToolOutput:
    from bio_programming_tools.utils.tool_instance import ToolInstance

    input_data = {
        "operation": "{operation_name}",
        "sequences": inputs.sequences,
        "param1": config.param1,
        "device": config.device,
    }

    output_data = ToolInstance.dispatch(
        "{tool_name}",
        input_data,
        script_path=Path(__file__).parent / "standalone" / "run.py",
        verbose=config.verbose,
    )

    return ToolOutput(
        results=output_data["results"],
        metadata={"param1": config.param1},
    )
```

**standalone/run.py** (or **inference.py** for AI models) -- JSON I/O entry point:
```python
"""
{ToolName} standalone runner for ToolInstance venv execution.
Usage (called by ToolInstance, not directly):
    python run.py <input.json> <output.json>  # CPU tools
    python inference.py <input.json> <output.json>  # AI model tools
"""
from __future__ import annotations

import json
import sys


def run_operation(input_data: dict) -> dict:
    """Run the main operation. Returns JSON-serializable dict."""
    import some_library

    results = some_library.run(input_data["sequences"], param=input_data["param1"])
    return {"results": results}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input.json> <output.json>", file=sys.stderr)
        sys.exit(1)

    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2]

    with open(input_json_path, "r") as f:
        input_data = json.load(f)

    operation = input_data["operation"]

    if operation == "operation_name":
        output_data = run_operation(input_data)
    else:
        raise ValueError(f"Unknown operation: {operation}")

    with open(output_json_path, "w") as f:
        json.dump(output_data, f)
```

**standalone/setup.sh**:
```bash
#!/bin/bash
set -euo pipefail

pip install uv
uv pip install -r requirements.txt

echo "Setup complete!"
```

**standalone/requirements.txt**:
```
some-library>=1.0.0
numpy>=1.24.0
```

---

## AI Model Tool (GPU)

```python
def run_tool_name(inputs: ToolInput, config: ToolConfig) -> ToolOutput:
    from bio_programming_tools.utils.tool_instance import ToolInstance

    result = ToolInstance.dispatch(
        "{tool_name}",
        {
            "operation": "run",
            "sequences": inputs.sequences,
            "param1": config.param1,
            "device": config.device,
        },
        script_path=Path(__file__).parent / "standalone" / "inference.py",
        verbose=config.verbose,
        reload_on=type(config).reload_fields(),  # Restart worker if device/checkpoint changes
    )

    return ToolOutput(results=result["results"])
```

### Batching Convention

GPU tools should include `batch_size: int = ConfigField(default=1, ...)` in their config.

**Rules:**
- Default is always `1` -- safe by default, prevents OOM errors
- The standalone `inference.py` implements the batching loop (chunking inputs, iterating)
- Generators and constraints pass `batch_size` through to tool configs -- they never batch themselves
- Higher `batch_size` = more GPU memory, higher throughput

---

## Compile-from-Source Tool

For tools distributed as C/C++ source (no prebuilt binaries or pip packages), compile during setup. No `binary_config.py` or `requirements.txt` needed.

**standalone/setup.sh**:
```bash
#!/bin/bash
set -euo pipefail

echo "Setting up {ToolName}..."

# Check for compiler
if ! command -v g++ &>/dev/null; then
    echo "ERROR: g++ not found. Install a C++ compiler (e.g., apt install g++)." >&2
    exit 1
fi

pip install uv

# Compile from source
BUILD_DIR=$(mktemp -d)
git clone --depth 1 https://github.com/{org}/{repo}.git "$BUILD_DIR/src"

BIN_DIR="$(dirname "$(which python)")"
g++ -O3 -ffast-math -lm -o "$BIN_DIR/{ToolBinary}" "$BUILD_DIR/src/{source}.cpp"

rm -rf "$BUILD_DIR"
echo "{ToolName} setup complete!"
```

**Key differences from CPU/GPU patterns:**
- Check for `g++`/`gcc`/`cmake` before compiling
- Use `BUILD_DIR` (not `TMPDIR`) to avoid shadowing the environment variable
- No `requirements.txt` or `binary_config.py`
- Binary is compiled directly into the venv's `bin/` directory

**Canonical examples:** TMalign (`tools/structure_alignment/tmalign/`) and USalign (`tools/structure_alignment/usalign/`)

---

## Caching Patterns

### Whole-result caching (for tools with single output)
```python
@tool(key="tool-key", ...)
@tool_cache("tool-key")
def run_tool_name(inputs, config) -> Output:
```
Note: `@tool_cache` goes BELOW `@tool`.

### Per-item caching (for tools processing lists/batches)
```python
@tool_cache_iterable(
    input_iterable_field="sequences",       # List field in Input
    output_iterable_field="results",        # List field in Output
    tool_name="tool-key",
)
@tool(key="tool-key", ...)
def run_tool_name(inputs, config) -> Output:
```
Note: `@tool_cache_iterable` goes ABOVE `@tool`.

Import from: `from bio_programming_tools.utils.tool_cache import tool_cache, tool_cache_iterable`

---

## Export Chain (`__init__.py` at all 4 levels)

**You MUST update ALL 4 levels.** Missing any level breaks imports.

### Level 1: Tool `__init__.py`
`tools/{category}/{tool_name}/__init__.py`:
```python
from .{tool_name} import (
    {ToolName}Config,
    {ToolName}Input,
    {ToolName}Output,
    run_{tool_name},
)

__all__ = [
    "{ToolName}Input",
    "{ToolName}Config",
    "{ToolName}Output",
    "run_{tool_name}",
]
```

### Level 2: Category `__init__.py`
`tools/{category}/__init__.py` -- add imports:
```python
# {ToolName}
from .{tool_name} import (
    {ToolName}Config,
    {ToolName}Input,
    {ToolName}Output,
    run_{tool_name},
)

# ... existing imports ...

__all__ = [
    # ... existing exports ...
    # {ToolName}
    "{ToolName}Input",
    "{ToolName}Config",
    "{ToolName}Output",
    "run_{tool_name}",
]
```

### Level 3: Master `tools/__init__.py`
`tools/__init__.py` -- add import block and __all__ entries:
```python
# {Category} - {ToolName}
from .{category} import (
    {ToolName}Config,
    {ToolName}Input,
    {ToolName}Output,
    run_{tool_name},
)

# In __all__:
__all__ = [
    # ... existing ...
    # {ToolName}
    "run_{tool_name}",
    "{ToolName}Input",
    "{ToolName}Config",
    "{ToolName}Output",
]
```

### Level 4: Package `__init__.py`
`bio_programming_tools/__init__.py` -- this file uses `from bio_programming_tools.tools import *` so no changes needed IF the tool is properly added to `tools/__init__.py`'s `__all__`.
