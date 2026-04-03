# implement-tool: Implementation Patterns

Reference file for the `implement-tool` skill. Patterns are tagged with which subagent or phase consumes them.

---

## Standalone CPU Tool (ToolInstance) — [Subagent 1: Standalone + Phase 2: Contract]

**Main tool file** (Phase 2 — Contract) — calls ToolInstance with `run.py`:
```python
from proto_tools.utils.tool_instance import ToolInstance

def run_tool_name(inputs: ToolInput, config: ToolConfig) -> ToolOutput:
    logger.debug("Using local venv for tool_name operation")

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
        config=config,
    )

    return ToolOutput(
        results=output_data["results"],
        metadata={"param1": config.param1},
    )
```

**standalone/run.py** (Subagent 1) — JSON I/O entry point:
```python
"""
{ToolName} standalone runner for ToolInstance venv execution.
Usage (called by ToolInstance, not directly):
    python run.py <input.json> <output.json>  # CPU tools
    python inference.py <input.json> <output.json>  # AI model tools
"""

import json
import sys


def run_operation(input_data: dict) -> dict:
    """Run the main operation. Returns JSON-serializable dict."""
    import some_library

    results = some_library.run(input_data["sequences"], param=input_data["param1"])
    return {"results": results}


# ---------------------------------------------------------------------------
# Device management protocol (required by DeviceManager)
# ---------------------------------------------------------------------------
def to_device(device: str) -> dict:
    """Passthrough for CPU tool — no persistent model to move."""
    return {"success": True, "device": device, "note": "CPU tool, auto-unloads"}


def get_memory_stats() -> dict:
    """CPU tool — no GPU memory to report."""
    return {"available": False, "framework": "cpu", "note": "CPU tool"}


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

**standalone/setup.sh** (Subagent 1) — sources `standalone_helpers.sh` (auto-copied) for shared functions:
```bash
#!/bin/bash
set -euo pipefail
source standalone_helpers.sh

pip install uv
uv pip install -r requirements.txt

echo "Setup complete!"
```

**standalone/setup.sh with weight download** (Subagent 1) — for tools that download model weights. See `tools/inverse_folding/fampnn/standalone/setup.sh`:
```bash
#!/bin/bash
set -euo pipefail
source standalone_helpers.sh

pip install uv
uv pip install -r requirements.txt

# Resolve weight directory based on PROTO_MODEL_CACHE
proto_resolve_weights_dir my_tool

if [ ! -f "$WEIGHTS_DIR/model.pt" ]; then
    echo "Downloading model weights..."
    wget -q -O "$WEIGHTS_DIR/model.pt" "https://example.com/model.pt"
fi

echo "Setup complete!"
```

**standalone/requirements.txt** (Subagent 1):
```
some-library>=1.0.0
numpy>=1.24.0
```

---

## AI Model Tool (GPU) — [Subagent 1: Standalone + Phase 2: Contract]

**Main tool file** (Phase 2 — Contract):
```python
from proto_tools.utils.tool_instance import ToolInstance

def run_tool_name(inputs: ToolInput, config: ToolConfig) -> ToolOutput:
    logger.debug("Using local venv for tool_name operation")

    result = ToolInstance.dispatch(
        "{tool_name}",
        {
            "operation": "run",
            "sequences": inputs.sequences,
            "param1": config.param1,
            "device": config.device,
        },
        script_path=Path(__file__).parent / "standalone" / "inference.py",
        config=config,
    )

    return ToolOutput(results=result["results"])
```

**standalone/inference.py** (Subagent 1) — GPU model pattern:
```python
"""
{ToolName} standalone inference for ToolInstance venv execution.
"""

import json
import sys
import os
from typing import Any

# Model class with lazy loading
class {ToolName}Model:
    def __init__(self):
        self._loaded = False
        self.device = None
        self.model = None

    def load(self, device: str, verbose: bool = False):
        """Load model weights to device."""
        import torch  # Heavy imports ONLY inside methods
        # For non-HF tools, resolve weight path:
        # from standalone_helpers import resolve_weights_dir
        # weights_dir = resolve_weights_dir("my_tool")
        # model_path = os.path.join(weights_dir, "model.pt")
        # Load model...
        self._loaded = True
        self.device = device

    def predict(self, input_data: dict) -> dict:
        """Run inference. Returns JSON-serializable dict."""
        if not self._loaded:
            self.load(input_data.get("device", "cuda"))
        # Run inference...
        return {"results": [...]}


def _serialize_output(value: Any) -> Any:
    """Recursively serialize tensors and arrays to JSON-safe types."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: _serialize_output(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_output(v) for v in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


_model = None

def dispatch(input_dict: dict) -> dict:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    if _model is None:
        _model = {ToolName}Model()

    operation = input_dict.get("operation", "predict")
    if operation == "predict":
        return _serialize_output(_model.predict(input_dict))
    else:
        raise ValueError(f"Unknown operation: {operation}")


# ---------------------------------------------------------------------------
# Device management protocol (required by DeviceManager)
# ---------------------------------------------------------------------------
def to_device(device: str) -> dict:
    """Move model to specified device (called by DeviceManager)."""
    global _model
    if _model is not None and _model._loaded:
        _model.to_device(device)
        return {"success": True, "device": device}
    else:
        return {"success": True, "device": device, "note": "model not loaded yet"}


def get_memory_stats() -> dict:
    """Report GPU memory usage (called by DeviceManager)."""
    from standalone_helpers import get_pytorch_memory_stats  # Auto-copied by worker bootstrap
    global _model
    device = _model.device if _model and hasattr(_model, "device") else 0
    return get_pytorch_memory_stats(device)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python inference.py <input.json> <output.json>")
    with open(sys.argv[1], "r") as f:
        input_data = json.load(f)
    result = dispatch(input_data)
    with open(sys.argv[2], "w") as f:
        json.dump(result, f)
```

**standalone/setup.sh for PyTorch tools** (Subagent 1):
```bash
#!/bin/bash
set -euo pipefail
source standalone_helpers.sh

echo "Installing uv package manager..."
pip install uv

proto_install_pytorch
# If torchvision/torchaudio needed: proto_install_pytorch "" torchvision

echo "Installing remaining dependencies..."
uv pip install -r requirements.txt

# Non-HF tools that download weights:
# proto_resolve_weights_dir my_tool
# wget -q -O "$WEIGHTS_DIR/model.pt" "https://example.com/model.pt"

echo "{ToolName} setup complete!"
```

### Downloading model weights in setup.sh

When a tool needs to download model weights or archives during setup:

- Use `curl -fsSL` (not `wget` — `wget` is not available in micromamba environments)
- Use `python -c "import zipfile; ..."` (not `unzip` — `unzip` is not available in micromamba environments)
- `set -euo pipefail` at the top ensures `curl` failures are fatal — no need for explicit error checks
- For prebuilt binaries, use `utils/install_binary.py` instead (see `notes/tool-environments.md`)

```bash
# Download model weights
WEIGHTS_DIR="${VENV_PATH}/weights"
mkdir -p "$WEIGHTS_DIR"

WEIGHTS_FILE="${WEIGHTS_DIR}/model.pt"
if [ ! -f "$WEIGHTS_FILE" ]; then
    echo "Downloading model weights..."
    curl -fsSL -o "$WEIGHTS_FILE" "https://example.com/model.pt"
fi

# Download and extract a zip archive
ARCHIVE="${WEIGHTS_DIR}/data.zip"
if [ ! -f "${WEIGHTS_DIR}/data.bin" ]; then
    echo "Downloading data archive..."
    curl -fsSL -o "$ARCHIVE" "https://example.com/data.zip"
    python -c "import zipfile; zipfile.ZipFile('${ARCHIVE}').extractall('${WEIGHTS_DIR}')"
    rm -f "$ARCHIVE"
fi
```

**standalone/setup.sh for JAX tools** (Subagent 1):
```bash
#!/bin/bash
set -euo pipefail
source standalone_helpers.sh

echo "Installing uv package manager..."
pip install uv

proto_install_jax MYTOOL

echo "Installing remaining dependencies..."
uv pip install -r requirements.txt

echo "{ToolName} setup complete!"
```

### Required Functions for AI Model Standalone Scripts

**All GPU tools must implement two protocol functions in their standalone scripts** (inference.py or run.py):

**IMPORTANT: Persistent vs Non-Persistent Tools**
- **Persistent tools**: Keep a global `_model` object loaded in memory between calls (e.g., ESMFold, Evo2, ESM2)
- **Non-persistent tools**: Create a new model instance on each call and unload it after (e.g., BioEmu)
- **CLI tools**: Spawn subprocesses that naturally unload after completion (e.g., Boltz2, RFDiffusion3)

#### 1. `to_device(device: str) -> dict`

Enables DeviceManager to move models between GPUs and CPU for resource management.

**Persistent PyTorch tools:**
```python
def to_device(device: str) -> dict:
    global _model
    if _model is not None and _model._loaded:
        _model.to_device(device)
        return {"success": True, "device": device}
    else:
        return {"success": True, "device": device, "note": "model not loaded yet"}
```

**CLI tools:**
```python
def to_device(device: str) -> dict:
    return {"success": True, "device": device, "note": "CLI tool, auto-unloads"}
```

#### 2. `get_memory_stats() -> dict`

Enables DeviceManager to query GPU memory usage for monitoring.

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

**Key points:**
- Both functions must be defined at module level (not inside classes)
- Import the appropriate helper from `standalone_helpers` (auto-copied by worker bootstrap)
- `to_device()` must return a dict with `{"success": bool, "device": str}`
- `get_memory_stats()` must return a dict with `{"available": bool, "framework": str, ...}`
- The standalone helpers already include the 'framework' key in all return paths — just call them directly

---

### CLI Tools with Subprocess Calls — [Subagent 1: Standalone]

**For tools that spawn CLI subprocesses** (e.g., Boltz2, RFDiffusion3, Protenix), use `get_subprocess_device_env()` from `standalone_helpers` to ensure correct device routing when parent has `CUDA_VISIBLE_DEVICES` set.

**Why this is needed:** When DeviceManager allocates a logical device (e.g., `cuda:2`), the worker subprocess inherits the parent's `CUDA_VISIBLE_DEVICES` (e.g., `0,1,5,7`). The CLI subprocess needs the physical GPU index (e.g., `5`) mapped from the logical index (2).

**standalone/run.py** — CLI subprocess pattern:
```python
"""
{ToolName} standalone runner for ToolInstance venv execution.

CRITICAL: This script runs in an isolated environment and CANNOT import from proto_tools.
Only import from: stdlib, requirements.txt dependencies, and standalone_helpers (auto-copied).
"""

import json
import subprocess
import sys

from standalone_helpers import get_subprocess_device_env  # Auto-copied by worker bootstrap


def run_operation(
    input_data: dict,
    device: str = "cuda",
) -> dict:
    """Run the CLI tool with correct device environment."""
    cmd = [
        "some-cli-tool",
        "--input", input_data["input_file"],
        "--output", input_data["output_dir"],
    ]

    # Get subprocess environment with correct CUDA_VISIBLE_DEVICES
    env = get_subprocess_device_env(device)

    # Run CLI subprocess with mapped device
    subprocess.run(cmd, check=True, text=True, env=env, encoding="utf-8")

    return {"status": "success"}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input.json> <output.json>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        input_data = json.load(f)

    device = input_data.get("device", "cuda")
    operation = input_data["operation"]

    if operation == "run":
        output_data = run_operation(input_data, device=device)
    else:
        raise ValueError(f"Unknown operation: {operation}")

    with open(sys.argv[2], "w") as f:
        json.dump(output_data, f)
```

**Key points:**
- Import `get_subprocess_device_env` from `standalone_helpers` (auto-copied by worker bootstrap)
- Accept `device` parameter in operation functions
- Call `env = get_subprocess_device_env(device)` before subprocess calls
- Pass `env=env` to `subprocess.run()` or `subprocess.Popen()`

---

### Batching Convention — [Subagent 1: Standalone + Phase 2: Contract]

GPU tools should include `batch_size: int = ConfigField(default=1, ...)` in their config.

**Rules:**
- Default is always `1` — safe by default, prevents OOM errors
- The standalone `inference.py` implements the batching loop (chunking inputs, iterating)
- Generators and constraints pass `batch_size` through to tool configs — they never batch themselves
- Higher `batch_size` = more GPU memory, higher throughput

---

## Compile-from-Source Tool — [Subagent 1: Standalone]

For tools distributed as C/C++ source (no prebuilt binaries or pip packages), compile during setup. No `binary_config.py` or `requirements.txt` needed.

**standalone/setup.sh** (Subagent 1):
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

## Caching Patterns — [Phase 2: Contract]

Add `cacheable=True` to the `@tool()` decorator. The wrapper auto-selects strategy:
- **Iterable tools** (have `iterable_input_field`) → per-item cache (strip/stitch)
- **Non-iterable tools** → whole-output cache

```python
# Per-item caching (tools processing lists/batches):
@tool(key="tool-key", ..., iterable_input_field="sequences", iterable_output_field="results", cacheable=True)
def run_tool_name(inputs, config) -> Output:

# Whole-output caching (tools with single output):
@tool(key="tool-key", ..., cacheable=True)
def run_tool_name(inputs, config) -> Output:
```

No separate imports needed — caching is built into the `@tool()` decorator.
Generative tools (e.g., samplers) should NOT set `cacheable=True`.

---

## Export Chain (`__init__.py` at all 4 levels) — [Subagent 5: Export Chain]

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
    "{ToolName}Config",
    "{ToolName}Input",
    "{ToolName}Output",
    "run_{tool_name}",
]
```

**Note:** Sort `__all__` alphabetically.

### Level 2: Category `__init__.py`
`tools/{category}/__init__.py` — add imports:
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
    "{ToolName}Config",
    "{ToolName}Input",
    "{ToolName}Output",
    "run_{tool_name}",
]
```

### Level 3: Master `tools/__init__.py`
`tools/__init__.py` — add import block and __all__ entries:
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
    "{ToolName}Config",
    "{ToolName}Input",
    "{ToolName}Output",
    "run_{tool_name}",
]
```

### Level 4: Package `__init__.py`
`proto_tools/__init__.py` — this file uses `from proto_tools.tools import *` so no changes needed IF the tool is properly added to `tools/__init__.py`'s `__all__`.
