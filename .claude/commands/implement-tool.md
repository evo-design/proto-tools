# Implement a New Tool

You are implementing a new bioinformatics tool in the bio-programming-tools codebase. This codebase has extremely strict conventions — every tool must follow the exact same patterns. Read this entire guide before writing any code.

## Step 0: Gather References

**Before writing ANY code, ask the user for the following:**

1. **Source GitHub repository URL** for the tool being wrapped (e.g., https://github.com/sokrypton/ColabFold)
2. **API documentation or client reference** (if applicable)
3. **Academic paper link/DOI** (if applicable)

**After receiving the references:**
- Use WebFetch to read the GitHub README and understand the tool's interface, inputs, outputs, and dependencies
- Read any API docs or paper abstracts to understand biological context
- Identify the tool's key parameters, input formats, and output formats

---

## Step 1: Tool Architecture

**Always use the standalone pattern** — every tool runs in an isolated venv via EnvManager:

```
tools/{category}/{tool_name}/
├── __init__.py
├── {tool_name}.py          # Input, Config, Output, run function (calls EnvManager)
├── standalone/
│   ├── setup.sh            # Creates venv, installs deps
│   ├── run.py OR inference.py  # run.py for CPU tools, inference.py for AI models
│   ├── requirements.txt    # Python dependencies
│   └── binary_config.py    # [optional] For external C/C++ binaries
└── README.md
```

**Optional: Shared Data Models** — If the category already has 2+ tools with overlapping schemas (e.g., structure_prediction, inverse_folding, causal_models), use a shared data models file at the category level:
```
tools/{category}/
├── shared_data_models.py   # Shared Input/Config/Output base classes
├── {tool_name}/
│   ├── __init__.py
│   ├── {tool_name}.py      # Extends shared models, calls EnvManager
│   ├── standalone/
│   │   ├── setup.sh
│   │   ├── run.py OR inference.py  # run.py for CPU tools, inference.py for AI models
│   │   ├── requirements.txt
│   │   └── binary_config.py   # [optional]
│   └── README.md
└── __init__.py
```

---

## Step 2: Create the Tool File

Every tool file has exactly 3 sections: **Data Models**, **Tool Implementation**, and uses exactly these imports.

### Complete Tool Template

```python
"""{ToolName} — brief description of what this tool does.

This module provides a standardized interface for {description of biological function}.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import ConfigDict, Field, computed_field, field_validator

from bio_programming_tools.utils.tool_io import BaseToolInput, BaseToolOutput
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig, ConfigField

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
class {ToolName}Input(BaseToolInput):
    """Input object for {ToolName}.

    Attributes:
        sequences (List[str]): Description of primary input data.
            Can be provided as:

            - A single string (e.g., ``"ATGCGT..."``)
            - A list of strings for batch processing

            Additional notes about input format or constraints.
    """

    sequences: List[str] = Field(
        description="Primary input data description"
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, v):
        """Convert single string to list."""
        if isinstance(v, str):
            return [v]
        if not v:
            raise ValueError("sequences must not be empty")
        return v


class {ToolName}Config(BaseConfig):
    """Configuration object for {ToolName}.

    Attributes:
        param1 (int): Description. Default: 4.
        param2 (float): Description. Range: (0, 1]. Default: 0.5.
    """

    # --- Primary parameters (shown in UI by default) ---
    param1: int = ConfigField(
        title="Parameter 1",
        default=4,
        ge=1,
        description="Human-readable description of this parameter",
    )
    param2: Literal["option_a", "option_b"] = ConfigField(
        title="Parameter 2",
        default="option_a",
        description="Choose between option_a and option_b",
    )

    # --- Advanced parameters (shown under "Advanced" in UI) ---
    advanced_param: float = ConfigField(
        title="Advanced Parameter",
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Parameter that users rarely need to change",
        advanced=True,
    )

    # --- Hidden parameters (not shown in UI) ---
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run on",
        hidden=True,
    )
    verbose: bool = ConfigField(
        title="Verbose",
        default=False,
        description="Whether to print status messages",
        hidden=True,
    )


class {ToolName}Output(BaseToolOutput):
    """Output from {ToolName}.

    Inherits metadata fields from BaseToolOutput: tool_id, execution_time,
    timestamp, success, warnings, errors, metadata. DO NOT redeclare these.

    Attributes:
        results (List[str]): Description of tool-specific results.
        num_results (int): Computed count of results.
    """

    # --- Tool-specific result fields only ---
    results: List[str] = Field(
        default_factory=list,
        description="Tool-specific results",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True  # Add this if using DataFrames or numpy arrays
    )

    # --- Computed properties for derived data ---
    @computed_field
    @property
    def num_results(self) -> int:
        """Total number of results."""
        return len(self.results)

    # --- Required abstract implementations ---
    @property
    def output_format_options(self) -> List[str]:
        return ["csv", "json"]

    @property
    def output_format_default(self) -> str:
        return "csv"

    def _export_output(self, export_path: str | Path, file_format: str):
        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "csv":
            # Write CSV output
            import csv
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["result"])
                for r in self.results:
                    writer.writerow([r])

        elif file_format == "json":
            import json
            with open(path, "w") as f:
                json.dump({"results": self.results}, f, indent=2)

        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================
@tool(
    key="{tool-key}",
    label="{Tool Display Label}",
    input={ToolName}Input,
    config={ToolName}Config,
    output={ToolName}Output,
    description="One-line description of what this tool does",
)
def run_{tool_name}(
    inputs: {ToolName}Input, config: {ToolName}Config
) -> {ToolName}Output:
    """Brief description of tool function.

    Longer description of what this tool does, what algorithm it uses,
    and any important biological context.

    Args:
        inputs ({ToolName}Input): Validated input containing primary data.
        config ({ToolName}Config): Validated configuration parameters.

    Returns:
        {ToolName}Output: Structured output containing results and metadata.

    Examples:
        >>> inputs = {ToolName}Input(sequences=["ATGCGT"])
        >>> config = {ToolName}Config(param1=4)
        >>> result = run_{tool_name}(inputs, config)
        >>> print(f"Found {{result.num_results}} results")

    See Also:
        - Tool GitHub: https://github.com/...
        - Paper: https://doi.org/...
    """
    # ... tool implementation here ...

    return {ToolName}Output(
        results=["result1", "result2"],
        metadata={
            "param1": config.param1,
            "param2": config.param2,
        },
    )
```

---

## Step 3: Implementation Patterns

### 3A: Standalone CPU Tool (EnvManager)

**Main tool file** — calls EnvManager with `run.py`:
```python
def run_tool_name(inputs: ToolInput, config: ToolConfig) -> ToolOutput:
    from bio_programming_tools.utils.env_manager import EnvManager

    venv_manager = EnvManager(model_name="{tool_name}")

    input_data = {
        "operation": "{operation_name}",
        "sequences": inputs.sequences,
        "param1": config.param1,
    }

    output_data = venv_manager.call_standalone_script_in_venv(
        script_path=Path(__file__).parent / "standalone" / "run.py",
        input_dict=input_data,
        device=config.device,
        verbose=config.verbose,
    )

    return ToolOutput(
        results=output_data["results"],
        metadata={"param1": config.param1},
    )
```

**standalone/run.py** (or **inference.py** for AI models) — JSON I/O entry point:
```python
"""
{ToolName} standalone runner for EnvManager venv execution.
Usage (called by EnvManager, not directly):
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

### 3B: AI Model Tool with the cloud runtime Fallback (GPU)

```python
def run_tool_name(inputs: ToolInput, config: ToolConfig) -> ToolOutput:
    from bio_programming_tools.utils import use_cloud_gpu

    if use_cloud_gpu():
        # Remote GPU execution via the cloud runtime
        logger.debug("Using the cloud runtime for {ToolName}")
        import _gpu_runtime

        ToolService = _gpu_runtime.Cls.from_name("bio-programming", "{ToolName}Service")
        result = ToolService().run.remote(
            sequences=inputs.sequences,
            param1=config.param1,
        )
    else:
        # Local GPU execution
        logger.debug("Using local GPU for {ToolName}")
        from bio_programming_tools.utils.env_manager import EnvManager

        venv_manager = EnvManager("{tool_name}")
        result = venv_manager.call_standalone_script_in_venv(
            script_path=Path(__file__).parent / "standalone" / "inference.py",
            input_dict={
                "operation": "run",
                "sequences": inputs.sequences,
                "param1": config.param1,
                "device": config.device,
            },
            device=config.device,
            verbose=config.verbose,
        )

    return ToolOutput(results=result["results"])
```

---

## Step 4: Caching

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

## Step 5: The `__init__.py` Export Chain

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
    "{ToolName}Input",
    "{ToolName}Config",
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
    "run_{tool_name}",
    "{ToolName}Input",
    "{ToolName}Config",
    "{ToolName}Output",
]
```

### Level 4: Package `__init__.py`
`bio_programming_tools/__init__.py` — this file uses `from bio_programming_tools.tools import *` so no changes needed IF the tool is properly added to `tools/__init__.py`'s `__all__`.

---

## Step 6: Write the README.md

Create `tools/{category}/{tool_name}/README.md` with:

```markdown
# {Tool Display Name}

## Overview
Brief biological context — what does this tool do and why is it useful?

## Key Parameters
- **param1**: What it controls, sensible defaults, when to change it
- **param2**: What it controls, recommended values

## Quick Start
\```python
from bio_programming_tools.tools import run_{tool_name}, {ToolName}Input, {ToolName}Config

inputs = {ToolName}Input(sequences=["ATGCGT..."])
config = {ToolName}Config(param1=4)
result = run_{tool_name}(inputs, config)
print(result.num_results)
\```

## Interpreting Results
How to interpret the output. What values mean biologically.

## References
- [Tool GitHub](https://github.com/...)
- [Paper](https://doi.org/...)
```

---

## Step 7: Write Tests

Create `tests/tool_tests/test_{tool_name}.py`:

```python
"""Tests for {ToolName} tool."""
import pytest
from bio_programming_tools.tools import run_{tool_name}, {ToolName}Input, {ToolName}Config


class Test{ToolName}:
    """Tests for {ToolName}."""

    def test_basic_execution(self):
        """Test basic tool execution with default config."""
        inputs = {ToolName}Input(sequences=["ATGCGT"])
        config = {ToolName}Config()
        result = run_{tool_name}(inputs, config)

        assert result.success
        assert result.num_results > 0
        assert result.execution_time > 0

    def test_single_string_input(self):
        """Test that single string is normalized to list."""
        inputs = {ToolName}Input(sequences="ATGCGT")
        assert isinstance(inputs.sequences, list)
        assert len(inputs.sequences) == 1

    def test_empty_input_raises(self):
        """Test that empty input raises ValidationError."""
        with pytest.raises(Exception):
            {ToolName}Input(sequences=[])

    def test_export_csv(self, tmp_path):
        """Test CSV export."""
        inputs = {ToolName}Input(sequences=["ATGCGT"])
        config = {ToolName}Config()
        result = run_{tool_name}(inputs, config)
        result.export(name="test_output", export_path=tmp_path, file_format="csv")
        assert (tmp_path / "test_output.csv").exists()
```

For GPU tools, add the marker:
```python
@pytest.mark.uses_gpu
class Test{ToolName}GPU:
    ...
```

---

## Step 8: Verify by Running the Tool

After implementing, do BOTH of the following:

### 8A: Run the tests
```bash
pytest tests/tool_tests/test_{tool_name}.py -v
```

### 8B: Run the tool directly
Write and execute a short verification script:

```python
"""Verify {tool_name} implementation."""
from bio_programming_tools.tools import run_{tool_name}, {ToolName}Input, {ToolName}Config

# Create input with realistic test data
inputs = {ToolName}Input(sequences=["ATGCGT..."])  # Use real biological data
config = {ToolName}Config()  # Default config

# Run the tool
result = run_{tool_name}(inputs, config)

# Verify
print(f"Success: {result.success}")
print(f"Execution time: {result.execution_time:.2f}s")
print(f"Results: {result.num_results}")

if not result.success:
    print(f"Errors: {result.errors}")
else:
    # Print tool-specific output to verify correctness
    print(result)
```

Run this script via Bash to confirm:
1. The tool imports correctly from `bio_programming_tools.tools`
2. Input validation works (try both single string and list)
3. The tool executes successfully
4. Output fields are populated correctly
5. Export works: `result.export(name="test", export_path="/tmp")`

If the tool requires GPU and none is available, verify at minimum that:
- The import chain works
- Input/Config/Output classes instantiate correctly
- The tool fails gracefully with a clear error about missing GPU

---

## Conventions Checklist

Before submitting, verify:

- [ ] File starts with `from __future__ import annotations`
- [ ] Uses `logging.getLogger(__name__)`, never `print()`
- [ ] Input extends `BaseToolInput`, uses `Field()` (not ConfigField)
- [ ] Config extends `BaseConfig`, uses `ConfigField()` (not bare Field)
- [ ] Output extends `BaseToolOutput`, does NOT redeclare inherited metadata fields
- [ ] Output implements `output_format_options`, `output_format_default`, `_export_output()`
- [ ] `@tool()` decorator has all 6 kwargs: key, label, input, config, output, description
- [ ] Run function signature: `def run_*(inputs: *Input, config: *Config) -> *Output`
- [ ] Run function returns Output with `metadata={}` dict of key parameters
- [ ] No try/except wrapping the tool logic — `@tool` decorator handles errors
- [ ] Google-style docstrings with Attributes (for classes) and Args/Returns/Examples (for functions)
- [ ] `__init__.py` exports at all 4 levels
- [ ] README.md in tool directory
- [ ] Tests written in `tests/tool_tests/` and passing
- [ ] Tool runs successfully end-to-end (verified via Step 8B)
- [ ] Biological coordinates are 1-indexed, inclusive (if applicable)

---

## Reference Implementations

When in doubt, read these canonical examples:

| Pattern | Example | File |
|---|---|---|
| Standalone + binary | BLAST | `tools/gene_annotation/blast/` |
| CPU standalone (run.py) | BLAST | `tools/gene_annotation/blast/standalone/run.py` |
| AI model standalone (inference.py) | ESMFold | `tools/structure_prediction/esmfold/standalone/inference.py` |
| GPU + the cloud runtime | Evo2 | `tools/causal_models/evo2/evo2_sample.py` |
| Shared data models | Inverse Folding | `tools/inverse_folding/shared_data_models.py` |
| Per-item caching | Orfipy | `tools/orf_prediction/orfipy/orfipy.py` |
| Scoring tool | Evo2 Score | `tools/causal_models/evo2/evo2_score.py` |
| Structure prediction | ESMFold | `tools/structure_prediction/esmfold/` |

Read the actual source files of these tools when implementing — they are the ground truth for conventions.
