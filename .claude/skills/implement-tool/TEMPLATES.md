# implement-tool: Code Templates

Reference file for the `implement-tool` skill. Templates are tagged with which phase or subagent consumes them.

**Placeholder glossary** (see SKILL.md for full definitions):

- `{toolkit}` — snake_case directory name (e.g., `evo2`, `pyrosetta`). **Strict** — drives the directory path.
- `{tool_key}` — kebab-case registration key (e.g., `evo2-sample`). **Strict** — passed to `@tool(key=...)`.
- `{tool_key_snake}` — snake_case form of `{tool_key}` (e.g., `evo2_sample`). **Strict** — core tool file name, `run_*` function, test file name.
- `{ToolName}` — PascalCase class-name prefix for this tool's `Input` / `Config` / `Output` (e.g., `Evo2Sample`, `ESMFoldPrediction`). **Developer's choice** — typically the PascalCase of `{tool_key}`, but pick whatever reads cleanly (e.g., `ESMFold` over `Esmfold`) as long as it's specific to this tool.
- `{tool_display_name}` — human-readable label (e.g., `"Evo 2"`).

---

## Complete Tool File Template — [Phase 2: Contract]

```python
"""{tool_display_name} {operation} tool."""

import logging
from typing import Literal

from pydantic import Field, computed_field, field_validator

from proto_tools.utils.tool_io import BaseToolInput, BaseToolOutput
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseConfig, ConfigField

logger = logging.getLogger(__name__)

# If using shared data models from the category, use labeled type aliases:
# Input:
# {ToolName}Input = SharedCategoryInput
# Config:
# {ToolName}Config = SharedCategoryConfig
# Output:
# {ToolName}Output = SharedCategoryOutput


# ============================================================================
# Data Models
# ============================================================================
class {ToolName}Input(BaseToolInput):
    """Input object for {tool_display_name}.

    Attributes:
        sequences (list[str]): Description of primary input data.
            Can be provided as:

            - A single string (e.g., ``"ATGCGT..."``)
            - A list of strings for batch processing

            Additional notes about input format or constraints.
    """

    sequences: list[str] = Field(
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

    Inherits device (default: "cpu"), verbose, and timeout from BaseConfig.
    GPU tools MUST override device default to "cuda".

    Attributes:
        param1 (int): Description. Default: 4.
        param2 (float): Description. Range: (0, 1]. Default: 0.5.
        device (str): Overridden for GPU tools. Inherited "cpu" for CPU tools.
    """

    # --- GPU tools ONLY: Override device default ---
    # GPU-enabled tools MUST set device="cuda" (DeviceManager allocates specific GPUs)
    # CPU-only tools inherit device="cpu" from BaseConfig — do NOT add a device field
    device: str = ConfigField(
        title="Device",
        default="cuda",  # REQUIRED for GPU tools (uses_gpu=True); OMIT this for CPU tools
        description="Device to run the model on (e.g., 'cuda', 'cpu')",
        hidden=True,
        include_in_key=False,
    )

    # --- Model configuration (use reload_on_change=True for worker restarts) ---
    # Parameters like model_checkpoint, model_size, etc. that require full
    # model reload should have reload_on_change=True
    model_checkpoint: str = ConfigField(
        title="Model Checkpoint",
        default="default-model",
        description="Model checkpoint to load",
        reload_on_change=True,  # Restart worker when this changes
    )

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

    # --- Batch processing (GPU tools) ---
    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Number of items to process per GPU forward pass",
        advanced=True,
    )

    # Note: verbose, timeout, and device are inherited from BaseConfig.
    # Only redeclare them if you need to override the default value.

class {ToolName}Output(BaseToolOutput):
    """Output from {ToolName}.

    Inherits metadata fields from BaseToolOutput: tool_id, execution_time,
    timestamp, success, warnings, errors, metadata. DO NOT redeclare these.

    Attributes:
        results (list[str]): Description of tool-specific results.
        num_results (int): Computed count of results.
    """

    # --- Tool-specific result fields only ---
    # All fields must be JSON-serializable Pydantic types (primitives, list, nested BaseModel).
    # Never use pd.DataFrame, numpy arrays, or arbitrary_types_allowed=True.
    # DataFrames are constructed lazily in _export_output() only.
    results: list[str] = Field(
        default_factory=list,
        description="Tool-specific results",
    )

    # --- Computed properties for derived data ---
    @computed_field
    @property
    def num_results(self) -> int:
        """Total number of results."""
        return len(self.results)

    # --- Required abstract implementations ---
    @property
    def output_format_options(self) -> list[str]:
        return ["csv", "json"]

    @property
    def output_format_default(self) -> str:
        return "csv"

    def _export_output(self, export_path: str | Path, file_format: str):
        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "csv":
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
def example_input():
    """Minimal valid input for testing and examples."""
    return {ToolName}Input(sequences=["ATGCGT"])


@tool(
    key="{tool-key}",  # kebab-case, must include action (e.g., "blast-search", "esmfold-prediction")
    label="{Tool Display Label}",  # Human-readable name for UI
    category="{category}",  # REQUIRED: Tool category (e.g., "structure_prediction", "gene_annotation")
    input_class={ToolName}Input,
    config_class={ToolName}Config,
    output_class={ToolName}Output,
    description="One-line description of what this tool does",
    uses_gpu=False,  # Set True for GPU/AI model tools (MUST also override device="cuda" in Config)
    device_count="1",  # Optional: Device count requirement ("1", "1-2", ">=1", "<=2"). Defaults to "1"
    example_input=example_input,  # Factory returning minimal valid input for parametrized tests
    # cacheable=True,  # Optional: enable wrapper cache
    # generative=True,  # Optional: diversified unseeded sampling/gradient/design outputs
)
def run_{tool_key_snake}(
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
        >>> result = run_{tool_key_snake}(inputs, config)
        >>> print(f"Found {result.num_results} results")

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

## Test Template — [Subagent 4: Tests]

Create `tests/{category}_tests/test_{tool_key_snake}.py`:

```python
"""Tests for {tool_display_name} tool."""
import pytest
from proto_tools.tools import run_{tool_key_snake}, {ToolName}Input, {ToolName}Config


class Test{ToolName}:
    """Tests for {tool_display_name}."""

    def test_basic_execution(self):
        """Test basic tool execution with default config."""
        inputs = {ToolName}Input(sequences=["ATGCGT"])
        config = {ToolName}Config()
        result = run_{tool_key_snake}(inputs, config)

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
        result = run_{tool_key_snake}(inputs, config)
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

## README.md Template — [Subagent 2: README + cite.bib]

Create `tools/{category}/{toolkit}/README.md`:

```markdown
# {Tool Display Name}

## Overview
Brief biological context — what does this tool do and why is it useful?

## When to Use This Tool

**Primary use cases:**
- Use case 1
- Use case 2

**When NOT to use this tool:**
- Anti-pattern 1
- Anti-pattern 2

## Biological Background

**What does this tool do?**
Description of the biological problem it solves.

**Why is this important?**
Real-world applications.

**Scientific foundation:**
How the algorithm/model works at a high level.

## Tool Catalog

| Tool | Input | Output | Use Case |
|------|-------|--------|----------|
| `{tool-key}` | Description | Description | Use case |

## Execution Modes

GPU/CPU requirements, memory estimates, timing.

## How It Works

Brief description of each operation.

## Input Parameters

### Operation Name (`{tool-key}`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `field1` | `Type` | Description |

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `param1` | `type` | `default` | Description |

## Output Specification

| Field | Type | Description |
|-------|------|-------------|
| `field1` | `type` | Description |

## Interpreting Results

How to interpret the output values biologically.

## Quick Start Examples

**Example 1: Basic usage**
\```python
from proto_tools.tools.{category}.{toolkit} import (
    run_{tool_key_snake}, {ToolName}Input, {ToolName}Config
)

inputs = {ToolName}Input(...)
config = {ToolName}Config(...)
result = run_{tool_key_snake}(inputs, config)
\```

## Best Practices & Gotchas

Parameter tuning guidance and common mistakes.

## References

- [Paper](https://doi.org/...)
- [GitHub](https://github.com/...)

## Related Tools

Tools often used together, alternatives.
```

---

## cite.bib Template — [Subagent 2: README + cite.bib]

Create `tools/{category}/{toolkit}/cite.bib`:

```bibtex
@article{author2024toolname,
  title={Title of the Paper},
  author={Author, First and Author, Second and others},
  journal={Journal Name},
  volume={1},
  number={1},
  pages={1--10},
  year={2024},
  publisher={Publisher Name},
  doi={10.1234/example.doi}
}
```

**Important:**
- Use the paper's DOI to find the correct BibTeX entry (most publishers provide this)
- If multiple tools in the same directory cite the same paper, they share the same `cite.bib`
- The BibTeX key format is typically `{firstauthor}{year}{toolname}` (e.g., `altschul1990blast`)

---

## Example Notebook Guidance — [Subagent 3: Example Notebook]

Create `tools/{category}/{toolkit}/examples/example.ipynb` with:

1. **Markdown title cell** with tool name, brief description, and link to paper
2. **Import cell** with exact imports from `proto_tools.tools.{category}.{toolkit}`
3. **API reference cells** with markdown tables documenting Input/Config/Output fields
4. **Execution cells** showing realistic usage with example data
5. **Export cell** demonstrating `result.export()`

Follow the pattern in existing notebooks (e.g., `tools/causal_models/evo2/examples/example.ipynb`). Key conventions:
- Use `proto-language` kernel with Python 3.12
- Include API reference tables with Field, Type, Default, Description columns
- Use realistic biological data (real sequences, not lorem ipsum)
- Show result inspection (printing key fields, accessing metrics)

---

## Verification Script Template — [Phase 4: Verify]

Write and execute a short verification script:

```python
"""Verify {tool_display_name} implementation."""
from proto_tools.tools import run_{tool_key_snake}, {ToolName}Input, {ToolName}Config

# Create input with realistic test data
inputs = {ToolName}Input(sequences=["ATGCGT..."])  # Use real biological data
config = {ToolName}Config()  # Default config

# Run the tool
result = run_{tool_key_snake}(inputs, config)

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
1. The tool imports correctly from `proto_tools.tools`
2. Input validation works (try both single string and list)
3. The tool executes successfully
4. Output fields are populated correctly
5. Export works: `result.export(name="test", export_path="/tmp")`

If the tool requires GPU and none is available, verify at minimum that:
- The import chain works
- Input/Config/Output classes instantiate correctly
- The tool fails gracefully with a clear error about missing GPU
