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

    # --- Secondary parameters ---
    secondary_param: float = ConfigField(
        title="Secondary Parameter",
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Parameter that users rarely need to change",
    )

    # --- Batch processing (GPU tools) ---
    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Number of items to process per GPU forward pass",
    )

    # --- Mutually exclusive fields (XOR group) ---
    # Tag siblings with the same `xor_group` slug + add a `@model_validator` to
    # enforce at runtime. See SKILL.md "Mutual-exclusion fields" for the pattern.
    # exact_count: int | None = ConfigField(default=None, ge=1, xor_group="amount", ...)
    # fraction: float | None = ConfigField(default=None, gt=0, le=1, xor_group="amount", ...)

    # Note: verbose, timeout, and device are inherited from BaseConfig.
    # Only redeclare them if you need to override the default value.

    # --- ToolPool resource declarations (properties, not fields) ---
    # `cpus_per_instance` defaults to None on BaseConfig: ToolPool dispatches
    # a single direct call and pool.cpus is ignored. KEEP this default for
    # most CPU tools — short per-item compute, internal threading, or
    # network IO all lose more than they gain from fan-out.
    #
    # OPT IN by overriding to a positive integer ONLY when:
    #   - per-call work is heavy enough to amortize subprocess startup
    #     (each worker holds its own venv in RAM)
    #   - the tool is single-threaded (or N-threaded) per call
    #   - items are embarrassingly parallel (no shared state across items)
    # Canonical opt-in: PyRosetta (heavy init, multi-second per pose).
    #
    # @property
    # def cpus_per_instance(self) -> int | None:
    #     """Opt in to ToolPool CPU fan-out — {reason: heavy init, single-threaded per call}."""
    #     return 1
    #
    # `gpus_per_instance` is auto-derived from the device string. Override
    # only when GPU need is decoupled from device (e.g. a use_gpu flag, or
    # a model-variant-dependent count); see Mmseqs2HomologySearchConfig for the pattern.

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
    # stochastic=True,  # Optional: outputs depend on config.seed (sampling/gradient/design/diffusion)
    # metrics_class={ToolName}Metrics,  # Optional: scalar metric container (plDDT, perplexity, etc.)
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

Flat functions only — no test classes (see CLAUDE.md, **Test Conventions**).

```python
"""Tests for {tool_display_name} tool."""
import pytest

from proto_tools.tools import run_{tool_key_snake}, {ToolName}Input, {ToolName}Config


def test_basic_execution():
    """Basic tool execution with default config."""
    inputs = {ToolName}Input(sequences=["ATGCGT"])
    config = {ToolName}Config()
    result = run_{tool_key_snake}(inputs, config)

    assert result.success
    assert result.num_results > 0
    assert result.execution_time > 0


def test_single_string_input():
    """Single string is normalized to list."""
    inputs = {ToolName}Input(sequences="ATGCGT")
    assert isinstance(inputs.sequences, list)
    assert len(inputs.sequences) == 1


def test_empty_input_raises():
    """Empty input raises ValidationError."""
    with pytest.raises(Exception):
        {ToolName}Input(sequences=[])


def test_export_csv(tmp_path):
    """CSV export writes the expected file."""
    inputs = {ToolName}Input(sequences=["ATGCGT"])
    config = {ToolName}Config()
    result = run_{tool_key_snake}(inputs, config)
    result.export(name="test_output", export_path=tmp_path, file_format="csv")
    assert (tmp_path / "test_output.csv").exists()


@pytest.mark.uses_gpu
def test_gpu_execution():
    """GPU dispatch test — auto-skipped if no GPU is visible."""
    inputs = {ToolName}Input(sequences=["ATGCGT"])
    config = {ToolName}Config(device="cuda")
    result = run_{tool_key_snake}(inputs, config)
    assert result.success
```

---

## README.md Template — [Subagent 2: README + cite.bib + license.yaml + links.yaml]

Create `tools/{category}/{toolkit}/README.md` using the structured template. Schemas / configs / output specs are auto-generated from Pydantic field descriptions, so the README focuses on prose — biology, when to use, and per-tool tips. The four H2 sections are canonical: `Overview`, `Background`, `Tools`, `Toolkit Notes`.

```markdown
<a href="https://bio-pro.mintlify.app/tools/{category-kebab}/{toolkit-kebab}"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,…" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,…&logoColor=white" alt="Use on Proto (coming soon)">

# {Toolkit Display Name}

> [!NOTE]
> **License:** {one-line license summary with link to upstream COPYING/LICENSE.}
>
> {For tools that federate over multiple sources, list each bundled dependency on its own bullet inside the same callout, with its license and a link.}

## Overview

[2–3 sentences: who built it, what it does, why useful. Link the upstream repo on first mention.]

## Background

[1–3 paragraphs: deeper biological and algorithmic context. Cite the paper(s) inline as `([First Author, Year](DOI))`. Link key concepts to Wikipedia or canonical references on first mention. Explain the scientific foundation at a level a senior researcher new to the subfield can follow.]

### Learning Resources

- [Link title](url) (Author or org) - one-line description of why this resource is useful.
- [Link title](url) (Author or org) - one-line description.

## Tools

### {Tool Display Name} (`{tool-key}`)

[One sentence on what the tool does and what it returns.]

#### Applications

[1–2 sentences on the typical research question this tool answers and how it fits in a pipeline.]

#### Usage Tips

- **Critical-knob name explained in bold.** Followed by a short explanation including units, defaults, and the most common failure mode.
- **Second tip in bold.** Explanation including any non-obvious interaction with other config fields.

[Repeat the H3 + H4 block per tool in the toolkit.]

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every {Toolkit Display Name} tool in this toolkit (`{tool-key-1}`, `{tool-key-2}`, ...).

- **First toolkit-wide note in bold.** Explanation covering execution mode (CPU/GPU, SIMD, etc.), install footprint, or memory characteristics that apply to every tool in the toolkit.
- **Second note in bold.** Explanation of cross-tool behaviour (shared resources, thresholds, threading model, etc.).
```

Conventions:
- Use the canonical four H2 sections (`Overview`, `Background`, `Tools`, `Toolkit Notes`) verbatim — `get_readme_sections()` parses on these exact names.
- `### Learning Resources` is optional under `## Background` and is excluded from `get_tool_docs()` by default — use it for user-facing explainers (blogs, talks, course pages) rather than agent-readable spec.
- Per-tool H3 heading must be `{Tool Display Name} (\`{tool-key}\`)` — the parser extracts the tool key from the backticks.
- `#### Applications` and `#### Usage Tips` are the canonical H4s under each tool.
- Mintlify URL pattern is `/tools/{category-kebab}/{toolkit-kebab}` — both segments kebab-cased (e.g. `gene-annotation/promoter-calculator`, `database-retrieval/sequence-fetch`).
- Read a recently-rewritten toolkit (e.g. `gene_annotation/pyhmmer/README.md`) to copy the exact badge SVG data and Toolkit Notes badge row.

---

## cite.bib Template — [Subagent 2: README + cite.bib + license.yaml + links.yaml]

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

## license.yaml Template — [Subagent 2: README + cite.bib + license.yaml + links.yaml]

Create `tools/{category}/{toolkit}/license.yaml`. **Required** — `test_license_consistency.py` enforces existence and schema. Verify the license from the upstream repo's `LICENSE`/`COPYING`; do not guess.

```yaml
code:
  spdx: MIT                         # SPDX id from the allowlist, or "Custom (<name>)"
  url: https://github.com/org/repo/blob/main/LICENSE
commercial_use: 'yes'               # 'yes' | 'no' | 'restricted'
redistribution: true
attribution_required: false
notes: One line on anything non-obvious (e.g. bundled deps under a different license).
last_updated: '2026-05-27'          # ISO YYYY-MM-DD; today's date
```

Schema rules (enforced by `test_license_consistency.py`):
- **Required top-level keys:** `code`, `commercial_use`, `redistribution`, `last_updated`.
- **`code` / `weights` / `data` blocks** each require `spdx` + `url`.
- **SPDX allowlist** (else use `spdx: "Custom (<name>)"`): `Apache-2.0`, `MIT`, `BSD-2-Clause`, `BSD-3-Clause`, `GPL-3.0`, `LGPL-3.0`, `MPL-2.0`, `CC0-1.0`, `CC-BY-4.0`, `CC-BY-SA-4.0`, `CC-BY-NC-4.0`, `CC-BY-NC-SA-4.0`, `AGPL-3.0`, `ISC`, `Unlicense`.
- **Text placement:** SPDX-allowlisted licenses must NOT inline a `text:` field — the canonical copy lives at `proto_tools/tools/_licenses/{spdx}.txt` (must exist; add it if introducing a new SPDX id). `Custom (...)` licenses MUST inline `text:`.
- **`commercial_use`** ∈ `{'yes', 'no', 'restricted'}`. **`weights.access`** (optional) ∈ `{'hf-gated', 'request'}`.
- Add an optional `weights:` block when the model weights carry a different license than the code, and a `data:` block (with `name`) for API-wrapper toolkits that fetch an external dataset.

---

## links.yaml Template — [Subagent 2: README + cite.bib + license.yaml + links.yaml]

Create `tools/{category}/{toolkit}/links.yaml`. **Required.** Canonical upstream links, each verified to resolve. Include only keys that apply:

```yaml
github: https://github.com/org/repo
website: https://tool.example.org          # optional homepage/docs
paper: https://doi.org/10.1234/example     # or `preprint:` for an arXiv/bioRxiv URL
huggingface: https://huggingface.co/org/model   # optional
organizations:                             # optional list of affiliated orgs/labs
  - Example Lab
```

Observed keys across the repo (use the right one, don't invent): `github`, `website`, `organizations`, `image`, `huggingface`, `paper`, `preprint`, `zenodo`, `model`, `blog`.

---

## Example Notebook Guidance — [Subagent 3: Example Notebook]

Create `tools/{category}/{toolkit}/examples/example.ipynb` with:

1. **Markdown title cell** with tool name, brief description, and link to paper
2. **Import cell** with exact imports from `proto_tools.tools.{category}.{toolkit}`
3. **API reference cells** that call `display_api_reference("{tool_key}", "input"/"config"/"output", "run_{tool_key_snake}")` to auto-render Input/Config/Output tables — never hand-write these tables (hand-written tables drift from the schema and require manual `\|` escaping for `X | None` types, which renders inconsistently across viewers)
4. **Execution cells** showing realistic usage with example data
5. **Export cell** demonstrating `result.export()`

Follow the pattern in existing notebooks (e.g., `tools/causal_models/evo2/examples/example.ipynb`). Key conventions:
- Kernelspec must be `{"name": "python3", "display_name": "proto-tools"}` — every example notebook ships with this exact metadata so `run_example_notebooks.py` resolves the kernel from the `proto-tools` conda env. Never set custom names like `proto-language`; they'll fail with `NoSuchKernel` outside the original author's machine.
- Use `display_api_reference()` for every Input/Config/Output table (see step 3 above) — `proto_tools.utils.notebook_docs` introspects the live Pydantic schema, so the table stays in sync with the code automatically.
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
