# AGENTS.md

Runtime guide for agents that consume proto-tools (LLM agents, MCP servers, custom dispatch harnesses). For development conventions (extending the framework, test patterns, error-handling policy), see [CLAUDE.md](CLAUDE.md).

`proto_tools` exposes a uniform API around every registered tool: discovery, structured prose documentation, Pydantic schemas, example inputs, citations, and the run functions. Every entry point is a classmethod on `ToolRegistry`, so a single import covers the full surface:

```python
from proto_tools.tools.tool_registry import ToolRegistry
```

Returned values are Pydantic v2 `BaseModel` instances or plain JSON-serializable types. Every result serializes via `.model_dump()` / `.model_dump_json()`, so the same API serves in-process Python callers and MCP / wire-protocol consumers.

## Primary entry point

`ToolRegistry.get_tool_docs(tool)` returns a single `ToolReadmeEntry` containing the tool-specific intro paragraph, `Applications`, `Usage Tips`, the toolkit's `Toolkit Notes`, and the parsed `license.yaml`. The toolkit notes and license are attached by default, so this one call provides the full conceptual context for a tool, including gating (`license["weights"]["access"]`) and usage terms.

```python
entry = ToolRegistry.get_tool_docs("esm2-embedding")
entry.intro
entry.usage_tips
entry.toolkit_notes  # toolkit-wide guidance, attached by default
entry.license        # parsed license.yaml incl. weights.access, attached by default
```

## Command-line interface

The same surface is available from the shell:

```bash
proto-tools list --category masked_models
proto-tools docs esm2-embedding              # full per-tool docs as text
proto-tools docs esm2-embedding --json       # structured payload
proto-tools docs run_esm2_embeddings         # any identifier form resolves
proto-tools schema esm2-embedding --input    # JSON Schema only
proto-tools example-input esm2-embedding     # minimal valid Input as JSON
proto-tools catalog --json                   # full registry, grouped by category
```

`proto-tools --help` lists every verb. `python -m proto_tools` is an equivalent entry point when the `proto-tools` script is shadowed in the environment.

Output defaults to human-readable text. The `--json` flag on any verb that returns structured data emits machine-readable Pydantic dumps suitable for subprocess callers. Error paths exit non-zero with the same `ValueError` messages the Python API raises (for example, `"Identifier 'esm2' is ambiguous; specify one of: …"`).

## Identifier shapes

Most APIs accept multiple identifier forms. The following table specifies what resolves where.

| Form | Example | Toolkit-level APIs | Per-tool APIs |
|---|---|---|---|
| Registry key | `"esm2-embedding"` | yes | yes |
| Run-function name (with `run_`) | `"run_esm2_embeddings"` | yes | yes |
| Run-function name (without `run_`) | `"esm2_embeddings"` | yes | yes |
| Docs path | `"masked-models/esm2"` | yes | yes if single-tool toolkit |
| Toolkit directory name | `"esm2"` | yes | raises (ambiguous; multiple tools) |

- Toolkit-level APIs (`get_readme`, `get_readme_section`, `get_readme_sections`) resolve to the toolkit directory. `"esm2"` resolves because the request targets the toolkit's README.
- Per-tool APIs (`get_tool_docs`, `get_input_doc`, `get_config_doc`, `get_output_doc`) must pinpoint one tool. A multi-tool toolkit identifier such as `"esm2"` raises `ValueError` with a `"specify one of: esm2-embedding, esm2-gradient, esm2-sample, esm2-score"` message. Single-tool toolkits (for example `"esmfold"`, which registers only `esmfold-prediction`) resolve.

## Discovery

```python
ToolRegistry.list_all()                          # every ToolSpec
ToolRegistry.count()                             # number of registered tools

ToolRegistry.list_categories()                   # ["binder_design", "causal_models", ...]
ToolRegistry.list_by_category("masked_models")   # tools in that category, key-sorted
ToolRegistry.catalog()                           # {category: [ToolSpec]} for the full registry

ToolRegistry.list_gpu_tools()                    # GPU-required subset
ToolRegistry.list_cpu_tools()                    # CPU-only subset
```

`ToolSpec` carries structured metadata for triaging a candidate without reading prose: `key`, `label`, `category`, `description`, `uses_gpu`, `device_count`, the Pydantic `input_model` / `config_model` / `output_model` classes, and the `function` reference.

## Reading prose docs

The four README extractors return progressively narrower slices of the same content.

```python
# Raw README text, verbatim.
text = ToolRegistry.get_readme("esm2-embedding")

# One named H2 section by exact heading text. Returns None if absent.
background = ToolRegistry.get_readme_section("esm2-embedding", "Background")

# Structured view of the whole README.
sections = ToolRegistry.get_readme_sections("esm2-embedding")
sections.title              # "ESM2"
sections.overview           # body of ## Overview
sections.background         # body of ## Background (excludes the optional
                            # ### Learning Resources subsection by default;
                            # it links to user-facing explainers such as blog
                            # posts and talks that agents do not need. Pass
                            # include_learning_resources=True to keep it)
sections.tools              # list[ToolReadmeEntry], one per registered tool
sections.toolkit_notes      # body of ## Toolkit Notes
sections.qc_pending         # True if the README still has the QC TODO callout
sections.other_sections     # any non-canonical H2s

# One tool's H3 subsection. The most common per-tool query.
entry = ToolRegistry.get_tool_docs("esm2-embedding")
entry.key                   # "esm2-embedding"
entry.label                 # "ESM2 Embeddings"
entry.intro                 # paragraph between the H3 and the first H4
entry.applications          # body of #### Applications
entry.usage_tips            # body of #### Usage Tips
entry.toolkit_notes         # ## Toolkit Notes body (attached by default)
entry.license               # parsed license.yaml incl. weights.access (attached by default)

# Exclude the toolkit notes for tool-specific content only:
ToolRegistry.get_tool_docs("esm2-embedding", include_toolkit_notes=False)
```

Shields.io badge HTML in the source README (for example the Toolkit Notes guide badges) is collapsed to plain `[label](url)` markdown links in the extracted text. The raw README returned by `get_readme` preserves the badge HTML unchanged.

For the Pydantic models:

```python
input_doc = ToolRegistry.get_input_doc("esm2-embedding")
config_doc = ToolRegistry.get_config_doc("esm2-embedding")
output_doc = ToolRegistry.get_output_doc("esm2-embedding")

config_doc.name             # "ESM2EmbeddingsConfig"
config_doc.docstring        # cleaned class docstring
for f in config_doc.fields:
    f.name, f.type_str, f.default, f.description, f.required
```

## Schemas and example inputs

For validating or constructing a call from JSON:

```python
ToolRegistry.get_schemas("esm2-embedding")        # {"inputs": ..., "config": ..., "output": ...}
ToolRegistry.get_input_schema("esm2-embedding")
ToolRegistry.get_config_schema("esm2-embedding")
ToolRegistry.get_output_schema("esm2-embedding")

example_input = ToolRegistry.get_example_input("esm2-embedding")
# -> ESM2EmbeddingsInput(sequences=["MKTL"])
```

Each `example_input` is a minimal, valid `Input` suitable for verifying a roundtrip or seeding a new call.

## Citation, links, license, docs URL

```python
ToolRegistry.get_citation("esm2-embedding")               # BibTeX string or None
ToolRegistry.get_doi("esm2-embedding")                    # DOI string or None
ToolRegistry.get_links("esm2-embedding")                  # {"github": "...", "huggingface": "...", ...}
ToolRegistry.get_license("esm2-embedding")                # parsed license.yaml
ToolRegistry.get_weights_access("esm2-embedding")         # "open" | "hf-gated" | "request"
ToolRegistry.get_docs_url("esm2-embedding")               # https://bio-pro.mintlify.app/tools/...
ToolRegistry.get_example_notebook_path("esm2-embedding")  # local Path to examples/example.ipynb
```

### Gated weights

Before calling a tool that loads model weights, check how the weights are
obtained:

```python
ToolRegistry.get_weights_access("esm3-embedding")   # "open" | "hf-gated" | "request"
```

`get_weights_access` normalizes the nested `license.yaml` `weights.access`
field (absent => `"open"`) so there is one value to branch on. The same value
is available raw as `get_license(...)["weights"]["access"]`, and via the CLI
(`proto-tools access <tool>`). The values:

- `"hf-gated"`: weights are behind a gated HuggingFace repo. The user must
  accept the provider's terms and set `HF_TOKEN`, or the tool raises before
  loading. (e.g. ESM3, AlphaGenome)
- `"request"`: weights are not publicly distributed and must be obtained from
  the provider out of band. (e.g. AlphaFold3)

`commercial_use` (`"yes"` / `"no"` / `"restricted"`) and `attribution_required`
(bool) carry the usage terms. The same restricted toolkits are also listed in
the repo-root README's "Gated model access" table.

## Calling a tool

Every tool follows the same pattern:

```
Input -> Config -> run_*() -> Output
```

The Pydantic Input and Config classes are available on the spec; the run function is the callable.

```python
from proto_tools.tools.masked_models.esm2 import (
    ESM2EmbeddingsConfig,
    ESM2EmbeddingsInput,
    run_esm2_embeddings,
)

# Equivalent lookup via the registry:
spec = ToolRegistry.get("esm2-embedding")
ESM2EmbeddingsInput = spec.input_model
ESM2EmbeddingsConfig = spec.config_model
run_esm2_embeddings = spec.function

result = run_esm2_embeddings(
    ESM2EmbeddingsInput(sequences=["MKTLIIA..."]),
    ESM2EmbeddingsConfig(model_checkpoint="esm2_t33_650M_UR50D"),
)
result.success            # bool
result.execution_time     # float, seconds
result.errors             # list[str] when success=False
result.results            # tool-specific payload
```

`Config` is optional at the call site; the `@tool` wrapper supplies defaults. Output models inherit standardized metadata fields (`tool_id`, `execution_time`, `success`, `errors`) in addition to the tool-specific payload.

## JSON and wire-protocol consumers

Every `ToolRegistry` API returns Pydantic v2 `BaseModel` instances or plain JSON-serializable types:

```python
ToolRegistry.get_tool_docs("esm2-embedding").model_dump_json()
ToolRegistry.get_config_doc("esm2-embedding").model_dump_json()
ToolRegistry.get_schemas("esm2-embedding")           # already plain dicts
```

Output models do not set `arbitrary_types_allowed=True` and contain no DataFrames or numpy arrays; every field is a primitive, a `list`, or a nested `BaseModel`. JSON Schema generation is well-defined across the entire surface.

## Caveats

- `qc_pending=True` indicates a README that has not yet been migrated to the structured template. `sections.tools` is empty in that case (the parser cannot find the canonical `## Tools` H3 layout), and `toolkit_notes` is empty. See [issue #743](https://github.com/evo-design/proto-tools/issues/743) for migration progress; both fields populate automatically as toolkits migrate.
- Tool execution is sandboxed. `run_*()` dispatches into an isolated micromamba environment per call by default (cold weights load, fresh subprocess). For batched or repeated calls, wrap the tool with `ToolInstance.persist()`; see the [Tool Persistence guide](https://bio-pro.mintlify.app/tools/guides/tool-persistence).
- GPU tools default to `device="cuda"`. Inspect `spec.uses_gpu` before dispatching, or use `list_cpu_tools()` to filter to CPU-only candidates.
- `config.seed` controls reproducibility for stochastic tools. Tools registered with `stochastic=True` skip the cache when the seed is unset; set an explicit `seed` in `Config` for reproducible output. The framework wires the seed through automatically for iterable dispatches.

## Reference

| Topic | Location |
|---|---|
| Python API of the doc extractors | [`proto_tools/utils/tool_docs.py`](proto_tools/utils/tool_docs.py) |
| Registry surface | [`proto_tools/tools/tool_registry.py`](proto_tools/tools/tool_registry.py) |
| Tool execution at runtime | [`notes/tool-environments.md`](notes/tool-environments.md), [`proto_tools/utils/tool_instance.py`](proto_tools/utils/tool_instance.py) |
| Rendered docs for any tool | https://bio-pro.mintlify.app/tools |
| Contribution conventions | [CLAUDE.md](CLAUDE.md) |
