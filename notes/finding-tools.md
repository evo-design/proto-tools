# Finding Tools

This note covers how to discover, inspect, and call every registered tool through `ToolRegistry` and the `proto-tools` CLI, namely discovery, prose documentation, Pydantic schemas, example inputs, citations, links, license and access metadata, and run functions. The same API supports in-process Python callers, scripts, notebooks, MCP/wire consumers, and the `proto-tools` CLI.

## Python Entry Point

```python
from proto_tools.tools.tool_registry import ToolRegistry
```

Returned values are Pydantic v2 `BaseModel` instances or plain JSON-serializable types. Outputs should round-trip through `.model_dump()` or `.model_dump_json()`.

## CLI and MCP

The CLI verbs `list`, `search`, `schema`, `example`, `info`, `run` and `workspace` mirror the MCP tools `list_tools`, `search_tools`, `get_tool_schema`, `get_tool_example`, `get_tool_info`, `run_tool` and `workspace_info`. The CLI, the local MCP server, and the hosted MCP server all call the same functions in `proto_tools/mcp/tools.py`, so they return the same results.

```bash
proto-tools search "predict a protein structure"
proto-tools list --category masked_models
proto-tools schema esm2-embedding
proto-tools example esm2-embedding --as-python
proto-tools info esm3-embedding              # links, citation, DOI, license, weights access
proto-tools run esm2-embedding --example
proto-tools run esm2-embedding --inputs @inputs.json --config '{"model_checkpoint": "esm2_t6_8M_UR50D"}'
proto-tools workspace --device modal
```

Mirrored verbs answer for this machine unless given `--device modal` or `--device proto`. `list` and `search` print text unless given `--json`; the others print the MCP payload as JSON and exit 1 when it reports `"ok": false`, including an unknown key, which comes back with `did_you_mean` suggestions.

`python -m proto_tools` is equivalent when the `proto-tools` executable is shadowed.

### CLI Only

These developer docs views have no MCP counterpart.

```bash
proto-tools catalog --json
proto-tools docs esm2-embedding
proto-tools docs esm2-embedding --json
proto-tools input esm2-embedding
proto-tools signature esm2-embedding
proto-tools notebook esm2-embedding
```

## Identifier Resolution

Most registry APIs accept multiple identifier forms.

| Form | Example | Toolkit-level APIs | Per-tool APIs |
|---|---|---|---|
| Registry key | `"esm2-embedding"` | yes | yes |
| Run-function name with `run_` | `"run_esm2_embeddings"` | yes | yes |
| Run-function name without `run_` | `"esm2_embeddings"` | yes | yes |
| Docs path | `"masked-models/esm2"` | yes | yes if single-tool toolkit |
| Toolkit directory name | `"esm2"` | yes | raises if ambiguous |

Toolkit-level README APIs resolve to the toolkit directory, while per-tool APIs must identify one registered operation. A multi-tool toolkit name raises `ValueError` with the valid tool keys.

## Discovery

```python
ToolRegistry.list_all()
ToolRegistry.count()
ToolRegistry.list_categories()
ToolRegistry.list_by_category("masked_models")
ToolRegistry.catalog()
ToolRegistry.list_gpu_tools()
ToolRegistry.list_cpu_tools()
ToolRegistry.list_local_cpu_tools()
```

`ToolRegistry.list_all()` and related list methods return `ToolSpec` objects, not string keys. To test membership or build a set of tool names, use `{spec.key for spec in ToolRegistry.list_all()}`, and do not call `set(ToolRegistry.list_all())`, because `ToolSpec` objects are Pydantic models and are not hashable.

Each `ToolSpec` carries structured metadata: registry key, label, category, description, device requirements, Pydantic input/config/output model classes, run function, iterable fields, cache behavior, and source path. `spec.toolkit_dir` is the directory of the defining file. `spec.is_external` is true when that file is outside `proto_tools/tools/` (a downstream package, a test file, or a notebook). External tools are registered when their module is imported, so they appear in `list_all()` only after that import; see `external-toolkits.md`.

## Documentation Extraction

Prefer structured doc APIs over ad hoc README parsing.

```python
ToolRegistry.get_readme("esm2-embedding")
ToolRegistry.get_readme_section("esm2-embedding", "Background")
ToolRegistry.get_readme_sections("esm2-embedding")
ToolRegistry.get_tool_docs("esm2-embedding")
ToolRegistry.get_input_doc("esm2-embedding")
ToolRegistry.get_config_doc("esm2-embedding")
ToolRegistry.get_output_doc("esm2-embedding")
```

`get_tool_docs()` returns one registered tool's README section plus toolkit notes and parsed license metadata by default. Pass `include_toolkit_notes=False` when only tool-specific prose is needed.

## Schemas and Examples

```python
ToolRegistry.get_schemas("esm2-embedding")
ToolRegistry.get_input_schema("esm2-embedding")
ToolRegistry.get_config_schema("esm2-embedding")
ToolRegistry.get_output_schema("esm2-embedding")
ToolRegistry.get_example_input("esm2-embedding")
ToolRegistry.get_example_config("miranda-scan")
```

`example_input()` values are minimal valid `Input` objects, useful for smoke tests, notebooks, and script templates. They carry real payloads, so for structure tools and model-context-length sequence tools they run to hundreds of KB; that size is inherent (`BorzoiInput` requires exactly 524,288 bp), not a fixture that could be trimmed. When you only need the symbol names and required field names, use `proto-tools signature <tool>`, which renders a fixed few hundred bytes for every tool.

A few examples cannot run on a default config, because the tool takes part of its request on the config: miranda's microRNA queries, or the reference genome a variant scorer reads. Those tools declare `example_config=`, a factory returning the config values the example needs. `get_example_config()` returns them, or `{}` for every other tool. `run_tool(use_example=True)` merges them under whatever config the caller passes, field by field.

## Citation, Links, License, and Access

```python
ToolRegistry.get_citation("esm2-embedding")
ToolRegistry.get_doi("esm2-embedding")
ToolRegistry.get_links("esm2-embedding")
ToolRegistry.get_license("esm2-embedding")
ToolRegistry.get_weights_access("esm3-embedding")
ToolRegistry.get_docs_url("esm2-embedding")
ToolRegistry.get_example_notebook_path("esm2-embedding")
```

`get_weights_access()` normalizes `license.yaml` into one of:

- `"open"`: weights are available without an additional access step.
- `"hf-gated"`: the user must accept provider terms and set `HF_TOKEN`.
- `"request"`: weights must be obtained from the provider out of band.

Check access before dispatching tools that load model weights.

`"open"` describes how the weights are *obtained*, not how they may be *used*.
A tool can download its weights with no gate and still carry restrictive terms —
`alphafold3` fetches its parameters from a public Google URL but its weights
license bars commercial use and redistribution. For usage rights read
`get_license()` (`commercial_use`, `redistribution`, `weights.text`), not
`get_weights_access()`.

## Calling a Tool

Every registered tool follows the same shape, where the `Input` and `Config` together feed the `run_*()` call that produces an `Output`:

<img src="assets/finding-tools/call-shape.svg" alt="Input and Config together feed run_*(), which produces Output." width="520">

```python
from proto_tools.tools.masked_models.esm2 import (
    ESM2EmbeddingsConfig,
    ESM2EmbeddingsInput,
    run_esm2_embeddings,
)

result = run_esm2_embeddings(
    ESM2EmbeddingsInput(sequences=["MKTLIIA..."]),
    ESM2EmbeddingsConfig(model_checkpoint="esm2_t33_650M_UR50D"),
)
```

Equivalent lookup through the registry:

```python
spec = ToolRegistry.get("esm2-embedding")
Input = spec.input_model
Config = spec.config_model
run_tool = spec.function

result = run_tool(Input(sequences=["MKTLIIA..."]), Config())
```

`Config` is optional at the public call site, since the decorator supplies defaults. Output models inherit standard metadata (`tool_id`, `execution_time`, `success`, `errors`) plus tool-specific payload fields.

## Persistence and Devices

Tool calls dispatch into isolated environments by default when a toolkit has a `standalone/` directory. For repeated or batched calls, use persistence or tool pools so models and environments stay warm. See `notes/tool-environments.md` for environment setup and device movement, and the guides for runtime examples.

GPU tools default to `device="cuda"` when their config supports device selection. Before dispatch, inspect `ToolSpec.uses_gpu`, list CPU/GPU subsets through the registry, and check storage and access requirements.

## JSON and Wire Consumers

Registry docs, schemas, example inputs, and outputs are Pydantic or JSON-serializable objects:

```python
ToolRegistry.get_tool_docs("esm2-embedding").model_dump_json()
ToolRegistry.get_config_doc("esm2-embedding").model_dump_json()
ToolRegistry.get_schemas("esm2-embedding")
```

Output fields must remain primitives, lists, dictionaries, or nested Pydantic models so JSON Schema generation and wire-protocol consumers stay reliable.
