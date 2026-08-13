# proto-tools: context for coding agents

You are driving **proto-tools**, a library of typed bioinformatics tool wrappers
(sequence, structure, ligand, genomic, and publication/database tools). Every
tool exposes a uniform Python API with validated input/output schemas, generated
docs, citations, licenses, and an isolated execution environment so heavy or
conflicting dependencies never collide. This primer is the starting point; pull
the long-form references linked at the bottom when you need depth.

## The one pattern every tool follows

```
Input -> Config -> run_*() -> Output
```

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

`Config` is optional at the call site; the decorator supplies defaults. Every
`Output` carries standard metadata (`tool_id`, `execution_time`, `success`,
`errors`) plus tool-specific payload fields. Biological coordinates are
1-indexed and inclusive.

## Discover tools offline with the CLI

The `proto-tools` command works on a clean `pip install` with no repo checkout.
Add `--json` to any verb that returns structured data for machine-readable
output. Resolve a tool by registry key (`esm2-embedding`), run-function name
(`run_esm2_embeddings`), or module path.

| Verb | What it gives you |
|---|---|
| `proto-tools list [--category C] [--gpu/--cpu]` | Registered tools, one per line |
| `proto-tools catalog` | Tools grouped by category |
| `proto-tools docs <tool>` | Intro, applications, usage tips, license |
| `proto-tools schema <tool> [--input/--config/--output]` | JSON Schema(s) |
| `proto-tools input/config/output <tool>` | Field-level model docs |
| `proto-tools signature <tool>` | Imports, symbol names, and required input fields for the call |
| `proto-tools example-input <tool> [--as-python]` | A minimal valid `Input`, as JSON or as a runnable snippet |
| `proto-tools example <tool>` | The toolkit example notebook as markdown |

## If you are connected over MCP, not writing Python

The MCP server exposes a fixed set of tools and no Python API: the imports,
`persist()` / `get()`, and `ToolPool` below do not apply, though the
`Input -> Config -> run -> Output` shape does. Everything is reached through
these:

| Tool | Use it for |
|---|---|
| `workspace_info` | Where calls land, and whether credentials are configured. Start here if anything looks misconfigured. |
| `list_tools` | What is available, with each tool's category, one-line summary, and whether it needs a GPU. Pass `category` to narrow. |
| `search_tools` | Finding a tool by description. Returns the best matches with the score each ranked on. |
| `get_tool_schema` | The input, config, and output schemas, before a first call. |
| `get_tool_example` | A known-good example input, showing shape rather than payload. |
| `get_tool_info` | Where a tool comes from: citation, DOI, the model's own links, and the implementation. |
| `run_tool` | Running one. |
| `deploy_tool` | Deploying an app to Modal, after the user approves the spend. |

Three things worth knowing before the first call:

- **Tool keys are `<model>-<action>`** — `esmfold-prediction`, not `esmfold`.
  Several actions usually exist for one model. If a key is rejected, the reply
  lists near matches; `search_tools` resolves a name you only half know.
- **Structure inputs take a file path or an http(s) URL** where the schema shows
  an object, so a file already on disk — another tool's output, say — never has
  to be read into the call. This is not uniform: an MSA takes its content.
- **Not every tool needs deploying.** Many are answered in the server's own
  process, because they need no GPU and no environment, or cannot be deployed at
  all. Those are listed as available with a note saying so, and `run_tool`
  reports where a call actually ran in `ran_on`.

## Don't guess symbol names from the registry key

Model and run-function names come from the toolkit, not the registry key, so
`esm2-embedding` lining up with `ESM2EmbeddingsInput` is the exception rather
than the rule. `blast-create-db` exports `CreateBlastDbInput` and
`run_create_blast_db`; `mafft-align` exports `MafftInput`, not `MafftAlignInput`.

Ask instead of guessing:

```bash
proto-tools signature blast-create-db
```

```python
from proto_tools.tools.sequence_alignment.blast.create_blast_db import (
    CreateBlastDbConfig,
    CreateBlastDbInput,
    run_create_blast_db,
)

result = run_create_blast_db(
    CreateBlastDbInput(fasta=...),
    CreateBlastDbConfig(),  # optional, omit for defaults
)  # -> CreateBlastDbOutput
```

`signature` is the cheap call: it renders symbol names and required field names
only, so it costs the same few hundred bytes for every tool. `example-input`
carries real values and scales with them, which for a structure or a
model-context-length window means hundreds of KB; reach for it when you want a
payload to actually run, not when you want the names.

From Python, when you already hold a registry key, call the tool through its
spec rather than importing anything (there is no `ToolRegistry.run()`):

```python
from proto_tools import ToolRegistry

spec = ToolRegistry.get("blast-create-db")
result = spec.function(ToolRegistry.get_example_input("blast-create-db"))
```

`spec.input_model`, `spec.config_model`, and `spec.output_model` give the
classes directly, and `Model.model_fields` gives their fields.

## Keep models warm and fan out across GPUs

Loading a model is the expensive step. Reuse a warm worker across a batch:

```python
from proto_tools.utils.tool_instance import ToolInstance

with ToolInstance.persist():        # every tool called in the block stays warm
    ...                             # shuts down automatically on block exit
```

Spread one batch across every available GPU:

```python
from proto_tools.utils import ToolPool

with ToolPool(gpus="all"):          # or gpus=2, or gpus=["cuda:0", "cuda:1"]
    ...                             # work is partitioned, run in parallel, reassembled in order
```

## Go deeper: long-form references on GitHub

These are the canonical developer notes; read them directly from GitHub when you
need more than the primer above (full index:
<https://github.com/evo-design/proto-tools/tree/main/notes>):

- **Finding & calling tools** — <https://github.com/evo-design/proto-tools/blob/main/notes/finding-tools.md>
- **Tool environments** (how isolated envs build/cache on first call) — <https://github.com/evo-design/proto-tools/blob/main/notes/tool-environments.md>
- **Tool persistence** (`persist()`, `persist_tool()`, `get()`) — <https://github.com/evo-design/proto-tools/blob/main/notes/tool-persistence.md>
- **Device management** (GPU allocation, LRU eviction, CPU offload) — <https://github.com/evo-design/proto-tools/blob/main/notes/device-management.md>
- **Model taste** (choosing models/validators for design tasks) — <https://github.com/evo-design/proto-tools/blob/main/notes/model-taste.md>
- **Storage** (`PROTO_HOME`, `PROTO_MODEL_CACHE`, where weights live) — <https://github.com/evo-design/proto-tools/blob/main/notes/storage.md>
- **Error handling** (raise-by-default policy, capture mode) — <https://github.com/evo-design/proto-tools/blob/main/notes/error-handling.md>
- **Troubleshooting** (cluster-specific env/GPU/storage problems) — <https://github.com/evo-design/proto-tools/blob/main/notes/troubleshooting.md>

Runnable guides and the rendered docs site: <https://proto.evodesign.org/docs/tools/introduction>
