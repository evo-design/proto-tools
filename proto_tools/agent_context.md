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
| `proto-tools example-input <tool>` | A minimal valid `Input` |
| `proto-tools example <tool>` | The toolkit example notebook as markdown |

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
