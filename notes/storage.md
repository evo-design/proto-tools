# Storage

This note covers where `proto_tools` stores model weights, tool environments, caches, and databases, and the environment variables that control those locations.

Tools download model weights on first use, and `PROTO_HOME` (defaults to `~/.proto/`) controls where those weights are stored.

```bash
# Recommended: add to ~/.bashrc
export PROTO_HOME=/path/to/your/proto_home
```

`PROTO_MODEL_CACHE` can override just the model weights location (defaults to `PROTO_HOME/proto_model_cache/`).
`PROTO_DATABASES_DIR` can override just the sequence-databases location (defaults to `PROTO_MODEL_CACHE/databases/`). See [Databases](#databases).

## Storage layout

Everything lives under `PROTO_HOME` regardless of install mode:

```
PROTO_HOME/                   (default: ~/.proto/)
├── proto_model_cache/        model weights (HF_HOME, TORCH_HOME, resolve_weights_dir)
│   ├── databases/            provisioned MMseqs2 databases (override: PROTO_DATABASES_DIR)
│   └── jax_cache/{toolkit}/  compiled JAX programs (JAX_COMPILATION_CACHE_DIR)
├── proto_tool_envs/          micromamba-managed tool venvs
├── uv_cache/                 uv package download cache (UV_CACHE_DIR)
├── pip_cache/                pip HTTP cache (PIP_CACHE_DIR)
└── .micromamba/              micromamba binary + package cache
```

## uv / pip package caches

Tool env builds use `uv pip install` (and occasionally `pip install`) to fetch Python wheels. By default, proto-tools routes both caches under `PROTO_HOME` so all disk usage is consolidated and cleanable atomically.

- `UV_CACHE_DIR` defaults to `PROTO_HOME/uv_cache/` (extracted archives + HTTP cache for `uv`)
- `PIP_CACHE_DIR` defaults to `PROTO_HOME/pip_cache/` (HTTP + wheel cache for `pip`)

Both are injected by `persistent_worker._build_subprocess_env()` via `setdefault`, so any value set in your shell overrides the default. To share the cache across projects, set them explicitly:

```bash
export UV_CACHE_DIR=~/.cache/uv
export PIP_CACHE_DIR=~/.cache/pip
```

The cache and tool envs share a filesystem by default, which lets `uv` hard-link from the cache into envs (saves bulk, because the extracted archive is not duplicated). Wiping `PROTO_HOME/proto_tool_envs/` preserves the cache, so the next rebuild is fast.

## JAX compilation cache

The first time a JAX model runs, XLA compiles it for your GPU. For large models like AlphaFold3 and AlphaGenome that takes several minutes. proto-tools saves the compiled model to disk, so the next run with the same input size, GPU, and JAX version loads it in seconds instead of compiling again.

This happens automatically for every JAX tool: `persistent_worker._build_subprocess_env()` sets `JAX_COMPILATION_CACHE_DIR` for each tool, and tools that don't use JAX ignore it. The cache is stored next to your model weights, so it survives env rebuilds, and on Modal it lives on the shared volume where every container can use it.

| `PROTO_MODEL_CACHE` | Cache location |
|---|---|
| *(unset, default)* | `{PROTO_HOME}/proto_model_cache/jax_cache/{toolkit}/` |
| `/absolute/path` | `/absolute/path/jax_cache/{toolkit}/` |
| `IN_ENV` | `{venv}/jax_cache/` |
| `NONE` | Not set, unless you set `JAX_COMPILATION_CACHE_DIR` yourself |

To put the cache somewhere else, or turn it off:

```bash
export JAX_COMPILATION_CACHE_DIR=/path/to/jax_cache   # one directory for every tool
export PROTO_JAX_COMPILATION_CACHE=0                 # disable the cache
```

A single tool can opt out on its own by setting the variable to an empty value in its `standalone/env_vars.txt`:

```text
[set]
JAX_COMPILATION_CACHE_DIR=
```

Leave the value truly empty. JAX treats an empty value as "off", but a space makes it write cache files into the working directory.

A cached model only matches the GPU type and JAX version that compiled it, so a mismatch just triggers a fresh compile, never a wrong result. That also makes the cache safe to share between users on the same model cache. It grows as you run new input sizes, GPUs, and JAX versions; you can delete `jax_cache/`, or one tool's folder inside it, at any time, and the next run recompiles.

## Modes

| Mode | HF_HOME | Non-HF weights | TORCH_HOME |
|------|---------|----------------|------------|
| *(unset, default)* | `{PROTO_HOME}/proto_model_cache/huggingface/` | `{PROTO_HOME}/proto_model_cache/{toolkit}/` | `{PROTO_HOME}/proto_model_cache/torch/` |
| `/absolute/path` | `/absolute/path/huggingface/` | `/absolute/path/{toolkit}/` | `/absolute/path/torch/` |
| `IN_ENV` | `{venv}/cache/huggingface/` | `{venv}/model_weight_cache/` | `{venv}/cache/torch/` |
| `NONE` | Parent `HF_HOME` passthrough | `{venv}/weights/` | Parent `TORCH_HOME` passthrough |

The default (`proto_model_cache/` under `PROTO_HOME`) keeps weights outside tool envs so they survive env rebuilds.

## Shared weights for teams

For teams sharing weights across collaborators, set `PROTO_MODEL_CACHE` to a shared directory while keeping `PROTO_HOME` per-user:

```bash
# Per-user: tool envs and micromamba (should NOT be shared between users,
# as different users may have different CUDA versions, library paths, etc.)
export PROTO_HOME=~/.proto

# Shared with collaborators: just model weights (safe for concurrent access;
# HuggingFace uses file locks internally to handle simultaneous downloads)
export PROTO_MODEL_CACHE=/shared/team/model_weights
```

Do **not** share `PROTO_HOME` itself across users, because tool environments are user-specific and should remain per-user. Only model weights (`PROTO_MODEL_CACHE`) are safe to share.

## Databases

Sequence databases for `mmseqs2-homology-search` (UniRef30, the ColabFold envdb, etc.) are large (tens to hundreds of GB each) and, once indexed, read-only. By default they live under `PROTO_MODEL_CACHE/databases/`, but `PROTO_DATABASES_DIR` overrides the databases root directly, so you can keep them on a separate filesystem from model weights, e.g. a high-capacity scratch volume:

```bash
# Databases on scratch; weights wherever PROTO_MODEL_CACHE points
export PROTO_DATABASES_DIR=/scratch/$USER/proto_databases
```

The override applies to **both** provisioning (`setup_databases.py` writes there) and runtime (the tool resolves datasets there), so a single export keeps them consistent, with no symlinks. Like weights, the databases root is safe to NFS-mount and share across collaborators (read-only after indexing). Provision into it with:

```bash
python -m proto_tools.tools.sequence_alignment.mmseqs2.setup_databases <dataset>   # e.g. uniref30-2302
```

`setup_databases.py --workdir <dir>` overrides the root for a single invocation instead.

## Per-tool override

`PROTO_{TOOL_NAME}_WEIGHTS_DIR` always wins, regardless of mode:

```bash
export PROTO_FAMPNN_WEIGHTS_DIR=/custom/path/fampnn
export PROTO_PROTENIX_WEIGHTS_DIR=/custom/path/protenix
```

<!-- docs:ignore start -->
## For tool authors

Non-HF tools call `resolve_weights_dir(toolkit)` from the `standalone_helpers` package:

```python
from standalone_helpers import resolve_weights_dir

weights_dir = resolve_weights_dir("my_tool")
if weights_dir:
    # Use weights_dir for model files
    ...
```

HF-based tools need no code changes. `persistent_worker.py` sets `HF_HOME` automatically.

For `setup.sh` scripts that download weights during environment setup, use the shared helper from `standalone_helpers.sh` (resolved off `PATH`):

```bash
source standalone_helpers.sh
proto_resolve_weights_dir my_tool
# $WEIGHTS_DIR is now set and the directory created
wget -q -O "$WEIGHTS_DIR/model.pt" "https://example.com/model.pt"
```

## Exceptions

- **ProteinMPNN**: Weights (~150 MB) live inside pip-installed ColabDesign. Inherently venv-local.
<!-- docs:ignore end -->

