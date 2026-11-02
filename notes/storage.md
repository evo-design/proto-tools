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

# Shared with collaborators: just model weights (safe for concurrent access:
# HuggingFace and the setup.sh download helpers lock simultaneous downloads)
export PROTO_MODEL_CACHE=/shared/team/model_weights
```

Concurrent setups that need the same file (two jobs, or two toolkits that share weights) take turns: the first downloads, the rest wait and then find it verified. The `setup.sh` helpers use `flock`, which works across nodes on filesystems that support it; this was tested across nodes on Sherlock's Lustre `$SCRATCH`. Without `flock(1)`, as on macOS, downloads run unlocked.

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

Use the shared helpers rather than writing download or path code in a tool:

- `require_weights_dir(name)`: like `resolve_weights_dir`, but raises instead of returning `None`. Use it instead of falling back to a temp directory.
- `download_weights(url, dest, sha256=..., md5=...)`: downloads once under a file lock, verifies the checksum, and renames into place, so concurrent workers on a shared cache never see a partial file. It refuses non-HTTPS URLs and sends `HF_TOKEN` only to `huggingface.co`. It is the Python counterpart of `proto_download_verified`.
- `export_weights_dir(env_var, name, prepend=False)`: points a library's own location variable (`FOUNDRY_CHECKPOINT_DIRS`, `CHAI_DOWNLOADS_DIR`, `PROTENIX_ROOT_DIR`) at the weights directory. `prepend=True` keeps an existing search path after it.

```python
from standalone_helpers import download_weights, export_weights_dir, require_weights_dir

export_weights_dir("FOUNDRY_CHECKPOINT_DIRS", "rf3", prepend=True)
ckpt = download_weights(URL, Path(require_weights_dir("my_tool")) / "model.pt", sha256=SHA256)
```

For `setup.sh` scripts that download weights during environment setup, use the shared helpers from `standalone_helpers.sh` (resolved off `PATH`):

```bash
source standalone_helpers.sh
proto_resolve_weights_dir my_tool
# $WEIGHTS_DIR is now set and the directory created
proto_download_verified "https://example.com/model.pt" "$WEIGHTS_DIR/model.pt" "$MODEL_SHA256"
```

`proto_download_verified` (checksum known), `proto_download` (no checksum), and `proto_download_archive` (tar into a directory) each lock their target, download into a temp file of their own, and move it into place only when complete, so concurrent setups sharing the cache never see or clobber a partial file. A file that is already present skips the lock, so read-only caches keep working. Wrap any other step that writes shared files in `proto_with_lock <lock_file> <command...>`; name the lock `.<file>.download.lock` beside a file or `.<dir>.extract.lock` beside a directory, matching the Python helpers, so bash and Python callers for the same target wait for each other.

Waiting has no fixed deadline, so a large download on a slow link is waited out however long it takes:

- `PROTO_LOCK_STALL_SECONDS` (default 600): a waiter gives up only when the holder's download (its temp file, or the directory being extracted) has not grown for this long. `proto_with_lock --progress <path>` gives custom steps the same behavior; without `--progress`, a waiter waits until the lock is free.
- `PROTO_DOWNLOAD_STALL_SECONDS` (default 300): the helpers' `curl` aborts a transfer that receives nothing for this long, which releases the lock before waiters give up, so a dead connection can't hold the cache. A slow transfer that is still receiving data is never aborted.
- `PROTO_LOCK_TIMEOUT_SECONDS` (default 0, off): an optional hard cap on the total wait.

## Exceptions

- **ProteinMPNN**: Weights (~150 MB) live inside pip-installed ColabDesign. Inherently venv-local.
<!-- docs:ignore end -->

