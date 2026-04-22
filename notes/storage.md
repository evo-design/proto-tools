# Storage

Tools download model weights on first use. `PROTO_HOME` (defaults to `~/.proto/`) controls where model weights are stored.

```bash
# Recommended: add to ~/.bashrc
export PROTO_HOME=/path/to/your/proto_home
```

`PROTO_MODEL_CACHE` can override just the model weights location (defaults to `PROTO_HOME/proto_model_cache/`).

## Storage layout

Everything lives under `PROTO_HOME` regardless of install mode:

```
PROTO_HOME/                   (default: ~/.proto/)
├── proto_model_cache/        model weights (HF_HOME, TORCH_HOME, resolve_weights_dir)
├── proto_tool_envs/          micromamba-managed tool venvs
├── uv_cache/                 uv package download cache (UV_CACHE_DIR)
├── pip_cache/                pip HTTP cache (PIP_CACHE_DIR)
└── .micromamba/              micromamba binary + package cache
```

## uv / pip package caches

Tool env builds use `uv pip install` (and occasionally `pip install`) to fetch Python wheels. By default, proto-tools routes both caches under `PROTO_HOME` so all disk usage is consolidated and cleanable atomically.

- `UV_CACHE_DIR` — defaults to `PROTO_HOME/uv_cache/` (extracted archives + HTTP cache for `uv`)
- `PIP_CACHE_DIR` — defaults to `PROTO_HOME/pip_cache/` (HTTP + wheel cache for `pip`)

Both are injected by `persistent_worker._build_subprocess_env()` via `setdefault`, so any value set in your shell overrides the default. To share the cache across projects, set them explicitly:

```bash
export UV_CACHE_DIR=~/.cache/uv
export PIP_CACHE_DIR=~/.cache/pip
```

The cache and tool envs share a filesystem by default, which lets `uv` hard-link from the cache into envs (saves bulk — the extracted archive is not duplicated). Wiping `PROTO_HOME/proto_tool_envs/` preserves the cache, so the next rebuild is fast.

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

Do **not** share `PROTO_HOME` itself across users. Tool environments are user-specific and should remain per-user. Only model weights (`PROTO_MODEL_CACHE`) are safe to share.

## Per-tool override

`PROTO_{TOOL_NAME}_WEIGHTS_DIR` always wins, regardless of mode:

```bash
export PROTO_FAMPNN_WEIGHTS_DIR=/custom/path/fampnn
export PROTO_PROTENIX_WEIGHTS_DIR=/custom/path/protenix
```

## For tool authors

Non-HF tools call `resolve_weights_dir(toolkit)` from `standalone_helpers.py`:

```python
from standalone_helpers import resolve_weights_dir

weights_dir = resolve_weights_dir("my_tool")
if weights_dir:
    # Use weights_dir for model files
    ...
```

HF-based tools need no code changes; `persistent_worker.py` sets `HF_HOME` automatically.

For `setup.sh` scripts that download weights during environment setup, use the shared helper from `standalone_helpers.sh` (auto-copied):

```bash
source standalone_helpers.sh
proto_resolve_weights_dir my_tool
# $WEIGHTS_DIR is now set and the directory created
wget -q -O "$WEIGHTS_DIR/model.pt" "https://example.com/model.pt"
```

## Exceptions

- **ProteinMPNN**: Weights (~150 MB) live inside pip-installed ColabDesign. Inherently venv-local.
- **AlphaFold3**: User-provided paths (`model_dir`, `db_dir`, `sif_path`). Not managed by `PROTO_MODEL_CACHE`.
