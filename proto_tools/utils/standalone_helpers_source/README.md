# Standalone Helpers

**These files are auto-copied to each tool's `standalone/` directory at runtime.**

Do NOT import from this directory directly. The bootstrap mechanism in
`_worker_bootstrap.py` and `tool_instance.py` copies these files into each
tool's standalone environment so they are available to `inference.py`, `run.py`,
and `setup.sh` scripts.

## Layout

- **`standalone_helpers/`** — Python package of helpers, split by concern:
  - `device.py` — subprocess device env, JAX device resolution, model/params device movement
  - `memory.py` — `get_pytorch_memory_stats`, `get_jax_memory_stats`
  - `seeding.py` — `get_random_int`, `set_torch_seed`, `set_jax_seed`, `enable_jax_compilation_cache`
  - `weights.py` — `resolve_weights_dir`
  - `compression.py` — `compress_array`, `is_compressed_array` (large-array IPC wire format)
  - `__init__.py` — re-exports every public name for backward compat

Imported by standalone scripts via `from standalone_helpers import ...` (package entry point) or
`from standalone_helpers.seeding import ...` (specific submodule).

Standalone worker environments receive this copied helper package and may not
have the full `proto_tools` package importable.

- **`standalone_helpers.sh`** — Bash helper functions for `setup.sh` scripts.
  Sourced via `source standalone_helpers.sh`. Provides `proto_install_pytorch`,
  `proto_install_jax`, `proto_install_cuda_toolkit`, `proto_resolve_weights_dir`,
  and `proto_check_gated_hf_repo`.

## Editing

Edit these source files here. The copies in `tools/*/standalone/` are
overwritten on every tool invocation and are gitignored. When adding a new
helper, add it to the appropriate submodule and re-export it from
`standalone_helpers/__init__.py`.
