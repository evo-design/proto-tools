# Standalone Helpers

**This directory is published to every tool subprocess; it is never copied.**

Tool environments contain the tool's dependencies but not `proto_tools`, so
`_build_subprocess_env()` puts this directory on the subprocess's `PYTHONPATH`
(for `inference.py` / `run.py`) and on its `PATH` (so `setup.sh` can
`source standalone_helpers.sh`), and exports it as
`PROTO_STANDALONE_HELPERS_DIR`. Nothing is written into the installed package,
so proto-tools works from a read-only install.

Do NOT import from this directory by its full dotted path in tool code; use the
published name, `from standalone_helpers import ...`.

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

Standalone worker environments import this package off `PYTHONPATH` and may not
have the full `proto_tools` package importable.

- **`standalone_helpers.sh`** — Bash helper functions for `setup.sh` scripts.
  Sourced via `source standalone_helpers.sh`. Provides `proto_install_pytorch`,
  `proto_install_jax`, `proto_install_cuda_toolkit`, `proto_resolve_weights_dir`,
  `proto_resolve_asset_availability`, `proto_check_gated_hf_repo`, and
  `proto_download_gdrive`.

## Editing

Edit these source files here; they take effect on the next tool invocation with
no environment rebuild. When adding a new helper, add it to the appropriate
submodule and re-export it from `standalone_helpers/__init__.py`.

Keep this directory free of anything that is not part of the published surface.
It lands on both `PATH` and `PYTHONPATH` inside every tool environment, so a
stray script or module here would leak into all of them.
