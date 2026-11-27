# Standalone Helpers

**This directory is made available to every tool subprocess without copying it.**

Tool environments contain the tool's dependencies but not `proto_tools`, so
`_build_subprocess_env()` puts this directory on the subprocess's `PYTHONPATH`
(for `inference.py` / `run.py`) and on its `PATH` (so `setup.sh` can
`source standalone_helpers.sh`), and exports it as
`PROTO_STANDALONE_HELPERS_DIR`. Nothing is written into the installed package,
so proto-tools works from a read-only install.

In tool code, import from the published name, `from standalone_helpers import ...`.
Do not use the full dotted path to this directory.

## Layout

- **`standalone_helpers/`** — Python package of helpers, split by concern:
  - `device.py` — subprocess device env, JAX device resolution, model/params device movement
  - `memory.py` — `get_pytorch_memory_stats`, `get_jax_memory_stats`
  - `msa_usage.py` — `count_a3m_rows`, `report_msa_rows_loaded` (log MSA rows a backend loaded; warn when supplied rows never reach it)
  - `seeding.py` — `get_random_int`, `set_torch_seed`, `set_jax_seed`
  - `weights.py` — `resolve_weights_dir`
  - `compression.py` — `compress_array`, `is_compressed_array` (large-array IPC wire format)
  - `recovery.py` — `is_checkpoint_corruption_error`, `retry_on_checkpoint_corruption` (re-download truncated checkpoints)
  - `__init__.py` — re-exports every public name for backward compat

Imported by standalone scripts via `from standalone_helpers import ...` (package entry point) or
`from standalone_helpers.seeding import ...` (specific submodule).

Standalone worker environments import this package off `PYTHONPATH` and may not
have the full `proto_tools` package importable.

- **`standalone_helpers.sh`** — Bash helper functions for `setup.sh` scripts.
  Sourced via `source standalone_helpers.sh`. Provides `proto_install_pytorch`,
  `proto_install_jax`, `proto_install_cuda_toolkit`, `proto_resolve_weights_dir`,
  `proto_check_gated_hf_repo`, `proto_download_gdrive`, `proto_download_verified`,
  `proto_download`, `proto_download_archive`, and `proto_is_truncated_zip`.

## Editing

Edit the source files in this directory. Changes take effect on the next tool invocation with
no environment rebuild. When adding a new helper, add it to the appropriate
submodule and re-export it from `standalone_helpers/__init__.py`.

Keep only published helpers in this directory. It is added to both `PATH` and
`PYTHONPATH` inside every tool environment, so any script or module placed here
becomes available to all tools.
