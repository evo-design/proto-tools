# Standalone Helpers

**These files are auto-copied to each tool's `standalone/` directory at runtime.**

Do NOT import from this directory directly. The bootstrap mechanism in
`_worker_bootstrap.py` and `tool_instance.py` copies these files into each
tool's standalone environment so they are available to `inference.py`, `run.py`,
and `setup.sh` scripts.

## Files

- **`standalone_helpers.py`** — Python helpers for device management, memory
  stats, weight directory resolution. Imported by standalone inference/run
  scripts via `from standalone_helpers import ...`.

- **`standalone_helpers.sh`** — Bash helper functions for `setup.sh` scripts.
  Sourced via `source standalone_helpers.sh`. Provides `proto_install_pytorch`,
  `proto_install_jax`, `proto_install_cuda_toolkit`, `proto_resolve_weights_dir`,
  and `proto_check_gated_hf_repo`.

## Editing

Edit these source files here. The copies in `tools/*/standalone/` are
overwritten on every tool invocation and are gitignored.
