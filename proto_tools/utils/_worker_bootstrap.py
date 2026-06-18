#!/usr/bin/env python
"""proto_tools/utils/_worker_bootstrap.py.

Usage:
    python _worker_bootstrap.py <standalone_script_path>

Protocol:

    Request:  {"id": "abc123", "input": { ... }}
    Response: {"id": "abc123", "result": { ... }}
    Error:    {"id": "abc123", "error": "traceback text"}

The standalone script is imported as a module. Its ``__main__`` block is
skipped because we import it, not run it.  We look for a ``dispatch``
function first; if absent, we fall back to the script's original
read-input / run-operation / write-output pattern by calling the
operations directly based on the ``operation`` key in the input dict.
"""

import contextlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

# Threshold above which response payloads spill to a temp file instead of the pipe (Linux pipe buffer is ~1 MB; 100 MB leaves ample headroom).
_FILE_FALLBACK_THRESHOLD = 100_000_000

# Commands accepted by the worker dispatch loop; keep in sync with the if/elif chain below.
_VALID_COMMANDS = ("to_device", "get_memory_stats")


def _copy_standalone_helpers(script_path: str) -> None:
    """Copy standalone helpers (Python package and shell file) to the tool's standalone directory.

    Checks if 'standalone' appears in the script's absolute path. If found,
    copies the ``standalone_helpers/`` package and ``standalone_helpers.sh``
    from ``utils/standalone_helpers_source/`` into that standalone directory.

    Args:
        script_path (str): Path to the standalone inference.py or run.py script
    """
    script = Path(script_path).resolve()

    if "standalone" not in script.parts:
        sys.stderr.write(
            f"[worker] Warning: Script {script_path} is not in a 'standalone' directory. "
            f"Skipping standalone helpers copy.\n"
        )
        return

    standalone_idx = script.parts.index("standalone")
    standalone_dir = Path(*script.parts[: standalone_idx + 1])

    # Source directory: utils/standalone_helpers_source/
    helpers_dir = Path(__file__).parent / "standalone_helpers_source"

    for name in ("standalone_helpers", "standalone_helpers.sh"):
        source = helpers_dir / name
        target = standalone_dir / name
        if not source.exists():
            continue
        # Remove any stale single-file standalone_helpers.py copy so it doesn't shadow the package on sys.path.
        if source.is_dir():
            stale_py = standalone_dir / f"{name}.py"
            if stale_py.exists():
                try:
                    stale_py.unlink()
                except OSError as exc:
                    sys.stderr.write(f"[worker] Warning: Failed to remove stale {stale_py}: {exc}\n")
        try:
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target)
        except Exception as exc:
            sys.stderr.write(f"[worker] Warning: Failed to copy {name}: {exc}\n")


def _remove_bootstrap_dir_from_sys_path() -> None:
    """Prevent the bootstrap directory from shadowing standalone dependencies.

    Persistent workers execute this file as a script, so Python places
    ``proto_tools/utils`` on ``sys.path``. That directory contains generic
    module names such as ``progress.py``; leaving it importable can shadow
    third-party packages inside isolated standalone tool environments.
    """
    bootstrap_dir = Path(__file__).resolve().parent
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != bootstrap_dir]


def _prepend_standalone_dir_to_sys_path(script_path: str) -> None:
    """Idempotently put the script's parent dir on sys.path so ``standalone_helpers`` resolves."""
    standalone_dir = str(Path(script_path).resolve().parent)
    if standalone_dir not in sys.path:
        sys.path.insert(0, standalone_dir)


def _install_subprocess_logging_bridge() -> None:
    """Trigger ``standalone_helpers.proto_logging.install`` for this subprocess.

    Imports ``standalone_helpers``; the package ``__init__`` gates ``install()``
    on ``TOOL_VENV_PATH`` so this is a no-op when the bootstrap is imported by
    parent-side code. Caller must have placed the standalone dir on ``sys.path``
    via :func:`_prepend_standalone_dir_to_sys_path` first.

    Failures here degrade the worker to "untagged stderr only": the worker still
    runs, but ``logger.info(..., update_status=True)`` calls won't reach the
    parent. We write a clear warning to stderr so the failure is visible at
    verbose level >= 3 and surfaces in crash context buffers.
    """
    try:
        import standalone_helpers  # noqa: F401  # import side effect: install()
    except Exception as exc:
        sys.stderr.write(f"[worker] logging bridge install failed: {type(exc).__name__}: {exc}\n")


def _load_module(script_path: str) -> Any:
    """Import a standalone script as a Python module.

    Caller must have placed the standalone dir on ``sys.path`` via
    :func:`_prepend_standalone_dir_to_sys_path` first.
    """
    path = Path(script_path).resolve()
    spec = importlib.util.spec_from_file_location("_standalone_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_standalone_module"] = module
    spec.loader.exec_module(module)
    return module


def _find_dispatch(module: Any) -> Any | None:
    """Return the module's ``dispatch`` function, or None."""
    return getattr(module, "dispatch", None)


def _build_legacy_dispatch(module: Any) -> Any:
    """Build a dispatch function from the module's ``run_{operation}`` functions.

    Most standalone scripts define top-level functions named like
    ``run_local_blast``.  We route by the ``operation`` key in the input dict.

    When there is exactly one ``run_*`` function and no ``operation`` key,
    we auto-route to it as a convenience for simple single-operation scripts.

    Args:
        module (Any): The imported standalone script module containing ``run_*`` functions.
    """
    # Pre-scan for run_* functions so we can auto-route single-function modules.
    run_funcs = {
        name: getattr(module, name)
        for name in dir(module)
        if name.startswith("run_") and callable(getattr(module, name))
    }

    def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
        """Route input_dict to the right run_{operation} function in the module."""
        operation = input_dict.get("operation")

        if operation is not None:
            func_name = f"run_{operation}"
            func = run_funcs.get(func_name)
            if func is not None:
                return func(input_dict)  # type: ignore[no-any-return]
            raise ValueError(
                f"Cannot dispatch operation {operation!r}: no function {func_name!r} found in "
                f"{module.__name__}; valid: {sorted(run_funcs) or ['(none)']}"
            )

        # No operation key; auto-route if there's exactly one run_* function.
        if len(run_funcs) == 1:
            func = next(iter(run_funcs.values()))
            return func(input_dict)  # type: ignore[no-any-return]

        available = ", ".join(sorted(run_funcs)) or "(none)"
        raise ValueError(
            f"Input dict must contain an 'operation' key for legacy dispatch "
            f"(module has {len(run_funcs)} run_* functions: {available})"
        )

    return dispatch


def _serialize(value: Any) -> Any:
    """Recursively serialize tensors, numpy arrays, etc. to JSON-safe types."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _send_response(json_out: Any, response_json: str) -> None:
    """Send a JSON response to the parent process.

    Small payloads use the PROTO_LENGTH-prefixed pipe protocol.  When the
    serialized JSON exceeds ``_FILE_FALLBACK_THRESHOLD`` bytes, the payload is
    written to a temporary file and a ``PROTO_FILE:<path>`` header is sent instead.
    This avoids pipe deadlocks on large responses.

    Args:
        json_out (Any): Writable file object for the JSON protocol pipe (stdout).
        response_json (str): Serialized JSON string to send.
    """
    if len(response_json) > _FILE_FALLBACK_THRESHOLD:
        fd, path = tempfile.mkstemp(suffix=".json", prefix="proto_worker_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(response_json)
        except Exception:
            # Clean up on write failure
            with contextlib.suppress(OSError):
                os.unlink(path)
            raise
        json_out.write(f"PROTO_FILE:{path}\n")
        json_out.flush()
    else:
        json_out.write(f"PROTO_LENGTH:{len(response_json)}\n")
        json_out.write(response_json)
        json_out.flush()


def main() -> None:
    if len(sys.argv) != 2:
        sys.stderr.write(f"Usage: {sys.argv[0]} <standalone_script_path>\n")
        sys.exit(1)

    # Redirect stdout to stderr so tool output doesn't pollute the JSON protocol pipe
    _json_out = sys.stdout
    sys.stdout = sys.stderr

    script_path = sys.argv[1]

    # Copy standalone helpers to the tool's directory if not present
    _copy_standalone_helpers(script_path)
    _remove_bootstrap_dir_from_sys_path()

    # Put the standalone dir on sys.path once; both the bridge install and the module load assume this.
    _prepend_standalone_dir_to_sys_path(script_path)

    # Install the worker logging bridge (via standalone_helpers) before loading the standalone module.
    _install_subprocess_logging_bridge()

    module = _load_module(script_path)

    dispatch = _find_dispatch(module)
    if dispatch is None:
        dispatch = _build_legacy_dispatch(module)

    # Signal ready
    sys.stderr.write(f"[worker] ready (script={script_path})\n")
    sys.stderr.flush()

    # Main loop: read JSON requests from stdin, write JSON responses to stdout
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            # Can't parse request; write error with no id
            error_response = {"id": None, "error": f"Invalid JSON: {exc}"}
            response_json = json.dumps(error_response, separators=(",", ":"))
            _send_response(_json_out, response_json)
            continue

        request_id = request.get("id")
        input_dict = request.get("input", {})

        try:
            # Check for special commands (to_device, etc.)
            if "command" in input_dict:
                command = input_dict["command"]
                if command == "to_device":
                    # Handle device move request
                    device = input_dict.get("device", "cpu")
                    if hasattr(module, "to_device"):
                        result = module.to_device(device)
                    else:
                        # Graceful fallback if tool doesn't support to_device
                        sys.stderr.write(f"[worker] Warning: {script_path} does not have to_device() function\n")
                        result = {"success": False, "error": "to_device not supported"}
                    result = _serialize(result)
                    response = {"id": request_id, "result": result}
                elif command == "get_memory_stats":
                    # Handle memory stats request
                    if hasattr(module, "get_memory_stats"):
                        result = module.get_memory_stats()
                    else:
                        # Graceful fallback if tool doesn't support get_memory_stats
                        sys.stderr.write(f"[worker] Warning: {script_path} does not have get_memory_stats() function\n")
                        result = {"available": False, "error": "get_memory_stats not supported"}
                    result = _serialize(result)
                    response = {"id": request_id, "result": result}
                else:
                    raise ValueError(f"Unknown command: {command!r}; valid: {list(_VALID_COMMANDS)}")
            else:
                # Normal dispatch
                result = dispatch(input_dict)
                result = _serialize(result)
                response = {"id": request_id, "result": result}
        except Exception:
            response = {"id": request_id, "error": traceback.format_exc()}

        # Length-prefixed protocol (or file fallback for large payloads)
        response_json = json.dumps(response, separators=(",", ":"))
        _send_response(_json_out, response_json)


if __name__ == "__main__":
    main()
