#!/usr/bin/env python
"""
Worker bootstrap — runs inside the tool's venv as a long-running process.

Usage (invoked by PersistentWorker, not directly):
    python _worker_bootstrap.py <standalone_script_path>

Protocol (stdin → stdout, one JSON object per line):

    Request:  {"id": "abc123", "input": { ... }}
    Response: {"id": "abc123", "result": { ... }}
    Error:    {"id": "abc123", "error": "traceback text"}

The standalone script is imported as a module. Its ``__main__`` block is
skipped because we import it, not run it.  We look for a ``dispatch``
function first; if absent, we fall back to the script's original
read-input / run-operation / write-output pattern by calling the
operations directly based on the ``operation`` key in the input dict.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any


def _copy_standalone_helpers(script_path: str) -> None:
    """Copy standalone_helpers.py to the tool's standalone directory.

    Checks if 'standalone' appears in the script's absolute path. If found,
    places standalone_helpers.py in that standalone directory so it's importable by all
    scripts in that tree.

    If the script is not in a standalone directory, logs a warning and skips the copy.
    This allows tools without standalone directories to still function.

    Args:
        script_path: Path to the standalone inference.py or run.py script
    """
    script = Path(script_path).resolve()

    # Check if 'standalone' is in the path
    if "standalone" not in script.parts:
        sys.stderr.write(
            f"[worker] Warning: Script {script_path} is not in a 'standalone' directory. "
            f"Skipping standalone_helpers.py copy.\n"
        )
        return

    # Find the standalone directory in the path
    standalone_idx = script.parts.index("standalone")
    standalone_dir = Path(*script.parts[:standalone_idx + 1])

    # Determine target path for standalone_helpers.py
    target_helpers = standalone_dir / "standalone_helpers.py"

    # Find the canonical source (in utils/)
    utils_dir = Path(__file__).parent
    source_helpers = utils_dir / "standalone_helpers.py"

    if not source_helpers.exists():
        # This should never happen in a proper install, but handle gracefully
        sys.stderr.write(
            f"[worker] Warning: Could not find {source_helpers} to copy helpers\n"
        )
        return

    # Always copy to ensure it's up to date
    try:
        shutil.copy2(source_helpers, target_helpers)
        sys.stderr.write(f"[worker] Copied standalone_helpers to {target_helpers}\n")
    except Exception as exc:
        # Don't fail worker startup if helpers copy fails - tool may not need it
        sys.stderr.write(
            f"[worker] Warning: Failed to copy standalone_helpers.py: {exc}\n"
        )


def _load_module(script_path: str) -> Any:
    """Import a standalone script as a Python module."""
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
                return func(input_dict)
            raise ValueError(
                f"Cannot dispatch operation '{operation}' — no function "
                f"'{func_name}' found in {module.__name__}"
            )

        # No operation key — auto-route if there's exactly one run_* function.
        if len(run_funcs) == 1:
            func = next(iter(run_funcs.values()))
            return func(input_dict)

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


def main() -> None:
    if len(sys.argv) != 2:
        sys.stderr.write(
            f"Usage: {sys.argv[0]} <standalone_script_path>\n"
        )
        sys.exit(1)

    # Redirect stdout to stderr so tool output doesn't pollute the JSON protocol pipe
    _json_out = sys.stdout
    sys.stdout = sys.stderr

    script_path = sys.argv[1]

    # Copy standalone helpers to the tool's directory if not present
    _copy_standalone_helpers(script_path)

    module = _load_module(script_path)

    dispatch = _find_dispatch(module)
    if dispatch is None:
        dispatch = _build_legacy_dispatch(module)

    # Signal ready
    sys.stderr.write(f"[worker] ready (script={script_path})\n")
    sys.stderr.flush()

    # Main loop: read JSON requests from stdin, write JSON responses to stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            # Can't parse request — write error with no id
            error_response = {"id": None, "error": f"Invalid JSON: {exc}"}
            response_json = json.dumps(error_response, separators=(",", ":"))
            _json_out.write(f"LENGTH:{len(response_json)}\n")
            _json_out.write(response_json)
            _json_out.flush()
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
                        sys.stderr.write(
                            f"[worker] Warning: {script_path} does not have to_device() function\n"
                        )
                        result = {"success": False, "error": "to_device not supported"}
                    result = _serialize(result)
                    response = {"id": request_id, "result": result}
                elif command == "get_memory_stats":
                    # Handle memory stats request
                    if hasattr(module, "get_memory_stats"):
                        result = module.get_memory_stats()
                    else:
                        # Graceful fallback if tool doesn't support get_memory_stats
                        sys.stderr.write(
                            f"[worker] Warning: {script_path} does not have get_memory_stats() function\n"
                        )
                        result = {"available": False, "error": "get_memory_stats not supported"}
                    result = _serialize(result)
                    response = {"id": request_id, "result": result}
                else:
                    raise ValueError(f"Unknown command: {command}")
            else:
                # Normal dispatch
                result = dispatch(input_dict)
                result = _serialize(result)
                response = {"id": request_id, "result": result}
        except Exception:
            response = {"id": request_id, "error": traceback.format_exc()}

        # Length-prefixed protocol: send byte count then JSON
        # This allows libraries to output warnings/logs without breaking JSON parsing
        response_json = json.dumps(response, separators=(",", ":"))
        _json_out.write(f"LENGTH:{len(response_json)}\n")
        _json_out.write(response_json)
        _json_out.flush()


if __name__ == "__main__":
    main()
