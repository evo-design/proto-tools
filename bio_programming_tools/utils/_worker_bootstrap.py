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
import sys
import traceback
from pathlib import Path
from typing import Any


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

    script_path = sys.argv[1]
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
            sys.stdout.write(json.dumps(error_response, separators=(",", ":")) + "\n")
            sys.stdout.flush()
            continue

        request_id = request.get("id")
        input_dict = request.get("input", {})

        try:
            result = dispatch(input_dict)
            result = _serialize(result)
            response = {"id": request_id, "result": result}
        except Exception:
            response = {"id": request_id, "error": traceback.format_exc()}

        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
