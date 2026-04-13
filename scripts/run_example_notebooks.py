#!/usr/bin/env python3
"""scripts/run_example_notebooks.py.

Re-execute (optional) and post-process every tool example notebook under
``proto_tools/tools/*/examples/`` to strip output mime types that don't
render outside the original execution context (ipywidgets progress bars,
live Plotly/Bokeh handles, etc.). The text/plain and text/html fallbacks
stay.

Typical usage::

    # Clean every notebook without executing (fast; handles notebooks that can't re-run)
    python scripts/run_example_notebooks.py --sanitize-only

    # Execute + sanitize one notebook
    python scripts/run_example_notebooks.py --only segmasker

    # Execute + sanitize every notebook (slow; requires GPU + model weights)
    python scripts/run_example_notebooks.py --timeout 1800

    # Just list what would be processed
    python scripts/run_example_notebooks.py --dry-run

Exits non-zero if any notebook fails to execute.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent

# Mime types that require a live kernel / runtime JS state to render, and that
# consumers of a static notebook (VS Code, JupyterLab, nbviewer, GitHub) cannot
# resolve. Always paired with a text/plain or text/html fallback — strip the
# live ones, keep the fallbacks.
_STRIP_MIMES = frozenset(
    {
        "application/vnd.jupyter.widget-view+json",  # ipywidgets (tqdm, sliders, etc.)
        "application/vnd.plotly.v1+json",  # plotly-native — text/html fallback renders fine
        "application/vnd.bokehjs_exec.v0+json",  # bokeh — HTML fallback renders fine
    }
)


def discover_notebooks(only: str | None) -> list[Path]:
    """Return every ``example.ipynb`` under ``proto_tools/tools``, filtered by ``only``."""
    notebooks = sorted(REPO_ROOT.glob("proto_tools/tools/**/examples/example.ipynb"))
    if only:
        notebooks = [n for n in notebooks if only in str(n.relative_to(REPO_ROOT))]
    return notebooks


def execute_notebook(path: Path, timeout: int) -> tuple[bool, str]:
    """Run the notebook in place via ``jupyter nbconvert --execute --inplace``.

    Returns:
        tuple[bool, str]: ``(success, message)`` — ``"ok"`` on success,
        or the last line of stderr/stdout on failure.
    """
    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 60,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    if result.returncode != 0:
        last_line = (result.stderr or result.stdout or "nbconvert exit non-zero").strip().split("\n")[-1]
        return False, last_line
    return True, "ok"


def sanitize(path: Path) -> tuple[int, set[str]]:
    """Strip non-renderable mime types from cell outputs; drop orphan ``widgets`` metadata.

    Returns:
        tuple[int, set[str]]: ``(count, seen_mimes)`` — count of mime entries stripped,
        and the full set of mime types that were encountered (for reporting).
    """
    nb = json.loads(path.read_text())
    stripped = 0
    seen_mimes: set[str] = set()

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            data = out.get("data")
            if not isinstance(data, dict):
                continue
            for mime in list(data.keys()):
                seen_mimes.add(mime)
                if mime in _STRIP_MIMES:
                    del data[mime]
                    stripped += 1

    # Drop notebook-level ``widgets`` metadata — refers to widget state we no longer persist.
    if "widgets" in nb.get("metadata", {}):
        del nb["metadata"]["widgets"]

    if stripped:
        path.write_text(json.dumps(nb, indent=1) + "\n")
    return stripped, seen_mimes


def _rel(path: Path) -> str:
    """Return ``path`` as a repo-relative string for user-facing messages."""
    return str(path.relative_to(REPO_ROOT))


def main() -> int:
    """Execute and/or sanitize example notebooks."""
    ap = argparse.ArgumentParser(
        description="Re-execute tool example notebooks and strip non-renderable widget outputs.",
    )
    ap.add_argument(
        "--only",
        default=None,
        help="Substring filter on notebook path (e.g. 'segmasker' or 'structure_prediction/alphafold2')",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Per-notebook execution timeout in seconds (default: 1800). Cell timeout matches this.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List the discovered notebooks without executing or sanitizing",
    )
    ap.add_argument(
        "--sanitize-only",
        action="store_true",
        help="Skip execution; only strip stale widget-view outputs. Fast; handles notebooks that can't re-run.",
    )
    args = ap.parse_args()

    notebooks = discover_notebooks(args.only)
    if not notebooks:
        print("No notebooks matched.", flush=True)
        return 1

    print(f"Found {len(notebooks)} notebook(s):", flush=True)
    for n in notebooks:
        print(f"  {_rel(n)}", flush=True)
    print(flush=True)

    if args.dry_run:
        return 0

    mode = "sanitize-only" if args.sanitize_only else f"execute+sanitize (timeout {args.timeout}s/notebook)"
    print(f"Mode: {mode}", flush=True)
    print(flush=True)

    failures: list[tuple[Path, str]] = []
    all_mimes: set[str] = set()
    total_stripped = 0

    progress = tqdm(notebooks, desc="Processing", unit="nb", file=sys.stderr)
    for nb_path in progress:
        rel = _rel(nb_path)
        progress.set_postfix_str(nb_path.parent.parent.name)

        if args.sanitize_only:
            ok, msg = True, "ok"
        else:
            ok, msg = execute_notebook(nb_path, args.timeout)

        if ok:
            stripped, mimes = sanitize(nb_path)
            total_stripped += stripped
            all_mimes.update(mimes)
            print(f"  ok    {rel}  (stripped {stripped})", flush=True)
        else:
            failures.append((nb_path, msg))
            print(f"  FAIL  {rel}: {msg}", flush=True)

    progress.close()

    print(flush=True)
    print(
        f"Summary: {len(notebooks) - len(failures)}/{len(notebooks)} notebooks processed; "
        f"{total_stripped} widget views stripped.",
        flush=True,
    )
    if all_mimes:
        print(f"Mime types seen across outputs: {sorted(all_mimes)}", flush=True)

    if failures:
        print(flush=True)
        print(f"FAILURES ({len(failures)}):", flush=True)
        for nb_path, msg in failures:
            print(f"  {_rel(nb_path)}: {msg}", flush=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
