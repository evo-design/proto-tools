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
import getpass
import json
import os
import re
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


# Text-bearing output mime types that may carry machine- or user-specific paths
# and identifiers worth redacting before committing.
_TEXT_MIMES = frozenset({"text/plain", "text/html", "text/markdown"})


def _ignored_warning_substrings() -> tuple[str, ...]:
    """Warning substrings the registry suppresses at runtime, mirrored for sanitizing.

    Notebooks executed before a substring was added to the registry's ignore list
    carry the warning baked into saved stderr output. Re-using the same list lets
    ``--sanitize-only`` scrub them without a full GPU re-run.
    """
    from proto_tools.tools.tool_registry import IGNORED_WARNING_SUBSTRINGS

    return tuple(IGNORED_WARNING_SUBSTRINGS)


def _scrub_warning_lines(text: object, ignored: tuple[str, ...]) -> tuple[object, int]:
    """Drop stderr stream lines containing a runtime-ignored warning substring.

    Returns ``(new_value, removed)`` preserving the original ``str``/``list[str]`` shape.
    """
    lines = text if isinstance(text, list) else str(text).splitlines(keepends=True)
    kept = [ln for ln in lines if not any(sub in str(ln) for sub in ignored)]
    removed = len(lines) - len(kept)
    if isinstance(text, str):
        return "".join(kept), removed
    return kept, removed


def _build_redaction_rules() -> list[tuple[re.Pattern[str], str]]:
    """Build ``(pattern, replacement)`` pairs for redacting identifiers.

    The rules strip machine- and user-specific identifiers from notebook outputs
    so they don't leak into the public repo. Covers absolute paths that descend
    into this repo (rewritten repo-relative), any user's home directory, the
    running user's own home and username, and the author field embedded in
    generated structure files (CIF/PDB).
    """
    repo = re.escape(REPO_ROOT.name)
    rules: list[tuple[re.Pattern[str], str]] = [
        # Absolute path descending into this repo -> repo-relative. The lookbehind
        # leaves URLs like https://github.com/org/<repo>/ untouched.
        (re.compile(r"(?<![\w:/])(?:/[\w.+-]+)+/" + repo + r"/"), REPO_ROOT.name + "/"),
        # Any user's home directory.
        (re.compile(r"/home/[^/\s\"']+"), "/home/user"),
        (re.compile(r"/Users/[^/\s\"']+"), "/Users/user"),
        # Author field written into generated CIF/PDB files.
        (re.compile(r"(_entry\.author\s+)\S+"), r"\1anonymous"),
    ]
    # The running user's own home (covers non-standard cluster homes) and name.
    home = os.path.expanduser("~")
    if home and home not in ("/", "/home", "/home/user"):
        rules.append((re.compile(re.escape(home)), "~"))
    user = getpass.getuser()
    if user and len(user) >= 3:
        rules.append((re.compile(r"\b" + re.escape(user) + r"\b"), "user"))
    return rules


_REDACTION_RULES = _build_redaction_rules()


def _redact_text(text: str) -> tuple[str, int]:
    """Apply every redaction rule to a string. Returns ``(new_text, count)``."""
    count = 0
    for pattern, replacement in _REDACTION_RULES:
        text, n = pattern.subn(replacement, text)
        count += n
    return text, count


def _redact_field(value: object) -> tuple[object, int]:
    """Redact a notebook text field that is a ``str`` or ``list[str]``."""
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, list):
        new_list: list[object] = []
        count = 0
        for item in value:
            if isinstance(item, str):
                red, n = _redact_text(item)
                new_list.append(red)
                count += n
            else:
                new_list.append(item)
        return new_list, count
    return value, 0


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


def sanitize(path: Path) -> tuple[int, int, int, set[str]]:
    """Sanitize one notebook's cell outputs in place for committing.

    Strips non-renderable mime types, scrubs stderr lines for warnings the registry
    suppresses at runtime, redacts machine/user-specific identifiers from cell
    outputs, and drops orphan ``widgets`` metadata.

    Returns:
        tuple[int, int, int, set[str]]: ``(stripped, scrubbed, redacted, seen_mimes)`` —
        mime entries stripped, stderr warning lines scrubbed, identifier redactions
        applied, and the full set of mime types encountered (for reporting).
    """
    nb = json.loads(path.read_text())
    ignored = _ignored_warning_substrings()
    stripped = 0
    scrubbed = 0
    redacted = 0
    seen_mimes: set[str] = set()

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        kept_outputs = []
        for out in cell.get("outputs", []):
            # Drop stderr lines for warnings the registry ignores at runtime; if the
            # whole output was noise, drop it rather than leave an empty stream.
            if out.get("output_type") == "stream" and out.get("name") == "stderr" and "text" in out:
                out["text"], n = _scrub_warning_lines(out["text"], ignored)
                scrubbed += n
                if not out["text"]:
                    continue
            # Stream text (stdout/stderr) and error tracebacks are plain text.
            if "text" in out:
                out["text"], n = _redact_field(out["text"])
                redacted += n
            if "traceback" in out:
                out["traceback"], n = _redact_field(out["traceback"])
                redacted += n
            data = out.get("data")
            if isinstance(data, dict):
                for mime in list(data.keys()):
                    seen_mimes.add(mime)
                    if mime in _STRIP_MIMES:
                        del data[mime]
                        stripped += 1
                    elif mime in _TEXT_MIMES:
                        data[mime], n = _redact_field(data[mime])
                        redacted += n
            kept_outputs.append(out)
        cell["outputs"] = kept_outputs

    # Drop notebook-level ``widgets`` metadata — refers to widget state we no longer persist.
    if "widgets" in nb.get("metadata", {}):
        del nb["metadata"]["widgets"]

    if stripped or scrubbed or redacted:
        # ensure_ascii=False keeps literal UTF-8 (em dashes, emoji, arrows), matching
        # nbformat's writer; the default would escape them and churn every notebook.
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return stripped, scrubbed, redacted, seen_mimes


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
    total_scrubbed = 0
    total_redacted = 0

    progress = tqdm(notebooks, desc="Processing", unit="nb", file=sys.stderr)
    for nb_path in progress:
        rel = _rel(nb_path)
        progress.set_postfix_str(nb_path.parent.parent.name)

        if args.sanitize_only:
            ok, msg = True, "ok"
        else:
            ok, msg = execute_notebook(nb_path, args.timeout)

        if ok:
            stripped, scrubbed, redacted, mimes = sanitize(nb_path)
            total_stripped += stripped
            total_scrubbed += scrubbed
            total_redacted += redacted
            all_mimes.update(mimes)
            print(f"  ok    {rel}  (stripped {stripped}, scrubbed {scrubbed}, redacted {redacted})", flush=True)
        else:
            failures.append((nb_path, msg))
            print(f"  FAIL  {rel}: {msg}", flush=True)

    progress.close()

    print(flush=True)
    print(
        f"Summary: {len(notebooks) - len(failures)}/{len(notebooks)} notebooks processed; "
        f"{total_stripped} widget views stripped, {total_scrubbed} warning lines scrubbed, "
        f"{total_redacted} identifiers redacted.",
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
