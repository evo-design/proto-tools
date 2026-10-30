#!/usr/bin/env python3
"""scripts/run_example_notebooks.py.

Re-execute (optional) and post-process every example notebook named
``example.ipynb`` under ``proto_tools/`` (tool and entity examples alike) to
strip output mime types that don't render outside the original execution
context (ipywidgets progress bars, live Plotly/Bokeh handles, etc.). The
text/plain and text/html fallbacks stay.

py3Dmol viewers are the exception: their text/html is a script that GitHub and
other static viewers strip, leaving a "3Dmol.js failed to load" box. Each one is
rendered in headless Chromium and replaced with an ``image/png`` snapshot.
Re-running the cell in Jupyter restores the interactive viewer. Snapshots need
Playwright (``pip install playwright && playwright install --with-deps chromium``)
(``--with-deps`` also installs the system fonts legend labels need) and network
access to the 3Dmol.js CDN; without Playwright the viewers are left as-is with a
warning. Pass ``--no-snapshots`` to skip them.

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
import base64
import getpass
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

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

# Marker py3Dmol's generated HTML uses for its viewer div; the digits name the JS viewer global.
_3DMOL_VIEWER_RE = re.compile(r"3dmolviewer_(\d+)")


def _join(value: object) -> str:
    """Join a notebook ``str | list[str]`` field into one string."""
    return "".join(value) if isinstance(value, list) else str(value)


def _3dmol_html(data: dict[str, Any]) -> str | None:
    """Return the py3Dmol viewer HTML in an output bundle, or ``None`` if it has none."""
    html = _join(data.get("text/html", ""))
    return html if _3DMOL_VIEWER_RE.search(html) else None


class Snapshotter:
    """Render py3Dmol viewer HTML to PNG in a shared headless Chromium.

    WebGL runs on SwiftShader, so no GPU or display is needed. The browser starts
    on the first render and stays open until ``close()``.
    """

    def __init__(self, timeout_ms: int = 30_000) -> None:
        """Configure the per-viewer render timeout; the browser starts lazily."""
        self._timeout_ms = timeout_ms
        self._playwright: Any = None
        self._browser: Any = None

    @staticmethod
    def available() -> bool:
        """Whether Playwright is importable."""
        try:
            import playwright.sync_api  # noqa: F401
        except ImportError:
            return False
        return True

    def render(self, html: str) -> bytes:
        """Screenshot the viewer (plus any legend wrapped around it) as PNG bytes."""
        if self._browser is None:
            from playwright.sync_api import sync_playwright

            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(
                args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
            )
        match = _3DMOL_VIEWER_RE.search(html)
        if match is None:
            raise ValueError("HTML has no py3Dmol viewer div")
        page = self._browser.new_page()
        try:
            page.set_content(f"<html><body style='margin:0'>{html}</body></html>")
            # py3Dmol sets the viewer global and renders in one callback; two frames let it composite.
            page.wait_for_function(
                f"() => typeof viewer_{match.group(1)} !== 'undefined' && viewer_{match.group(1)} !== null",
                timeout=self._timeout_ms,
            )
            page.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
            # The first top-level div is the viewer itself, or the wrapper holding viewer + legend.
            png: bytes = page.locator("body > div").first.screenshot(timeout=self._timeout_ms)
        finally:
            page.close()
        return png

    def close(self) -> None:
        """Shut down the browser if it was started."""
        if self._browser is not None:
            self._browser.close()
            self._playwright.stop()
            self._browser = self._playwright = None


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
    """Return every ``example.ipynb`` under ``proto_tools``, filtered by ``only``."""
    notebooks = sorted(REPO_ROOT.glob("proto_tools/**/example.ipynb"))
    if only:
        notebooks = [n for n in notebooks if only in str(n.relative_to(REPO_ROOT))]
    return notebooks


def execute_notebook(path: Path, timeout: int) -> tuple[bool, str]:
    """Run the notebook in place via ``jupyter nbconvert --execute --inplace``.

    Pinned to the interpreter running this script, in two places, because both resolve through
    ``PATH`` otherwise and a notebook then executes against whichever environment happens to come
    first — importing a different proto-tools than the one being tested, and failing on tools that
    checkout does not have:

    * ``jupyter`` itself, hence ``sys.executable -m jupyter`` rather than the bare command.
    * the kernel, whose ``kernel.json`` says ``"argv": ["python", ...]`` with no path, so
      ``PATH`` carries this interpreter's directory for the subprocess to resolve against.

    Returns:
        tuple[bool, str]: ``(success, message)`` — ``"ok"`` on success,
        or the last line of stderr/stdout on failure.
    """
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        str(path),
    ]
    env = {**os.environ, "PATH": os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")])}
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 60,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    if result.returncode != 0:
        last_line = (result.stderr or result.stdout or "nbconvert exit non-zero").strip().split("\n")[-1]
        return False, last_line
    return True, "ok"


def _snapshot_3dmol(data: dict[str, Any], snapshotter: Snapshotter, where: str) -> bool:
    """Replace a py3Dmol output bundle's HTML with a PNG snapshot, in place.

    Only ``image/png`` and any existing ``text/plain`` are kept: static viewers would
    otherwise prefer the HTML and show its load-failure box. A failed render leaves
    the bundle untouched and logs why.

    Returns:
        bool: Whether the bundle was replaced.
    """
    html = _3dmol_html(data)
    if html is None:
        return False
    try:
        png = snapshotter.render(html)
    except Exception as exc:
        print(f"  WARN  {where}: py3Dmol snapshot failed, leaving HTML: {exc}", flush=True)
        return False
    for mime in list(data):
        if mime != "text/plain":
            del data[mime]
    data["image/png"] = base64.b64encode(png).decode("ascii")
    return True


def sanitize(path: Path, snapshotter: Snapshotter | None = None) -> tuple[int, int, int, int, set[str]]:
    """Sanitize one notebook's cell outputs in place for committing.

    Strips non-renderable mime types, scrubs stderr lines for warnings the registry
    suppresses at runtime, redacts machine/user-specific identifiers from cell
    outputs, and drops orphan ``widgets`` metadata. With a ``snapshotter``, py3Dmol
    viewers are replaced by PNG snapshots.

    Returns:
        tuple[int, int, int, int, set[str]]: ``(stripped, scrubbed, redacted, snapshots,
        seen_mimes)`` — mime entries stripped, stderr warning lines scrubbed, identifier
        redactions applied, py3Dmol viewers snapshotted, and the full set of mime types
        encountered before any changes (for reporting).
    """
    nb = json.loads(path.read_text())
    ignored = _ignored_warning_substrings()
    stripped = 0
    scrubbed = 0
    redacted = 0
    snapshots = 0
    seen_mimes: set[str] = set()

    for cell_idx, cell in enumerate(nb.get("cells", [])):
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
                seen_mimes.update(data)
                if snapshotter is not None and _snapshot_3dmol(data, snapshotter, f"{_rel(path)} cell {cell_idx}"):
                    snapshots += 1
                for mime in list(data.keys()):
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

    if stripped or scrubbed or redacted or snapshots:
        # ensure_ascii=False keeps literal UTF-8 (em dashes, emoji, arrows), matching
        # nbformat's writer; the default would escape them and churn every notebook.
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return stripped, scrubbed, redacted, snapshots, seen_mimes


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
    ap.add_argument(
        "--no-snapshots",
        action="store_true",
        help="Leave py3Dmol viewers as HTML instead of replacing them with PNG snapshots.",
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
    snapshotter: Snapshotter | None = None
    if not args.no_snapshots:
        if Snapshotter.available():
            snapshotter = Snapshotter()
        else:
            print(
                "WARNING: playwright not installed; py3Dmol viewers stay as HTML and won't render on GitHub. "
                "Install with `pip install playwright && playwright install --with-deps chromium`.",
                flush=True,
            )
    print(flush=True)

    failures: list[tuple[Path, str]] = []
    all_mimes: set[str] = set()
    total_stripped = 0
    total_scrubbed = 0
    total_redacted = 0
    total_snapshots = 0

    progress = tqdm(notebooks, desc="Processing", unit="nb", file=sys.stderr)
    try:
        for nb_path in progress:
            rel = _rel(nb_path)
            progress.set_postfix_str(nb_path.parent.parent.name)

            if args.sanitize_only:
                ok, msg = True, "ok"
            else:
                ok, msg = execute_notebook(nb_path, args.timeout)

            if ok:
                stripped, scrubbed, redacted, snapshots, mimes = sanitize(nb_path, snapshotter)
                total_stripped += stripped
                total_scrubbed += scrubbed
                total_redacted += redacted
                total_snapshots += snapshots
                all_mimes.update(mimes)
                print(
                    f"  ok    {rel}  (stripped {stripped}, scrubbed {scrubbed}, redacted {redacted}, "
                    f"snapshots {snapshots})",
                    flush=True,
                )
            else:
                failures.append((nb_path, msg))
                print(f"  FAIL  {rel}: {msg}", flush=True)
    finally:
        progress.close()
        if snapshotter is not None:
            snapshotter.close()

    print(flush=True)
    print(
        f"Summary: {len(notebooks) - len(failures)}/{len(notebooks)} notebooks processed; "
        f"{total_stripped} widget views stripped, {total_scrubbed} warning lines scrubbed, "
        f"{total_redacted} identifiers redacted, {total_snapshots} py3Dmol viewers snapshotted.",
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
