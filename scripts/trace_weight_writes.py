#!/usr/bin/env python3
"""scripts/trace_weight_writes.py.

Run a command and report the files it wrote, grouped by folder. Use this to check
a toolkit's downloads before and after moving them to its model-cache directory.

Files modified after the command starts are collected under ``PROTO_HOME``, ``$HOME``, and
the temp dir. Tool envs and package caches under ``PROTO_HOME`` are summarized in one line
each, since an env build writes thousands of files that are not weights.

Usage::

    python scripts/trace_weight_writes.py -- proto-tools run esm2-embedding --example
    python scripts/trace_weight_writes.py --depth 5 -- proto-tools run fampnn-design --example
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("trace_weight_writes")

# Under PROTO_HOME: env builds and package caches, reported as one total each
_SUMMARIZED = ("proto_tool_envs", "uv_cache", ".micromamba")
# Never worth walking under $HOME
_SKIPPED_HOME = (".git", "node_modules", ".claude")


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _newer_files(root: Path, since: float, skip: set[Path], skip_names: tuple[str, ...]) -> list[tuple[Path, int]]:
    """Return (path, size) for files under ``root`` modified after ``since``."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        dirnames[:] = [d for d in dirnames if here / d not in skip and d not in skip_names]
        for name in filenames:
            path = here / name
            try:
                st = path.lstat()
            except OSError:
                continue
            if st.st_mtime >= since or st.st_ctime >= since:
                found.append((path, st.st_size))
    return found


def _group(files: list[tuple[Path, int]], root: Path, depth: int) -> dict[Path, tuple[int, int]]:
    """Group files by their first ``depth`` path components below ``root``: {folder: (count, bytes)}."""
    groups: dict[Path, list[int]] = defaultdict(lambda: [0, 0])
    for path, size in files:
        rel = path.relative_to(root)
        key = root.joinpath(*rel.parts[: min(depth, len(rel.parts) - 1)]) if len(rel.parts) > 1 else root
        groups[key][0] += 1
        groups[key][1] += size
    return {k: (v[0], v[1]) for k, v in groups.items()}


def main() -> int:
    """Run the traced command and print where it wrote files."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--depth", type=int, default=4, help="Folder depth below each root to group by")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to run, after --")
    args = parser.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        parser.error("no command given")

    proto_home = Path(os.environ.get("PROTO_HOME") or Path.home() / ".proto").expanduser().resolve()
    home = Path.home().resolve()
    tmp = Path(tempfile.gettempdir()).resolve()

    # Some filesystems store whole-second mtimes; back off so same-second writes still count
    since = time.time() - 1
    rc = subprocess.call(cmd)

    summarized = {proto_home / name for name in _SUMMARIZED}
    print(f"\nexit code: {rc}\n")
    for root in dict.fromkeys((proto_home, home, tmp)):
        if not root.is_dir():
            continue
        skip = set(summarized) | ({proto_home} if root != proto_home else set())
        files = _newer_files(root, since, skip, _SKIPPED_HOME if root == home else ())
        print(f"== {root} ({len(files)} files, {_human(sum(s for _, s in files))})")
        for folder, (count, size) in sorted(_group(files, root, args.depth).items()):
            print(f"   {_human(size):>9}  {count:>6} files  {folder}")
        if root == proto_home:
            for folder in sorted(summarized):
                if folder.is_dir():
                    sub = _newer_files(folder, since, set(), ())
                    if sub:
                        print(
                            f"   {_human(sum(s for _, s in sub)):>9}  {len(sub):>6} files  {folder}  (env/cache build)"
                        )
    return rc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
