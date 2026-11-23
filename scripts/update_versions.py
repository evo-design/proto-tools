#!/usr/bin/env python3
"""scripts/update_versions.py.

Record each toolkit's current source hash in ``versions.yaml`` to satisfy
``tests/style_consistency_tests/test_toolkit_versions.py``. Toolkits whose top
entry already matches are left alone. See ``notes/versioning.md``.

Usage::

    python scripts/update_versions.py --bump patch            # new patch entry per changed toolkit
    python scripts/update_versions.py --bump minor esmfold    # only the named toolkits
    python scripts/update_versions.py --in-place              # replace the top entry's hash
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import get_args

# Hash this checkout, not whichever one the installed proto_tools points at: an editable
# install from another worktree would otherwise be hashed and have its versions.yaml rewritten.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from proto_tools.utils.tool_versions import VERSIONS_FILE, Bump, packaged_toolkit_dirs, record_source_hash  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    """Parse arguments and update the selected toolkits' ``versions.yaml`` files."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bump", choices=get_args(Bump), help="add a new top entry at this level")
    mode.add_argument(
        "--in-place",
        action="store_true",
        help="replace the top entry's hash; for an entry already added on this branch, or a hashing change",
    )
    parser.add_argument("toolkits", nargs="*", help="toolkit directory names to update (default: all)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dirs = packaged_toolkit_dirs()
    if args.toolkits:
        unknown = set(args.toolkits) - {d.name for d in dirs}
        if unknown:
            parser.error(f"unknown toolkits: {', '.join(sorted(unknown))}")
        dirs = [d for d in dirs if d.name in args.toolkits]

    updated = 0
    for toolkit_dir in dirs:
        version = record_source_hash(toolkit_dir, args.bump)
        if version is not None:
            updated += 1
            logger.info("%s/%s: %s", toolkit_dir.name, VERSIONS_FILE, version)
    logger.info("%d of %d toolkits updated", updated, len(dirs))


if __name__ == "__main__":
    main()
