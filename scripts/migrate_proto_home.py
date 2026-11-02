#!/usr/bin/env python3
"""scripts/migrate_proto_home.py.

Move model weights from the old flat model-cache layout into per-toolkit folders.

Workers run the same migration automatically the first time a migrated toolkit starts, so
this script is only needed to migrate everything up front, or to preview the moves. It
covers every toolkit with an old path to move and acts on ``PROTO_MODEL_CACHE``
(default ``{PROTO_HOME}/proto_model_cache``). ``IN_ENV`` and ``NONE`` modes have nothing to
migrate.

Usage::

    python scripts/migrate_proto_home.py --dry-run
    python scripts/migrate_proto_home.py
    python scripts/migrate_proto_home.py --toolkit esm2 --cache-root /shared/team/model_weights
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from proto_tools.utils.weights_layout import default_cache_root, migrate_toolkit, toolkits_to_migrate

logger = logging.getLogger("migrate_proto_home")


def main() -> int:
    """Migrate the requested toolkits and print each move."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print the moves without making them")
    parser.add_argument("--toolkit", action="append", help="Migrate only this toolkit (repeatable)")
    parser.add_argument("--cache-root", type=Path, help="Model cache to migrate (default: from PROTO_MODEL_CACHE)")
    args = parser.parse_args()

    cache_root = args.cache_root or default_cache_root()
    if cache_root is None:
        logger.info("PROTO_MODEL_CACHE is IN_ENV or NONE; nothing to migrate")
        return 0

    toolkits = args.toolkit or toolkits_to_migrate(cache_root)

    verb = "would move" if args.dry_run else "moved"
    total = 0
    for toolkit in toolkits:
        for old, new in migrate_toolkit(toolkit, cache_root, dry_run=args.dry_run):
            logger.info("%s: %s %s -> %s", toolkit, verb, old, new)
            total += 1
    logger.info("%d path(s) %s under %s", total, verb, cache_root)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # migrate_toolkit logs each move itself; this script prints them once, with the dry-run wording
    logging.getLogger("proto_tools").setLevel(logging.WARNING)
    sys.exit(main())
