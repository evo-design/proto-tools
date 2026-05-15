"""Allow ``python -m proto_tools`` as an alternate CLI entry point.

Same surface as the ``proto-tools`` console script registered in
``pyproject.toml`` — see ``proto_tools.cli.main`` for the verbs.
"""

from __future__ import annotations

import sys

from proto_tools.cli import main

if __name__ == "__main__":
    sys.exit(main())
