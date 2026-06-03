"""proto_tools - A unified library of computational and AI tools for biology.

This package provides a collection of bioinformatics tools and entity definitions
for working with biological sequences, structures, and computational biology tasks.
"""

__version__ = "0.0.0"

import sys as _sys
from pathlib import Path as _Path

# Put source-tree standalone_helpers/ on sys.path so parent-side imports of standalone modules can resolve `from standalone_helpers import X` (subprocess inserts the per-tool copy ahead of this).
_sys.path.append(str(_Path(__file__).parent / "utils" / "standalone_helpers_source"))

from proto_tools.utils.logging_config import (  # noqa: F401
    get_logger,
    install_logger_class,
    install_spinner_handler,
    setup_logging,
)

# Make every getLogger return a ProtoLogger so update_status=True works as a kwarg.
install_logger_class()

# Parent-side spinner takeover: update_status=True records (parent or bridged) update the spinner subtitle.
install_spinner_handler()

from proto_tools.entities import *  # noqa: E402, F403
from proto_tools.tools import *  # noqa: E402, F403
