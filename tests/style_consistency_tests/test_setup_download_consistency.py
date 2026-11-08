"""Consistency check for downloads in standalone setup scripts.

Weights downloads may take as long as they need while they progress, and fail only once they
stall. The ``standalone_helpers.sh`` download helpers (``proto_download``,
``proto_download_verified``, ``proto_download_archive``, ...) enforce that with curl's
``--speed-limit``/``--speed-time`` stall window, and add locking and temp-file staging on top. A
``setup.sh`` that calls ``curl`` or ``wget`` itself skips all of that, and ``--max-time`` caps the
whole transfer so a slow but healthy download fails.
"""

import re
from pathlib import Path

import pytest

_PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent / "proto_tools"

_SETUP_SCRIPTS = sorted(_PACKAGE_DIR.rglob("setup.sh"))

_RAW_DOWNLOAD = re.compile(r"(?<![\w-])(curl|wget)\s")


def _code_lines(path: Path) -> list[tuple[int, str]]:
    """Return ``(line number, line)`` for each non-comment line of a shell script."""
    return [(i, line) for i, line in enumerate(path.read_text().splitlines(), 1) if not line.lstrip().startswith("#")]


@pytest.mark.parametrize("setup", _SETUP_SCRIPTS, ids=lambda p: p.relative_to(_PACKAGE_DIR).as_posix())
def test_setup_downloads_through_the_helpers(setup: Path):
    """``setup.sh`` never runs ``curl``/``wget`` directly or caps a transfer with ``--max-time``."""
    offending = [
        f"line {i}: {line.strip()}"
        for i, line in _code_lines(setup)
        if _RAW_DOWNLOAD.search(line) or "--max-time" in line
    ]
    assert not offending, (
        f"{setup.relative_to(_PACKAGE_DIR)} downloads without the standalone_helpers.sh helpers "
        f"(use proto_download / proto_download_verified / proto_download_archive):\n" + "\n".join(offending)
    )
