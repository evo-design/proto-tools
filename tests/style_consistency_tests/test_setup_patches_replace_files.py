"""Consistency check for setup scripts that patch installed packages.

uv hardlinks each installed file from its cache into every env that installs the same package, so
envs share one copy of torch. A ``setup.sh`` that rewrites an installed file in place (Python's
``open(path, "w")`` or ``Path.write_text``) edits that shared copy, and the patch shows up in every
other env holding the file. A patch must write a new file and swap it in with ``os.replace``, which
gives the env its own copy. ``sed -i`` already does that.
"""

import re
from pathlib import Path

import pytest

_PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent / "proto_tools"

# Scripts that edit what pip or uv installed into the env
_PATCHING_SCRIPTS = sorted(
    path
    for path in _PACKAGE_DIR.rglob("setup.sh")
    if re.search(r"SITE_PACKAGES|getsitepackages|site-packages", path.read_text())
)

# An in-place Python write, unless it writes the temporary file that os.replace swaps in
_IN_PLACE_WRITE = re.compile(r"open\((?![^)]*\.proto-patch)[^)]*,\s*['\"]w['\"]|\.write_text\(")


@pytest.mark.parametrize("setup", _PATCHING_SCRIPTS, ids=lambda p: p.relative_to(_PACKAGE_DIR).as_posix())
def test_patches_to_installed_files_replace_them(setup: Path):
    """A patch writes ``<file>.proto-patch`` and ``os.replace``-s it over the file, never rewriting in place."""
    offending = [
        f"line {i}: {line.strip()}"
        for i, line in enumerate(setup.read_text().splitlines(), 1)
        if _IN_PLACE_WRITE.search(line)
    ]
    assert not offending, (
        f"{setup.relative_to(_PACKAGE_DIR)} rewrites a file in place, which also patches every env sharing "
        "it through uv's hardlinks. Write '<file>.proto-patch' and os.replace it over the file:\n"
        + "\n".join(offending)
    )
