"""Toolkits that share a model-cache folder must pin the same version of what they put there.

Several toolkits read weights another toolkit downloads, so one copy serves both: bindcraft,
freebindcraft, and germinal read ``alphafold2``'s AF2 params, and germinal reads ``chai1``'s Chai-1
files and ``ablang``'s AbLang2 weights. The cache keys these folders by name, not version, so
whichever toolkit downloads first decides what the others load. If one toolkit bumps its pin and
the other doesn't, the second one silently loads weights for the wrong version.

Each entry in ``_SHARED`` lists where every toolkit in a sharing group declares its version; the
test fails when they disagree, so a version bump has to move the whole group together.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
_TOOLS = _REPO / "proto_tools" / "tools"
_HELPERS_SH = _REPO / "proto_tools" / "utils" / "standalone_helpers_source" / "standalone_helpers.sh"

_AF2_TAR = r"alphafold_params_(\d{4}-\d{2}-\d{2})\.tar"

# Shared folder -> (file, regex capturing the version, required) for every toolkit that loads it.
# "required" means the file must pin a version: an unpinned install floats to whatever is newest.
_SHARED: dict[str, list[tuple[Path, str, bool]]] = {
    "alphafold2 AF2 params": [
        (_HELPERS_SH, _AF2_TAR, True),
        # Toolkits that call the shared helper no longer name the tar; any that still do must agree
        *(
            (_TOOLS / path / "standalone" / "setup.sh", _AF2_TAR, False)
            for path in (
                "structure_prediction/alphafold2",
                "binder_design/bindcraft",
                "binder_design/freebindcraft",
                "binder_design/germinal",
            )
        ),
    ],
    "chai1 Chai-1 weights": [
        (_TOOLS / "structure_prediction/chai1/standalone/requirements.txt", r"chai[-_]lab==([\w.]+)", True),
        (_TOOLS / "binder_design/germinal/standalone/setup.sh", r"chai[-_]lab==([\w.]+)", True),
    ],
    **{
        f"ablang {model} archive": [
            (_TOOLS / path / "standalone" / "inference.py", rf'"{model}": \("(https://[^"]+)"', True)
            for path in ("masked_models/ablang", "binder_design/germinal")
        ]
        for model in ("ablang1-heavy", "ablang2-paired")
    },
    "ablang AbLang2 weights": [
        (_TOOLS / "masked_models/ablang/standalone/requirements.txt", r"ablang2==([\w.]+)", True),
        (_TOOLS / "binder_design/germinal/standalone/setup.sh", r"ablang2==([\w.]+)", True),
    ],
}


@pytest.mark.parametrize("group", sorted(_SHARED))
def test_shared_weights_versions_agree(group: str) -> None:
    """Every place a sharing group declares its version names the same one, and none is unpinned."""
    versions: dict[str, set[str]] = {}
    for path, pattern, required in _SHARED[group]:
        rel = str(path.relative_to(_REPO))
        found = set(re.findall(pattern, path.read_text()))
        assert found or not required, f"{group}: {rel} must pin a version (an unpinned install may drift)"
        if found:
            versions[rel] = found
    distinct = set().union(*versions.values())
    assert len(distinct) == 1, f"{group}: toolkits sharing this folder pin different versions: {versions}"
