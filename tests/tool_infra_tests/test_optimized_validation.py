"""Validation of user-supplied data must survive ``python -O``.

``python -O`` strips every ``assert`` from the bytecode, so an assertion guarding
a public entry point silently disappears under optimized execution. Pydantic
models are unaffected, since their validators raise explicitly and pydantic-core
does its own checking, but plain helpers that users call with their own arrays
have to raise rather than assert. This runs the affected paths in an optimized
subprocess, which is the only way to observe the stripping.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_GEOMETRY_PROBE = """
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, sys.argv[1])

from proto_tools.entities import (
    adjacent_distances,
    distances_to_centroid,
    get_centroid,
    pairwise_distances,
)

if __debug__:
    raise RuntimeError("probe did not run under python -O")

for function in (pairwise_distances, adjacent_distances, get_centroid, distances_to_centroid):
    try:
        function(np.zeros((2, 2)))
    except ValueError as exc:
        if "shape (N, 3)" not in str(exc):
            raise RuntimeError(f"unexpected error from {function.__name__}: {exc}") from exc
    else:
        raise RuntimeError(f"{function.__name__} validation disappeared under python -O")

    try:
        function([[0.0, 0.0, 0.0]])
    except TypeError:
        pass
    else:
        raise RuntimeError(f"{function.__name__} type validation disappeared under python -O")
"""


def test_geometry_validation_survives_python_optimized_mode() -> None:
    """The public geometry helpers still reject bad input when assertions are stripped."""
    repository = Path(__file__).parents[2]
    result = subprocess.run(
        [sys.executable, "-O", "-c", _GEOMETRY_PROBE, str(repository)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
