"""scripts/_pytest_collect_runnable.py.

Tiny pytest plugin loaded by ``list_incomplete_tests.py`` to enumerate
the tests that would *actually run* under a given pytest invocation —
i.e. collected items minus those any skip / skipif marker would skip on
this machine.

We hook ``pytest_collection_modifyitems`` with ``trylast=True`` so the
project conftest has already applied its custom-marker-driven skips
(``uses_gpu``, ``skip_ci``, ``test_on_platforms``, etc.). Then we call
pytest's own ``evaluate_skip_marks`` for the final verdict, print one
``RUNNABLE: <nodeid>`` line per surviving item, and clear ``items`` so
nothing actually executes.
"""

from __future__ import annotations

import pytest
from _pytest.skipping import evaluate_skip_marks


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    runnable = [item.nodeid for item in items if evaluate_skip_marks(item) is None]
    for nid in runnable:
        print(f"RUNNABLE: {nid}")
    # Drop everything so pytest exits without running tests.
    items[:] = []
