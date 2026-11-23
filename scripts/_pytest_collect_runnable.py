"""scripts/_pytest_collect_runnable.py.

List the tests that would run for a given pytest command on this machine.

``list_incomplete_tests.py`` loads this plugin to collect tests and exclude
those covered by ``skip`` or ``skipif`` markers.

The ``pytest_collection_modifyitems`` hook runs with ``trylast=True``, after
the project conftest has applied skips from custom markers
(``uses_gpu``, ``skip_ci``, ``test_on_platforms``, etc.). Then we call
pytest's ``evaluate_skip_marks`` to check the remaining skip markers, print one
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
