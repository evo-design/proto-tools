"""Which progress bar ``progress_bar()`` returns, for each kind of output stream.

A notebook is not a tty. An ``isatty`` gate placed ahead of the notebook check therefore
makes the notebook renderer unreachable, which is how notebooks silently lost their spinner
for the length of one PR. The order of these two checks is the whole point of the tests.
"""

import io
import logging
import threading
from unittest.mock import patch

import pytest
from tqdm import tqdm

from proto_tools.utils.base_config import HOSTED_ENV_VAR
from proto_tools.utils.progress import (
    _AnimatedProgressBar,
    _NotebookProgressBar,
    has_active_progress_bar,
    progress_bar,
)


def _bar(*, notebook: bool, interactive: bool, disabled: bool = False):
    """Build a bar with the three conditions that select a renderer pinned, then close it.

    Closing matters. An animated bar registers itself in a process-wide stack, and a test that
    leaves one there makes every later test think a bar is already on screen.
    """
    with (
        patch("proto_tools.utils.progress._in_notebook", return_value=notebook),
        patch("proto_tools.utils.progress._is_interactive", return_value=interactive),
        patch("proto_tools.utils.progress._is_disabled", return_value=disabled),
    ):
        bar = progress_bar(total=1, desc="test")
    bar.close()
    return bar


def test_a_notebook_gets_the_notebook_bar_despite_not_being_a_tty():
    """The regression. A Jupyter kernel's stderr is a ZMQ stream, so ``isatty`` is False."""
    assert isinstance(_bar(notebook=True, interactive=False), _NotebookProgressBar)


def test_a_terminal_gets_the_animated_bar():
    assert isinstance(_bar(notebook=False, interactive=True), _AnimatedProgressBar)


def test_a_redirected_stream_gets_a_plain_bar():
    """Without a tty to redraw on, every frame would land as its own line in a log."""
    bar = _bar(notebook=False, interactive=False)
    assert isinstance(bar, tqdm)
    assert not isinstance(bar, _AnimatedProgressBar | _NotebookProgressBar)


def test_a_redirected_stream_does_not_emit_progress_frames():
    """Captured agent commands should not fill logs with tqdm refreshes."""
    stream = io.StringIO()
    with (
        patch("proto_tools.utils.progress._in_notebook", return_value=False),
        patch("proto_tools.utils.progress._is_interactive", return_value=False),
    ):
        bar = progress_bar(total=2, desc="Running learned model", file=stream)
    bar.update(2)
    bar.close()
    assert bar.disable is True
    assert stream.getvalue() == ""


def test_an_explicit_captured_stream_overrides_interactive_stderr():
    """The selected output file controls rendering when stderr itself is a tty."""
    stream = io.StringIO()
    with (
        patch("proto_tools.utils.progress._in_notebook", return_value=False),
        patch("proto_tools.utils.progress._is_interactive", return_value=True),
    ):
        bar = progress_bar(total=1, file=stream)
    bar.update(1)
    bar.close()
    assert bar.disable is True
    assert stream.getvalue() == ""


def test_disabling_beats_every_other_condition():
    """``PROTO_NO_SPINNER`` has to win, including where a renderer would otherwise apply."""
    for notebook, interactive in ((True, True), (True, False), (False, True)):
        bar = _bar(notebook=notebook, interactive=interactive, disabled=True)
        assert not isinstance(bar, _AnimatedProgressBar | _NotebookProgressBar)


# ============================================================================
# A hosted worker reports a counting bar as status phases
# ============================================================================
@pytest.fixture
def hosted(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Pin a redirected stream in a hosted worker, and hand back a settable fake clock."""
    clock = [0.0]
    monkeypatch.setenv(HOSTED_ENV_VAR, "1")
    monkeypatch.setattr("proto_tools.utils.progress._in_notebook", lambda: False)
    monkeypatch.setattr("proto_tools.utils.progress._is_interactive", lambda: False)
    monkeypatch.setattr("proto_tools.utils.progress.time.monotonic", lambda: clock[0])
    return clock


def _phases(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if getattr(record, "update_status", False)]


def test_a_hosted_bar_reports_its_count_at_human_pace(hosted, caplog):
    """The first update shows at once, then one per interval, and reaching the total always shows."""
    with caplog.at_level(logging.INFO, logger="proto_tools.utils.progress"):
        bar = progress_bar(total=10, desc="Folding", unit="structure")
        for step in range(10):
            hosted[0] = step * 2.0  # an update every 2s against a 5s interval
            bar.update(1)
        bar.close()

    assert _phases(caplog) == [
        "Folding (1/10 structure)",
        "Folding (4/10 structure)",
        "Folding (7/10 structure)",
        "Folding (10/10 structure)",
    ]
    assert not has_active_progress_bar(), "a hosted bar must leave set_substatus on its logging fallback"


def test_a_hosted_bar_counts_iteration_without_a_total(hosted, caplog):
    """A generator has no length, so the phase carries the count alone."""
    with caplog.at_level(logging.INFO, logger="proto_tools.utils.progress"):
        items = list(progress_bar((i for i in range(3)), desc="Designing", unit="seq"))

    assert items == [0, 1, 2]
    assert _phases(caplog) == ["Designing (1 seq)"]


def test_a_hosted_bar_starts_no_thread(hosted, monkeypatch):
    """A monitor thread is cost a hosted worker pays for nothing, even on a disabled tqdm."""
    monkeypatch.setattr(tqdm, "monitor", None)  # otherwise an earlier bar's monitor hides a new one
    before = set(threading.enumerate())
    bar = progress_bar(total=2, desc="Folding")
    bar.update(2)
    bar.close()

    assert set(threading.enumerate()) <= before


@pytest.mark.parametrize(
    "kwargs", [{"show_bar": False}, {"disable": True}], ids=["status-only spinner", "caller disabled"]
)
def test_a_hosted_bar_without_a_count_to_report_says_nothing(hosted, caplog, kwargs):
    """A status-only spinner carries no count, and a caller that disabled its bar asked for silence."""
    with caplog.at_level(logging.INFO, logger="proto_tools.utils.progress"):
        bar = progress_bar(total=1, desc="Running tool", **kwargs)
        bar.update(1)
        bar.close()

    assert _phases(caplog) == []


def test_a_local_redirected_bar_reports_nothing(monkeypatch, caplog):
    """Outside a hosted worker a redirected stream stays silent, as it always has."""
    monkeypatch.delenv(HOSTED_ENV_VAR, raising=False)
    with caplog.at_level(logging.INFO, logger="proto_tools.utils.progress"):
        bar = _bar(notebook=False, interactive=False)
        bar.update(1)

    assert type(bar) is tqdm
    assert _phases(caplog) == []
