"""Which progress bar ``progress_bar()`` returns, for each kind of output stream.

A notebook is not a tty. An ``isatty`` gate placed ahead of the notebook check therefore
makes the notebook renderer unreachable, which is how notebooks silently lost their spinner
for the length of one PR. The order of these two checks is the whole point of the tests.
"""

from unittest.mock import patch

from tqdm import tqdm

from proto_tools.utils.progress import _AnimatedProgressBar, _NotebookProgressBar, progress_bar


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


def test_disabling_beats_every_other_condition():
    """``PROTO_NO_SPINNER`` has to win, including where a renderer would otherwise apply."""
    for notebook, interactive in ((True, True), (True, False), (False, True)):
        bar = _bar(notebook=notebook, interactive=interactive, disabled=True)
        assert not isinstance(bar, _AnimatedProgressBar | _NotebookProgressBar)
