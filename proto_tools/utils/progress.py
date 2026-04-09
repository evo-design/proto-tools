"""Animated progress bars and status indicators for long-running operations.

Provides ``progress_bar()`` as a drop-in ``tqdm`` replacement that shows an
animated dots spinner in the description prefix.  Infrastructure code calls
``set_substatus()`` to update the description as phases change (env setup →
subprocess launch → inference).  A ``contextvars.ContextVar`` bridge connects
the two without any direct coupling.

Styles are registered in ``SPINNER_STYLES`` — add new entries there
to create new animation patterns.
"""

from __future__ import annotations

import contextvars
import logging
import os
import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto


# ============================================================================
# Style registry — add new styles here
# ============================================================================
def _kitt_frames(width: int = 7) -> list[str]:
    """Generate Knight Rider / KITT-style bouncing bar frames.

    Args:
        width (int): Number of positions in the bar.

    Returns:
        list[str]: Frame strings like ``[□■□□□□□]``.
    """
    frames = []
    for i in range(width):
        bar = ["□"] * width
        bar[i] = "■"
        frames.append("[" + "".join(bar) + "]")
    for i in range(width - 2, 0, -1):
        bar = ["□"] * width
        bar[i] = "■"
        frames.append("[" + "".join(bar) + "]")
    return frames


def _equalizer_frames() -> list[str]:
    """Generate vertical bar equalizer frames.

    Returns:
        list[str]: Frame strings using Unicode block elements.
    """
    blocks = " ▁▂▃▄▅▆▇█"
    patterns = [
        [8, 3, 5, 2],
        [6, 5, 7, 3],
        [3, 8, 4, 6],
        [5, 6, 2, 8],
        [7, 2, 8, 5],
        [4, 7, 6, 3],
        [2, 5, 3, 7],
        [6, 4, 8, 4],
        [8, 6, 2, 6],
        [3, 8, 5, 2],
        [5, 3, 7, 8],
        [7, 5, 4, 5],
    ]
    return ["".join(blocks[h] for h in p) for p in patterns]


class SpinnerStyle:
    """Definition of a spinner animation style.

    Attributes:
        frames (list[str]): Animation frame strings.
        interval (float): Seconds between frame updates.
    """

    def __init__(self, frames: list[str], interval: float = 0.08) -> None:
        """Create a spinner style.

        Args:
            frames (list[str]): Animation frame strings.
            interval (float): Seconds between frame updates.
        """
        self.frames = frames
        self.interval = interval


SPINNER_STYLES: dict[str, SpinnerStyle] = {
    "dots": SpinnerStyle(
        frames=["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
        interval=0.08,
    ),
    "kitt": SpinnerStyle(
        frames=_kitt_frames(4),
        interval=0.12,
    ),
    "equalizer": SpinnerStyle(
        frames=_equalizer_frames(),
        interval=0.12,
    ),
    "bar": SpinnerStyle(
        frames=["|", "/", "-", "\\"],
        interval=0.10,
    ),
}


# ============================================================================
# Helpers
# ============================================================================
def _is_interactive() -> bool:
    """Check if stderr is connected to an interactive terminal."""
    try:
        return sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


def _is_disabled() -> bool:
    """Check if spinners are disabled via environment variable."""
    return os.environ.get("PROTO_NO_SPINNER", "").strip() in ("1", "true", "yes")


def _in_notebook() -> bool:
    """Check if running inside a Jupyter notebook."""
    try:
        from IPython import get_ipython  # type: ignore[attr-defined]

        shell = get_ipython()  # type: ignore[no-untyped-call]
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


# ============================================================================
# Context variable bridge
# ============================================================================
_active_bar: contextvars.ContextVar[_NotebookProgressBar | _AnimatedProgressBar | None] = contextvars.ContextVar(
    "_active_bar", default=None
)


# Notebook-friendly interval (slower to avoid flooding widget updates)
_NOTEBOOK_INTERVAL = 0.3


# ============================================================================
# Notebook progress bar (tqdm.auto widget with spinner thread)
# ============================================================================
class _NotebookProgressBar(tqdm_auto):  # type: ignore[type-arg]
    r"""Notebook-friendly progress bar using tqdm.auto's IPython widget.

    Uses the native IPython widget that updates in place, avoiding ghost
    lines from ``\r`` carriage returns.  A background spinner thread
    animates at a slower interval than the terminal variant.
    """

    def __init__(self, *args: Any, spinner_style: str = "dots", **kwargs: Any) -> None:  # noqa: D417
        """Create a notebook progress bar.

        All positional and keyword args are forwarded to tqdm.auto.

        Args:
            spinner_style (str): Animation style name from ``SPINNER_STYLES``.
        """
        self._base_desc = kwargs.get("desc", "") or ""
        self._style = SPINNER_STYLES.get(spinner_style, SPINNER_STYLES["dots"])
        self._stop_event = threading.Event()
        self._token = _active_bar.set(self)
        super().__init__(*args, **kwargs)

        self._spinner_thread = threading.Thread(target=self._animate, daemon=True)
        self._spinner_thread.start()

    def _animate(self) -> None:
        """Background thread that cycles spinner frames in the description."""
        frames = self._style.frames
        idx = 0
        while not self._stop_event.is_set():
            frame = frames[idx % len(frames)]
            try:
                self.set_description(f"{frame} {self._base_desc}", refresh=True)
            except Exception:
                break
            idx += 1
            self._stop_event.wait(_NOTEBOOK_INTERVAL)

    def _set_base_desc(self, message: str) -> None:
        """Update the description text (spinner keeps animating).

        Args:
            message (str): New description text.
        """
        self._base_desc = message

    def _set_style(self, style_name: str) -> None:
        """Switch the spinner animation style.

        Args:
            style_name (str): Style name from ``SPINNER_STYLES``.
        """
        if style_name in SPINNER_STYLES:
            self._style = SPINNER_STYLES[style_name]

    def close(self) -> None:
        """Stop the spinner thread and close the progress bar."""
        if self._stop_event.is_set():
            super().close()
            return
        self._stop_event.set()
        self._spinner_thread.join(timeout=1.0)
        self.set_description(self._base_desc, refresh=True)
        _active_bar.reset(self._token)
        super().close()


# ============================================================================
# Animated progress bar (tqdm subclass, terminal only)
# ============================================================================
class _AnimatedProgressBar(tqdm):  # type: ignore[type-arg]
    """tqdm progress bar with animated spinner in the description prefix.

    A background daemon thread cycles through spinner frames, updating the
    description via ``set_description()``.  The base description can be changed
    at any time via ``_set_base_desc()`` (called by ``set_substatus()``).
    """

    def __init__(self, *args: Any, spinner_style: str = "dots", **kwargs: Any) -> None:  # noqa: D417
        """Create an animated progress bar.

        All positional and keyword args are forwarded to tqdm.

        Args:
            spinner_style (str): Animation style name from ``SPINNER_STYLES``.
        """
        self._base_desc = kwargs.pop("desc", "") or ""
        self._style = SPINNER_STYLES.get(spinner_style, SPINNER_STYLES["dots"])
        self._stop_event = threading.Event()
        self._token = _active_bar.set(self)

        super().__init__(*args, desc=self._base_desc, **kwargs)  # type: ignore[call-overload]

        self._spinner_thread = threading.Thread(target=self._animate, daemon=True)
        self._spinner_thread.start()

    def _animate(self) -> None:
        """Background thread that cycles spinner frames in the description."""
        frames = self._style.frames
        interval = self._style.interval
        idx = 0
        while not self._stop_event.is_set():
            frame = frames[idx % len(frames)]
            try:
                self.set_description(f"{frame} {self._base_desc}", refresh=True)
            except Exception:
                break
            idx += 1
            self._stop_event.wait(interval)

    def _set_base_desc(self, message: str) -> None:
        """Update the base description (the text after the spinner prefix).

        Args:
            message (str): New description text.
        """
        self._base_desc = message

    def _set_style(self, style_name: str) -> None:
        """Switch the spinner animation style.

        Args:
            style_name (str): Style name from ``SPINNER_STYLES``.
        """
        if style_name in SPINNER_STYLES:
            self._style = SPINNER_STYLES[style_name]

    def close(self) -> None:
        """Stop the spinner thread and close the progress bar."""
        if self._stop_event.is_set():
            super().close()
            return
        self._stop_event.set()
        self._spinner_thread.join(timeout=1.0)
        # Restore clean description for the final rendered line
        self.set_description(self._base_desc, refresh=True)
        _active_bar.reset(self._token)
        super().close()


# ============================================================================
# Public API
# ============================================================================
def progress_bar(*args: Any, spinner_style: str = "dots", **kwargs: Any) -> tqdm:  # type: ignore[type-arg]  # noqa: D417
    """Create a tqdm progress bar with an animated spinner in the description.

    Drop-in replacement for ``tqdm()``.  A background thread animates a
    spinner prefix (e.g. ``⠙``) in the description.  Infrastructure code
    can call ``set_substatus()`` to update the description as phases change.

    If spinners are disabled (``PROTO_NO_SPINNER=1``), returns a plain
    ``tqdm`` bar.

    All positional and keyword args are forwarded to tqdm.

    Args:
        spinner_style (str): Animation style name from ``SPINNER_STYLES``.

    Returns:
        tqdm: A tqdm-compatible progress bar instance.

    Example::

        from proto_tools.utils.progress import progress_bar

        pbar = progress_bar(total=10, desc="Folding structures", unit="structure")
        for item in items:
            result = process(item)
            pbar.update(1)
        pbar.close()
    """
    if _is_disabled() or kwargs.get("disable"):
        return tqdm(*args, **kwargs)
    if _in_notebook():
        return _NotebookProgressBar(*args, spinner_style=spinner_style, **kwargs)
    return _AnimatedProgressBar(*args, spinner_style=spinner_style, **kwargs)


def set_substatus(message: str, style: str | None = None) -> None:
    """Update the active progress bar's description to reflect a new phase.

    If a ``progress_bar()`` is active in the current context, updates its
    base description (the spinner keeps animating).  Optionally switches
    the spinner style.  If no progress bar is active, falls back to logging.

    This is the main integration point for infrastructure code
    (``tool_instance.py``) to communicate status without coupling to tqdm.

    Args:
        message (str): Status message (e.g. ``"Setting up environment"``).
        style (str | None): Optional spinner style name from ``SPINNER_STYLES``.

    Example::

        set_substatus("Starting subprocess")
        set_substatus("Running inference", style="kitt")
    """
    bar = _active_bar.get(None)
    if bar is not None:
        bar._set_base_desc(message)
        if style is not None:
            bar._set_style(style)
    else:
        # Fallback: standalone spinner for code running outside a progress bar
        _fallback_log(message)


def _fallback_log(message: str) -> None:
    """Log a substatus message when no progress bar is active.

    Args:
        message (str): Status message to display.
    """
    logger = logging.getLogger("proto_tools.utils.progress")
    logger.info(message)


# ============================================================================
# Standalone status indicator (internal fallback, not exported)
# ============================================================================
_spinner_state: dict[str, Any] = {
    "stop_event": threading.Event(),
    "thread": threading.Thread(),
}


@contextmanager
def status_indicator(message: str, style: str = "dots") -> Generator[None, None, None]:
    """Show an animated status indicator while a block executes.

    Internal fallback used when no ``progress_bar()`` is active.
    Writes only to stderr (never stdout).  Auto-disables in non-TTY
    environments (CI, pipes) or when ``PROTO_NO_SPINNER=1`` is set.

    Args:
        message (str): Status message displayed next to the animation.
        style (str): Animation style name from ``SPINNER_STYLES``.
    """
    if _is_disabled():
        yield
        return

    resolved = SPINNER_STYLES.get(
        os.environ.get("PROTO_SPINNER_STYLE", style),
        SPINNER_STYLES["dots"],
    )

    if not _is_interactive():
        sys.stderr.write(f"{message}...\n")
        sys.stderr.flush()
        yield
        return

    stop_event = threading.Event()
    _spinner_state["stop_event"] = stop_event
    frames = resolved.frames
    interval = resolved.interval
    stream: Any = sys.stderr

    def _spin() -> None:
        idx = 0
        while not stop_event.is_set():
            frame = frames[idx % len(frames)]
            stream.write(f"\r{frame} {message}...")
            stream.flush()
            idx += 1
            stop_event.wait(interval)
        stream.write("\r\033[2K")
        stream.flush()

    thread = threading.Thread(target=_spin, daemon=True)
    _spinner_state["thread"] = thread
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=1.0)
