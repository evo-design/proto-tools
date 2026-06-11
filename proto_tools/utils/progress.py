"""Animated progress bars and status indicators for long-running operations.

Provides ``progress_bar()`` as a drop-in ``tqdm`` replacement that shows an
animated dots spinner in the description prefix.  Infrastructure code calls
``set_substatus()`` to update the description as phases change (env setup →
subprocess launch → inference).  A process-wide active-bar stack connects
the two without any direct coupling, and stays visible across thread
boundaries (worker drain threads call ``set_substatus`` too).

Styles are registered in ``SPINNER_STYLES`` — add new entries there
to create new animation patterns.
"""

from __future__ import annotations

import contextlib
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


def _cloud_frames(track: int = 3) -> list[str]:
    """Generate cloud-dispatch frames: a pulse bouncing between a computer and a cloud.

    Evokes the round-trip to hosted execution. Each frame is fixed-width (one ``•``
    pulse over a ``·`` track, flanked by the two emoji), so the pulse animates in place
    and the badge stays short enough to leave room for the status message.

    Args:
        track (int): Number of dot positions between the two icons.

    Returns:
        list[str]: Frame strings like ``💻·•·☁️``.
    """
    positions = list(range(track)) + list(range(track - 2, 0, -1))
    frames = []
    for pos in positions:
        dots = ["·"] * track
        dots[pos] = "•"
        frames.append("💻" + "".join(dots) + "☁️")
    return frames


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
    "cloud": SpinnerStyle(
        frames=_cloud_frames(),
        interval=0.28,
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
# Active progress bar registry
# ============================================================================
# Process-wide stack of active bars. We use a lock-protected module-level list
# instead of a ContextVar so worker drain threads (which run in their own
# threading.Thread without inheriting the main thread's context) can still
# resolve the current bar from ``set_substatus``. Nested ``progress_bar()``
# blocks push/pop on the stack so the innermost bar wins.
_active_bars: list[_NotebookProgressBar | _AnimatedProgressBar] = []
_active_bars_lock = threading.Lock()


def _push_active_bar(bar: _NotebookProgressBar | _AnimatedProgressBar) -> None:
    with _active_bars_lock:
        _active_bars.append(bar)


def _pop_active_bar(bar: _NotebookProgressBar | _AnimatedProgressBar) -> None:
    """Remove ``bar`` from the active stack (idempotent; safe under nesting)."""
    with _active_bars_lock, contextlib.suppress(ValueError):
        _active_bars.remove(bar)


def _current_active_bar() -> _NotebookProgressBar | _AnimatedProgressBar | None:
    with _active_bars_lock:
        return _active_bars[-1] if _active_bars else None


def has_active_progress_bar() -> bool:
    """Check if a progress bar is currently active anywhere in this process."""
    return _current_active_bar() is not None


_current_tool_function: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_tool_function", default=None
)


def get_current_tool_function() -> str | None:
    """Return the name of the currently executing tool function, if any."""
    return _current_tool_function.get()


def set_current_tool_function(name: str | None) -> contextvars.Token[str | None]:
    """Set the currently executing tool function name.

    Args:
        name (str | None): Function name, or None to clear.

    Returns:
        contextvars.Token[str | None]: Token for restoring the previous value.
    """
    return _current_tool_function.set(name)


def reset_current_tool_function(token: contextvars.Token[str | None]) -> None:
    """Restore the previous tool function name from a token.

    Args:
        token (contextvars.Token[str | None]): Token from ``set_current_tool_function()``.
    """
    _current_tool_function.reset(token)


# Notebook-friendly interval (slower to avoid flooding widget updates)
_NOTEBOOK_INTERVAL = 0.3

# Fixed display width (chars) for the description text. Substatus messages are
# padded with spaces if shorter and truncated with an ellipsis if longer, so
# the bar never shifts horizontally as the substatus changes length.
_DESC_DISPLAY_WIDTH = 40


def _format_desc_for_display(message: str, width: int = _DESC_DISPLAY_WIDTH) -> str:
    """Pad or truncate a substatus message to a fixed display width."""
    if len(message) > width:
        return message[: width - 1] + "…"
    return message.ljust(width)


def _compose_desc(prefix: str | None, leader: str, base_desc: str) -> str:
    """Compose a spinner description line: optional prefix + frame/icon + padded message."""
    padded = _format_desc_for_display(base_desc)
    if prefix:
        return f"{prefix} {leader} {padded}"
    return f"{leader} {padded}"


# ============================================================================
# Notebook progress bar (tqdm.auto widget with spinner thread)
# ============================================================================
class _NotebookProgressBar(tqdm_auto):  # type: ignore[type-arg]
    r"""Notebook-friendly progress bar using tqdm.auto's IPython widget.

    Uses the native IPython widget that updates in place, avoiding ghost
    lines from ``\r`` carriage returns.  A background spinner thread
    animates at a slower interval than the terminal variant.
    """

    def __init__(  # noqa: D417
        self, *args: Any, spinner_style: str = "dots", show_bar: bool = True, prefix: str | None = None, **kwargs: Any
    ) -> None:
        """Create a notebook progress bar.

        All positional and keyword args are forwarded to tqdm.auto.

        Args:
            spinner_style (str): Animation style name from ``SPINNER_STYLES``.
            show_bar (bool): Whether to show the progress bar widget. When False,
                only the spinner, description, and elapsed time are visible.
            prefix (str | None): Optional static badge rendered before the
                animated frame (e.g. ``"☁"`` for cloud dispatch).
        """
        self._base_desc = kwargs.get("desc", "") or ""
        self._style = SPINNER_STYLES.get(spinner_style, SPINNER_STYLES["dots"])
        self._show_bar = show_bar
        self._prefix = prefix
        self._stop_event = threading.Event()
        _push_active_bar(self)
        super().__init__(*args, **kwargs)

        if not show_bar and hasattr(self, "container") and hasattr(self.container, "children"):
            for child in self.container.children:
                if type(child).__name__ in ("FloatProgress", "IntProgress"):
                    child.layout.display = "none"

        self._spinner_thread = threading.Thread(target=self._animate, daemon=True)
        self._spinner_thread.start()

    def _animate(self) -> None:
        """Background thread that cycles spinner frames in the description."""
        frames = self._style.frames
        idx = 0
        while not self._stop_event.is_set():
            frame = frames[idx % len(frames)]
            try:
                self.set_description(_compose_desc(self._prefix, frame, self._base_desc), refresh=True)
            except Exception:
                break
            idx += 1
            self._stop_event.wait(_NOTEBOOK_INTERVAL)

    def _set_substatus(self, message: str) -> None:
        """Update the description text shown next to the spinner (bar position unchanged).

        The displayed text is padded/truncated to a fixed width by
        :func:`_format_desc_for_display`, so the rest of the bar stays in a
        stable column regardless of message length.

        Args:
            message (str): Substatus text (e.g. ``"Loading checkpoint"``).
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
        # See _AnimatedProgressBar.close() for why status-only spinners skip
        # the icon \u2014 same misleading-\u2718 problem on success.
        if self._show_bar:
            completed = self.total is not None and self.n >= self.total
            icon = "\u2714" if completed else "\u2718"
            self.set_description(_compose_desc(self._prefix, icon, self._base_desc), refresh=True)
        _pop_active_bar(self)
        super().close()


# ============================================================================
# Animated progress bar (tqdm subclass, terminal only)
# ============================================================================
class _AnimatedProgressBar(tqdm):  # type: ignore[type-arg]
    """tqdm progress bar with animated spinner in the description prefix.

    A background daemon thread cycles through spinner frames, updating the
    description via ``set_description()``. ``set_substatus()`` replaces the
    description text; :func:`_format_desc_for_display` pads/truncates it to a
    fixed width so the bar never shifts horizontally as substatus length
    changes.
    """

    def __init__(  # noqa: D417
        self, *args: Any, spinner_style: str = "dots", show_bar: bool = True, prefix: str | None = None, **kwargs: Any
    ) -> None:
        """Create an animated progress bar.

        All positional and keyword args are forwarded to tqdm.

        Args:
            spinner_style (str): Animation style name from ``SPINNER_STYLES``.
            show_bar (bool): Whether the bar widget is rendered. When False
                (status-only spinner used by tool dispatch), ``close()`` clears
                the line instead of pinning a final icon — the trailing
                "Tool X: completed in Ns" log line already conveys the result.
            prefix (str | None): Optional static badge rendered before the
                animated frame (e.g. ``"☁"`` to flag cloud dispatch). Stays
                fixed across frames so it's a stable glanceable marker.
        """
        self._base_desc = kwargs.pop("desc", "") or ""
        self._style = SPINNER_STYLES.get(spinner_style, SPINNER_STYLES["dots"])
        self._show_bar = show_bar
        self._prefix = prefix
        self._stop_event = threading.Event()
        _push_active_bar(self)

        # Status-only spinners shouldn't leave their final frame on screen.
        if not show_bar:
            kwargs.setdefault("leave", False)

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
                self.set_description(_compose_desc(self._prefix, frame, self._base_desc), refresh=True)
            except Exception:
                break
            idx += 1
            self._stop_event.wait(interval)

    def _set_substatus(self, message: str) -> None:
        """Update the description text shown next to the spinner (bar position unchanged).

        The displayed text is padded/truncated to a fixed width by
        :func:`_format_desc_for_display`, so the rest of the bar stays in a
        stable column regardless of message length.

        Args:
            message (str): Substatus text (e.g. ``"Loading checkpoint"``).
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
        # Real progress bars: pin a final \u2714/\u2718 based on n vs total.
        # Status-only spinners (show_bar=False): skip the icon and let
        # leave=False (set in __init__) clear the line on super().close().
        # The n vs total heuristic is misleading for status spinners because
        # tool dispatch never calls update(), so n stays at 0 and every
        # successful run would render a red \u2718.
        if self._show_bar:
            completed = self.total is not None and self.n >= self.total
            icon = "\033[32m\u2714\033[0m" if completed else "\033[31m\u2718\033[0m"
            self.set_description(_compose_desc(self._prefix, icon, self._base_desc), refresh=True)
        _pop_active_bar(self)
        super().close()


# ============================================================================
# Public API
# ============================================================================
def progress_bar(  # noqa: D417
    *args: Any, spinner_style: str = "dots", show_bar: bool = True, prefix: str | None = None, **kwargs: Any
) -> tqdm:  # type: ignore[type-arg]
    """Create a tqdm progress bar with an animated spinner in the description.

    Drop-in replacement for ``tqdm()``.  A background thread animates a
    spinner prefix (e.g. ``⠙``) in the description.  Infrastructure code
    can call ``set_substatus()`` to update the description as phases change.

    If spinners are disabled (``PROTO_NO_SPINNER=1``), returns a plain
    ``tqdm`` bar.

    All positional and keyword args are forwarded to tqdm.

    Args:
        spinner_style (str): Animation style name from ``SPINNER_STYLES``.
        show_bar (bool): Whether to show the progress bar widget. When False,
            only the spinner, description, and elapsed time are visible.
            Only affects notebook rendering.
        prefix (str | None): Optional static badge rendered before the animated
            frame (e.g. ``"☁"`` to flag cloud-dispatched runs). Held fixed
            across frames so it functions as a glanceable mode indicator.

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
        return _NotebookProgressBar(*args, spinner_style=spinner_style, show_bar=show_bar, prefix=prefix, **kwargs)
    return _AnimatedProgressBar(*args, spinner_style=spinner_style, show_bar=show_bar, prefix=prefix, **kwargs)


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
    bar = _current_active_bar()
    if bar is not None:
        bar._set_substatus(message)
        if style is not None:
            bar._set_style(style)
    else:
        # Fallback: standalone spinner for code running outside a progress bar
        _fallback_log(message)


def update_active_substatus(message: str) -> None:
    """Update the active progress bar's substatus, or no-op if none is active.

    The bar-only sink for log-driven spinner updates (``SpinnerFromLogsHandler``).
    Unlike :func:`set_substatus` it never logs on the no-bar path; re-logging here
    would loop back through the handler that routed the record in.
    """
    bar = _current_active_bar()
    if bar is not None:
        bar._set_substatus(message)


def _fallback_log(message: str) -> None:
    """Log a substatus message when no progress bar is active.

    Flagged ``update_status`` so a log-driven spinner downstream still advances
    even though this process has no bar to drive directly: the parent's
    ``SpinnerFromLogsHandler``, or a cloud client replaying captured worker logs.
    """
    logger = logging.getLogger("proto_tools.utils.progress")
    logger.info(message, extra={"update_status": True})


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
