"""
Persistent subprocess worker for long-running tool processes.

Manages a subprocess that stays alive between calls, communicating via
stdin/stdout JSON-line protocol. This avoids reloading models on every call.
"""

from __future__ import annotations

import json
import logging
import os
import select
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any

from .device import determine_visible_devices

logger = logging.getLogger(__name__)

# Jupyter/IPython env vars that cause failures in subprocesses
_JUPYTER_BLOCKLIST = {
    "MPLBACKEND",
    "DISPLAY",
    "JPY_PARENT_PID",
    "JPY_SESSION_NAME",
    "PYDEVD_USE_CYTHON",
    "PYDEVD_USE_FRAME_EVAL",
    "JUPYTER_PLATFORM_DIRS",
    "JUPYTER_DATA_DIR",
    "JUPYTER_CONFIG_DIR",
    "JUPYTER_RUNTIME_DIR",
    "JUPYTER_PATH",
}

# CUDA library paths that interfere with bundled libraries in venvs
_CUDA_BLOCKLIST = {
    "LD_LIBRARY_PATH",
}

# Conda env vars that confuse pip/uv inside isolated venvs
_CONDA_BLOCKLIST = {
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "CONDA_PYTHON_EXE",
    "CONDA_PROMPT_MODIFIER",
    "CONDA_SHLVL",
    "CONDA_EXE",
    "_CE_CONDA",
    "_CONDA_EXE",
    "_CONDA_ROOT",
}

_BLOCKED_ENV_VARS = _JUPYTER_BLOCKLIST | _CUDA_BLOCKLIST | _CONDA_BLOCKLIST


def _merge_colon_paths(*values: str) -> str:
    """Merge colon-separated path strings while preserving order and deduplicating."""
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        for part in value.split(":"):
            part = part.strip()
            if not part or part in seen:
                continue
            merged.append(part)
            seen.add(part)
    return ":".join(merged)


def _discover_tool_ld_library_paths(tool_venv_path: Path | str | None) -> list[str]:
    """Discover CUDA-related library directories from a tool venv."""
    if not tool_venv_path:
        return []

    venv_path = Path(tool_venv_path)
    candidates: list[Path] = [
        venv_path / "cuda_env" / "lib",
        venv_path / "cuda_env" / "lib64",
    ]
    candidates.extend(venv_path.glob("lib/python*/site-packages/nvidia/*/lib"))

    found: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        if path.is_dir():
            as_str = str(path)
            if as_str not in seen:
                seen.add(as_str)
                found.append(as_str)
    return found


def _clean_env(
    device: str = "cpu",
    tool_venv_path: Path | str | None = None,
) -> dict[str, str]:
    """Build a clean env dict for subprocess execution."""
    from .system_info import capture_subprocess_env

    env = {k: v for k, v in os.environ.items() if k not in _BLOCKED_ENV_VARS}
    parent_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "").strip()
    tool_ld_library_path = ":".join(_discover_tool_ld_library_paths(tool_venv_path))
    merged_ld_library_path = _merge_colon_paths(
        tool_ld_library_path,
        parent_ld_library_path,
    )
    if merged_ld_library_path:
        env["LD_LIBRARY_PATH"] = merged_ld_library_path

    env["CUDA_VISIBLE_DEVICES"] = determine_visible_devices(device=device)
    if device == "cpu":
        # Prevent JAX from probing CUDA plugins in CPU-only subprocesses.
        env["JAX_PLATFORMS"] = "cpu"
    else:
        env.pop("JAX_PLATFORMS", None)

    # Capture for environment reporting
    capture_subprocess_env(env)

    return env


class PersistentWorker:
    """A long-running subprocess that accepts JSON requests on stdin and
    returns JSON responses on stdout.

    The subprocess runs ``_worker_bootstrap.py`` inside a tool's venv,
    which discovers and imports the tool's standalone script module. The
    model loads once on first request and stays resident for subsequent calls.

    Parameters
    ----------
    tool_name : str
        Name of the tool (e.g. ``"esm2"``, ``"blast"``).
    venv_path : Path
        Path to the tool's venv (e.g. ``.venvs/esm2_env``).
    script_path : Path
        Path to the standalone script (e.g. ``standalone/inference.py``).
    device : str
        Device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc.).
    """

    def __init__(
        self,
        tool_name: str,
        venv_path: Path,
        script_path: Path,
        device: str = "cpu",
    ) -> None:
        self.tool_name = tool_name
        self.venv_path = venv_path
        self.script_path = script_path
        self.device = device
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._stderr_thread: threading.Thread | None = None
        self._stderr_lines: list[str] = []

    @property
    def alive(self) -> bool:
        """Check if the worker subprocess is running."""
        return self._process is not None and self._process.poll() is None

    def _drain_stderr(self) -> None:
        """Background thread: read stderr lines from the worker process."""
        if self._process is None or self._process.stderr is None:
            return
        for line in self._process.stderr:
            text = line.rstrip("\n")
            if text:
                self._stderr_lines.append(text)
                logger.debug("[%s worker stderr] %s", self.tool_name, text)

    def start(self) -> None:
        """Spawn the worker subprocess if not already running."""
        if self.alive:
            return

        python_exe = str(self.venv_path / "bin" / "python")
        bootstrap = str(Path(__file__).parent / "_worker_bootstrap.py")
        env = _clean_env(self.device, tool_venv_path=self.venv_path)
        env["TOOL_VENV_PATH"] = str(self.venv_path)

        logger.debug(
            "Starting persistent worker for %s (script=%s, device=%s)",
            self.tool_name,
            self.script_path,
            self.device,
        )

        self._stderr_lines.clear()
        self._process = subprocess.Popen(
            [python_exe, bootstrap, str(self.script_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,  # line-buffered
        )

        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def send(
        self,
        input_dict: dict[str, Any],
        *,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Send a request to the worker and return the response.

        Thread-safe — only one request at a time.

        Parameters
        ----------
        input_dict : dict
            JSON-serializable input for the standalone script.
        timeout : int | None
            Maximum seconds to wait for a response.  *None* means block
            indefinitely.  On timeout the worker is killed (its state is
            unknown) and :class:`TimeoutError` is raised.

        Returns
        -------
        dict
            The script's JSON output.

        Raises
        ------
        RuntimeError
            If the worker crashes or returns an error.
        TimeoutError
            If the worker does not respond within *timeout* seconds.
        """
        with self._lock:
            if not self.alive:
                self.start()

            assert self._process is not None
            assert self._process.stdin is not None
            assert self._process.stdout is not None

            request_id = uuid.uuid4().hex[:8]
            request = {"id": request_id, "input": input_dict}
            request_line = json.dumps(request, separators=(",", ":")) + "\n"

            try:
                self._process.stdin.write(request_line)
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                stderr_tail = "\n".join(self._stderr_lines[-20:])
                raise RuntimeError(
                    f"Worker for {self.tool_name} crashed while sending request.\n"
                    f"stderr:\n{stderr_tail}"
                ) from exc

            # Read response line (with optional timeout).
            # select() on pipe fds is Unix-only; this module assumes Linux.
            if timeout is not None:
                ready, _, _ = select.select(
                    [self._process.stdout.fileno()], [], [], timeout
                )
                if not ready:
                    self.stop()
                    raise TimeoutError(
                        f"Worker for {self.tool_name} timed out after {timeout}s"
                    )
            response_line = self._process.stdout.readline()
            if not response_line:
                stderr_tail = "\n".join(self._stderr_lines[-20:])
                raise RuntimeError(
                    f"Worker for {self.tool_name} closed stdout unexpectedly.\n"
                    f"stderr:\n{stderr_tail}"
                )

            try:
                response = json.loads(response_line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Worker for {self.tool_name} returned invalid JSON: "
                    f"{response_line!r}"
                ) from exc

            if response.get("id") != request_id:
                raise RuntimeError(
                    f"Worker for {self.tool_name} returned mismatched request id: "
                    f"expected {request_id}, got {response.get('id')}"
                )

            if "error" in response:
                raise RuntimeError(
                    f"Worker for {self.tool_name} returned an error:\n"
                    f"{response['error']}"
                )

            return response["result"]

    def stop(self) -> None:
        """Terminate the worker subprocess."""
        if self._process is not None:
            try:
                if self._process.stdin and not self._process.stdin.closed:
                    self._process.stdin.close()
                self._process.terminate()
                self._process.wait(timeout=10)
            except Exception:
                self._process.kill()
                self._process.wait(timeout=5)
            finally:
                self._process = None
                logger.debug("Stopped persistent worker for %s", self.tool_name)
