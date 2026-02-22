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

# ============================================================================
# Whitelist-based environment isolation
# ============================================================================

# Vars passed through from the parent process to tool subprocesses.
# Everything else is blocked — conda, jupyter, mamba, etc. never leak.
_BASE_PASSTHROUGH = {
    # Identity — many tools/libs resolve ~ or check $USER
    "HOME",
    "USER",
    "LOGNAME",
    # Locale — C extensions and text processing break without these
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LC_MESSAGES",
    "LC_NUMERIC",
    "LC_TIME",
    "LC_COLLATE",
    "LC_MONETARY",
    "LC_PAPER",
    # Temp directories — subprocesses write scratch files here
    "TMPDIR",
    "TEMP",
    "TMP",
    # Shell — needed by subprocess.Popen when shell=True and by some tools
    "SHELL",
    # XDG dirs — model caches (HF, torch) respect these for default locations
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    # Network proxy — tools download model weights and need proxy config
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    # Conda/Micromamba environment info — needed for micromamba-created environments
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "CONDA_SHLVL",
}

# Standard system binary dirs — always present in reconstructed PATH
_SYSTEM_PATH_DIRS = [
    "/usr/local/sbin",
    "/usr/local/bin",
    "/usr/sbin",
    "/usr/bin",
    "/sbin",
    "/bin",
]

# Prepended to PATH for GPU tools so nvcc/nvidia-smi are available
_CUDA_BIN_DIR = "/usr/local/cuda/bin"


def _parse_env_vars_file(
    path: Path | None,
) -> dict[str, list[str]]:
    """Parse a ``standalone/env_vars.txt`` file.

    Returns a dict with ``"passthrough"`` and ``"set"`` keys, each
    mapping to a list of variable names (passthrough) or ``KEY=VALUE``
    strings (set).  Missing or empty files return empty lists.

    File format::

        [passthrough]
        HF_TOKEN
        HUGGING_FACE_HUB_TOKEN

        [set]
        MY_VAR=${VENV_PATH}/data

    Lines starting with ``#`` are comments.  Blank lines are ignored.
    """
    result: dict[str, list[str]] = {"passthrough": [], "set": []}
    if path is None or not path.exists():
        return result

    current_section: str | None = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].lower()
            if section in result:
                current_section = section
            else:
                logger.warning(
                    "Unknown section %r in %s, ignoring", section, path
                )
                current_section = None
            continue
        if current_section is not None:
            result[current_section].append(line)
    return result


def _discover_tool_ld_library_paths(tool_env_path: Path | str | None) -> list[str]:
    """Discover CUDA-related library directories from a tool venv."""
    if not tool_env_path:
        return []

    env_path = Path(tool_env_path)
    candidates: list[Path] = [
        env_path / "cuda_env" / "lib",
        env_path / "cuda_env" / "lib64",
    ]
    candidates.extend(env_path.glob("lib/python*/site-packages/nvidia/*/lib"))

    found: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        if path.is_dir():
            as_str = str(path)
            if as_str not in seen:
                seen.add(as_str)
                found.append(as_str)
    return found


def _build_subprocess_env(
    device: str = "cpu",
    tool_env_path: Path | str | None = None,
    tool_env_vars: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    """Build a clean env dict for subprocess execution.

    Uses a **whitelist** approach: starts with an empty dict and only
    copies explicitly allowed variables from the parent environment.
    This prevents conda, jupyter, mamba, and other host-specific vars
    from leaking into isolated tool venvs.

    Parameters
    ----------
    device
        Target device (``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc.).
    tool_env_path
        Path to the tool's isolated venv.  Used to reconstruct PATH
        and LD_LIBRARY_PATH.
    tool_env_vars
        Parsed env_vars.txt contents (from :func:`_parse_env_vars_file`).
        ``"passthrough"`` entries are copied from the parent env;
        ``"set"`` entries are literal ``KEY=VALUE`` assignments with
        optional ``${VENV_PATH}`` interpolation.
    """
    from .system_info import capture_subprocess_env

    env: dict[str, str] = {}

    # 1. Copy only whitelisted vars from parent
    for var in _BASE_PASSTHROUGH:
        val = os.environ.get(var)
        if val is not None:
            env[var] = val

    # 2. Reconstruct PATH: venv/bin + (cuda bin for GPU) + system dirs
    path_parts: list[str] = []
    if tool_env_path:
        path_parts.append(str(Path(tool_env_path) / "bin"))
    if device != "cpu":
        path_parts.append(_CUDA_BIN_DIR)
    path_parts.extend(_SYSTEM_PATH_DIRS)
    env["PATH"] = ":".join(path_parts)

    # 3. Reconstruct LD_LIBRARY_PATH from venv CUDA libs only (no parent)
    tool_ld_paths = _discover_tool_ld_library_paths(tool_env_path)
    if tool_ld_paths:
        env["LD_LIBRARY_PATH"] = ":".join(tool_ld_paths)

    # 4. Device visibility
    env["CUDA_VISIBLE_DEVICES"] = determine_visible_devices(device=device)
    if device == "cpu":
        env["JAX_PLATFORMS"] = "cpu"

    # 5. Per-venv torch cache isolation
    if tool_env_path:
        env["TORCH_HOME"] = str(Path(tool_env_path) / "cache" / "torch")

    # 6. Inject compute environment detection (hardware-aware PyTorch/JAX specs)
    from .compute_deps import detect_compute_environment
    compute_env = detect_compute_environment()
    env.update(compute_env)

    # 7. Apply tool-specific env vars from env_vars.txt
    if tool_env_vars:
        venv_str = str(Path(tool_env_path)) if tool_env_path else ""

        for var_name in tool_env_vars.get("passthrough", []):
            val = os.environ.get(var_name)
            if val is not None:
                env[var_name] = val
            else:
                logger.debug(
                    "env_vars.txt requests passthrough of %r but it is "
                    "not set in the parent environment",
                    var_name,
                )

        for entry in tool_env_vars.get("set", []):
            if "=" not in entry:
                logger.warning("Malformed [set] entry in env_vars.txt: %r", entry)
                continue
            key, val = entry.split("=", 1)
            env[key] = val.replace("${VENV_PATH}", venv_str)

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
    env_path : Path
        Path to the tool's environment (e.g. ``tool_envs/esm2_env``).
    script_path : Path
        Path to the standalone script (e.g. ``standalone/inference.py``).
    device : str
        Device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc.).
    tool_env_vars : dict | None
        Parsed env_vars.txt contents for this tool.
    """

    def __init__(
        self,
        tool_name: str,
        env_path: Path,
        script_path: Path,
        device: str = "cpu",
        tool_env_vars: dict[str, list[str]] | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.env_path = env_path
        self.script_path = script_path
        self.device = device
        self.tool_env_vars = tool_env_vars
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._stderr_thread: threading.Thread | None = None
        self._stderr_lines: list[str] = []

    @property
    def alive(self) -> bool:
        """Check if the worker subprocess is running."""
        return self._process is not None and self._process.poll() is None

    def _drain_stderr(self) -> None:
        """Background thread: read stderr lines from the worker process.

        Deduplicates consecutive identical lines to reduce log noise from
        progress bars and repeated status messages.
        """
        if self._process is None or self._process.stderr is None:
            return
        prev_line: str | None = None
        repeat_count = 0
        for line in self._process.stderr:
            text = line.rstrip("\n")
            if text:
                self._stderr_lines.append(text)
                if text == prev_line:
                    repeat_count += 1
                    continue
                if repeat_count > 0:
                    logger.debug(
                        "[%s worker stderr] (previous line repeated %d more times)",
                        self.tool_name,
                        repeat_count,
                    )
                repeat_count = 0
                logger.debug("[%s worker stderr] %s", self.tool_name, text)
                prev_line = text
        if repeat_count > 0:
            logger.debug(
                "[%s worker stderr] (previous line repeated %d more times)",
                self.tool_name,
                repeat_count,
            )

    def start(self) -> None:
        """Spawn the worker subprocess if not already running."""
        if self.alive:
            return

        python_exe = str(self.env_path / "bin" / "python")
        bootstrap = str(Path(__file__).parent / "_worker_bootstrap.py")
        env = _build_subprocess_env(
            self.device,
            tool_env_path=self.env_path,
            tool_env_vars=self.tool_env_vars,
        )
        env["TOOL_VENV_PATH"] = str(self.env_path)

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

            # Read length header (length-prefixed protocol)
            # Format: "LENGTH:<bytes>\n"
            # This allows workers to output warnings/logs without breaking JSON parsing
            # Skip any non-LENGTH lines (warnings, logs, etc.) until we find the header
            prev_stdout_line: str | None = None
            stdout_repeat_count = 0
            while True:
                length_line = self._process.stdout.readline()
                if not length_line:
                    stderr_tail = "\n".join(self._stderr_lines[-20:])
                    raise RuntimeError(
                        f"Worker for {self.tool_name} closed stdout unexpectedly.\n"
                        f"stderr:\n{stderr_tail}"
                    )

                length_line = length_line.strip()
                if length_line.startswith("LENGTH:"):
                    if stdout_repeat_count > 0:
                        logger.debug(
                            "[%s worker stdout] (previous line repeated %d more times)",
                            self.tool_name,
                            stdout_repeat_count,
                        )
                    break
                # Non-LENGTH line (warning/log) - log and skip, dedup consecutive
                if length_line == prev_stdout_line:
                    stdout_repeat_count += 1
                    continue
                if stdout_repeat_count > 0:
                    logger.debug(
                        "[%s worker stdout] (previous line repeated %d more times)",
                        self.tool_name,
                        stdout_repeat_count,
                    )
                stdout_repeat_count = 0
                logger.debug(
                    "[%s worker stdout] %s", self.tool_name, length_line
                )
                prev_stdout_line = length_line

            try:
                json_length = int(length_line.split(":", 1)[1])
            except (ValueError, IndexError) as exc:
                raise RuntimeError(
                    f"Worker for {self.tool_name} sent invalid LENGTH header: "
                    f"{length_line!r}"
                ) from exc

            # Read exactly json_length bytes
            response_bytes = self._process.stdout.read(json_length)
            if len(response_bytes) != json_length:
                raise RuntimeError(
                    f"Worker for {self.tool_name} sent incomplete JSON: "
                    f"expected {json_length} bytes, got {len(response_bytes)}"
                )

            try:
                response = json.loads(response_bytes)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Worker for {self.tool_name} returned invalid JSON: "
                    f"{response_bytes!r}"
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
