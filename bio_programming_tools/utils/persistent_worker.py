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
import signal
import subprocess
import tempfile
import threading
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================


@lru_cache(maxsize=1)
def _load_bpt_env() -> dict[str, str]:
    """Load defaults from ``.bpt.env`` at the project root.

    Simple ``KEY=VALUE`` format (no shell expansion).  Surrounding quotes
    on values are stripped so ``BPT_MODEL_CACHE="/path"`` works like
    ``BPT_MODEL_CACHE=/path``.  Lines starting with ``#`` and blank lines
    are ignored.  Missing file returns ``{}``.
    """
    env_file = Path(__file__).parent.parent.parent / ".bpt.env"
    if not env_file.is_file():
        return {}
    result: dict[str, str] = {}
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def _normalize_progress_line(line: str) -> str:
    """Normalize progress bar lines to detect and skip similar consecutive lines.

    Detects common progress bar patterns and normalizes variable parts:
    - Progress bar visualizations: |███████▊  |, [=====>    ], etc.
    - Percentages: 5%, 100%
    - Ratios: 123/456, 1234 of 5678
    - Time estimates: [00:00<00:04], 1:23 elapsed
    - Rates: 980.92it/s, 1.2MB/s, 45.3%
    - Additional trailing info after common separators

    Returns the normalized line template for similarity comparison.
    """
    import re

    normalized = line

    # Normalize progress bar visualizations (|███|, [===>], etc.)
    normalized = re.sub(r'[\|\[\(\{][^\|\]\)\}]*[\|\]\)\}]', '|BAR|', normalized)

    # Normalize percentages
    normalized = re.sub(r'\d+\.?\d*%', 'N%', normalized)

    # Normalize count ratios
    normalized = re.sub(r'\d+/\d+', 'N/N', normalized)
    normalized = re.sub(r'\d+\s+of\s+\d+', 'N of N', normalized)

    # Normalize time estimates ([HH:MM<HH:MM], <1:23, elapsed: 0:45)
    normalized = re.sub(r'\[\d+:\d+(?::\d+)?<\d+:\d+(?::\d+)?\]', '[T<T]', normalized)
    normalized = re.sub(r'<\d+:\d+(?::\d+)?', '<T', normalized)
    normalized = re.sub(r'elapsed:\s*\d+:\d+(?::\d+)?', 'elapsed: T', normalized)

    # Normalize rates (it/s, MB/s) and sizes (1.2MB, 500KB)
    normalized = re.sub(r'\d+\.?\d*\s*[A-Za-z]+/s', 'N/s', normalized)
    normalized = re.sub(r'\d+\.?\d*\s*[KMGT]?B\b', 'NB', normalized)

    # Strip trailing info after separators in detected progress bars
    if 'N%' in normalized or 'N/N' in normalized or '|BAR|' in normalized:
        normalized = re.sub(r'(N%.*?N/s.*?),.*$', r'\1', normalized)
        normalized = re.sub(r'(N%.*?N/s.*?)\s+[-|]\s+.*$', r'\1', normalized)

    # Normalize remaining standalone numbers
    normalized = re.sub(r'\b\d+\.?\d*\b', 'N', normalized)

    return normalized


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
    # HuggingFace — model download paths
    "HF_HOME",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    # Model weights management
    "BPT_MODEL_CACHE",
    # Network proxy — tools download model weights and need proxy config
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
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
        Path to the tool's isolated venv.  Used to reconstruct PATH.
    tool_env_vars
        Parsed env_vars.txt contents (from :func:`_parse_env_vars_file`).
    """
    from .system_info import capture_subprocess_env

    env: dict[str, str] = {}

    # Copy only whitelisted vars from parent
    for var in _BASE_PASSTHROUGH:
        val = os.environ.get(var)
        if val is not None:
            env[var] = val

    # Apply .bpt.env defaults (env vars from parent always take precedence)
    for key, value in _load_bpt_env().items():
        env.setdefault(key, value)

    # Ensure HF_TOKEN is explicitly set even if the token was stored
    # in a file (~/.cache/huggingface/token or ~/.git-credentials).
    # Subprocesses may have a different HF_HOME, so file-based tokens
    # wouldn't be found at the redirected path.
    if "HF_TOKEN" not in env:
        from .auth import resolve_hf_token

        token = resolve_hf_token()
        if token:
            env["HF_TOKEN"] = token

    # Pass through per-tool weight directory overrides
    for var, val in os.environ.items():
        if var.startswith("BPT_") and var.endswith("_WEIGHTS_DIR"):
            env[var] = val

    # Reconstruct PATH: venv/bin > cuda/bin (GPU) > parent PATH > system dirs
    path_parts: list[str] = []
    if tool_env_path:
        path_parts.append(str(Path(tool_env_path) / "bin"))
    if device != "cpu":
        path_parts.append(_CUDA_BIN_DIR)
    for p in os.environ.get("PATH", "").split(":"):
        if p and p not in path_parts:
            path_parts.append(p)
    for d in _SYSTEM_PATH_DIRS:
        if d not in path_parts:
            path_parts.append(d)
    env["PATH"] = ":".join(path_parts)

    # Pass through parent's CUDA_VISIBLE_DEVICES (device placement handled by model.to())
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if parent_cvd is not None:
        env["CUDA_VISIBLE_DEVICES"] = parent_cvd

    # Generous timeouts for pip/uv — NFS extraction can exceed the defaults
    env.setdefault("UV_HTTP_TIMEOUT", "300")
    env.setdefault("PIP_DEFAULT_TIMEOUT", "300")

    # Prevent JAX from preallocating GPU memory at import time (harmless for PyTorch)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    # Set HF_HOME and TORCH_HOME based on BPT_MODEL_CACHE
    # Default: repo-local model_cache/ directory (survives env rebuilds)
    _default_cache = str(Path(__file__).parent.parent.parent / "model_cache")
    cache_mode = env.get("BPT_MODEL_CACHE", _default_cache)
    if cache_mode == "IN_ENV":
        # Each tool stores weights in its own venv
        if tool_env_path:
            env["HF_HOME"] = str(Path(tool_env_path) / "cache" / "huggingface")
            env["TORCH_HOME"] = str(Path(tool_env_path) / "cache" / "torch")
    elif cache_mode == "NONE":
        pass  # HF_HOME already copied from parent via _BASE_PASSTHROUGH
    else:
        # Absolute or relative path → shared directory
        cache_path = Path(cache_mode)
        env["HF_HOME"] = str(cache_path / "huggingface")
        if tool_env_path:
            env["TORCH_HOME"] = str(cache_path / "torch")

    # Inject compute environment detection (hardware-aware PyTorch/JAX specs)
    from .compute_deps import detect_compute_environment
    compute_env = detect_compute_environment()
    env.update(compute_env)

    # Apply tool-specific env vars from env_vars.txt
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

    # Point CONDA_PREFIX and VIRTUAL_ENV at tool env (read parent first for conda/lib below)
    parent_conda_prefix = os.environ.get("CONDA_PREFIX")
    if tool_env_path:
        env["CONDA_PREFIX"] = str(Path(tool_env_path))
        env["VIRTUAL_ENV"] = str(Path(tool_env_path))

    # Build LD_LIBRARY_PATH: tool [set] > parent LD > conda/lib
    ld_parts: list[str] = []

    # Tool-specific [set] LD_LIBRARY_PATH goes first (already in env from step 6)
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    if existing_ld:
        ld_parts.extend(existing_ld.split(":"))

    # Parent LD_LIBRARY_PATH (NVIDIA driver, CUDA, module-loaded libs, MKL, etc.)
    parent_ld = os.environ.get("LD_LIBRARY_PATH", "")
    for p in parent_ld.split(":"):
        if p and p not in ld_parts:
            ld_parts.append(p)

    # Conda shared libs (libgomp, libstdc++, etc.) — may not be in parent LD
    if parent_conda_prefix:
        conda_lib = str(Path(parent_conda_prefix) / "lib")
        if conda_lib not in ld_parts:
            ld_parts.append(conda_lib)

    if ld_parts:
        env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    else:
        env.pop("LD_LIBRARY_PATH", None)

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

        Skips consecutive duplicate and similar lines to reduce log noise from
        progress bars and repeated status messages.
        """
        if self._process is None or self._process.stderr is None:
            return
        prev_line: str | None = None
        prev_normalized: str | None = None
        for line in self._process.stderr:
            text = line.rstrip()  # Strip all trailing whitespace
            if text:
                self._stderr_lines.append(text)
                # Fast path: skip exact duplicates
                if text == prev_line:
                    continue
                # Slow path: normalize and check for progress bar similarity
                normalized = _normalize_progress_line(text)
                if normalized == prev_normalized:
                    continue
                logger.debug("[%s worker stderr] %s", self.tool_name, text)
                prev_line = text
                prev_normalized = normalized

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
            start_new_session=True,  # own process group for clean shutdown
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

            # Read header, skipping non-protocol lines (warnings/logs).
            # Header is either PROTO_LENGTH:<n> (pipe payload) or PROTO_FILE:<path>
            # (large payload written to a temp file by the worker).
            prev_stdout_line: str | None = None
            prev_stdout_normalized: str | None = None
            while True:
                header_line = self._process.stdout.readline()
                if not header_line:
                    stderr_tail = "\n".join(self._stderr_lines[-20:])
                    raise RuntimeError(
                        f"Worker for {self.tool_name} closed stdout unexpectedly.\n"
                        f"stderr:\n{stderr_tail}"
                    )

                header_line = header_line.strip()
                if header_line.startswith("PROTO_LENGTH:") or header_line.startswith("PROTO_FILE:"):
                    break
                # Non-protocol line (warning/log) — skip duplicates / progress bars
                if header_line == prev_stdout_line:
                    continue
                normalized = _normalize_progress_line(header_line)
                if normalized == prev_stdout_normalized:
                    continue
                logger.debug(
                    "[%s worker stdout] %s", self.tool_name, header_line
                )
                prev_stdout_line = header_line
                prev_stdout_normalized = normalized

            # Branch on header type
            if header_line.startswith("PROTO_FILE:"):
                # Large payload — worker wrote JSON to a temp file
                file_path = header_line.split(":", 1)[1]
                if not file_path.startswith(tempfile.gettempdir()):
                    logger.warning(
                        "Worker for %s sent file path outside temp directory: %s",
                        self.tool_name, file_path,
                    )
                try:
                    with open(file_path) as f:
                        response = json.load(f)
                finally:
                    try:
                        os.unlink(file_path)
                    except OSError:
                        pass
            else:
                # Standard LENGTH-prefixed pipe payload
                try:
                    json_length = int(header_line.split(":", 1)[1])
                except (ValueError, IndexError) as exc:
                    raise RuntimeError(
                        f"Worker for {self.tool_name} sent invalid LENGTH header: "
                        f"{header_line!r}"
                    ) from exc

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

    def _killpg(self, sig: int) -> None:
        """Send *sig* to the worker's process group, ignoring already-dead."""
        try:
            os.killpg(self._process.pid, sig)
        except OSError:
            pass

    def stop(self) -> None:
        """Terminate the worker and all of its child processes."""
        if self._process is not None:
            try:
                if self._process.stdin and not self._process.stdin.closed:
                    self._process.stdin.close()
                self._killpg(signal.SIGTERM)          # graceful shutdown
                self._process.wait(timeout=10)
            except Exception:
                self._killpg(signal.SIGKILL)          # force kill
                self._process.wait(timeout=5)         # reap zombie
            finally:
                for pipe in (self._process.stdout, self._process.stderr):
                    if pipe and not pipe.closed:
                        try:
                            pipe.close()
                        except Exception:
                            pass
                self._process = None
                logger.debug("Stopped persistent worker for %s", self.tool_name)
