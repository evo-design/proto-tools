"""proto_tools/utils/persistent_worker.py.

Manages a subprocess that stays alive between calls, communicating via
stdin/stdout JSON-line protocol. This avoids reloading models on every call.
"""

import collections
import contextlib
import functools
import json
import logging
import os
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import IO, Any

logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

# Default per-worker bounded stderr buffer size, used by _crash_context. Override via PROTO_WORKER_STDERR_BUFFER_LINES.
_DEFAULT_STDERR_BUFFER_LINES = 20

# Read response framing: chunk size per os.read, and the post-header window to drain the body.
_READ_CHUNK = 65536
_BODY_GRACE_SECONDS = 30


def _verbose_to_log_level(verbose: int) -> int:
    """Map the 0-3 ``verbose`` scale to a stdlib logging level for console handlers.

    | verbose | log level |
    | 0       | WARNING   |
    | 1       | INFO      |
    | 2       | DEBUG     |
    | 3       | DEBUG     | (raw subprocess stderr also teed)
    """
    if verbose <= 0:
        return logging.WARNING
    if verbose == 1:
        return logging.INFO
    return logging.DEBUG


def _apply_verbose_to_console_handlers(verbose: int) -> None:
    """Set every console ``StreamHandler`` reachable from ``proto_tools`` to the verbose-mapped level.

    Filters live on the *handlers*, not on the originating logger, so file
    handlers and :class:`SpinnerFromLogsHandler` continue to receive records at
    every level. Walks ``proto_tools`` and ``root`` since ``setup_logging``
    attaches its console handler to root.
    """
    target_level = _verbose_to_log_level(verbose)
    for logger_name in ("proto_tools", ""):
        for h in logging.getLogger(logger_name).handlers:
            if type(h) is logging.StreamHandler:
                h.setLevel(target_level)


def _stderr_buffer_lines() -> int:
    """Read ``PROTO_WORKER_STDERR_BUFFER_LINES`` as a positive int, else default."""
    raw = os.environ.get("PROTO_WORKER_STDERR_BUFFER_LINES")
    if raw is None:
        return _DEFAULT_STDERR_BUFFER_LINES
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_STDERR_BUFFER_LINES
    return max(1, n)


# ============================================================================
# Subprocess stderr demultiplexing (shared by PersistentWorker and one-shot)
# ============================================================================
# Tagged lines (``\x00LOG\x00<json>``) are demuxed by `_reemit_tagged_line`; untagged lines hit `_handle_raw_stderr_line` (ring buffer + optional raw_tee at verbose level 3). See notes/logging.md for the full architecture.


def _handle_raw_stderr_line(line: str, buffer: "collections.deque[str]", raw_tee: bool) -> None:
    r"""Handle one untagged stderr line: ring buffer + optional tee to parent stderr.

    Progress-bar lines (tqdm-style ``\r<frame1>\r...<final>\n``) keep only the
    last ``\r``-separated segment so the final completion state survives.

    Args:
        line (str): The raw line read from the subprocess's stderr.
        buffer (collections.deque[str]): Bounded ring buffer to append to.
        raw_tee (bool): If True, mirror the line to the parent's ``sys.stderr``.
    """
    if "\r" in line:
        line = line.rsplit("\r", 1)[-1]
    text = line.rstrip()
    if not text:
        return
    buffer.append(text)
    if raw_tee:
        sys.stderr.write(line)
        sys.stderr.flush()


def _reemit_tagged_line(
    parent_logger: logging.Logger,
    line: str,
    buffer: "collections.deque[str]",
) -> None:
    r"""Parse one ``\x00LOG\x00<json>\n`` line and re-emit it on the parent logger.

    On JSON parse failure, the line is treated as untagged via
    :func:`_handle_raw_stderr_line` with ``raw_tee=False`` (defensive fallback;
    a malformed tag is almost certainly noise that happened to start with the
    sentinel — don't show it to the user even at verbose level 2).

    Args:
        parent_logger (logging.Logger): Logger under which to re-emit
            (typically ``logging.getLogger("proto_tools.worker.{toolkit}")``).
        line (str): The raw tagged line, including the ``\x00LOG\x00`` prefix.
        buffer (collections.deque[str]): Ring buffer for the malformed-line
            fallback path.
    """
    from proto_tools.utils.logging_config import _TAG_PREFIX

    try:
        payload = json.loads(line[len(_TAG_PREFIX) :])
    except (json.JSONDecodeError, ValueError):
        _handle_raw_stderr_line(line, buffer, raw_tee=False)
        return
    record_name: str = payload.get("name", "unknown")
    # Strip subprocess-side "worker." prefix; parent_logger is already proto_tools.worker.{toolkit}, so the suffix re-attaches as proto_tools.worker.{toolkit}.{name}.
    suffix = record_name.removeprefix("worker.")
    target = parent_logger.getChild(suffix) if suffix else parent_logger
    level_name: str = payload.get("level", "INFO")
    # Map level name to int via attribute lookup (str→int via getLevelName is deprecated).
    level_attr = getattr(logging, level_name, None)
    level = level_attr if isinstance(level_attr, int) else logging.INFO
    msg: str = payload.get("msg", "")
    update_status = bool(payload.get("update_status", False))
    target.log(level, "%s", msg, extra={"update_status": update_status})


def _drain_subprocess_stderr(
    stream: IO[str],
    parent_logger: logging.Logger,
    buffer: "collections.deque[str]",
    raw_tee: bool,
) -> None:
    """Read ``stream`` until EOF, demuxing each line to the appropriate path.

    Used by :meth:`PersistentWorker._drain_stderr` (long-lived worker) and by
    :meth:`ToolInstance._run_oneshot` (ephemeral subprocess) so both subprocess
    modes produce the same parent-side records and ring-buffer state.

    Args:
        stream (IO[str]): Subprocess stderr (text-mode pipe).
        parent_logger (logging.Logger): Logger under which to re-emit tagged
            records (children created via ``getChild``).
        buffer (collections.deque[str]): Bounded ring buffer for untagged lines.
        raw_tee (bool): If True, untagged lines are mirrored to parent stderr.
    """
    from proto_tools.utils.logging_config import _TAG_PREFIX

    for line in stream:
        if line.startswith(_TAG_PREFIX):
            _reemit_tagged_line(parent_logger, line, buffer)
            continue
        _handle_raw_stderr_line(line, buffer, raw_tee)


# ============================================================================
# Whitelist-based environment isolation
# ============================================================================

# Vars passed through from the parent process to tool subprocesses.
# Everything else is blocked; conda, jupyter, mamba, etc. never leak.
_BASE_PASSTHROUGH = {
    # Identity: many tools/libs resolve ~ or check $USER
    "HOME",
    "USER",
    "LOGNAME",
    # Locale: C extensions and text processing break without these
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LC_MESSAGES",
    "LC_NUMERIC",
    "LC_TIME",
    "LC_COLLATE",
    "LC_MONETARY",
    "LC_PAPER",
    # Temp directories: subprocesses write scratch files here
    "TMPDIR",
    "TEMP",
    "TMP",
    # Shell: needed by subprocess.Popen when shell=True and by some tools
    "SHELL",
    # XDG dirs: model caches (HF, torch) respect these for default locations
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    # HuggingFace: model download paths
    "HF_HOME",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    # Model weights management
    "PROTO_HOME",
    "PROTO_MODEL_CACHE",
    # uv/pip package caches: let user-set values override our defaults
    "UV_CACHE_DIR",
    "PIP_CACHE_DIR",
    # Network proxy: tools download model weights and need proxy config
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}

# Standard system binary dirs, always present in reconstructed PATH
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


# Fallback locations searched if ldconfig is unavailable or doesn't list libcuda.
_DRIVER_LIB_FALLBACK_DIRS = (
    "/usr/lib64",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
    "/usr/lib/wsl/lib",
    "/lib64",
)


@functools.lru_cache(maxsize=1)
def _find_driver_lib_dir() -> str | None:
    """Locate the host directory containing ``libcuda.so.1`` (NVIDIA driver lib).

    Pip-bundled CUDA wheels never ship the driver lib (it must come from the
    installed driver). When ``[no_passthrough] LD_LIBRARY_PATH`` strips the
    parent's value to avoid ABI-shadowing pip-bundled libs, this helper finds
    just the driver dir so subprocesses can still ``dlopen`` it.

    Probes in order: ``ldconfig -p`` (canonical on bare metal), parent
    ``LD_LIBRARY_PATH`` entries (catches apptainer/singularity-style containers
    where the driver is bind-mounted under ``/.singularity.d/libs/`` or
    ``/usr/local/nvidia/lib*``), then hardcoded fallback dirs.

    Returns:
        str | None: Absolute path to the driver-lib directory, or ``None`` if
            not found.
    """
    try:
        out = subprocess.check_output(["ldconfig", "-p"], text=True, stderr=subprocess.DEVNULL, timeout=2)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, OSError):
        out = ""
    for line in out.splitlines():
        if "libcuda.so.1" not in line or "=>" not in line:
            continue
        _, lib_path = line.rsplit("=>", 1)
        p = Path(lib_path.strip())
        if p.exists():
            return str(p.parent)

    for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if entry and (Path(entry) / "libcuda.so.1").exists():
            return entry

    for candidate in _DRIVER_LIB_FALLBACK_DIRS:
        if (Path(candidate) / "libcuda.so.1").exists():
            return candidate

    return None


def _parse_env_vars_file(
    path: Path | None,
) -> dict[str, list[str]]:
    """Parse a ``standalone/env_vars.txt`` file.

    Returns a dict with ``"passthrough"``, ``"set"``, and ``"no_passthrough"``
    keys.  ``passthrough`` and ``no_passthrough`` map to variable names;
    ``set`` maps to ``KEY=VALUE`` strings.  Missing or empty files return
    empty lists for all sections.

    Args:
        path (Path | None): Path to the ``env_vars.txt`` file, or None.

    File format::

        [passthrough]
        HF_TOKEN
        HUGGING_FACE_HUB_TOKEN

        [set]
        MY_VAR=${VENV_PATH}/data

        [no_passthrough]
        LD_LIBRARY_PATH

    ``[no_passthrough]`` blocks the parent process's value for the named
    var from leaking into the subprocess: the whitelisted parent copy
    is skipped, and for ``LD_LIBRARY_PATH`` neither the parent's value
    nor ``$CONDA_PREFIX/lib`` get appended.  The host directory holding
    ``libcuda.so.1`` is still appended (driver libs are never bundled
    in pip wheels), so opting out doesn't break GPU access on hosts
    that rely on the system driver.  ``[set]`` values still apply.
    Use this for tools whose pip-bundled libs (e.g. JAX's RPATH'd
    CUDA wheels) get ABI-shadowed by the parent's libs.

    Lines starting with ``#`` are comments.  Blank lines are ignored.
    """
    result: dict[str, list[str]] = {"passthrough": [], "set": [], "no_passthrough": []}
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
                logger.warning("Unknown section %r in %s, ignoring", section, path)
                current_section = None
            continue
        if current_section is not None:
            result[current_section].append(line)
    return result


def _build_subprocess_env(
    device: str = "cpu",
    tool_env_path: Path | str | None = None,
    tool_env_vars: dict[str, list[str]] | None = None,
    env_overrides: dict[str, str] | None = None,
    pin_visible_devices: bool = False,
) -> dict[str, str]:
    """Build a clean env dict for subprocess execution.

    Uses a **whitelist** approach: starts with an empty dict and only
    copies explicitly allowed variables from the parent environment.
    This prevents conda, jupyter, mamba, and other host-specific vars
    from leaking into isolated tool venvs.

    Args:
        device (str): Target device (``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc.).
        tool_env_path (Path | str | None): Path to the tool's isolated venv.
        tool_env_vars (dict[str, list[str]] | None): Parsed env_vars.txt contents.
        env_overrides (dict[str, str] | None): Caller-supplied env vars applied
            as the final layer, after every other source. Takes precedence over
            passthroughs, ``env_vars.txt``, and tool-cache defaults. Used by
            ToolPool's CPU fan-out path to pin per-worker
            ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` / etc., but available to
            any caller that needs to override a specific variable for one
            worker subprocess.
        pin_visible_devices (bool): Restrict ``CUDA_VISIBLE_DEVICES`` to ``device``'s physical GPU(s) (local cuda:0..N-1) instead of the parent's value. For JAX tools.
    """
    from proto_tools.utils.system_info import capture_subprocess_env

    env: dict[str, str] = {}

    # Vars in [no_passthrough] don't inherit anything from the parent.
    no_passthrough = set((tool_env_vars or {}).get("no_passthrough", []))

    # Copy only whitelisted vars from parent
    for var in _BASE_PASSTHROUGH:
        if var in no_passthrough:
            continue
        val = os.environ.get(var)
        if val is not None:
            env[var] = val

    # Ensure HF_TOKEN is explicitly set even if the token was stored
    # in a file (~/.cache/huggingface/token or ~/.git-credentials).
    # Subprocesses may have a different HF_HOME, so file-based tokens
    # wouldn't be found at the redirected path.
    if "HF_TOKEN" not in env:
        from proto_tools.utils.auth import resolve_hf_token

        token = resolve_hf_token()
        if token:
            env["HF_TOKEN"] = token

    # Pass through per-tool weight directory overrides
    env.update(
        {var: val for var, val in os.environ.items() if var.startswith("PROTO_") and var.endswith("_WEIGHTS_DIR")}
    )

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

    # CUDA_VISIBLE_DEVICES: pinned workers (JAX) see only their physical GPU(s) as local cuda:0..N-1; else pass parent's through.
    if pin_visible_devices:
        from proto_tools.utils.device import determine_visible_devices

        env["CUDA_VISIBLE_DEVICES"] = determine_visible_devices(device)  # "" for cpu; physical indices for cuda
        if device == "cpu":
            env["JAX_PLATFORMS"] = "cpu"  # hide GPUs so JAX initializes on CPU only
    else:
        parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if parent_cvd is not None:
            env["CUDA_VISIBLE_DEVICES"] = parent_cvd

    # Generous timeouts for pip/uv; NFS extraction can exceed the defaults
    env.setdefault("UV_HTTP_TIMEOUT", "300")
    env.setdefault("PIP_DEFAULT_TIMEOUT", "300")

    # Route uv/pip caches under PROTO_HOME so all proto-tools disk usage
    # is consolidated and cleanable atomically. Respects user overrides
    # (UV_CACHE_DIR / PIP_CACHE_DIR are copied in from _BASE_PASSTHROUGH
    # above; setdefault only fills when unset).
    from proto_tools.utils.proto_home import get_proto_home

    env.setdefault("UV_CACHE_DIR", str(get_proto_home() / "uv_cache"))
    env.setdefault("PIP_CACHE_DIR", str(get_proto_home() / "pip_cache"))

    # Prevent JAX from preallocating GPU memory at import time (harmless for PyTorch)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    # Always propagate PROTO_HOME so standalone helpers resolve the same root
    if "PROTO_HOME" not in env:
        env["PROTO_HOME"] = str(get_proto_home())

    # Set HF_HOME and TORCH_HOME based on PROTO_MODEL_CACHE
    # Default: PROTO_HOME/proto_model_cache/ directory (survives env rebuilds)
    _default_cache = str(get_proto_home() / "proto_model_cache")
    cache_mode = env.get("PROTO_MODEL_CACHE", _default_cache)
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
    from proto_tools.utils.compute_deps import detect_compute_environment

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
                    "env_vars.txt requests passthrough of %r but it is not set in the parent environment",
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
    # (parent LD and conda /lib are skipped if LD_LIBRARY_PATH is in [no_passthrough];
    # in that case just the host driver-lib dir is appended so libcuda.so.1 still resolves)
    ld_parts: list[str] = []
    ld_no_passthrough = "LD_LIBRARY_PATH" in no_passthrough

    # Tool-specific [set] LD_LIBRARY_PATH goes first (already in env from step 6)
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    if existing_ld:
        ld_parts.extend(existing_ld.split(":"))

    if not ld_no_passthrough:
        # Parent LD_LIBRARY_PATH (NVIDIA driver, CUDA, module-loaded libs, MKL, etc.)
        parent_ld = os.environ.get("LD_LIBRARY_PATH", "")
        for p in parent_ld.split(":"):
            if p and p not in ld_parts:
                ld_parts.append(p)

        # Conda shared libs (libgomp, libstdc++, etc.), may not be in parent LD
        if parent_conda_prefix:
            conda_lib = str(Path(parent_conda_prefix) / "lib")
            if conda_lib not in ld_parts:
                ld_parts.append(conda_lib)
    else:
        # libcuda.so.1 (host driver lib) is never in pip wheels; append just its dir.
        driver_dir = _find_driver_lib_dir()
        if driver_dir:
            if driver_dir not in ld_parts:
                ld_parts.append(driver_dir)
        else:
            logger.warning(
                "Could not locate libcuda.so.1 via ldconfig, parent LD_LIBRARY_PATH, "
                "or fallback dirs %s; subprocess may fail to load the CUDA driver. "
                "This is expected on hosts where pip-bundled CUDA wheels are "
                "self-sufficient (e.g. on some managed GPU platforms).",
                list(_DRIVER_LIB_FALLBACK_DIRS),
            )

    if ld_parts:
        env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    else:
        env.pop("LD_LIBRARY_PATH", None)

    # Caller-supplied overrides (applied last so they win over every other source).
    if env_overrides:
        env.update(env_overrides)

    # Capture for environment reporting
    capture_subprocess_env(env)

    return env


class PersistentWorker:
    """A long-running subprocess that accepts JSON requests on stdin and.

    returns JSON responses on stdout.

    The subprocess runs ``_worker_bootstrap.py`` inside a tool's venv,
    which discovers and imports the tool's standalone script module. The
    model loads once on first request and stays resident for subsequent calls.

    Parameters
    ----------
    toolkit : str
        Name of the tool (e.g. ``"esm2"``, ``"blast"``).
    env_path : Path
        Path to the tool's environment (e.g. ``tool_envs/esm2_env``).
    script_path : Path
        Path to the standalone script (e.g. ``standalone/inference.py``).
    device : str
        Device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc.).
    tool_env_vars : dict | None
        Parsed env_vars.txt contents for this tool.
    verbose : int
        Effective verbosity on the 0-3 scale (already merged with
        ``PROTO_WORKER_VERBOSE`` by the caller). Drives both the parent-side
        log level on ``proto_tools.worker.{toolkit}`` and whether untagged
        subprocess stderr is teed to the parent's ``sys.stderr``. Latched at
        construction time so the drain thread never races a mid-run env-var
        toggle.
    env_overrides : dict | None
        Caller-supplied env vars applied as the final layer when the worker
        subprocess is spawned. Used by ToolPool's CPU fan-out path to pin
        per-worker thread budgets (``OMP_NUM_THREADS`` etc.); harmless to
        any other caller.
    """

    def __init__(
        self,
        toolkit: str,
        env_path: Path,
        script_path: Path,
        device: str = "cpu",
        tool_env_vars: dict[str, list[str]] | None = None,
        verbose: int = 0,
        env_overrides: dict[str, str] | None = None,
        pin_visible_devices: bool = False,
    ) -> None:
        """Initialize PersistentWorker."""
        self.toolkit = toolkit
        self.env_path = env_path
        self.script_path = script_path
        self.device = device
        self.tool_env_vars = tool_env_vars
        self._verbose = verbose
        self._env_overrides = env_overrides
        # When True, spawn with CUDA_VISIBLE_DEVICES restricted to ``device``'s physical GPU(s) (local cuda:0..N-1).
        self._pin_visible_devices = pin_visible_devices
        self._process: subprocess.Popen | None = None  # type: ignore[type-arg]
        self._lock = threading.Lock()
        self._stderr_thread: threading.Thread | None = None
        # Bounded ring buffer for crash context; size from PROTO_WORKER_STDERR_BUFFER_LINES (default 20).
        self._stderr_lines: collections.deque[str] = collections.deque(maxlen=_stderr_buffer_lines())

    @property
    def alive(self) -> bool:
        """Check if the worker subprocess is running."""
        return self._process is not None and self._process.poll() is None

    def _crash_context(self) -> str:
        """Return ``"process exit=X; last stderr: Y"`` for embedding in error messages."""
        exit_code = self._process.poll() if self._process is not None else "no-process"
        # Buffer is a bounded deque (default 20 lines); join the whole thing.
        stderr_tail = " | ".join(self._stderr_lines) or "<no stderr>"
        return f"process exit={exit_code}; last stderr: {stderr_tail}"

    def _drain_stderr(self) -> None:
        """Drain this worker's stderr via the shared :func:`_drain_subprocess_stderr` helper.

        Applies ``self._verbose`` to the ``proto_tools`` console StreamHandler so
        records visible in the terminal match the documented level mapping
        (0=WARNING, 1=INFO, 2=DEBUG, 3=DEBUG+raw_tee). The toolkit logger stays
        at DEBUG so all records still propagate to file handlers and to
        :class:`SpinnerFromLogsHandler` for spinner takeover.
        """
        if self._process is None or self._process.stderr is None:
            return
        parent_logger = logging.getLogger(f"proto_tools.worker.{self.toolkit}")
        _apply_verbose_to_console_handlers(self._verbose)
        raw_tee = self._verbose >= 3
        _drain_subprocess_stderr(self._process.stderr, parent_logger, self._stderr_lines, raw_tee)

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
            env_overrides=self._env_overrides,
            pin_visible_devices=self._pin_visible_devices,
        )
        env["TOOL_VENV_PATH"] = str(self.env_path)

        logger.debug(
            "Starting persistent worker for %s (script=%s, device=%s)",
            self.toolkit,
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

        Thread-safe. Only one request at a time.

        Args:
            input_dict (dict[str, Any]): JSON-serializable input for the standalone script.
            timeout (int | None): Maximum seconds to wait for a response. None means
                block indefinitely. On timeout the worker is killed and TimeoutError is raised.

        Returns:
            dict[str, Any]: The script's JSON output.

        Raises:
            RuntimeError: If the worker crashes or returns an error.
            TimeoutError: If the worker does not respond within *timeout* seconds.
        """
        with self._lock:
            if not self.alive:
                self.start()

            if self._process is None or self._process.stdin is None or self._process.stdout is None:
                raise RuntimeError(f"{self.toolkit} worker not ready ({self._crash_context()})")

            request_id = uuid.uuid4().hex[:8]
            request = {"id": request_id, "input": input_dict}
            request_line = json.dumps(request, separators=(",", ":")) + "\n"

            try:
                self._process.stdin.write(request_line)
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise RuntimeError(
                    f"{self.toolkit} worker crashed sending request {request_id} ({self._crash_context()})"
                ) from exc

            # Kill the worker if we fail to read a matching response. The worker is in its own
            # process group and survives parent-side cancellation by the host platform;
            # without this, the worker's late response sits in the pipe and corrupts the next send.
            response_matched = False
            try:
                response = self._read_response(request_id, timeout)

                if response.get("id") != request_id:
                    raise RuntimeError(
                        f"{self.toolkit} worker returned mismatched id "
                        f"(expected {request_id}, got {response.get('id')}); {self._crash_context()}"
                    )
                response_matched = True
            finally:
                if not response_matched:
                    self.stop()

            if "error" in response:
                # Capture crash context before any stop() (which clears the process state it reads).
                error_context = self._crash_context()
                from proto_tools.utils.device import is_gpu_acquisition_error

                # A GPU context-acquisition failure poisons this process (the CUDA/JAX backend caches the init failure); tear it down so the next dispatch (e.g. a registry retry) cold-starts a fresh worker.
                if is_gpu_acquisition_error(response["error"]):
                    self.stop()
                raise RuntimeError(
                    f"{self.toolkit} worker error for request {request_id}: {response['error']}; {error_context}"
                )

            return response["result"]  # type: ignore[no-any-return]

    def _read_response(self, request_id: str, timeout: int | None) -> dict[str, Any]:
        """Read one framed worker response, bounding the whole wait by *timeout* (None = block)."""
        if timeout is None:
            return self._read_response_blocking(request_id)
        return self._read_response_deadline(request_id, timeout)

    def _read_response_blocking(self, request_id: str) -> dict[str, Any]:
        """Legacy buffered read path (unbounded); used only when no timeout is set."""
        assert self._process is not None and self._process.stdout is not None
        # Header is PROTO_LENGTH:<n> (pipe payload) or PROTO_FILE:<path> (temp-file payload).
        # proto_tools own logs route to stderr; anything else on stdout is third-party noise we skip.
        while True:
            header_line = self._process.stdout.readline()
            if not header_line:
                raise RuntimeError(
                    f"{self.toolkit} worker closed stdout before responding to request {request_id} "
                    f"({self._crash_context()})"
                )
            header_line = header_line.strip()
            if header_line.startswith(("PROTO_LENGTH:", "PROTO_FILE:")):
                break

        if header_line.startswith("PROTO_FILE:"):
            return self._read_file_payload(header_line)

        try:
            json_length = int(header_line.split(":", 1)[1])
        except (ValueError, IndexError) as exc:
            raise RuntimeError(
                f"{self.toolkit} worker sent invalid LENGTH header {header_line!r} "
                f"for request {request_id} ({self._crash_context()})"
            ) from exc

        response_bytes = self._process.stdout.read(json_length)
        if len(response_bytes) != json_length:
            raise RuntimeError(
                f"{self.toolkit} worker sent incomplete JSON for request {request_id}: "
                f"expected {json_length} bytes, got {len(response_bytes)} ({self._crash_context()})"
            )
        return self._decode_body(request_id, response_bytes)

    def _read_response_deadline(self, request_id: str, timeout: int) -> dict[str, Any]:
        """Deadline-bounded raw-fd read: the entire wait for a response is capped at *timeout*.

        Reads raw bytes off the stdout fd (not the buffered text stream, which is select-unsafe)
        and frames the protocol itself, so a worker that emits native fd-1 noise and then stalls
        still trips the timeout instead of blocking forever in readline().
        """
        assert self._process is not None and self._process.stdout is not None
        fd = self._process.stdout.fileno()
        buf = bytearray()

        # Phase 1: wait for the PROTO_* header, skipping noise lines. Compute happens here.
        deadline = time.monotonic() + timeout
        header_line = None
        while header_line is None:
            nl = buf.find(b"\n")
            if nl != -1:
                line = buf[:nl].decode("utf-8", "replace").strip()
                del buf[: nl + 1]
                if line.startswith(("PROTO_LENGTH:", "PROTO_FILE:")):
                    header_line = line
                continue  # noise line, or already-buffered bytes still to scan
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                raise TimeoutError(
                    f"{self.toolkit} worker timed out waiting for response to request {request_id} after {timeout}s"
                )
            chunk = os.read(fd, _READ_CHUNK)
            if not chunk:
                raise RuntimeError(
                    f"{self.toolkit} worker closed stdout before responding to request {request_id} "
                    f"({self._crash_context()})"
                )
            buf.extend(chunk)

        if header_line.startswith("PROTO_FILE:"):
            return self._read_file_payload(header_line)

        try:
            json_length = int(header_line.split(":", 1)[1])
        except (ValueError, IndexError) as exc:
            raise RuntimeError(
                f"{self.toolkit} worker sent invalid LENGTH header {header_line!r} "
                f"for request {request_id} ({self._crash_context()})"
            ) from exc

        # Phase 2: drain the body. Compute is done, so a fresh short grace window bounds the
        # transfer without risking killing a finished worker that produced a large result.
        body_deadline = time.monotonic() + _BODY_GRACE_SECONDS
        while len(buf) < json_length:
            remaining = body_deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                break
            chunk = os.read(fd, min(_READ_CHUNK, json_length - len(buf)))
            if not chunk:
                break
            buf.extend(chunk)

        if len(buf) < json_length:
            raise RuntimeError(
                f"{self.toolkit} worker sent incomplete JSON for request {request_id}: "
                f"expected {json_length} bytes, got {len(buf)} ({self._crash_context()})"
            )
        # Body is ASCII (bootstrap serializes with ensure_ascii=True), so PROTO_LENGTH counts bytes.
        return self._decode_body(request_id, bytes(buf[:json_length]))

    def _read_file_payload(self, header_line: str) -> dict[str, Any]:
        """Read and unlink a PROTO_FILE temp payload (shared by both read paths)."""
        file_path = header_line.split(":", 1)[1]
        if not file_path.startswith(tempfile.gettempdir()):
            logger.warning(
                "Worker for %s sent file path outside temp directory: %s",
                self.toolkit,
                file_path,
            )
        try:
            with open(file_path) as f:
                return json.load(f)  # type: ignore[no-any-return]
        finally:
            with contextlib.suppress(OSError):
                os.unlink(file_path)

    def _decode_body(self, request_id: str, response_bytes: bytes | str) -> dict[str, Any]:
        """Parse a PROTO_LENGTH JSON body, raising RuntimeError on malformed payloads."""
        try:
            return json.loads(response_bytes)  # type: ignore[no-any-return]
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{self.toolkit} worker returned invalid JSON for request {request_id} "
                f"({self._crash_context()}); payload: {response_bytes!r}"
            ) from exc

    def _killpg(self, sig: int) -> None:
        """Send *sig* to the worker's process group, ignoring already-dead."""
        with contextlib.suppress(OSError):
            assert self._process is not None
            os.killpg(self._process.pid, sig)

    def stop(self) -> None:
        """Terminate the worker and all of its child processes."""
        if self._process is not None:
            try:
                if self._process.stdin and not self._process.stdin.closed:
                    self._process.stdin.close()
                self._killpg(signal.SIGTERM)  # graceful shutdown
                self._process.wait(timeout=10)
            except Exception:
                self._killpg(signal.SIGKILL)  # force kill
                self._process.wait(timeout=5)  # reap zombie
            finally:
                for pipe in (self._process.stdout, self._process.stderr):
                    if pipe and not pipe.closed:
                        with contextlib.suppress(Exception):
                            pipe.close()
                self._process = None
                logger.debug("Stopped persistent worker for %s", self.toolkit)
