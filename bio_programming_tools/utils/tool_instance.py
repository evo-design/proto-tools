"""
ToolInstance — isolated-venv execution for tool wrappers.

**One-shot by default**: ``dispatch()`` runs an ephemeral subprocess —
no leaked workers, no GPU memory retained after the call.

**Opt-in persistence**: use ``persist()`` (auto-cache all tools),
``persist_tool()`` (tool-specific), or ``get()`` (manual lifecycle)
when you need to keep a worker alive across calls.

Device is always driven by ``config.device`` (a ``BaseConfig`` field)
which flows through ``input_dict["device"]``.  Persistent workers
auto-restart when any ``reload_on_change`` config field changes between
calls (device, model checkpoint, etc.).  Standalone scripts must NOT
check for config changes themselves — the ToolInstance layer handles
restarts.  Any config field that affects model initialization must be
marked ``reload_on_change=True`` in the tool's Config class.

Usage::

    # Default — safe, no leak (device comes from config → input_dict)
    result = ToolInstance.dispatch("esm2", {"device": "cuda", ...})

    # Auto-persist everything (recommended) — all tools auto-cached
    with ToolInstance.persist():
        run_esmfold(inputs, config)       # auto-cached on first call
        run_esm2_score(inputs2, config2)  # also auto-cached
        run_esmfold(inputs3, config)      # reuses cached worker
    # everything cleaned up on exit

    # Tool-specific persistence — named instances / multi-GPU
    with ToolInstance.persist_tool("esmfold"):
        for i in range(500):
            output = run_esmfold(inputs, config)  # reuses worker

    # Manual persistence — power user
    tool = ToolInstance.get("esmfold")
    output = run_esmfold(inputs, config)
    tool.shutdown()  # also evicts from cache

    # Cache-isolated scope (for test fixtures, batch jobs, etc.)
    with ToolInstance.scope():
        tool = ToolInstance.get("esm2")
        result = tool.run(...)
    # worker killed, previous cache restored
"""

from __future__ import annotations

import atexit
import contextvars
import datetime
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar

from .base_config import DEFAULT_TIMEOUT
from .persistent_worker import PersistentWorker, _clean_env

logger = logging.getLogger(__name__)

# ============================================================================
# Singleton registry
# ============================================================================
_lock = threading.Lock()  # protects _instances dict
_instances: dict[str, ToolInstance] = {}
atexit.register(lambda: ToolInstance.clear_all())

_scope_override: contextvars.ContextVar[dict[str, ToolInstance] | None] = (
    contextvars.ContextVar("_scope_override", default=None)
)
_persist_mode: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_persist_mode", default=False
)


def _active_cache() -> dict[str, ToolInstance]:
    """Return the scope-local cache if inside scope(), else the global cache."""
    override = _scope_override.get()
    return override if override is not None else _instances


class ToolInstance:
    """Manage an isolated venv for a tool and (optionally) a persistent worker.

    Callers should use :meth:`dispatch` (one-shot, default) or opt into
    persistence via :meth:`persist_tool` (scoped) or :meth:`get` (manual).
    """

    _tool_dir_cache: ClassVar[dict[str, Path] | None] = None
    _build_failures: ClassVar[dict[str, str]] = {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def get(
        cls,
        tool_name: str,
        *,
        instance_name: str | None = None,
    ) -> ToolInstance:
        """Return (or create) a ToolInstance for *tool_name*.

        Parameters
        ----------
        tool_name : str
            Model-level folder name (e.g. ``"esm2"``, ``"progen2"``).
        instance_name : str | None
            Explicit cache key.  When *None*, the instance is cached
            under *tool_name* so that different operations on the same
            model share one worker.
        """
        key = instance_name if instance_name is not None else tool_name
        with _lock:
            cache = _active_cache()
            if key in cache:
                logger.debug("Returning cached ToolInstance for key=%r", key)
                return cache[key]
        # Create outside lock — __init__ is lightweight now (no venv build)
        new_inst = cls(tool_name)
        with _lock:
            cache = _active_cache()
            # Double-check — another thread may have created it
            if key in cache:
                logger.debug("Returning cached ToolInstance for key=%r", key)
                return cache[key]
            logger.debug("Creating new ToolInstance for %r (key=%r)", tool_name, key)
            cache[key] = new_inst
            if not hasattr(new_inst, "_cache_keys"):
                new_inst._cache_keys = set()
            new_inst._cache_keys.add(key)
            return new_inst

    @classmethod
    def clear_all(cls) -> None:
        """Stop all workers and clear the instance cache."""
        with _lock:
            cache = _active_cache()
            logger.debug("Clearing all cached ToolInstances (%d)", len(cache))
            snapshot = list(cache.values())
            cache.clear()
        for inst in snapshot:
            inst.shutdown()

    @classmethod
    def shutdown_instance(cls, instance_name: str) -> None:
        """Shut down and remove a single cached instance by its cache key.

        This is the escape-hatch for when you don't have the object handy.
        Prefer calling :meth:`shutdown` on the instance directly when
        possible.  No-op if *instance_name* is not in the cache.
        """
        with _lock:
            inst = _active_cache().pop(instance_name, None)
        if inst is not None:
            logger.debug("Closing ToolInstance for key=%r", instance_name)
            inst.shutdown()

    @classmethod
    def dispatch(
        cls,
        tool_name: str,
        input_dict: dict[str, Any],
        *,
        instance: str | ToolInstance | None = None,
        script_path: Path | str | None = None,
        verbose: bool = False,
        reload_on: set[str] | None = None,
    ) -> dict[str, Any]:
        """Run a tool, reusing a cached persistent instance if one exists.

        This is the primary entry point for tool wrappers.  When no
        persistent instance is cached, runs an ephemeral one-shot
        subprocess (no leak).  When a persistent instance exists (via
        :meth:`persistent` or :meth:`get`), reuses it.

        Timeout is read from ``input_dict["timeout"]`` (falling back to
        ``DEFAULT_TIMEOUT``).  The constant is defined once in
        ``base_config.py`` and shared by both ``BaseConfig.timeout`` and
        this fallback.

        Parameters
        ----------
        tool_name : str
            Model-level folder name (e.g. ``"esm2"``).
        input_dict : dict
            JSON-serializable input for the standalone script.
            Should include ``"device"`` when the tool needs a
            specific device (e.g. ``"cuda"``).
        instance : str | ToolInstance | None
            A :class:`ToolInstance` object to use directly, a string cache
            key for persistent instance lookup, or *None* to use
            *tool_name* as the cache key.
        script_path : Path | str | None
            Override the default standalone script.
        verbose : bool
            Whether to log progress.
        reload_on : set[str] | None
            Config field names whose value changes should trigger a
            persistent worker restart.  Defaults to ``{"device"}`` for
            backward compatibility.
        """
        timeout: int | None = input_dict.get("timeout", DEFAULT_TIMEOUT)
        # Path 1: caller passed a ToolInstance object directly
        if isinstance(instance, ToolInstance):
            logger.debug("dispatch(%s): using provided ToolInstance", tool_name)
            return instance.run(
                input_dict,
                script_path=script_path,
                verbose=verbose,
                timeout=timeout,
                reload_on=reload_on,
            )
        # Path 2: look up by string cache key (or tool_name as default key)
        key = instance if instance is not None else tool_name
        with _lock:
            cached = _active_cache().get(key)
        if cached is not None:
            logger.debug(
                "dispatch(%s): reusing cached instance (key=%r)", tool_name, key
            )
            return cached.run(
                input_dict,
                script_path=script_path,
                verbose=verbose,
                timeout=timeout,
                reload_on=reload_on,
            )
        # Path 3: persist mode — auto-create and cache instead of one-shot
        if _persist_mode.get():
            logger.debug(
                "dispatch(%s): persist mode, auto-caching instance", tool_name
            )
            inst = cls.get(tool_name)
            return inst.run(
                input_dict,
                script_path=script_path,
                verbose=verbose,
                timeout=timeout,
                reload_on=reload_on,
            )
        # Path 4: no cached instance — ephemeral one-shot subprocess
        logger.debug("dispatch(%s): no cached instance, running one-shot", tool_name)
        return cls._oneshot(
            tool_name,
            input_dict,
            script_path=script_path,
            verbose=verbose,
            timeout=timeout,
        )

    @classmethod
    def _oneshot(
        cls,
        tool_name: str,
        input_dict: dict[str, Any],
        *,
        script_path: Path | str | None = None,
        verbose: bool = False,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Run a tool in an ephemeral subprocess — no caching, no worker."""
        inst = cls(tool_name)
        effective_script = Path(script_path) if script_path else inst.script_path
        return inst._run_oneshot(
            input_dict,
            script_path=effective_script,
            verbose=verbose,
            timeout=timeout,
        )

    @classmethod
    @contextmanager
    def persist_tool(
        cls,
        tool_name: str,
        *,
        instance_name: str | None = None,
    ):
        """Context manager that caches a persistent worker for its duration.

        Tool wrappers called inside the block will find the cached
        instance via :meth:`dispatch` and reuse it::

            # Single instance — implicit dispatch works
            with ToolInstance.persist_tool("esmfold"):
                for i in range(500):
                    output = run_esmfold(inputs, config)

            # Multiple instances — first caches, second must be passed explicitly
            with ToolInstance.persist_tool("esm2") as inst_a:
                out_a = run_esm2_score(inputs_a, config)  # implicit dispatch
                with ToolInstance.persist_tool("esm2") as inst_b:
                    out_b = run_esm2_score(inputs_b, config, instance=inst_b)

            # Named instance — always cached under the given name
            with ToolInstance.persist_tool("esmfold", instance_name="fold-gpu0"):
                output = run_esmfold(inputs, config)

        The worker is shut down and removed from the cache on exit.
        """
        if instance_name is not None:
            # Named: always cache under the given name
            inst = cls.get(tool_name, instance_name=instance_name)
            try:
                yield inst
            finally:
                cls.shutdown_instance(instance_name)
        else:
            # Anonymous: atomically check-and-claim the tool_name slot
            with _lock:
                cache = _active_cache()
                if tool_name not in cache:
                    inst = cls(tool_name)
                    cache[tool_name] = inst
                    if not hasattr(inst, "_cache_keys"):
                        inst._cache_keys = set()
                    inst._cache_keys.add(tool_name)
                    owns_slot = True
                else:
                    owns_slot = False
            if owns_slot:
                try:
                    yield inst
                finally:
                    cls.shutdown_instance(tool_name)
            else:
                # Slot taken: don't cache (user must pass instance=)
                logger.warning(
                    "A persistent instance for %r with the default "
                    "instance_name is already cached. This new instance "
                    "will NOT be automatically used by tool calls and "
                    "must be explicitly specified — pass instance=<this "
                    "instance> to run_*() calls, or use instance_name= "
                    "to give it a unique cache key.",
                    tool_name,
                )
                inst = cls(tool_name)
                try:
                    yield inst
                finally:
                    inst.shutdown()

    @classmethod
    @contextmanager
    def scope(cls):
        """Context manager that provides an isolated ToolInstance cache.

        Uses a ``contextvars.ContextVar`` so only the current
        thread/async-task sees the scoped cache.  Other threads
        continue using the global ``_instances`` unaffected::

            with ToolInstance.scope():
                tool = ToolInstance.get("esm2")
                result = tool.run(...)
            # worker killed, previous cache restored
        """
        scoped_cache: dict[str, ToolInstance] = {}
        token = _scope_override.set(scoped_cache)
        try:
            yield
        finally:
            _scope_override.reset(token)
            for inst in scoped_cache.values():
                inst.shutdown()

    @classmethod
    @contextmanager
    def persist(cls):
        """Context manager that auto-caches tools on first dispatch.

        Any tool called inside the block via :meth:`dispatch` is
        automatically cached on first use — subsequent calls to the
        same tool reuse the warm worker.  On exit, all auto-created
        instances are shut down and GPU memory is freed.

        Uses an isolated :meth:`scope` internally, so auto-created
        instances never pollute the global cache::

            with ToolInstance.persist():
                run_esmfold(inputs, config)       # auto-cached
                run_esm2_score(inputs2, config2)  # also auto-cached
                run_esmfold(inputs3, config)      # reuses cached
            # everything cleaned up on exit

        Nestable — each ``persist()`` block gets its own scope.
        Thread-safe — persist mode is per-thread (uses contextvars).
        """
        token = _persist_mode.set(True)
        try:
            with cls.scope():
                yield
        finally:
            _persist_mode.reset(token)

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def __init__(self, tool_name: str) -> None:
        self.tool_name = self._validate_tool_name(tool_name)
        self.device = "cpu"

        venv_root = self._get_venvs_root()
        venv_root.mkdir(parents=True, exist_ok=True)

        self.venv_path = venv_root / f"{tool_name}_env"
        self.setup_script = self._find_setup_script(tool_name)
        self.script_path = self._find_script(tool_name)

        self._venv_ready = False
        self._cache_keys: set[str] = set()
        self._instance_lock = threading.Lock()  # protects _worker lifecycle
        self._worker: PersistentWorker | None = None
        # Tracks previous reload-field values for restart detection
        self._reload_params: dict[str, Any] = {}

    def _ensure_venv(self) -> None:
        """Build the tool's venv if it doesn't exist or is broken.

        Called lazily on first actual execution (not during ``__init__``),
        so that the double-check-locking loser in ``get()`` discards only
        a lightweight object — not one that already built a venv.

        Fails fast if this tool already failed to build in this process.
        On a cross-session failure (FAILED STATUS.txt from a previous run),
        logs a warning and retries the build.
        """
        if getattr(self, "_venv_ready", False):
            return
        if self.tool_name in self._build_failures:
            tail = self._build_failures[self.tool_name]
            hint = f"\n{tail}" if tail else ""
            raise RuntimeError(
                f"'{self.tool_name}' may not be compatible with your "
                f"system. Check logs for details.{hint}"
            )
        if not self.venv_path.exists() or not self._is_venv_ok():
            # Check for a stale failure from a previous session
            status_file = self.venv_path / "STATUS.txt"
            if status_file.exists():
                status = status_file.read_text()
                if status.startswith("FAILED"):
                    current_hash = self._setup_hash()
                    if f"Setup hash: {current_hash}" in status:
                        summary = self._failure_summary()
                        hint = f": {summary}" if summary else ""
                        logger.warning(
                            "'%s' previously failed to build with the "
                            "same setup.sh (hash=%s)%s. Retrying — if "
                            "this keeps failing, the tool may not be "
                            "compatible with your system, or you may "
                            "need to accept a license agreement (e.g. "
                            "Hugging Face). Check logs for details.",
                            self.tool_name,
                            current_hash,
                            hint,
                        )
                    else:
                        logger.info(
                            "setup.sh changed for %s — retrying venv build",
                            self.tool_name,
                        )
            try:
                self._create_venv()
            except Exception as exc:
                # _create_venv populates _build_failures with the stderr
                # tail on setup.sh failure; for other exceptions, store
                # the message as fallback.
                if self.tool_name not in self._build_failures:
                    self._build_failures[self.tool_name] = str(exc)
                raise
        self._venv_ready = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        input_dict: dict[str, Any],
        *,
        script_path: Path | str | None = None,
        verbose: bool = False,
        timeout: int | None = None,
        reload_on: set[str] | None = None,
    ) -> dict[str, Any]:
        """Execute *input_dict* in the tool's venv and return the result.

        Parameters
        ----------
        input_dict : dict
            JSON-serializable input for the standalone script.
        script_path : Path | str | None
            Override the default standalone script. Useful for tools with
            multiple scripts (e.g. colabfold local vs remote).
        verbose : bool
            Whether to log progress.
        timeout : int | None
            Maximum seconds to wait.
        reload_on : set[str] | None
            Config field names whose value changes should trigger a
            persistent worker restart.  Defaults to ``{"device"}``.
        """
        effective_script = Path(script_path) if script_path else self.script_path
        with self._instance_lock:
            return self._run_persistent(
                input_dict,
                script_path=effective_script,
                verbose=verbose,
                timeout=timeout,
                reload_on=reload_on,
            )

    def ensure_ready(self) -> None:
        """Ensure the tool's venv is built and ready.

        Call this when you need direct access to venv binaries or files
        without running the tool through :meth:`dispatch` or :meth:`run`.
        See ``tests/dummy_data/create_mini_mmseqs_db.py`` for an example
        (used to access the mmseqs binary directly for database setup).
        """
        self._ensure_venv()

    def shutdown(self) -> None:
        """Stop the persistent worker (if any) and remove from cache."""
        # Defensive: atexit may call shutdown() on a partially-initialized
        # instance if __init__ raised after the atexit handler was registered.
        instance_lock = getattr(self, "_instance_lock", None)
        if instance_lock is not None:
            with instance_lock:
                worker = getattr(self, "_worker", None)
                self._worker = None
        else:
            worker = getattr(self, "_worker", None)
        if worker is not None:
            name = getattr(self, "tool_name", "?")
            logger.debug("Shutting down persistent worker for %s", name)
            worker.stop()
        # Evict from cache by stored keys
        with _lock:
            cache = _active_cache()
            for k in getattr(self, "_cache_keys", set()):
                cache.pop(k, None)

    # ------------------------------------------------------------------
    # Persistent execution
    # ------------------------------------------------------------------
    def _run_persistent(
        self,
        input_dict: dict[str, Any],
        *,
        script_path: Path | None = None,
        verbose: bool = False,
        timeout: int | None = None,
        reload_on: set[str] | None = None,
    ) -> dict[str, Any]:
        """Execute via the persistent worker, restarting if config changed.

        Compares the current reload-field values against the previous call.
        If any tracked field changed (or the script path changed), the
        existing worker is stopped and a new one is created.  On the very
        first call the worker is lazily created.

        Standalone scripts must NOT check for config changes themselves —
        the ToolInstance layer handles restarts via ``reload_on_change``
        fields in the tool's Config class.
        """
        self._ensure_venv()
        sp = script_path or self.script_path
        # Default to {"device"} for backward compat with tools that don't
        # pass reload_on explicitly.
        reload_keys = reload_on or {"device"}
        reload_params = {k: input_dict.get(k) for k in reload_keys}

        if self._worker is not None:
            script_changed = self._worker.script_path != sp
            params_changed = reload_params != self._reload_params
            if script_changed or params_changed:
                if params_changed:
                    changed = {
                        k
                        for k in reload_keys
                        if self._reload_params.get(k) != reload_params.get(k)
                    }
                    logger.info(
                        "Config changed (%s) for %s, restarting worker",
                        ", ".join(
                            f"{k}: {self._reload_params.get(k)!r} → {reload_params[k]!r}"
                            for k in sorted(changed)
                        ),
                        self.tool_name,
                    )
                self._worker.stop()
                self._worker = None

        self._reload_params = reload_params
        # Cache device for PersistentWorker constructor
        self.device = input_dict.get("device", "cpu")

        if self._worker is None:
            self._worker = PersistentWorker(
                tool_name=self.tool_name,
                venv_path=self.venv_path,
                script_path=sp,
                device=self.device,
            )
        return self._worker.send(input_dict, timeout=timeout)

    # ------------------------------------------------------------------
    # One-shot execution
    # ------------------------------------------------------------------
    def _run_oneshot(
        self,
        input_dict: dict[str, Any],
        *,
        script_path: Path | None = None,
        verbose: bool = False,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Run in an ephemeral subprocess (no persistent worker).

        Writes *input_dict* to a temp JSON file, invokes the venv's Python
        on *script_path* with the input/output paths as arguments, reads the
        output JSON, and converts ``subprocess.TimeoutExpired`` to
        ``TimeoutError``.
        """
        self._ensure_venv()
        sp = script_path or self.script_path
        device = input_dict.get("device", self.device)
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.json"
            output_path = Path(tmp) / "output.json"

            with open(input_path, "w") as f:
                json.dump(input_dict, f)

            # Sets CUDA_VISIBLE_DEVICES based on device string
            env = _clean_env(device, tool_venv_path=self.venv_path)
            env["TOOL_VENV_PATH"] = str(self.venv_path)
            python_exe = str(self.venv_path / "bin" / "python")

            if verbose:
                logger.info(
                    "Running %s (one-shot) with device=%s",
                    sp.name,
                    device,
                )

            try:
                subprocess.run(
                    [python_exe, str(sp), str(input_path), str(output_path)],
                    env=env,
                    text=True,
                    check=True,
                    timeout=timeout,
                    stdout=None if verbose else subprocess.PIPE,
                    stderr=None if verbose else subprocess.PIPE,
                )
            except subprocess.TimeoutExpired:
                raise TimeoutError(
                    f"Tool {self.tool_name} timed out after {timeout}s"
                ) from None

            with open(output_path) as f:
                return json.load(f)

    # ------------------------------------------------------------------
    # Venv management
    # ------------------------------------------------------------------
    @staticmethod
    def _get_venvs_root() -> Path:
        """Determine the ``.venvs`` root directory.

        For editable installs (``pip install -e .``), finds the project root
        by walking up from this file looking for ``pyproject.toml``, then
        uses ``project_root/.venvs/``.

        For non-editable installs (``pip install .``), the package is copied
        into site-packages and there's no project root.  Falls back to a
        user-level cache directory:
        ``$XDG_CACHE_HOME/bio_programming_tools/.venvs/`` or
        ``~/.cache/bio_programming_tools/.venvs/``.
        """
        for parent in Path(__file__).resolve().parents:
            if (parent / "pyproject.toml").exists():
                return parent / ".venvs"
        cache_home = Path(
            os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
        )
        return cache_home / "bio_programming_tools" / ".venvs"

    @classmethod
    def _get_tool_dirs(cls) -> dict[str, Path]:
        """Return a ``{tool_name: dir_path}`` mapping for all standalone tools.

        The mapping is computed once (single ``rglob``) and cached for the
        lifetime of the process.
        """
        if cls._tool_dir_cache is not None:
            return cls._tool_dir_cache
        tools_dir = Path(__file__).parent.parent / "tools"
        cls._tool_dir_cache = {
            item.name: item
            for item in tools_dir.rglob("*")
            if item.is_dir() and (item / "standalone").exists()
        }
        return cls._tool_dir_cache

    @classmethod
    def _validate_tool_name(cls, tool_name: str) -> str:
        """Raise ValueError if *tool_name* has no standalone/ directory."""
        tool_dirs = cls._get_tool_dirs()
        if tool_name in tool_dirs:
            return tool_name
        raise ValueError(
            f"Invalid tool name: {tool_name!r}. "
            f"Available tools with standalone dirs: {sorted(tool_dirs)}"
        )

    @classmethod
    def _find_setup_script(cls, tool_name: str) -> Path:
        """Return the path to standalone/setup.sh for *tool_name*."""
        tool_dir = cls._get_tool_dirs().get(tool_name)
        if tool_dir is not None:
            setup = tool_dir / "standalone" / "setup.sh"
            if setup.is_file():
                return setup
        raise ValueError(f"No setup.sh found for tool {tool_name!r}")

    @classmethod
    def _find_script(cls, tool_name: str) -> Path:
        """Find the main standalone script (inference.py or run.py)."""
        tool_dir = cls._get_tool_dirs().get(tool_name)
        if tool_dir is not None:
            standalone_dir = tool_dir / "standalone"
            # Prefer inference.py, then run.py
            for name in ("inference.py", "run.py"):
                candidate = standalone_dir / name
                if candidate.is_file():
                    return candidate
            # Fall back to first .py file that isn't __init__.py or config
            for py_file in sorted(standalone_dir.glob("*.py")):
                if py_file.name not in ("__init__.py", "binary_config.py"):
                    return py_file
        raise ValueError(f"No standalone script found for tool {tool_name!r}")

    def _setup_hash(self) -> str:
        """Return a short SHA-256 hash of setup.sh for change detection."""
        return hashlib.sha256(self.setup_script.read_bytes()).hexdigest()[:16]

    @staticmethod
    def _stderr_tail(stderr: str | None, max_lines: int = 10) -> str:
        """Return the last few non-empty lines of stderr for error messages."""
        if not stderr:
            return ""
        lines = [l for l in stderr.strip().splitlines() if l.strip()]
        return "\n".join(lines[-max_lines:])

    def _failure_summary(self) -> str:
        """Extract a one-line error summary from a FAILED STATUS.txt."""
        status_file = self.venv_path / "STATUS.txt"
        if not status_file.exists():
            return ""
        skip = (
            "FAILED", "Return code:", "Command:",
            "Setup hash:", "Timestamp:", "STDERR:",
        )
        for line in status_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith(skip):
                return line[:200]
        return ""

    def _is_venv_ok(self) -> bool:
        """Check STATUS.txt and verify the Python executable works."""
        status_file = self.venv_path / "STATUS.txt"
        if not status_file.exists():
            return False
        try:
            status = status_file.read_text().strip()
            if not status.startswith("SUCCESS"):
                return False
            python_exe = self.venv_path / "bin" / "python"
            if not python_exe.exists():
                return False
            result = subprocess.run(
                [str(python_exe), "--version"],
                capture_output=True,
                timeout=30,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _create_venv(self) -> None:
        """Create (or rebuild) the tool's isolated venv.

        Removes a broken existing venv, creates a fresh one via
        ``python -m venv``, runs ``standalone/setup.sh`` inside it, and
        writes STATUS.txt to record success or failure.
        """
        status_file = self.venv_path / "STATUS.txt"

        # Remove broken venv
        if self.venv_path.exists() and not self._is_venv_ok():
            logger.info("Removing broken venv at %s", self.venv_path)
            shutil.rmtree(self.venv_path)

        # Create fresh venv
        logger.info("Setting up venv for %s ...", self.tool_name)
        subprocess.run(
            [sys.executable, "-m", "venv", "--copies", str(self.venv_path)],
            check=True,
        )

        # Run setup.sh
        subprocess.run(["chmod", "+x", str(self.setup_script)], check=True)
        env = _clean_env(self.device, tool_venv_path=self.venv_path)
        env["VENV_PATH"] = str(self.venv_path.absolute())
        env["PYTHON_EXE"] = str(self.venv_path.absolute() / "bin" / "python")
        env["PIP_EXE"] = str(self.venv_path.absolute() / "bin" / "pip")

        activate = self.venv_path.absolute() / "bin" / "activate"
        proc = subprocess.Popen(
            ["bash", "-c", f"source {activate} && {self.setup_script}"],
            cwd=self.setup_script.parent,
            env=env,
            stdout=None,
            stderr=subprocess.PIPE,
            text=True,
        )
        _, stderr_output = proc.communicate()

        if proc.returncode == 0:
            status_file.write_text("SUCCESS")
            logger.debug("Venv setup completed for %s", self.tool_name)
        else:
            logger.error(
                "Venv setup failed for %s (exit %d)", self.tool_name, proc.returncode
            )
            if stderr_output:
                logger.error("stderr: %s", stderr_output)
            status_file.write_text(
                f"FAILED\n\n"
                f"Return code: {proc.returncode}\n"
                f"Command: {self.setup_script}\n"
                f"Setup hash: {self._setup_hash()}\n"
                f"Timestamp: {datetime.datetime.now()}\n\n"
                f"STDERR:\n{stderr_output or ''}\n"
            )
            tail = self._stderr_tail(stderr_output)
            self._build_failures[self.tool_name] = tail
            hint = f"\n{tail}" if tail else ""
            raise RuntimeError(
                f"'{self.tool_name}' may not be compatible with your "
                f"system. setup.sh failed (exit {proc.returncode}).{hint}"
            )
