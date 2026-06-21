"""proto_tools/utils/tool_instance.py.

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

    # Default - safe, no leak (device comes from config → input_dict)
    result = ToolInstance.dispatch("esm2", {"device": "cuda", ...})

    # Auto-persist everything (recommended) - all tools auto-cached
    with ToolInstance.persist():
        run_esmfold(inputs, config)       # auto-cached on first call
        run_esm2_score(inputs2, config2)  # also auto-cached
        run_esmfold(inputs3, config)      # reuses cached worker
    # everything cleaned up on exit

    # Tool-specific persistence - named instances / multi-GPU
    with ToolInstance.persist_tool("esmfold"):
        for i in range(500):
            output = run_esmfold(inputs, config)  # reuses worker

    # Manual persistence - power user
    tool = ToolInstance.get("esmfold")
    output = run_esmfold(inputs, config)
    tool.shutdown()  # also evicts from cache

    # Cache-isolated scope (for test fixtures, batch jobs, etc.)
    with ToolInstance.scope():
        tool = ToolInstance.get("esm2")
        result = tool.run(...)
    # worker killed, previous cache restored
"""

import atexit
import collections
import contextvars
import datetime
import hashlib
import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, ClassVar

from proto_tools.utils._worker_bootstrap import _copy_standalone_helpers as copy_standalone_helpers
from proto_tools.utils.base_config import DEFAULT_TIMEOUT, BaseConfig
from proto_tools.utils.device import parse_device_string
from proto_tools.utils.device_manager import DeviceManager
from proto_tools.utils.logging_config import verbose_level_from_env
from proto_tools.utils.persistent_worker import (
    PersistentWorker,
    _apply_verbose_to_console_handlers,
    _build_subprocess_env,
    _drain_subprocess_stderr,
    _parse_env_vars_file,
    _stderr_buffer_lines,
)
from proto_tools.utils.progress import get_current_tool_function, has_active_progress_bar, progress_bar, set_substatus
from proto_tools.utils.tool_io import MissingAssetError

logger = logging.getLogger(__name__)

# Seconds to wait for a worker to complete a to_device move.
DEVICE_MOVE_TIMEOUT = 200

# conda corrupts relocated binaries past its 255-byte prefix; cap the envs root.
MAX_TOOL_ENVS_ROOT_LEN = 200
# The full env prefix (root + the per-env leaf) must also stay within the same
# 255-byte limit. The root cap above leaves headroom for typical leaves; this
# guards the actual prefix length, which longer env names (e.g. overrides) grow.
MAX_ENV_PREFIX_LEN = 255

# ============================================================================
# Singleton registry
# ============================================================================
_lock = threading.Lock()  # protects _instances dict
_instances: dict[str, "ToolInstance"] = {}

_scope_override: contextvars.ContextVar[dict[str, "ToolInstance"] | None] = contextvars.ContextVar(
    "_scope_override", default=None
)
_persist_mode: contextvars.ContextVar[bool] = contextvars.ContextVar("_persist_mode", default=False)

# Set by the ``@tool()`` wrapper around each dispatch so ``ToolInstance`` can
# read per-tool flags (e.g. ``gpu_only``) without plumbing the ``ToolSpec``
# through every call site. Contents: {"key": str, "gpu_only": bool,
# "pin_visible_devices": bool}.
_current_tool_invocation: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "_current_tool_invocation", default=None
)


def _device_change_needed(current: str, requested: str) -> bool:
    """Whether a worker on ``current`` must move to satisfy ``requested`` (auto/equivalent device forms = no move)."""
    if not requested:
        return False
    try:
        cur, req = parse_device_string(current), parse_device_string(requested)
    except ValueError:
        return current != requested  # unparseable form: compare verbatim
    if req.devices is None:  # auto request (cuda / cudaxN) satisfied by any same-kind, same-count allocation
        return not (cur.kind == req.kind and cur.count == req.count)
    return (cur.kind, cur.devices) != (req.kind, req.devices)


def _local_visible_device(device: str) -> str:
    """Local device a CVD-pinned worker sees: its N assigned GPUs become cuda:0..N-1; cpu unchanged."""
    if not device or device == "cpu":
        return "cpu"
    try:
        count = parse_device_string(device).count
    except ValueError:
        count = 1
    return ",".join(f"cuda:{i}" for i in range(count))


# Overlay of toolkit -> ToolInstance used by _auto_persist_scope. Checked by
# dispatch() Path 2 BEFORE the shared cache. ContextVar-backed so two
# concurrent callers each with their own instance don't cross-contaminate.
_auto_persist_overlay: contextvars.ContextVar[dict[str, "ToolInstance"] | None] = contextvars.ContextVar(
    "_auto_persist_overlay", default=None
)


def _active_cache() -> dict[str, "ToolInstance"]:
    """Return the scope-local cache if inside scope(), else the global cache."""
    override = _scope_override.get()
    return override if override is not None else _instances


def _lookup_instance(key: str) -> "ToolInstance | None":
    """Return the cached instance for *key*, checking the auto-persist overlay first.

    Read-only — use :func:`_active_cache` for mutations. Overlays are
    ContextVar-backed so they take precedence and stay per-task/thread,
    preventing concurrent callers with distinct instances from colliding.
    """
    overlay = _auto_persist_overlay.get()
    if overlay is not None and key in overlay:
        return overlay[key]
    with _lock:
        return _active_cache().get(key)


def _resolve_instance_or_raise(
    instance: "str | ToolInstance | None",
    toolkit: str,
) -> "ToolInstance | None":
    """Normalize the ``instance`` kwarg shared by ``dispatch`` and the @tool wrapper.

    Semantics:
      * ``None`` -> returns ``None`` (caller's default-dispatch path).
      * ``ToolInstance`` -> returns it unchanged (caller owns the handle).
      * ``str`` -> looked up via :func:`_lookup_instance`. If found, returns it.
        If not found AND ``_persist_mode`` is active, returns ``None`` so
        dispatch can create-and-register under the string key. If not found
        AND ``_persist_mode`` is inactive, raises :class:`ValueError` — a
        string is a reference, not a creation request.

    Args:
        instance (str | ToolInstance | None): The caller-supplied ``instance``
            value. Strings are treated as cache keys that must already exist
            outside of a ``persist()`` / ``persist_tool()`` context.
        toolkit (str): Toolkit name, used in the error message to suggest
            a remediation (``ToolInstance.get`` / ``persist_tool``).

    Returns:
        ToolInstance | None: The resolved instance, or ``None`` when the
            caller passed ``None`` or an unresolved string inside persist mode.

    Raises:
        ValueError: When ``instance`` is a string not present in
            ``_instances`` and ``_persist_mode`` is inactive.
    """
    if instance is None or isinstance(instance, ToolInstance):
        return instance
    cached = _lookup_instance(instance)
    if cached is not None:
        return cached
    if _persist_mode.get():
        # Explicit opt-in: persist() / persist_tool() authorize auto-creation
        # under the supplied key. Let dispatch handle the actual `.get()`.
        return None
    raise ValueError(
        f"No cached ToolInstance for instance_name={instance!r}. "
        f"Create one with ToolInstance.get({toolkit!r}, instance_name={instance!r}) "
        f"or wrap the call in ToolInstance.persist_tool({toolkit!r}, "
        f"instance_name={instance!r}) before dispatching."
    )


def _run_setup_script(
    setup_script: Path,
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    toolkit: str,
) -> tuple[int, str]:
    """Run ``setup.sh``, capturing stdout+stderr to ``log_path``.

    Honors two environment variables on the caller side:

    - ``PROTO_ENV_VERBOSE=1`` mirrors each line of the subprocess output to
      this process's stderr as it arrives, so users see install progress
      live instead of waiting for the subprocess to finish.
    - ``PROTO_ENV_LOG_DIR=<path>`` copies the completed log to
      ``<PROTO_ENV_LOG_DIR>/<toolkit>_setup.log`` after the subprocess
      exits. Useful for inspecting setup output when the env directory
      itself is ephemeral — e.g. pointing ``PROTO_ENV_LOG_DIR`` at a
      persistent scratch path so a failed build's log is retrievable
      even after the env is rolled back.

    Args:
        setup_script (Path): Absolute path to ``setup.sh``.
        cwd (Path): Working directory for the subprocess.
        env (dict[str, str]): Environment variables for the subprocess.
        log_path (Path): Where to write the setup log. The parent
            directory must already exist.
        toolkit (str): Used to name the mirrored log file when
            ``PROTO_ENV_LOG_DIR`` is set.

    Returns:
        tuple[int, str]: Process return code and the full combined
            stdout+stderr as a string.
    """
    verbose = os.environ.get("PROTO_ENV_VERBOSE") == "1"
    mirror_dir = os.environ.get("PROTO_ENV_LOG_DIR")

    proc = subprocess.Popen(
        ["bash", str(setup_script)],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc.stdout is not None  # stdout=PIPE guarantees this

    with open(log_path, "wb") as log_file:
        for line in iter(proc.stdout.readline, b""):
            log_file.write(line)
            log_file.flush()
            if verbose:
                sys.stderr.buffer.write(line)
                sys.stderr.buffer.flush()
    proc.wait()

    if mirror_dir:
        mirror_path = Path(mirror_dir)
        with suppress(OSError):
            mirror_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(log_path, mirror_path / f"{toolkit}_setup.log")

    combined_output = log_path.read_text(errors="replace") if log_path.exists() else ""
    return proc.returncode, combined_output


class ToolInstance:
    """Manage an isolated venv for a tool and (optionally) a persistent worker.

    Callers should use :meth:`dispatch` (one-shot, default) or opt into
    persistence via :meth:`persist_tool` (scoped) or :meth:`get` (manual).
    """

    _tool_dir_cache: ClassVar[dict[str, Path] | None] = None
    _build_failures: ClassVar[dict[str, str]] = {}
    _env_build_locks: ClassVar[dict[str, threading.Lock]] = {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def get(
        cls,
        toolkit: str,
        *,
        instance_name: str | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> "ToolInstance":
        """Return (or create) a ToolInstance for *toolkit*.

        Passing ``instance_name="K"`` registers the instance in the shared
        cache under ``"K"`` so later ``dispatch(..., instance="K")`` calls
        can reference it. Strings are references: outside of a :meth:`persist`
        / :meth:`persist_tool` context, an unknown name raises ``ValueError``.

        Args:
            toolkit (str): Worker-group folder name (e.g. ``"esm2"``, ``"progen2"``).
                Accepts either a toolkit or a registered tool_key; tool_keys are
                normalized to their toolkit.
            instance_name (str | None): Explicit cache key. When None, the instance
                is cached under *toolkit* so that different operations on the same
                model share one worker.
            env_overrides (dict[str, str] | None): Per-instance env vars passed
                to the worker subprocess on spawn. Only applied when this call
                actually creates the instance — if the cache already holds an
                instance for ``key``, its existing overrides are kept. Used by
                ToolPool's CPU fan-out to pin per-worker thread budgets.
        """
        toolkit = cls._normalize_toolkit(toolkit)
        key = instance_name if instance_name is not None else toolkit
        with _lock:
            cache = _active_cache()
            if key in cache:
                logger.debug("Returning cached ToolInstance for key=%r", key)
                return cache[key]
        # Create outside lock - __init__ is lightweight now (no venv build)
        new_inst = cls(toolkit, env_overrides=env_overrides)
        with _lock:
            cache = _active_cache()
            # Double-check - another thread may have created it
            if key in cache:
                logger.debug("Returning cached ToolInstance for key=%r", key)
                return cache[key]
            logger.debug("Creating new ToolInstance for %r (key=%r)", toolkit, key)
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

        Args:
            instance_name (str): Cache key of the instance to shut down.
        """
        with _lock:
            inst = _active_cache().pop(instance_name, None)
        if inst is not None:
            logger.debug("Closing ToolInstance for key=%r", instance_name)
            inst.shutdown()

    @classmethod
    def dispatch(
        cls,
        toolkit: str,
        input_dict: dict[str, Any],
        *,
        instance: "str | ToolInstance | None" = None,
        script_path: Path | str | None = None,
        config: BaseConfig | None = None,
    ) -> dict[str, Any]:
        """Run a tool, reusing a cached persistent instance if one exists.

        This is the primary entry point for tool wrappers.  When no
        persistent instance is cached, runs an ephemeral one-shot
        subprocess (no leak).  When a persistent instance exists (via
        :meth:`persist`, :meth:`persist_tool`, or :meth:`get`), reuses it.

        Args:
            toolkit (str): Worker-group folder name (e.g. ``"esm2"``).
                Accepts either a toolkit or a registered tool_key; tool_keys
                are normalized to their toolkit.
            input_dict (dict[str, Any]): JSON-serializable input for the standalone script.
            instance (str | ToolInstance | None): How to route the call.
                A ``ToolInstance`` is used directly. A string is a **reference**
                to an already-registered cache entry — it must exist in
                ``_instances`` (via :meth:`get` or :meth:`persist_tool`), else
                ``ValueError`` is raised. Inside a :meth:`persist` /
                :meth:`persist_tool` context strings may auto-create under the
                named key. ``None`` falls back to the toolkit-keyed default
                (cached instance, persist-mode auto-create, or one-shot).
            script_path (Path | str | None): Override the default standalone script.
            config (BaseConfig | None): Tool configuration object. When provided,
                verbose, timeout, and reload_on are derived automatically.

        Raises:
            ValueError: If ``instance`` is a string not present in
                ``_instances`` and no persist context is active.
        """
        toolkit = cls._normalize_toolkit(toolkit)
        # Read effective_timeout() so tools can derive the cap from other fields.
        if config is not None:
            verbose = config.verbose
            timeout = config.effective_timeout()
            reload_on = type(config).reload_fields()
        else:
            verbose = 0
            timeout = DEFAULT_TIMEOUT
            reload_on = None
        # Path 1: caller passed a ToolInstance object directly - use as-is.
        if isinstance(instance, ToolInstance):
            logger.debug("dispatch(%s): using provided ToolInstance", toolkit)
            return instance.run(
                input_dict,
                script_path=script_path,
                verbose=verbose,
                timeout=timeout,
                reload_on=reload_on,
            )
        # Path 2: string reference - must resolve, or be inside persist mode.
        # Raises ValueError otherwise (strings are references, not creation requests).
        if isinstance(instance, str):
            resolved = _resolve_instance_or_raise(instance, toolkit)
            if resolved is not None:
                logger.debug("dispatch(%s): reusing cached instance (key=%r)", toolkit, instance)
                return resolved.run(
                    input_dict,
                    script_path=script_path,
                    verbose=verbose,
                    timeout=timeout,
                    reload_on=reload_on,
                )
            # Inside persist mode with an unresolved string: create + register under it.
            logger.debug("dispatch(%s): persist mode, auto-caching instance (key=%r)", toolkit, instance)
            return cls.get(toolkit, instance_name=instance).run(
                input_dict,
                script_path=script_path,
                verbose=verbose,
                timeout=timeout,
                reload_on=reload_on,
            )
        # Path 3: instance is None - toolkit-keyed default.
        cached = _lookup_instance(toolkit)
        if cached is not None:
            logger.debug("dispatch(%s): reusing cached instance (key=%r)", toolkit, toolkit)
            return cached.run(
                input_dict,
                script_path=script_path,
                verbose=verbose,
                timeout=timeout,
                reload_on=reload_on,
            )
        if _persist_mode.get():
            logger.debug("dispatch(%s): persist mode, auto-caching instance (key=%r)", toolkit, toolkit)
            return cls.get(toolkit).run(
                input_dict,
                script_path=script_path,
                verbose=verbose,
                timeout=timeout,
                reload_on=reload_on,
            )
        # Path 4: no cached instance - ephemeral one-shot subprocess.
        logger.debug("dispatch(%s): no cached instance, running one-shot", toolkit)
        return cls._oneshot(
            toolkit,
            input_dict,
            script_path=script_path,
            verbose=verbose,
            timeout=timeout,
        )

    @classmethod
    def _oneshot(
        cls,
        toolkit: str,
        input_dict: dict[str, Any],
        *,
        script_path: Path | str | None = None,
        verbose: int = 0,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Run a tool in an ephemeral subprocess — no caching, no worker.

        For GPU devices, acquires a transient lease from DeviceManager
        to prevent concurrent one-shot calls from stomping the same GPU.

        Args:
            toolkit (str): Model-level folder name (e.g. ``"esm2"``).
            input_dict (dict[str, Any]): JSON-serializable input for the standalone script.
            script_path (Path | str | None): Override the default standalone script.
            verbose (int): Verbosity level on the 0/1/2/3 scale; see :class:`BaseConfig`.
            timeout (int | None): Maximum execution time in seconds.
        """
        inst = cls(toolkit)
        effective_script = Path(script_path) if script_path else inst.script_path
        device = input_dict.get("device")

        if device and device.startswith("cuda"):
            dm = DeviceManager.get_instance()
            lease_timeout = float((timeout or DEFAULT_TIMEOUT) + 60)
            with dm.lease(toolkit, device=device, timeout=lease_timeout) as allocated:
                leased_input = {**input_dict, "device": allocated}
                return inst._run_oneshot(
                    leased_input,
                    script_path=effective_script,
                    verbose=verbose,
                    timeout=timeout,
                )
        else:
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
        toolkit: str,
        *,
        instance_name: str | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> Generator["ToolInstance", None, None]:
        """Context manager that caches a persistent worker for its duration.

        Tool wrappers called inside the block will find the cached
        instance via :meth:`dispatch` and reuse it::

            with ToolInstance.persist_tool("esmfold"):
                for i in range(500):
                    output = run_esmfold(inputs, config)

        The worker is shut down and removed from the cache on exit.

        Args:
            toolkit (str): Worker-group folder name (e.g. ``"esmfold"``).
                Accepts either a toolkit or a registered tool_key; tool_keys are
                normalized to their toolkit.
            instance_name (str | None): Explicit cache key. When None, uses toolkit.
            env_overrides (dict[str, str] | None): Per-instance env vars passed
                to the worker subprocess on spawn. Only applied when this call
                creates the instance.
        """
        toolkit = cls._normalize_toolkit(toolkit)
        if instance_name is not None:
            # Named: always cache under the given name
            inst = cls.get(toolkit, instance_name=instance_name, env_overrides=env_overrides)
            try:
                yield inst
            finally:
                cls.shutdown_instance(instance_name)
        else:
            # Anonymous: atomically check-and-claim the toolkit slot
            with _lock:
                cache = _active_cache()
                if toolkit not in cache:
                    inst = cls(toolkit, env_overrides=env_overrides)
                    cache[toolkit] = inst
                    if not hasattr(inst, "_cache_keys"):
                        inst._cache_keys = set()
                    inst._cache_keys.add(toolkit)
                    owns_slot = True
                else:
                    owns_slot = False
            if owns_slot:
                try:
                    yield inst
                finally:
                    cls.shutdown_instance(toolkit)
            else:
                # Slot taken: don't cache (user must pass instance=)
                logger.warning(
                    "A persistent instance for %r with the default "
                    "instance_name is already cached. This new instance "
                    "will NOT be automatically used by tool calls and "
                    "must be explicitly specified — pass instance=<this "
                    "instance> to run_*() calls, or use instance_name= "
                    "to give it a unique cache key.",
                    toolkit,
                )
                inst = cls(toolkit, env_overrides=env_overrides)
                try:
                    yield inst
                finally:
                    inst.shutdown()

    @classmethod
    @contextmanager
    def scope(cls) -> Generator[None, None, None]:
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
    def persist(cls) -> Generator[None, None, None]:
        """Context manager that auto-caches tools on first dispatch.

        Any tool called inside the block via :meth:`dispatch` is
        automatically cached on first use — subsequent calls to the
        same tool reuse the warm worker.  On exit, all auto-created
        instances are shut down and GPU memory is freed.

        Uses an isolated :meth:`scope` internally, so auto-created
        instances never pollute the global cache::

            with ToolInstance.persist():
                run_esmfold(inputs, config)  # auto-cached
                run_esm2_score(inputs2, config2)  # also auto-cached
                run_esmfold(inputs3, config)  # reuses cached
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

    @classmethod
    @contextmanager
    def _auto_persist_scope(
        cls,
        toolkit: str,
        *,
        instance: "ToolInstance | None" = None,
    ) -> Generator[None, None, None]:
        """Internal: seed a thread-local overlay so preprocess + dispatch share one worker.

        Called by the ``@tool`` wrapper when a tool has a custom ``preprocess``
        or an explicit ``instance=`` is passed. Inner same-toolkit dispatches
        find the seeded instance via ``dispatch`` Path 2 and reuse it.

        No-op (nesting-safe) if *toolkit* is already visible via the overlay
        or the shared cache. A caller-provided *instance* is seeded but not
        shut down on exit (caller owns it); a fresh instance is created and
        shut down on exit.

        Raises:
            ValueError: If ``instance.toolkit`` does not match *toolkit*.
        """
        toolkit = cls._normalize_toolkit(toolkit)
        if instance is not None and instance.toolkit != toolkit:
            raise ValueError(
                f"ToolInstance toolkit mismatch: instance.toolkit={instance.toolkit!r} != outer toolkit={toolkit!r}"
            )
        if _lookup_instance(toolkit) is not None:
            yield
            return

        scoped_inst = instance if instance is not None else cls(toolkit)
        overlay = _auto_persist_overlay.get() or {}
        token = _auto_persist_overlay.set({**overlay, toolkit: scoped_inst})
        try:
            yield
        finally:
            _auto_persist_overlay.reset(token)
            if instance is None:
                if _persist_mode.get():
                    with _lock:
                        cache = _active_cache()
                        if not hasattr(scoped_inst, "_cache_keys"):
                            scoped_inst._cache_keys = set()
                        if cache.setdefault(toolkit, scoped_inst) is scoped_inst:
                            scoped_inst._cache_keys.add(toolkit)
                        else:
                            # Slot already claimed (e.g. a concurrent call) — drop ours.
                            scoped_inst.shutdown()
                else:
                    scoped_inst.shutdown()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def __init__(self, toolkit: str, *, env_overrides: dict[str, str] | None = None) -> None:
        """Initialize ToolInstance.

        Args:
            toolkit (str): Either a toolkit (folder name like ``"pyrosetta"``)
                or a registered tool_key (``"pyrosetta-energy"``). Tool keys are
                auto-normalized to their toolkit via the registry.
            env_overrides (dict[str, str] | None): Per-instance env vars
                forwarded to the worker subprocess on spawn (final layer,
                wins over every other source). Used by ToolPool's CPU
                fan-out to pin ``OMP_NUM_THREADS`` etc. per worker.
        """
        # Accept either a toolkit or a tool_key for ergonomic flexibility.
        toolkit = self._normalize_toolkit(toolkit)
        self.toolkit = self._validate_toolkit(toolkit)
        self.device = "cpu"

        env_root = self._get_tool_envs_root()
        env_root.mkdir(parents=True, exist_ok=True)

        env_def_dir, env_name = self._resolve_env_def(toolkit)
        self.env_name = env_name
        self.env_path = env_root / f"{env_name}_env"
        if len(str(self.env_path)) > MAX_ENV_PREFIX_LEN:
            raise RuntimeError(
                f"Tool environment path is too long for conda's binary-relocation limit: "
                f"{self.env_path} exceeds {MAX_ENV_PREFIX_LEN} chars and would corrupt env "
                f"binaries. Set PROTO_HOME to a shorter path."
            )
        self.setup_script = env_def_dir / "setup.sh"
        self.script_path = self._find_script(toolkit)
        self._tool_env_vars = _parse_env_vars_file(env_def_dir / "env_vars.txt")
        self._env_overrides = env_overrides

        self._env_ready = False
        self._cache_keys: set[str] = set()
        self._instance_lock = threading.RLock()  # protects _worker lifecycle
        self._worker: PersistentWorker | None = None
        # Tracks previous reload-field values for restart detection
        self._reload_params: dict[str, Any] = {}
        # Latches True if any dispatch sets gpu_only; eviction then uses the
        # worker-kill path. See notes/tool-environments.md.
        self._gpu_only: bool = False
        # Latches True if any dispatch sets pin_visible_devices (CVD-pinned worker, respawns on device change).
        self._pin_visible_devices: bool = False

    def _ensure_env(self) -> None:
        """Build the tool's environment if it doesn't exist or is broken.

        Called lazily on first actual execution (not during ``__init__``),
        so that the double-check-locking loser in ``get()`` discards only
        a lightweight object — not one that already built an environment.

        Fails fast if this tool already failed to build in this process.
        On a cross-session failure (FAILED STATUS.txt from a previous run),
        logs a warning and retries the build.
        """
        if getattr(self, "_env_ready", False):
            return
        if self.toolkit in self._build_failures:
            tail = self._build_failures[self.toolkit]
            sentinel = self._parse_asset_sentinel(tail)
            if sentinel is not None:
                toolkit, asset_kind = sentinel
                raise MissingAssetError(toolkit, asset_kind, tail)
            raise RuntimeError(
                f"{self.toolkit}: previously failed setup; last error: {tail or '<no stderr>'}{self._fix_env_hint()}"
            )
        # Show one-time notice if using default storage locations
        from proto_tools.utils.proto_home import show_first_run_notice

        show_first_run_notice()

        # Serialize builds per env_name so concurrent calls for the same env (incl. shared envs) don't race micromamba create.
        with _lock:
            if self.env_name not in self._env_build_locks:
                self._env_build_locks[self.env_name] = threading.Lock()
        with self._env_build_locks[self.env_name]:
            # Re-check under the build lock in case the race winner already recorded a failure.
            if self.toolkit in self._build_failures:
                tail = self._build_failures[self.toolkit]
                sentinel = self._parse_asset_sentinel(tail)
                if sentinel is not None:
                    toolkit, asset_kind = sentinel
                    raise MissingAssetError(toolkit, asset_kind, tail)
                raise RuntimeError(f"{self.toolkit}: previously failed setup; last error: {tail or '<no stderr>'}")

            if not self.env_path.exists() or not self._is_env_ok():
                if not self.env_path.exists():
                    logger.info(
                        "First-time setup for %s. Installing dependencies. "
                        "This is a one-time process; subsequent runs will start much faster.",
                        self.toolkit,
                    )
                # Check for a stale status from a previous session
                status_file = self.env_path / "STATUS.txt"
                if status_file.exists():
                    status = status_file.read_text()
                    if status.startswith("SUCCESS"):
                        logger.info(
                            "Setup files changed for %s, rebuilding environment",
                            self.toolkit,
                        )
                    elif status.startswith("FAILED"):
                        current_hash = self._setup_hash()
                        if f"Setup hash: {current_hash}" in status:
                            summary = self._failure_summary()
                            hint = f": {summary}" if summary else ""
                            logger.warning(
                                "'%s' previously failed to build with the "
                                "same setup files (hash=%s)%s. Retrying — "
                                "if this keeps failing, the tool may not be "
                                "compatible with your system, or you may "
                                "need to accept a license agreement (e.g. "
                                "Hugging Face). Check logs for details.",
                                self.toolkit,
                                current_hash,
                                hint,
                            )
                        else:
                            logger.info(
                                "Setup files changed for %s — retrying venv build",
                                self.toolkit,
                            )
                try:
                    self._create_env()
                except Exception as exc:
                    # Store exception message as fallback if _create_env didn't populate _build_failures
                    if self.toolkit not in self._build_failures:
                        self._build_failures[self.toolkit] = str(exc)
                    raise
        self._env_ready = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        input_dict: dict[str, Any],
        *,
        script_path: Path | str | None = None,
        verbose: int = 0,
        timeout: int | None = None,
        reload_on: set[str] | None = None,
    ) -> dict[str, Any]:
        """Execute *input_dict* in the tool's venv and return the result.

        Args:
            input_dict (dict[str, Any]): JSON-serializable input for the standalone script.
            script_path (Path | str | None): Override the default standalone script.
            verbose (int): Verbosity level on the 0/1/2/3 scale; see :class:`BaseConfig`.
            timeout (int | None): Maximum seconds to wait.
            reload_on (set[str] | None): Config field names whose value changes should
                trigger a persistent worker restart.
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
        self._ensure_env()

    def shutdown(self, remove_from_cache: bool = True) -> None:
        """Stop the persistent worker (if any) and optionally remove from cache.

        Args:
            remove_from_cache (bool): If True, removes the instance from the cache.
                If False, keeps it so it can be restarted on next use.
        """
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
            name = getattr(self, "toolkit", "?")
            logger.debug("Shutting down persistent worker for %s", name)
            worker.stop()

        # Release device from DeviceManager
        cache_keys: set[str] = getattr(self, "_cache_keys", set())
        if cache_keys:
            instance_name = next(iter(cache_keys))
            device_manager = DeviceManager.get_instance()
            device_manager.release_device(instance_name)

        # Optionally evict from cache by stored keys
        if remove_from_cache:
            with _lock:
                cache = _active_cache()
                for k in cache_keys:
                    cache.pop(k, None)

    def _to(self, device: str) -> "ToolInstance":
        """Move this tool instance to a different device (internal).

        Called by device mismatch detection in ``_run_persistent`` (e.g.,
        recovery from CPU after eviction, or explicit device change).
        Not part of the public API — device changes should be driven by
        the config's ``device`` field at ``run()`` time.

        Sends a ``to_device`` command to the persistent worker subprocess,
        which moves the model to the specified device. Updates DeviceManager
        allocation tracking.

        Args:
            device (str): Target device (e.g., ``"cpu"``, ``"cuda"``, ``"cuda:0"``).

        Returns:
            ToolInstance: Self, for method chaining.
        """
        with self._instance_lock:
            # Get instance name for DeviceManager
            instance_name = next(iter(self._cache_keys)) if self._cache_keys else self.toolkit

            # If no worker exists yet, just update internal device tracking
            if self._worker is None:
                self.device = device
                logger.debug(
                    "ToolInstance._to(%s): No worker yet, device will be used on first run",
                    device,
                )
                return self

            # Send to_device command to worker
            def _move_worker(target_device: str) -> None:
                """Callback to send to_device command to worker subprocess."""
                if self._worker is not None:
                    command = {"command": "to_device", "device": target_device}
                    try:
                        result = self._worker.send(command, timeout=DEVICE_MOVE_TIMEOUT)
                        if not result.get("success", False):
                            logger.warning("Worker to_device returned success=False: %s", result)
                    except Exception as e:
                        logger.error("Failed to move worker to %s: %s", target_device, e)
                        raise

            # Update DeviceManager allocation and invoke move
            device_manager = DeviceManager.get_instance()
            resolved = device_manager.move_to_device(
                instance_name=instance_name,
                target_device=device,
                worker_callback=_move_worker,
            )

            # Update internal tracking with resolved device (e.g., "cuda" → "cuda:0")
            if resolved is not None:
                self.device = resolved

            return self

    def get_memory_stats(self) -> dict[str, Any]:
        """Get memory statistics from the worker process.

        Queries the persistent worker subprocess for its current memory usage.
        Only works with PyTorch and JAX tools that have implemented the
        ``get_memory_stats()`` function in their standalone/inference.py.

        Returns per-instance memory usage as reported by the framework (torch.cuda
        or JAX), which tracks only the memory allocated by this specific model
        instance within the worker process.

        Returns:
            dict[str, Any]: Memory statistics with standardized keys (available,
                framework, allocated_bytes, max_allocated_bytes) plus framework-specific
                keys. Returns ``{"available": False}`` if no worker is running.

        Example:
            >>> tool = ToolInstance.get("esm2", instance_name="esm2_1")
            >>> with ToolInstance.persist_tool("esm2", instance_name="esm2_1"):
            ...     result = run_esm2_sample(inputs, config, instance="esm2_1")
            ...     stats = tool.get_memory_stats()
            ...     if stats["available"]:
            ...         print(f"ESM2 using {stats['allocated_bytes'] / 1e9:.2f} GB")
        """
        with self._instance_lock:
            # If no worker, can't get stats
            if self._worker is None:
                return {
                    "available": False,
                    "error": "No worker running - start persistent worker first",
                }

            # Send get_memory_stats command to worker
            command = {"command": "get_memory_stats"}
            try:
                return self._worker.send(command, timeout=10)
            except Exception as e:
                logger.error("Failed to get memory stats: %s", e)
                return {"available": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Warmup timeout - per-config first-run detection
    # ------------------------------------------------------------------
    def _config_marker_path(self, reload_params: dict[str, Any]) -> Path:
        """Return the marker file path for a specific reload-param combination.

        Args:
            reload_params (dict[str, Any]): Current reload_on_change field values.

        Each unique set of reload_on_change values (e.g., model_name) gets its
        own marker, so switching to a new checkpoint triggers the warmup timeout.
        """
        key = hashlib.sha256(json.dumps(reload_params, sort_keys=True, default=str).encode()).hexdigest()[:12]
        return self.env_path / f".warmup_complete_{key}"

    def _needs_warmup(self, reload_params: dict[str, Any]) -> bool:
        """Check if this config combination has never completed successfully.

        Args:
            reload_params (dict[str, Any]): Current reload_on_change field values.

        Returns True if this specific set of reload_on_change values has
        never been run before, indicating checkpoints may need to be
        downloaded.
        """
        return not self._config_marker_path(reload_params).exists()

    def _mark_warmup_complete(self, reload_params: dict[str, Any]) -> None:
        """Mark that a config combination completed successfully."""
        with suppress(OSError):
            self._config_marker_path(reload_params).touch()

    def _apply_warmup_timeout(
        self,
        timeout: int | None,
        reload_params: dict[str, Any] | None = None,
    ) -> int | None:
        """Apply warm-up timeout for first run of a config combination.

        On the first use of a reload_on_change config combination, tools
        may need to download large checkpoint files. Use an extended
        timeout (60 minutes or the configured timeout, whichever is
        larger) to allow time for downloads.

        ``timeout=None`` is preserved as-is (no warm-up extension).

        Args:
            timeout (int | None): The configured timeout in seconds, or None for no timeout.
            reload_params (dict[str, Any] | None): Current reload_on_change field values.

        Returns:
            int | None: The effective timeout to use (extended on first run).
        """
        # Honor explicit opt-out: no warm-up extension when the user asked for no cap.
        if timeout is None:
            return None
        WARMUP_TIMEOUT = 3600  # 60 minutes
        params = reload_params or {}
        if self._needs_warmup(params):
            effective_timeout = max(WARMUP_TIMEOUT, timeout)
            if params:
                config_desc = ", ".join(f"{k}={v!r}" for k, v in sorted(params.items()))
                logger.info(
                    "First run of %s with %s. Model weights may need to download; "
                    "subsequent runs with this configuration will be faster.",
                    self.toolkit,
                    config_desc,
                )
            logger.debug(
                "Using extended warm-up timeout: %ds (configured: %ds)",
                effective_timeout,
                timeout,
            )
            return effective_timeout
        return timeout

    # ------------------------------------------------------------------
    # Persistent execution
    # ------------------------------------------------------------------
    def _run_persistent(
        self,
        input_dict: dict[str, Any],
        *,
        script_path: Path | None = None,
        verbose: int = 0,
        timeout: int | None = None,
        reload_on: set[str] | None = None,
    ) -> dict[str, Any]:
        """Execute via the persistent worker, restarting if config changed.

        Compares the current reload-field values against the previous call.
        If any tracked field changed (or the script path changed), the
        existing worker is stopped and a new one is created.  On the very
        first call the worker is lazily created.

        Args:
            input_dict (dict[str, Any]): JSON-serializable input for the standalone script.
            script_path (Path | None): Override the default standalone script.
            verbose (int): Verbosity level on the 0/1/2/3 scale. ``PROTO_WORKER_VERBOSE``
                env var acts as a global ceiling override (``max(verbose, env)``).
                Effective level >= 3 makes the worker's drain thread mirror untagged
                subprocess stderr to the parent's ``sys.stderr``.
            timeout (int | None): Maximum seconds to wait.
            reload_on (set[str] | None): Config field names that trigger worker restart on change.
        """
        self._ensure_env()
        sp = script_path or self.script_path
        # ``reload_on is None`` = caller isn't driving reload tracking; don't
        # compare against or overwrite the tracked state. Empty set still tracks.
        track_reload = reload_on is not None
        reload_params = {k: input_dict.get(k) for k in (reload_on or set())} if track_reload else None

        # Latch gpu_only / pin_visible_devices for later dispatches and the eviction callback.
        _invocation = _current_tool_invocation.get() or {}
        if _invocation.get("gpu_only"):
            self._gpu_only = True
        if _invocation.get("pin_visible_devices"):
            self._pin_visible_devices = True

        # Pinned workers have a fixed CUDA_VISIBLE_DEVICES and cannot move in-process; respawn on a device change.
        if self._worker is not None and self._pin_visible_devices:
            requested_device = input_dict.get("device", "")
            if _device_change_needed(self.device, requested_device):
                set_substatus("Restarting worker")
                self._worker.stop()
                self._worker = None

        if self._worker is not None:
            script_changed = self._worker.script_path != sp
            params_changed = reload_params is not None and reload_params != self._reload_params
            if script_changed or params_changed:
                if params_changed:
                    assert reload_params is not None
                    changed = {k for k in (reload_on or set()) if self._reload_params.get(k) != reload_params.get(k)}
                    logger.info(
                        "Config changed (%s) for %s, restarting worker",
                        ", ".join(
                            f"{k}: {self._reload_params.get(k)!r} → {reload_params[k]!r}" for k in sorted(changed)
                        ),
                        self.toolkit,
                    )
                set_substatus("Restarting worker")
                self._worker.stop()
                self._worker = None

        if track_reload:
            assert reload_params is not None
            self._reload_params = reload_params

        # Get instance name for DeviceManager (use first cache key if available)
        instance_name = next(iter(self._cache_keys)) if self._cache_keys else self.toolkit

        if self._worker is None:
            # Request device from DeviceManager
            device = input_dict.get("device", "cpu")
            device_manager = DeviceManager.get_instance()

            # Eviction callback - sends to_device directly to avoid lock ordering deadlock
            def eviction_callback(action: str) -> None:
                if action == "cpu":
                    # gpu_only/pinned tools can't be CPU-offloaded in-process; kill the worker (next dispatch respawns).
                    if self._gpu_only or self._pin_visible_devices:
                        worker = self._worker
                        self._worker = None
                        if worker is not None:
                            logger.warning(
                                "Evicting %s %s: killing worker (in-process CPU "
                                "offload not supported; fresh worker spawns on next "
                                "dispatch).",
                                "gpu_only tool" if self._gpu_only else "pinned tool",
                                self.toolkit,
                            )
                            worker.stop()  # never raises
                        self.device = "cpu"
                        return
                    worker = self._worker
                    if worker is not None:
                        command = {"command": "to_device", "device": "cpu"}
                        try:
                            result = worker.send(command, timeout=DEVICE_MOVE_TIMEOUT)
                            if not result.get("success", False):
                                logger.warning(
                                    "Worker to_device(cpu) returned success=False during eviction: %s",
                                    result,
                                )
                        except Exception as e:
                            logger.error("Failed to move worker to cpu during eviction: %s", e)
                            raise
                    self.device = "cpu"
                elif action == "shutdown":
                    # Stop worker directly to avoid lock ordering deadlock with shutdown()
                    worker = self._worker
                    self._worker = None
                    if worker is not None:
                        worker.stop()  # never raises

            allocated_device = device_manager.request_device(
                toolkit=self.toolkit,
                instance_name=instance_name,
                device=device,
                eviction_callback=eviction_callback,
            )

            # Track the logical device; a pinned worker places on its local cuda:0..N-1, otherwise the logical device.
            self.device = allocated_device
            input_dict["device"] = (
                _local_visible_device(allocated_device) if self._pin_visible_devices else allocated_device
            )

            set_substatus("Starting worker")
            # Latch effective verbose (max of config.verbose and PROTO_WORKER_VERBOSE) on the worker.
            effective_verbose = max(int(verbose), verbose_level_from_env())
            self._worker = PersistentWorker(
                toolkit=self.toolkit,
                env_path=self.env_path,
                script_path=sp,
                device=self.device,
                tool_env_vars=self._tool_env_vars,
                verbose=effective_verbose,
                env_overrides=self._env_overrides,
                pin_visible_devices=self._pin_visible_devices,
            )
        else:
            # Live worker that needs no physical-device change (pinned ones that do were respawned above).
            device_manager = DeviceManager.get_instance()
            requested_device = input_dict.get("device", "")

            # Non-pinned workers move in-process; pinned workers never reach here with a pending change.
            if not self._pin_visible_devices and _device_change_needed(self.device, requested_device):
                self._to(requested_device)

            # Update last-used timestamp on each dispatch
            device_manager.update_last_used(instance_name)

            # Worker placement target: a pinned worker's local cuda:0..N-1, otherwise the resolved logical device.
            input_dict["device"] = _local_visible_device(self.device) if self._pin_visible_devices else self.device

        # Apply warm-up timeout for first use of this config combination
        effective_timeout = self._apply_warmup_timeout(timeout, reload_params)

        set_substatus(f"Running {self.toolkit}")
        # _mark_warmup_complete only runs after a successful send(); a raising
        # send() short-circuits it, so a failed run never marks warmup complete.
        result = self._worker.send(input_dict, timeout=effective_timeout)
        if reload_params is not None:
            self._mark_warmup_complete(reload_params)
        return result

    # ------------------------------------------------------------------
    # One-shot execution
    # ------------------------------------------------------------------
    def _run_oneshot(
        self,
        input_dict: dict[str, Any],
        *,
        script_path: Path | None = None,
        verbose: int = 0,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Run in an ephemeral subprocess (no persistent worker).

        Writes *input_dict* to a temp JSON file, invokes the venv's Python on
        *script_path* with the input/output paths as arguments, reads the
        output JSON, and converts ``subprocess.TimeoutExpired`` to
        ``TimeoutError``.

        Subprocess stderr is always piped and demultiplexed by a daemon drain
        thread that calls :func:`proto_tools.utils.persistent_worker._drain_subprocess_stderr`.
        Tagged JSON log lines flow back to the parent's
        ``proto_tools.worker.{toolkit}.*`` logger; untagged lines accumulate in
        a bounded ring buffer used for crash context, and are also teed to the
        parent's stderr at verbose level >= 3.

        stdout is ``DEVNULL`` at levels 0/1/2 (one-shot uses files for IPC, not
        stdout), and inherited (``None``) at level 3 so library prints surface.

        Args:
            input_dict (dict[str, Any]): JSON-serializable input for the standalone script.
            script_path (Path | None): Override the default standalone script.
            verbose (int): Verbosity level on the 0-3 scale; see :class:`BaseConfig`.
                ``PROTO_WORKER_VERBOSE`` env var acts as a global ceiling override.
            timeout (int | None): Maximum seconds to wait.
        """
        self._ensure_env()
        sp = script_path or self.script_path
        copy_standalone_helpers(sp)  # type: ignore[arg-type]
        device = input_dict.get("device", self.device)
        # Effective level = max(per-call config, env-var ceiling).
        effective_verbose = max(int(verbose), verbose_level_from_env())
        inherit_stdout = effective_verbose >= 3
        raw_tee = effective_verbose >= 3
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.json"
            output_path = Path(tmp) / "output.json"

            with open(input_path, "w") as f:
                json.dump(input_dict, f)

            # Sets CUDA_VISIBLE_DEVICES based on device string
            env = _build_subprocess_env(
                device,
                tool_env_path=self.env_path,
                tool_env_vars=self._tool_env_vars,
            )
            env["TOOL_VENV_PATH"] = str(self.env_path)
            python_exe = str(self.env_path / "bin" / "python")

            if effective_verbose >= 1:
                logger.info(
                    "Running %s (one-shot) with device=%s",
                    sp.name,
                    device,
                )

            # Apply warm-up timeout for first use of this config
            effective_timeout = self._apply_warmup_timeout(timeout)

            # Show a spinner only when no tool-level progress bar is active
            # (tools like ESMFold create their own bar for batch iteration)
            show_spinner = not has_active_progress_bar()
            display_name = get_current_tool_function() or self.toolkit
            pbar = progress_bar(
                total=1,
                desc=f"Running {display_name}",
                bar_format="{desc} [{elapsed}]",
                show_bar=False,
                disable=not show_spinner,
            )

            ring_buffer: collections.deque[str] = collections.deque(maxlen=_stderr_buffer_lines())
            parent_logger = logging.getLogger(f"proto_tools.worker.{self.toolkit}")
            _apply_verbose_to_console_handlers(effective_verbose)
            cmd = [python_exe, str(sp), str(input_path), str(output_path)]

            try:
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    text=True,
                    bufsize=1,  # line-buffered so the drain thread sees lines promptly
                    stdout=None if inherit_stdout else subprocess.DEVNULL,
                    stderr=subprocess.PIPE,  # always PIPE so the drain thread can demux tagged JSON
                    start_new_session=True,  # own process group so timeouts can SIGKILL grandchildren
                )
                assert proc.stderr is not None  # PIPE guarantees this
                drain_thread = threading.Thread(
                    target=_drain_subprocess_stderr,
                    args=(proc.stderr, parent_logger, ring_buffer, raw_tee),
                    daemon=True,
                )
                drain_thread.start()

                try:
                    rc = proc.wait(timeout=effective_timeout)
                except subprocess.TimeoutExpired:
                    # killpg, not proc.kill, so grandchildren don't orphan with live CUDA handles.
                    with suppress(OSError):
                        os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        with suppress(OSError):
                            os.killpg(proc.pid, signal.SIGKILL)
                        with suppress(Exception):
                            proc.wait(timeout=5)
                    raise TimeoutError(f"{self.toolkit}: timed out after {effective_timeout}s") from None
                finally:
                    drain_thread.join(timeout=2)

                if rc != 0:
                    tail = " | ".join(ring_buffer) or "<no stderr>"
                    logger.error(
                        "Tool %s failed (exit %d):\n%s",
                        self.toolkit,
                        rc,
                        tail,
                    )
                    raise subprocess.CalledProcessError(rc, cmd, stderr=tail)

                pbar.update(1)
            finally:
                pbar.close()

            with open(output_path) as f:
                result = json.load(f)
                self._mark_warmup_complete({})
                return result  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Venv management
    # ------------------------------------------------------------------
    @staticmethod
    def _get_tool_envs_root() -> Path:
        """Return ``PROTO_HOME/proto_tool_envs/``.

        Always uses ``PROTO_HOME`` regardless of install mode; raises if too deep
        for conda's 255-byte binary-relocation limit (set a shorter ``PROTO_HOME``).
        See :func:`~.proto_home.get_proto_home`.
        """
        from proto_tools.utils.proto_home import get_proto_home

        root = get_proto_home() / "proto_tool_envs"
        if len(str(root)) > MAX_TOOL_ENVS_ROOT_LEN:
            raise RuntimeError(
                f"PROTO_HOME is too deep for conda's binary-relocation limit: {root} "
                f"exceeds {MAX_TOOL_ENVS_ROOT_LEN} chars and would corrupt env binaries. "
                f"Set PROTO_HOME to a shorter path."
            )
        return root

    @staticmethod
    def _get_micromamba_root() -> Path:
        """Return ``PROTO_HOME/.micromamba/``.

        Always uses ``PROTO_HOME`` regardless of install mode.
        See :func:`~.proto_home.get_proto_home`.
        """
        from proto_tools.utils.proto_home import get_proto_home

        return get_proto_home() / ".micromamba"

    @staticmethod
    def _configure_micromamba_root(mamba_root: Path) -> None:
        """Configure proto-tools' private micromamba root."""
        condarc = mamba_root / "condarc"
        desired = "register_envs: false"
        try:
            mamba_root.mkdir(parents=True, exist_ok=True)
            if condarc.exists():
                existing = condarc.read_text()
                lines = existing.splitlines()
                found = False
                updated_lines = []
                for line in lines:
                    if line.lstrip().startswith("register_envs:"):
                        updated_lines.append(desired)
                        found = True
                    else:
                        updated_lines.append(line)
                if not found:
                    updated_lines.append(desired)
                content = "\n".join(updated_lines) + "\n"
            else:
                existing = ""
                content = f"{desired}\n"

            if content != existing:
                condarc.write_text(content)
        except OSError as e:
            logger.warning("Could not configure micromamba root at %s: %s", mamba_root, e)

    @staticmethod
    def _is_managed_env_registration(path: Path, managed_roots: tuple[Path, ...]) -> bool:
        """Return whether ``path`` is under a proto-tools managed env root."""
        try:
            resolved = path.expanduser().resolve(strict=False)
        except OSError:
            return False
        return any(resolved == root or root in resolved.parents for root in managed_roots)

    @staticmethod
    def _cleanup_registered_micromamba_envs() -> None:
        """Remove proto-tools managed envs from conda's global registry."""
        registry = Path.home() / ".conda" / "environments.txt"
        if not registry.exists():
            return

        managed_roots = (
            ToolInstance._get_tool_envs_root().resolve(strict=False),
            ToolInstance._get_foundation_env_path().resolve(strict=False),
        )
        try:
            lines = registry.read_text().splitlines(keepends=True)
        except OSError as e:
            logger.warning("Could not read conda environment registry at %s: %s", registry, e)
            return

        kept = []
        removed = 0
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                env_path = Path(stripped)
                if ToolInstance._is_managed_env_registration(env_path, managed_roots):
                    removed += 1
                    continue
            kept.append(line)

        if removed == 0:
            return

        tmp_registry = registry.with_name(f"{registry.name}.{os.getpid()}.tmp")
        try:
            tmp_registry.write_text("".join(kept))
            tmp_registry.replace(registry)
            logger.info("Removed %d proto-tools env registrations from %s", removed, registry)
        except OSError as e:
            logger.warning("Could not update conda environment registry at %s: %s", registry, e)
            with suppress(OSError):
                tmp_registry.unlink()

    @staticmethod
    def _ensure_micromamba() -> Path:
        """Ensure micromamba is installed, download if missing.

        Uses a file lock to prevent concurrent downloads when multiple
        processes (e.g. pytest-xdist workers) race to install micromamba.

        Returns:
            Path: Path to micromamba binary.
        """
        mamba_root = ToolInstance._get_micromamba_root()
        mamba_bin = mamba_root / "bin" / "micromamba"

        if mamba_bin.exists():
            ToolInstance._configure_micromamba_root(mamba_root)
            ToolInstance._cleanup_registered_micromamba_envs()
            return mamba_bin

        # File lock prevents concurrent downloads from parallel processes
        from filelock import FileLock

        mamba_root.mkdir(parents=True, exist_ok=True)
        lock = FileLock(mamba_root / ".install.lock", timeout=300)
        with lock:
            # Re-check after acquiring lock - another process may have installed it
            ToolInstance._configure_micromamba_root(mamba_root)
            ToolInstance._cleanup_registered_micromamba_envs()
            if mamba_bin.exists():
                return mamba_bin

            logger.info("Downloading micromamba to %s...", mamba_root)

            system = platform.system()
            arch = platform.machine()

            if system == "Linux":
                if arch == "x86_64":
                    platform_id = "linux-64"
                elif arch == "aarch64" or arch == "arm64":
                    platform_id = "linux-aarch64"
                else:
                    raise RuntimeError(f"Unsupported Linux architecture: {arch}")
            elif system == "Darwin":  # macOS
                if arch == "x86_64":
                    platform_id = "osx-64"
                elif arch == "arm64":
                    platform_id = "osx-arm64"
                else:
                    raise RuntimeError(f"Unsupported macOS architecture: {arch}")
            else:
                raise RuntimeError(f"Unsupported operating system: {system} (arch: {arch})")

            urls = [
                f"https://micro.mamba.pm/api/micromamba/{platform_id}/latest",
                f"https://conda.anaconda.org/conda-forge/{platform_id}/micromamba-1.5.12-0.tar.bz2",
            ]
            last_err: Exception | None = None
            set_substatus("Downloading micromamba")
            for url in urls:
                # Attempt 1: curl + tar subprocess (fast, well-tested)
                try:
                    result = subprocess.run(
                        ["curl", "-Ls", "--retry", "2", "--retry-delay", "3", url],
                        check=True,
                        capture_output=True,
                    )
                    subprocess.run(
                        ["tar", "-xvj", "-C", str(mamba_root), "bin/micromamba"],
                        input=result.stdout,
                        check=True,
                        capture_output=True,
                    )
                    mamba_bin.chmod(0o755)
                    logger.info("Micromamba installed successfully from %s", url)
                    return mamba_bin
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    logger.info("curl/tar download failed (%s), trying urllib fallback", e)

                # Attempt 2: Python stdlib fallback (no external deps required)
                try:
                    import io
                    import tarfile as _tarfile
                    import urllib.request as _urllib_request

                    with _urllib_request.urlopen(url, timeout=120) as resp:  # noqa: S310
                        data = resp.read()
                    with _tarfile.open(fileobj=io.BytesIO(data), mode="r:bz2") as tar:
                        member = tar.getmember("bin/micromamba")
                        tar.extract(member, path=str(mamba_root), filter="data")
                    mamba_bin.chmod(0o755)
                    logger.info("Micromamba installed via urllib fallback from %s", url)
                    return mamba_bin
                except Exception as e:
                    last_err = e
                    logger.warning("Failed to download micromamba from %s: %s", url, e)
                    continue
            raise RuntimeError(f"Failed to download/extract micromamba from all sources: {last_err}")

    # ============================================================================
    # Foundation environment - shared system tools for all standalone envs
    # ============================================================================

    @staticmethod
    def _get_foundation_env_path() -> Path:
        """Return ``PROTO_HOME/.foundation_env/``."""
        from proto_tools.utils.proto_home import get_proto_home

        return get_proto_home() / ".foundation_env"

    @staticmethod
    def _ensure_foundation_env() -> Path | None:
        """Ensure a foundation environment is available, or confirm host satisfies it.

        The foundation env is a shared micromamba environment at
        ``PROTO_HOME/.foundation_env/`` containing git, curl, make, cmake,
        pkg-config, gcc, and gxx from conda-forge. Standalone tool setup
        scripts get its ``bin/`` prepended to PATH so they can compile and
        download regardless of the host system.

        Skipped entirely when the host already provides those tools at a
        recent enough version — see ``foundation_env.host_has_foundation_tools``.
        Override with ``PROTO_USE_FOUNDATION_ENV=1`` to force-install or ``=0``
        to force-skip (without probing).

        Uses a file lock to prevent concurrent creation when multiple processes
        race to set up the foundation env.

        Returns:
            Path | None: Path to the foundation env root, or ``None`` if the
                host satisfies the contract and creation was skipped.
        """
        from proto_tools.utils.foundation_env import MIN_FOUNDATION_GCC, host_has_foundation_tools

        override = os.environ.get("PROTO_USE_FOUNDATION_ENV")
        if override == "0":
            logger.debug("PROTO_USE_FOUNDATION_ENV=0 — skipping foundation env")
            return None
        if override != "1" and host_has_foundation_tools():
            logger.debug("Host provides foundation tools — skipping foundation env")
            return None

        foundation_path = ToolInstance._get_foundation_env_path()
        marker = foundation_path / ".ready"

        if marker.exists():
            return foundation_path

        from filelock import FileLock

        # Only create parent dir - micromamba create needs the prefix to not
        # exist (or be a valid conda env). The lock file lives in the parent.
        foundation_path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(foundation_path.parent / ".foundation_env.lock", timeout=600)
        with lock:
            # Re-check after acquiring lock - another process may have created it
            if marker.exists():
                return foundation_path

            mamba_bin = ToolInstance._ensure_micromamba()
            mamba_root = ToolInstance._get_micromamba_root()

            logger.info("Creating foundation environment at %s...", foundation_path)
            set_substatus("Setting up foundation environment (git, curl, gcc)")

            mamba_env = {**os.environ, "MAMBA_ROOT_PREFIX": str(mamba_root)}
            # Pin to MIN_FOUNDATION_GCC as a floor - keeps the installed env at
            # least as new as the threshold we accept on the host, so behavior
            # matches whether tools fall through to the host or use this env.
            gcc_pin = f"gcc>={MIN_FOUNDATION_GCC}"
            gxx_pin = f"gxx>={MIN_FOUNDATION_GCC}"
            try:
                subprocess.run(
                    [
                        str(mamba_bin),
                        "create",
                        "-y",
                        "-p",
                        str(foundation_path),
                        "git",
                        "curl",
                        "make",
                        "cmake",
                        "pkg-config",
                        gcc_pin,
                        gxx_pin,
                        "-c",
                        "conda-forge",
                    ],
                    check=True,
                    capture_output=True,
                    env=mamba_env,
                )
            except subprocess.CalledProcessError as e:
                detail = ToolInstance._called_process_detail(e)
                logger.error("Failed to create foundation environment:\n%s", detail)
                raise RuntimeError(
                    f"foundation_env: micromamba create failed at {foundation_path} (exit {e.returncode}): "
                    f"{' | '.join(detail.splitlines()) or '<no stderr>'}"
                ) from e

            marker.touch()
            logger.info("Foundation environment ready at %s", foundation_path)
            return foundation_path

    @staticmethod
    def _validate_python_version(version_str: str, source: str) -> str:
        """Validate `major.minor[.patch]` with major == 3, minor >= 8.

        Args:
            version_str (str): The version string to validate.
            source (str): Context for error messages (file path, key, etc.).

        Returns:
            str: The version string unchanged on success.

        Raises:
            ValueError: If format is invalid, components are not integers, or version < 3.8.
        """
        parts = version_str.split(".")
        if len(parts) not in (2, 3):
            raise ValueError(
                f"Invalid Python version format in {source}: '{version_str}'. "
                f"Expected format: '3.11' or '3.11.5' (major.minor or major.minor.patch)"
            )
        try:
            major = int(parts[0])
            minor = int(parts[1])
            if len(parts) == 3:
                _ = int(parts[2])  # Validate patch is numeric
        except ValueError:
            raise ValueError(
                f"Invalid Python version in {source}: '{version_str}'. Version components must be integers."
            ) from None
        if major != 3 or minor < 8:
            raise ValueError(f"Unsupported Python version in {source}: '{version_str}'. Requires Python 3.8 or higher.")
        return version_str

    @staticmethod
    def _parse_python_version(content: str, platform_key: str, source: str) -> str:
        """Parse python_version.txt content and resolve for the given platform.

        Format: keyed lines `key: value` with a required `default` key and
        optional platform overrides. Comments (`#` to end of line) and blank
        lines are ignored. Whitespace around `:` is stripped; keys are
        lowercased.

        Three-tier lookup, most specific wins:
            1. Exact platform key (e.g. ``linux-aarch64``)
            2. OS-only key (e.g. ``linux``)
            3. ``default`` (required catch-all)

        Args:
            content (str): Raw file contents.
            platform_key (str): Lookup key for the current platform, formatted as
                ``f"{platform.system().lower()}-{platform.machine()}"`` (e.g.
                ``"linux-aarch64"``).
            source (str): File path or other identifier used in error messages.

        Returns:
            str: The resolved Python version string (e.g. ``"3.11"``).

        Raises:
            ValueError: If the file is empty after stripping comments, contains a
                line without ``:``, has duplicate keys, is missing the required
                ``default`` key, or any value fails version validation.
        """
        versions: dict[str, str] = {}
        for raw_line in content.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(
                    f"Invalid line in {source}: '{line}'. "
                    f"Expected 'key: value' format (e.g., 'default: 3.11', 'linux-aarch64: 3.10')."
                )
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip()
            if not key:
                raise ValueError(f"Invalid line in {source}: '{line}'. Empty key before ':'.")
            if key in versions:
                raise ValueError(f"Duplicate key '{key}' in {source}.")
            versions[key] = ToolInstance._validate_python_version(value, f"{source} (key '{key}')")

        if not versions:
            raise ValueError(
                f"python_version.txt at {source} has no entries after stripping comments and "
                f"blank lines. Expected at least 'default: <version>'."
            )

        if "default" not in versions:
            raise ValueError(
                f"python_version.txt at {source} is missing required 'default' key. "
                f"Every python_version.txt must declare a default version like 'default: 3.11'."
            )

        # Three-tier lookup: most specific wins.
        if platform_key in versions:
            return versions[platform_key]
        os_key = platform_key.partition("-")[0]
        if os_key and os_key in versions:
            return versions[os_key]
        return versions["default"]

    def _get_python_version(self) -> str:
        """Get Python version for this tool from python_version.txt.

        Every tool with a ``standalone/`` directory must ship a
        ``standalone/python_version.txt`` that pins its Python version. A
        missing file is a hard error — see :meth:`_parse_python_version` for
        the format spec, including per-platform overrides.

        Returns:
            str: The resolved Python version string (e.g. ``"3.12"``).

        Raises:
            FileNotFoundError: If ``standalone/python_version.txt`` is missing.
            RuntimeError: If the file exists but cannot be read.
            ValueError: If the file content is malformed (see ``_parse_python_version``).
        """
        version_file = self.setup_script.parent / "python_version.txt"
        if not version_file.exists():
            raise FileNotFoundError(
                f"Tool {self.toolkit!r} is missing required standalone/python_version.txt "
                f"at {version_file}. Every tool must pin its Python version explicitly. "
                f"Create the file with a single line like 'default: 3.12'. "
                f"See notes/tool-environments.md for the format."
            )
        try:
            content = version_file.read_text()
        except Exception as e:
            raise RuntimeError(f"Failed to read {version_file} for tool '{self.toolkit}': {e}") from e
        platform_key = f"{platform.system().lower()}-{platform.machine()}"
        return self._parse_python_version(content, platform_key, str(version_file))

    _HELPER_ARTIFACTS = frozenset({"standalone_helpers", "standalone_helpers.sh", "standalone_helpers.py"})

    @staticmethod
    def _has_valid_standalone(standalone_dir: Path) -> bool:
        """Check that a standalone/ directory contains a setup.sh or runnable script."""
        if (standalone_dir / "setup.sh").is_file():
            return True
        return any((standalone_dir / name).is_file() for name in ("inference.py", "run.py"))

    @classmethod
    def _cleanup_stale_standalone_dirs(cls, tools_dir: Path | None = None) -> None:
        """Remove orphaned standalone helper artifacts left after tool category moves.

        When a tool moves between categories, git removes tracked files but
        gitignored runtime copies (standalone_helpers/, standalone_helpers.sh)
        persist at the old path. This detects and removes those phantom dirs.
        """
        if tools_dir is None:
            tools_dir = Path(__file__).parent.parent / "tools"

        for standalone_dir in list(tools_dir.rglob("standalone")):
            if not standalone_dir.is_dir():
                continue
            if cls._has_valid_standalone(standalone_dir):
                continue
            contents = {p.name for p in standalone_dir.iterdir()}
            if contents - cls._HELPER_ARTIFACTS - {"__pycache__"}:
                continue

            tool_dir = standalone_dir.parent
            logger.info(
                "Removing stale standalone helpers at %s (tool was likely moved to a different category)",
                standalone_dir,
            )
            shutil.rmtree(standalone_dir, ignore_errors=True)

            if tool_dir.exists():
                remaining = set(tool_dir.iterdir())
                if not remaining or remaining == {tool_dir / "__pycache__"}:
                    shutil.rmtree(tool_dir, ignore_errors=True)

    @classmethod
    def _get_tool_dirs(cls) -> dict[str, Path]:
        """Return a ``{toolkit: dir_path}`` mapping for all standalone tools.

        The mapping is computed once (single ``rglob``) and cached for the
        lifetime of the process. Phantom directories (stale helper copies
        without setup.sh or scripts) are filtered out and cleaned up.
        """
        if cls._tool_dir_cache is not None:
            return cls._tool_dir_cache

        tools_dir = Path(__file__).parent.parent / "tools"
        candidates: dict[str, list[Path]] = {}
        for standalone_dir in tools_dir.rglob("standalone"):
            if not standalone_dir.is_dir():
                continue
            if cls._has_valid_standalone(standalone_dir):
                candidates.setdefault(standalone_dir.parent.name, []).append(standalone_dir.parent)

        result: dict[str, Path] = {}
        for name, dirs in candidates.items():
            if len(dirs) == 1:
                result[name] = dirs[0]
            else:
                with_init = [d for d in dirs if (d / "__init__.py").is_file()]
                if len(with_init) == 1:
                    result[name] = with_init[0]
                    logger.warning(
                        "Tool %r found at multiple paths; using %s. Stale: %s",
                        name,
                        with_init[0],
                        [d for d in dirs if d != with_init[0]],
                    )
                else:
                    ordered = sorted(dirs)
                    result[name] = ordered[0]
                    logger.warning(
                        "Tool %r has multiple valid paths: %s. Using %s.",
                        name,
                        dirs,
                        ordered[0],
                    )

        cls._tool_dir_cache = result
        cls._cleanup_stale_standalone_dirs(tools_dir)
        return cls._tool_dir_cache

    @classmethod
    def _validate_toolkit(cls, toolkit: str) -> str:
        """Raise ValueError if *toolkit* has no standalone/ directory."""
        tool_dirs = cls._get_tool_dirs()
        if toolkit in tool_dirs:
            return toolkit
        raise ValueError(
            f"Invalid toolkit: {toolkit!r}. Available worker groups with standalone dirs: {sorted(tool_dirs)}"
        )

    @classmethod
    def _normalize_toolkit(cls, identifier: str) -> str:
        """Accept either a toolkit or a tool_key; return a toolkit.

        Because one toolkit = one persistent subprocess serving all tools
        in that folder, any tool_key inside a folder is operationally
        equivalent to the toolkit itself for worker-management purposes.
        Callers can pass either form; tool_keys are resolved to their
        toolkit via the registry.

        Invariant: this assumes one-worker-per-toolkit. If that ever
        changes (e.g. per-tool_key workers), remove this normalization.

        Args:
            identifier (str): Either a toolkit (folder name) or a tool
                registration key (``@tool(key=...)``).

        Returns:
            str: The toolkit identifier.
        """
        # Lazy import to avoid circular dep: tool_registry imports tool_instance
        from proto_tools.tools.tool_registry import ToolRegistry

        spec = ToolRegistry._registry.get(identifier)
        if spec is not None:
            resolved = spec.source_file.parent.name
            logger.debug("normalized tool_key %r to toolkit %r", identifier, resolved)
            return resolved
        # Already a toolkit - or invalid, in which case _validate_toolkit or
        # _resolve_env_def will surface a clear error downstream.
        return identifier

    @classmethod
    def _shared_envs_root(cls) -> Path:
        """Return the root directory for shared environment definitions.

        Shared envs let multiple tools (e.g. different model families that ship
        in the same Python package) reuse a single micromamba environment on
        disk. A tool opts in by placing ``standalone/shared_env.txt`` containing
        the shared env's directory name. See ``notes/tool-environments.md``.

        Returns:
            Path: Absolute path to ``proto_tools/shared_envs/``.
        """
        return Path(__file__).parent.parent / "shared_envs"

    @classmethod
    def _standalone_override_dir(cls, toolkit: str) -> Path | None:
        """Return a user override standalone dir from ``PROTO_<TOOLKIT>_STANDALONE_DIR``.

        The variable lets a user replace a tool's packaged ``standalone/`` env
        definition with their own editable copy (see ``proto-tools
        eject-standalone``), e.g. to patch ``setup.sh`` for a specific cluster or
        project. Returns ``None`` when the variable is unset.

        Raises:
            ValueError: If the variable is set but the path is not an existing
                directory, or is missing a required env-def file (``setup.sh``,
                ``python_version.txt``). The message names the variable, the
                resolved path, and how to produce a valid directory.
        """
        # toolkit is the underscored folder name here, but sanitize hyphens
        # defensively so the result is always a valid shell variable name.
        var = f"PROTO_{toolkit.upper().replace('-', '_')}_STANDALONE_DIR"
        raw = os.environ.get(var)
        if not raw:
            return None
        path = Path(raw).expanduser()
        if not path.is_dir():
            raise ValueError(
                f"{var}={raw!r} is set but no directory exists there. Create one with "
                f"`proto-tools eject-standalone {toolkit}` and point {var} at it, or unset {var}."
            )
        missing = [f for f in ("setup.sh", "python_version.txt") if not (path / f).is_file()]
        if missing:
            raise ValueError(
                f"{var} points to {path}, but a standalone override directory must contain "
                f"setup.sh and python_version.txt (missing: {missing}). "
                f"`proto-tools eject-standalone {toolkit}` produces a valid starting point."
            )
        return path.resolve()

    @classmethod
    def _resolve_env_def(cls, toolkit: str) -> tuple[Path, str]:
        """Resolve a tool's environment-definition directory and env name.

        Honors a ``PROTO_<TOOLKIT>_STANDALONE_DIR`` override when set (validated by
        ``_standalone_override_dir``); the override builds under an isolated env
        name so it never clobbers the packaged env and per-project overrides do
        not collide on disk. Otherwise delegates to ``_packaged_env_def``.

        Returns:
            tuple[Path, str]: ``(env_def_dir, env_name)``. ``env_path`` is
                ``env_root / f"{env_name}_env"``.
        """
        override = cls._standalone_override_dir(toolkit)
        if override is not None:
            digest = hashlib.sha256(str(override).encode()).hexdigest()[:8]
            return override, f"{toolkit}__override_{digest}"
        return cls._packaged_env_def(toolkit)

    @classmethod
    def eject_standalone(cls, toolkit: str, dest_root: Path) -> Path:
        """Copy a tool's packaged standalone env-def dir to ``dest_root/<toolkit>/``.

        Gives the user an editable, working-directory copy of every file the
        tool's environment is built from (``setup.sh``, ``python_version.txt``,
        ``requirements.txt``, ``env_vars.txt``, ...), as the starting point for a
        ``PROTO_<TOOLKIT>_STANDALONE_DIR`` override. Always copies the *packaged*
        definition (resolving shared envs), ignoring any active override.

        Returns:
            Path: The destination directory (``dest_root/<toolkit>``).

        Raises:
            ValueError: If the toolkit is unknown.
            FileExistsError: If the destination already exists.
        """
        toolkit = cls._validate_toolkit(cls._normalize_toolkit(toolkit))
        src, _ = cls._packaged_env_def(toolkit)
        dest = (Path(dest_root) / toolkit).resolve()
        if dest.exists():
            raise FileExistsError(f"{dest} already exists; remove it or pass a different --dir.")
        shutil.copytree(src, dest)
        return dest

    @classmethod
    def _packaged_env_def(cls, toolkit: str) -> tuple[Path, str]:
        """Resolve the packaged environment-definition directory and env name.

        If the tool's standalone dir contains ``shared_env.txt``, the env def
        comes from ``shared_envs/<name>/`` and the env name is ``<name>``;
        multiple tools opting into the same shared env will share one physical
        env on disk. Otherwise the env def is the tool's own ``standalone/``
        directory and the env name is the tool's directory name.

        Args:
            toolkit (str): The tool's directory name (e.g. ``"esm3"``).

        Returns:
            tuple[Path, str]: ``(env_def_dir, env_name)``. ``env_def_dir``
                contains ``setup.sh`` (and optionally ``requirements.txt``,
                ``python_version.txt``, ``env_vars.txt``). ``env_name`` is used
                to compute ``env_path = env_root / f"{env_name}_env"``.

        Raises:
            ValueError: If the tool dir is unknown, ``shared_env.txt`` points to
                a non-existent shared env, the tool dir contains both
                ``shared_env.txt`` and ``setup.sh`` (ambiguous), or no
                ``setup.sh`` is found at the resolved location.
        """
        tool_dir = cls._get_tool_dirs().get(toolkit)
        if tool_dir is None:
            raise ValueError(f"Unknown tool {toolkit!r}; valid: {sorted(cls._get_tool_dirs())}")

        standalone_dir = tool_dir / "standalone"
        marker = standalone_dir / "shared_env.txt"
        local_setup = standalone_dir / "setup.sh"

        if marker.is_file():
            # Catch any stray env-def file alongside the marker - silently
            # falling back to the shared dir would mask migration mistakes.
            stray_env_def_files = [
                name
                for name in ("setup.sh", "requirements.txt", "python_version.txt", "env_vars.txt")
                if (standalone_dir / name).is_file()
            ]
            if stray_env_def_files:
                raise ValueError(
                    f"Tool {toolkit!r} has shared_env.txt alongside stray env-def file(s) "
                    f"{stray_env_def_files} in {standalone_dir}; delete the local file(s) "
                    "to use the shared env, or delete shared_env.txt to use a private env."
                )
            shared_name = marker.read_text().strip()
            if not shared_name:
                raise ValueError(f"shared_env.txt for tool {toolkit!r} is empty")
            env_def_dir = cls._shared_envs_root() / shared_name
            shared_setup = env_def_dir / "setup.sh"
            if not shared_setup.is_file():
                raise ValueError(
                    f"Tool {toolkit!r} references shared env {shared_name!r}, but {shared_setup} does not exist."
                )
            return env_def_dir, shared_name

        if not local_setup.is_file():
            raise ValueError(f"No setup.sh found for tool {toolkit!r} (looked in {standalone_dir})")
        return standalone_dir, toolkit

    @classmethod
    def _find_script(cls, toolkit: str) -> Path:
        """Find the main standalone script (inference.py or run.py)."""
        tool_dir = cls._get_tool_dirs().get(toolkit)
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
        raise ValueError(f"No standalone script found for tool {toolkit!r}")

    def _setup_hash(self) -> str:
        """Short SHA-256 of setup.sh + requirements.txt + env_vars.txt + python_version.txt for change detection."""
        h = hashlib.sha256()
        h.update(self.setup_script.read_bytes())
        req = self.setup_script.parent / "requirements.txt"
        if req.exists():
            h.update(req.read_bytes())
        env_vars = self.setup_script.parent / "env_vars.txt"
        if env_vars.exists():
            h.update(env_vars.read_bytes())
        python_version_file = self.setup_script.parent / "python_version.txt"
        if python_version_file.exists():
            h.update(python_version_file.read_bytes())
            # Include the resolved version too - ensures the hash differs across
            # platforms when keyed-form overrides are in use, even if the file
            # content is identical (matters when PROTO_HOME is on shared storage).
            h.update(self._get_python_version().encode())
        return h.hexdigest()[:16]

    @staticmethod
    def _stderr_tail(stderr: str | None) -> str:
        """Return all non-empty lines of stderr for error messages."""
        if not stderr:
            return ""
        lines = [line for line in stderr.strip().splitlines() if line.strip()]
        return "\n".join(lines)

    @classmethod
    def _called_process_detail(cls, exc: subprocess.CalledProcessError) -> str:
        """Decode captured stderr (falling back to stdout) from a failed run.

        ``capture_output=True`` leaves the bytes on the exception but the
        default ``CalledProcessError`` message drops them, so callers must pull
        them out to surface anything useful.
        """

        def _decode(stream: object) -> str:
            if isinstance(stream, bytes):
                return stream.decode("utf-8", errors="replace")
            return stream if isinstance(stream, str) else ""

        # Tail each stream before falling back, so a blank-only stderr (which
        # tails to "") still yields stdout instead of short-circuiting on it.
        return cls._stderr_tail(_decode(exc.stderr)) or cls._stderr_tail(_decode(exc.stdout))

    @staticmethod
    def _parse_asset_sentinel(output: str | None) -> tuple[str, str] | None:
        """Detect the ``proto_resolve_asset_availability`` sentinel in output.

        The shell helper prints
        ``[proto-tools] ASSET_NOT_AVAILABLE: <toolkit>:<asset_kind>`` to
        stderr and exits 64. Returns ``(toolkit, asset_kind)`` when
        present (any line — checks the whole output, not just the tail, so it
        survives even if subsequent shell teardown adds noise after exit).

        Args:
            output (str | None): Combined stdout/stderr from the failed
                subprocess.

        Returns:
            tuple[str, str] | None: ``(toolkit, asset_kind)`` if the sentinel
                is present, else ``None``.
        """
        if not output:
            return None
        prefix = "[proto-tools] ASSET_NOT_AVAILABLE: "
        for line in output.splitlines():
            stripped = line.strip()
            if stripped.startswith(prefix):
                payload = stripped[len(prefix) :].strip()
                if ":" in payload:
                    toolkit, asset_kind = payload.split(":", 1)
                    return toolkit.strip(), asset_kind.strip()
        return None

    def _fix_env_hint(self) -> str:
        """Guidance appended to env-build failures, pointing at the ``fix-env`` skill."""
        var = f"PROTO_{self.toolkit.upper().replace('-', '_')}_STANDALONE_DIR"
        return (
            f"\n\nThis is an environment-build failure: {self.toolkit}'s standalone setup "
            "(setup.sh, etc.) is fully editable to match any hardware/software. "
            "Use the proto-tools `fix-env` skill — eject, patch, and iterate until it builds:\n"
            f"  proto-tools eject-standalone {self.toolkit}\n"
            f'  export {var}="$PWD/proto_standalone/{self.toolkit}"\n'
            "  # re-run with PROTO_ENV_VERBOSE=1, edit setup.sh, re-run."
        )

    def _failure_summary(self) -> str:
        """Extract a one-line error summary from a FAILED STATUS.txt."""
        status_file = self.env_path / "STATUS.txt"
        if not status_file.exists():
            return ""
        skip = (
            "FAILED",
            "Return code:",
            "Command:",
            "Setup hash:",
            "Timestamp:",
            "STDERR:",
        )
        for raw_line in status_file.read_text().splitlines():
            line = raw_line.strip()
            if line and not line.startswith(skip):
                return line[:200]
        return ""

    def _is_env_ok(self) -> bool:
        """Check STATUS.txt, verify the Python executable, and compare setup hash."""
        status_file = self.env_path / "STATUS.txt"
        if not status_file.exists():
            return False
        try:
            status = status_file.read_text()
            if not status.startswith("SUCCESS"):
                return False
            python_exe = self.env_path / "bin" / "python"
            if not python_exe.exists():
                return False
            result = subprocess.run(
                [str(python_exe), "--version"],
                capture_output=True,
                timeout=30,
                check=False,
            )
            if result.returncode != 0:
                return False
            # Compare stored hash to current setup files
            current_hash = self._setup_hash()
            return f"Setup hash: {current_hash}" in status
        except Exception:
            return False

    def _create_env(self) -> None:
        """Create (or rebuild) the tool's isolated environment.

        Removes a broken existing env, creates a fresh one via
        ``micromamba create``, runs ``standalone/setup.sh`` with the tool
        env on PATH, and writes STATUS.txt to record success or failure.
        """
        status_file = self.env_path / "STATUS.txt"

        # Remove env if setup files changed
        if self.env_path.exists() and not self._is_env_ok():
            logger.info("Removing environment at %s", self.env_path)
            shutil.rmtree(self.env_path)

        # Ensure micromamba is available
        mamba_bin = self._ensure_micromamba()

        # Get Python version for this tool
        python_version = self._get_python_version()

        # Create fresh micromamba environment
        set_substatus(f"Setting up environment for {self.toolkit}")
        # Set MAMBA_ROOT_PREFIX to the .micromamba dir (same filesystem as
        # tool_envs/) so the package cache lives alongside the envs and
        # micromamba can hardlink instead of copying.
        mamba_env = {**os.environ, "MAMBA_ROOT_PREFIX": str(mamba_bin.parent.parent)}
        try:
            subprocess.run(
                [
                    str(mamba_bin),
                    "create",
                    "-y",
                    "-p",
                    str(self.env_path),
                    f"python={python_version}",
                    "pip",
                    "uv",
                    "-c",
                    "conda-forge",
                ],
                check=True,
                capture_output=True,
                env=mamba_env,
            )
        except subprocess.CalledProcessError as e:
            detail = self._called_process_detail(e)
            logger.error(
                "micromamba create failed for %s (exit %s):\n%s",
                self.toolkit,
                e.returncode,
                detail,
            )
            self._build_failures[self.toolkit] = detail
            raise RuntimeError(
                f"{self.toolkit}: micromamba create failed (exit {e.returncode}): "
                f"{detail or '<no output>'}{self._fix_env_hint()}"
            ) from e

        # Run setup.sh directly (not via micromamba run, which overwrites PATH
        # and strips conda prefix - breaking access to git, curl, gcc, etc.)
        try:
            subprocess.run(["chmod", "+x", str(self.setup_script)], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            detail = self._called_process_detail(e)
            self._build_failures[self.toolkit] = detail
            raise RuntimeError(
                f"{self.toolkit}: failed to chmod +x {self.setup_script} (exit {e.returncode}): "
                f"{detail or '<no output>'}{self._fix_env_hint()}"
            ) from e

        # Ensure foundation environment (provides git, curl, gcc, g++).
        # Returns None when the host already provides these - see
        # foundation_env.host_has_foundation_tools for the contract.
        foundation_path = self._ensure_foundation_env()

        env = _build_subprocess_env(
            self.device,
            tool_env_path=self.env_path,
            tool_env_vars=self._tool_env_vars,
        )

        # Prepend foundation env bin/ so setup scripts always have access to
        # git, curl, gcc. Priority: tool_env/bin > foundation_env/bin > rest.
        # Skipped when foundation_path is None (host satisfies the contract).
        if foundation_path is not None:
            foundation_bin = str(foundation_path / "bin")
            path_parts = env["PATH"].split(":")
            if foundation_bin not in path_parts:
                tool_bin = str(self.env_path / "bin")
                if tool_bin in path_parts:
                    idx = path_parts.index(tool_bin) + 1
                    path_parts.insert(idx, foundation_bin)
                else:
                    path_parts.insert(0, foundation_bin)
                env["PATH"] = ":".join(path_parts)
            env["FOUNDATION_ENV_PATH"] = str(foundation_path.absolute())

        env["VENV_PATH"] = str(self.env_path.absolute())
        env["PYTHON_EXE"] = str(self.env_path.absolute() / "bin" / "python")
        env["PIP_EXE"] = str(self.env_path.absolute() / "bin" / "pip")
        env["MAMBA_BIN"] = str(mamba_bin.absolute())
        env["PACKAGE_ROOT"] = str(Path(__file__).parent.parent.parent.absolute())

        # Copy standalone_helpers.sh so setup.sh can source it
        sh_source = Path(__file__).parent / "standalone_helpers_source" / "standalone_helpers.sh"
        sh_target = self.setup_script.parent / "standalone_helpers.sh"
        if sh_source.exists():
            with suppress(Exception):
                shutil.copy2(sh_source, sh_target)

        returncode, combined_output = _run_setup_script(
            self.setup_script,
            cwd=self.setup_script.parent,
            env=env,
            log_path=self.env_path / "setup.log",
            toolkit=self.toolkit,
        )

        if returncode == 0:
            status_file.write_text(f"SUCCESS\nSetup hash: {self._setup_hash()}\n")
            logger.debug("Environment setup completed for %s", self.toolkit)
        else:
            tail = self._stderr_tail(combined_output)
            logger.error(
                "Environment setup failed for %s (exit %d):\n%s",
                self.toolkit,
                returncode,
                tail,
            )
            if combined_output:
                logger.debug("Full setup output for %s:\n%s", self.toolkit, combined_output)
            status_file.write_text(
                f"FAILED\n\n"
                f"Return code: {returncode}\n"
                f"Command: {self.setup_script}\n"
                f"Setup hash: {self._setup_hash()}\n"
                f"Timestamp: {datetime.datetime.now()}\n\n"
                f"OUTPUT:\n{combined_output or ''}\n"
            )
            self._build_failures[self.toolkit] = tail

            # Asset-not-provisioned signalled by proto_resolve_asset_availability:
            # raise the typed exception so the test layer converts it to a skip
            # instead of a hard failure.
            sentinel = self._parse_asset_sentinel(combined_output)
            if sentinel is not None:
                toolkit, asset_kind = sentinel
                raise MissingAssetError(toolkit, asset_kind, tail)

            raise RuntimeError(
                f"{self.toolkit}: setup.sh failed (exit {returncode}): {tail or '<no stderr>'}{self._fix_env_hint()}"
            )


atexit.register(ToolInstance.clear_all)
