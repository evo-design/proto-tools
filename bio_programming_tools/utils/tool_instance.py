"""
bio_programming_tools/utils/tool_instance.py

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

from .base_config import DEFAULT_TIMEOUT, BaseConfig
from ._worker_bootstrap import _copy_standalone_helpers as copy_standalone_helpers
from .device_manager import DeviceManager
from .persistent_worker import (
    PersistentWorker,
    _build_subprocess_env,
    _parse_env_vars_file,
)

logger = logging.getLogger(__name__)

# Seconds to wait for a worker to complete a to_device move.
DEVICE_MOVE_TIMEOUT = 200

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

        Args:
            tool_name (str): Model-level folder name (e.g. ``"esm2"``, ``"progen2"``).
            instance_name (str | None): Explicit cache key. When None, the instance is
                cached under *tool_name* so that different operations on the same
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
        tool_name: str,
        input_dict: dict[str, Any],
        *,
        instance: str | ToolInstance | None = None,
        script_path: Path | str | None = None,
        config: BaseConfig | None = None,
    ) -> dict[str, Any]:
        """Run a tool, reusing a cached persistent instance if one exists.

        This is the primary entry point for tool wrappers.  When no
        persistent instance is cached, runs an ephemeral one-shot
        subprocess (no leak).  When a persistent instance exists (via
        :meth:`persistent` or :meth:`get`), reuses it.

        Args:
            tool_name (str): Model-level folder name (e.g. ``"esm2"``).
            input_dict (dict[str, Any]): JSON-serializable input for the standalone script.
            instance (str | ToolInstance | None): A ToolInstance object to use directly,
                a string cache key for persistent instance lookup, or None.
            script_path (Path | str | None): Override the default standalone script.
            config (BaseConfig | None): Tool configuration object. When provided,
                verbose, timeout, and reload_on are derived automatically.
        """
        # Derive execution parameters from config
        if config is not None:
            verbose = config.verbose
            timeout = config.timeout
            reload_on = type(config).reload_fields()
        else:
            verbose = False
            timeout = DEFAULT_TIMEOUT
            reload_on = None
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
                "dispatch(%s): persist mode, auto-caching instance (key=%r)",
                tool_name, key,
            )
            inst = cls.get(tool_name, instance_name=key)
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
        """Run a tool in an ephemeral subprocess — no caching, no worker.

        For GPU devices, acquires a transient lease from DeviceManager
        to prevent concurrent one-shot calls from stomping the same GPU.

        Args:
            tool_name (str): Model-level folder name (e.g. ``"esm2"``).
            input_dict (dict[str, Any]): JSON-serializable input for the standalone script.
            script_path (Path | str | None): Override the default standalone script.
            verbose (bool): Whether to print status messages.
            timeout (int | None): Maximum execution time in seconds.
        """
        inst = cls(tool_name)
        effective_script = Path(script_path) if script_path else inst.script_path
        device = input_dict.get("device")

        if device and device.startswith("cuda"):
            dm = DeviceManager.get_instance()
            lease_timeout = float((timeout or DEFAULT_TIMEOUT) + 60)
            with dm.lease(tool_name, device=device, timeout=lease_timeout) as allocated:
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
        tool_name: str,
        *,
        instance_name: str | None = None,
    ):
        """Context manager that caches a persistent worker for its duration.

        Tool wrappers called inside the block will find the cached
        instance via :meth:`dispatch` and reuse it::

            with ToolInstance.persist_tool("esmfold"):
                for i in range(500):
                    output = run_esmfold(inputs, config)

        The worker is shut down and removed from the cache on exit.

        Args:
            tool_name (str): Model-level folder name (e.g. ``"esmfold"``).
            instance_name (str | None): Explicit cache key. When None, uses tool_name.
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

        env_root = self._get_tool_envs_root()
        env_root.mkdir(parents=True, exist_ok=True)

        self.env_path = env_root / f"{tool_name}_env"
        self.setup_script = self._find_setup_script(tool_name)
        self.script_path = self._find_script(tool_name)
        self._tool_env_vars = _parse_env_vars_file(
            self.setup_script.parent / "env_vars.txt"
        )

        self._env_ready = False
        self._cache_keys: set[str] = set()
        self._instance_lock = threading.RLock()  # protects _worker lifecycle
        self._worker: PersistentWorker | None = None
        # Tracks previous reload-field values for restart detection
        self._reload_params: dict[str, Any] = {}

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
        if self.tool_name in self._build_failures:
            tail = self._build_failures[self.tool_name]
            hint = f"\n{tail}" if tail else ""
            raise RuntimeError(
                f"'{self.tool_name}' may not be compatible with your "
                f"system. Check logs for details.{hint}"
            )
        if not self.env_path.exists() or not self._is_env_ok():
            # Check for a stale status from a previous session
            status_file = self.env_path / "STATUS.txt"
            if status_file.exists():
                status = status_file.read_text()
                if status.startswith("SUCCESS"):
                    logger.info(
                        "Setup files changed for %s, rebuilding environment",
                        self.tool_name,
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
                            self.tool_name,
                            current_hash,
                            hint,
                        )
                    else:
                        logger.info(
                            "Setup files changed for %s — retrying venv build",
                            self.tool_name,
                        )
            try:
                self._create_env()
            except Exception as exc:
                # Store exception message as fallback if _create_env didn't populate _build_failures
                if self.tool_name not in self._build_failures:
                    self._build_failures[self.tool_name] = str(exc)
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
        verbose: bool = False,
        timeout: int | None = None,
        reload_on: set[str] | None = None,
    ) -> dict[str, Any]:
        """Execute *input_dict* in the tool's venv and return the result.

        Args:
            input_dict (dict[str, Any]): JSON-serializable input for the standalone script.
            script_path (Path | str | None): Override the default standalone script.
            verbose (bool): Whether to log progress.
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
            name = getattr(self, "tool_name", "?")
            logger.debug("Shutting down persistent worker for %s", name)
            worker.stop()

        # Release device from DeviceManager
        cache_keys = getattr(self, "_cache_keys", set())
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

    def _to(self, device: str) -> ToolInstance:
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
            instance_name = (
                next(iter(self._cache_keys)) if self._cache_keys else self.tool_name
            )

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
                            logger.warning(
                                "Worker to_device returned success=False: %s", result
                            )
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

        Returns
        -------
        dict[str, Any]
            Memory statistics dictionary with standardized keys across frameworks:

            **Common keys (all frameworks):**

            - ``available`` (bool): Whether stats are available
            - ``framework`` (str): "pytorch", "jax", or error info
            - ``allocated_bytes`` (int): Currently allocated GPU memory in bytes
            - ``max_allocated_bytes`` (int): Peak allocated memory since program start

            **PyTorch-specific keys:**

            - ``reserved_bytes`` (int): Reserved memory in PyTorch cache (not yet allocated)

            **JAX-specific keys:**

            - ``device_kind`` (str): Device type (e.g., "gpu", "cpu")
            - Legacy: ``bytes_in_use``, ``peak_bytes_in_use`` (same as standardized keys)

            **CLI tools or tools without support:**

            - ``available`` (bool): False
            - ``error`` (str): Error message or "not supported"

        Example
        -------
        >>> tool = ToolInstance.get("esm2", instance_name="esm2_1")
        >>> with ToolInstance.persist_tool("esm2", instance_name="esm2_1"):
        ...     result = run_esm2_sample(inputs, config, instance="esm2_1")
        ...     stats = tool.get_memory_stats()
        ...     if stats["available"]:
        ...         print(f"ESM2 using {stats['allocated_bytes'] / 1e9:.2f} GB")

        Notes
        -----
        - Returns ``{"available": False}`` if no worker is running
        - Only reports memory for THIS instance, not total GPU memory
        - For total GPU memory across all processes, use
          ``DeviceManager.get_gpu_memory_used("cuda:0")``
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
                result = self._worker.send(command, timeout=10)
                return result
            except Exception as e:
                logger.error("Failed to get memory stats: %s", e)
                return {"available": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Warmup timeout — per-config first-run detection
    # ------------------------------------------------------------------
    def _config_marker_path(self, reload_params: dict[str, Any]) -> Path:
        """Return the marker file path for a specific reload-param combination.

        Args:
            reload_params (dict[str, Any]): Current reload_on_change field values.

        Each unique set of reload_on_change values (e.g., model_name) gets its
        own marker, so switching to a new checkpoint triggers the warmup timeout.
        """
        key = hashlib.sha256(
            json.dumps(reload_params, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]
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
        try:
            self._config_marker_path(reload_params).touch()
        except OSError:
            pass  # non-critical — worst case warmup timeout applies again

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

        Args:
            timeout (int | None): The configured timeout in seconds, or None for no timeout.
            reload_params (dict[str, Any] | None): Current reload_on_change field values.

        Returns:
            int | None: The effective timeout to use (extended on first run).
        """
        WARMUP_TIMEOUT = 3600  # 60 minutes
        params = reload_params or {}
        if self._needs_warmup(params):
            if timeout is None:
                effective_timeout = WARMUP_TIMEOUT
            else:
                effective_timeout = max(WARMUP_TIMEOUT, timeout)
            if timeout is None or effective_timeout > timeout:
                config_desc = ""
                if params:
                    config_desc = " (config: %s)" % ", ".join(
                        f"{k}={v!r}" for k, v in sorted(params.items())
                    )
                logger.info(
                    "First run of %s%s detected, using extended warm-up timeout: "
                    "%ds (configured: %s)",
                    self.tool_name,
                    config_desc,
                    effective_timeout,
                    f"{timeout}s" if timeout is not None else "None",
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
        verbose: bool = False,
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
            verbose (bool): Whether to log progress.
            timeout (int | None): Maximum seconds to wait.
            reload_on (set[str] | None): Config field names that trigger worker restart on change.
        """
        self._ensure_env()
        sp = script_path or self.script_path
        reload_keys = reload_on or set()
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

        # Get instance name for DeviceManager (use first cache key if available)
        instance_name = (
            next(iter(self._cache_keys)) if self._cache_keys else self.tool_name
        )

        if self._worker is None:
            # Request device from DeviceManager
            device = input_dict.get("device", "cpu")
            device_manager = DeviceManager.get_instance()

            # Eviction callback — sends to_device directly to avoid lock ordering deadlock
            def eviction_callback(action: str) -> None:
                if action == "cpu":
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
                        try:
                            worker.stop()
                        except Exception as e:
                            logger.error(
                                "Failed to stop worker during RESTART eviction: %s", e
                            )

            allocated_device = device_manager.request_device(
                tool_name=self.tool_name,
                instance_name=instance_name,
                device=device,
                eviction_callback=eviction_callback,
            )

            # Override input_dict device with allocated device
            self.device = allocated_device
            input_dict["device"] = allocated_device

            self._worker = PersistentWorker(
                tool_name=self.tool_name,
                env_path=self.env_path,
                script_path=sp,
                device=self.device,
                tool_env_vars=self._tool_env_vars,
            )
        else:
            device_manager = DeviceManager.get_instance()
            requested_device = input_dict.get("device", "")

            # Move worker if config requests a different device than current
            needs_move = False
            if requested_device == "cuda" and self.device.startswith("cuda"):
                # Already on a GPU, generic "cuda" is satisfied
                pass
            elif requested_device and self.device != requested_device:
                needs_move = True

            if needs_move:
                self._to(requested_device)

            # Update last-used timestamp on each dispatch
            device_manager.update_last_used(instance_name)

            # Override input_dict device with worker's actual device
            # (resolves generic "cuda" to specific "cuda:0", etc.)
            input_dict["device"] = self.device

        # Apply warm-up timeout for first use of this config combination
        effective_timeout = self._apply_warmup_timeout(timeout, reload_params)

        try:
            result = self._worker.send(input_dict, timeout=effective_timeout)
            self._mark_warmup_complete(reload_params)
            return result
        except Exception:
            # Don't mark complete on failure
            raise

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

        Args:
            input_dict (dict[str, Any]): JSON-serializable input for the standalone script.
            script_path (Path | None): Override the default standalone script.
            verbose (bool): Whether to log progress.
            timeout (int | None): Maximum seconds to wait.
        """
        self._ensure_env()
        sp = script_path or self.script_path
        copy_standalone_helpers(sp)
        device = input_dict.get("device", self.device)
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

            if verbose:
                logger.info(
                    "Running %s (one-shot) with device=%s",
                    sp.name,
                    device,
                )

            # Apply warm-up timeout for first use of this config
            effective_timeout = self._apply_warmup_timeout(timeout)

            try:
                subprocess.run(
                    [python_exe, str(sp), str(input_path), str(output_path)],
                    env=env,
                    text=True,
                    check=True,
                    timeout=effective_timeout,
                    stdout=None if verbose else subprocess.PIPE,
                    stderr=None if verbose else subprocess.PIPE,
                )
            except subprocess.CalledProcessError as e:
                tail = self._stderr_tail(e.stderr or e.stdout or "")
                logger.error(
                    "Tool %s failed (exit %d):\n%s",
                    self.tool_name, e.returncode, tail,
                )
                raise
            except subprocess.TimeoutExpired:
                raise TimeoutError(
                    f"Tool {self.tool_name} timed out after {effective_timeout}s"
                ) from None

            with open(output_path) as f:
                result = json.load(f)
                self._mark_warmup_complete({})
                return result

    # ------------------------------------------------------------------
    # Venv management
    # ------------------------------------------------------------------
    @staticmethod
    def _get_tool_envs_root() -> Path:
        """Determine the ``tool_envs`` root directory.

        For editable installs (``pip install -e .``), finds the project root
        by walking up from this file looking for ``pyproject.toml``, then
        uses ``project_root/tool_envs/``.

        For non-editable installs (``pip install .``), the package is copied
        into site-packages and there's no project root.  Falls back to a
        user-level cache directory:
        ``$XDG_CACHE_HOME/bio_programming_tools/tool_envs/`` or
        ``~/.cache/bio_programming_tools/tool_envs/``.
        """
        for parent in Path(__file__).resolve().parents:
            if (parent / "pyproject.toml").exists():
                return parent / "tool_envs"
        cache_home = Path(
            os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
        )
        return cache_home / "bio_programming_tools" / "tool_envs"

    @staticmethod
    def _get_micromamba_root() -> Path:
        """Determine the ``.micromamba`` directory for global micromamba install.

        Uses same logic as tool_envs: editable install → project root, otherwise
        → user cache directory.
        """
        for parent in Path(__file__).resolve().parents:
            if (parent / "pyproject.toml").exists():
                return parent / ".micromamba"
        cache_home = Path(
            os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
        )
        return cache_home / "bio_programming_tools" / ".micromamba"

    @staticmethod
    def _ensure_micromamba() -> Path:
        """Ensure micromamba is installed, download if missing.

        Returns path to micromamba binary.
        """
        mamba_root = ToolInstance._get_micromamba_root()
        mamba_bin = mamba_root / "bin" / "micromamba"

        if mamba_bin.exists():
            return mamba_bin

        # Download micromamba
        logger.info("Downloading micromamba to %s...", mamba_root)
        mamba_root.mkdir(parents=True, exist_ok=True)

        import platform
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
            raise RuntimeError(
                f"Unsupported operating system: {system} (arch: {arch})"
            )

        urls = [
            f"https://micro.mamba.pm/api/micromamba/{platform_id}/latest",
            f"https://conda.anaconda.org/conda-forge/{platform_id}/micromamba-1.5.12-0.tar.bz2",
        ]
        last_err: Exception | None = None
        for url in urls:
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
            except subprocess.CalledProcessError as e:
                last_err = e
                logger.warning("Failed to download micromamba from %s: %s", url, e)
                continue
        raise RuntimeError(
            f"Failed to download/extract micromamba from all sources: {last_err}"
        )

    def _get_python_version(self) -> str:
        """Get Python version for this tool from python_version.txt.

        Looks for `standalone/python_version.txt` in the tool's directory.
        Returns version string (e.g., "3.11") or defaults to current Python version.

        Validates format: must be major.minor or major.minor.patch (e.g., "3.11" or "3.11.5").
        """
        version_file = self.setup_script.parent / "python_version.txt"

        if not version_file.exists():
            # Default to current Python version
            return f"{sys.version_info.major}.{sys.version_info.minor}"

        try:
            version_str = version_file.read_text().strip()
        except Exception as e:
            raise RuntimeError(
                f"Failed to read {version_file} for tool '{self.tool_name}': {e}"
            )

        # Validate format
        if not version_str:
            raise ValueError(
                f"python_version.txt for tool '{self.tool_name}' is empty. "
                f"Expected format: '3.11' or '3.11.5'"
            )

        parts = version_str.split(".")
        if len(parts) not in (2, 3):
            raise ValueError(
                f"Invalid Python version format in {version_file}: '{version_str}'. "
                f"Expected format: '3.11' or '3.11.5' (major.minor or major.minor.patch)"
            )

        try:
            major = int(parts[0])
            minor = int(parts[1])
            if len(parts) == 3:
                _ = int(parts[2])  # Validate patch is numeric
        except ValueError:
            raise ValueError(
                f"Invalid Python version in {version_file}: '{version_str}'. "
                f"Version components must be integers."
            )

        # Check reasonable bounds
        if major != 3 or minor < 8:
            raise ValueError(
                f"Unsupported Python version in {version_file}: '{version_str}'. "
                f"Requires Python 3.8 or higher."
            )

        return version_str

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
        """Short SHA-256 of setup.sh + requirements.txt + env_vars.txt + python_version.txt for change detection."""
        h = hashlib.sha256()
        h.update(self.setup_script.read_bytes())
        req = self.setup_script.parent / "requirements.txt"
        if req.exists():
            h.update(req.read_bytes())
        env_vars = self.setup_script.parent / "env_vars.txt"
        if env_vars.exists():
            h.update(env_vars.read_bytes())
        python_version = self.setup_script.parent / "python_version.txt"
        if python_version.exists():
            h.update(python_version.read_bytes())
        return h.hexdigest()[:16]

    @staticmethod
    def _stderr_tail(stderr: str | None, max_lines: int = 10) -> str:
        """Return the last few non-empty lines of stderr for error messages."""
        if not stderr:
            return ""
        lines = [l for l in stderr.strip().splitlines() if l.strip()]
        return "\n".join(lines[-max_lines:])

    def _failure_summary(self) -> str:
        """Extract a one-line error summary from a FAILED STATUS.txt."""
        status_file = self.env_path / "STATUS.txt"
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
            if f"Setup hash: {current_hash}" not in status:
                return False
            return True
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
        logger.info(
            "Setting up environment for %s (Python %s)...",
            self.tool_name,
            python_version
        )
        # Set MAMBA_ROOT_PREFIX to the .micromamba dir (same filesystem as
        # tool_envs/) so the package cache lives alongside the envs and
        # micromamba can hardlink instead of copying.
        mamba_env = {**os.environ, "MAMBA_ROOT_PREFIX": str(mamba_bin.parent.parent)}
        subprocess.run(
            [
                str(mamba_bin), "create", "-y", "-p", str(self.env_path),
                f"python={python_version}",
                "pip", "uv",
                "-c", "conda-forge"
            ],
            check=True,
            capture_output=True,
            env=mamba_env,
        )

        # Run setup.sh directly (not via micromamba run, which overwrites PATH
        # and strips conda prefix — breaking access to git, curl, gcc, etc.)
        subprocess.run(["chmod", "+x", str(self.setup_script)], check=True)
        env = _build_subprocess_env(
            self.device,
            tool_env_path=self.env_path,
            tool_env_vars=self._tool_env_vars,
        )
        env["VENV_PATH"] = str(self.env_path.absolute())
        env["PYTHON_EXE"] = str(self.env_path.absolute() / "bin" / "python")
        env["PIP_EXE"] = str(self.env_path.absolute() / "bin" / "pip")
        env["MAMBA_BIN"] = str(mamba_bin.absolute())
        env["PACKAGE_ROOT"] = str(Path(__file__).parent.parent.parent.absolute())

        # Copy standalone_helpers.sh so setup.sh can source it
        sh_source = Path(__file__).parent / "standalone_helpers_source" / "standalone_helpers.sh"
        sh_target = self.setup_script.parent / "standalone_helpers.sh"
        if sh_source.exists():
            try:
                shutil.copy2(sh_source, sh_target)
            except Exception:
                pass  # Non-critical — only needed by setup.sh that source it

        proc = subprocess.Popen(
            ["bash", str(self.setup_script)],
            cwd=self.setup_script.parent,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        raw_output, _ = proc.communicate()
        combined_output = raw_output.decode("utf-8", errors="replace") if raw_output else ""

        if proc.returncode == 0:
            status_file.write_text(
                f"SUCCESS\nSetup hash: {self._setup_hash()}\n"
            )
            logger.debug("Environment setup completed for %s", self.tool_name)
        else:
            tail = self._stderr_tail(combined_output)
            logger.error(
                "Environment setup failed for %s (exit %d):\n%s",
                self.tool_name, proc.returncode, tail,
            )
            if combined_output:
                logger.debug("Full setup output for %s:\n%s", self.tool_name, combined_output)
            status_file.write_text(
                f"FAILED\n\n"
                f"Return code: {proc.returncode}\n"
                f"Command: {self.setup_script}\n"
                f"Setup hash: {self._setup_hash()}\n"
                f"Timestamp: {datetime.datetime.now()}\n\n"
                f"OUTPUT:\n{combined_output or ''}\n"
            )
            self._build_failures[self.tool_name] = tail
            hint = f"\n{tail}" if tail else ""
            raise RuntimeError(
                f"'{self.tool_name}' may not be compatible with your "
                f"system. setup.sh failed (exit {proc.returncode}).{hint}"
            )
