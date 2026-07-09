"""Run Python code or a script inside a tool's isolated environment."""

from __future__ import annotations

import collections
import logging
import os
import signal
import subprocess
import tempfile
import threading
from contextlib import ExitStack, suppress
from pathlib import Path
from typing import TYPE_CHECKING

from proto_tools.utils._worker_bootstrap import _copy_standalone_helpers
from proto_tools.utils.device_manager import DeviceManager
from proto_tools.utils.logging_config import verbose_level_from_env
from proto_tools.utils.persistent_worker import (
    _apply_verbose_to_console_handlers,
    _build_subprocess_env,
    _drain_subprocess_stderr,
    _stderr_buffer_lines,
)
from proto_tools.utils.tool_instance import ToolInstance

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


def run_in_env(
    toolkit: str,
    *,
    code: str | None = None,
    script: Path | str | None = None,
    args: Sequence[str] = (),
    device: str = "cpu",
    timeout: int = 600,
    verbose: int = 1,
) -> str:
    """Run Python inside ``toolkit``'s isolated environment and return its stdout.

    Exactly one of ``code`` or ``script`` must be provided. The environment is built if
    it does not yet exist (build time is not counted against ``timeout``). In-env
    proto-logging is streamed to the parent's ``proto_tools.worker.{toolkit}`` logger;
    only stdout is captured and returned.

    For ``cuda`` devices a transient :class:`DeviceManager` lease is held for the run — it
    is never evicted and honors shared-GPU mode (``allow_multiple_per_device``) — and the
    resolved device is exported to the program as ``RUN_IN_ENV_DEVICE``. ``timeout`` bounds
    the whole run, so set it generously when the program downloads weights on first use.

    Args:
        toolkit (str): Toolkit whose environment to run in (for example ``"esm2"``).
        code (str | None): A Python source string to execute.
        script (Path | str | None): Path to a Python script to execute.
        args (Sequence[str]): Command-line arguments passed to the code/script as ``sys.argv[1:]``.
        device (str): Device to lease and run on; the resolved value is exported as ``RUN_IN_ENV_DEVICE``.
        timeout (int): Maximum seconds to wait for the run.
        verbose (int): Verbosity for re-emitted logs; ``>= 3`` also tees raw stderr.

    Returns:
        str: The captured standard output of the run.

    Raises:
        ValueError: If not exactly one of ``code`` / ``script`` is provided.
        RuntimeError: If the subprocess exits with a non-zero status.
        TimeoutError: If the run exceeds ``timeout`` seconds.
    """
    if (code is None) == (script is None):
        raise ValueError("run_in_env requires exactly one of code= or script=.")

    instance = ToolInstance(toolkit)
    instance._ensure_env()
    _copy_standalone_helpers(str(instance.script_path))

    standalone_dir = str(instance.script_path.parent)
    python_exe = str(instance.env_path / "bin" / "python")

    effective_verbose = max(int(verbose), verbose_level_from_env())
    raw_tee = effective_verbose >= 3
    ring_buffer: collections.deque[str] = collections.deque(maxlen=_stderr_buffer_lines())
    parent_logger = logging.getLogger(f"proto_tools.worker.{toolkit}")
    _apply_verbose_to_console_handlers(effective_verbose)

    with ExitStack() as stack:
        # Lease GPUs through DeviceManager so run_in_env shares the same arbitration as tool
        # workers. Transient leases are never evicted and honor allow_multiple_per_device
        # (shared-GPU mode). Non-cuda devices need no lease.
        if device.startswith("cuda"):
            dm = DeviceManager.get_instance()
            run_device = stack.enter_context(dm.lease(toolkit, device=device, timeout=float(timeout + 60)))
        else:
            run_device = device

        env = _build_subprocess_env(
            run_device,
            tool_env_path=instance.env_path,
            tool_env_vars=instance._tool_env_vars,
        )
        env["TOOL_VENV_PATH"] = str(instance.env_path)
        env["RUN_IN_ENV_DEVICE"] = run_device  # resolved device for the in-env program to use
        env["PYTHONPATH"] = os.pathsep.join(p for p in (standalone_dir, env.get("PYTHONPATH", "")) if p)

        logger.debug("run_in_env: %s on device=%s (%s)", toolkit, run_device, "script" if script else "code")

        with tempfile.TemporaryDirectory() as tmp:
            if script is not None:
                target = Path(script)
            else:
                target = Path(tmp) / "_run_in_env.py"
                target.write_text(code or "")
            cmd = [python_exe, str(target), *args]

            proc = subprocess.Popen(  # noqa: S603
                cmd,
                env=env,
                text=True,
                bufsize=1,  # line-buffered so the drain thread sees lines promptly
                stdout=subprocess.PIPE,  # the data channel: captured and returned
                stderr=subprocess.PIPE,  # PIPE so the drain thread can demux tagged JSON logs
                start_new_session=True,  # own process group so timeouts can SIGKILL grandchildren
            )  # stdout/stderr are non-None because both are PIPE

            stdout_chunks: list[str] = []

            def _collect_stdout(stream: object) -> None:
                stdout_chunks.append(stream.read())  # type: ignore[attr-defined]

            stdout_thread = threading.Thread(target=_collect_stdout, args=(proc.stdout,), daemon=True)
            stderr_thread = threading.Thread(
                target=_drain_subprocess_stderr,
                args=(proc.stderr, parent_logger, ring_buffer, raw_tee),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()

            try:
                proc.wait(timeout=timeout)
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
                raise TimeoutError(f"run_in_env({toolkit!r}) timed out after {timeout}s.") from None
            finally:
                stdout_thread.join(timeout=2)
                stderr_thread.join(timeout=2)

    if proc.returncode != 0:
        tail = " | ".join(ring_buffer) or "<no stderr>"
        raise RuntimeError(f"run_in_env({toolkit!r}) failed (exit {proc.returncode}).\nstderr:\n{tail}")
    return "".join(stdout_chunks)
