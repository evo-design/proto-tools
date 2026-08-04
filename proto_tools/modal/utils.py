"""Shared utilities for Modal deployment services."""

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import modal

from proto_tools.modal.progress import container_progress
from proto_tools.utils.base_config import BaseConfig
from proto_tools.utils.tool_io import BaseToolOutput, MissingAssetError

_utils_logger = logging.getLogger(__name__)


# Startup GPU-readiness gate. A freshly-scheduled GPU container can report 0
# GPUs via nvidia-smi for a few seconds before the driver attaches; wait it out
# once at startup, off the per-request path. Module-level so tests can shrink.
_GPU_READY_TIMEOUT_S = 120.0
_GPU_READY_POLL_INITIAL_S = 0.25
_GPU_READY_POLL_MAX_S = 5.0


def ensure_gpu_ready(toolkit: str, *, timeout_s: float = _GPU_READY_TIMEOUT_S) -> int:
    """Block until the container's GPU is visible. Call first in a GPU service's startup hook.

    A freshly-scheduled GPU container can report 0 GPUs for a few seconds before
    the driver attaches; waiting once at startup keeps that race off the
    per-request path. Polls nvidia-smi with backoff until a GPU appears. Fast
    paths leave CPU containers unaffected: returns at once when a GPU is already
    visible, and raises at once when nvidia-smi is absent (no GPU to wait for).
    Only GPU services should call this.

    Relation to proto-tools: ``DeviceManager`` already retries this same race
    (~2.5s) the first time a tool requests a device (``_resolve_gpu_devices``).
    This gate is the earlier, longer backstop — it runs *before* ``persist_tool``
    loads the model onto CUDA and waits up to ``timeout_s``, where the 2.5s
    budget proved too short under cold starts. Do not call it from a
    ``@modal.enter(snap=True)`` snapshot hook: that runs on CPU during snapshot
    creation, so the gate would always time out.

    Args:
        toolkit (str): Tool name, for log/error lines.
        timeout_s (float): Max seconds to wait before giving up.

    Returns:
        int: Number of visible GPUs (>= 1).

    Raises:
        RuntimeError: nvidia-smi absent, ``CUDA_VISIBLE_DEVICES`` lists no usable
            devices, or no GPU within ``timeout_s`` — a GPU-less or misconfigured
            container, surfaced loudly at startup.
    """
    from proto_tools.utils.device import number_of_visible_gpus, nvidia_smi_present

    visible: int = number_of_visible_gpus()
    if visible > 0:
        return visible
    if not nvidia_smi_present():
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")
        raise RuntimeError(
            f"{toolkit}: no GPUs visible and nvidia-smi absent — not a GPU host "
            f"(CUDA_VISIBLE_DEVICES={cvd}); set device='cpu' to run on CPU"
        )

    # A CUDA_VISIBLE_DEVICES that lists no usable indices ("", "   ", "," or ",,")
    # makes number_of_visible_gpus() report 0 *permanently* without ever consulting
    # nvidia-smi, so polling could never recover — that's a deliberate "no GPU"
    # signal, not the cold-start race (which shows CVD unset or valid indices). The
    # check mirrors proto-tools' own emptiness test (drop blank comma entries) so it
    # tracks that permanent-0 state exactly; fail fast instead of burning the timeout.
    cvd_raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd_raw is not None and not [d for d in cvd_raw.split(",") if d.strip()]:
        raise RuntimeError(
            f"{toolkit}: CUDA_VISIBLE_DEVICES={cvd_raw!r} lists no usable devices — no GPU "
            f"can become visible, so the readiness poll cannot recover; set device='cpu' to run on CPU"
        )

    started = time.monotonic()
    deadline = started + timeout_s
    interval = _GPU_READY_POLL_INITIAL_S
    while time.monotonic() < deadline:
        time.sleep(interval)
        visible = number_of_visible_gpus()
        if visible > 0:
            _utils_logger.warning(
                "%s: GPU became visible after ~%.1fs at startup (cold-start driver-readiness race)",
                toolkit,
                time.monotonic() - started,
            )
            return visible
        interval = min(interval * 2, _GPU_READY_POLL_MAX_S)

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")
    raise RuntimeError(
        f"{toolkit}: no GPUs visible after waiting {timeout_s:.0f}s at startup "
        f"(CUDA_VISIBLE_DEVICES={cvd}); container has no usable GPU"
    )


def env_for() -> dict[str, str]:
    """Return the build-time env vars every Modal service should set.

    Applied via ``.env(env_for())`` *before* ``.run_function(_warmup, ..., include_source=False)``
    so it covers both the build-time warmup and the runtime container.
    ``PROTO_CAPTURE_ERRORS`` is intentionally NOT here — see
    :data:`RUNTIME_ENV` for why.

    Tool envs (micromamba) live on the container's ephemeral fs and are
    baked into the image layer at warmup time. Only model weights are
    routed onto a Modal Volume (``/weights``), where large blob writes
    work fine.

    ``PROTO_ENV_LOG_DIR`` points at a path on the model-cache volume so
    each toolkit's ``setup.sh`` log (mirrored by
    ``proto_tools.utils.tool_instance._run_setup``) survives the build
    container and can be retrieved with::

        modal volume ls proto-cache _deploy_logs/
        modal volume get proto-cache _deploy_logs/<toolkit>_setup.log /tmp/

    Persistence requires ``/weights`` to be mounted on the warmup
    function. Every service file in ``deployment/`` (GPU and CPU)
    mounts it via ``run_function(..., volumes={"/weights": MODEL_CACHE})``
    so the mirror lands on the persistent volume regardless of the
    underlying tool. ``proto_tools._run_setup`` ignores ``OSError`` on
    the copy, so a future service that forgets the mount degrades
    silently — its log goes to ephemeral fs and is lost on container
    exit, but the deploy itself still succeeds.

    Env-on-volume — i.e. routing ``PROTO_HOME`` onto the ``bio-tool-envs``
    Volume — is currently disabled because Modal Volumes v1 hit two hard
    walls: (1) a 500K inode quota that ~30 conda envs blow past in a
    single deploy, and (2) ``os.rename()`` returning EPERM on the FUSE
    mount, which breaks both micromamba and uv. Volumes v2 lifts the
    inode cap and may also fix rename semantics; tracked separately.

    To switch env-on-volume back on (after v2 migration is verified):
    take a ``toolkit: str`` arg again and add to the returned dict::

        home = f"/envs/{toolkit}"
        "PROTO_HOME": home,
        "PROTO_ENV_LOG_DIR": f"{home}/.deploy_logs",

    Then re-add a ``TOOL_ENVS`` env-storage volume (a ``modal.Volume`` named
    for tool envs, distinct from the ``proto-cache`` model cache) to
    ``proto_tools/modal/app.py`` and mount it at ``/envs`` on each service's
    ``image.run_function(...)`` and ``@app.cls(volumes=...)``.

    Returns:
        dict[str, str]: Mapping to pass to ``modal.Image.env(...)`` and
            inherit on every container of the service.
    """
    return {
        "PROTO_ENV_VERBOSE": "1",  # streams install output live
        "PROTO_MODEL_CACHE": "/weights",
        "PROTO_ENV_LOG_DIR": "/weights/_deploy_logs",
        "PROTO_NO_SPINNER": "1",
    }


# Runtime-only env, applied via ``.env(RUNTIME_ENV)`` *after*
# ``.run_function(_warmup, ..., include_source=False)`` so it does NOT affect build-time warmup.
#
# - During build (warmup): PROTO_CAPTURE_ERRORS is unset → @tool raises on
#   tool failures → Modal's run_function sees the exception → deploy goes red.
#   The point: a broken tool fails the deploy instead of hiding behind a
#   success=False envelope.
# - At runtime (incl. smoke tests): PROTO_CAPTURE_ERRORS=1 → @tool captures
#   into success=False envelopes that ride back through Modal RPC intact.
#   Modal's exception serialization is lossy (custom types collapse to
#   RemoteError) and Modal autoscaling/retry would replay deterministic tool
#   failures and burn GPU minutes. Capture-mode preserves the traceback and
#   hands the caller a structured result it can inspect.
RUNTIME_ENV: dict[str, str] = {"PROTO_CAPTURE_ERRORS": "1"}


# Don't add ``examples`` — :func:`stage_all_excluded_fixtures` reads
# .pdb/.json/.fasta/.a3m/.hmm from per-tool examples/ dirs at build time.
LOCAL_DIR_IGNORE: list[str] = [
    ".git",
    # Deploy logs are written under the working directory, which is this tree when deploying
    # from a clone. Mounting them lets a build invalidate its own source mid-run.
    "logs",
    "__pycache__",
    "*.pyc",
    "*.egg-info",
    "tests",
    "tutorials",
    "notes",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "*.ipynb",
    "*.md",
    "tool_envs",
    ".DS_Store",
    ".vscode",
    ".idea",
]


# Envelope fields the caller can recompute, so they don't ride back over the
# wire. ``metadata`` is kept: it's tool-specific payload and the only place that
# telemetry survives into the result. success/errors/warnings also stay — on a
# caught exception they're all that carries the traceback.
_BASE_OUTPUT_FIELDS = frozenset({"tool_id", "execution_time", "timestamp"})


def strip_base_fields(output_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip caller-recomputable envelope fields from a tool result dict.

    Args:
        output_dict (dict[str, Any]): Result from ``ToolOutput.model_dump()``.

    Returns:
        dict[str, Any]: Tool-specific fields plus ``success``/``errors``/``warnings``.
    """
    return {k: v for k, v in output_dict.items() if k not in _BASE_OUTPUT_FIELDS}


def apply_tool_envelope(
    run_fn: Callable[..., BaseToolOutput],
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run *run_fn* and reduce its ``BaseToolOutput`` to a serializable result dict."""
    try:
        result = run_fn(*args, **kwargs)
    except MissingAssetError as e:
        # The parts travel separately so the client can rebuild the exception rather than parse a
        # sentence. A caller distinguishes "this machine lacks the weights" from a real failure by
        # catching the type, which is the whole reason MissingAssetError exists.
        return {
            "success": False,
            "errors": [f"MissingAssetError: {e}"],
            "warnings": [],
            "missing_asset": True,
            "missing_asset_toolkit": e.toolkit,
            "missing_asset_kind": e.asset_kind,
        }
    return strip_base_fields(result.model_dump())


def dispatch_tool_call(
    run_fn: Callable[..., BaseToolOutput],
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Modal-worker entry point for one tool call.

    Results return inline in the function's return value. Modal moves any
    payload over 2 MiB through its own transport automatically, billed to
    the workspace running the container, so nothing needs configuring.
    Worker logs are read with ``modal app logs <app>``.
    """
    # This container hosts the tool for someone else, so a config that would reach for a large
    # local corpus adjusts itself before preprocess runs. Set per call rather than at import, since
    # the value has to be visible wherever the call actually executes.
    os.environ["PROTO_IS_HOSTED_ENV"] = "1"

    # @app.cls(timeout=...) is Modal's outer wall; null the inner timer so it can't fire early.
    configs = [arg for arg in (*args, *kwargs.values()) if isinstance(arg, BaseConfig)]
    for config in configs:
        config.timeout = None

    # Stream this call's log output back to the caller's spinner when they asked for it. Wired here
    # rather than in each service method, so every deployed tool reports progress on the same terms.
    partition = next((config._progress_partition for config in configs if config._progress_partition), None)
    level = next((config._progress_level for config in configs if config._progress_partition), logging.INFO)
    with container_progress(partition, level=level):
        return apply_tool_envelope(run_fn, *args, **kwargs)


def apply_standalone_overrides(image: modal.Image, tool: str, source_dir: Path | str) -> modal.Image:
    """Overlay ``proto_tools/modal/standalone_overrides/{tool}/`` onto the upstream standalone dir inside a Modal image.

    Resolves the tool's on-disk location via ``ToolRegistry`` (so the
    helper follows proto-tools directory reorganizations without updates)
    and rebases that path onto ``/pkg/proto-tools``, the container mount
    point. The override dir is a partial overlay — only listed files are
    copied; everything else comes from upstream untouched.

    Proto-tools' ``ToolInstance`` fingerprints the standalone dir (SHA-256
    of setup.sh + requirements.txt + env_vars.txt + python_version.txt), so
    changing any override file invalidates the cached micromamba env and
    triggers a clean rebuild on the next warmup call.

    Args:
        image (modal.Image): The Modal image to extend. Must already have
            ``add_local_dir(PROTO_TOOLS, "/pkg/proto-tools", ...)`` applied.
        tool (str): Tool directory name (e.g. ``"boltz2"``), matching the
            on-disk tool package name whose standalone directory is overlaid.
        source_dir (Path | str): Directory holding ``standalone_overrides/``,
            normally the service module's own directory.

    Returns:
        modal.Image: The image with override files copied in, or the
            original image unchanged if no override dir exists for the tool.

    Raises:
        LookupError: If ``tool`` is not registered in ``ToolRegistry``.
    """
    override_src = Path(source_dir) / "standalone_overrides"
    if not override_src.is_dir():
        return image

    import proto_tools
    from proto_tools.tools import ToolRegistry

    spec = next(
        (s for s in ToolRegistry._registry.values() if s.source_file.parent.name == tool),
        None,
    )
    if spec is None:
        raise LookupError(
            f"apply_standalone_overrides: no registered tool with on-disk "
            f"directory '{tool}'. Check spelling or that the tool is imported "
            f"before this helper runs."
        )

    # Resolve the tool's path relative to the proto_tools package root so we
    # don't depend on whether proto-tools was installed from source (editable)
    # or into site-packages — `spec.source_file` tracks the install location
    # but the path within `proto_tools/` is stable either way.
    proto_tools_pkg_root = Path(proto_tools.__file__).resolve().parent
    rel_tool_dir = spec.source_file.parent.relative_to(proto_tools_pkg_root)
    from proto_tools.modal.base_images import container_package_root

    container_standalone = f"{container_package_root()}/{rel_tool_dir.as_posix()}/standalone"

    return image.add_local_dir(
        str(override_src),
        f"/pkg/overrides/{tool}",
        copy=True,
    ).run_commands(
        f"cp -rf /pkg/overrides/{tool}/. {container_standalone}/",
    )


def _stage_excluded_fixtures() -> None:
    """Run inside the Modal image during build to stage wheel-excluded fixtures.

    Module-level so ``modal.Image.run_function`` can serialize it via
    cloudpickle. See :func:`stage_all_excluded_fixtures` for the full
    rationale.
    """
    import shutil
    from pathlib import Path

    import proto_tools
    from proto_tools.modal.base_images import container_package_root

    src_root = Path(container_package_root())
    dst_root = Path(proto_tools.__file__).parent
    if src_root == dst_root:
        return
    for ext in (".pdb", ".json", ".fasta", ".a3m", ".hmm"):
        for src in src_root.rglob(f"*{ext}"):
            dst = dst_root / src.relative_to(src_root)
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)


def stage_all_excluded_fixtures(image: modal.Image) -> modal.Image:
    """Copy wheel-excluded fixtures from the mounted tree into the installed package.

    Covers ``.pdb``/``.json``/``.fasta``/``.a3m``/``.hmm``.

    The proto-tools wheel's ``[tool.setuptools.package-data]`` whitelists
    ``.sh``/``.txt``/``.csv``/``.smi``/``.cif``/``.bib``/``.yaml`` and
    excludes everything else, including the example fixtures.
    Any tool whose ``example_input()`` reads such a file from
    ``Path(__file__).parent`` therefore raises ``FileNotFoundError`` inside
    the Modal container, even though the file is present in the editable
    install used during local dev.

    The image build already mounts the entire tree at
    ``/pkg/proto-tools`` (via ``add_local_dir``), so the fixtures *are* in
    the container — just not where ``example_input()`` looks. This helper
    adds a ``run_function`` build step that, after ``pip_install``, walks
    ``/pkg/proto-tools/proto_tools`` for every fixture extension and
    copies it to the matching path under the installed package dir
    (resolved at build time via ``proto_tools.__file__``). Idempotent:
    skips files that already exist.

    The staging logic lives in :func:`_stage_excluded_fixtures` so it can
    be cloudpickled into the build container — earlier ``run_commands``
    plus heredoc attempts hit Modal's Dockerfile parser, which mistook
    Python ``from`` lines for Dockerfile ``FROM`` instructions.

    Call this between the proto-tools install and ``run_function(_warmup, ..., include_source=False)``
    in any service whose warmup or runtime code goes through a tool's
    ``example_input()``::

        image = (
            CPU_BASE.add_local_dir(  # or GPU_BASE
                str(PROTO_TOOLS), "/pkg/proto-tools", copy=True, ignore=LOCAL_DIR_IGNORE
            )
            .pip_install("/pkg/proto-tools", extra_options="--no-deps")
            .env(env_for())
        )
        image = stage_all_excluded_fixtures(image).run_function(_warmup, include_source=False)

    Args:
        image (modal.Image): The Modal image to extend. Must already have
            ``add_local_dir(PROTO_TOOLS, "/pkg/proto-tools", ...)`` and the
            proto-tools install (``pip_install("/pkg/proto-tools", ...)`` or
            an editable equivalent) applied.

    Returns:
        modal.Image: The image with one extra build step that stages
            every excluded fixture into the installed package.
    """
    return image.run_function(_stage_excluded_fixtures, include_source=False)
