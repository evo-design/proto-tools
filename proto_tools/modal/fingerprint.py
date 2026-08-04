"""Hashes that detect drift between local proto-tools and a deployment.

Covers a tool's schemas, its source, and its standalone environment.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

from pydantic import BaseModel

# Bump when the hashing rules change. Without this, redefining a fingerprint
# makes every deployment report drift at once, which is indistinguishable from
# everything actually being broken.
ALGORITHM = 2

# JSON-schema keys that describe presentation or local behaviour rather than
# the wire contract. ``default`` is here because callers dump every field with
# its default already resolved, so a deployed default never applies — changing
# one cannot affect a call, and warning about it would be noise.
_IGNORED_SCHEMA_KEYS = frozenset(
    {
        "description",
        "title",
        "examples",
        "default",
        "include_in_key",
        "reload_on_change",
        "_field_type",
    }
)

_HASH_LENGTH = 12


class ToolFingerprint(BaseModel):
    """Schema and environment hashes for one tool key."""

    algorithm: int = ALGORITHM
    schema_hash: str
    code_hash: str
    env_hash: str
    env_name: str


def _strip(node: Any) -> Any:
    """Recursively drop presentation-only keys and sort for stable ordering."""
    if isinstance(node, dict):
        return {k: _strip(v) for k, v in sorted(node.items()) if k not in _IGNORED_SCHEMA_KEYS}
    if isinstance(node, list):
        return [_strip(v) for v in node]
    return node


def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()[:_HASH_LENGTH]


def schema_hash(tool_key: str) -> str:
    """Hash the wire contract of ``tool_key``.

    Pydantic inlines referenced models into ``$defs``, so a change to a shared
    entity such as ``Structure`` shifts every tool that accepts one without any
    dependency tracking here.
    """
    from proto_tools.tools import ToolRegistry

    spec = ToolRegistry.get(tool_key)
    models = (spec.input_model, spec.config_model, spec.output_model)
    return _digest(json.dumps([_strip(m.model_json_schema()) for m in models], sort_keys=True))


def env_hash(toolkit: str) -> tuple[str, str]:
    """Hash a toolkit's env-definition directory. Returns ``(hash, env_name)``.

    Covers every file in the directory tree, including ``standalone_helpers/``,
    so the standalone ``inference.py`` and the helpers it calls are both included — proto-tools' own ``_setup_hash`` deliberately omits it, since it
    fingerprints the environment rather than the behaviour, and behaviour drift
    is exactly what this is meant to catch.

    ``env_name`` differs from ``toolkit`` when the toolkit redirects to a shared
    environment via ``shared_env.txt``. Both are empty for a tool that owns no
    environment, which cannot drift.
    """
    from proto_tools.utils.tool_instance import ToolInstance

    try:
        env_dir, env_name = ToolInstance._resolve_env_def(toolkit)
    except ValueError:
        # A tool with no standalone environment -- an HTTP-only database fetch, say --
        # has no environment to drift. Schema and code hashes still apply.
        return "", ""
    digest = hashlib.sha256()
    for path in sorted(p for p in env_dir.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        # Relative, so the hash does not move with the tree's location on disk.
        digest.update(path.relative_to(env_dir).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:_HASH_LENGTH], env_name


# Framework modules every tool shares. Hashing them would move every fingerprint at once
# on any framework edit, which is indistinguishable from everything being broken -- and a
# change there that alters the wire contract already moves ``schema_hash``.
_FRAMEWORK_PREFIXES = ("proto_tools/utils/", "proto_tools/tools/tool_registry.py")

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _defining_files(tool_key: str) -> list[pathlib.Path]:
    """Return the first-party files defining a tool's behaviour and its models.

    Autodetected: the run function's own module, plus the defining module of every
    class in each model's MRO, so a shared base such as
    ``structure_prediction/shared_data_models.py`` is covered without being declared.
    """
    import inspect

    from proto_tools.tools import ToolRegistry

    spec = ToolRegistry.get(tool_key)
    found: set[pathlib.Path] = {pathlib.Path(spec.source_file).resolve()}
    for model in (spec.input_model, spec.config_model, spec.output_model):
        for klass in getattr(model, "__mro__", ()):
            try:
                origin = inspect.getsourcefile(klass)
            except TypeError:
                continue
            if origin:
                found.add(pathlib.Path(origin).resolve())

    keep = []
    for path in found:
        try:
            rel = path.relative_to(_PACKAGE_ROOT).as_posix()
        except ValueError:
            continue  # third-party or stdlib
        if rel.startswith("proto_tools/") and not rel.startswith(_FRAMEWORK_PREFIXES):
            keep.append(path)
    return sorted(keep)


def code_hash(tool_key: str) -> str:
    """Hash the first-party source defining a tool, so behaviour drift is visible.

    ``schema_hash`` covers the wire contract and ``env_hash`` the standalone
    environment; neither moves when a run function's logic changes in place.

    Args:
        tool_key (str): Registry key of the tool.

    Returns:
        str: Truncated SHA-256 over the tool's defining sources.
    """
    digest = hashlib.sha256()
    for path in _defining_files(tool_key):
        digest.update(path.relative_to(_PACKAGE_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:_HASH_LENGTH]


def toolkit_for(tool_key: str) -> str:
    """Return the on-disk toolkit directory name owning ``tool_key``."""
    from proto_tools.tools import ToolRegistry

    return str(ToolRegistry.get(tool_key).source_file.parent.name)


def fingerprint(tool_key: str) -> ToolFingerprint:
    """Build the full fingerprint for one tool key."""
    env, env_name = env_hash(toolkit_for(tool_key))
    return ToolFingerprint(
        schema_hash=schema_hash(tool_key), code_hash=code_hash(tool_key), env_hash=env, env_name=env_name
    )


def fingerprints_for_service(service_class: str) -> dict[str, ToolFingerprint]:
    """Fingerprint every tool key registered to ``service_class``."""
    from proto_tools.modal.client import available_tools

    return {
        key: fingerprint(key) for key, (cls_name, _method) in available_tools().items() if cls_name == service_class
    }


# Must match the volume each service mounts at /weights. Kept here rather than
# imported from deployment/ so the client can run this check without importing
# the service modules, which construct Modal images at import time.
_MANIFEST_DIR = "/_fingerprints"


def manifest_path(service_class: str) -> str:
    """Volume path holding the recorded fingerprints for one service."""
    return f"{_MANIFEST_DIR}/{service_class}.json"


def write_manifest(service_class: str, environment: str | None = None) -> int:
    """Record the local fingerprints of ``service_class`` onto the cache volume.

    Called after a successful deploy, from the same checkout that was deployed,
    so the recorded values describe what actually shipped. Returns the number of
    tools recorded.
    """
    import io

    import modal

    fps = fingerprints_for_service(service_class)
    if not fps:
        return 0
    payload = json.dumps(
        {"algorithm": ALGORITHM, "tools": {k: v.model_dump() for k, v in fps.items()}},
        sort_keys=True,
    ).encode()

    from proto_tools.modal.app import CACHE_VOLUME_NAME

    volume = modal.Volume.from_name(CACHE_VOLUME_NAME, environment_name=environment, create_if_missing=True)
    with volume.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(payload), manifest_path(service_class))
    return len(fps)


def read_manifest(service_class: str) -> dict[str, Any] | None:
    """Read a service's recorded fingerprints, or ``None`` if absent or unreadable.

    Absent is normal: a deployment made without ``scripts/deploy.py``, or one
    predating this mechanism, simply has nothing recorded.
    """
    import modal

    try:
        from proto_tools.modal.app import CACHE_VOLUME_NAME

        volume = modal.Volume.from_name(CACHE_VOLUME_NAME)
        raw = b"".join(volume.read_file(manifest_path(service_class)))
        parsed: dict[str, Any] = json.loads(raw)
        return parsed
    except Exception:
        return None


def drift_warnings(tool_key: str, service_class: str) -> list[str]:
    """Return human-readable drift warnings for ``tool_key``, empty if aligned.

    Never raises: a check that breaks a working call is worse than a missed
    warning, so any failure reads as "nothing to report".
    """
    try:
        recorded = read_manifest(service_class)
        if recorded is None:
            return []
        if recorded.get("algorithm") != ALGORITHM:
            return [
                f"{tool_key}: deployment recorded fingerprints with algorithm "
                f"{recorded.get('algorithm')}, this client uses {ALGORITHM}; cannot compare."
            ]
        entry = recorded.get("tools", {}).get(tool_key)
        if entry is None:
            return []

        local = fingerprint(tool_key)
        out = []
        if entry.get("schema_hash") != local.schema_hash:
            out.append(
                f"{tool_key}: the deployed tool's schema differs from your local proto-tools "
                f"(deployed {entry.get('schema_hash')}, local {local.schema_hash}). Calls may fail, "
                f"or fields a newer deployment returns may be silently dropped. Redeploy to realign."
            )
        if entry.get("code_hash") != local.code_hash:
            out.append(
                f"{tool_key}: the deployed tool's code differs from your local proto-tools "
                f"(deployed {entry.get('code_hash')}, local {local.code_hash}). Results may not be "
                f"comparable with local runs. Redeploy to realign."
            )
        if entry.get("env_hash") != local.env_hash:
            out.append(
                f"{tool_key}: the deployed tool's environment differs from your local proto-tools "
                f"(deployed {entry.get('env_hash')}, local {local.env_hash}). Calls will still "
                f"succeed, but results may not be comparable to a local run."
            )
        return out
    except Exception:
        return []
