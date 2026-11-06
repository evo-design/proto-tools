"""proto_tools/utils/weights_manifest.py.

Record the weight files a toolkit's setup leaves on disk, and find the ones that later go missing.

``setup.sh`` downloads weights only when an environment is built, but the files can disappear
afterwards while the environment still looks built: a scratch filesystem purges old files, or a
user clears part of the model cache. After a successful setup, ``write_manifest`` records every
file in the folders that setup filled, meaning each folder ``proto_resolve_weights_dir`` resolved
(which covers shared folders such as ``alphafold2/`` and ``PROTO_{TOOLKIT}_WEIGHTS_DIR``
overrides), together with the settings that located them. ``weights_problem`` compares that
record with the disk, and ``ToolInstance`` rebuilds the environment when anything is gone, which
runs ``setup.sh`` again and downloads it.

Only setup's folders are recorded: what a worker downloads while it runs (Hugging Face and torch
caches, ``download_weights``) is fetched again by the worker, so its loss needs no new setup.
Which files count, and how their presence is checked, is shared with every other completeness
check (``standalone_helpers.completeness``): caches the toolkit regenerates on its own
(``jax_cache/``, ``compile_cache/``, Hugging Face's ``xet/`` chunk cache), dotfiles, and
in-progress downloads are not recorded.
"""

import json
import logging
import os
from pathlib import Path

from proto_tools.utils.proto_home import get_proto_home
from proto_tools.utils.standalone_helpers_source.standalone_helpers.completeness import list_files, missing_files

logger = logging.getLogger(__name__)

MANIFEST_NAME = "weights_manifest.json"
# setup.sh appends "<toolkit>\t<dir>" here for every weights folder it resolves.
RECORD_NAME = "weights_dirs.tsv"
RECORD_ENV_VAR = "PROTO_WEIGHTS_RECORD"

_MANIFEST_VERSION = 1
_REGENERATED = ("jax_cache", "compile_cache", "xet")


def _weights_location(toolkits: set[str]) -> dict[str, str | None]:
    """The settings that decide where ``toolkits``' weights live."""
    location: dict[str, str | None] = {
        "PROTO_HOME": str(get_proto_home()),
        "PROTO_MODEL_CACHE": os.environ.get("PROTO_MODEL_CACHE") or None,
    }
    for toolkit in sorted(toolkits):
        var = f"PROTO_{toolkit.upper()}_WEIGHTS_DIR"
        location[var] = os.environ.get(var) or None
    return location


def _read_record(record: Path) -> dict[str, str]:
    """Toolkit to weights folder, from the lines ``proto_resolve_weights_dir`` appended."""
    if not record.is_file():
        return {}
    folders = {}
    for line in record.read_text().splitlines():
        toolkit, _, folder = line.partition("\t")
        if toolkit and folder:
            folders[toolkit] = folder
    return folders


def write_manifest(env_path: Path, toolkit: str) -> None:
    """Record the weight files present after ``toolkit``'s setup succeeded.

    Args:
        env_path (Path): The tool environment; holds the setup record and receives the manifest.
        toolkit (str): The toolkit whose setup just ran.
    """
    folders = _read_record(env_path / RECORD_NAME)
    files = {os.path.abspath(folder): list_files(folder, skip_dirs=_REGENERATED) for folder in folders.values()}
    manifest = {
        "version": _MANIFEST_VERSION,
        "location": _weights_location({toolkit, *folders}),
        "files": files,
    }
    (env_path / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1, sort_keys=True))
    logger.debug("Recorded %d weight files for %s", sum(map(len, files.values())), toolkit)


def weights_problem(env_path: Path) -> str | None:
    """Why the weights recorded for ``env_path`` are no longer usable, or ``None`` when they are.

    Args:
        env_path (Path): A tool environment whose setup succeeded.

    Returns:
        str | None: A description of the problem: no manifest (an environment built before
            weights were tracked), weights now located elsewhere, or files that are gone.
    """
    manifest_path = env_path / MANIFEST_NAME
    if not manifest_path.is_file():
        return "it has no record of its weight files yet"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as e:
        return f"its weights record is unreadable ({e})"
    if manifest.get("version") != _MANIFEST_VERSION:
        return "its weights record has an old format"

    recorded_location = manifest.get("location", {})
    toolkits = {
        var[len("PROTO_") : -len("_WEIGHTS_DIR")].lower() for var in recorded_location if var.endswith("_WEIGHTS_DIR")
    }
    current_location = _weights_location(toolkits)
    moved = sorted(var for var in current_location if current_location[var] != recorded_location.get(var))
    if moved:
        return f"its weights location changed ({', '.join(moved)})"

    missing = [
        os.path.join(folder, rel)
        for folder, rels in manifest.get("files", {}).items()
        for rel in missing_files(folder, rels)
    ]
    if missing:
        return f"{len(missing)} weight file(s) are missing, e.g. {missing[0]}"
    return None
