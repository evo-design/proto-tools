"""Provision registered datasets from the command line.

User-invoked CLI over :func:`proto_tools.databases.provision.provision`, which downloads and builds
each dataset into the databases root (``$PROTO_DATABASES_DIR`` if set, else
``$PROTO_MODEL_CACHE/databases/``; ``--workdir`` overrides per-invocation). Idempotent: re-runs
skip already-provisioned datasets and already-downloaded files. Adding a dataset is a new entry in
``databases/entries/``, with no change here.

Usage::

    # Default (no args): UniRef30 only, smallest viable bootstrap
    python -m proto_tools.tools.sequence_alignment.mmseqs2.setup_databases

    # Specific datasets
    python -m ...setup_databases uniref30-2302 small-bfd

    # Named preset matching a predictor's preferred_datasets
    python -m ...setup_databases --preset af3-protein

    # List datasets / presets
    python -m ...setup_databases --list

    # Custom cache root (datasets still go in <root>/<dataset_name>/)
    python -m ...setup_databases --workdir /custom/path uniref30-2302

Prerequisites:
    The script auto-resolves ``mmseqs`` from the ``mmseqs2`` toolkit's
    micromamba env at ``$PROTO_HOME/proto_tool_envs/mmseqs2_env/bin/mmseqs``,
    building the env on first run if needed (no PATH setup required by the
    caller). ``zstd`` (for ``.zst``-distributed datasets) is still expected on
    PATH where applicable.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from proto_tools.databases import DatasetRegistry, get_databases_root
from proto_tools.databases.provision import provision
from proto_tools.utils.tool_instance import ToolInstance

logger = logging.getLogger(__name__)

# ============================================================================
# Presets — mirror predictor `preferred_datasets` defaults from the design doc
# ============================================================================

_AF3_PROTEIN = [
    "uniref90-2022-05",
    "mgnify-2022-05",
    "small-bfd",
    "uniprot-2021-04",
    "pdb-seqres-2022-09-28",
]
_AF3_RNA = [
    "rnacentral-active-90-80",
    "rfam-14-9-90-80",
    "nt-rna-2023-02-23-90-80",
]

PRESETS: dict[str, list[str]] = {
    # Default of the `mmseqs2-homology-search` Config.
    "colabfold-default": ["uniref30-2302"],
    "colabfold-with-envdb": ["uniref30-2302", "colabfold-envdb-202108"],
    # AlphaFast / AF3 reference set
    "af3-protein": _AF3_PROTEIN,
    "af3-rna": _AF3_RNA,
    "af3-all": _AF3_PROTEIN + _AF3_RNA,
    # Lightning-Boltz `--mode alphafold3` (no PDB seqres, no RNA — Boltz-2
    # doesn't consume them)
    "boltz2-af3": ["uniref90-2022-05", "mgnify-2022-05", "small-bfd", "uniprot-2021-04"],
}


# ============================================================================
# CLI
# ============================================================================


def _list_available() -> None:
    """Print available datasets and presets."""
    print("Datasets (registered in proto_tools.databases):\n")
    print(f"  {'name':<32} {'type':<8} {'size (download / built)':<32}")
    print(f"  {'-' * 32} {'-' * 8} {'-' * 32}")
    for key in DatasetRegistry.list_all():
        entry = DatasetRegistry.get(key)
        size = f"~{entry.total_download_bytes / 1e9:.0f} GB / ~{entry.total_disk_bytes / 1e9:.0f} GB"
        print(f"  {key:<32} {entry.molecule_type:<8} {size:<32}")
    print("\nPresets:\n")
    for name, datasets in PRESETS.items():
        print(f"  {name}")
        for d in datasets:
            print(f"    - {d}")
    print(f"\nDefault cache root: {get_databases_root()}")


def _resolve_targets(args: argparse.Namespace) -> list[str]:
    """Combine positional dataset arguments and ``--preset`` into the final list."""
    targets: list[str] = []
    if args.preset:
        if args.preset not in PRESETS:
            raise SystemExit(f"Unknown preset {args.preset!r}. Available: {', '.join(PRESETS)}")
        targets.extend(PRESETS[args.preset])
    targets.extend(args.datasets)
    if not targets:
        # Default: smallest viable bootstrap (UniRef30 only)
        targets = PRESETS["colabfold-default"]
    # De-dup preserving order
    return list(dict.fromkeys(targets))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns process exit code."""
    parser = argparse.ArgumentParser(
        prog="setup_databases",
        description=__doc__.split("\n\n")[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset registry keys to provision (e.g. uniref30-2302). "
        "Defaults to 'colabfold-default' preset (uniref30-2302 only) when neither "
        "datasets nor --preset are given.",
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS),
        default=None,
        help="Named bundle to provision. Combines additively with positional datasets.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help=f"Override the databases root for this run. Datasets go in <workdir>/<slug>/ (default: {get_databases_root()}).",
    )
    parser.add_argument("--list", action="store_true", help="List available datasets and presets, then exit.")
    parser.add_argument(
        "--force", action="store_true", help="Re-download and re-run build steps even if a dataset looks provisioned."
    )
    args = parser.parse_args(argv)

    if args.list:
        _list_available()
        return 0

    # Provisioning reports progress through logging; show it on the terminal.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Toolkit-bundled mmseqs is at $PROTO_HOME/proto_tool_envs/mmseqs2_env/bin/mmseqs; building the env on first run (matches tests/dummy_data/create_mini_mmseqs_db.py) so recipes find `mmseqs` via PATH without caller setup.
    _ensure_mmseqs_on_path()

    targets = _resolve_targets(args)

    print(f"Will provision {len(targets)} dataset(s): {', '.join(targets)}")

    # Validate all names up front so we fail fast before downloading anything.
    registered = set(DatasetRegistry.list_all())
    unknown = [name for name in targets if name not in registered]
    if unknown:
        print(f"Error: unknown dataset(s) {unknown}. Registered: {sorted(registered)}", file=sys.stderr)
        return 2

    return _provision_each(targets, workdir=args.workdir, force=args.force)


def _ensure_mmseqs_on_path() -> None:
    """Build the mmseqs2 toolkit env if needed and prepend its bin to PATH.

    Idempotent. ``ToolInstance.ensure_ready()`` is a no-op when the env already
    exists. The first call builds it, which can take minutes. The bin prepend is process-local
    (doesn't leak beyond this script invocation) and only adds the toolkit's
    bin, so system utilities (``tar``, ``mv``, etc.) resolve unchanged.
    """
    tool = ToolInstance.get("mmseqs2")
    tool.ensure_ready()
    mmseqs_bin = tool.env_path / "bin" / "mmseqs"
    if not mmseqs_bin.exists():
        raise RuntimeError(
            f"mmseqs binary not found at {mmseqs_bin} after toolkit env setup; "
            "the mmseqs2 standalone env may have failed to build."
        )
    os.environ["PATH"] = f"{mmseqs_bin.parent}{os.pathsep}{os.environ.get('PATH', '')}"


def _provision_each(targets: list[str], *, workdir: Path | None, force: bool) -> int:
    """Provision each dataset, stopping with a clear error on the first failure."""
    for name in targets:
        try:
            provision(name, workdir=workdir, force=force)
        except subprocess.CalledProcessError as e:
            print(f"\nERROR: provisioning {name} failed at step: {e.cmd}", file=sys.stderr)
            return 1
        except RuntimeError as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            return 1

    print("\nAll datasets provisioned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
