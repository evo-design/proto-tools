"""Provision MMseqs2 databases for proto-tools.

User-invoked CLI; no auto-download anywhere else in the codebase. Reads
the registry entries in ``proto_tools.tools.sequence_alignment.databases``
to know what URLs to fetch, what indexing commands to run, and where on
disk each dataset belongs (``$PROTO_MODEL_CACHE/databases/<name>/`` by
default). Idempotent: re-runs skip already-provisioned datasets and
already-downloaded files.

Each registered ``DatasetEntry`` declares its own ``urls`` (download
specs) and ``index_recipe.steps`` (argv to run after download). This
script is a generic executor — adding a new dataset is a new entry
in ``databases/entries/``, no script changes needed.

Usage::

    # Default (no args): UniRef30 only, smallest viable bootstrap
    python -m proto_tools.tools.sequence_alignment.mmseqs2_homology_search.setup_databases

    # Specific datasets
    python -m ...setup_databases uniref30-2302 small-bfd

    # Named preset matching a predictor's preferred_datasets
    python -m ...setup_databases --preset af3-protein

    # List datasets / presets
    python -m ...setup_databases --list

    # Custom cache root (datasets still go in <root>/<dataset_name>/)
    python -m ...setup_databases --workdir /custom/path uniref30-2302

Prerequisites:
    The ``mmseqs`` binary must be on ``PATH`` (and ``zstd`` for any
    ``.zst``-distributed dataset). The mmseqs2-homology-search tool's
    micromamba env ships an ``mmseqs`` at
    ``$PROTO_HOME/proto_tool_envs/mmseqs2_homology_search_env/bin/`` —
    after the tool has been dispatched once that env is created and you
    can append it to PATH for this script. Resolving the binary
    automatically from the tool env is a planned follow-up.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from proto_tools.tools.sequence_alignment.databases import (
    DatasetEntry,
    DatasetRegistry,
    DownloadSpec,
    IndexStep,
    dataset_slug,
    get_databases_root,
    get_dataset_dir,
)

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
    # Default of the legacy `colabfold-search` setup_databases.sh and the
    # current `mmseqs2-homology-search` Config default.
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
# Provisioning
# ============================================================================


def _resolve_cache_dir(name: str, workdir: Path | None) -> Path:
    """Return where ``name`` should be installed.

    Without ``--workdir``, uses the registry's standard
    ``$PROTO_MODEL_CACHE/databases/<slug>/``. With ``--workdir``, places
    each dataset under ``<workdir>/<slug>/`` so multiple datasets stay
    isolated even with a custom root.
    """
    if workdir is None:
        return get_dataset_dir(name)
    return workdir / dataset_slug(name)


def _which_downloader() -> tuple[str, list[str]]:
    """Pick the best available HTTP downloader: aria2c (parallel) > curl > wget.

    Returns ``(name, base_argv)`` where ``base_argv`` is the static prefix
    to which we append output / URL.
    """
    if shutil.which("aria2c"):
        return "aria2c", [
            "aria2c",
            "--max-connection-per-server=8",
            "--allow-overwrite=true",
            "--auto-file-renaming=false",
        ]
    if shutil.which("curl"):
        return "curl", ["curl", "-L", "--fail"]
    if shutil.which("wget"):
        return "wget", ["wget", "--no-verbose"]
    raise RuntimeError("No download tool found in PATH. Install aria2c, curl, or wget.")


def _download_file(spec: DownloadSpec, cache_dir: Path, downloader: tuple[str, list[str]]) -> bool:
    """Download a single ``DownloadSpec`` into ``cache_dir``. Returns True on success.

    Idempotent: skips if the target file already exists. ``downloader`` is the
    ``(name, base_argv)`` tuple resolved once per provisioning run by the
    caller (so we don't re-``shutil.which`` on every file).
    """
    target = cache_dir / spec.filename
    if target.exists():
        print(f"  [skip] {spec.filename} already present")
        return True

    name, base_argv = downloader
    print(f"  [download] {spec.filename} via {name}: {spec.url}")

    if name == "aria2c":
        cmd = [*base_argv, "-o", spec.filename, "-d", str(cache_dir), spec.url]
    elif name == "curl":
        cmd = [*base_argv, "-o", str(target), spec.url]
    else:  # wget
        cmd = [*base_argv, "-O", str(target), spec.url]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        if not spec.required:
            print(f"  [warn] optional file {spec.filename} download failed (continuing): {e}")
            return True
        raise
    # TODO: verify spec.sha256 when set — currently every registered entry
    # passes sha256=None, so this branch is unreachable. Wire up hashlib
    # comparison the first time we register an entry with a populated checksum.
    return True


def _run_step(step: IndexStep, cache_dir: Path, name: str) -> None:
    """Run a single ``IndexStep`` argv with ``cwd=cache_dir``.

    Substitutes ``{name}`` in any argv element with the dataset's registry
    name (snake-cased to match filesystem conventions).
    """
    cmd = [arg.replace("{name}", dataset_slug(name)) for arg in step.command]
    print(f"  [step] {step.description}")
    print(f"          $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cache_dir, check=True)


def _is_provisioned(entry: DatasetEntry, cache_dir: Path) -> bool:
    """A dataset is considered provisioned when every declared output file is present."""
    if not entry.index_recipe.output_files:
        # No declared outputs — fall back to the .dbtype check
        return (cache_dir / f"{entry.db_prefix}.dbtype").is_file()
    for output_file in entry.index_recipe.output_files:
        path = cache_dir / output_file.replace("{name}", dataset_slug(entry.name))
        # Allow glob patterns in output_files for consistency with mmseqs filename conventions
        if "*" in output_file:
            if not list(cache_dir.glob(output_file.replace("{name}", dataset_slug(entry.name)))):
                return False
        elif not path.exists():
            return False
    return True


def provision(name: str, workdir: Path | None = None, *, force: bool = False) -> Path:
    """Provision a single dataset by registry name.

    Args:
        name (str): Registered dataset key (e.g. ``"uniref30-2302"``).
        workdir (Path | None): Override the standard cache root. When set,
            the dataset goes to ``<workdir>/<slug>/`` (still isolated per dataset).
        force (bool): Re-download / re-run index steps even if the dataset
            looks provisioned.

    Returns:
        Path: The dataset's cache directory.
    """
    entry = DatasetRegistry.get(name)
    cache_dir = _resolve_cache_dir(name, workdir)

    print(f"\n=== {name} ({entry.display_name}) ===")
    print(f"  type:    {entry.molecule_type}")
    print(f"  cache:   {cache_dir}")
    print(
        f"  size:    ~{entry.total_download_bytes / 1e9:.1f} GB download / ~{entry.total_disk_bytes / 1e9:.1f} GB indexed"
    )

    if not force and _is_provisioned(entry, cache_dir):
        print("  [skip] already provisioned (use --force to re-run)")
        return cache_dir

    # --force: wipe the cache dir before re-running so stale outputs don't
    # collide with this run (e.g. ``zstd -d`` prompting on an existing FASTA,
    # or ``mmseqs createdb`` refusing to overwrite a partial DB).
    if force and cache_dir.exists():
        print("  [force] wiping existing cache dir before re-provisioning")
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("\n  Downloading sources...")
    downloader = _which_downloader()
    for spec in entry.urls:
        _download_file(spec, cache_dir, downloader)

    print("\n  Running index recipe...")
    for step in entry.index_recipe.steps:
        _run_step(step, cache_dir, name)

    print(f"\n  ✓ {name} ready at {cache_dir}")
    return cache_dir


# ============================================================================
# CLI
# ============================================================================


def _list_available() -> None:
    """Print available datasets and presets."""
    print("Datasets (registered in proto_tools.tools.sequence_alignment.databases):\n")
    print(f"  {'name':<32} {'type':<8} {'size (download / indexed)':<32}")
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
    """Combine positional dataset args + ``--preset`` into the final list."""
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
    seen: set[str] = set()
    unique: list[str] = []
    for name in targets:
        if name not in seen:
            unique.append(name)
            seen.add(name)
    return unique


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
        help=f"Override the cache root. Datasets still go in <workdir>/<slug>/ (default: {get_databases_root()}).",
    )
    parser.add_argument("--list", action="store_true", help="List available datasets and presets, then exit.")
    parser.add_argument(
        "--force", action="store_true", help="Re-download and re-run index steps even if a dataset looks provisioned."
    )
    args = parser.parse_args(argv)

    if args.list:
        _list_available()
        return 0

    targets = _resolve_targets(args)

    print(f"Will provision {len(targets)} dataset(s): {', '.join(targets)}")

    # Validate all names up front so we fail fast before downloading anything.
    registered = set(DatasetRegistry.list_all())
    unknown = [name for name in targets if name not in registered]
    if unknown:
        print(f"Error: unknown dataset(s) {unknown}. Registered: {sorted(registered)}", file=sys.stderr)
        return 2

    return _provision_each(targets, workdir=args.workdir, force=args.force)


def _provision_each(targets: list[str], *, workdir: Path | None, force: bool) -> int:
    """Provision each dataset; abort on first failure with a clear error."""
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
