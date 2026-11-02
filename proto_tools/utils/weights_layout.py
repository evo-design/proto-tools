"""proto_tools/utils/weights_layout.py.

Per-toolkit layout of the model cache, and migration from the old flat layout.

Every framework cache a toolkit writes to lands in one folder named by its weights key::

    {PROTO_MODEL_CACHE}/{key}/
    ├── huggingface/   HF_HOME
    ├── torch/         TORCH_HOME
    ├── jax_cache/     JAX_COMPILATION_CACHE_DIR
    └── weights/       resolve_weights_dir()

The weights key is the toolkit name unless ``standalone/weights_key.txt`` names another
toolkit's folder, so toolkits that load the same checkpoints share one copy.

The old layout put ``huggingface/``, ``torch/``, and ``jax_cache/{toolkit}/`` at the cache
root, and ``resolve_weights_dir(name)`` files directly in ``{name}/``. Each migrated toolkit
lists what it owns in ``MIGRATIONS``; the first worker a process starts moves every migrated
toolkit's paths into the new layout, so a toolkit that reads another's weights (bindcraft
reading alphafold2's) never finds them half-moved. ``scripts/migrate_proto_home.py`` runs the
same moves by hand. ``IN_ENV`` and ``NONE`` modes are not affected.
"""

import contextlib
import fcntl
import logging
import os
import shutil
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Subfolders of a toolkit folder that belong to the new layout.
LAYOUT_SUBDIRS = ("huggingface", "torch", "jax_cache", "weights")

# Set in the subprocess env to the comma-separated migrated toolkits, so the standalone weights
# helpers resolve {name}/weights for exactly those names, whichever toolkit is asking.
MIGRATED_ENV_VAR = "PROTO_WEIGHTS_MIGRATED"

# Toolkits switched to the per-toolkit layout. Transitional: toolkits are added one at a time
# while their downloads are audited, and the gate is removed once every toolkit is migrated.
MIGRATED_TOOLKITS: frozenset[str] = frozenset(
    {
        "ablang",
        "alphafold2",
        "alphafold3",
        "alphagenome",
        "bindcraft",
        "bioemu",
        "blast",
        "boltz2",
        "borzoi",
        "caliby",
        "ccd_lookup",
        "chai1",
        "cherimoya",
        "codonfm",
        "crispr_tracr_rna",
        "deeppbs_specificity",
        "dssp",
        "enformer",
        "esm2",
        "esm3",
        "esm_if1",
        "esmc",
        "esmfold",
        "esmfold2",
        "evo1",
        "evo2",
        "fampnn",
        "foldmason",
        "foldseek",
        "freebindcraft",
        "germinal",
        "glm2",
        "iglm",
        "ipsae",
        "ligandmpnn",
        "mafft",
        "malinois",
        "meme",
        "metal3d",
        "minced",
        "minerva",
        "miranda",
        "mmseqs2",
        "mock_cli_multi_gpu_tool",
        "mock_cli_tool",
        "mock_cpu_tool",
        "mock_jax_multi_gpu_tool",
        "mock_jax_tool",
        "mock_pytorch_multi_gpu_tool",
        "mock_pytorch_tool",
        "na_mpnn_specificity",
        "netmhcpan",
        "opendde",
        "orfipy",
        "pangolin",
        "parade",
        "primer3",
        "prodigal",
        "progen2",
        "progen3",
        "promoter_calculator",
        "proteinmpnn",
        "protenix",
        "puffin",
        "pyhmmer",
        "pymol_rmsd",
        "pyrosetta",
        "rf3",
        "rfd4",
        "rfdiffusion3",
        "segmasker",
        "splice_transformer",
        "spliceai",
        "spliceai2",
        "tmalign",
        "usalign",
        "viennarna",
        "vina",
        "x3dna",
    }
)


@dataclass(frozen=True)
class HFRepo:
    """A Hugging Face repo in the old shared ``huggingface/hub/`` cache.

    Attributes:
        repo_id (str): Repo id, e.g. ``"facebook/esm2_t6_8M_UR50D"``.
        repo_type (str): ``"model"``, ``"dataset"``, or ``"space"``.
    """

    repo_id: str
    repo_type: str = "model"

    @property
    def cache_dirname(self) -> str:
        """Return the hub cache folder name, e.g. ``models--facebook--esm2_t6_8M_UR50D``."""
        return f"{self.repo_type}s--{self.repo_id.replace('/', '--')}"


@dataclass(frozen=True)
class WeightsDir:
    """A ``resolve_weights_dir(name)`` folder whose files move from ``{name}/`` into ``{name}/weights/``.

    Attributes:
        name (str): The name passed to ``resolve_weights_dir``.
    """

    name: str


@dataclass(frozen=True)
class TorchHubFile:
    """A file in the old shared ``torch/`` cache, relative to it, e.g. ``hub/checkpoints/x.pt``.

    Attributes:
        relpath (str): Path relative to the old ``TORCH_HOME``.
    """

    relpath: str


@dataclass(frozen=True)
class HFRepoPattern:
    """Hub repos in the old shared cache whose folder names match a glob.

    For repo families too large to list, e.g. ``"models--biohub--ESMC-*-sae-*"``.

    Attributes:
        pattern (str): Glob matched against ``huggingface/hub/`` folder names.
    """

    pattern: str


@dataclass(frozen=True)
class HFHomePath:
    """A non-repo path a library keeps under the old shared ``HF_HOME``, e.g. ``"evo2_vllm"``.

    Attributes:
        relpath (str): Path or glob relative to the old ``HF_HOME``.
    """

    relpath: str


@dataclass(frozen=True)
class CachePath:
    """An explicit move between two paths relative to the cache root.

    For files a toolkit kept in its own folder that now live in a shared one, e.g. germinal's
    private AF2 params moving to ``alphafold2/weights/params``.

    Attributes:
        old (str): Old path relative to the cache root.
        new (str): New path relative to the cache root.
    """

    old: str
    new: str


MigrationEntry = HFRepo | HFRepoPattern | HFHomePath | WeightsDir | TorchHubFile | CachePath

# What each migrated toolkit downloaded under the old layout. An entry is added in the same
# commit that adds the toolkit to MIGRATED_TOOLKITS. The old jax_cache/{toolkit}/ moves for
# every migrated toolkit without an entry.
MIGRATIONS: dict[str, tuple[MigrationEntry, ...]] = {
    "alphafold3": (WeightsDir("alphafold3"),),
    "bioemu": (HFRepo("microsoft/bioemu"),),
    "alphagenome": (
        HFRepo("google/alphagenome-all-folds"),
        WeightsDir("alphagenome"),
    ),
    "netmhcpan": (WeightsDir("netmhcpan"),),
    "x3dna": (WeightsDir("x3dna"),),
    "evo2": (
        *(
            HFRepo(f"gbrixi/evo2_{name}_vllm")
            for name in ("20b", "7b", "7b_base", "7b_262k", "1b_base", "7b_microviridae")
        ),
        *(
            HFRepo(f"arcinstitute/evo2_{name}")
            for name in ("40b", "7b", "20b", "40b_base", "7b_base", "1b_base", "7b_262k")
        ),
        HFRepo("evo-design/evo-2-7b-8k-microviridae"),
        HFHomePath("evo2_vllm"),
        HFHomePath("evo2_*.pt"),
    ),
    "deeppbs_specificity": (WeightsDir("deeppbs_specificity"),),
    "spliceai2": (WeightsDir("spliceai2"),),
    "evo1": (
        HFRepo("evo-design/evo-1.5-8k-base"),
        HFRepo("togethercomputer/evo-1-8k-base"),
        HFRepo("togethercomputer/evo-1-131k-base"),
        HFRepo("LongSafari/evo-1-8k-crispr"),
        HFRepo("LongSafari/evo-1-8k-transposon"),
    ),
    "germinal": (
        CachePath("germinal/params", "alphafold2/weights/params"),
        WeightsDir("germinal"),
    ),
    "alphafold2": (WeightsDir("alphafold2"),),
    "ablang": (WeightsDir("ablang"),),
    "foldseek": (WeightsDir("foldseek"),),
    "boltz2": (WeightsDir("boltz2"),),
    "chai1": (WeightsDir("chai1"),),
    "opendde": (WeightsDir("opendde"), HFRepo("aurekaresearch/OpenDDE")),
    "protenix": (WeightsDir("protenix"),),
    "rfd4": (HFRepo("nvidia/RFD4-Proteina"),),
    "esmc": (
        *(HFRepo(f"biohub/ESMC-{size}") for size in ("300M", "600M", "6B")),
        HFRepoPattern("models--biohub--ESMC-*-sae-*"),
        HFRepoPattern("models--biohub--esmc-*-2024-12"),
    ),
    "esm3": (HFRepo("biohub/esm3-sm-open-v1"),),
    "esmfold2": (
        HFRepo("biohub/ESMFold2"),
        HFRepo("biohub/ESMFold2-Fast"),
    ),
    "esmfold": (HFRepo("facebook/esmfold_v1"),),
    "na_mpnn_specificity": (WeightsDir("na_mpnn_specificity"),),
    "puffin": (WeightsDir("puffin"),),
    "rfdiffusion3": (WeightsDir("rfdiffusion3"),),
    "rf3": (WeightsDir("rf3"),),
    "ccd_lookup": (WeightsDir("ccd_lookup"),),
    "caliby": (WeightsDir("caliby"),),
    "malinois": (WeightsDir("malinois"),),
    "parade": (WeightsDir("parade"),),
    "codonfm": (WeightsDir("codonfm"),),
    "borzoi": (
        *(
            HFRepo(f"johahi/{name}-replicate-{rep}{suffix}")
            for name, suffix in (("flashzoi", ""), ("borzoi", ""), ("borzoi", "-mouse"))
            for rep in range(4)
        ),
    ),
    "cherimoya": (HFRepo("programmable-genomics/CATv1"),),
    "splice_transformer": (HFRepo("brianhie/SpTransformer"),),
    "enformer": (HFRepo("EleutherAI/enformer-official-rough"),),
    "minerva": (
        HFRepo("gbrixi/minerva-mlm"),
        HFRepo("gbrixi/minerva-mlm-8k"),
    ),
    "glm2": (
        HFRepo("tattabio/gLM2_150M"),
        HFRepo("tattabio/gLM2_650M"),
    ),
    "progen3": (*(HFRepo(f"Profluent-Bio/progen3-{size}") for size in ("112m", "219m", "339m", "762m", "1b", "3b")),),
    "progen2": (
        *(
            HFRepo(f"hugohrban/progen2-{size}")
            for size in ("small", "medium", "base", "oas", "large", "BFD90", "xlarge")
        ),
    ),
    "metal3d": (WeightsDir("metal3d"),),
    "esm_if1": (WeightsDir("esm_if1"),),
    "esm2": (
        *(
            HFRepo(f"fredzzp/{ckpt}")
            for ckpt in (
                "esm2_t6_8M_UR50D",
                "esm2_t12_35M_UR50D",
                "esm2_t30_150M_UR50D",
                "esm2_t33_650M_UR50D",
                "esm2_t36_3B_UR50D",
            )
        ),
        HFRepo("facebook/esm2_t48_15B_UR50D"),
    ),
    "fampnn": (WeightsDir("fampnn"),),
    "ligandmpnn": (WeightsDir("ligandmpnn"),),
}


def uses_toolkit_layout(toolkit: str) -> bool:
    """Return whether ``toolkit`` stores its weights in the per-toolkit layout.

    Args:
        toolkit (str): Toolkit name.

    Returns:
        bool: ``True`` when the toolkit has been migrated.
    """
    return toolkit in MIGRATED_TOOLKITS


def weights_key(toolkit: str) -> str:
    """Return the model-cache folder name for ``toolkit``.

    Args:
        toolkit (str): Toolkit name.

    Returns:
        str: The name in ``standalone/weights_key.txt``, or ``toolkit`` when there is none.

    Raises:
        ValueError: If ``weights_key.txt`` exists but is empty.
    """
    # Lazy import to avoid circular dep: tool_instance imports persistent_worker, which imports this
    from proto_tools.utils.tool_instance import ToolInstance

    tool_dir = ToolInstance._find_tool_dir(toolkit)
    marker = tool_dir / "standalone" / "weights_key.txt" if tool_dir is not None else None
    if marker is None or not marker.is_file():
        return toolkit
    key = marker.read_text().strip()
    if not key:
        raise ValueError(f"weights_key.txt for toolkit {toolkit!r} is empty ({marker})")
    return key


def migrated_names_env() -> dict[str, str]:
    """Return the env var that tells ``resolve_weights_dir`` which names use ``{name}/weights``.

    Returns:
        dict[str, str]: ``PROTO_WEIGHTS_MIGRATED`` set to the comma-separated migrated toolkits.
    """
    return {MIGRATED_ENV_VAR: ",".join(sorted(MIGRATED_TOOLKITS))}


def toolkit_cache_env(toolkit: str, cache_root: Path) -> dict[str, str]:
    """Return the env vars that point the HF and torch caches at ``toolkit``'s folder.

    The JAX compilation cache is resolved separately because a user-set value and
    ``PROTO_JAX_COMPILATION_CACHE=0`` both take precedence over it.

    Args:
        toolkit (str): Toolkit name.
        cache_root (Path): The model cache root (``PROTO_MODEL_CACHE`` path mode).

    Returns:
        dict[str, str]: ``HF_HOME`` and ``TORCH_HOME``.
    """
    folder = cache_root / weights_key(toolkit)
    return {
        "HF_HOME": str(folder / "huggingface"),
        "TORCH_HOME": str(folder / "torch"),
    }


@contextlib.contextmanager
def _migration_lock(cache_root: Path) -> Iterator[None]:
    """Hold an exclusive lock on the model cache so concurrent workers don't move the same paths."""
    cache_root.mkdir(parents=True, exist_ok=True)
    with open(cache_root / ".migration.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _planned_moves(toolkit: str, cache_root: Path) -> list[tuple[Path, Path]]:
    """Return the (old, new) path pairs for ``toolkit``, before checking what exists on disk."""
    folder = cache_root / weights_key(toolkit)
    moves = [(cache_root / "jax_cache" / toolkit, folder / "jax_cache")]
    for entry in MIGRATIONS.get(toolkit, ()):
        if isinstance(entry, HFRepo):
            hub = Path("huggingface") / "hub" / entry.cache_dirname
            moves.append((cache_root / hub, folder / hub))
        elif isinstance(entry, HFRepoPattern):
            old_hub = cache_root / "huggingface" / "hub"
            moves.extend(
                (repo, folder / "huggingface" / "hub" / repo.name) for repo in sorted(old_hub.glob(entry.pattern))
            )
        elif isinstance(entry, HFHomePath):
            old_home = cache_root / "huggingface"
            moves.extend(
                (path, folder / "huggingface" / path.relative_to(old_home))
                for path in sorted(old_home.glob(entry.relpath))
            )
        elif isinstance(entry, TorchHubFile):
            moves.append((cache_root / "torch" / entry.relpath, folder / "torch" / entry.relpath))
        elif isinstance(entry, CachePath):
            moves.append((cache_root / entry.old, cache_root / entry.new))
        else:
            # Old files sit beside the new layout subfolders in the same {name}/ folder
            old_dir = cache_root / entry.name
            if old_dir.is_dir():
                moves.extend(
                    (child, old_dir / "weights" / child.name)
                    for child in sorted(old_dir.iterdir())
                    if child.name not in LAYOUT_SUBDIRS
                )
    return moves


def _materialize_shared_blobs(repo_dir: Path) -> None:
    """Replace a hub repo's links into the cache-wide shared blob store with regular files.

    ``huggingface_hub`` >= 1.x stores Xet files once at ``hub/blobs/<xx>/<hash>`` and links each
    repo's ``blobs/<etag>`` to it with a relative symlink, which breaks when the repo moves to
    another hub. Each payload is hard-linked into the repo (copied across filesystems), giving
    the classic per-repo layout every ``huggingface_hub`` version reads, and the repo is dropped
    from the store's ``.refs`` manifest; a store entry with no references left is deleted.

    Args:
        repo_dir (Path): A ``models--org--name`` folder inside a hub cache.
    """
    blobs = repo_dir / "blobs"
    if not blobs.is_dir():
        return
    for link in sorted(blobs.iterdir()):
        if not link.is_symlink():
            continue
        payload = (blobs / os.readlink(link)).resolve()
        if not payload.is_file():
            continue
        tmp = link.with_name(f"{link.name}.migrating")
        try:
            os.link(payload, tmp)
        except OSError:
            shutil.copy2(payload, tmp)
        os.replace(tmp, link)
        manifest = payload.with_name(f"{payload.name}.refs")
        ref = f"{repo_dir.name}/blobs/{link.name}"
        refs = [r for r in manifest.read_text().splitlines() if r.strip()] if manifest.is_file() else [ref]
        remaining = [r for r in refs if r != ref]
        if remaining:
            manifest.write_text("".join(f"{r}\n" for r in remaining))
        else:
            for stale in (payload, manifest, payload.with_name(f"{payload.name}.lock")):
                stale.unlink(missing_ok=True)


def migrate_toolkit(toolkit: str, cache_root: Path, *, dry_run: bool = False) -> list[tuple[Path, Path]]:
    """Move ``toolkit``'s weights from the old flat layout into its per-toolkit folder.

    Safe to repeat: a path is moved only when it exists at the old location and not at the new
    one. When both exist the old path is left alone and a warning is logged.

    Args:
        toolkit (str): Toolkit name.
        cache_root (Path): The model cache root.
        dry_run (bool): Report the moves without touching the filesystem.

    Returns:
        list[tuple[Path, Path]]: The (old, new) pairs moved, or that would be moved in a dry run.
    """
    with contextlib.ExitStack() as stack:
        if not dry_run:
            stack.enter_context(_migration_lock(cache_root))
        moved = []
        for old, new in _planned_moves(toolkit, cache_root):
            if not old.exists() and not old.is_symlink():
                continue
            if old.is_dir() and not old.is_symlink() and not any(old.iterdir()):
                # An empty folder an old setup.sh created carries nothing to move
                if not dry_run:
                    old.rmdir()
                continue
            if new.exists():
                logger.warning("%s: not migrating %s, %s already exists; remove one of them", toolkit, old, new)
                continue
            if not dry_run:
                if old.parent.name == "hub":
                    _materialize_shared_blobs(old)
                new.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old), str(new))
                logger.info("%s: migrated %s -> %s", toolkit, old, new)
            moved.append((old, new))
        return moved


_migrated_roots: set[str] = set()
_migrated_lock = threading.Lock()


def migrate_all_once(cache_root: Path) -> None:
    """Migrate every toolkit in ``MIGRATED_TOOLKITS`` at most once per cache root in this process.

    Run before any worker starts, not only the toolkit's own, because a toolkit may read another
    toolkit's weights folder. Each toolkit costs a few ``stat`` calls when nothing is left to move.

    Args:
        cache_root (Path): The model cache root.
    """
    with _migrated_lock:
        if str(cache_root) in _migrated_roots:
            return
        _migrated_roots.add(str(cache_root))
    for toolkit in sorted(MIGRATED_TOOLKITS):
        try:
            migrate_toolkit(toolkit, cache_root)
        except OSError as exc:  # noqa: PERF203 -- one toolkit's failure must not stop the rest
            # A shared cache may be read-only for this user; the tool then downloads to the new layout
            logger.warning("%s: weights migration skipped (%s); old weights stay in place", toolkit, exc)


def default_cache_root() -> Path | None:
    """Return the model cache root for the current env, or ``None`` in ``IN_ENV``/``NONE`` mode.

    Returns:
        Path | None: ``PROTO_MODEL_CACHE`` when it is a path, else ``{PROTO_HOME}/proto_model_cache``.
    """
    from proto_tools.utils.proto_home import get_proto_home

    mode = os.environ.get("PROTO_MODEL_CACHE") or str(get_proto_home() / "proto_model_cache")
    if mode in ("IN_ENV", "NONE"):
        return None
    return Path(mode)
