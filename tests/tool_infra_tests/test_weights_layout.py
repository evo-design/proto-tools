"""Tests for the per-toolkit model-cache layout and its migration from the flat layout."""

import logging
import os
import subprocess
from pathlib import Path

import pytest

from proto_tools.utils import weights_layout
from proto_tools.utils.persistent_worker import _build_subprocess_env
from proto_tools.utils.standalone_helpers_source.standalone_helpers import resolve_weights_dir
from proto_tools.utils.tool_instance import ToolInstance
from proto_tools.utils.weights_layout import (
    HFHomePath,
    HFRepo,
    HFRepoPattern,
    TorchHubFile,
    WeightsDir,
    migrate_toolkit,
    weights_key,
)

_HELPERS_SH = Path(weights_layout.__file__).parent / "standalone_helpers_source" / "standalone_helpers.sh"


@pytest.fixture
def migrated(monkeypatch):
    """Put ``mocktool`` on the per-toolkit layout with the given migration entries."""

    def _migrate(*entries, toolkit: str = "mocktool"):
        monkeypatch.setattr(weights_layout, "MIGRATED_TOOLKITS", frozenset({toolkit}))
        monkeypatch.setattr(weights_layout, "MIGRATIONS", {toolkit: entries})

    return _migrate


@pytest.fixture
def key_file(monkeypatch, tmp_path: Path):
    """Give ``mocktool`` a toolkit dir whose ``standalone/weights_key.txt`` holds the given text."""

    def _write(text: str) -> None:
        tool_dir = tmp_path / "tool"
        (tool_dir / "standalone").mkdir(parents=True)
        (tool_dir / "standalone" / "weights_key.txt").write_text(text)
        monkeypatch.setattr(ToolInstance, "_find_tool_dir", classmethod(lambda cls, name: tool_dir))

    return _write


def _touch(path: Path, text: str = "w") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# ── weights_key ──────────────────────────────────────────────────────────────


def test_weights_key_reads_shared_folder(key_file):
    """weights_key.txt points a toolkit at another toolkit's folder."""
    key_file("alphafold2\n")
    assert weights_key("mocktool") == "alphafold2"


def test_weights_key_empty_file_raises(key_file):
    """An empty weights_key.txt is a mistake, not a silent fallback to the toolkit name."""
    key_file("  \n")
    with pytest.raises(ValueError, match="empty"):
        weights_key("mocktool")


# ── subprocess env ───────────────────────────────────────────────────────────


def test_migrated_toolkit_caches_live_in_its_folder(monkeypatch, tmp_path: Path, migrated, key_file):
    """HF, torch, and JAX caches all resolve under {cache}/{weights_key}/ for a migrated toolkit."""
    migrated()
    key_file("shared")
    monkeypatch.setenv("PROTO_MODEL_CACHE", str(tmp_path))
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("PROTO_JAX_COMPILATION_CACHE", raising=False)

    env = _build_subprocess_env(device="cpu", tool_env_path=tmp_path / "venv", toolkit="mocktool")

    folder = tmp_path / "shared"
    assert env["HF_HOME"] == str(folder / "huggingface")
    assert env["TORCH_HOME"] == str(folder / "torch")
    assert env["JAX_COMPILATION_CACHE_DIR"] == str(folder / "jax_cache")
    assert env["PROTO_WEIGHTS_MIGRATED"] == "mocktool"


def test_unmigrated_toolkit_keeps_flat_layout(monkeypatch, tmp_path: Path, migrated):
    """Toolkits not yet migrated keep the shared HF cache, but still learn which names are migrated."""
    migrated(toolkit="othertool")
    monkeypatch.setenv("PROTO_MODEL_CACHE", str(tmp_path))

    env = _build_subprocess_env(device="cpu", tool_env_path=tmp_path / "venv", toolkit="mocktool")

    assert env["HF_HOME"] == str(tmp_path / "huggingface")
    assert env["PROTO_WEIGHTS_MIGRATED"] == "othertool"


@pytest.mark.parametrize("mode", ["IN_ENV", "NONE"])
def test_in_env_and_none_modes_ignore_layout(monkeypatch, tmp_path: Path, migrated, mode):
    """IN_ENV and NONE modes behave exactly as before, migrated or not."""
    migrated()
    monkeypatch.setenv("PROTO_MODEL_CACHE", mode)
    monkeypatch.delenv("HF_HOME", raising=False)

    env = _build_subprocess_env(device="cpu", tool_env_path=tmp_path / "venv", toolkit="mocktool")

    assert "PROTO_WEIGHTS_MIGRATED" not in env
    expected_hf = str(tmp_path / "venv" / "cache" / "huggingface") if mode == "IN_ENV" else None
    assert env.get("HF_HOME") == expected_hf


def test_user_jax_cache_beats_toolkit_layout(monkeypatch, tmp_path: Path, migrated):
    """A user-set JAX_COMPILATION_CACHE_DIR still wins for a migrated toolkit."""
    migrated()
    monkeypatch.setenv("PROTO_MODEL_CACHE", str(tmp_path))
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/custom/jax")

    env = _build_subprocess_env(device="cpu", toolkit="mocktool")

    assert env["JAX_COMPILATION_CACHE_DIR"] == "/custom/jax"


def test_worker_env_migrates_old_weights(monkeypatch, tmp_path: Path, migrated):
    """Building a migrated toolkit's env moves its old weights before the subprocess starts."""
    migrated(HFRepo("org/model"))
    monkeypatch.setattr(weights_layout, "_migrated_roots", set())
    monkeypatch.setenv("PROTO_MODEL_CACHE", str(tmp_path))
    _touch(tmp_path / "huggingface" / "hub" / "models--org--model" / "config.json")

    _build_subprocess_env(device="cpu", toolkit="mocktool")

    assert (tmp_path / "mocktool" / "huggingface" / "hub" / "models--org--model" / "config.json").is_file()


# ── resolve_weights_dir ──────────────────────────────────────────────────────


def _shell_weights_dir(env: dict[str, str], name: str) -> str:
    script = f'source "{_HELPERS_SH}" && proto_resolve_weights_dir {name} && printf %s "$WEIGHTS_DIR"'
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, check=True).stdout


@pytest.mark.parametrize("explicit_cache", [True, False])
def test_resolve_weights_dir_toolkit_layout(monkeypatch, tmp_path: Path, explicit_cache):
    """Under the layout marker, Python and shell helpers both resolve {cache}/{name}/weights."""
    monkeypatch.setenv("PROTO_HOME", str(tmp_path / "home"))
    if explicit_cache:
        monkeypatch.setenv("PROTO_MODEL_CACHE", str(tmp_path / "cache"))
        cache = tmp_path / "cache"
    else:
        monkeypatch.delenv("PROTO_MODEL_CACHE", raising=False)
        cache = tmp_path / "home" / "proto_model_cache"
    monkeypatch.setenv("PROTO_WEIGHTS_MIGRATED", "esm2,fampnn")
    monkeypatch.delenv("PROTO_FAMPNN_WEIGHTS_DIR", raising=False)

    expected = str(cache / "fampnn" / "weights")
    assert resolve_weights_dir("fampnn") == expected
    assert _shell_weights_dir(dict(os.environ), "fampnn") == expected


def test_resolve_weights_dir_follows_requested_name(monkeypatch, tmp_path: Path):
    """The weights/ subfolder depends on the name asked for, so bindcraft reading alphafold2 finds it."""
    monkeypatch.setenv("PROTO_MODEL_CACHE", str(tmp_path))
    monkeypatch.setenv("PROTO_WEIGHTS_MIGRATED", "alphafold2")
    for name in ("alphafold2", "bindcraft"):
        monkeypatch.delenv(f"PROTO_{name.upper()}_WEIGHTS_DIR", raising=False)
    env = dict(os.environ)

    assert resolve_weights_dir("alphafold2") == str(tmp_path / "alphafold2" / "weights")
    assert resolve_weights_dir("bindcraft") == str(tmp_path / "bindcraft")
    assert _shell_weights_dir(env, "alphafold2") == str(tmp_path / "alphafold2" / "weights")
    # A name that merely contains a migrated name is not migrated
    assert _shell_weights_dir(env, "alphafold") == str(tmp_path / "alphafold")


def test_unmigrated_worker_migrates_other_toolkits(monkeypatch, tmp_path: Path, migrated):
    """Any worker start migrates every migrated toolkit, so a reader of another's folder finds it moved."""
    migrated(WeightsDir("mocktool"))
    monkeypatch.setattr(weights_layout, "_migrated_roots", set())
    monkeypatch.setenv("PROTO_MODEL_CACHE", str(tmp_path))
    _touch(tmp_path / "mocktool" / "params" / "model.npz")

    _build_subprocess_env(device="cpu", toolkit="othertool")

    assert (tmp_path / "mocktool" / "weights" / "params" / "model.npz").is_file()


def test_resolve_weights_dir_override_ignores_layout(monkeypatch, tmp_path: Path):
    """PROTO_{TOOL}_WEIGHTS_DIR is used as given, with no weights/ appended."""
    monkeypatch.setenv("PROTO_WEIGHTS_MIGRATED", "fampnn")
    monkeypatch.setenv("PROTO_FAMPNN_WEIGHTS_DIR", str(tmp_path / "mine"))

    assert resolve_weights_dir("fampnn") == str(tmp_path / "mine")
    assert _shell_weights_dir(dict(os.environ), "fampnn") == str(tmp_path / "mine")


# ── migrate_toolkit ──────────────────────────────────────────────────────────


def test_migrate_moves_every_entry_kind(tmp_path: Path, migrated):
    """HF repos, torch files, resolve_weights_dir files, and the JAX cache all land in the toolkit folder."""
    migrated(HFRepo("org/model"), TorchHubFile("hub/checkpoints/m.pt"), WeightsDir("mocktool"))
    _touch(tmp_path / "huggingface" / "hub" / "models--org--model" / "snapshots" / "abc" / "w.bin")
    _touch(tmp_path / "huggingface" / "hub" / "models--org--other" / "w.bin")
    _touch(tmp_path / "torch" / "hub" / "checkpoints" / "m.pt")
    _touch(tmp_path / "mocktool" / "model.pt")
    _touch(tmp_path / "jax_cache" / "mocktool" / "entry")

    migrate_toolkit("mocktool", tmp_path)

    folder = tmp_path / "mocktool"
    assert (folder / "huggingface" / "hub" / "models--org--model" / "snapshots" / "abc" / "w.bin").is_file()
    assert (folder / "torch" / "hub" / "checkpoints" / "m.pt").is_file()
    assert (folder / "weights" / "model.pt").is_file()
    assert (folder / "jax_cache" / "entry").is_file()
    assert not (folder / "model.pt").exists()
    # Repos the toolkit does not list stay in the shared cache
    assert (tmp_path / "huggingface" / "hub" / "models--org--other" / "w.bin").is_file()


def test_migrate_is_idempotent(tmp_path: Path, migrated):
    """A second run finds nothing to move, including the layout subfolders it created."""
    migrated(WeightsDir("mocktool"))
    _touch(tmp_path / "mocktool" / "model.pt")

    assert len(migrate_toolkit("mocktool", tmp_path)) == 1
    assert migrate_toolkit("mocktool", tmp_path) == []


def test_migrate_keeps_both_on_conflict(tmp_path: Path, migrated, caplog):
    """When the new location already exists, the old one is left in place with a warning."""
    migrated(WeightsDir("mocktool"))
    _touch(tmp_path / "mocktool" / "model.pt", "old")
    _touch(tmp_path / "mocktool" / "weights" / "model.pt", "new")

    with caplog.at_level(logging.WARNING, logger="proto_tools.utils.weights_layout"):
        assert migrate_toolkit("mocktool", tmp_path) == []

    assert (tmp_path / "mocktool" / "model.pt").read_text() == "old"
    assert (tmp_path / "mocktool" / "weights" / "model.pt").read_text() == "new"
    assert "already exists" in caplog.text


def test_migrate_dry_run_touches_nothing(tmp_path: Path, migrated):
    """A dry run reports the moves and leaves the filesystem as it was."""
    migrated(HFRepo("org/model"))
    old = _touch(tmp_path / "huggingface" / "hub" / "models--org--model" / "w.bin")

    moves = migrate_toolkit("mocktool", tmp_path, dry_run=True)

    assert moves == [(old.parent, tmp_path / "mocktool" / "huggingface" / "hub" / "models--org--model")]
    assert old.is_file()
    assert not (tmp_path / "mocktool").exists()


def test_migrate_shared_key_moves_into_shared_folder(tmp_path: Path, migrated, key_file):
    """A toolkit sharing another's weights migrates its repos into that folder."""
    migrated(HFRepo("org/model"))
    key_file("esm2")
    _touch(tmp_path / "huggingface" / "hub" / "models--org--model" / "w.bin")

    migrate_toolkit("mocktool", tmp_path)

    assert (tmp_path / "esm2" / "huggingface" / "hub" / "models--org--model" / "w.bin").is_file()


def _shared_blob_repo(hub: Path, repo: str, etag: str, xet_hash: str) -> Path:
    """Lay out a hub repo whose blob links into the shared store, as huggingface_hub 1.x does."""
    payload = _touch(hub / "blobs" / xet_hash[:2] / xet_hash, "weights")
    _touch(hub / "blobs" / ".huggingface-shared-blobs", "1\n")
    repo_blobs = hub / repo / "blobs"
    repo_blobs.mkdir(parents=True)
    (repo_blobs / etag).symlink_to(Path("..", "..", "blobs", xet_hash[:2], xet_hash))
    snapshot = hub / repo / "snapshots" / "rev"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").symlink_to(Path("..", "..", "blobs", etag))
    manifest = payload.with_name(f"{xet_hash}.refs")
    manifest.write_text(
        manifest.read_text() + f"{repo}/blobs/{etag}\n" if manifest.exists() else f"{repo}/blobs/{etag}\n"
    )
    return payload


def test_migrate_hf_repo_with_shared_blobs(tmp_path: Path, migrated):
    """Moved repos keep readable weights when their blobs lived in the cache-wide shared store."""
    migrated(HFRepo("org/model"))
    hub = tmp_path / "huggingface" / "hub"
    xet_hash = "ab" + "0" * 62
    payload = _shared_blob_repo(hub, "models--org--model", "e1", xet_hash)
    _shared_blob_repo(hub, "models--org--other", "e2", xet_hash)

    migrate_toolkit("mocktool", tmp_path)

    moved = tmp_path / "mocktool" / "huggingface" / "hub" / "models--org--model"
    assert (moved / "snapshots" / "rev" / "model.safetensors").read_text() == "weights"
    # The other repo still references the shared payload, so it stays, minus our manifest entry
    assert payload.read_text() == "weights"
    assert payload.with_name(f"{xet_hash}.refs").read_text() == "models--org--other/blobs/e2\n"


def test_migrate_drops_unreferenced_shared_blob(tmp_path: Path, migrated):
    """A shared payload referenced only by the moved repo is removed from the old store."""
    migrated(HFRepo("org/model"))
    xet_hash = "cd" + "0" * 62
    payload = _shared_blob_repo(tmp_path / "huggingface" / "hub", "models--org--model", "e1", xet_hash)

    migrate_toolkit("mocktool", tmp_path)

    assert not payload.exists()
    assert not payload.with_name(f"{xet_hash}.refs").exists()


def test_migrate_hf_repo_pattern_and_hf_home_path(tmp_path: Path, migrated):
    """Repo families match by glob, and library caches kept beside hub/ move with HF_HOME."""
    migrated(HFRepoPattern("models--org--sae-*"), HFHomePath("evo2_vllm"))
    hub = tmp_path / "huggingface" / "hub"
    _touch(hub / "models--org--sae-k64" / "w.bin")
    _touch(hub / "models--org--sae-k128" / "w.bin")
    _touch(hub / "models--org--base" / "w.bin")
    _touch(tmp_path / "huggingface" / "evo2_vllm" / "evo2_40b_hf" / "model.safetensors")

    migrate_toolkit("mocktool", tmp_path)

    new_hf = tmp_path / "mocktool" / "huggingface"
    assert (new_hf / "hub" / "models--org--sae-k64" / "w.bin").is_file()
    assert (new_hf / "hub" / "models--org--sae-k128" / "w.bin").is_file()
    assert (new_hf / "evo2_vllm" / "evo2_40b_hf" / "model.safetensors").is_file()
    assert (hub / "models--org--base" / "w.bin").is_file()
