"""Tests for the record of weight files a toolkit's setup leaves on disk."""

from pathlib import Path

import pytest

from proto_tools.utils.weights_manifest import RECORD_NAME, weights_problem, write_manifest


@pytest.fixture
def cache(monkeypatch, tmp_path: Path) -> Path:
    """A model cache root, with the setup's record of its own and the shared ``alphafold2`` weights folders."""
    root = tmp_path / "cache"
    monkeypatch.setenv("PROTO_MODEL_CACHE", str(root))
    monkeypatch.delenv("PROTO_BINDCRAFT_WEIGHTS_DIR", raising=False)
    monkeypatch.delenv("PROTO_ALPHAFOLD2_WEIGHTS_DIR", raising=False)
    for rel in ("bindcraft/weights/model.pt", "bindcraft/huggingface/hub/blob", "alphafold2/weights/params/a.npz"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"w")
    env = tmp_path / "env"
    env.mkdir()
    (env / RECORD_NAME).write_text(f"alphafold2\t{root}/alphafold2/weights\nbindcraft\t{root}/bindcraft/weights\n")
    return root


def test_intact_weights_need_no_setup(cache: Path, tmp_path: Path):
    write_manifest(tmp_path / "env", "bindcraft")
    assert weights_problem(tmp_path / "env") is None


@pytest.mark.parametrize("deleted", ["bindcraft/weights/model.pt", "alphafold2/weights/params"])
def test_deleted_weights_need_setup(cache: Path, tmp_path: Path, deleted: str):
    """A missing file, in the toolkit's own weights folder or a shared folder setup filled, is reported."""
    write_manifest(tmp_path / "env", "bindcraft")
    target = cache / deleted
    if target.is_dir():
        for path in target.iterdir():
            path.unlink()
        target.rmdir()
    else:
        target.unlink()

    problem = weights_problem(tmp_path / "env")

    assert problem is not None and "missing" in problem


def test_runtime_downloads_and_transient_files_are_not_tracked(cache: Path, tmp_path: Path):
    """Worker downloads, caches, dotfiles, and in-progress downloads may come and go without a new setup."""
    transient = [
        "bindcraft/huggingface/hub/runtime-blob",
        "bindcraft/compile_cache/triton/kernel",
        "bindcraft/weights/jax_cache/program",
        "bindcraft/weights/.params_complete",
        "bindcraft/weights/blob2.incomplete",
        "bindcraft/weights/model.pt.lock",
    ]
    for rel in transient:
        (cache / rel).parent.mkdir(parents=True, exist_ok=True)
        (cache / rel).write_bytes(b"t")
    write_manifest(tmp_path / "env", "bindcraft")

    for rel in transient:
        (cache / rel).unlink()

    assert weights_problem(tmp_path / "env") is None


@pytest.mark.parametrize("variable", ["PROTO_MODEL_CACHE", "PROTO_ALPHAFOLD2_WEIGHTS_DIR"])
def test_moved_weights_location_needs_setup(cache: Path, tmp_path: Path, monkeypatch, variable: str):
    """Pointing the cache, or a folder setup filled, somewhere new means its weights must be fetched there."""
    write_manifest(tmp_path / "env", "bindcraft")
    monkeypatch.setenv(variable, str(tmp_path / "elsewhere"))

    assert weights_problem(tmp_path / "env") == f"its weights location changed ({variable})"


def test_env_without_a_record_needs_setup(tmp_path: Path):
    """An environment built before weights were tracked cannot vouch for them."""
    assert weights_problem(tmp_path) == "it has no record of its weight files yet"
