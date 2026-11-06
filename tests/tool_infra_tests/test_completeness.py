"""Tests for knowing a directory is complete by the files it holds (standalone_helpers.completeness)."""

import subprocess
import threading
from pathlib import Path

import pytest

from proto_tools.utils.standalone_helpers_source.standalone_helpers.completeness import (
    COMPLETE_MARKER,
    ensure_complete,
    list_files,
    missing_files,
    missing_recorded_files,
    record_files,
)

_HELPERS = Path(__file__).resolve().parents[2] / "proto_tools/utils/standalone_helpers_source/standalone_helpers.sh"


def _write(root: Path, *rels: str) -> None:
    for rel in rels:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(rel)


def _bash(command: str) -> int:
    return subprocess.run(["bash", "-c", f'source "{_HELPERS}"; {command}'], check=False).returncode


def test_list_files_skips_dotfiles_transient_downloads_and_named_caches(tmp_path: Path):
    _write(tmp_path, "model.pt", "sub/vocab.txt", ".lock", ".git/HEAD", "blob.incomplete", "x.tmp", "jax_cache/p")

    assert list_files(tmp_path, skip_dirs=["jax_cache"]) == ["model.pt", "sub/vocab.txt"]
    assert list_files(tmp_path / "absent") == []


def test_missing_files_reports_files_and_whole_directories_that_are_gone(tmp_path: Path):
    _write(tmp_path, "a.pt", "sub/b.pt")

    assert missing_files(tmp_path, ["a.pt", "gone.pt", "sub/b.pt", "lost/c.pt"]) == ["gone.pt", "lost/c.pt"]


def test_recorded_files_vouch_for_a_directory_only_while_present(tmp_path: Path):
    _write(tmp_path, "a.pt", "sub/b.pt")
    assert missing_recorded_files(tmp_path) is None

    record_files(tmp_path)
    assert missing_recorded_files(tmp_path) == []
    (tmp_path / "sub/b.pt").unlink()
    assert missing_recorded_files(tmp_path) == ["sub/b.pt"]

    (tmp_path / COMPLETE_MARKER).write_text("")
    assert missing_recorded_files(tmp_path) == [COMPLETE_MARKER], "an empty marker records nothing"


@pytest.mark.parametrize("writer", ["python", "bash"])
def test_bash_and_python_markers_agree(tmp_path: Path, writer: str):
    """Either side can write the marker and either side reads it the same way."""
    _write(tmp_path, "a.pt", "sub/b.pt", ".hidden", "c.lock")
    if writer == "python":
        record_files(tmp_path)
    else:
        assert _bash(f'proto_record_files "{tmp_path}"') == 0

    assert (tmp_path / COMPLETE_MARKER).read_text() == "a.pt\nsub/b.pt\n"
    assert _bash(f'proto_recorded_files_complete "{tmp_path}"') == 0
    (tmp_path / "a.pt").unlink()
    assert _bash(f'proto_recorded_files_complete "{tmp_path}"') != 0
    assert missing_recorded_files(tmp_path) == ["a.pt"]


def _counting_build(builds: list[Path], *files: str):
    def build(directory: Path) -> None:
        builds.append(directory)
        _write(directory, *files)

    return build


@pytest.mark.parametrize("stage", [False, True])
def test_ensure_complete_builds_once_and_again_after_a_purge(tmp_path: Path, stage: bool):
    dest, builds = tmp_path / "weights", []
    build = _counting_build(builds, "model.pt", "sub/vocab.txt")

    ensure_complete(dest, build, stage=stage)
    ensure_complete(dest, build, stage=stage)
    assert len(builds) == 1

    (dest / "sub/vocab.txt").unlink()
    (dest / "stale-partial").write_text("x")
    ensure_complete(dest, build, stage=stage)

    assert len(builds) == 2
    assert list_files(dest) == ["model.pt", "sub/vocab.txt"], "the incomplete copy was replaced, not patched"
    assert sorted(p.name for p in tmp_path.iterdir() if not p.name.endswith(".lock")) == ["weights"]


def test_ensure_complete_customizations(tmp_path: Path):
    """``reset=False`` keeps what is there for a build that adds what is missing; ``record`` picks what counts."""
    dest, builds = tmp_path / "weights", []
    _write(dest, "kept.txt")

    def add_missing(directory: Path) -> None:
        builds.append(directory)
        if not (directory / "model.pt").exists():
            _write(directory, "model.pt")

    ensure_complete(dest, add_missing, reset=False, record=lambda d: ["model.pt"], marker=".ready")
    assert (dest / "kept.txt").exists() and (dest / ".ready").read_text() == "model.pt\n"

    (dest / "kept.txt").unlink()
    ensure_complete(dest, add_missing, reset=False, record=lambda d: ["model.pt"], marker=".ready")
    assert len(builds) == 1, "only the recorded files decide completeness"

    ensure_complete(dest, add_missing, reset=False, record=lambda d: ["model.pt"], marker=".ready", force=True)
    assert len(builds) == 2


def test_failed_staged_build_leaves_the_previous_copy_and_no_staging_behind(tmp_path: Path):
    dest = tmp_path / "weights"
    _write(dest, "old.pt")

    def failing(directory: Path) -> None:
        _write(directory, "half.pt")
        raise RuntimeError("download failed")

    with pytest.raises(RuntimeError, match="download failed"):
        ensure_complete(dest, failing, stage=True)

    assert list_files(dest) == ["old.pt"]
    assert missing_recorded_files(dest) is None, "no marker: still incomplete"
    assert [p.name for p in tmp_path.iterdir() if ".stage." in p.name] == []


def test_concurrent_callers_build_once(tmp_path: Path):
    dest, builds = tmp_path / "weights", []
    build = _counting_build(builds, "model.pt")

    threads = [threading.Thread(target=ensure_complete, args=(dest, build), kwargs={"stage": True}) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(builds) == 1
    assert missing_recorded_files(dest) == []


def test_build_finds_an_incomplete_dest_removed(tmp_path: Path):
    """A build that acts only when its target is absent (boltz extracting mols/, git clone) still runs."""
    dest = tmp_path / "mols"
    _write(dest, "a.pkl")
    record_files(dest)
    (dest / "a.pkl").unlink()
    seen = []

    def extract_if_absent(directory: Path) -> None:
        seen.append(directory.exists())
        _write(directory, "a.pkl")

    ensure_complete(dest, extract_if_absent)

    assert seen == [False]
    assert missing_recorded_files(dest) == []


def test_adopt_records_a_copy_built_before_records_instead_of_rebuilding_it(tmp_path: Path):
    """Valid weights from before records existed are kept; a copy failing the old check is still rebuilt."""
    dest, builds = tmp_path / "weights", []
    build = _counting_build(builds, "model.pt", "vocab.txt")
    _write(dest, "model.pt", "vocab.txt")

    ensure_complete(dest, build, adopt=lambda d: (d / "model.pt").exists())
    assert builds == []
    assert missing_recorded_files(dest) == []

    other = tmp_path / "other"
    _write(other, "vocab.txt")
    ensure_complete(other, build, adopt=lambda d: (d / "model.pt").exists())
    assert len(builds) == 1
