"""Tests for the download-aware request deadline."""

import subprocess
import sys
from pathlib import Path

import pytest

from proto_tools.utils import download_progress
from proto_tools.utils.download_progress import DownloadWatch, ProgressDeadline, _tree_bytes
from proto_tools.utils.persistent_worker import _build_subprocess_env, _download_dirs


class _Clock:
    """A hand-advanced stand-in for ``time.monotonic``."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(download_progress.time, "monotonic", fake)
    return fake


def _grow(path: Path, n: int = download_progress._MIN_GROWTH_BYTES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as f:
        f.write(b"x" * n)


# ── Measuring growth ────────────────────────────────────────────────────────


def test_tree_bytes_skips_symlinks_and_missing_dirs(tmp_path: Path):
    blob = tmp_path / "blobs" / "abc"
    _grow(blob, 100)
    (tmp_path / "snapshots").mkdir()
    (tmp_path / "snapshots" / "model.bin").symlink_to(blob)
    assert _tree_bytes(tmp_path) == 100
    assert _tree_bytes(tmp_path / "missing") == 0


def test_watch_reports_growth_only_after_a_baseline(tmp_path: Path):
    watch = DownloadWatch([tmp_path / "a", tmp_path / "b", tmp_path / "a"])
    assert not watch.grew()  # baseline
    _grow(tmp_path / "b" / "x.incomplete")
    assert watch.grew()
    assert not watch.grew()
    (tmp_path / "b" / "x.incomplete").unlink()
    assert not watch.grew()  # shrinking is not progress
    _grow(tmp_path / "a" / "y.tmp")
    assert watch.grew()


# ── The charge rule ─────────────────────────────────────────────────────────


def test_deadline_extends_only_across_intervals_with_growth(
    tmp_path: Path, clock: _Clock, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("PROTO_DOWNLOAD_POLL_SECONDS", "5")
    blob = tmp_path / "hub" / "blob.incomplete"
    deadline = ProgressDeadline(10, [tmp_path], label="tk")

    clock.now = 5
    deadline.tick()  # baseline
    _grow(blob)
    clock.now = 10
    deadline.tick()  # grew over 5-10: credited
    assert deadline.download_seconds == 5
    assert deadline.downloading
    clock.now = 14
    assert not deadline.expired()

    clock.now = 15
    deadline.tick()  # nothing new: the clock runs again
    assert not deadline.downloading
    assert deadline.expired()
    assert deadline.timeout_note() == " (excluding 5s spent downloading weights)"


def test_small_writes_are_not_a_download(tmp_path: Path, clock: _Clock, monkeypatch: pytest.MonkeyPatch):
    """A cache or log that grows a little each interval must not keep a wedged worker alive."""
    monkeypatch.setenv("PROTO_DOWNLOAD_POLL_SECONDS", "5")
    deadline = ProgressDeadline(10, [tmp_path], label="tk")
    clock.now = 5
    deadline.tick()  # baseline
    _grow(tmp_path / "embeds_cache" / "seq.npy", download_progress._MIN_GROWTH_BYTES - 1)
    clock.now = 10
    deadline.tick()
    assert deadline.download_seconds == 0
    assert deadline.expired()


def test_download_credit_is_capped(tmp_path: Path, clock: _Clock, monkeypatch: pytest.MonkeyPatch):
    """Growth past ``PROTO_DOWNLOAD_MAX_SECONDS`` of credit no longer extends the deadline."""
    monkeypatch.setenv("PROTO_DOWNLOAD_POLL_SECONDS", "5")
    monkeypatch.setenv("PROTO_DOWNLOAD_MAX_SECONDS", "7")
    deadline = ProgressDeadline(10, [tmp_path], label="tk")
    for now in (5, 10, 15, 20):  # baseline, then growth in every interval
        clock.now = now
        _grow(tmp_path / "blob.incomplete")
        deadline.tick()
    assert deadline.download_seconds == 7
    clock.now = 17
    assert deadline.expired()


def test_deadline_without_downloads_expires_on_schedule(tmp_path: Path, clock: _Clock):
    deadline = ProgressDeadline(4, [tmp_path], label="tk")
    assert deadline.wait_slice() == 2  # the baseline is taken halfway through a short timeout
    clock.now = 4
    deadline.tick()
    assert deadline.expired()
    assert deadline.timeout_note() == ""


def test_deadline_with_no_timeout_never_expires(tmp_path: Path, clock: _Clock):
    deadline = ProgressDeadline(None, [tmp_path], label="tk")
    clock.now = 1e9
    deadline.tick()
    assert deadline.wait_slice() is None
    assert not deadline.expired()


def test_wait_process_returns_exit_code_and_raises_when_due(tmp_path: Path):
    quick = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
    assert ProgressDeadline(30, [tmp_path], label="tk").wait_process(quick) == 3

    slow = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            ProgressDeadline(0.5, [tmp_path], label="tk").wait_process(slow)
        assert slow.poll() is None  # the caller decides how to stop it
    finally:
        slow.kill()
        slow.wait()


# ── Which folders count as downloads ────────────────────────────────────────


def test_download_dirs_in_path_mode_are_the_toolkit_download_folders(tmp_path: Path):
    env = {"PROTO_MODEL_CACHE": str(tmp_path), "HF_HOME": str(tmp_path / "tk" / "huggingface")}
    env["TORCH_HOME"] = str(tmp_path / "tk" / "torch")
    assert _download_dirs("tk", env, tmp_path / "venv") == [
        tmp_path / "tk" / "huggingface",
        tmp_path / "tk" / "torch",
        tmp_path / "tk" / "weights",
    ]


@pytest.mark.parametrize(("mode", "weights"), [("IN_ENV", "model_weight_cache"), ("NONE", "weights")])
def test_download_dirs_in_venv_modes_use_the_venv_weights_dir(tmp_path: Path, mode: str, weights: str):
    assert _download_dirs("tk", {"PROTO_MODEL_CACHE": mode}, tmp_path) == [tmp_path / weights]


def test_download_dirs_honor_a_weights_dir_override(tmp_path: Path):
    env = {"PROTO_MODEL_CACHE": str(tmp_path), "PROTO_TK_WEIGHTS_DIR": str(tmp_path / "mine")}
    assert _download_dirs("tk", env, None) == [tmp_path / "mine"]


def test_download_dirs_leave_out_compile_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """JIT compile time stays charged: no compile or JAX cache folder is watched."""
    monkeypatch.setenv("PROTO_MODEL_CACHE", str(tmp_path))
    env = _build_subprocess_env(device="cpu", tool_env_path=tmp_path / "venv", toolkit="tk")
    watched = {str(d) for d in _download_dirs("tk", env, tmp_path / "venv")}
    assert env["JAX_COMPILATION_CACHE_DIR"] not in watched
    assert not any("compile_cache" in d for d in watched)
    assert str(tmp_path / "tk" / "weights") in watched
