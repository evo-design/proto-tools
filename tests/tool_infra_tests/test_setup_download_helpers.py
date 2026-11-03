"""tests/tool_infra_tests/test_setup_download_helpers.py.

Tests for the setup.sh download helpers in standalone_helpers.sh, run in bash against file:// URLs.
"""

import hashlib
import io
import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

_HELPERS = (
    Path(__file__).resolve().parents[2]
    / "proto_tools"
    / "utils"
    / "standalone_helpers_source"
    / "standalone_helpers.sh"
)


def _run(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'set -euo pipefail; source "{_HELPERS}"; {command}'],
        capture_output=True,
        text=True,
        check=False,
    )


def _run_concurrently(command: str, n: int, env: dict[str, str] | None = None) -> list[tuple[int, str]]:
    """Start ``n`` copies of a helper command at once; return each one's exit code and output."""
    procs = [
        subprocess.Popen(
            ["bash", "-c", f'set -euo pipefail; source "{_HELPERS}"; {command}'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, **(env or {})},
        )
        for _ in range(n)
    ]
    return [(proc.wait(), proc.stdout.read()) for proc in procs]


_needs_flock = pytest.mark.skipif(shutil.which("flock") is None, reason="flock(1) not available")


def _zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        # A fixed timestamp keeps the bytes identical across calls; writestr otherwise stamps the current time
        archive.writestr(zipfile.ZipInfo("archive/data.pkl", date_time=(1980, 1, 1, 0, 0, 0)), b"x" * 4096)
    return buffer.getvalue()


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "remote" / "model.pt"
    path.parent.mkdir()
    path.write_bytes(_zip_bytes())
    return path


@pytest.mark.parametrize(
    ("content", "truncated"),
    [
        (_zip_bytes(), False),
        (_zip_bytes()[:-64], True),
        (b"\x80\x02legacy torch pickle", False),
    ],
)
def test_is_truncated_zip(tmp_path: Path, content: bytes, truncated: bool) -> None:
    path = tmp_path / "model.pt"
    path.write_bytes(content)
    assert (_run(f'proto_is_truncated_zip "{path}"').returncode == 0) is truncated


def test_download_fetches_into_place(tmp_path: Path, source: Path) -> None:
    dest = tmp_path / "weights" / "model.pt"
    assert _run(f'proto_download "file://{source}" "{dest}"').returncode == 0
    assert dest.read_bytes() == source.read_bytes()
    assert not list(dest.parent.glob("*.tmp")) and not list(dest.parent.glob(".*.tmp"))


@pytest.mark.parametrize("existing", [_zip_bytes(), b"\x80\x02legacy torch pickle"])
def test_download_keeps_complete_file(tmp_path: Path, existing: bytes) -> None:
    dest = tmp_path / "model.pt"
    dest.write_bytes(existing)
    assert _run(f'proto_download "file://{tmp_path}/missing.pt" "{dest}"').returncode == 0
    assert dest.read_bytes() == existing


@pytest.mark.parametrize("existing", [b"", _zip_bytes()[:-64]])
def test_download_replaces_incomplete_file(tmp_path: Path, source: Path, existing: bytes) -> None:
    dest = tmp_path / "model.pt"
    dest.write_bytes(existing)
    assert _run(f'proto_download "file://{source}" "{dest}"').returncode == 0
    assert dest.read_bytes() == source.read_bytes()


def test_download_failure_leaves_nothing_behind(tmp_path: Path) -> None:
    dest = tmp_path / "model.pt"
    assert _run(f'proto_download "file://{tmp_path}/missing.pt" "{dest}"').returncode != 0
    # The 0-byte lock file is kept on purpose (see proto_with_lock); nothing else may remain
    assert [p.name for p in tmp_path.iterdir()] in ([], [".model.pt.download.lock"])


def test_download_archive_extracts_once(tmp_path: Path) -> None:
    archive = tmp_path / "params.tar"
    with tarfile.open(archive, "w") as tar:
        payload = tmp_path / "params_model_1.npz"
        payload.write_bytes(_zip_bytes())
        tar.add(payload, arcname=payload.name)
    dest = tmp_path / "params"

    assert _run(f'proto_download_archive "file://{archive}" "{dest}"').returncode == 0
    assert (dest / "params_model_1.npz").read_bytes() == _zip_bytes()
    archive.unlink()
    assert _run(f'proto_download_archive "file://{archive}" "{dest}"').returncode == 0


def test_download_archive_failure_is_not_marked_complete(tmp_path: Path) -> None:
    dest = tmp_path / "params"
    (dest).mkdir()
    (dest / "params_model_1.npz").write_bytes(b"partial")

    assert _run(f'proto_download_archive "file://{tmp_path}/missing.tar" "{dest}"').returncode != 0
    assert not (dest / ".proto_download_complete").exists()


# ── Concurrent callers sharing a model cache ─────────────────────────────────


@_needs_flock
def test_with_lock_runs_holders_one_at_a_time(tmp_path: Path) -> None:
    """Commands under the same lock never overlap: every start is followed by its own end."""
    log = tmp_path / "log"
    results = _run_concurrently(
        f'proto_with_lock "{tmp_path}/.lock" bash -c \'echo start >> "{log}"; sleep 0.3; echo end >> "{log}"\'', 3
    )
    assert [rc for rc, _ in results] == [0, 0, 0]
    assert log.read_text().split() == ["start", "end"] * 3


@_needs_flock
def test_with_lock_times_out(tmp_path: Path) -> None:
    """A holder that outlasts PROTO_LOCK_TIMEOUT_SECONDS makes the waiter fail instead of hang."""
    holder = subprocess.Popen(["flock", "-x", str(tmp_path / ".lock"), "sleep", "5"])
    try:
        subprocess.run(["sleep", "0.5"], check=True)
        result = _run_concurrently(
            f'proto_with_lock "{tmp_path}/.lock" true', 1, env={"PROTO_LOCK_TIMEOUT_SECONDS": "1"}
        )
    finally:
        holder.kill()
    assert result[0][0] != 0
    assert "timed out waiting for lock" in result[0][1]


@_needs_flock
def test_concurrent_verified_downloads_fetch_once(tmp_path: Path) -> None:
    """Setups racing for one file: one downloads, the rest find it verified, nothing is left over."""
    source = tmp_path / "remote" / "trunk.pt"
    source.parent.mkdir()
    source.write_bytes(os.urandom(8 << 20))
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    dest = tmp_path / "weights" / "models_v2" / "trunk.pt"
    dest.parent.mkdir(parents=True)

    results = _run_concurrently(f'proto_download_verified "file://{source}" "{dest}" "{sha}"', 4)

    assert [rc for rc, _ in results] == [0, 0, 0, 0]
    assert dest.read_bytes() == source.read_bytes()
    assert sum("Downloading trunk.pt" in out for _, out in results) == 1
    assert sorted(p.name for p in dest.parent.iterdir()) == [".trunk.pt.download.lock", "trunk.pt"]


@_needs_flock
def test_concurrent_archive_downloads_extract_once(tmp_path: Path) -> None:
    """Setups racing for one archive: one extracts, the rest find the completion marker."""
    archive = tmp_path / "params.tar"
    payload = tmp_path / "params_model_1.npz"
    payload.write_bytes(_zip_bytes())
    with tarfile.open(archive, "w") as tar:
        tar.add(payload, arcname=payload.name)
    dest = tmp_path / "weights" / "params"

    results = _run_concurrently(f'proto_download_archive "file://{archive}" "{dest}"', 4)

    assert [rc for rc, _ in results] == [0, 0, 0, 0]
    assert sum("Downloading and extracting" in out for _, out in results) == 1
    assert (dest / "params_model_1.npz").read_bytes() == _zip_bytes()
