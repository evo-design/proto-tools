"""tests/tool_infra_tests/test_setup_download_helpers.py.

Tests for the setup.sh download helpers in standalone_helpers.sh, run in bash against file:// URLs.
"""

import hashlib
import http.server
import io
import os
import shutil
import subprocess
import tarfile
import threading
import time
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
    assert "timed out after 1s waiting for lock" in result[0][1]


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


@_needs_flock
def test_with_lock_passes_exit_status_and_releases_on_failure(tmp_path: Path) -> None:
    """A failing command fails the call (so set -e setups abort) without keeping the lock."""
    failed = _run(f'proto_with_lock "{tmp_path}/.lock" bash -c "exit 7"')
    assert failed.returncode == 7
    after = _run_concurrently(f'proto_with_lock "{tmp_path}/.lock" true', 1, env={"PROTO_LOCK_TIMEOUT_SECONDS": "1"})
    assert after[0][0] == 0


@_needs_flock
def test_with_lock_is_released_when_holder_is_killed(tmp_path: Path) -> None:
    """Killing the holder with kill -9 frees the lock: the kernel drops flocks of dead processes."""
    marker = tmp_path / "holding"
    holder = subprocess.Popen(
        [
            "bash",
            "-c",
            f'source "{_HELPERS}"; proto_with_lock "{tmp_path}/.lock" bash -c \'touch "{marker}"; sleep 60\'',
        ],
        start_new_session=True,
    )
    try:
        for _ in range(50):
            if marker.exists():
                break
            subprocess.run(["sleep", "0.1"], check=True)
        assert marker.exists()
        os.killpg(holder.pid, 9)
        holder.wait()
        waiter = _run_concurrently(
            f'proto_with_lock "{tmp_path}/.lock" true', 1, env={"PROTO_LOCK_TIMEOUT_SECONDS": "5"}
        )
    finally:
        if holder.poll() is None:
            os.killpg(holder.pid, 9)
    assert waiter[0][0] == 0


@_needs_flock
def test_with_lock_is_reentrant(tmp_path: Path) -> None:
    """Taking a lock already held by this call chain runs straight through instead of deadlocking."""
    nested = f'proto_with_lock "{tmp_path}/.lock" proto_with_lock "{tmp_path}/.lock" echo inner'
    result = _run_concurrently(nested, 1, env={"PROTO_LOCK_TIMEOUT_SECONDS": "2"})
    assert result[0] == (0, "inner\n")


@_needs_flock
def test_with_lock_keeps_outer_lock_while_nesting_another(tmp_path: Path) -> None:
    """An inner lock on a different file does not release the outer one."""
    outer, inner = tmp_path / ".outer.lock", tmp_path / ".inner.lock"
    holder = subprocess.Popen(
        [
            "bash",
            "-c",
            f'source "{_HELPERS}"; proto_with_lock "{outer}" proto_with_lock "{inner}" bash -c \'touch "{tmp_path}/go"; sleep 2\'',
        ]
    )
    try:
        for _ in range(50):
            if (tmp_path / "go").exists():
                break
            subprocess.run(["sleep", "0.1"], check=True)
        probe = subprocess.run(["flock", "-n", str(outer), "true"], check=False)
    finally:
        holder.wait()
    assert probe.returncode != 0, "outer lock was free while the inner one was held"


def test_with_lock_runs_unlocked_without_flock(tmp_path: Path) -> None:
    """Without flock(1) (macOS), the command still runs, unlocked as before."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("bash", "dirname", "mkdir", "basename"):
        (bin_dir / tool).symlink_to(shutil.which(tool))
    result = subprocess.run(
        ["bash", "-c", f'source "{_HELPERS}"; proto_with_lock "{tmp_path}/.lock" echo ran'],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": str(bin_dir)},
    )
    assert (result.returncode, result.stdout) == (0, "ran\n")


@_needs_flock
def test_verified_download_skips_lock_for_present_file(tmp_path: Path, source: Path) -> None:
    """A verified file in a read-only cache is accepted without creating a lock file."""
    cache = tmp_path / "readonly"
    cache.mkdir()
    dest = cache / "model.pt"
    dest.write_bytes(source.read_bytes())
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    cache.chmod(0o555)
    try:
        result = _run(f'proto_download_verified "file://{source}" "{dest}" "{sha}"')
    finally:
        cache.chmod(0o755)
    assert result.returncode == 0, result.stderr
    assert "already present and verified" in result.stdout
    assert sorted(p.name for p in cache.iterdir()) == ["model.pt"]


@pytest.mark.parametrize("helper", ["proto_download", "proto_download_verified"])
def test_download_removes_temps_left_by_a_killed_run(tmp_path: Path, source: Path, helper: str) -> None:
    """A temp file from a download that was killed mid-transfer is cleaned up by the next one."""
    dest = tmp_path / "weights" / "model.pt"
    dest.parent.mkdir()
    stale = dest.parent / ".model.pt.Ab12Cd.tmp"
    stale.write_bytes(b"partial transfer")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    args = f'"file://{source}" "{dest}"' + (f' "{sha}"' if helper == "proto_download_verified" else "")
    assert _run(f"{helper} {args}").returncode == 0
    assert dest.read_bytes() == source.read_bytes()
    assert not stale.exists()


@_needs_flock
def test_different_files_in_one_folder_download_in_parallel(tmp_path: Path) -> None:
    """Locks are per target file: holding one file's lock does not block a sibling's download."""
    source = tmp_path / "remote" / "b.pt"
    source.parent.mkdir()
    source.write_bytes(_zip_bytes())
    weights = tmp_path / "weights"
    weights.mkdir()
    holder = subprocess.Popen(["flock", "-x", str(weights / ".a.pt.download.lock"), "sleep", "5"])
    try:
        subprocess.run(["sleep", "0.3"], check=True)
        result = _run_concurrently(
            f'proto_download "file://{source}" "{weights}/b.pt"', 1, env={"PROTO_LOCK_TIMEOUT_SECONDS": "1"}
        )
    finally:
        holder.kill()
    assert result[0][0] == 0
    assert (weights / "b.pt").read_bytes() == _zip_bytes()


# Fast waits for the progress tests: check every second, call 2 s without growth a stall
_FAST_WAIT = {"PROTO_LOCK_POLL_SECONDS": "1", "PROTO_LOCK_STALL_SECONDS": "2"}


def _hold_lock(lock: Path, script: str) -> subprocess.Popen:
    """Start a holder that takes ``lock`` and runs ``script``; return once it holds the lock."""
    started = lock.with_name(lock.name + ".held")
    holder = subprocess.Popen(
        ["bash", "-c", f'source "{_HELPERS}"; proto_with_lock "{lock}" bash -c \'touch "{started}"; {script}\'']
    )
    for _ in range(50):
        if started.exists():
            return holder
        time.sleep(0.1)
    holder.kill()
    raise AssertionError("holder never took the lock")


@_needs_flock
def test_waiter_outlasts_stall_window_while_holder_makes_progress(tmp_path: Path) -> None:
    """A slow but live download keeps its waiters waiting, however long it takes."""
    growing = tmp_path / "part.tmp"
    holder = _hold_lock(tmp_path / ".lock", f'for i in 1 2 3 4 5 6; do printf x >> "{growing}"; sleep 1; done')
    started = time.monotonic()
    result = _run_concurrently(f'proto_with_lock --progress "{growing}" "{tmp_path}/.lock" true', 1, env=_FAST_WAIT)
    holder.wait()
    assert result[0][0] == 0, result[0][1]
    assert time.monotonic() - started > 4  # waited well past the 2 s stall window


@_needs_flock
def test_waiter_gives_up_on_a_stalled_holder(tmp_path: Path) -> None:
    """A holder whose progress path stops growing is reported instead of waited on forever."""
    growing = tmp_path / "part.tmp"
    growing.write_bytes(b"x")
    holder = _hold_lock(tmp_path / ".lock", "sleep 20")
    try:
        started = time.monotonic()
        result = _run_concurrently(f'proto_with_lock --progress "{growing}" "{tmp_path}/.lock" true', 1, env=_FAST_WAIT)
        elapsed = time.monotonic() - started
    finally:
        holder.kill()
    assert result[0][0] != 0
    assert "made no progress" in result[0][1]
    assert elapsed < 10


@_needs_flock
def test_waiter_has_no_hard_cap_by_default(tmp_path: Path) -> None:
    """Without PROTO_LOCK_TIMEOUT_SECONDS, a waiter with no progress path waits until the lock is free."""
    holder = _hold_lock(tmp_path / ".lock", "sleep 4")
    result = _run_concurrently(f'proto_with_lock "{tmp_path}/.lock" true', 1, env={"PROTO_LOCK_POLL_SECONDS": "1"})
    holder.wait()
    assert result[0][0] == 0, result[0][1]


@_needs_flock
def test_verified_download_waits_out_a_slow_holder(tmp_path: Path, source: Path) -> None:
    """The helper's waiter tracks the holder's temp file, then finds the finished file verified."""
    dest = tmp_path / "weights" / "model.pt"
    dest.parent.mkdir()
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    tmp = dest.parent / ".model.pt.Hold01.tmp"
    holder = _hold_lock(
        dest.parent / ".model.pt.download.lock",
        f'for i in 1 2 3 4 5; do printf x >> "{tmp}"; sleep 1; done; cp "{source}" "{tmp}"; mv "{tmp}" "{dest}"',
    )
    result = _run_concurrently(f'proto_download_verified "file://{source}" "{dest}" "{sha}"', 1, env=_FAST_WAIT)
    holder.wait()
    assert result[0][0] == 0, result[0][1]
    assert "already present and verified" in result[0][1]
    assert dest.read_bytes() == source.read_bytes()


class _TricklingHandler(http.server.BaseHTTPRequestHandler):
    """Send a small body slowly (~200 B/s), well under the old 1 KB/s abort threshold."""

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", str(1200))
        self.end_headers()
        for _ in range(6):
            self.wfile.write(b"x" * 200)
            self.wfile.flush()
            time.sleep(1)

    def log_message(self, *args: object) -> None:
        pass


def _serve(handler: type[http.server.BaseHTTPRequestHandler]) -> http.server.ThreadingHTTPServer:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_download_keeps_a_slow_transfer_that_is_still_progressing(tmp_path: Path) -> None:
    """A download that keeps receiving data is never aborted, however slow, even past the stall window."""
    server = _serve(_TricklingHandler)
    dest = tmp_path / "model.pt"
    try:
        result = _run_concurrently(
            f'proto_download "http://127.0.0.1:{server.server_address[1]}/model.pt" "{dest}"',
            1,
            env={"PROTO_DOWNLOAD_STALL_SECONDS": "2"},
        )
    finally:
        server.shutdown()
    assert result[0][0] == 0, result[0][1]
    assert dest.read_bytes() == b"x" * 1200


class _StallingHandler(http.server.BaseHTTPRequestHandler):
    """Promise a large body, send a few bytes, then go silent."""

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", str(10 << 20))
        self.end_headers()
        self.wfile.write(b"x" * 4096)
        self.wfile.flush()
        time.sleep(30)

    def log_message(self, *args: object) -> None:
        pass


def test_download_aborts_a_stalled_transfer(tmp_path: Path) -> None:
    """A transfer that receives nothing for PROTO_DOWNLOAD_STALL_SECONDS fails and leaves no temp file."""
    server = _serve(_StallingHandler)
    dest = tmp_path / "model.pt"
    try:
        started = time.monotonic()
        result = _run_concurrently(
            f'proto_download "http://127.0.0.1:{server.server_address[1]}/model.pt" "{dest}"',
            1,
            env={"PROTO_DOWNLOAD_STALL_SECONDS": "2"},
        )
        elapsed = time.monotonic() - started
    finally:
        server.shutdown()
    assert result[0][0] != 0
    assert "failed to download" in result[0][1]
    assert elapsed < 20
    assert not dest.exists() and not list(tmp_path.glob(".model.pt.*.tmp"))


@_needs_flock
def test_concurrent_af2_param_setups_download_once(tmp_path: Path) -> None:
    """Setups of the AF2-param toolkits racing on one cache: one extracts, all find a complete copy."""
    archive = tmp_path / "alphafold_params.tar"
    with tarfile.open(archive, "w") as tar:
        for model in range(1, 6):
            for variant in ("", "_ptm", "_multimer_v3"):
                member = tmp_path / f"params_model_{model}{variant}.npz"
                member.write_bytes(_zip_bytes())
                tar.add(member, arcname=member.name)
    env = {"PROTO_MODEL_CACHE": str(tmp_path / "cache"), "_PROTO_AF2_PARAMS_URL": f"file://{archive}"}

    results = _run_concurrently("proto_ensure_af2_params", 3, env=env)

    assert [rc for rc, _ in results] == [0, 0, 0], [out for _, out in results]
    assert sum("Downloading AlphaFold2 parameters" in out for _, out in results) == 1
    params = tmp_path / "cache" / "alphafold2" / "weights" / "params"
    assert len(list(params.glob("params_model_*.npz"))) == 15
    assert (params / ".params_complete").exists()
