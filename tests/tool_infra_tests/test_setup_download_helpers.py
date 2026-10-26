"""tests/tool_infra_tests/test_setup_download_helpers.py.

Tests for the setup.sh download helpers in standalone_helpers.sh, run in bash against file:// URLs.
"""

import io
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


def _zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("archive/data.pkl", b"x" * 4096)
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
    assert not dest.with_name("model.pt.part").exists()


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
    assert list(tmp_path.iterdir()) == []


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
