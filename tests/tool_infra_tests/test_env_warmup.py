"""tests/tool_infra_tests/test_env_warmup.py.

Tests for the tool-environment shared-library warm-up.

What warming *achieves* -- turning a million mmap page faults into a handful of sequential
reads -- cannot be asserted here: it depends on a page cache, a parallel filesystem and a
cold node, none of which exist in a test. The measurements live in the module docstring and
in the PR.

What is tested is everything that decides *whether it runs*, which is where the risk is. A
warm-up that fires on local disk wastes a second of everyone's time, one that fires twice
wastes it repeatedly, and one that raises breaks a run it was only ever meant to speed up.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from proto_tools.utils.env_warmup import (
    _filesystem_type,
    _is_network_filesystem,
    _large_shared_libraries,
    _marker_path,
    warm_environment,
)

_BIG = 16 * 2**20


def _env_with_libraries(tmp_path: Path, **sizes: int) -> Path:
    """A tool env laid out like micromamba's, holding the named libraries."""
    site_packages = tmp_path / "env" / "lib" / "python3.12" / "site-packages" / "torch" / "lib"
    site_packages.mkdir(parents=True)
    for name, size in sizes.items():
        (site_packages / f"{name}.so").write_bytes(b"\0" * size)
    return tmp_path / "env"


@pytest.fixture
def on_lustre(monkeypatch):
    """Report every path as living on Lustre."""
    monkeypatch.setattr("proto_tools.utils.env_warmup._is_network_filesystem", lambda _path: True)


@pytest.fixture
def fresh_marker(monkeypatch, tmp_path):
    """Keep warm-up markers out of the real /dev/shm, so tests cannot see each other's."""
    monkeypatch.setattr(
        "proto_tools.utils.env_warmup._marker_path",
        lambda env_path, fingerprint: tmp_path / f"marker-{fingerprint}",
    )


# ── Deciding whether to warm at all ──────────────────────────────────────────


def test_the_longest_matching_mount_wins(monkeypatch):
    """A container can show a short prefix as local while the real storage is mounted deeper.

    On Sherlock /oak is a local xfs stub inside Apptainer and /oak/stanford is the Lustre
    mount holding the data, so a shortest-match lookup would skip exactly the case this
    module exists for.
    """
    monkeypatch.setattr(
        "proto_tools.utils.env_warmup._mount_table",
        lambda: [("/", "xfs"), ("/oak", "xfs"), ("/oak/stanford", "lustre")],
    )
    assert _filesystem_type(Path("/oak/stanford/groups/x/env")) == "lustre"
    assert _filesystem_type(Path("/oak/other")) == "xfs"
    assert _filesystem_type(Path("/elsewhere")) == "xfs"


@pytest.mark.parametrize(
    "fs_type,expected",
    [
        ("lustre", True),
        ("nfs4", True),
        ("gpfs", True),
        ("fuse.sshfs", True),  # matched by prefix
        ("xfs", False),
        ("ext4", False),
        ("tmpfs", False),
        ("some_future_fs", False),  # unknown means skip, never guess
    ],
)
def test_only_network_filesystems_are_warmed(monkeypatch, fs_type, expected):
    """Only a network filesystem is worth warming.

    Warming reads whole files where mmap reads only the pages touched, so on local storage it
    costs more than it saves.
    """
    monkeypatch.setattr("proto_tools.utils.env_warmup._mount_table", lambda: [("/", fs_type)])
    assert _is_network_filesystem(Path("/env")) is expected


def test_no_mount_table_means_no_warming(monkeypatch):
    """Anywhere /proc/mounts does not exist, skipping is the safe answer."""
    monkeypatch.setattr("proto_tools.utils.env_warmup._mount_table", list)
    assert _is_network_filesystem(Path("/env")) is False


def test_a_local_env_is_left_alone(tmp_path, monkeypatch, fresh_marker):
    """The whole point of the filesystem gate: no reads, no marker, no cost."""
    monkeypatch.setattr("proto_tools.utils.env_warmup._mount_table", lambda: [("/", "ext4")])
    env = _env_with_libraries(tmp_path, libbig=_BIG)
    assert warm_environment(env, "hash") == 0


# ── The opt-out, in both directions ──────────────────────────────────────────


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
def test_the_env_var_can_turn_warming_off(tmp_path, monkeypatch, value, on_lustre, fresh_marker):
    """An operator who does not want this must be able to say so."""
    monkeypatch.setenv("PROTO_WARM_ENV", value)
    env = _env_with_libraries(tmp_path, libbig=_BIG)
    assert warm_environment(env, "hash") == 0


def test_the_env_var_can_force_warming_on_a_local_filesystem(tmp_path, monkeypatch, fresh_marker):
    """Forcing it is how someone tests the path, or overrides a filesystem we classify wrongly."""
    monkeypatch.setattr("proto_tools.utils.env_warmup._mount_table", lambda: [("/", "ext4")])
    monkeypatch.setenv("PROTO_WARM_ENV", "1")
    env = _env_with_libraries(tmp_path, libbig=_BIG)
    assert warm_environment(env, "hash") == _BIG


# ── Choosing what to read ────────────────────────────────────────────────────


def test_only_large_libraries_are_read(tmp_path):
    """A few big CUDA libraries are nearly all the bytes; hundreds of small ones are overhead."""
    env = _env_with_libraries(tmp_path, libbig=_BIG, libsmall=1024)
    assert [path.name for path in _large_shared_libraries(env)] == ["libbig.so"]


def test_libraries_are_read_largest_first(tmp_path):
    """Under a time budget, the biggest files are the ones worth getting in."""
    env = _env_with_libraries(tmp_path, libmid=_BIG * 2, libbig=_BIG * 3, libsmall=_BIG)
    assert [path.name for path in _large_shared_libraries(env)] == ["libbig.so", "libmid.so", "libsmall.so"]


def test_a_symlinked_python_dir_is_not_read_twice(tmp_path):
    """Conda leaves lib/python3.1 -> python3.12, which would otherwise double every library."""
    env = _env_with_libraries(tmp_path, libbig=_BIG)
    (env / "lib" / "python3.1").symlink_to("python3.12")
    assert len(_large_shared_libraries(env)) == 1


def test_an_env_with_no_large_libraries_reads_nothing(tmp_path, on_lustre, fresh_marker):
    """A CPU-only toolkit should not pay for a mechanism aimed at CUDA libraries."""
    env = _env_with_libraries(tmp_path, libsmall=1024)
    assert warm_environment(env, "hash") == 0


# ── Doing it once per node ───────────────────────────────────────────────────


def test_the_second_call_is_a_no_op(tmp_path, on_lustre, fresh_marker):
    """The page cache is already full; re-reading would be pure waste on every dispatch."""
    env = _env_with_libraries(tmp_path, libbig=_BIG)
    assert warm_environment(env, "hash") == _BIG
    assert warm_environment(env, "hash") == 0


def test_a_rebuilt_env_is_warmed_again(tmp_path, on_lustre, fresh_marker):
    """A new fingerprint means new libraries on disk, so the old marker must not suppress it."""
    env = _env_with_libraries(tmp_path, libbig=_BIG)
    assert warm_environment(env, "hash-one") == _BIG
    assert warm_environment(env, "hash-two") == _BIG


def test_the_marker_is_node_local_and_scoped_to_the_user(tmp_path):
    """Shared storage would make one node's warm-up silence every other node's."""
    marker = _marker_path(tmp_path / "env", "hash")
    node_local = {Path("/dev/shm"), Path(tempfile.gettempdir())}  # noqa: S108 -- the paths under test
    assert marker.parent in node_local
    assert str(os.getuid()) in marker.name


def test_a_symlink_planted_at_the_marker_is_not_followed(tmp_path, monkeypatch, on_lustre):
    """The marker lives in a world-writable tmpfs, so it must not become a write primitive."""
    marker = tmp_path / "marker"
    victim = tmp_path / "victim"
    victim.write_text("untouched")
    marker.symlink_to(victim)
    monkeypatch.setattr("proto_tools.utils.env_warmup._marker_path", lambda *_: marker)

    env = _env_with_libraries(tmp_path, libbig=_BIG)
    warm_environment(env, "hash")

    assert victim.read_text() == "untouched"


# ── Never breaking the run it was meant to speed up ──────────────────────────


def test_a_missing_env_is_survivable(tmp_path, on_lustre, fresh_marker):
    """Warming races nothing important; an env that is not there is simply not warmed."""
    assert warm_environment(tmp_path / "absent", "hash") == 0


def test_an_unexpected_error_is_swallowed(tmp_path, monkeypatch, fresh_marker):
    """An optimization must never be the reason a tool fails to start."""
    monkeypatch.setattr(
        "proto_tools.utils.env_warmup._is_network_filesystem",
        lambda _path: (_ for _ in ()).throw(RuntimeError("mount table exploded")),
    )
    assert warm_environment(tmp_path / "env", "hash") == 0


def test_the_time_budget_is_honoured(tmp_path, on_lustre, fresh_marker):
    """A filesystem slow enough to blow the budget must not hold up the run indefinitely."""
    env = _env_with_libraries(tmp_path, libbig=_BIG, libmid=_BIG, libsmall=_BIG)
    assert warm_environment(env, "hash", budget_seconds=0.0) == 0
