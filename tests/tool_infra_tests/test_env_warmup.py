"""tests/tool_infra_tests/test_env_warmup.py.

Tests for the tool-environment shared-library warm-up.

What warming *achieves* cannot be asserted here: it needs a cold node and a filesystem whose page
faults are slow, neither of which exists in a test. The measurements live in the module docstring.
What is tested is what decides whether it runs and what it reads, and that it never raises.
"""

from __future__ import annotations

import mmap
import os
from pathlib import Path

import pytest

from proto_tools.utils import env_warmup
from proto_tools.utils.env_warmup import _large_shared_libraries, warm_environment

_BIG = 16 * 2**20
_SLOW = env_warmup._SLOW_FAULT_BYTES_PER_SECOND / 100
_FAST = env_warmup._SLOW_FAULT_BYTES_PER_SECOND * 100


def _env_with_libraries(tmp_path: Path, **sizes: int) -> Path:
    """A tool env laid out like micromamba's, holding the named libraries in site-packages."""
    torch_lib = tmp_path / "env" / "lib" / "python3.12" / "site-packages" / "torch" / "lib"
    torch_lib.mkdir(parents=True)
    for name, size in sizes.items():
        (torch_lib / f"{name}.so").write_bytes(b"\0" * size)
    return tmp_path / "env"


@pytest.fixture
def faults(monkeypatch):
    """Report the env as uncached, set the fault throughput the probe reports, record what it probed."""
    probed: list[Path] = []
    monkeypatch.setattr(env_warmup, "_resident_fraction", lambda libraries: 0.0)

    def _set(throughput: float) -> list[Path]:
        monkeypatch.setattr(env_warmup, "_fault_throughput", lambda path, pages: probed.append(path) or throughput)
        return probed

    monkeypatch.delenv("PROTO_WARM_ENV", raising=False)
    return _set


@pytest.mark.parametrize(("throughput", "expected"), [(_SLOW, _BIG), (_FAST, 0)])
def test_only_slow_page_faults_are_worth_a_pre_read(tmp_path, faults, throughput, expected):
    """Lustre-slow faults are read ahead; on a fast filesystem the lazy path is already cheaper."""
    faults(throughput)
    assert warm_environment(_env_with_libraries(tmp_path, libbig=_BIG), "hash") == expected


def test_residency_is_read_page_by_page(tmp_path):
    """The kernel reports residency page by page, which the residency checks and the probe window rely on."""
    library = _env_with_libraries(tmp_path, libbig=_BIG + 1) / "lib/python3.12/site-packages/torch/lib/libbig.so"
    pages = env_warmup._page_residency(library)
    if pages is None:
        pytest.skip("mincore is unavailable here")
    assert len(pages) == _BIG // mmap.PAGESIZE + 1


def test_the_library_with_the_most_uncached_pages_is_probed(tmp_path, monkeypatch, faults):
    """Envs hardlink shared libraries, so the largest may be cached by another env while the rest are cold."""
    probed = faults(_FAST)
    env = _env_with_libraries(tmp_path, libbig=_BIG * 2, libmid=_BIG)
    cached = {"libbig.so"}
    monkeypatch.setattr(
        env_warmup, "_page_residency", lambda path: bytes([path.name in cached]) * (path.stat().st_size // 4096)
    )
    warm_environment(env, "hash")
    assert [path.name for path in probed] == ["libmid.so"]


@pytest.mark.parametrize(
    ("value", "throughput", "expected", "probes"), [("0", _SLOW, 0, 0), ("1", _FAST, _BIG, 0), ("auto", _FAST, 0, 1)]
)
def test_the_env_var_overrides_the_measurement(tmp_path, monkeypatch, faults, value, throughput, expected, probes):
    """PROTO_WARM_ENV=0 never reads and =1 always reads, without probing; anything else measures."""
    probed = faults(throughput)
    monkeypatch.setenv("PROTO_WARM_ENV", value)
    assert warm_environment(_env_with_libraries(tmp_path, libbig=_BIG), "hash") == expected
    assert len(probed) == probes


def test_large_libraries_are_chosen_once_each_largest_first(tmp_path):
    """Conda's lib/ counts, small files and symlinks do not, and the biggest come first."""
    env = _env_with_libraries(tmp_path, libmid=_BIG * 2, libsmall=1024)
    (env / "lib" / "libcudnn.so.9").write_bytes(b"\0" * _BIG * 3)
    (env / "lib" / "libcudnn.so").symlink_to("libcudnn.so.9")
    (env / "lib" / "python3.1").symlink_to("python3.12")

    chosen = _large_shared_libraries(env, "hash", deadline=float("inf"))
    assert [path.name for path in chosen] == ["libcudnn.so.9", "libmid.so"]


def test_the_library_list_is_reused_until_the_env_is_rebuilt(tmp_path, monkeypatch):
    """The walk is paid once per build; a new fingerprint walks again."""
    env = _env_with_libraries(tmp_path, libbig=_BIG)
    first = _large_shared_libraries(env, "build-one", deadline=float("inf"))

    walks: list[Path] = []
    monkeypatch.setattr(env_warmup, "_scan_libraries", lambda path, deadline: walks.append(path) or [])
    assert _large_shared_libraries(env, "build-one", deadline=float("inf")) == first
    assert walks == []
    assert _large_shared_libraries(env, "build-two", deadline=float("inf")) == []
    assert walks == [env]

    first[0].unlink()  # e.g. a uv pip install into the env: a listed library that is gone means the list is stale
    _large_shared_libraries(env, "build-one", deadline=float("inf"))
    assert walks == [env, env]


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through read-only permissions")
def test_a_read_only_env_is_still_warmed(tmp_path, faults):
    """A team's read-only env cannot keep the list, so it is walked each time instead."""
    faults(_SLOW)
    env = _env_with_libraries(tmp_path, libbig=_BIG)
    env.chmod(0o555)
    try:
        assert warm_environment(env, "hash") == _BIG
        assert not (env / env_warmup._LIBRARY_LIST).exists()
    finally:
        env.chmod(0o755)


def test_the_time_budget_is_honoured(tmp_path, faults):
    """A filesystem slow enough to blow the budget must not hold up the run."""
    faults(_SLOW)
    env = _env_with_libraries(tmp_path, libbig=_BIG, libmid=_BIG)
    assert warm_environment(env, "hash", budget_seconds=0.0) == 0


@pytest.mark.parametrize("broken", ["missing env", "probe raises"])
def test_a_warm_up_that_cannot_run_is_survivable(tmp_path, monkeypatch, broken):
    """An optimization must never be the reason a tool fails to start."""
    monkeypatch.delenv("PROTO_WARM_ENV", raising=False)
    env = tmp_path / "absent"
    if broken == "probe raises":
        env = _env_with_libraries(tmp_path, libbig=_BIG)
        monkeypatch.setattr(env_warmup, "_fault_throughput", lambda path, pages: (_ for _ in ()).throw(OSError("boom")))
    assert warm_environment(env, "hash") == 0


def test_the_probe_times_pages_that_are_not_cached():
    """A probe that re-timed a window an earlier probe had cached would call a cold env warm."""
    cold_mid = bytes([1] * 500 + [0] * 100 + [1] * 50 + [0] * 300 + [1] * 50)
    start, end = env_warmup._probe_window(cold_mid)
    assert (start, end) == (650, 950), "the longest uncached run, not the cached middle"
    assert env_warmup._probe_window(bytes([1] * 1000)) is None


@pytest.mark.parametrize(("resident", "probes"), [(0.95, 0), (0.5, 1)])
def test_a_mostly_cached_env_is_not_probed(tmp_path, monkeypatch, faults, resident, probes):
    """Residency, not fault speed, answers "already warm": probing it would cache the sample."""
    probed = faults(_SLOW)
    monkeypatch.setattr(env_warmup, "_resident_fraction", lambda libraries: resident)
    warm_environment(_env_with_libraries(tmp_path, libbig=_BIG), "hash")
    assert len(probed) == probes
