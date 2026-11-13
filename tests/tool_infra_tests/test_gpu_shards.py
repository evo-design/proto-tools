"""tests/tool_infra_tests/test_gpu_shards.py.

Splitting a session across GPUs (``tests/_gpu_shards.py``): every test runs exactly once, and a
toolkit never spans two shards.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests import _gpu_shards as s


class _Marker(SimpleNamespace):
    pass


def _item(nodeid: str, toolkit: str | None = None, gpus: int = 0) -> SimpleNamespace:
    markers = [_Marker(args=(gpus,))] if gpus else []
    return SimpleNamespace(
        nodeid=nodeid, _tk=toolkit, iter_markers=lambda name: iter(markers if name == "uses_gpu" else [])
    )


@pytest.fixture(autouse=True)
def _toolkit_from_item():
    with patch("tests._warm_workers.toolkit_of", side_effect=lambda item: item._tk):
        yield


def test_multi_gpu_and_device_manager_tests_are_serial() -> None:
    assert s.is_serial(_item("tests/x.py::t", gpus=2))
    assert s.is_serial(_item("tests/tool_infra_tests/test_device_manager/test_auto_pin.py::t", gpus=1))
    assert not s.is_serial(_item("tests/x.py::t", "esm2", gpus=1))


def test_assignment_is_longest_first_and_deterministic() -> None:
    items = [_item(f"a.py::t{i}", tk) for i, tk in enumerate(["big", "big", "mid", "small", None])]
    weights = {"a.py::t0": 50, "a.py::t1": 50, "a.py::t2": 60, "a.py::t3": 10, "a.py::t4": 5}
    assignment = s.assign_shards(items, 2, weights)
    assert assignment == {"big": 0, "mid": 1, "small": 1, s.NO_TOOLKIT: 1}
    assert s.assign_shards(list(reversed(items)), 2, weights) == assignment


def _config(**options):
    calls = []

    def pytest_deselected(*, items):
        calls.append(items)

    hook = SimpleNamespace(pytest_deselected=pytest_deselected)
    return SimpleNamespace(getoption=lambda name: options.get(name.lstrip("-").replace("-", "_")), hook=hook), calls


def test_phases_and_shards_partition_the_session() -> None:
    items = [
        _item("tests/a.py::esm2_1", "esm2", gpus=1),
        _item("tests/a.py::esm2_2", "esm2", gpus=1),
        _item("tests/b.py::boltz", "boltz2", gpus=1),
        _item("tests/c.py::cpu", None),
        _item("tests/d.py::two_gpu", "esm2", gpus=2),
        _item("tests/tool_infra_tests/test_device_manager/test_x.py::dm", None, gpus=1),
    ]
    picked = {}
    for name, options in {
        "serial": {"gpu_phase": "serial"},
        "s0": {"gpu_phase": "parallel", "shard": "0/2"},
        "s1": {"gpu_phase": "parallel", "shard": "1/2"},
    }.items():
        config, _ = _config(**options)
        chosen = list(items)
        s.select(config, chosen)
        picked[name] = {i.nodeid for i in chosen}
    everything = {i.nodeid for i in items}
    assert picked["serial"] == {"tests/d.py::two_gpu", "tests/tool_infra_tests/test_device_manager/test_x.py::dm"}
    assert picked["serial"] | picked["s0"] | picked["s1"] == everything
    assert not (picked["serial"] & picked["s0"] or picked["serial"] & picked["s1"] or picked["s0"] & picked["s1"])
    # A toolkit lands whole in one shard, even across files.
    assert {"tests/a.py::esm2_1", "tests/a.py::esm2_2"} <= picked["s0"] or {
        "tests/a.py::esm2_1",
        "tests/a.py::esm2_2",
    } <= picked["s1"]


def test_no_phase_selects_everything() -> None:
    config, calls = _config()
    items = [_item("tests/a.py::t", "esm2", gpus=1)]
    s.select(config, items)
    assert len(items) == 1 and calls == []


def test_a_bad_shard_is_a_usage_error() -> None:
    for value in ("2/2", "x/2", "1"):
        with pytest.raises(pytest.UsageError):
            s.parse_shard(value)


def test_junit_durations_map_to_node_ids(tmp_path: Path) -> None:
    (tmp_path / "tests" / "pkg").mkdir(parents=True)
    (tmp_path / "tests" / "pkg" / "test_mod.py").write_text("")
    xml = tmp_path / "run.xml"
    xml.write_text(
        '<testsuites><testsuite name="pytest">'
        '<testcase classname="tests.pkg.test_mod" name="test_a[x]" time="12.5"/>'
        '<testcase classname="tests.pkg.test_mod.TestK" name="test_b" time="3"/>'
        "</testsuite></testsuites>"
    )
    assert s.junit_durations(xml, tmp_path) == {
        "tests/pkg/test_mod.py::test_a[x]": 12.5,
        "tests/pkg/test_mod.py::TestK::test_b": 3.0,
    }
