"""tests/tool_infra_tests/test_warm_workers.py.

The suite's run ordering and its session pool of warm workers (``tests/_warm_workers.py``).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests import _warm_workers as w

# ── The pool ──────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_cache():
    """Stand-in for ToolInstance's cache: get() creates and caches, shutdown_instance() removes."""
    cache: dict[str, MagicMock] = {}

    def get(toolkit, **_):
        return cache.setdefault(toolkit, MagicMock(name=toolkit))

    def shutdown_instance(key):
        inst = cache.pop(key, None)
        if inst is not None:
            inst.shutdown()

    with (
        patch("proto_tools.utils.tool_instance.ToolInstance.get", side_effect=get),
        patch("proto_tools.utils.tool_instance.ToolInstance.shutdown_instance", side_effect=shutdown_instance),
        patch("proto_tools.utils.tool_instance._lookup_instance", side_effect=cache.get),
    ):
        yield cache


def test_pool_reuses_a_held_worker(fake_cache) -> None:
    pool = w.WarmWorkers()
    pool.acquire("esm2", "t1")
    first = fake_cache["esm2"]
    pool.acquire("esm2", "t2")
    assert fake_cache["esm2"] is first
    assert pool.report() == {"esm2": {"started": 1, "reused": 1}}


def test_pool_releases_what_the_next_test_does_not_use(fake_cache) -> None:
    pool = w.WarmWorkers()
    pool.acquire("esm2", "t1")
    pool.acquire("esm3", "t1")
    esm2 = fake_cache["esm2"]
    pool.release_except({"esm3"}, "t2")
    esm2.shutdown.assert_called_once()
    assert set(fake_cache) == {"esm3"}
    pool.release_except(set(), None)
    assert fake_cache == {}
    assert pool.report()["esm2"]["released"] == 1


def test_pool_starts_again_after_something_else_stopped_the_worker(fake_cache) -> None:
    """A test that calls ToolInstance.clear_all() (or resets DeviceManager) must not leave the pool stale."""
    pool = w.WarmWorkers()
    pool.acquire("esm2", "t1")
    stale = fake_cache.pop("esm2")  # what clear_all does
    pool.acquire("esm2", "t2")
    assert fake_cache["esm2"] is not stale
    assert pool.report()["esm2"] == {"started": 2, "lost": 1}


def test_releasing_a_replaced_worker_stops_only_the_pools_own(fake_cache) -> None:
    pool = w.WarmWorkers()
    pool.acquire("esm2", "t1")
    ours = fake_cache.pop("esm2")
    theirs = fake_cache.setdefault("esm2", MagicMock(name="theirs"))  # a test registered its own
    pool.release_except(set(), None)
    ours.shutdown.assert_called_once()
    theirs.shutdown.assert_not_called()


# ── Ordering ──────────────────────────────────────────────────────────────────


def _item(nodeid: str, toolkit: str | None, group: tuple = (), opted: bool = True) -> SimpleNamespace:
    return SimpleNamespace(nodeid=nodeid, _tk=toolkit, _group=group, _opted=opted)


@pytest.fixture
def fake_classifier():
    with (
        patch.object(w, "toolkit_of", side_effect=lambda item: item._tk),
        patch.object(w, "restart_group", side_effect=lambda item, tk: item._group),
        patch.object(w, "warm_eligible", side_effect=lambda item: item._opted),
    ):
        yield


def test_order_groups_toolkits_and_restart_groups_and_keeps_everything_else(fake_classifier) -> None:
    items = [
        _item("a.py::cpu1", None),
        _item("a.py::esm2_big", "esm2", (("ckpt", "big"),)),
        _item("b.py::boltz", "boltz2"),
        _item("a.py::esm2_small", "esm2", (("ckpt", "small"),)),
        _item("c.py::cpu2", None),
        _item("d.py::esm2_big_again", "esm2", (("ckpt", "big"),)),
        _item("e.py::seed_esm2", "esm2", (("ckpt", "big"),), opted=False),
        _item("f.py::esm2_big_late", "esm2", (("ckpt", "big"),)),
    ]
    before = sorted(i.nodeid for i in items)
    w.order_items(items)
    assert [i.nodeid for i in items] == [
        "a.py::cpu1",
        "c.py::cpu2",  # tests that run no tool keep their order, first
        "a.py::esm2_big",  # esm2's first restart group, opted-in files first, in file order
        "d.py::esm2_big_again",
        "f.py::esm2_big_late",
        "e.py::seed_esm2",  # a file that did not opt in comes after, so it never sees the warm worker
        "a.py::esm2_small",
        "b.py::boltz",
    ]
    assert sorted(i.nodeid for i in items) == before  # reordered, never added or dropped


# ── What a test runs ──────────────────────────────────────────────────────────


def test_a_tool_key_parameter_names_the_toolkit() -> None:
    from proto_tools.tools.tool_registry import ToolRegistry

    spec = ToolRegistry.get("esm2-score")
    item = SimpleNamespace(
        get_closest_marker=lambda name: None,
        callspec=SimpleNamespace(params={"tool_spec": spec}),
    )
    assert w.toolkit_of(item) == "esm2"
    item.callspec.params = {"key": "esm2-score"}
    assert w.toolkit_of(item) == "esm2"


def test_restart_group_reads_parameters_defaults_and_literals() -> None:
    def run_esm2_score_case():
        from proto_tools.tools.masked_models.esm2.esm2_score import ESM2ScoringConfig, run_esm2_score

        run_esm2_score(None, ESM2ScoringConfig(model_checkpoint="esm2_t6_8M_UR50D"))

    item = SimpleNamespace(
        get_closest_marker=lambda name: None, callspec=None, function=run_esm2_score_case, module=None
    )
    assert w.restart_group(item, "esm2") == (("model_checkpoint", "'esm2_t6_8M_UR50D'"),)

    item = SimpleNamespace(
        get_closest_marker=lambda name: None,
        callspec=SimpleNamespace(params={"checkpoint": "esm2_t12_35M_UR50D"}),
        function=lambda: None,
        module=None,
    )
    assert w.restart_group(item, "esm2") == (("model_checkpoint", "'esm2_t12_35M_UR50D'"),)


def test_a_worker_config_marker_wins() -> None:
    marker = SimpleNamespace(args=(), kwargs={"model_checkpoint": "evo2_20b"})
    item = SimpleNamespace(get_closest_marker=lambda name: marker if name == "worker_config" else None)
    assert w.restart_group(item, "evo2") == (("model_checkpoint", "'evo2_20b'"),)
