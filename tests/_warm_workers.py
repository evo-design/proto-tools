"""tests/_warm_workers.py.

Run each toolkit's tests together, and keep its persistent worker warm across test files.

Loading a model is most of what a GPU test costs. Files that opt into persistence
(``make_persistent_fixture``, or ``WARM_WORKERS = True`` for a file that serves several toolkits)
used to keep a worker only until the file ended, and a test that changed a ``reload_on_change``
field (a different checkpoint, say) restarted it. This plugin does two things:

1. **Order.** After collection, tool tests are sorted by toolkit, then by restart group (the
   ``reload_on_change`` values the test runs with), then by file, keeping their original order
   otherwise. Tests with no toolkit keep their order and run first. Parametrized tests are not
   changed; only where each case runs.
2. **Keep warm.** Opted-in files take their worker from a session-level pool. After each test the
   pool looks at the next one: a worker it will use stays, and every other one is released, so a
   toolkit's worker lives across files for as long as its tests run back to back. A held worker is
   only ever visible to tests in opted-in files, as before, so tests that expect a fresh process
   (the seed tests' non-persistent variant, say) still get one.

``--random-order`` shuffles instead, and ``--no-toolkit-order`` keeps collection order (the pool
still releases a worker as soon as the next test does not use it). Every decision is logged under
``proto_tools.tests.warm_workers``; ``PROTO_WARM_WORKERS_REPORT`` names a JSON file for a per-toolkit
summary of what was started, reused and released.
"""

from __future__ import annotations

import ast
import functools
import inspect
import json
import logging
import os
import textwrap
import typing
from collections import Counter, defaultdict
from typing import Any

import pytest

logger = logging.getLogger("proto_tools.tests.warm_workers")

PLUGIN_NAME = "warm_workers"

#: module ``__name__`` -> toolkits its ``make_persistent_fixture`` calls keep warm.
_MODULE_TOOLKITS: dict[str, list[str]] = defaultdict(list)


def register_module_toolkit(module_name: str, toolkit: str) -> None:
    """Record that ``module_name`` keeps ``toolkit`` warm (called by ``make_persistent_fixture``)."""
    if toolkit not in _MODULE_TOOLKITS[module_name]:
        _MODULE_TOOLKITS[module_name].append(toolkit)


# ── What a test runs ──────────────────────────────────────────────────────────


@functools.cache
def _registry() -> dict[str, Any]:
    from proto_tools.tools.tool_registry import ToolRegistry

    return {spec.key: spec for spec in ToolRegistry.list_all()}


def _toolkit_of_spec(spec: Any) -> str:
    return spec.source_file.parent.name


@functools.cache
def _toolkit_names() -> frozenset[str]:
    return frozenset(_toolkit_of_spec(s) for s in _registry().values())


@functools.cache
def _function_toolkits() -> dict[str, list[Any]]:
    """Tool function name (``run_esm2_score``) -> the specs it runs."""
    by_name: dict[str, list[Any]] = defaultdict(list)
    for spec in _registry().values():
        name = getattr(spec.function, "__name__", None)
        if name:
            by_name[name].append(spec)
    return by_name


def _toolkit_of_value(value: Any) -> str | None:
    from proto_tools.tools.tool_registry import ToolSpec

    if isinstance(value, ToolSpec):
        return _toolkit_of_spec(value)
    if isinstance(value, str):
        spec = _registry().get(value)
        if spec is not None:
            return _toolkit_of_spec(spec)
        if value in _toolkit_names():
            return value
    return None


@functools.cache
def _source_names_and_literals(func: Any) -> tuple[frozenset[str], frozenset[str]]:
    """The bare names and string literals a test function's body uses."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    except (OSError, TypeError, SyntaxError):
        return frozenset(), frozenset()
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    literals = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    return frozenset(names), frozenset(literals)


def _source_specs(item: pytest.Item) -> list[Any]:
    """Specs whose tool function the test body calls by name."""
    func = getattr(item, "function", None)
    if func is None:
        return []
    names, _ = _source_names_and_literals(func)
    by_name = _function_toolkits()
    return [spec for name in names for spec in by_name.get(name, ())]


def warm_eligible(item: pytest.Item) -> bool:
    """Whether the test's file opted into persistent workers."""
    module = getattr(item, "module", None)
    if module is None:
        return False
    return bool(_MODULE_TOOLKITS.get(module.__name__)) or bool(getattr(module, "WARM_WORKERS", False))


def toolkit_of(item: pytest.Item) -> str | None:
    """The toolkit a test runs, or None for a test that runs no tool worker.

    In order: a ``toolkit`` marker; a ``ToolSpec``, tool key or toolkit name among its parameters;
    the toolkit its file keeps warm (when there is exactly one); the toolkit of the tool functions
    its body calls (when they all belong to one). The last two only for tests that opted into
    persistence or use a GPU, so CPU unit tests keep their order.
    """
    marker = item.get_closest_marker("toolkit")
    if marker is not None and marker.args:
        return str(marker.args[0])
    callspec = getattr(item, "callspec", None)
    if callspec is not None:
        for value in callspec.params.values():
            toolkit = _toolkit_of_value(value)
            if toolkit is not None:
                return toolkit
    if not (warm_eligible(item) or item.get_closest_marker("uses_gpu") is not None):
        return None
    module = getattr(item, "module", None)
    registered = _MODULE_TOOLKITS.get(module.__name__, []) if module is not None else []
    if len(registered) == 1:
        return registered[0]
    toolkits = {_toolkit_of_spec(spec) for spec in _source_specs(item)}
    if registered:
        toolkits &= set(registered)
    return toolkits.pop() if len(toolkits) == 1 else None


# ── Which worker configuration it needs ──────────────────────────────────────


def _literal_choices(annotation: Any) -> tuple[Any, ...]:
    choices: list[Any] = []
    for arg in (annotation, *typing.get_args(annotation)):
        if typing.get_origin(arg) is typing.Literal:
            choices.extend(typing.get_args(arg))
    return tuple(choices)


def _field_default(info: Any) -> Any:
    from pydantic_core import PydanticUndefined

    return None if info.default is PydanticUndefined else info.default


@functools.cache
def _reload_choices(toolkit: str) -> dict[str, frozenset[Any]]:
    """``reload_on_change`` field -> its known values over every config in ``toolkit``.

    A ``Literal`` field's allowed values; for any other field, the defaults its tools declare
    (caliby's pack and sample checkpoints, say).
    """
    choices: dict[str, set[Any]] = defaultdict(set)
    for spec in _registry().values():
        if _toolkit_of_spec(spec) != toolkit:
            continue
        model = spec.config_model
        fields = model.reload_fields() if hasattr(model, "reload_fields") else None
        for name in fields or ():
            info = model.model_fields.get(name)
            if info is None:
                continue
            literal = _literal_choices(info.annotation)
            choices[name].update(literal or [v for v in [_field_default(info)] if isinstance(v, str)])
    return {name: frozenset(values) for name, values in choices.items()}


def _reload_defaults(spec: Any) -> dict[str, Any]:
    """The ``reload_on_change`` defaults of one tool's config."""
    model = spec.config_model
    fields = model.reload_fields() if hasattr(model, "reload_fields") else None
    defaults: dict[str, Any] = {}
    for name in fields or ():
        info = model.model_fields.get(name)
        if info is not None and (default := _field_default(info)) is not None:
            defaults[name] = default
    return defaults


def restart_group(item: pytest.Item, toolkit: str) -> tuple[tuple[str, str], ...]:
    """The ``reload_on_change`` values a test runs ``toolkit`` with, as far as collection can tell.

    A ``worker_config(field=value)`` marker wins. Otherwise: the file's ``worker_config_for(params)``
    hook, when it defines one; the defaults of the tools the test body calls; any parameter or
    string literal that is one of the field's allowed choices. Two tests in the same group can
    share a worker without a restart.
    """
    marker = item.get_closest_marker("worker_config")
    if marker is not None:
        return tuple(sorted((k, repr(v)) for k, v in marker.kwargs.items()))
    choices = _reload_choices(toolkit)
    values: dict[str, Any] = {}
    for spec in _source_specs(item):
        if _toolkit_of_spec(spec) == toolkit:
            values.update(_reload_defaults(spec))
    callspec = getattr(item, "callspec", None)
    params = dict(callspec.params) if callspec is not None else {}
    func = getattr(item, "function", None)
    literals = _source_names_and_literals(func)[1] if func is not None else frozenset()
    for field, allowed in choices.items():
        for candidate in [*params.values(), *sorted(literals)]:
            try:
                if candidate in allowed:
                    values[field] = candidate
                    break
            except TypeError:
                continue
    hook = getattr(getattr(item, "module", None), "worker_config_for", None)
    if callable(hook):
        values.update(hook(params))
    return tuple(sorted((k, repr(v)) for k, v in values.items()))


# ── Ordering ──────────────────────────────────────────────────────────────────


def order_items(items: list[pytest.Item]) -> None:
    """Sort tool tests by toolkit, restart group and file; keep everything else first, in order."""
    toolkit_rank: dict[str, int] = {}
    group_rank: dict[tuple[str, tuple], int] = {}
    module_rank: dict[str, int] = {}
    keyed = []
    for index, item in enumerate(items):
        module_rank.setdefault(item.nodeid.split("::", 1)[0], len(module_rank))
        toolkit = toolkit_of(item)
        if toolkit is None:
            keyed.append(((0, index), item))
            continue
        group = restart_group(item, toolkit)
        t = toolkit_rank.setdefault(toolkit, len(toolkit_rank))
        g = group_rank.setdefault((toolkit, group), len(group_rank))
        # Opted-in files first within a group, so their worker carries straight across them.
        opted = 0 if warm_eligible(item) else 1
        keyed.append(((1, t, g, opted, module_rank[item.nodeid.split("::", 1)[0]], index), item))
    keyed.sort(key=lambda pair: pair[0])
    items[:] = [item for _, item in keyed]
    logger.debug("ordered %d tests into %d toolkits, %d restart groups", len(items), len(toolkit_rank), len(group_rank))


# ── The pool ──────────────────────────────────────────────────────────────────


class WarmWorkers:
    """Session-level persistent workers, kept while consecutive tests use them."""

    def __init__(self) -> None:
        self._held: dict[str, Any] = {}
        self.stats: dict[str, Counter] = defaultdict(Counter)

    def acquire(self, toolkit: str, nodeid: str) -> None:
        """Make sure ``toolkit`` has a cached persistent instance for the coming test."""
        from proto_tools.utils.tool_instance import ToolInstance, _lookup_instance

        held = self._held.get(toolkit)
        cached = _lookup_instance(toolkit)
        if held is not None and cached is held:
            self.stats[toolkit]["reused"] += 1
            logger.info("warm workers: keeping %s warm for %s", toolkit, nodeid)
            return
        if held is not None:
            # Something else stopped it (clear_all, a reset DeviceManager): start over.
            self.stats[toolkit]["lost"] += 1
            logger.info("warm workers: %s was stopped outside the pool; starting it again", toolkit)
        instance = ToolInstance.get(toolkit)
        self._held[toolkit] = instance
        self.stats[toolkit]["started"] += 1
        logger.info("warm workers: starting %s for %s", toolkit, nodeid)

    def release(self, toolkit: str, reason: str) -> None:
        from proto_tools.utils.tool_instance import ToolInstance, _lookup_instance

        instance = self._held.pop(toolkit, None)
        if instance is None:
            return
        if _lookup_instance(toolkit) is instance:
            ToolInstance.shutdown_instance(toolkit)
        else:
            instance.shutdown()
        self.stats[toolkit]["released"] += 1
        logger.info("warm workers: releasing %s (%s)", toolkit, reason)

    def release_except(self, keep: set[str], next_nodeid: str | None) -> None:
        for toolkit in list(self._held):
            if toolkit in keep:
                continue
            reason = f"next test {next_nodeid} does not use it" if next_nodeid else "end of session"
            self.release(toolkit, reason)

    def report(self) -> dict[str, dict[str, int]]:
        return {toolkit: dict(counter) for toolkit, counter in sorted(self.stats.items())}


def wanted_toolkits(item: pytest.Item | None) -> set[str]:
    """The toolkits the pool keeps for ``item``: its file's, and its own in a WARM_WORKERS file."""
    if item is None or not warm_eligible(item):
        return set()
    module = item.module
    wanted = set(_MODULE_TOOLKITS.get(module.__name__, ()))
    if getattr(module, "WARM_WORKERS", False):
        toolkit = toolkit_of(item)
        if toolkit is not None:
            wanted.add(toolkit)
    return wanted


class WarmWorkersPlugin:
    """Orders the session and owns the pool; registered by ``tests/conftest.py``."""

    def __init__(self, config: pytest.Config) -> None:
        self.config = config
        self.pool = WarmWorkers()

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config: pytest.Config, items: list[pytest.Item]) -> None:
        if config.getoption("--random-order") or config.getoption("--no-toolkit-order"):
            return
        order_items(items)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_teardown(self, item: pytest.Item, nextitem: pytest.Item | None):
        yield
        self.pool.release_except(wanted_toolkits(nextitem), nextitem.nodeid if nextitem else None)

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        self.pool.release_except(set(), None)
        path = os.environ.get("PROTO_WARM_WORKERS_REPORT")
        if path:
            with open(path, "w") as fh:
                json.dump(self.pool.report(), fh, indent=2, sort_keys=True)


def pool(config: pytest.Config) -> WarmWorkers | None:
    plugin = config.pluginmanager.get_plugin(PLUGIN_NAME)
    return plugin.pool if plugin is not None else None
