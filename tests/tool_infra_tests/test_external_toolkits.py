"""tests/tool_infra_tests/test_external_toolkits.py.

Tests for out-of-tree toolkits: tools defined by a downstream package, outside ``proto_tools/tools/``.

Each test writes a throwaway package to ``tmp_path`` (``<pkg>/tools/testing/<toolkit>/``) whose tool reuses the
``mock_cpu_tool`` models and its stdlib-only ``standalone/``, then imports it. End-to-end tests seed a ready env
(``bin/python`` linked to this interpreter plus a matching ``STATUS.txt``), so no environment is ever built.
"""

import hashlib
import importlib
import os
import shutil
import sys
import uuid
from pathlib import Path

import pytest

import proto_tools.tools.testing.mock_cpu_tool.mock_cpu_tool as mock_cpu_tool
from proto_tools.mcp.tools import runs_in_process
from proto_tools.tools.testing.mock_cpu_tool.mock_cpu_tool import MockCPUToolInput
from proto_tools.tools.tool_registry import TOOLS_DIR, ToolRegistry
from proto_tools.utils import ToolInstance

_MOCK_STANDALONE = Path(mock_cpu_tool.__file__).parent / "standalone"

_TOOL_MODULE = '''"""Out-of-tree test toolkit."""

from typing import Any

from proto_tools.tools.testing.mock_cpu_tool.mock_cpu_tool import (
    MockCPUToolConfig,
    MockCPUToolInput,
    MockCPUToolOutput,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ToolInstance


class ExtToolConfig(MockCPUToolConfig):
    """Config for the out-of-tree test toolkit."""

    @property
    def cpus_per_instance(self) -> int | None:
        """Run in a single worker rather than fanning out."""
        return None


@tool(
    key={key!r},
    label="External Test Tool",
    category="testing",
    input_class=MockCPUToolInput,
    config_class=ExtToolConfig,
    output_class=MockCPUToolOutput,
    description="Out-of-tree test tool",
    iterable_input_fields=["items"],
    iterable_output_field="results",
    max_chunk_size=None,
    local_only={local_only!r},
)
def run_ext(inputs: MockCPUToolInput, config: ExtToolConfig, instance: Any = None) -> MockCPUToolOutput:
    """Dispatch to this toolkit's standalone worker."""
    result = ToolInstance.dispatch(
        {toolkit!r}, {{"items": list(inputs.items), "device": config.device}}, instance=instance, config=config
    )
    return MockCPUToolOutput(**result)
'''


@pytest.fixture
def make_toolkit(tmp_path, monkeypatch):
    """Factory that writes and imports an out-of-tree toolkit; teardown unregisters everything it made."""
    monkeypatch.syspath_prepend(str(tmp_path))
    made: list[tuple[str, str]] = []

    def _make(
        toolkit: str = "ext_test_tk",
        *,
        key: str | None = None,
        package: str | None = None,
        standalone: bool = True,
        local_only: str | None = None,
        in_init: bool = False,
        standalone_files: tuple[str, ...] = (),
    ):
        package = package or f"extpkg_{uuid.uuid4().hex[:8]}"
        key = key or f"{toolkit.replace('_', '-')}-{uuid.uuid4().hex[:6]}"
        made.append((package, toolkit))

        # Package skeleton: <pkg>/tools/testing/<toolkit>/<toolkit>.py, or the toolkit's __init__.py with in_init
        package_dir = tmp_path / package
        toolkit_dir = package_dir / "tools" / "testing" / toolkit
        toolkit_dir.mkdir(parents=True)
        for pkg_dir in (package_dir, package_dir / "tools", package_dir / "tools" / "testing", toolkit_dir):
            (pkg_dir / "__init__.py").touch()
        tool_file = toolkit_dir / ("__init__.py" if in_init else f"{toolkit}.py")
        tool_file.write_text(_TOOL_MODULE.format(key=key, toolkit=toolkit, local_only=local_only))
        if standalone:
            shutil.copytree(_MOCK_STANDALONE, toolkit_dir / "standalone", ignore=shutil.ignore_patterns("__pycache__"))
            for name in standalone_files:
                (toolkit_dir / "standalone" / name).touch()
        module = f"{package}.tools.testing.{toolkit}"
        # A second toolkit in an already-imported package is a new entry in a directory the import system has listed.
        importlib.invalidate_caches()
        return importlib.import_module(module if in_init else f"{module}.{toolkit}")

    yield _make

    # Unregister everything the factory made, so session-wide list_all() checks never see it.
    packages = {package for package, _ in made}
    for key in [k for k, s in ToolRegistry._registry.items() if s.function.__module__.split(".")[0] in packages]:
        del ToolRegistry._registry[key]
    for _, toolkit in made:
        ToolInstance.shutdown_instance(toolkit)
        ToolInstance._build_failures.pop(toolkit, None)
    for name in [name for name in sys.modules if name.split(".")[0] in packages]:
        del sys.modules[name]


def _spec(module):
    """Return the registered spec for a test toolkit module's tool."""
    return next(spec for spec in ToolRegistry._registry.values() if spec.function.__module__ == module.__name__)


def _seed_env(toolkit: str) -> ToolInstance:
    """Mark the toolkit's env as built against this interpreter, so dispatch skips setup."""
    instance = ToolInstance(toolkit)
    (instance.env_path / "bin").mkdir(parents=True)
    (instance.env_path / "bin" / "python").symlink_to(sys.executable)
    (instance.env_path / "STATUS.txt").write_text(f"SUCCESS\nSetup hash: {instance._setup_hash()}\n")
    return instance


@pytest.fixture
def proto_home(tmp_path, monkeypatch):
    """Point PROTO_HOME at a private dir so seeded envs never touch the real env root."""
    home = tmp_path / "home"
    monkeypatch.setenv("PROTO_HOME", str(home))
    return home


# ── Registration ─────────────────────────────────────────────────────────────


def test_packaged_tool_is_not_external():
    spec = ToolRegistry.get("mock-cpu-tool-run")
    assert spec.is_external is False
    assert spec.toolkit_dir == TOOLS_DIR / "testing" / "mock_cpu_tool"


def test_external_toolkit_is_external_and_local_only(make_toolkit):
    """An out-of-tree toolkit with an environment can only run locally, and says so."""
    spec = _spec(make_toolkit())
    assert spec.is_external and spec.has_standalone_env
    assert spec.local_only.startswith(f"{spec.key} is defined outside proto_tools")
    assert runs_in_process(spec.key)


def test_external_toolkit_keeps_its_own_local_only_reason(make_toolkit):
    spec = _spec(make_toolkit(local_only="reads a licence file from local disk."))
    assert spec.local_only == "reads a licence file from local disk."


def test_external_in_process_tool_gets_no_local_only_default(make_toolkit):
    """Without a standalone/ env, an out-of-tree tool keeps the packaged-tool routing rules."""
    assert _spec(make_toolkit(standalone=False)).local_only is None


def test_external_toolkit_named_like_a_packaged_toolkit_is_refused(make_toolkit):
    """Sharing a packaged toolkit's name would share its worker and env, so registration refuses."""
    with pytest.raises(ValueError, match=r"same name as the packaged toolkit .*mock_cpu_tool"):
        make_toolkit("mock_cpu_tool", key="ext-shadow-run")
    assert "ext-shadow-run" not in ToolRegistry._registry


def test_two_external_toolkits_with_one_name_are_refused(make_toolkit):
    first = _spec(make_toolkit("ext_dup_tk"))
    with pytest.raises(ValueError, match=rf"same name as .*defines {first.key!r}"):
        make_toolkit("ext_dup_tk")
    assert first.key in ToolRegistry._registry


def test_external_toolkit_with_a_standalone_helpers_copy_is_refused(make_toolkit):
    """A local standalone_helpers file would shadow the helpers every worker is given, so registration refuses."""
    with pytest.raises(ValueError, match=r"standalone_helpers\.py, a name reserved"):
        make_toolkit("ext_helpers_tk", key="ext-helpers-run", standalone_files=("standalone_helpers.py",))
    assert "ext-helpers-run" not in ToolRegistry._registry


def test_external_in_process_tool_may_share_a_packaged_name(make_toolkit):
    """Name-keyed state only exists for toolkits with an environment, so the rule applies only to those."""
    spec = _spec(make_toolkit("mock_cpu_tool", standalone=False))
    assert spec.is_external and not spec.has_standalone_env


# ── Toolkit directories and environment naming ───────────────────────────────


def test_external_toolkit_dir_resolves_while_registered(make_toolkit):
    """An out-of-tree toolkit resolves from the registry, outside the cached packaged scan."""
    spec = _spec(make_toolkit("ext_dirs_tk"))
    assert ToolInstance._find_tool_dir("ext_dirs_tk") == spec.toolkit_dir
    assert "ext_dirs_tk" not in ToolInstance._get_tool_dirs()
    assert ToolInstance._find_tool_dir("mock_cpu_tool") == TOOLS_DIR / "testing" / "mock_cpu_tool"

    del ToolRegistry._registry[spec.key]
    assert ToolInstance._find_tool_dir("ext_dirs_tk") is None


def test_external_env_name_is_namespaced_by_package(make_toolkit):
    """The env is keyed by the defining package, so same-named toolkits from two packages never share one."""
    module = make_toolkit("ext_env_tk", package="extpkg_envname")
    _, env_name = ToolInstance._resolve_env_def("ext_env_tk")
    digest = hashlib.sha256(b"extpkg_envname.tools.testing.ext_env_tk").hexdigest()[:8]
    assert env_name == f"ext_env_tk__ext_{digest}"
    assert ToolInstance._resolve_env_def("ext_env_tk") == (_spec(module).toolkit_dir / "standalone", env_name)


def test_external_env_name_is_the_same_for_a_tool_in_the_toolkit_init(make_toolkit):
    """A tool defined in the toolkit's __init__.py hashes the toolkit package, like one in a sibling module."""
    make_toolkit("ext_init_tk", package="extpkg_init", in_init=True)
    digest = hashlib.sha256(b"extpkg_init.tools.testing.ext_init_tk").hexdigest()[:8]
    assert ToolInstance._resolve_env_def("ext_init_tk")[1] == f"ext_init_tk__ext_{digest}"


def test_standalone_dir_override_applies_to_external_toolkits(make_toolkit, tmp_path, monkeypatch):
    make_toolkit("ext_override_tk")
    override = tmp_path / "override"
    shutil.copytree(_MOCK_STANDALONE, override, ignore=shutil.ignore_patterns("__pycache__", "*.py"))
    monkeypatch.setenv("PROTO_EXT_OVERRIDE_TK_STANDALONE_DIR", str(override))
    env_def_dir, env_name = ToolInstance._resolve_env_def("ext_override_tk")
    assert env_def_dir == override.resolve()
    assert env_name.startswith("ext_override_tk__override_")


def _join_shared_env(module, shared_name: str) -> Path:
    """Swap a test toolkit's private env definition for a ``shared_env.txt`` naming ``shared_name``."""
    standalone = _spec(module).toolkit_dir / "standalone"
    for name in ("setup.sh", "python_version.txt"):
        (standalone / name).unlink()
    (standalone / "shared_env.txt").write_text(f"{shared_name}\n")
    return standalone.parents[3]


def _make_package_shared_env(package_dir: Path, shared_name: str) -> Path:
    """Write a shared env definition into a test package's ``shared_envs/``."""
    env_def = package_dir / "shared_envs" / shared_name
    shutil.copytree(_MOCK_STANDALONE, env_def, ignore=shutil.ignore_patterns("__pycache__", "*.py"))
    return env_def


def test_external_toolkit_can_use_a_packaged_shared_env(make_toolkit):
    """An out-of-tree toolkit joining a packaged shared env builds that env, under its packaged name."""
    _join_shared_env(make_toolkit("ext_shared_tk"), "glm2_base")
    env_def_dir, env_name = ToolInstance._resolve_env_def("ext_shared_tk")
    assert env_name == "glm2_base"
    assert env_def_dir == ToolInstance._shared_envs_root() / "glm2_base"


def test_package_shared_envs_are_shared_within_a_package_and_separate_across_packages(make_toolkit):
    """Toolkits in one package share its shared env; another package's same-named shared env is a separate env."""
    package_dir = _join_shared_env(make_toolkit("ext_pkg_shared_a", package="extpkg_shared"), "ext_test_shared")
    _join_shared_env(make_toolkit("ext_pkg_shared_b", package="extpkg_shared"), "ext_test_shared")
    other_dir = _join_shared_env(make_toolkit("ext_pkg_shared_c", package="extpkg_shared_other"), "ext_test_shared")
    env_def = _make_package_shared_env(package_dir, "ext_test_shared")
    other_def = _make_package_shared_env(other_dir, "ext_test_shared")

    env_def_a, env_name_a = ToolInstance._resolve_env_def("ext_pkg_shared_a")
    assert env_def_a == env_def
    assert env_name_a.startswith("ext_test_shared__ext_")
    assert ToolInstance._resolve_env_def("ext_pkg_shared_b") == (env_def, env_name_a)
    env_def_c, env_name_c = ToolInstance._resolve_env_def("ext_pkg_shared_c")
    assert env_def_c == other_def
    assert env_name_c.startswith("ext_test_shared__ext_")
    assert env_name_c != env_name_a


def test_external_shared_env_name_collision_with_a_packaged_one_raises(make_toolkit):
    """A package shared env named like a packaged one is ambiguous, so resolving it fails naming both dirs."""
    package_dir = _join_shared_env(make_toolkit("ext_shared_clash_tk"), "glm2_base")
    env_def = _make_package_shared_env(package_dir, "glm2_base")
    with pytest.raises(ValueError, match=rf"(?s){env_def}.*{ToolInstance._shared_envs_root() / 'glm2_base'}"):
        ToolInstance._resolve_env_def("ext_shared_clash_tk")


def test_borrowed_env_follows_the_toolkits_override(make_toolkit, tmp_path, monkeypatch):
    """A user override of the borrowed toolkit's env applies to every toolkit running in it."""
    _join_shared_env(make_toolkit("ext_borrow_override_tk"), "toolkit:mock_cpu_tool")
    override = tmp_path / "override"
    shutil.copytree(_MOCK_STANDALONE, override, ignore=shutil.ignore_patterns("__pycache__", "*.py"))
    monkeypatch.setenv("PROTO_MOCK_CPU_TOOL_STANDALONE_DIR", str(override))
    env_def_dir, env_name = ToolInstance._resolve_env_def("ext_borrow_override_tk")
    assert env_def_dir == override.resolve()
    assert (env_def_dir, env_name) == ToolInstance._resolve_env_def("mock_cpu_tool")


def test_borrowing_an_unknown_toolkits_env_raises(make_toolkit):
    """A ``toolkit:`` reference to a name proto-tools does not ship fails, listing the valid toolkits."""
    _join_shared_env(make_toolkit("ext_borrow_unknown_tk"), "toolkit:not_a_toolkit")
    with pytest.raises(ValueError, match="unknown toolkit 'not_a_toolkit'"):
        ToolInstance._resolve_env_def("ext_borrow_unknown_tk")


# ── Dispatch ─────────────────────────────────────────────────────────────────


def test_external_toolkit_refuses_a_remote_device(make_toolkit):
    module = make_toolkit()
    with pytest.raises(ValueError, match="defined outside proto_tools"):
        module.run_ext(MockCPUToolInput(items=[1]), module.ExtToolConfig(device="modal"))


def test_external_toolkit_runs_one_shot(make_toolkit, proto_home, monkeypatch):
    """Each call starts a fresh worker from the toolkit's standalone/ in its namespaced env."""
    module = make_toolkit("ext_oneshot_tk")
    instance = _seed_env("ext_oneshot_tk")
    assert instance.env_path.parent == proto_home / "proto_tool_envs"
    monkeypatch.setattr(ToolInstance, "_create_env", lambda self: pytest.fail("seeded env was rebuilt"))

    config = module.ExtToolConfig(device="cpu")
    first = module.run_ext(MockCPUToolInput(items=[1, 2]), config)
    second = module.run_ext(MockCPUToolInput(items=[3]), config)
    assert [r.item for r in first.results] == [1, 2]
    assert len({r.pid for r in first.results}) == 1
    assert first.results[0].pid != second.results[0].pid
    assert first.results[0].pid != os.getpid()


def test_external_toolkit_runs_in_a_borrowed_env(make_toolkit, proto_home, monkeypatch):
    """The worker runs the out-of-tree toolkit's script in the borrowed env, building nothing."""
    module = make_toolkit("ext_borrow_run_tk")
    _join_shared_env(module, "toolkit:mock_cpu_tool")
    seeded = _seed_env("mock_cpu_tool")
    monkeypatch.setattr(ToolInstance, "_create_env", lambda self: pytest.fail("borrowed env was rebuilt"))

    instance = ToolInstance("ext_borrow_run_tk")
    assert instance.env_path == seeded.env_path
    assert instance.setup_script == _MOCK_STANDALONE / "setup.sh"
    assert instance.script_path == _spec(module).toolkit_dir / "standalone" / "inference.py"
    result = module.run_ext(MockCPUToolInput(items=[4, 5]), module.ExtToolConfig(device="cpu"))
    assert [r.item for r in result.results] == [4, 5]


def test_external_toolkit_runs_persistent(make_toolkit, proto_home, monkeypatch):
    """Under persist_tool, calls share one warm worker."""
    module = make_toolkit("ext_persist_tk")
    _seed_env("ext_persist_tk")
    monkeypatch.setattr(ToolInstance, "_create_env", lambda self: pytest.fail("seeded env was rebuilt"))

    config = module.ExtToolConfig(device="cpu")
    with ToolInstance.persist_tool(_spec(module).key):
        first = module.run_ext(MockCPUToolInput(items=[1]), config)
        second = module.run_ext(MockCPUToolInput(items=[2]), config)
    assert first.results[0].process_unique_id == second.results[0].process_unique_id
