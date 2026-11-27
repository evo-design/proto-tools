"""tests/tool_infra_tests/test_dependency_caps.py.

Checks the hand-set caps in ``proto_tools/utils/dependency_caps.txt`` and how env builds apply them.
These tests only read the caps; they never set or raise one.
"""

import fnmatch
import re
import subprocess
from pathlib import Path
from unittest.mock import patch

try:
    import tomllib
except ModuleNotFoundError:  # requires-python allows 3.10, where tomllib does not exist
    import tomli as tomllib  # type: ignore[no-redef]

import pytest

from proto_tools.utils import compute_deps
from proto_tools.utils.dependency_caps import CAPS_FILE, cap_applies, parse_caps, write_env_caps
from proto_tools.utils.tool_instance import ToolInstance

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "proto_tools"

# Package families a global cap may name: those tied to the GPU, driver or compiler. A cap outside
# these needs a deliberate change here; a break in one tool belongs in that tool's setup.sh.
CAPPABLE = re.compile(
    r"^(torch|triton|vllm|flash-attn|jax|jaxlib|jax-cuda1\d-(plugin|pjrt)|cuequivariance(-.+)?|nvidia-.+)$"
)
# A version bound on a package, e.g. ``torch==2.11.0+cu129`` or ``jax[cuda12]>=0.4.20``
SPEC = re.compile(
    r"(?<![\w.-])(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)(\[[^\]]*\])?\s*(?P<op>===|==|~=|>=|>)\s*(?P<version>\d+(\.\d+)*)"
)
# A wheel installed by URL pins its version in the file name, e.g. ``.../vllm-0.26.0+cu129-cp38-abi3-...whl``
WHEEL = re.compile(r"/(?P<name>[A-Za-z0-9][A-Za-z0-9_.]*)-(?P<version>\d+(\.\d+)*)[^/\s\"']*\.whl")


def _release(version: str) -> tuple[int, ...]:
    """Return a version's release numbers, ignoring any local (``+cu129``) or post suffix."""
    return tuple(int(part) for part in re.match(r"\d+(\.\d+)*", version).group().split("."))  # type: ignore[union-attr]


def _below(version: str, cap: str) -> bool:
    """Return whether ``version`` sits below ``cap``, comparing release numbers padded with zeros."""
    v, c = _release(version), _release(cap)
    width = max(len(v), len(c))
    return v + (0,) * (width - len(v)) < c + (0,) * (width - len(c))


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _packaged_caps() -> dict[str, str]:
    """Return ``{package: cap version}`` for the packaged caps."""
    caps = {}
    for cap in parse_caps(CAPS_FILE.read_text()):
        name, version = cap.requirement.split("<")
        assert name not in caps, f"dependency_caps.txt caps {name} twice"
        caps[name] = version
    return caps


# ── The caps file ─────────────────────────────────────────────────────────


def test_every_cap_is_scoped_and_cites_its_break():
    """Each line caps a GPU-coupled package, says which machines it applies to, and links the break."""
    caps = parse_caps(CAPS_FILE.read_text())
    assert caps, "dependency_caps.txt has no caps"
    for cap in caps:
        name = cap.requirement.split("<")[0]
        assert CAPPABLE.match(name), f"{name} is not a GPU-, driver- or compiler-coupled package; cap it in its tool"
        assert re.search(r"https?://", cap.reason), f"{cap.requirement}: cite the break with a link"


def test_a_malformed_cap_line_is_rejected():
    """A line without a condition, or with an unknown one, fails rather than applying everywhere."""
    for line in ("torch<2.15", "torch<2.15  # breaks things", "torch<2.15  # cuda<13: reason (https://x)"):
        with pytest.raises(ValueError, match="dependency caps"):
            parse_caps(line)


@pytest.mark.parametrize(
    ("condition", "driver", "applies"),
    [
        ("all", None, True),
        ("all", 590, True),
        ("driver<580", 550, True),
        ("driver<580", 579, True),
        ("driver<580", 580, False),
        ("driver<580", None, False),
    ],
)
def test_caps_apply_only_where_their_condition_holds(condition: str, driver: int | None, applies: bool):
    """A driver cap binds below its driver; a machine with no NVIDIA driver gets none of them."""
    assert cap_applies(condition, driver) is applies


def _declared_bounds() -> list[tuple[str, str, str, str]]:
    """Every ``==``/``>=``/``>``/``~=`` bound and URL-pinned wheel, as ``(file, package, operator, version)``."""
    bounds = []
    # setup.sh is read whole: installs span continuation lines and specs often sit in variables
    for path in sorted(PACKAGE_ROOT.rglob("*requirements*.txt")) + sorted(PACKAGE_ROOT.rglob("setup.sh")):
        file = str(path.relative_to(REPO_ROOT))
        for raw in path.read_text().splitlines():
            line = re.sub(r"(^|\s)#.*", "", raw)
            bounds.extend((file, _canonical(m["name"]), m["op"], m["version"]) for m in SPEC.finditer(line))
            bounds.extend((file, _canonical(m["name"]), "==", m["version"]) for m in WHEEL.finditer(line))
    for minimum, _maximum, _variant in compute_deps._TORCH_COMPATIBILITY.values():
        bounds.append(("proto_tools/utils/compute_deps.py", "torch", ">=", minimum))
    for minimum, _maximum in compute_deps._JAX_COMPATIBILITY.values():
        bounds.append(("proto_tools/utils/compute_deps.py", "jax", ">=", minimum))
    return bounds


def test_no_cap_excludes_a_pinned_or_minimum_version():
    """Every cap sits above every pin and minimum a tool declares, so no env becomes impossible to build."""
    caps = _packaged_caps()
    bounds = _declared_bounds()
    found = {(file.split("/")[-3], name, version) for file, name, _op, version in bounds}
    # Known pins, so a scan that silently stops finding bounds fails here
    for known in [("evo2", "torch", "2.11.0"), ("evo2", "vllm", "0.26.0"), ("borzoi", "flash-attn", "2.8.3")]:
        assert known in found, f"the scan no longer finds {known}; is it broken?"
    conflicts = [
        f"{file}: {name}{op}{version} vs cap <{caps[name]}"
        for file, name, op, version in bounds
        if name in caps and not _below(version, caps[name])
    ]
    assert not conflicts, "caps exclude versions tools require:\n" + "\n".join(conflicts)


def test_no_setup_sh_replaces_the_caps_with_a_constraint_flag():
    """A ``--constraint``/``-c`` flag makes uv and pip ignore ``UV_CONSTRAINT``/``PIP_CONSTRAINT``, dropping the caps."""
    install = re.compile(r"\b((uv pip|pip3?|python -m pip) (install|compile|sync)|proto_install_\w+)\b")
    flag = re.compile(r"(\s--constraint[\s=]|\s-c\s)")
    offenders = []
    for path in sorted(PACKAGE_ROOT.rglob("setup.sh")):
        lines = path.read_text().replace("\\\n", " ").splitlines()
        offenders += [
            f"{path.relative_to(REPO_ROOT)}: {line.strip()}"
            for line in lines
            if install.search(line) and flag.search(line)
        ]
    assert not offenders, "add constraints to UV_CONSTRAINT instead of passing a flag:\n" + "\n".join(offenders)


# ── Applying the caps ─────────────────────────────────────────────────────


def _make_env_dir(inst: ToolInstance, cmd: list[str]) -> subprocess.CompletedProcess:
    """Stand in for ``micromamba create`` (and any host probe) by creating the env directory."""
    inst.env_path.mkdir(parents=True, exist_ok=True)
    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")


def _build_env(tmp_path: Path, driver: int | None, returncode: int = 0) -> tuple[ToolInstance, dict[str, str]]:
    """Run ``_create_env`` on a machine with ``driver`` (``None``: no NVIDIA GPU), returning setup.sh's env."""
    inst = ToolInstance.__new__(ToolInstance)
    inst.toolkit = "fake_tool"
    inst.device = "cpu"
    inst.env_path = tmp_path / "fake_env"
    inst.setup_script = tmp_path / "setup.sh"
    inst.setup_script.write_text("#!/bin/bash\n")
    (tmp_path / "python_version.txt").write_text("default: 3.12\n")
    inst._tool_env_vars = {"passthrough": [], "set": [], "no_passthrough": []}
    machine = {"DETECTED_COMPUTE_PLATFORM": "cpu" if driver is None else "cuda"}
    if driver is not None:
        machine["DETECTED_DRIVER_VERSION"] = str(driver)
    seen: dict[str, str] = {}

    def _setup(*args, **kwargs):
        seen.update(kwargs["env"])
        return returncode, "error: no solution for torch"

    with (
        patch.object(inst, "_ensure_micromamba", return_value=Path("/fake/micromamba")),
        patch.object(inst, "_ensure_foundation_env", return_value=None),
        patch.object(inst, "_clean_package_archives"),
        patch.object(inst, "_schedule_uv_prune"),
        patch("proto_tools.utils.tool_instance.write_manifest"),
        patch("proto_tools.utils.compute_deps.detect_compute_environment", return_value=machine),
        patch(
            "proto_tools.utils.tool_instance.subprocess.run",
            side_effect=lambda *a, **k: _make_env_dir(inst, a[0]),
        ),
        patch("proto_tools.utils.tool_instance._run_setup_script", side_effect=_setup),
    ):
        inst._create_env()
    return inst, seen


@pytest.fixture
def no_constraints(monkeypatch: pytest.MonkeyPatch) -> None:
    """No constraints of the user's own."""
    for var in ("UV_CONSTRAINT", "PIP_CONSTRAINT"):
        monkeypatch.delenv(var, raising=False)


def _requirements(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line and not line.startswith("#")]


@pytest.mark.usefixtures("no_constraints")
def test_a_cuda12_driver_builds_under_its_caps(tmp_path: Path):
    """On driver 550, setup.sh installs under the driver<580 caps, which the env keeps a record of."""
    inst, env = _build_env(tmp_path, driver=550)

    env_caps = inst.env_path / "dependency_caps.txt"
    expected = [c.requirement for c in parse_caps(CAPS_FILE.read_text()) if cap_applies(c.condition, 550)]
    assert expected, "no packaged cap applies to driver 550; pick a driver the caps cover"
    assert _requirements(env_caps) == expected
    assert env["UV_CONSTRAINT"] == str(env_caps.absolute())
    assert env["PIP_CONSTRAINT"] == str(env_caps.absolute())


@pytest.mark.usefixtures("no_constraints")
@pytest.mark.parametrize("driver", [590, None])
def test_machines_no_cap_covers_install_the_newest_releases(tmp_path: Path, driver: int | None):
    """A CUDA 13 driver or a machine without an NVIDIA GPU installs under the `all` caps only."""
    inst, env = _build_env(tmp_path, driver=driver)

    everywhere = [c.requirement for c in parse_caps(CAPS_FILE.read_text()) if c.condition == "all"]
    assert _requirements(inst.env_path / "dependency_caps.txt") == everywhere
    if everywhere:
        assert env["UV_CONSTRAINT"] == str((inst.env_path / "dependency_caps.txt").absolute())
    else:
        assert "UV_CONSTRAINT" not in env
        assert "PIP_CONSTRAINT" not in env


@pytest.mark.parametrize(
    ("detected", "gets_driver_caps"),
    [({"DETECTED_DRIVER_VERSION": "0"}, True), ({"DETECTED_DRIVER_VERSION": "n/a"}, False), ({}, False)],
)
def test_an_unreadable_driver_falls_back_safely(tmp_path: Path, detected: dict[str, str], gets_driver_caps: bool):
    """compute_deps reports an unparseable driver as 0, which gets the driver caps; no driver gets none."""
    driver_caps = [c.requirement for c in parse_caps(CAPS_FILE.read_text()) if c.condition.startswith("driver<")]
    assert driver_caps, "no driver cap to check"
    write_env_caps(tmp_path, {"DETECTED_COMPUTE_PLATFORM": "cuda", **detected})
    applied = _requirements(tmp_path / "dependency_caps.txt")
    assert all((cap in applied) is gets_driver_caps for cap in driver_caps)


def test_a_users_own_constraints_stay_out_of_tool_envs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Constraints set for the caller's own project don't leak into an isolated tool env's build."""
    monkeypatch.setenv("UV_CONSTRAINT", "/home/me/project-constraints.txt")
    monkeypatch.setenv("PIP_CONSTRAINT", "/home/me/project-constraints.txt")
    inst, env = _build_env(tmp_path, driver=550)

    env_caps = str((inst.env_path / "dependency_caps.txt").absolute())
    assert env["UV_CONSTRAINT"] == env_caps
    assert env["PIP_CONSTRAINT"] == env_caps


def test_the_caps_file_ships_in_the_wheel():
    """A wheel install reads the caps from the package, so pyproject must ship the file."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        patterns = tomllib.load(fh)["tool"]["setuptools"]["package-data"]["*"]
    assert any(fnmatch.fnmatch(CAPS_FILE.name, pattern) for pattern in patterns), (
        f"{CAPS_FILE.name} is not matched by [tool.setuptools.package-data] in pyproject.toml"
    )
