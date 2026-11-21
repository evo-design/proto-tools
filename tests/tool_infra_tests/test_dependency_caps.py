"""tests/tool_infra_tests/test_dependency_caps.py.

Checks the hand-set caps in ``proto_tools/utils/dependency_caps.txt`` and how env builds apply them.
These tests only read the caps; they never set or raise one.
"""

import re
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from proto_tools.utils import compute_deps
from proto_tools.utils.dependency_caps import CAPS_ENV_VAR, CAPS_FILE, constraint_env, resolve_caps_file
from proto_tools.utils.tool_instance import ToolInstance

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "proto_tools"

# Package families a cap may name: those tied to the GPU, driver or compiler. A cap outside these
# needs a deliberate change here, so plain Python libraries stay uncapped.
CAPPABLE = re.compile(
    r"^(torch|triton|vllm|flash-attn|jax|jaxlib|jax-cuda1\d-(plugin|pjrt)|cuequivariance(-.+)?|nvidia-.+)$"
)
# Released in lockstep: jax requires the matching jaxlib and plugins, so their caps must agree.
JAX_LOCKSTEP = ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "jax-cuda13-plugin", "jax-cuda13-pjrt")

LINE = re.compile(r"^(?P<name>[a-z0-9][a-z0-9-]*)<(?P<cap>\d+(\.\d+)*)\s+#\s+(?P<note>.+)$")
TESTED = re.compile(
    r"^tested (?P<version>\d+(\.\d+)*(\.post\d+)?) \| (?P<date>\d{4}-\d{2}-\d{2}) \| (?P<hardware>[^|]+) \| (?P<evidence>.+)$"
)
FOLLOWS = re.compile(r"^follows (?P<name>[a-z0-9-]+)$")
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


def _caps() -> dict[str, tuple[str, str]]:
    """Parse the caps file into ``{package: (cap, note)}``."""
    caps = {}
    for raw in CAPS_FILE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = LINE.match(line)
        assert match, f"dependency_caps.txt: {line!r} is not `<package><<version>  # <note>`"
        assert match["name"] not in caps, f"dependency_caps.txt caps {match['name']} twice"
        caps[match["name"]] = (match["cap"], match["note"].strip())
    return caps


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _cuda12_build(name: str) -> str:
    """Return the CUDA 12 build's name for a CUDA 13 package (``-cu13`` or ``-cuda13``)."""
    return re.sub(r"(cu|cuda)13", r"\g<1>12", name)


def _cuda13_build(name: str) -> str:
    """Return the CUDA 13 build's name for a CUDA 12 package (``-cu12`` or ``-cuda12``)."""
    return re.sub(r"(cu|cuda)12", r"\g<1>13", name)


# ── The caps file ─────────────────────────────────────────────────────────


def test_every_cap_is_a_cappable_package_with_its_evidence():
    """Each line caps a GPU-coupled package and says what was tested, when, where and in which run."""
    caps = _caps()
    assert caps, "dependency_caps.txt has no caps"
    for name, (cap, note) in caps.items():
        assert CAPPABLE.match(name), f"{name} is not a GPU-, driver- or compiler-coupled package; leave it uncapped"
        tested, follows = TESTED.match(note), FOLLOWS.match(note)
        assert tested or follows, f"{name}: note {note!r} is neither `tested ...` nor `follows ...`"
        if tested:
            assert _below(tested["version"], cap), f"{name}: tested {tested['version']} is not below its cap <{cap}"


def test_a_cuda13_build_follows_its_cuda12_counterpart():
    """A ``follows`` line names a capped cu12 package with the same bound, and each cu12 cap has its cu13 line."""
    caps = _caps()
    for name, (cap, note) in caps.items():
        if follows := FOLLOWS.match(note):
            target = follows["name"]
            assert target != name and target == _cuda12_build(name), f"{name} may only follow its CUDA 12 build"
            assert target in caps, f"{name} follows {target}, which has no cap"
            assert caps[target][0] == cap, f"{name}<{cap} differs from {target}<{caps[target][0]}"
        if _cuda13_build(name) != name:
            assert _cuda13_build(name) in caps, f"{name} is capped but its CUDA 13 build is not"


def test_jax_packages_share_one_cap():
    """jax, jaxlib and the CUDA plugins are released together, so differing caps would leave nothing to install."""
    caps = _caps()
    bounds = {name: caps[name][0] for name in JAX_LOCKSTEP if name in caps}
    assert len(set(bounds.values())) <= 1, f"jax packages have different caps: {bounds}"


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
    """A cap above every pin and minimum a tool declares, so no env becomes impossible to build."""
    caps = _caps()
    bounds = _declared_bounds()
    found = {(file.split("/")[-3], name, version) for file, name, _op, version in bounds}
    # Known pins, so a scan that silently stops finding bounds fails here
    for known in [("evo2", "torch", "2.11.0"), ("evo2", "vllm", "0.26.0"), ("borzoi", "flash-attn", "2.8.3")]:
        assert known in found, f"the scan no longer finds {known}; is it broken?"
    conflicts = [
        f"{file}: {name}{op}{version} vs cap <{caps[name][0]}"
        for file, name, op, version in bounds
        if name in caps and not _below(version, caps[name][0])
    ]
    assert not conflicts, "caps exclude versions tools require:\n" + "\n".join(conflicts)


# ── Applying the caps ─────────────────────────────────────────────────────


def test_resolve_caps_file_follows_the_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Unset uses the packaged caps, ``0`` turns them off, a path replaces them, a missing path fails loudly."""
    monkeypatch.delenv(CAPS_ENV_VAR, raising=False)
    assert resolve_caps_file() == CAPS_FILE

    monkeypatch.setenv(CAPS_ENV_VAR, "0")
    assert resolve_caps_file() is None

    own = tmp_path / "my_caps.txt"
    own.write_text("torch<2.20\n")
    monkeypatch.setenv(CAPS_ENV_VAR, str(own))
    assert resolve_caps_file() == own

    monkeypatch.setenv(CAPS_ENV_VAR, str(tmp_path / "missing.txt"))
    with pytest.raises(FileNotFoundError, match=CAPS_ENV_VAR):
        resolve_caps_file()


def test_constraint_env_keeps_a_users_own_constraints():
    """The caps join constraints the user already set rather than replacing them."""
    env = constraint_env(Path("/env/dependency_caps.txt"), {"UV_CONSTRAINT": "/home/me/uv.txt"})
    assert env == {
        "UV_CONSTRAINT": "/env/dependency_caps.txt /home/me/uv.txt",
        "PIP_CONSTRAINT": "/env/dependency_caps.txt",
    }
    assert constraint_env(None, {"PIP_CONSTRAINT": "/home/me/pip.txt"}) == {"PIP_CONSTRAINT": "/home/me/pip.txt"}
    assert constraint_env(None, {}) == {}


def _make_env_dir(inst: ToolInstance, cmd: list[str]) -> subprocess.CompletedProcess:
    """Stand in for ``micromamba create`` (and any host probe) by creating the env directory."""
    inst.env_path.mkdir(parents=True, exist_ok=True)
    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")


def _build_env(tmp_path: Path, returncode: int = 0) -> tuple[ToolInstance, dict[str, str]]:
    """Run ``_create_env`` with micromamba and setup.sh stubbed, returning the env setup.sh would get."""
    inst = ToolInstance.__new__(ToolInstance)
    inst.toolkit = "fake_tool"
    inst.device = "cpu"
    inst.env_path = tmp_path / "fake_env"
    inst.setup_script = tmp_path / "setup.sh"
    inst.setup_script.write_text("#!/bin/bash\n")
    (tmp_path / "python_version.txt").write_text("default: 3.12\n")
    inst._tool_env_vars = {"passthrough": [], "set": [], "no_passthrough": []}
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
        patch(
            "proto_tools.utils.tool_instance.subprocess.run",
            side_effect=lambda *a, **k: _make_env_dir(inst, a[0]),
        ),
        patch("proto_tools.utils.tool_instance._run_setup_script", side_effect=_setup),
    ):
        inst._create_env()
    return inst, seen


def test_env_builds_install_under_the_caps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """setup.sh runs with uv and pip constrained by the env's own copy of the caps."""
    monkeypatch.delenv(CAPS_ENV_VAR, raising=False)
    monkeypatch.delenv("UV_CONSTRAINT", raising=False)
    monkeypatch.delenv("PIP_CONSTRAINT", raising=False)
    inst, env = _build_env(tmp_path)

    copy = inst.env_path / "dependency_caps.txt"
    assert copy.read_text() == CAPS_FILE.read_text(), "the env records the caps it was built under"
    assert env["UV_CONSTRAINT"] == str(copy.absolute())
    assert env["PIP_CONSTRAINT"] == str(copy.absolute())


def test_env_builds_without_caps_when_turned_off(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """``PROTO_DEPENDENCY_CAPS=0`` builds with no constraints of ours."""
    monkeypatch.setenv(CAPS_ENV_VAR, "0")
    monkeypatch.delenv("UV_CONSTRAINT", raising=False)
    monkeypatch.delenv("PIP_CONSTRAINT", raising=False)
    inst, env = _build_env(tmp_path)

    assert not (inst.env_path / "dependency_caps.txt").exists()
    assert "UV_CONSTRAINT" not in env
    assert "PIP_CONSTRAINT" not in env


@pytest.mark.parametrize("caps", ["on", "0"])
def test_a_users_own_constraints_reach_setup_sh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caps: str):
    """The subprocess env copies only allowlisted variables; the user's constraints still reach setup.sh."""
    if caps == "on":
        monkeypatch.delenv(CAPS_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(CAPS_ENV_VAR, caps)
    monkeypatch.setenv("UV_CONSTRAINT", "/home/me/uv.txt")
    monkeypatch.setenv("PIP_CONSTRAINT", "/home/me/pip.txt")
    inst, env = _build_env(tmp_path)

    ours = f"{(inst.env_path / 'dependency_caps.txt').absolute()} " if caps == "on" else ""
    assert env["UV_CONSTRAINT"] == f"{ours}/home/me/uv.txt"
    assert env["PIP_CONSTRAINT"] == f"{ours}/home/me/pip.txt"


def test_a_failed_build_names_the_caps_and_the_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """When a capped build fails, the error says caps were on and how to build without them."""
    monkeypatch.delenv(CAPS_ENV_VAR, raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        _build_env(tmp_path, returncode=1)
    assert "dependency caps" in str(excinfo.value)
    assert f"{CAPS_ENV_VAR}=0" in str(excinfo.value)
