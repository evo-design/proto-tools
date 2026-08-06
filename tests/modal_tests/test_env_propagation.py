"""TEMPORARY — diagnostic tests for PROTO_MODEL_CACHE propagation.

Written while investigating why per-toolkit weight directories were not
appearing on the cache volume. These exercise the real proto-tools code paths
that decide where a tool writes its weights, without touching Modal.

Delete once the volume question is settled, or promote whichever of these
turn out to be worth keeping.
"""

import re
import subprocess

from tests.modal_tests.helpers import REPO, service_modules

CACHE_MOUNT = "/weights"


def toolkits_we_ship() -> set[str]:
    """Toolkit directory names backing the services in this repo, for toolkits that own an env.

    Everything below concerns what reaches a tool's subprocess. A deployed toolkit with no
    ``standalone/`` directory has no such subprocess and no ``env_vars.txt`` to inspect — its
    service calls the tool function in the container's own interpreter — so including it would ask
    ``_resolve_env_def`` for an environment that was never meant to exist.
    """
    from proto_tools.modal import available_tools
    from proto_tools.tools import ToolRegistry

    specs = (ToolRegistry.get(k) for k in available_tools())
    return {spec.source_file.parent.name for spec in specs if spec.has_standalone_env}


# --------------------------------------------------------------------------
# 1. Our side: env_for() must be hermetic and universally applied.
# --------------------------------------------------------------------------


def test_env_for_ignores_ambient_environment(monkeypatch):
    """A developer's own PROTO_* vars must not leak into the built image."""
    from proto_tools.modal.utils import env_for

    baseline = env_for()
    monkeypatch.setenv("PROTO_MODEL_CACHE", "/somewhere/else")
    monkeypatch.setenv("PROTO_HOME", "/nowhere/not-this")
    monkeypatch.setenv("PROTO_RF3_WEIGHTS_DIR", "/nowhere/override")
    assert env_for() == baseline, "env_for() is reading ambient environment"
    assert baseline["PROTO_MODEL_CACHE"] == CACHE_MOUNT


def test_every_service_applies_env_for():
    """A service that skips .env(env_for()) gets no PROTO_MODEL_CACHE at all."""
    missing = [str(p.relative_to(REPO)) for p in service_modules() if "env_for()" not in p.read_text()]
    assert not missing, f"services not applying env_for(): {missing}"


def test_no_service_overrides_the_cache_path():
    """Services may extend env_for(), but must not redirect the weights cache."""
    offenders = []
    for path in service_modules():
        text = path.read_text()
        # an assignment of the form "VAR": "..." after the env_for() spread
        offenders.extend(
            f"{path.relative_to(REPO)} sets {var}"
            for var in ("PROTO_MODEL_CACHE", "PROTO_HOME")
            if re.search(rf'"{var}"\s*:', text)
        )
    assert not offenders, "services redirecting the cache: " + "; ".join(offenders)


# --------------------------------------------------------------------------
# 2. proto-tools side: does PROTO_MODEL_CACHE survive into setup.sh?
# --------------------------------------------------------------------------


def test_no_shipped_toolkit_blocks_proto_model_cache():
    """A [no_passthrough] entry would strip the variable before setup.sh runs."""
    from proto_tools.utils.persistent_worker import _parse_env_vars_file
    from proto_tools.utils.tool_instance import ToolInstance

    blockers = []
    for toolkit in sorted(toolkits_we_ship()):
        env_dir, _ = ToolInstance._resolve_env_def(toolkit)
        path = env_dir / "env_vars.txt"
        if not path.is_file():
            continue
        parsed = _parse_env_vars_file(path)
        if "PROTO_MODEL_CACHE" in parsed.get("no_passthrough", []):
            blockers.append(toolkit)
    assert not blockers, f"toolkits blocking PROTO_MODEL_CACHE: {blockers}"


def test_subprocess_env_carries_proto_model_cache(monkeypatch):
    """The env handed to setup.sh must carry PROTO_MODEL_CACHE unchanged."""
    from proto_tools.utils.persistent_worker import _build_subprocess_env, _parse_env_vars_file
    from proto_tools.utils.tool_instance import ToolInstance

    monkeypatch.setenv("PROTO_MODEL_CACHE", CACHE_MOUNT)
    dropped = []
    for toolkit in sorted(toolkits_we_ship()):
        env_dir, _ = ToolInstance._resolve_env_def(toolkit)
        path = env_dir / "env_vars.txt"
        tool_env_vars = _parse_env_vars_file(path) if path.is_file() else None

        env = _build_subprocess_env("cuda", tool_env_path="/nowhere/env", tool_env_vars=tool_env_vars)
        if env.get("PROTO_MODEL_CACHE") != CACHE_MOUNT:
            dropped.append(f"{toolkit} (got {env.get('PROTO_MODEL_CACHE')!r})")
    assert not dropped, f"toolkits whose setup.sh would not see PROTO_MODEL_CACHE: {dropped}"


# --------------------------------------------------------------------------
# 3. The shell resolver — the code that actually picks the download path.
# --------------------------------------------------------------------------


def resolve_via_shell(toolkit: str, env: dict[str, str]) -> str:
    """Run proto_resolve_weights_dir exactly as setup.sh does, and report WEIGHTS_DIR."""
    import proto_tools

    helper = (
        __import__("pathlib").Path(proto_tools.__file__).parent
        / "utils"
        / "standalone_helpers_source"
        / "standalone_helpers.sh"
    )
    script = f'set -e\nsource "{helper}"\nproto_resolve_weights_dir {toolkit}\necho "$WEIGHTS_DIR"\n'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, check=False)
    assert out.returncode == 0, f"helper failed: {out.stderr[-400:]}"
    return out.stdout.strip().splitlines()[-1]


def test_shell_resolver_targets_the_volume(tmp_path):
    """An absolute PROTO_MODEL_CACHE must resolve to <mount>/rf3.

    Uses a temp dir as the mount because the helper mkdir -p's the target,
    and the test cannot create /weights locally.
    """
    mount = str(tmp_path / "weights")
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "PROTO_MODEL_CACHE": mount}
    assert resolve_via_shell("rf3", env) == f"{mount}/rf3"


def test_shell_resolver_falls_back_without_the_variable(tmp_path):
    """Absent PROTO_MODEL_CACHE, weights land under PROTO_HOME — off the volume."""
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "PROTO_HOME": str(tmp_path / "proto")}
    got = resolve_via_shell("rf3", env)
    assert got == str(tmp_path / "proto" / "proto_model_cache" / "rf3")


def test_per_tool_override_beats_the_volume(tmp_path):
    """PROTO_<TOOL>_WEIGHTS_DIR wins over PROTO_MODEL_CACHE — a way weights escape."""
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "PROTO_MODEL_CACHE": CACHE_MOUNT,
        "PROTO_RF3_WEIGHTS_DIR": str(tmp_path / "elsewhere"),
    }
    assert resolve_via_shell("rf3", env) == str(tmp_path / "elsewhere")
