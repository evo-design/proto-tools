"""tests/functional_tests/test_non_editable_install.py.

Functional test: verify PROTO_HOME path resolution for non-editable pip install.

Run from demo-project/ (NOT the repo root) to ensure Python loads from
site-packages, not the local source directory.

Usage:
    export PROTO_HOME=/oak/stanford/groups/euan/projects/viggiano/.proto
    export PROTO_MODEL_CACHE=/scratch/users/viggiano/model_weights/proto-tools
    cd demo-project/
    python test_non_editable_install.py
"""

import os


def main():
    proto_home = os.environ.get("PROTO_HOME")
    proto_model_cache = os.environ.get("PROTO_MODEL_CACHE")

    print("=" * 70)
    print("NON-EDITABLE INSTALL FUNCTIONAL TEST")
    print("=" * 70)
    print()

    # ── Step 1: Verify env vars are set ──────────────────────────────────
    print("[1] Environment variables")
    assert proto_home, "PROTO_HOME must be set (we're on Sherlock; ~/.proto/ would fill $HOME)"
    print(f"    PROTO_HOME       = {proto_home}")
    print(f"    PROTO_MODEL_CACHE = {proto_model_cache or '(not set, will use PROTO_HOME/proto_model_cache/)'}")
    print()

    # ── Step 2: Verify we're loading from site-packages ──────────────────
    print("[2] Package location")
    import proto_tools

    pkg_file = proto_tools.__file__
    print(f"    __file__ = {pkg_file}")
    assert "site-packages" in pkg_file, (
        f"Expected to load from site-packages but got: {pkg_file}\n"
        f"Are you running from the repo root? Run from demo-project/ instead."
    )
    print("    PASS: Loading from site-packages (non-editable)")
    print()

    # ── Step 3: Verify PROTO_HOME resolution ─────────────────────────────
    print("[3] PROTO_HOME resolution")
    from proto_tools.utils.proto_home import get_proto_home

    resolved = get_proto_home()
    print(f"    get_proto_home() = {resolved}")
    assert str(resolved) == os.path.realpath(os.path.expanduser(proto_home)), f"Expected {proto_home}, got {resolved}"
    print("    PASS: Matches PROTO_HOME env var")
    print()

    # ── Step 4: Verify tool_envs goes to PROTO_HOME (non-editable) ──────
    print("[4] Tool envs location")
    from proto_tools.utils.tool_instance import ToolInstance

    tool_envs = ToolInstance._get_tool_envs_root()
    micromamba = ToolInstance._get_micromamba_root()
    print(f"    tool_envs  = {tool_envs}")
    print(f"    micromamba  = {micromamba}")
    assert str(tool_envs) == os.path.join(str(resolved), "proto_tool_envs"), (
        f"Expected PROTO_HOME/proto_tool_envs, got {tool_envs}"
    )
    assert str(micromamba) == os.path.join(str(resolved), ".micromamba"), (
        f"Expected PROTO_HOME/.micromamba, got {micromamba}"
    )
    print("    PASS: Tool envs and micromamba under PROTO_HOME")
    print()

    # ── Step 5: Verify model cache resolution ────────────────────────────
    print("[5] Model cache location")
    from proto_tools.utils.persistent_worker import _build_subprocess_env

    env = _build_subprocess_env(device="cpu")
    hf_home = env.get("HF_HOME", "(not set)")
    proto_home_in_env = env.get("PROTO_HOME", "(not set)")
    print(f"    subprocess PROTO_HOME = {proto_home_in_env}")
    print(f"    subprocess HF_HOME    = {hf_home}")

    if proto_model_cache:
        expected_hf = os.path.join(proto_model_cache, "huggingface")
    else:
        expected_hf = os.path.join(str(resolved), "proto_model_cache", "huggingface")

    assert hf_home == expected_hf, f"Expected HF_HOME={expected_hf}, got {hf_home}"
    print("    PASS: HF_HOME points to correct model cache")
    print()

    # ── Step 6: Verify resolve_weights_dir ───────────────────────────────
    print("[6] resolve_weights_dir (standalone helper)")
    # Simulate what a subprocess would see
    old_env = os.environ.copy()
    os.environ["PROTO_HOME"] = str(resolved)
    if proto_model_cache:
        os.environ["PROTO_MODEL_CACHE"] = proto_model_cache

    from proto_tools.utils.standalone_helpers_source.standalone_helpers import (
        resolve_weights_dir,
    )

    weights = resolve_weights_dir("test_tool")
    print(f"    resolve_weights_dir('test_tool') = {weights}")

    if proto_model_cache:
        expected = os.path.join(proto_model_cache, "test_tool")
    else:
        expected = os.path.join(str(resolved), "proto_model_cache", "test_tool")

    assert weights == expected, f"Expected {expected}, got {weights}"
    print("    PASS: Weights resolve to correct location")

    # Clean up the test directory we just created
    import shutil

    if weights and os.path.exists(weights):
        shutil.rmtree(weights)

    os.environ.clear()
    os.environ.update(old_env)
    print()

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("ALL CHECKS PASSED")
    print("=" * 70)
    print()
    print("Storage layout for this install:")
    print(f"  Model weights:  {proto_model_cache or os.path.join(str(resolved), 'proto_model_cache')}/")
    print(f"  Tool envs:      {tool_envs}/")
    print(f"  Micromamba:      {micromamba}/")


if __name__ == "__main__":
    main()
