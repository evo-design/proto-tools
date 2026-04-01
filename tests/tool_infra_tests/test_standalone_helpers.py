"""tests/tool_infra_tests/test_standalone_helpers.py

Tests for standalone_helpers.py."""

import os

import pytest

from proto_tools.utils.device import determine_visible_devices
from proto_tools.utils.standalone_helpers_source.standalone_helpers import (
    get_subprocess_device_env,
)

# ── Consistency: standalone_helpers vs main codebase ──────────────────────────


def test_get_subprocess_device_env_matches_determine_visible_devices_cpu():
    """Verify CPU device handling matches between standalone and main."""
    env = get_subprocess_device_env("cpu")
    expected = determine_visible_devices("cpu")

    assert env["CUDA_VISIBLE_DEVICES"] == expected, \
        "get_subprocess_device_env('cpu') should match determine_visible_devices('cpu')"


@pytest.mark.uses_gpu
def test_get_subprocess_device_env_matches_determine_visible_devices_single(monkeypatch):
    """Verify single GPU handling matches between standalone and main."""
    # No parent CUDA_VISIBLE_DEVICES
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    env = get_subprocess_device_env("cuda:0")
    expected = determine_visible_devices("cuda:0")

    assert env["CUDA_VISIBLE_DEVICES"] == expected, \
        "get_subprocess_device_env should match determine_visible_devices for cuda:0"


def test_get_subprocess_device_env_matches_determine_visible_devices_with_parent(monkeypatch):
    """Verify device mapping matches when parent has CUDA_VISIBLE_DEVICES set."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5,7")

    # Test single device mapping
    env = get_subprocess_device_env("cuda:0")
    expected = determine_visible_devices("cuda:0")
    assert env["CUDA_VISIBLE_DEVICES"] == expected, \
        "Logical cuda:0 should map to physical GPU 3"

    env = get_subprocess_device_env("cuda:1")
    expected = determine_visible_devices("cuda:1")
    assert env["CUDA_VISIBLE_DEVICES"] == expected, \
        "Logical cuda:1 should map to physical GPU 5"

    env = get_subprocess_device_env("cuda:2")
    expected = determine_visible_devices("cuda:2")
    assert env["CUDA_VISIBLE_DEVICES"] == expected, \
        "Logical cuda:2 should map to physical GPU 7"


def test_get_subprocess_device_env_matches_multi_device(monkeypatch):
    """Verify multi-device handling matches between standalone and main."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,5,7")

    # Test multi-device shorthand: cuda:2,3
    env = get_subprocess_device_env("cuda:2,3")
    expected = determine_visible_devices("cuda:2,3")
    assert env["CUDA_VISIBLE_DEVICES"] == expected, \
        "cuda:2,3 should map to physical GPUs 5,7"

    # Test multi-device verbose: cuda:0,cuda:1
    env = get_subprocess_device_env("cuda:0,cuda:1")
    expected = determine_visible_devices("cuda:0,cuda:1")
    assert env["CUDA_VISIBLE_DEVICES"] == expected, \
        "cuda:0,cuda:1 should map to physical GPUs 0,1"


# ── Standalone helpers specific tests ─────────────────────────────────────────


def test_get_subprocess_device_env_returns_full_env():
    """Verify get_subprocess_device_env returns a complete environment dict."""
    env = get_subprocess_device_env("cpu")

    # Should be a dict
    assert isinstance(env, dict)

    # Should contain CUDA_VISIBLE_DEVICES
    assert "CUDA_VISIBLE_DEVICES" in env

    # Should contain other environment variables (it's a copy of os.environ)
    assert "PATH" in env


def test_get_subprocess_device_env_doesnt_modify_parent_env(monkeypatch):
    """Verify get_subprocess_device_env doesn't modify os.environ."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    original = os.environ.get("CUDA_VISIBLE_DEVICES")

    # Get subprocess env for different device
    env = get_subprocess_device_env("cuda:1")

    # Original environment should be unchanged
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == original

    # Returned env should have mapped value
    assert env["CUDA_VISIBLE_DEVICES"] != original


def test_get_subprocess_device_env_warns_without_parent_cuda_visible_devices(monkeypatch, caplog):
    """Verify warning is logged when CUDA_VISIBLE_DEVICES is not set."""
    import logging

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with caplog.at_level(logging.WARNING):
        get_subprocess_device_env("cuda:0")

    # Should log a warning
    assert any("CUDA_VISIBLE_DEVICES not set" in record.message for record in caplog.records)


def test_get_subprocess_device_env_handles_invalid_device_format(monkeypatch, caplog):
    """Verify graceful handling of invalid device formats."""
    import logging

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")

    with caplog.at_level(logging.WARNING):
        env = get_subprocess_device_env("invalid-device-string")

    # Should log a warning about unexpected format
    assert any("Unexpected device format" in record.message for record in caplog.records)

    # Should still return an env dict with no GPU access (invalid = no devices)
    assert isinstance(env, dict)
    assert env["CUDA_VISIBLE_DEVICES"] == ""


def test_get_subprocess_device_env_handles_index_out_of_range(monkeypatch, caplog):
    """Verify graceful handling when device index exceeds parent's devices."""
    import logging

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")  # Only 2 devices

    with caplog.at_level(logging.ERROR):
        env = get_subprocess_device_env("cuda:5")  # Index 5 doesn't exist

    # Should log an error
    assert any("exceeds parent CUDA_VISIBLE_DEVICES length" in record.message for record in caplog.records)

    # Should return env with unchanged CUDA_VISIBLE_DEVICES (fallback)
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_get_subprocess_device_env_handles_spaces_in_parent(monkeypatch):
    """Verify handling of spaces in parent CUDA_VISIBLE_DEVICES."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", " 3, 5 , 7 ")

    env = get_subprocess_device_env("cuda:0")
    expected = determine_visible_devices("cuda:0")

    # Should strip spaces and map correctly
    assert env["CUDA_VISIBLE_DEVICES"] == expected


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_get_subprocess_device_env_multi_device_non_contiguous_mapping(monkeypatch):
    """Verify multi-device mapping with non-contiguous parent devices.

    Parent CUDA_VISIBLE_DEVICES=0,3,5 (3 physical GPUs).
    Requesting cuda:0,2 (logical indices 0 and 2) should map to
    physical GPUs 0 and 5, giving CUDA_VISIBLE_DEVICES=0,5.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,3,5")

    env = get_subprocess_device_env("cuda:0,2")

    assert env["CUDA_VISIBLE_DEVICES"] == "0,5"


def test_get_subprocess_device_env_single_parent_device(monkeypatch):
    """Verify behavior when parent has only one visible device."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")

    # Only cuda:0 should be valid
    env = get_subprocess_device_env("cuda:0")
    expected = determine_visible_devices("cuda:0")

    assert env["CUDA_VISIBLE_DEVICES"] == expected


def test_get_subprocess_device_env_empty_parent(monkeypatch):
    """Verify behavior when CUDA_VISIBLE_DEVICES is set but empty."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    # Empty CUDA_VISIBLE_DEVICES falls through to no-parent path in
    # get_subprocess_device_env, which uses the device index directly.
    env = get_subprocess_device_env("cuda:0")
    assert env["CUDA_VISIBLE_DEVICES"] == "0"


# ── JAX environment variables ─────────────────────────────────────────────────


def test_gpu_subprocess_removes_jax_preallocation_restrictions(monkeypatch):
    """GPU subprocesses should allow JAX preallocation (ephemeral process)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    # Simulate persistent_worker.py having set these restrictions
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

    env = get_subprocess_device_env("cuda:0")

    # Subprocess gets GPU access; preallocation restrictions should be removed
    assert "XLA_PYTHON_CLIENT_PREALLOCATE" not in env
    assert "XLA_PYTHON_CLIENT_ALLOCATOR" not in env
    assert "JAX_PLATFORMS" not in env


def test_cpu_subprocess_sets_jax_platforms_cpu(monkeypatch):
    """CPU subprocesses should set JAX_PLATFORMS=cpu."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    env = get_subprocess_device_env("cpu")

    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["JAX_PLATFORMS"] == "cpu"


def test_gpu_subprocess_removes_jax_platforms_if_inherited(monkeypatch):
    """GPU subprocesses should remove JAX_PLATFORMS if inherited from parent."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")  # Leftover from parent

    env = get_subprocess_device_env("cuda:0")

    # Subprocess gets GPU; should not be forced to CPU
    assert "JAX_PLATFORMS" not in env


def test_multi_gpu_subprocess_removes_jax_restrictions(monkeypatch):
    """Multi-GPU subprocesses should also remove JAX preallocation restrictions."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,5,7")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

    env = get_subprocess_device_env("cuda:2,3")

    assert env["CUDA_VISIBLE_DEVICES"] == "5,7"
    assert "XLA_PYTHON_CLIENT_PREALLOCATE" not in env
    assert "XLA_PYTHON_CLIENT_ALLOCATOR" not in env


def test_invalid_device_empty_cvd_gets_jax_cpu(monkeypatch):
    """Invalid device format resulting in empty CVD should set JAX to CPU."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    env = get_subprocess_device_env("not-a-device")

    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["JAX_PLATFORMS"] == "cpu"


# ── resolve_weights_dir ──────────────────────────────────────────────────────

from proto_tools.utils.standalone_helpers_source.standalone_helpers import (
    resolve_weights_dir,
)


def test_resolve_weights_dir_default_uses_proto_home(monkeypatch, tmp_path):
    """Default mode (no PROTO_MODEL_CACHE) returns {PROTO_HOME}/proto_model_cache/{tool}."""
    proto_home = tmp_path / ".proto"
    monkeypatch.setenv("PROTO_HOME", str(proto_home))
    monkeypatch.delenv("PROTO_MODEL_CACHE", raising=False)
    monkeypatch.delenv("PROTO_FAMPNN_WEIGHTS_DIR", raising=False)

    result = resolve_weights_dir("fampnn")

    assert result == str(proto_home / "proto_model_cache" / "fampnn")
    assert os.path.isdir(result)


def test_resolve_weights_dir_in_env_explicit(monkeypatch, tmp_path):
    """Explicit PROTO_MODEL_CACHE=IN_ENV stores weights in the tool venv."""
    monkeypatch.setenv("PROTO_MODEL_CACHE", "IN_ENV")
    monkeypatch.setenv("TOOL_VENV_PATH", str(tmp_path))
    monkeypatch.delenv("PROTO_FAMPNN_WEIGHTS_DIR", raising=False)

    result = resolve_weights_dir("fampnn")

    assert result == str(tmp_path / "model_weight_cache")


def test_resolve_weights_dir_shared_path_mode(monkeypatch, tmp_path):
    """Absolute path mode returns /path/{tool_name}/."""
    shared = tmp_path / "shared_weights"
    shared.mkdir()
    monkeypatch.setenv("PROTO_MODEL_CACHE", str(shared))
    monkeypatch.delenv("PROTO_PROTENIX_WEIGHTS_DIR", raising=False)

    result = resolve_weights_dir("protenix")

    assert result == str(shared / "protenix")
    assert os.path.isdir(result)


def test_resolve_weights_dir_none_mode(monkeypatch, tmp_path):
    """NONE mode falls back to {venv}/weights/ (matching shell helper)."""
    venv = tmp_path / "tool_env"
    venv.mkdir()
    monkeypatch.setenv("PROTO_MODEL_CACHE", "NONE")
    monkeypatch.setenv("VENV_PATH", str(venv))
    monkeypatch.delenv("PROTO_BOLTZ2_WEIGHTS_DIR", raising=False)
    monkeypatch.delenv("TOOL_VENV_PATH", raising=False)

    result = resolve_weights_dir("boltz2")

    assert result == str(venv / "weights")
    assert os.path.isdir(result)


def test_resolve_weights_dir_per_tool_override_beats_mode(monkeypatch, tmp_path):
    """PROTO_{TOOL}_WEIGHTS_DIR overrides PROTO_MODEL_CACHE."""
    override_dir = tmp_path / "my_custom_dir"
    monkeypatch.setenv("PROTO_MODEL_CACHE", "NONE")
    monkeypatch.setenv("PROTO_FAMPNN_WEIGHTS_DIR", str(override_dir))

    result = resolve_weights_dir("fampnn")

    assert result == str(override_dir)
    assert os.path.isdir(result)


def test_resolve_weights_dir_per_tool_override_beats_in_env(monkeypatch, tmp_path):
    """PROTO_{TOOL}_WEIGHTS_DIR overrides IN_ENV mode too."""
    override_dir = tmp_path / "override"
    monkeypatch.setenv("PROTO_MODEL_CACHE", "IN_ENV")
    monkeypatch.setenv("TOOL_VENV_PATH", str(tmp_path / "venv"))
    monkeypatch.setenv("PROTO_ESM_IF1_WEIGHTS_DIR", str(override_dir))

    result = resolve_weights_dir("esm_if1")

    assert result == str(override_dir)


def test_resolve_weights_dir_creates_leaf_directory(monkeypatch, tmp_path):
    """resolve_weights_dir creates the leaf directory and tool subdirectory."""
    parent = tmp_path / "existing"
    parent.mkdir()
    shared = parent / "weights"
    monkeypatch.setenv("PROTO_MODEL_CACHE", str(shared))
    monkeypatch.delenv("PROTO_FAMPNN_WEIGHTS_DIR", raising=False)

    result = resolve_weights_dir("fampnn")

    assert os.path.isdir(result)
    assert result == str(shared / "fampnn")


def test_resolve_weights_dir_creates_explicit_path(monkeypatch, tmp_path):
    """resolve_weights_dir creates the directory when given an explicit path."""
    cache_path = tmp_path / "custom_cache"
    monkeypatch.setenv("PROTO_MODEL_CACHE", str(cache_path))
    monkeypatch.delenv("PROTO_FAMPNN_WEIGHTS_DIR", raising=False)

    result = resolve_weights_dir("fampnn")

    assert result == str(cache_path / "fampnn")
    assert os.path.isdir(result)


def test_resolve_weights_dir_in_env_no_venv(monkeypatch):
    """IN_ENV with no TOOL_VENV_PATH/VENV_PATH returns None."""
    monkeypatch.setenv("PROTO_MODEL_CACHE", "IN_ENV")
    monkeypatch.delenv("TOOL_VENV_PATH", raising=False)
    monkeypatch.delenv("VENV_PATH", raising=False)
    monkeypatch.delenv("PROTO_FAMPNN_WEIGHTS_DIR", raising=False)

    result = resolve_weights_dir("fampnn")

    assert result is None


def test_resolve_weights_dir_no_proto_home_falls_back_to_default(monkeypatch, tmp_path):
    """When PROTO_HOME is also unset, falls back to ~/.proto/proto_model_cache/."""
    monkeypatch.delenv("TOOL_VENV_PATH", raising=False)
    monkeypatch.delenv("VENV_PATH", raising=False)
    monkeypatch.delenv("PROTO_HOME", raising=False)
    monkeypatch.delenv("PROTO_MODEL_CACHE", raising=False)
    monkeypatch.delenv("PROTO_FAMPNN_WEIGHTS_DIR", raising=False)

    result = resolve_weights_dir("fampnn")

    home = os.path.expanduser("~")
    assert result == os.path.join(home, ".proto", "proto_model_cache", "fampnn")
