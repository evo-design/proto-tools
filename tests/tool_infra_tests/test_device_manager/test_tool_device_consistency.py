"""tests/tool_infra_tests/test_device_manager/test_tool_device_consistency.py.

Tests for tool protocol compliance (to_device, get_memory_stats, etc).
"""

import re

import pytest

from proto_tools.tools import ToolRegistry

_all_gpu_specs = [spec for spec in ToolRegistry.list_all() if spec.uses_gpu]


def _find_standalone_script(tool_spec):
    """Return the standalone script path for a tool spec, or None."""
    tool_dir = tool_spec.source_file.parent
    standalone_dir = tool_dir / "standalone"

    for name in ("inference.py", "run.py"):
        if (standalone_dir / name).exists():
            return standalone_dir / name

    return None


# ── Standalone protocol compliance ──────────────────────────────────────


@pytest.mark.parametrize("tool_spec", _all_gpu_specs, ids=lambda spec: spec.key)
def test_standalone_protocol_compliance(tool_spec):
    """Verify that a GPU tool's standalone script follows all DeviceManager protocols.

    Reads the standalone script once and checks all applicable requirements:
    - Has a module-level to_device(device: str) -> dict function
    - to_device() returns a dict with "success" key
    - If to_device() delegates to _model.to_device(), the model class has that method
    - If the tool has a model class to_device(), it uses move_model_to_device() or unload->reload
    - If the tool uses subprocess calls, it uses get_subprocess_device_env() helper
    - If the tool imports JAX, it uses centralized resolve_jax_device() helper
    """
    tool_key = tool_spec.key

    standalone_script = _find_standalone_script(tool_spec)
    assert standalone_script is not None, (
        f"GPU tool {tool_key} has no standalone script (expected standalone/inference.py or run.py)"
    )

    content = standalone_script.read_text()
    violations = []

    # --- 1-4. to_device() protocol — PyTorch/CLI move in-process; pinned (JAX) tools respawn and omit to_device(). ---
    if tool_spec.pin_visible_devices:
        # Pinned tools must not expose the module-level to_device() dispatch (a class-level .to_device() at load is fine).
        assert re.search(r"^def to_device\(", content, re.MULTILINE) is None, (
            f"{tool_key} is pin_visible_devices=True but defines a module-level to_device(); "
            "pinned tools respawn on device change and must not implement it"
        )
    elif "def to_device(" not in content:
        violations.append("Missing to_device() function")
    else:
        # --- 2. Correct signature: def to_device(device: str) -> dict[str, Any]: ---
        sig_pattern = r"def to_device\(device:\s*str\)\s*->\s*dict(\[str,\s*Any\])?:"
        if not re.search(sig_pattern, content):
            violations.append(
                "to_device() has wrong signature (expected: def to_device(device: str) -> dict[str, Any]:)"
            )

        # --- 3. Module-level to_device returns {"success": ...} ---
        module_pattern = r"^def to_device\(device:\s*str\)\s*->\s*dict(\[str,\s*Any\])?:"
        match = re.search(module_pattern, content, re.MULTILINE)
        if match:
            rest = content[match.end() :]
            next_fn = re.search(r"\ndef |^def ", rest, re.MULTILINE)
            section = rest[: next_fn.start()] if next_fn else rest
            if '"success"' not in section and "'success'" not in section:
                violations.append("to_device() return value missing 'success' key")

        # --- 4. If global to_device() calls _model.to_device(), model class must have it ---
        if "_model.to_device(" in content:
            class_method = re.search(r"class\s+\w+.*?^[ \t]+def\s+to_device\s*\(", content, re.MULTILINE | re.DOTALL)
            if not class_method:
                violations.append("to_device() calls _model.to_device() but no model class has that method")

    # --- 5. Model class to_device() must use move_model_to_device() helper
    #         OR a clean unload->reload cycle (for opaque models like AlphaGenome
    #         that cannot be moved via jax.device_put or model.to()) ---
    class_to_device = re.search(
        r"class\s+\w+.*?^\s+def\s+to_device\s*\(self,\s*device:\s*str\)\s*->\s*None:", content, re.MULTILINE | re.DOTALL
    )
    if class_to_device:
        has_move_helper = bool(
            re.search(r"from \.?standalone_helpers import\s.*move_model_to_device", content, re.DOTALL)
        ) and bool(re.search(r"move_model_to_device\s*\(", content))
        has_unload_reload = bool(re.search(r"self\.unload\s*\(", content))
        if not has_move_helper and not has_unload_reload:
            violations.append(
                "Model class has to_device() but doesn't use move_model_to_device() helper "
                "or a clean unload->reload cycle"
            )

    # --- 6. CLI tools with subprocess must use get_subprocess_device_env() ---
    has_subprocess = bool(re.search(r"\bsubprocess\.(run|Popen|call|check_output)\s*\(", content))
    if has_subprocess:
        has_env_import = bool(re.search(r"from \.?standalone_helpers import .*get_subprocess_device_env", content))
        has_env_usage = bool(re.search(r"get_subprocess_device_env\s*\(", content))
        has_env_param = bool(re.search(r"subprocess\.\w+\([^)]*env\s*=", content))
        if not has_env_import or not has_env_usage:
            violations.append("Uses subprocess calls but doesn't use get_subprocess_device_env() helper")
        elif not has_env_param:
            violations.append("Calls get_subprocess_device_env() but doesn't pass env= to subprocess")

    # --- 7. JAX tools must use centralized resolve_jax_device() ---
    is_jax = bool(re.search(r"^import jax\b|^from jax\b", content, re.MULTILINE))
    if is_jax:
        has_jax_import = bool(re.search(r"from \.?standalone_helpers import\s.*resolve_jax_device", content, re.DOTALL))
        if not has_jax_import:
            violations.append("JAX tool doesn't import resolve_jax_device from standalone_helpers")
        has_local_def = bool(re.search(r"^def _?resolve_jax_device\s*\(", content, re.MULTILINE))
        if has_local_def:
            violations.append("JAX tool defines its own resolve_jax_device() instead of using standalone_helpers")

    assert not violations, f"Tool {tool_key} has protocol violations in {standalone_script}:\n" + "\n".join(
        f"  - {v}" for v in violations
    )


# ── Config-level checks ────────────────────────────────────────────────


@pytest.mark.parametrize("tool_spec", _all_gpu_specs, ids=lambda spec: spec.key)
def test_gpu_tools_default_to_generic_cuda(tool_spec):
    """Test that GPU-enabled tool defaults to 'cuda' (not 'cuda:0' or specific GPUs).

    This ensures consistency with DeviceManager, which expects tools to request
    generic 'cuda' and handles allocation to specific devices automatically.
    """
    tool_key = tool_spec.key

    config_model = tool_spec.config_model
    device_field_info = config_model.model_fields.get("device")

    assert device_field_info is not None, f"{tool_key}: missing device field"

    default_device = device_field_info.default

    assert default_device != "cpu", f"{tool_key}: GPU tool defaults to 'cpu' (should be 'cuda' or 'cudaxN')"

    is_generic_cuda = default_device == "cuda" or re.match(r"^cudax\d+$", default_device)
    assert is_generic_cuda, (
        f"{tool_key}: defaults to '{default_device}' (should be 'cuda' for single-GPU or 'cudaxN' for multi-GPU)"
    )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.uses_gpu
@pytest.mark.integration
@pytest.mark.parametrize(
    "tool_spec",
    _all_gpu_specs,
    ids=lambda spec: spec.key,
)
def test_get_memory_stats_via_worker(tool_spec):
    """Test that get_memory_stats() actually works when called via worker subprocess.

    This is a runtime test that:
    1. Starts a worker for each GPU tool
    2. Sends the get_memory_stats command
    3. Verifies the response has the expected format
    """
    from proto_tools.utils.tool_instance import ToolInstance

    tool_key = tool_spec.key

    toolkit = tool_spec.source_file.parent.name

    with ToolInstance.persist_tool(toolkit) as instance:
        try:
            command = {"command": "get_memory_stats"}
            script_path = tool_spec.source_file.parent / "standalone" / "inference.py"
            if not script_path.exists():
                script_path = tool_spec.source_file.parent / "standalone" / "run.py"

            from proto_tools.utils.base_config import BaseConfig

            cfg = BaseConfig(verbose=False, timeout=1800)
            result = ToolInstance.dispatch(
                toolkit,
                command,
                instance=instance,
                script_path=script_path,
                config=cfg,
            )
        except Exception:
            if instance._worker is not None:
                result = instance._worker.send(command, timeout=1800)
            else:
                raise

        assert isinstance(result, dict), f"Tool {tool_key}: get_memory_stats() returned {type(result)}, expected dict"

        assert "available" in result, (
            f"Tool {tool_key}: get_memory_stats() response missing 'available' key. Got: {result.keys()}"
        )

        assert "framework" in result, (
            f"Tool {tool_key}: get_memory_stats() response missing 'framework' key. Got: {result.keys()}"
        )

        if result["available"]:
            assert result["framework"] in ["pytorch", "jax"], (
                f"Tool {tool_key}: unexpected framework '{result['framework']}', expected 'pytorch' or 'jax'"
            )

            if result["framework"] == "pytorch":
                assert "allocated_bytes" in result, f"Tool {tool_key}: PyTorch memory stats missing 'allocated_bytes'"
                assert isinstance(result["allocated_bytes"], (int, float)), (
                    f"Tool {tool_key}: allocated_bytes should be numeric, got {type(result['allocated_bytes'])}"
                )

            elif result["framework"] == "jax":
                assert "bytes_in_use" in result, f"Tool {tool_key}: JAX memory stats missing 'bytes_in_use'"
                assert isinstance(result["bytes_in_use"], (int, float)), (
                    f"Tool {tool_key}: bytes_in_use should be numeric, got {type(result['bytes_in_use'])}"
                )

        else:
            assert "reason" in result, f"Tool {tool_key}: get_memory_stats() returned available=False but no 'reason'"
