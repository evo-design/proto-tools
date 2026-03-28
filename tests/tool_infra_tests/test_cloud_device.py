"""tests/tool_infra_tests/test_cloud_device.py

Tests for device='cloud' single-tool cloud dispatch."""

from __future__ import annotations

import pytest
from pydantic import Field

from bio_programming_tools.tools.tool_registry import ToolRegistry
from bio_programming_tools.utils import BaseConfig, ConfigField
from bio_programming_tools.utils.tool_io import BaseToolInput
from tests.tool_infra_tests.test_export_functionality import MockToolOutputBase


# ── Mock data models ─────────────────────────────────────────────────────────

class CloudTestInput(BaseToolInput):
    input_data: str = Field(description="Input data")


class CloudTestConfig(BaseConfig):
    param: str = ConfigField(default="default", description="Param")


class CloudTestOutput(MockToolOutputBase):
    result: str = Field(description="Result")


@pytest.fixture
def clean_registry():
    """Provide a clean registry, restore after test."""
    original_registry = ToolRegistry._registry.copy()
    original_backend = ToolRegistry._execution_backend
    original_batch = ToolRegistry._execution_backend_batch
    ToolRegistry._registry.clear()
    yield ToolRegistry
    ToolRegistry._registry = original_registry
    ToolRegistry._execution_backend = original_backend
    ToolRegistry._execution_backend_batch = original_batch


def _register_tool(registry, key="cloud-test"):
    """Register a simple mock tool and return its spec."""
    @registry.register(
        key=key,
        label=key,
        category="test",
        input_class=CloudTestInput,
        config_class=CloudTestConfig,
        output_class=CloudTestOutput,
        description=key,
        uses_gpu=True,
    )
    def run_tool(inputs, config=None, instance=None):
        return CloudTestOutput(result=f"local:{inputs.input_data}")

    return registry.get(key)


# ── Tests ────────────────────────────────────────────────────────────────────

def test_cloud_device_dispatches_to_backend(clean_registry):
    """device='cloud' calls the execution backend and returns its result."""
    spec = _register_tool(clean_registry)
    backend_called_with = {}

    def mock_backend(tool_key, inputs, config):
        backend_called_with["tool_key"] = tool_key
        backend_called_with["input_data"] = inputs.input_data
        return CloudTestOutput(result=f"cloud:{inputs.input_data}")

    clean_registry.set_execution_backend(mock_backend)

    inputs = CloudTestInput(input_data="test")
    config = CloudTestConfig(param="v", device="cloud")
    result = spec.function(inputs, config)

    assert result.success is True
    assert result.result == "cloud:test"
    assert result.tool_id == "cloud-test"
    assert result.execution_time > 0
    assert backend_called_with["tool_key"] == "cloud-test"


def test_cloud_device_no_backend_raises(clean_registry):
    """device='cloud' with no backend registered raises RuntimeError."""
    spec = _register_tool(clean_registry)

    inputs = CloudTestInput(input_data="test")
    config = CloudTestConfig(param="v", device="cloud")

    with pytest.raises(RuntimeError, match="no cloud backend is registered"):
        spec.function(inputs, config)


def test_cloud_device_backend_returns_none_raises(clean_registry):
    """device='cloud' with backend returning None (unsupported tool) raises."""
    spec = _register_tool(clean_registry)

    def mock_backend(tool_key, inputs, config):
        return None  # Tool not supported

    clean_registry.set_execution_backend(mock_backend)

    inputs = CloudTestInput(input_data="test")
    config = CloudTestConfig(param="v", device="cloud")

    with pytest.raises(RuntimeError, match="does not support this tool"):
        spec.function(inputs, config)


def test_cpu_device_runs_locally(clean_registry):
    """device='cpu' does not trigger the cloud intercept path."""
    spec = _register_tool(clean_registry)

    def mock_backend(tool_key, inputs, config):
        # In the normal retry loop, backend is tried first.
        # Return None to fall through to local execution.
        return None

    clean_registry.set_execution_backend(mock_backend)

    inputs = CloudTestInput(input_data="test")
    config = CloudTestConfig(param="v", device="cpu")
    result = spec.function(inputs, config)

    # Should run locally (backend returned None, fell through)
    assert result.success is True
    assert result.result == "local:test"


def test_cloud_device_skips_local_device_validation(clean_registry):
    """device='cloud' should not trigger GPU device validation."""
    # Register a tool that requires exactly 2 GPUs
    @clean_registry.register(
        key="multi-gpu",
        label="Multi GPU",
        category="test",
        input_class=CloudTestInput,
        config_class=CloudTestConfig,
        output_class=CloudTestOutput,
        description="Needs 2 GPUs",
        uses_gpu=True,
        device_count="2",
    )
    def run_tool(inputs, config=None, instance=None):
        return CloudTestOutput(result="local")

    def mock_backend(tool_key, inputs, config):
        return CloudTestOutput(result="cloud")

    clean_registry.set_execution_backend(mock_backend)

    spec = clean_registry.get("multi-gpu")
    inputs = CloudTestInput(input_data="test")
    # device="cloud" should NOT raise ValueError about device count
    config = CloudTestConfig(param="v", device="cloud")
    result = spec.function(inputs, config)

    assert result.success is True
    assert result.result == "cloud"


def test_cloud_device_backend_exception_returns_error_output(clean_registry):
    """Backend execution error returns structured success=False output."""
    spec = _register_tool(clean_registry)

    def failing_backend(tool_key, inputs, config):
        raise ConnectionError("cloud service unavailable")

    clean_registry.set_execution_backend(failing_backend)

    inputs = CloudTestInput(input_data="test")
    config = CloudTestConfig(param="v", device="cloud")
    result = spec.function(inputs, config)

    assert result.success is False
    assert "cloud service unavailable" in result.errors[0]
    assert result.tool_id == "cloud-test"
