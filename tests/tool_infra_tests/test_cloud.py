"""tests/tool_infra_tests/test_cloud.py.

Tests for device="cloud" dispatch via proto_tools.cloud.use_api_backend.

``proto_client`` is stubbed via ``sys.modules`` so tests run without the
real SDK installed and without hitting a network.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import Field

from proto_tools.cloud import (
    disable_api_backend,
    is_api_backend_enabled,
    use_api_backend,
)
from proto_tools.tools.tool_registry import ToolRegistry
from proto_tools.utils import BaseConfig, ConfigField
from proto_tools.utils.tool_io import BaseToolInput
from tests.tool_infra_tests.test_export_functionality import MockToolOutputBase


class _CloudInput(BaseToolInput):
    payload: str = Field(description="Input payload")


class _CloudConfig(BaseConfig):
    temperature: float = ConfigField(default=1.0, ge=0.0)


class _CloudOutput(MockToolOutputBase):
    result: str = Field(description="Result payload")


class _PreprocessConfig(_CloudConfig):
    local_path: str | None = ConfigField(default=None, hidden=True)

    def preprocess(self, inputs: BaseToolInput) -> BaseToolInput:
        raise AssertionError("cloud dispatch should not run local preprocess")


@dataclass
class _StubResponse:
    """Stand-in for proto_client.models.JobStatusResponse."""

    result: Any


@dataclass
class _StubToolsNamespace:
    output_to_return: Any = None
    raise_on_run: BaseException | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def run(
        self,
        tool_key: str,
        inputs: dict[str, Any],
        config: dict[str, Any] | None = None,
        poll_interval: float = 1.0,
        timeout: float = 600.0,
        *,
        output_model: type | None = None,
    ) -> _StubResponse:
        self.calls.append(
            {
                "tool_key": tool_key,
                "inputs": inputs,
                "config": config,
                "poll_interval": poll_interval,
                "timeout": timeout,
                "output_model": output_model,
            }
        )
        if self.raise_on_run is not None:
            raise self.raise_on_run
        return _StubResponse(result=self.output_to_return)


class _StubProtoClient:
    """Fake proto_client.ProtoClient for tests."""

    last_instance: _StubProtoClient | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.tools = _StubToolsNamespace()
        _StubProtoClient.last_instance = self


class _FakeProtoAuthError(Exception):
    """Stand-in for ``proto_client.errors.ProtoAuthError`` (401/403)."""

    def __init__(self, message: str, *, status_code: int, request_id: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.request_id = request_id


@pytest.fixture
def fake_proto_client(monkeypatch):
    """Install a fake ``proto_client`` module + stub key so use_api_backend() can construct a client."""
    fake_module = types.ModuleType("proto_client")
    fake_module.ProtoClient = _StubProtoClient  # type: ignore[attr-defined]
    fake_errors = types.ModuleType("proto_client.errors")
    fake_errors.ProtoAuthError = _FakeProtoAuthError  # type: ignore[attr-defined]
    fake_module.errors = fake_errors  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "proto_client", fake_module)
    monkeypatch.setitem(sys.modules, "proto_client.errors", fake_errors)
    monkeypatch.setenv("PROTO_API_KEY", "test-stub-key")
    _StubProtoClient.last_instance = None
    yield _StubProtoClient
    _StubProtoClient.last_instance = None


@pytest.fixture
def clean_registry():
    """Clean tool registry + cloud state for each test."""
    original_registry = ToolRegistry._registry.copy()
    original_dispatch = ToolRegistry._try_dispatch
    ToolRegistry._registry.clear()
    yield ToolRegistry
    ToolRegistry._registry = original_registry
    disable_api_backend()
    ToolRegistry._try_dispatch = original_dispatch  # type: ignore[method-assign]


def _register_cloud_tool(registry, key: str):
    """Register a tool whose local impl fails — proves we routed to cloud."""

    def _must_not_run_locally(inputs, config, instance=None):
        del inputs, config, instance
        raise AssertionError(f"Local execution of {key!r} should not happen when device='cloud'")

    registry.register(
        key=key,
        label=key,
        category="test",
        input_class=_CloudInput,
        config_class=_CloudConfig,
        output_class=_CloudOutput,
        description=key,
    )(_must_not_run_locally)
    return registry.get(key)


# ─ parse_device_string / is_api_backend_enabled ──────────────────────────────


def test_parse_device_string_accepts_cloud():
    """parse_device_string returns DeviceSpec(kind='cloud') for 'cloud'."""
    from proto_tools.utils.device import parse_device_string

    spec = parse_device_string("cloud")
    assert spec.kind == "cloud"
    assert spec.devices == ["cloud"]
    assert spec.count == 1


def test_is_api_backend_enabled_defaults_false():
    """Before use_api_backend() is called, the flag is off."""
    assert is_api_backend_enabled() is False


# ─ use_api_backend wiring ────────────────────────────────────────────────────


def test_use_api_backend_routes_device_cloud(fake_proto_client, clean_registry):
    """device='cloud' dispatches through proto-client and returns the validated output."""
    expected = _CloudOutput(result="from-api")
    use_api_backend(poll_interval=0.25, timeout=5.0, api_key="test-key")
    assert is_api_backend_enabled() is True

    client = fake_proto_client.last_instance
    assert client is not None
    assert client.kwargs == {"api_key": "test-key"}
    client.tools.output_to_return = expected

    spec = _register_cloud_tool(clean_registry, "api-tool")
    result = spec.function(_CloudInput(payload="hi"), _CloudConfig(device="cloud"))

    assert isinstance(result, _CloudOutput)
    assert result.result == "from-api"
    assert result.success is True
    assert result.tool_id == "api-tool"

    assert len(client.tools.calls) == 1
    call = client.tools.calls[0]
    assert call["tool_key"] == "api-tool"
    assert call["inputs"] == {"payload": "hi"}
    # device='cloud' is the client-side routing signal; the server picks its own
    # physical device, so the dispatcher strips device before sending.
    assert "device" not in call["config"]
    assert call["poll_interval"] == 0.25
    assert call["timeout"] == 5.0
    assert call["output_model"] is _CloudOutput


def test_device_cloud_dispatches_before_preprocess(fake_proto_client, clean_registry):
    """device='cloud' sends the original request instead of running local preprocess first."""
    expected = _CloudOutput(result="from-api")
    use_api_backend()
    fake_proto_client.last_instance.tools.output_to_return = expected

    clean_registry.register(
        key="preprocess-tool",
        label="preprocess-tool",
        category="test",
        input_class=_CloudInput,
        config_class=_PreprocessConfig,
        output_class=_CloudOutput,
        description="preprocess-tool",
    )(lambda inputs, config, instance=None: _CloudOutput(result="local"))
    spec = clean_registry.get("preprocess-tool")

    result = spec.function(_CloudInput(payload="raw"), _PreprocessConfig(device="cloud"))

    assert result.result == "from-api"
    call = fake_proto_client.last_instance.tools.calls[0]
    assert call["inputs"] == {"payload": "raw"}
    assert "local_path" not in call["config"]


def test_device_cloud_uses_custom_dispatch_without_proto_client_backend(clean_registry):
    """Other ToolRegistry dispatch hooks can handle device='cloud' without proto_tools.cloud."""
    original_dispatch = ToolRegistry._try_dispatch

    clean_registry.register(
        key="custom-cloud-tool",
        label="custom-cloud-tool",
        category="test",
        input_class=_CloudInput,
        config_class=_PreprocessConfig,
        output_class=_CloudOutput,
        description="custom-cloud-tool",
    )(lambda inputs, config, instance=None: _CloudOutput(result="local"))
    spec = clean_registry.get("custom-cloud-tool")
    seen: dict[str, Any] = {}

    def _custom_dispatch(cls, key, inputs, config):
        del cls
        seen["key"] = key
        seen["inputs"] = inputs.model_dump()
        seen["config"] = config.model_dump(exclude_none=True)
        return _CloudOutput(result="custom")

    ToolRegistry._try_dispatch = classmethod(_custom_dispatch)
    try:
        result = spec.function(_CloudInput(payload="raw"), _PreprocessConfig(device="cloud"))
    finally:
        ToolRegistry._try_dispatch = original_dispatch

    assert result.result == "custom"
    assert seen["key"] == "custom-cloud-tool"
    assert seen["inputs"] == {"payload": "raw"}
    assert seen["config"]["device"] == "cloud"


def test_device_cpu_falls_through_to_local(fake_proto_client, clean_registry):
    """After use_api_backend(), non-cloud devices must still run locally."""
    use_api_backend()

    def _local_impl(inputs, config, instance=None):
        del config, instance
        return _CloudOutput(result=f"local:{inputs.payload}")

    clean_registry.register(
        key="local-tool",
        label="local-tool",
        category="test",
        input_class=_CloudInput,
        config_class=_CloudConfig,
        output_class=_CloudOutput,
        description="local-tool",
    )(_local_impl)
    spec = clean_registry.get("local-tool")

    result = spec.function(_CloudInput(payload="y"), _CloudConfig(device="cpu"))

    assert result.result == "local:y"
    assert result.success is True
    assert fake_proto_client.last_instance.tools.calls == []


def test_device_cloud_without_use_api_backend_raises(clean_registry):
    """device='cloud' without use_api_backend() surfaces a clear configuration error."""
    assert is_api_backend_enabled() is False
    spec = _register_cloud_tool(clean_registry, "no-backend")

    with pytest.raises(RuntimeError, match="Proto's remote execution backend is not enabled"):
        spec.function(_CloudInput(payload="x"), _CloudConfig(device="cloud"))


def test_use_api_backend_without_proto_client_raises(monkeypatch):
    """use_api_backend() raises a helpful ImportError if proto-client is missing."""
    import builtins

    monkeypatch.delitem(sys.modules, "proto_client", raising=False)
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "proto_client":
            raise ImportError("No module named 'proto_client'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    with pytest.raises(ImportError, match=r"proto-tools\[cloud\]"):
        use_api_backend()


# ─ failure handling ──────────────────────────────────────────────────────────


def test_transport_error_propagates_to_caller(fake_proto_client, clean_registry):
    """Non-auth SDK failures (network, 5xx, etc.) propagate unchanged — not buried as the placeholder."""
    use_api_backend()
    fake_proto_client.last_instance.tools.raise_on_run = ConnectionError("network down")

    spec = _register_cloud_tool(clean_registry, "transport-failing-tool")

    with pytest.raises(ConnectionError, match="network down"):
        spec.function(_CloudInput(payload="x"), _CloudConfig(device="cloud"))


def test_real_tool_failure_propagates_to_caller(fake_proto_client, clean_registry):
    """A RuntimeError from proto-client (failed/cancelled job) propagates with the original message."""
    use_api_backend()
    fake_proto_client.last_instance.tools.raise_on_run = RuntimeError(
        "Job fc-1 failed: ValueError: chain 'A' not present"
    )

    spec = _register_cloud_tool(clean_registry, "failing-cloud-tool")

    with pytest.raises(RuntimeError, match="chain 'A'"):
        spec.function(_CloudInput(payload="x"), _CloudConfig(device="cloud"))


def test_unexpected_result_type_raises(fake_proto_client, clean_registry):
    """If proto-client returns a non-matching result type, cloud dispatch flags it."""
    use_api_backend()
    fake_proto_client.last_instance.tools.output_to_return = {"result": "ok"}

    spec = _register_cloud_tool(clean_registry, "type-check-tool")

    with pytest.raises(Exception, match="expected _CloudOutput"):
        spec.function(_CloudInput(payload="x"), _CloudConfig(device="cloud"))


# ─ disable ───────────────────────────────────────────────────────────────────


def test_disable_api_backend_restores_default(fake_proto_client, clean_registry):
    """disable_api_backend() clears the flag and restores local dispatch."""
    use_api_backend()
    assert is_api_backend_enabled() is True

    disable_api_backend()
    assert is_api_backend_enabled() is False

    def _local_impl(inputs, config, instance=None):
        del inputs, config, instance
        return _CloudOutput(result="local-after-disable")

    clean_registry.register(
        key="post-disable-tool",
        label="post-disable-tool",
        category="test",
        input_class=_CloudInput,
        config_class=_CloudConfig,
        output_class=_CloudOutput,
        description="post-disable-tool",
    )(_local_impl)
    spec = clean_registry.get("post-disable-tool")

    result = spec.function(_CloudInput(payload="q"), _CloudConfig(device="cpu"))
    assert result.result == "local-after-disable"
    del fake_proto_client


def test_use_api_backend_called_twice_replaces_client(fake_proto_client, clean_registry):
    """Re-calling use_api_backend() swaps the active client."""
    use_api_backend(api_key="first")
    first = fake_proto_client.last_instance
    use_api_backend(api_key="second")
    second = fake_proto_client.last_instance

    assert first is not second
    assert second.kwargs == {"api_key": "second"}

    second.tools.output_to_return = _CloudOutput(result="second-client")
    spec = _register_cloud_tool(clean_registry, "swap-tool")
    result = spec.function(_CloudInput(payload="z"), _CloudConfig(device="cloud"))

    assert result.result == "second-client"
    assert first.tools.calls == []
    assert len(second.tools.calls) == 1
