"""tests/tool_infra_tests/test_cloud.py.

Tests for device="cloud" dispatch via proto_tools.cloud.dispatch_to_cloud.

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

from proto_tools.tools.tool_registry import ToolRegistry
from proto_tools.utils import BaseConfig, ConfigField
from proto_tools.utils.tool_io import BaseToolInput
from tests.tool_infra_tests.test_export_functionality import MockToolOutputBase


class _CloudInput(BaseToolInput):
    payload: str = Field(description="Input payload")


class _CloudConfig(BaseConfig):
    temperature: float = ConfigField(default=1.0, ge=0.0, title="Temperature", description="Temperature parameter")


class _SlowCloudConfig(_CloudConfig):
    timeout: int | None = ConfigField(default=1200, ge=1, title="Timeout", description="Timeout in seconds")


class _CloudOutput(MockToolOutputBase):
    result: str = Field(description="Result payload")


class _CloudAssetOutput(MockToolOutputBase):
    structure: str = Field(description="Decoded structure text")
    scores: list[list[float]] = Field(description="Decoded score matrix")


class _PreprocessConfig(_CloudConfig):
    local_path: str | None = ConfigField(default=None, title="Local Path", description="Local path override")

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


@dataclass
class _StubAssetsNamespace:
    decoded_by_id: dict[str, Any] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def decode(self, ref: dict[str, Any]) -> Any:
        self.calls.append(ref)
        return self.decoded_by_id[ref["id"]]


class _StubProtoClient:
    """Fake proto_client.ProtoClient for tests."""

    last_instance: _StubProtoClient | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.tools = _StubToolsNamespace()
        self.assets = _StubAssetsNamespace()
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
    """Install a fake ``proto_client`` module + stub key so dispatch_to_cloud can construct a client."""
    fake_module = types.ModuleType("proto_client")
    fake_module.ProtoClient = _StubProtoClient  # type: ignore[attr-defined]
    fake_errors = types.ModuleType("proto_client.errors")
    fake_errors.ProtoAuthError = _FakeProtoAuthError  # type: ignore[attr-defined]
    fake_module.errors = fake_errors  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "proto_client", fake_module)
    monkeypatch.setitem(sys.modules, "proto_client.errors", fake_errors)
    monkeypatch.setenv("PROTO_API_KEY", "test-stub-key")

    # Clear the lru_cache so each test gets a fresh client.
    from proto_tools.cloud import _get_client

    _get_client.cache_clear()
    _StubProtoClient.last_instance = None
    yield _StubProtoClient
    _get_client.cache_clear()
    _StubProtoClient.last_instance = None


@pytest.fixture
def clean_registry():
    """Clean tool registry state for each test."""
    original_registry = ToolRegistry._registry.copy()
    ToolRegistry._registry.clear()
    yield ToolRegistry
    ToolRegistry._registry = original_registry


@pytest.fixture
def arm_stub_client(monkeypatch):
    """Helper: monkey-patch ``_StubProtoClient.__init__`` to seed tools/assets state at construction.

    Required because the dispatcher constructs ``ProtoClient`` lazily inside the
    call path, so we can't reach into ``last_instance`` before the SDK call has
    already happened. Each test calls ``arm_stub_client(setup=...)`` to inject
    output, raise_on_run, or asset state into the stub at construction time.
    """
    original_init = _StubProtoClient.__init__

    def _arm(setup):
        def _init_with_setup(self, **kwargs):
            original_init(self, **kwargs)
            setup(self)

        monkeypatch.setattr(_StubProtoClient, "__init__", _init_with_setup)

    return _arm


def _register_cloud_tool(
    registry,
    key: str,
    output_class: type[MockToolOutputBase] = _CloudOutput,
    config_class: type[BaseConfig] = _CloudConfig,
):
    """Register a non-local_cpu tool (``uses_gpu=True``) whose local impl fails — proves we routed to cloud."""

    def _must_not_run_locally(inputs, config, instance=None):
        del inputs, config, instance
        raise AssertionError(f"Local execution of {key!r} should not happen when device='cloud'")

    registry.register(
        key=key,
        label=key,
        category="test",
        input_class=_CloudInput,
        config_class=config_class,
        output_class=output_class,
        description=key,
        uses_gpu=True,
    )(_must_not_run_locally)
    return registry.get(key)


# ─ parse_device_string ───────────────────────────────────────────────────────


def test_parse_device_string_accepts_cloud():
    """parse_device_string returns DeviceSpec(kind='cloud') for 'cloud'."""
    from proto_tools.utils.device import parse_device_string

    spec = parse_device_string("cloud")
    assert spec.kind == "cloud"
    assert spec.devices == ["cloud"]
    assert spec.count == 1


# ─ device='cloud' dispatch ───────────────────────────────────────────────────


def test_device_cloud_returns_validated_output(fake_proto_client, arm_stub_client, clean_registry):
    """A registered tool with device='cloud' returns the validated cloud response."""
    arm_stub_client(lambda c: setattr(c.tools, "output_to_return", {"result": "from-api"}))
    spec = _register_cloud_tool(clean_registry, "api-tool-2")

    result = spec.function(_CloudInput(payload="hi"), _CloudConfig(device="cloud"))

    assert isinstance(result, _CloudOutput)
    assert result.result == "from-api"
    assert result.success is True
    assert result.tool_id == "api-tool-2"

    client = fake_proto_client.last_instance
    assert client is not None
    assert len(client.tools.calls) == 1
    call = client.tools.calls[0]
    assert call["tool_key"] == "api-tool-2"
    assert call["inputs"] == {"payload": "hi"}
    # device='cloud' is the client-side routing signal; the server picks its
    # own physical device, so the dispatcher strips device before sending.
    assert "device" not in call["config"]
    assert call["output_model"] is None


def test_device_cloud_uses_tool_config_timeout(fake_proto_client, arm_stub_client, clean_registry):
    """Cloud dispatch passes the selected tool config's effective timeout to the SDK."""
    arm_stub_client(lambda c: setattr(c.tools, "output_to_return", {"result": "from-api"}))
    spec = _register_cloud_tool(clean_registry, "slow-api-tool", config_class=_SlowCloudConfig)

    spec.function(_CloudInput(payload="hi"), _SlowCloudConfig(device="cloud"))

    client = fake_proto_client.last_instance
    assert client is not None
    assert client.tools.calls[0]["timeout"] == 1200.0


def test_cloud_output_assets_are_decoded_before_validation(fake_proto_client, arm_stub_client, clean_registry):
    """Cloud dispatch materializes output AssetRefs before applying the proto-tools output model."""

    def _seed(c):
        c.assets.decoded_by_id = {
            "structure_asset": "data_test\n#",
            "scores_asset": [[0.91, 0.87]],
        }
        c.tools.output_to_return = {
            "structure": {
                "id": "structure_asset",
                "kind": "output",
                "mime_type": "chemical/x-mmcif",
                "url": "https://api.test/api/v1/assets/structure_asset",
            },
            "scores": {
                "id": "scores_asset",
                "kind": "output",
                "mime_type": "application/json+gzip",
                "url": "https://api.test/api/v1/assets/scores_asset",
            },
        }

    arm_stub_client(_seed)
    spec = _register_cloud_tool(clean_registry, "asset-tool", output_class=_CloudAssetOutput)

    result = spec.function(_CloudInput(payload="hi"), _CloudConfig(device="cloud"))

    assert isinstance(result, _CloudAssetOutput)
    assert result.structure == "data_test\n#"
    assert result.scores == [[0.91, 0.87]]
    client = fake_proto_client.last_instance
    assert client is not None
    assert [ref["id"] for ref in client.assets.calls] == ["structure_asset", "scores_asset"]


def test_device_cloud_dispatches_before_preprocess(fake_proto_client, arm_stub_client, clean_registry):
    """device='cloud' sends the original request instead of running local preprocess first."""
    arm_stub_client(lambda c: setattr(c.tools, "output_to_return", {"result": "from-api"}))
    clean_registry.register(
        key="preprocess-tool",
        label="preprocess-tool",
        category="test",
        input_class=_CloudInput,
        config_class=_PreprocessConfig,
        output_class=_CloudOutput,
        description="preprocess-tool",
        uses_gpu=True,
    )(lambda inputs, config, instance=None: _CloudOutput(result="local"))
    spec = clean_registry.get("preprocess-tool")

    result = spec.function(_CloudInput(payload="raw"), _PreprocessConfig(device="cloud"))

    assert result.result == "from-api"
    client = fake_proto_client.last_instance
    assert client is not None
    call = client.tools.calls[0]
    assert call["inputs"] == {"payload": "raw"}
    assert "local_path" not in call["config"]


def test_device_cpu_runs_locally(fake_proto_client, clean_registry):
    """Non-cloud devices must still run locally, even when fake_proto_client is installed."""

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
    assert fake_proto_client.last_instance is None  # client never constructed


# ─ local_cpu no-op ───────────────────────────────────────────────────────────


def test_device_cloud_noops_for_local_cpu(fake_proto_client, clean_registry):
    """device='cloud' on a local_cpu tool runs locally without hitting the cloud client."""

    def _local_impl(inputs, config, instance=None):
        del config, instance
        return _CloudOutput(result=f"local:{inputs.payload}")

    clean_registry.register(
        key="local-cpu-tool",
        label="local-cpu-tool",
        category="test",
        input_class=_CloudInput,
        config_class=_CloudConfig,
        output_class=_CloudOutput,
        description="pure-python tool — no GPU, no standalone env",
    )(_local_impl)
    spec = clean_registry.get("local-cpu-tool")
    assert spec.local_cpu is True

    result = spec.function(_CloudInput(payload="hi"), _CloudConfig(device="cloud"))

    assert result.result == "local:hi"
    assert result.success is True
    assert fake_proto_client.last_instance is None


def test_device_cloud_noop_leaves_caller_config_untouched(clean_registry):
    """The wrapper's device rewrite must not mutate the caller's config object."""

    def _local_impl(inputs, config, instance=None):
        del inputs, instance
        assert config.device == "cpu"
        return _CloudOutput(result="ok")

    clean_registry.register(
        key="local-cpu-config-untouched",
        label="local-cpu-config-untouched",
        category="test",
        input_class=_CloudInput,
        config_class=_CloudConfig,
        output_class=_CloudOutput,
        description="verifies caller-config immutability under cloud no-op",
    )(_local_impl)
    spec = clean_registry.get("local-cpu-config-untouched")
    caller_config = _CloudConfig(device="cloud")

    spec.function(_CloudInput(payload="x"), caller_config)

    assert caller_config.device == "cloud"


# ─ failure handling ──────────────────────────────────────────────────────────


def test_no_api_key_raises_cloud_status(monkeypatch, clean_registry):
    """device='cloud' with no PROTO_API_KEY surfaces the coming-soon status."""
    monkeypatch.delenv("PROTO_API_KEY", raising=False)
    # Need a stub proto_client so the ImportError path doesn't fire first.
    fake_module = types.ModuleType("proto_client")
    fake_module.ProtoClient = _StubProtoClient  # type: ignore[attr-defined]
    fake_errors = types.ModuleType("proto_client.errors")
    fake_errors.ProtoAuthError = _FakeProtoAuthError  # type: ignore[attr-defined]
    fake_module.errors = fake_errors  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "proto_client", fake_module)
    monkeypatch.setitem(sys.modules, "proto_client.errors", fake_errors)
    spec = _register_cloud_tool(clean_registry, "no-key")

    with pytest.raises(NotImplementedError, match="coming soon"):
        spec.function(_CloudInput(payload="x"), _CloudConfig(device="cloud"))


def test_missing_proto_client_raises_install_hint(monkeypatch, clean_registry):
    """If proto-client isn't installed, the user gets a `pip install proto-client` hint."""
    import builtins

    monkeypatch.delitem(sys.modules, "proto_client", raising=False)
    monkeypatch.delitem(sys.modules, "proto_client.errors", raising=False)
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "proto_client" or name.startswith("proto_client."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    monkeypatch.setenv("PROTO_API_KEY", "test-stub-key")

    spec = _register_cloud_tool(clean_registry, "no-sdk")

    with pytest.raises(ImportError, match=r"pip install proto-client"):
        spec.function(_CloudInput(payload="x"), _CloudConfig(device="cloud"))


def test_invalid_api_key_raises_permission_error(fake_proto_client, arm_stub_client, clean_registry):
    """A ProtoAuthError surfaces as PermissionError with the invalid-key guidance."""
    arm_stub_client(lambda c: setattr(c.tools, "raise_on_run", _FakeProtoAuthError("unauthorized", status_code=401)))
    spec = _register_cloud_tool(clean_registry, "bad-key-tool")

    with pytest.raises(PermissionError, match="API key was rejected"):
        spec.function(_CloudInput(payload="x"), _CloudConfig(device="cloud"))


def test_transport_error_propagates_to_caller(fake_proto_client, arm_stub_client, clean_registry):
    """Non-auth SDK failures (network, 5xx, etc.) propagate unchanged — not buried as the placeholder."""
    arm_stub_client(lambda c: setattr(c.tools, "raise_on_run", ConnectionError("network down")))
    spec = _register_cloud_tool(clean_registry, "transport-failing-tool")

    with pytest.raises(ConnectionError, match="network down"):
        spec.function(_CloudInput(payload="x"), _CloudConfig(device="cloud"))


def test_real_tool_failure_propagates_to_caller(fake_proto_client, arm_stub_client, clean_registry):
    """A RuntimeError from proto-client (failed/cancelled job) propagates with the original message."""
    arm_stub_client(
        lambda c: setattr(c.tools, "raise_on_run", RuntimeError("Job fc-1 failed: ValueError: chain 'A' not present"))
    )
    spec = _register_cloud_tool(clean_registry, "failing-cloud-tool")

    with pytest.raises(RuntimeError, match="chain 'A'"):
        spec.function(_CloudInput(payload="x"), _CloudConfig(device="cloud"))


def test_unexpected_result_type_raises(fake_proto_client, arm_stub_client, clean_registry):
    """If proto-client returns a non-conforming result, cloud dispatch flags it."""
    arm_stub_client(lambda c: setattr(c.tools, "output_to_return", {"unexpected": "ok"}))
    spec = _register_cloud_tool(clean_registry, "type-check-tool")

    with pytest.raises(Exception, match="does not conform to _CloudOutput"):
        spec.function(_CloudInput(payload="x"), _CloudConfig(device="cloud"))


# ─ api_key override ──────────────────────────────────────────────────────────


def test_explicit_api_key_overrides_env(fake_proto_client, arm_stub_client, monkeypatch, clean_registry):
    """dispatch_to_cloud's api_key kwarg takes precedence over PROTO_API_KEY."""
    monkeypatch.setenv("PROTO_API_KEY", "env-key")
    arm_stub_client(lambda c: setattr(c.tools, "output_to_return", {"result": "ok"}))
    _register_cloud_tool(clean_registry, "explicit-key-tool")

    from proto_tools.cloud import dispatch_to_cloud

    dispatch_to_cloud(
        "explicit-key-tool",
        _CloudInput(payload="hi"),
        _CloudConfig(device="cloud"),
        api_key="explicit-key",
    )

    client = fake_proto_client.last_instance
    assert client is not None
    assert client.kwargs == {"api_key": "explicit-key"}
