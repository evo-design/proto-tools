"""tests/tool_infra_tests/test_cloud.py.

Tests for device="cloud" dispatch via proto_tools.cloud.dispatch_to_cloud.

``proto_client`` is stubbed via ``sys.modules`` so tests run without the
real SDK installed and without hitting a network.
"""

from __future__ import annotations

import logging
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
class _StubJobStatus:
    """Stand-in for proto_client.models.JobStatus enum entries."""

    value: str


@dataclass
class _StubJobStatusResponse:
    """Stand-in for proto_client.models.JobStatusResponse."""

    status: _StubJobStatus
    result: Any = None
    error: str | None = None


@dataclass
class _StubToolsNamespace:
    output_to_return: Any = None
    # Raised by ``submit`` so auth + transport failures surface like the real SDK.
    raise_on_run: BaseException | None = None
    final_status: str = "completed"
    status_sequence: list[str] = field(
        default_factory=list
    )  # statuses returned across successive get() calls; else final_status
    log_records: list[Any] = field(default_factory=list)
    submit_calls: list[dict[str, Any]] = field(default_factory=list)
    get_calls: list[dict[str, Any]] = field(default_factory=list)
    log_iter_calls: list[dict[str, Any]] = field(default_factory=list)

    # Back-compat alias so older assertions reading ``client.tools.calls`` still resolve to submit history.
    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.submit_calls

    def submit(
        self,
        tool_key: str,
        inputs: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> str:
        self.submit_calls.append({"tool_key": tool_key, "inputs": inputs, "config": config})
        if self.raise_on_run is not None:
            raise self.raise_on_run
        return f"stub-job-{tool_key}"

    def get(self, tool_key: str, job_id: str) -> _StubJobStatusResponse:
        self.get_calls.append({"tool_key": tool_key, "job_id": job_id})
        status_value = self.status_sequence.pop(0) if self.status_sequence else self.final_status
        return _StubJobStatusResponse(
            status=_StubJobStatus(status_value),
            result=self.output_to_return,
            error="stub failure" if status_value == "failed" else None,
        )

    def iter_job_logs(
        self,
        tool_key: str,
        job_id: str,
        *,
        follow: bool = False,
        level: list[str] | None = None,
        stream: list[str] | None = None,
        since: int | None = None,
        limit: int | None = None,
    ) -> Any:
        self.log_iter_calls.append(
            {"tool_key": tool_key, "job_id": job_id, "follow": follow, "level": level, "stream": stream}
        )
        yield from self.log_records


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


@pytest.fixture(autouse=True)
def _default_cloud_hostable(monkeypatch):
    """Default synthetic test tools (no ``license.yaml``) hostable so the gate doesn't block dispatch."""
    monkeypatch.setattr(ToolRegistry, "get_license", lambda key: {"redistribution": True})


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


# ─ cloud hostability gate (license redistribution) ───────────────────────────


def test_is_cloud_hostable_reads_redistribution(monkeypatch):
    """``is_cloud_hostable`` reflects the tool's license ``redistribution`` flag, failing closed."""
    from proto_tools.cloud import is_cloud_hostable

    monkeypatch.setattr(ToolRegistry, "get_license", lambda key: {"redistribution": True})
    assert is_cloud_hostable("any-tool") is True

    monkeypatch.setattr(ToolRegistry, "get_license", lambda key: {"redistribution": False})
    assert is_cloud_hostable("any-tool") is False

    # Missing license, missing field, and unknown key all fail closed.
    monkeypatch.setattr(ToolRegistry, "get_license", lambda key: None)
    assert is_cloud_hostable("any-tool") is False

    monkeypatch.setattr(ToolRegistry, "get_license", lambda key: {})
    assert is_cloud_hostable("any-tool") is False

    def _raise(key):
        raise ValueError("unknown tool")

    monkeypatch.setattr(ToolRegistry, "get_license", _raise)
    assert is_cloud_hostable("any-tool") is False


def test_is_cloud_hostable_fails_closed_on_malformed_license(monkeypatch):
    """A corrupt/unreadable license.yaml (a non-ValueError from get_license) blocks cloud instead of crashing the run."""
    import yaml

    from proto_tools.cloud import is_cloud_hostable

    def _raise_yaml(key):
        raise yaml.YAMLError("could not parse license.yaml")

    monkeypatch.setattr(ToolRegistry, "get_license", _raise_yaml)
    assert is_cloud_hostable("any-tool") is False

    def _raise_os(key):
        raise OSError("permission denied reading license.yaml")

    monkeypatch.setattr(ToolRegistry, "get_license", _raise_os)
    assert is_cloud_hostable("any-tool") is False


def test_cloud_blocks_non_redistributable_tool(fake_proto_client, arm_stub_client, monkeypatch, clean_registry):
    """device='cloud' on a non-redistributable tool fails fast with a clear error and never dispatches."""
    arm_stub_client(lambda c: setattr(c.tools, "output_to_return", {"result": "should-not-run"}))
    spec = _register_cloud_tool(clean_registry, "gated-tool")

    # This tool's license forbids redistribution → not hostable on Proto's cloud.
    monkeypatch.setattr(ToolRegistry, "get_license", lambda key: {"redistribution": False})

    with pytest.raises(ValueError, match="redistribution"):
        spec.function(_CloudInput(payload="x"), _CloudConfig(device="cloud"))

    # The gate is purely local: no client constructed, no submit attempted.
    assert fake_proto_client.last_instance is None


def test_cloud_blocks_unsupported_config(fake_proto_client, clean_registry):
    """device='cloud' with a config whose ``cloud_unsupported_reason()`` is non-None fails fast, never dispatches."""

    class _LocalOnlyConfig(_CloudConfig):
        def cloud_unsupported_reason(self) -> str | None:
            return "needs a local database file"

    spec = _register_cloud_tool(clean_registry, "local-only-config", config_class=_LocalOnlyConfig)

    with pytest.raises(ValueError, match="needs a local database file"):
        spec.function(_CloudInput(payload="x"), _LocalOnlyConfig(device="cloud"))

    # Fails fast locally: no client constructed, no submit attempted.
    assert fake_proto_client.last_instance is None


def test_base_config_cloud_unsupported_reason_defaults_none():
    """The default hook returns None so ordinary tools still dispatch to cloud."""
    assert _CloudConfig().cloud_unsupported_reason() is None


def test_local_file_tools_declare_cloud_unsupported():
    """Local-file tools (blast local-DB search, pyhmmer hmmscan/hmmsearch) declare themselves cloud-unsupported."""
    from proto_tools.tools.gene_annotation.pyhmmer.hmmscan import PyHmmscanConfig
    from proto_tools.tools.gene_annotation.pyhmmer.hmmsearch import PyHmmsearchConfig
    from proto_tools.tools.sequence_alignment.blast.blast_search import BlastSearchConfig

    assert BlastSearchConfig(search_mode="local", local_db="/db").cloud_unsupported_reason() is not None
    assert BlastSearchConfig(search_mode="online").cloud_unsupported_reason() is None  # online (NCBI) is cloud-OK
    assert PyHmmscanConfig().cloud_unsupported_reason() is not None
    assert PyHmmsearchConfig().cloud_unsupported_reason() is not None


def test_local_resource_override_configs_declare_cloud_unsupported():
    """Optional local-path overrides (weights dir, artifact, denoiser YAML) fail cloud only when set."""
    from proto_tools.tools.causal_models.evo2.evo2_sample import Evo2SampleConfig
    from proto_tools.tools.causal_models.evo2.evo2_score import Evo2ScoringConfig
    from proto_tools.tools.causal_models.progen2.progen2_sample import ProGen2SampleConfig
    from proto_tools.tools.causal_models.progen2.progen2_score import ProGen2ScoringConfig
    from proto_tools.tools.sequence_scoring.malinois.malinois_score import (
        MalinoisGradientConfig,
        MalinoisScoreConfig,
    )
    from proto_tools.tools.structure_dynamics.bioemu.bioemu_sample import BioEmuConfig

    # Defaults dispatch fine (managed download / cache / unset).
    for cfg in (
        Evo2SampleConfig(),
        Evo2ScoringConfig(),
        ProGen2SampleConfig(),
        ProGen2ScoringConfig(),
        MalinoisScoreConfig(),
        MalinoisGradientConfig(),
        BioEmuConfig(),
    ):
        assert cfg.cloud_unsupported_reason() is None

    # An explicitly-set local override is rejected before dispatch.
    assert Evo2SampleConfig(local_path="/w").cloud_unsupported_reason() is not None
    assert Evo2ScoringConfig(local_path="/w").cloud_unsupported_reason() is not None
    assert ProGen2SampleConfig(local_path="/w").cloud_unsupported_reason() is not None
    assert ProGen2ScoringConfig(local_path="/w").cloud_unsupported_reason() is not None
    assert MalinoisScoreConfig(artifact_path="/a").cloud_unsupported_reason() is not None
    assert MalinoisScoreConfig(malinois_dir="/d").cloud_unsupported_reason() is not None
    assert MalinoisGradientConfig(malinois_dir="/d").cloud_unsupported_reason() is not None
    assert BioEmuConfig(denoiser_config="steer.yaml").cloud_unsupported_reason() is not None


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


def test_device_cloud_uses_tool_config_timeout(fake_proto_client, arm_stub_client, monkeypatch, clean_registry):
    """Cloud dispatch passes the selected tool config's effective timeout to the poll loop."""
    arm_stub_client(lambda c: setattr(c.tools, "output_to_return", {"result": "from-api"}))
    spec = _register_cloud_tool(clean_registry, "slow-api-tool", config_class=_SlowCloudConfig)

    # Spy on _poll_until_terminal — the dispatcher's poll loop owns the timeout now (no single SDK call carries it).
    import proto_tools.cloud as cloud_mod

    captured: dict[str, Any] = {}
    real_poll = cloud_mod._poll_until_terminal

    def _spy_poll(client, key, job_id, poll_interval, timeout):
        captured["timeout"] = timeout
        captured["poll_interval"] = poll_interval
        return real_poll(client, key, job_id, poll_interval, timeout)

    monkeypatch.setattr(cloud_mod, "_poll_until_terminal", _spy_poll)

    spec.function(_CloudInput(payload="hi"), _SlowCloudConfig(device="cloud"))

    assert captured["timeout"] == 1200.0


class _NoTimeoutCloudConfig(_CloudConfig):
    timeout: int | None = ConfigField(default=None, ge=1, title="Timeout", description="No inference cap")


def test_device_cloud_none_timeout_polls_until_terminal(
    fake_proto_client, arm_stub_client, monkeypatch, clean_registry
):
    """config.timeout=None must poll until terminal (no client deadline), not cap at a fixed value."""
    arm_stub_client(lambda c: setattr(c.tools, "output_to_return", {"result": "from-api"}))
    spec = _register_cloud_tool(clean_registry, "unbounded-api-tool", config_class=_NoTimeoutCloudConfig)

    import proto_tools.cloud as cloud_mod

    captured: dict[str, Any] = {}
    real_poll = cloud_mod._poll_until_terminal

    def _spy_poll(client, key, job_id, poll_interval, timeout):
        captured["timeout"] = timeout
        return real_poll(client, key, job_id, poll_interval, timeout)

    monkeypatch.setattr(cloud_mod, "_poll_until_terminal", _spy_poll)

    spec.function(_CloudInput(payload="hi"), _NoTimeoutCloudConfig(device="cloud"))

    assert captured["timeout"] is None


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


def test_device_cloud_noops_for_local_cpu_without_cloud_hostability(
    monkeypatch,
    fake_proto_client,
    clean_registry,
):
    """device='cloud' on a local_cpu tool runs locally without cloud hostability or transport."""

    def _get_license_should_not_run(key):
        raise AssertionError(f"local_cpu cloud no-op should not check cloud hostability for {key!r}")

    monkeypatch.setattr(ToolRegistry, "get_license", _get_license_should_not_run)

    def _local_impl(inputs, config, instance=None):
        del instance
        assert config.device == "cpu"
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
    """device='cloud' with no PROTO_API_KEY surfaces the set-up-your-key guidance."""
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

    with pytest.raises(NotImplementedError, match="workspace/keys"):
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

    with pytest.raises(PermissionError, match="couldn't validate"):
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


# ─ log streaming ─────────────────────────────────────────────────────────────


@dataclass
class _StubLogRecord:
    """Stand-in for proto_client.models.LogRecord."""

    msg: str
    level: str = "info"
    stream: str = "system"
    seq: int = 0
    type: str = "record"
    update_status: bool = False


@dataclass
class _StubLogsEnd:
    """Stand-in for proto_client.models.LogsEnd terminator."""

    type: str = "end"
    reason: str = "completed"
    final_seq: int = 0


def test_cloud_streams_remote_logs_through_local_logger(
    fake_proto_client, arm_stub_client, caplog, monkeypatch, clean_registry
):
    """LogRecord rows from the server are replayed via ``proto_tools.cloud.remote`` so the user sees live progress."""
    # See verbose-mapping test: conftest pins PROTO_WORKER_VERBOSE=3 globally.
    monkeypatch.delenv("PROTO_WORKER_VERBOSE", raising=False)

    def _seed(c):
        c.tools.output_to_return = {"result": "ok"}
        c.tools.log_records = [
            _StubLogRecord(msg="Starting ESMFold", level="info", stream="system", seq=1),
            _StubLogRecord(msg="Folded 1/3", level="info", stream="stdout", seq=2),
            _StubLogRecord(msg="GPU OOM averted", level="warning", stream="system", seq=3),
            _StubLogsEnd(final_seq=3),
            # After the LogsEnd terminator the streaming helper must stop — this record must never appear in caplog.
            _StubLogRecord(msg="should not be replayed", level="info", stream="system", seq=4),
        ]

    arm_stub_client(_seed)
    spec = _register_cloud_tool(clean_registry, "streaming-tool")

    caplog.set_level(logging.DEBUG, logger="proto_tools.cloud.remote")
    spec.function(_CloudInput(payload="x"), _CloudConfig(device="cloud", verbose=1))

    messages = [r.message for r in caplog.records if r.name == "proto_tools.cloud.remote"]
    assert "Starting ESMFold" in messages
    assert "Folded 1/3" in messages
    assert "GPU OOM averted" in messages
    assert "should not be replayed" not in messages

    # The warning record must map to Python WARNING, not INFO.
    levels = {r.message: r.levelno for r in caplog.records if r.name == "proto_tools.cloud.remote"}
    assert levels["GPU OOM averted"] == logging.WARNING
    assert levels["Starting ESMFold"] == logging.INFO

    # iter_job_logs must have been called once with follow=True and the verbose=1 filters.
    client = fake_proto_client.last_instance
    assert client is not None
    assert len(client.tools.log_iter_calls) == 1
    call = client.tools.log_iter_calls[0]
    assert call["follow"] is True
    assert call["stream"] == ["system", "stdout"]
    assert call["level"] == ["info", "notice", "warning", "error", "critical", "alert", "emergency"]


def test_cloud_verbose_levels_map_to_server_filters(fake_proto_client, arm_stub_client, monkeypatch, clean_registry):
    """``config.verbose`` translates 1:1 to ``iter_job_logs`` level/stream filters so cloud mirrors local verbosity."""
    # Clear PROTO_WORKER_VERBOSE (conftest pins it to 3) so config.verbose drives each case.
    monkeypatch.delenv("PROTO_WORKER_VERBOSE", raising=False)
    arm_stub_client(lambda c: setattr(c.tools, "output_to_return", {"result": "ok"}))

    for verbose, expected_level, expected_stream in [
        # verbose=0 is "quiet": cloud streams only warnings+ (spinner conveys liveness instead of info chatter), mirroring local quiet mode.
        (0, ["warning", "error", "critical", "alert", "emergency"], ["system"]),
        (1, ["info", "notice", "warning", "error", "critical", "alert", "emergency"], ["system", "stdout"]),
        (2, None, ["system", "stdout"]),
        (3, None, None),
    ]:
        key = f"verbose-tool-{verbose}"
        spec = _register_cloud_tool(clean_registry, key)
        spec.function(_CloudInput(payload="x"), _CloudConfig(device="cloud", verbose=verbose))
        call = fake_proto_client.last_instance.tools.log_iter_calls[-1]
        assert call["level"] == expected_level, f"verbose={verbose}"
        assert call["stream"] == expected_stream, f"verbose={verbose}"


def test_cloud_dispatch_opens_cloud_spinner(fake_proto_client, arm_stub_client, monkeypatch, clean_registry):
    """The cloud dispatch path opens a progress_bar with the computer↔cloud spinner so a glance shows it's a cloud run."""
    arm_stub_client(lambda c: setattr(c.tools, "output_to_return", {"result": "ok"}))
    spec = _register_cloud_tool(clean_registry, "spinner-tool")

    # Spy on ``progress_bar`` to capture how dispatch invokes it (kwargs are the contract).
    import proto_tools.cloud as cloud_mod

    captured: dict[str, Any] = {}
    real_progress_bar = cloud_mod.progress_bar

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        return real_progress_bar(*args, **kwargs)

    monkeypatch.setattr(cloud_mod, "progress_bar", _spy)

    spec.function(_CloudInput(payload="x"), _CloudConfig(device="cloud"))

    # Emoji terminals get the "cloud" pulse style (no separate prefix); non-UTF falls back to dots + [cloud].
    assert (captured.get("spinner_style"), captured.get("prefix")) in {("cloud", None), ("dots", "[cloud]")}
    # The description shows the live phase, opening at "queued".
    assert captured.get("desc") == "queued"
    # status-only spinner: no progress bar widget, only the spinner + description
    assert captured.get("show_bar") is False


def test_cloud_spinner_substatus_tracks_job_status(fake_proto_client, arm_stub_client, monkeypatch, clean_registry):
    """The cloud spinner's substatus follows job status while polling (queued → running) before the result returns."""
    import proto_tools.cloud as cloud_mod

    seen: list[str] = []
    monkeypatch.setattr(cloud_mod, "set_substatus", lambda msg, *a, **k: seen.append(msg))
    monkeypatch.setattr(cloud_mod.time, "sleep", lambda _s: None)  # don't actually wait between polls

    def _seed(c):
        c.tools.output_to_return = {"result": "ok"}
        c.tools.status_sequence = ["running", "completed"]  # first poll running, then terminal

    arm_stub_client(_seed)
    spec = _register_cloud_tool(clean_registry, "status-tool")
    spec.function(_CloudInput(payload="x"), _CloudConfig(device="cloud"))

    assert "running" in seen  # the running status was mapped onto the spinner substatus


def test_strip_logger_prefix():
    """The worker logger-name prefix is removed from phase markers; ordinary 'word:' messages are kept."""
    from proto_tools.cloud import _strip_logger_prefix

    assert (
        _strip_logger_prefix("proto_tools.worker.esmfold.esmfold-prediction: Loading ESMFold model: v1 on cuda")
        == "Loading ESMFold model: v1 on cuda"
    )
    assert _strip_logger_prefix("proto_tools.utils.progress: Running esmfold") == "Running esmfold"
    # No dotted logger head -> leave untouched (message may legitimately start with 'word: ').
    assert _strip_logger_prefix("Done: 5 sequences") == "Done: 5 sequences"
    assert _strip_logger_prefix("Folding 1 complex(es), num_recycles=3") == "Folding 1 complex(es), num_recycles=3"


def test_cloud_replay_strips_logger_prefix_on_plain_lines(
    fake_proto_client, arm_stub_client, monkeypatch, caplog, clean_registry
):
    """Cloud log lines read like local: the worker logger-name prefix is stripped on every replayed line."""
    monkeypatch.delenv("PROTO_WORKER_VERBOSE", raising=False)

    def _seed(c):
        c.tools.output_to_return = {"result": "ok"}
        c.tools.log_records = [
            _StubLogRecord(msg="proto_tools.utils.progress: Starting worker", level="info", stream="stdout", seq=1),
            _StubLogsEnd(final_seq=1),
        ]

    arm_stub_client(_seed)
    spec = _register_cloud_tool(clean_registry, "prefix-tool")
    caplog.set_level(logging.DEBUG, logger="proto_tools.cloud.remote")
    spec.function(_CloudInput(payload="x"), _CloudConfig(device="cloud", verbose=1))

    msgs = [r.message for r in caplog.records if r.name == "proto_tools.cloud.remote"]
    assert "Starting worker" in msgs
    assert "proto_tools.utils.progress: Starting worker" not in msgs


def test_cloud_update_status_records_drive_spinner_substatus(
    fake_proto_client, arm_stub_client, monkeypatch, caplog, clean_registry
):
    """update_status records replay with the flag so the local spinner handler drives the bar; plain lines don't."""
    monkeypatch.delenv("PROTO_WORKER_VERBOSE", raising=False)

    def _seed(c):
        c.tools.output_to_return = {"result": "ok"}
        c.tools.log_records = [
            _StubLogRecord(msg="Loading checkpoint", level="info", stream="stdout", seq=1, update_status=True),
            _StubLogRecord(msg="Folded 1/3", level="info", stream="stdout", seq=2),
            _StubLogsEnd(final_seq=2),
        ]

    arm_stub_client(_seed)
    spec = _register_cloud_tool(clean_registry, "phase-tool")
    caplog.set_level(logging.DEBUG, logger="proto_tools.cloud.remote")
    spec.function(_CloudInput(payload="x"), _CloudConfig(device="cloud", verbose=1))

    # The flag rides through onto the replayed record; SpinnerFromLogsHandler (tested separately) routes it to the bar.
    by_msg = {r.message: r for r in caplog.records if r.name == "proto_tools.cloud.remote"}
    assert getattr(by_msg["Loading checkpoint"], "update_status", False) is True
    assert getattr(by_msg["Folded 1/3"], "update_status", False) is False


def test_cloud_log_stream_failure_does_not_break_run(fake_proto_client, arm_stub_client, caplog, clean_registry):
    """A broken log stream must not fail the run — the result is what matters."""

    def _seed(c):
        c.tools.output_to_return = {"result": "ok"}

        def _broken_logs(*args, **kwargs):
            raise ConnectionError("log stream down")
            yield  # unreachable; keep this a generator function

        c.tools.iter_job_logs = _broken_logs

    arm_stub_client(_seed)
    spec = _register_cloud_tool(clean_registry, "log-fail-tool")

    caplog.set_level(logging.DEBUG, logger="proto_tools.cloud")
    result = spec.function(_CloudInput(payload="x"), _CloudConfig(device="cloud"))
    assert result.result == "ok"


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
