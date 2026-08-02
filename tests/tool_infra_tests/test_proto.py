"""tests/tool_infra_tests/test_cloud.py.

Tests for device="proto" dispatch via proto_tools.proto.dispatch_to_proto.

The absorbed client is stubbed by attribute so tests run without touching the
real SDK installed and without hitting a network.
"""

from __future__ import annotations

import logging
import types
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import Field

from proto_tools.tools.tool_registry import ToolRegistry
from proto_tools.utils import BaseConfig, ConfigField
from proto_tools.utils.base_config import INTERNAL_STATE_KEY
from proto_tools.utils.progress import remote_connecting_status
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
    """Config whose preprocess rewrites the payload, making its effect visible in the dispatch."""

    local_path: str | None = ConfigField(default=None, title="Local Path", description="Local path override")

    def preprocess(self, inputs: BaseToolInput) -> BaseToolInput:
        """Mark the payload so the test can tell prepared inputs from raw ones."""
        return type(inputs)(payload=f"prepared:{inputs.payload}")


@dataclass
class _StubJobStatus:
    """Stand-in for the client's job-status values."""

    value: str


@dataclass
class _StubJobStatusResponse:
    """Stand-in for the client's job-status response."""

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
    """Fake ProtoClient for tests."""

    last_instance: _StubProtoClient | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.tools = _StubToolsNamespace()
        self.assets = _StubAssetsNamespace()
        _StubProtoClient.last_instance = self


class _FakeProtoAuthError(Exception):
    """Stand-in for ``ProtoAuthError`` (401/403)."""

    def __init__(self, message: str, *, status_code: int, request_id: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.request_id = request_id


@pytest.fixture
def fake_proto_client(monkeypatch):
    """Point the client at a stub and set a key so dispatch_to_proto can construct one."""
    monkeypatch.setattr("proto_tools.proto._client.ProtoClient", _StubProtoClient)
    monkeypatch.setattr("proto_tools.proto._client.ProtoAuthError", _FakeProtoAuthError)
    monkeypatch.setenv("PROTO_API_KEY", "test-stub-key")

    # Clear the lru_cache so each test gets a fresh client.
    from proto_tools.proto import _get_client

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
    call path, so we cannot reach into ``last_instance`` before the call has
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
        raise AssertionError(f"Local execution of {key!r} should not happen when device='proto'")

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


def test_is_proto_hostable_reads_redistribution(monkeypatch):
    """``is_proto_hostable`` reflects the tool's license ``redistribution`` flag, failing closed."""
    from proto_tools.proto import is_proto_hostable

    monkeypatch.setattr(ToolRegistry, "get_license", lambda key: {"redistribution": True})
    assert is_proto_hostable("any-tool") is True

    monkeypatch.setattr(ToolRegistry, "get_license", lambda key: {"redistribution": False})
    assert is_proto_hostable("any-tool") is False

    # Missing license, missing field, and unknown key all fail closed.
    monkeypatch.setattr(ToolRegistry, "get_license", lambda key: None)
    assert is_proto_hostable("any-tool") is False

    monkeypatch.setattr(ToolRegistry, "get_license", lambda key: {})
    assert is_proto_hostable("any-tool") is False

    def _raise(key):
        raise ValueError("unknown tool")

    monkeypatch.setattr(ToolRegistry, "get_license", _raise)
    assert is_proto_hostable("any-tool") is False


def test_is_proto_hostable_fails_closed_on_malformed_license(monkeypatch):
    """A corrupt/unreadable license.yaml (a non-ValueError from get_license) blocks cloud instead of crashing the run."""
    import yaml

    from proto_tools.proto import is_proto_hostable

    def _raise_yaml(key):
        raise yaml.YAMLError("could not parse license.yaml")

    monkeypatch.setattr(ToolRegistry, "get_license", _raise_yaml)
    assert is_proto_hostable("any-tool") is False

    def _raise_os(key):
        raise OSError("permission denied reading license.yaml")

    monkeypatch.setattr(ToolRegistry, "get_license", _raise_os)
    assert is_proto_hostable("any-tool") is False


def test_proto_blocks_non_redistributable_tool(fake_proto_client, arm_stub_client, monkeypatch, clean_registry):
    """device='proto' on a non-redistributable tool fails fast with a clear error and never dispatches."""
    arm_stub_client(lambda c: setattr(c.tools, "output_to_return", {"result": "should-not-run"}))
    spec = _register_cloud_tool(clean_registry, "gated-tool")

    # This tool's license forbids redistribution → not hostable on Proto's cloud.
    monkeypatch.setattr(ToolRegistry, "get_license", lambda key: {"redistribution": False})

    with pytest.raises(ValueError, match="redistribution"):
        spec.function(_CloudInput(payload="x"), _CloudConfig(device="proto"))

    # The gate is purely local: no client constructed, no submit attempted.
    assert fake_proto_client.last_instance is None


def test_proto_blocks_unsupported_config(fake_proto_client, clean_registry):
    """device='proto' with a config whose ``remote_unsupported_reason()`` is non-None fails fast, never dispatches."""

    class _LocalOnlyConfig(_CloudConfig):
        def remote_unsupported_reason(self, device: str) -> str | None:
            return "needs a local database file"

    spec = _register_cloud_tool(clean_registry, "local-only-config", config_class=_LocalOnlyConfig)

    with pytest.raises(ValueError, match="needs a local database file"):
        spec.function(_CloudInput(payload="x"), _LocalOnlyConfig(device="proto"))

    # Fails fast locally: no client constructed, no submit attempted.
    assert fake_proto_client.last_instance is None


def test_base_config_remote_unsupported_reason_defaults_none():
    """The default hook returns None so ordinary tools still dispatch to cloud."""
    assert _CloudConfig().remote_unsupported_reason("proto") is None


def test_local_file_tools_declare_cloud_unsupported():
    """Local-file tools (blast local-DB search, pyhmmer hmmscan/hmmsearch) declare themselves cloud-unsupported."""
    from proto_tools.tools.gene_annotation.pyhmmer.hmmscan import PyHmmscanConfig
    from proto_tools.tools.gene_annotation.pyhmmer.hmmsearch import PyHmmsearchConfig
    from proto_tools.tools.sequence_alignment.blast.blast_search import BlastSearchConfig

    assert BlastSearchConfig(search_mode="local", local_db="/db").remote_unsupported_reason("proto") is not None
    assert (
        BlastSearchConfig(search_mode="online").remote_unsupported_reason("proto") is None
    )  # online (NCBI) is cloud-OK
    assert PyHmmscanConfig().remote_unsupported_reason("proto") is not None
    assert PyHmmsearchConfig().remote_unsupported_reason("proto") is not None


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
        assert cfg.remote_unsupported_reason("proto") is None

    # An explicitly-set local override is rejected before dispatch.
    assert Evo2SampleConfig(local_path="/w").remote_unsupported_reason("proto") is not None
    assert Evo2ScoringConfig(local_path="/w").remote_unsupported_reason("proto") is not None
    assert ProGen2SampleConfig(local_path="/w").remote_unsupported_reason("proto") is not None
    assert ProGen2ScoringConfig(local_path="/w").remote_unsupported_reason("proto") is not None
    assert MalinoisScoreConfig(artifact_path="/a").remote_unsupported_reason("proto") is not None
    assert MalinoisScoreConfig(malinois_dir="/d").remote_unsupported_reason("proto") is not None
    assert MalinoisGradientConfig(malinois_dir="/d").remote_unsupported_reason("proto") is not None
    assert BioEmuConfig(denoiser_config="steer.yaml").remote_unsupported_reason("proto") is not None


# ─ parse_device_string ───────────────────────────────────────────────────────


def test_parse_device_string_accepts_cloud():
    """parse_device_string returns DeviceSpec(kind='cloud') for 'cloud'."""
    from proto_tools.utils.device import parse_device_string

    spec = parse_device_string("proto")
    assert spec.kind == "proto"
    assert spec.devices == ["proto"]
    assert spec.count == 1


# ─ device='proto' dispatch ───────────────────────────────────────────────────


def test_device_cloud_returns_validated_output(fake_proto_client, arm_stub_client, clean_registry):
    """A registered tool with device='proto' returns the validated cloud response."""
    arm_stub_client(lambda c: setattr(c.tools, "output_to_return", {"result": "from-api"}))
    spec = _register_cloud_tool(clean_registry, "api-tool-2")

    result = spec.function(_CloudInput(payload="hi"), _CloudConfig(device="proto"))

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
    # device='proto' is the client-side routing signal; the server picks its
    # own physical device, so the dispatcher strips device before sending.
    assert "device" not in call["config"]


def test_device_cloud_uses_tool_config_timeout(fake_proto_client, arm_stub_client, monkeypatch, clean_registry):
    """Cloud dispatch passes the selected tool config's effective timeout to the poll loop."""
    arm_stub_client(lambda c: setattr(c.tools, "output_to_return", {"result": "from-api"}))
    spec = _register_cloud_tool(clean_registry, "slow-api-tool", config_class=_SlowCloudConfig)

    # Spy on _poll_until_terminal — the dispatcher's poll loop owns the timeout now (no single SDK call carries it).
    import proto_tools.proto as cloud_mod

    captured: dict[str, Any] = {}
    real_poll = cloud_mod._poll_until_terminal

    def _spy_poll(client, key, job_id, poll_interval, timeout):
        captured["timeout"] = timeout
        captured["poll_interval"] = poll_interval
        return real_poll(client, key, job_id, poll_interval, timeout)

    monkeypatch.setattr(cloud_mod, "_poll_until_terminal", _spy_poll)

    spec.function(_CloudInput(payload="hi"), _SlowCloudConfig(device="proto"))

    assert captured["timeout"] == 1200.0


class _NoTimeoutCloudConfig(_CloudConfig):
    timeout: int | None = ConfigField(default=None, ge=1, title="Timeout", description="No inference cap")


def test_device_cloud_none_timeout_polls_until_terminal(
    fake_proto_client, arm_stub_client, monkeypatch, clean_registry
):
    """config.timeout=None must poll until terminal (no client deadline), not cap at a fixed value."""
    arm_stub_client(lambda c: setattr(c.tools, "output_to_return", {"result": "from-api"}))
    spec = _register_cloud_tool(clean_registry, "unbounded-api-tool", config_class=_NoTimeoutCloudConfig)

    import proto_tools.proto as cloud_mod

    captured: dict[str, Any] = {}
    real_poll = cloud_mod._poll_until_terminal

    def _spy_poll(client, key, job_id, poll_interval, timeout):
        captured["timeout"] = timeout
        return real_poll(client, key, job_id, poll_interval, timeout)

    monkeypatch.setattr(cloud_mod, "_poll_until_terminal", _spy_poll)

    spec.function(_CloudInput(payload="hi"), _NoTimeoutCloudConfig(device="proto"))

    assert captured["timeout"] is None


def test_proto_output_assets_are_decoded_before_validation(fake_proto_client, arm_stub_client, clean_registry):
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

    result = spec.function(_CloudInput(payload="hi"), _CloudConfig(device="proto"))

    assert isinstance(result, _CloudAssetOutput)
    assert result.structure == "data_test\n#"
    assert result.scores == [[0.91, 0.87]]
    client = fake_proto_client.last_instance
    assert client is not None
    assert [ref["id"] for ref in client.assets.calls] == ["structure_asset", "scores_asset"]


def test_device_cloud_preprocesses_before_dispatching(fake_proto_client, arm_stub_client, clean_registry):
    """device='proto' prepares the request locally and tells the server preprocess already ran."""
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

    result = spec.function(_CloudInput(payload="raw"), _PreprocessConfig(device="proto"))

    assert result.result == "from-api"
    client = fake_proto_client.last_instance
    assert client is not None
    call = client.tools.calls[0]
    assert call["inputs"] == {"payload": "prepared:raw"}, "the server should receive prepared inputs"
    envelope = call["config"][INTERNAL_STATE_KEY]
    # Asserted per key rather than whole: the envelope carries whatever framework state has to
    # cross the boundary, so pinning it exactly breaks every time something is added.
    assert envelope["preprocess_completed"] is True
    assert envelope["client_identity"], "a hosted worker needs to know who it is acting for"
    assert "local_path" not in call["config"]


def test_device_cpu_runs_locally(fake_proto_client, clean_registry):
    """Local devices must still run locally, even with a key configured and the client stubbed."""

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
    """device='proto' on a local_cpu tool runs locally without cloud hostability or transport."""

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

    result = spec.function(_CloudInput(payload="hi"), _CloudConfig(device="proto"))

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
    caller_config = _CloudConfig(device="proto")

    spec.function(_CloudInput(payload="x"), caller_config)

    assert caller_config.device == "proto"


# ─ failure handling ──────────────────────────────────────────────────────────


def test_no_api_key_raises_cloud_status(monkeypatch, clean_registry):
    """device='proto' with no PROTO_API_KEY surfaces the set-up-your-key guidance."""
    monkeypatch.delenv("PROTO_API_KEY", raising=False)
    monkeypatch.setattr("proto_tools.proto._client.ProtoClient", _StubProtoClient)
    monkeypatch.setattr("proto_tools.proto._client.ProtoAuthError", _FakeProtoAuthError)
    spec = _register_cloud_tool(clean_registry, "no-key")

    with pytest.raises(NotImplementedError, match="workspace/keys"):
        spec.function(_CloudInput(payload="x"), _CloudConfig(device="proto"))


def test_invalid_api_key_raises_permission_error(fake_proto_client, arm_stub_client, clean_registry):
    """A ProtoAuthError surfaces as PermissionError with the invalid-key guidance."""
    arm_stub_client(lambda c: setattr(c.tools, "raise_on_run", _FakeProtoAuthError("unauthorized", status_code=401)))
    spec = _register_cloud_tool(clean_registry, "bad-key-tool")

    with pytest.raises(PermissionError, match="couldn't validate"):
        spec.function(_CloudInput(payload="x"), _CloudConfig(device="proto"))


def test_transport_error_propagates_to_caller(fake_proto_client, arm_stub_client, clean_registry):
    """Non-auth SDK failures (network, 5xx, etc.) propagate unchanged — not buried as the placeholder."""
    arm_stub_client(lambda c: setattr(c.tools, "raise_on_run", ConnectionError("network down")))
    spec = _register_cloud_tool(clean_registry, "transport-failing-tool")

    with pytest.raises(ConnectionError, match="network down"):
        spec.function(_CloudInput(payload="x"), _CloudConfig(device="proto"))


def test_real_tool_failure_propagates_to_caller(fake_proto_client, arm_stub_client, clean_registry):
    """A RuntimeError from a failed or cancelled job propagates with the original message."""
    arm_stub_client(
        lambda c: setattr(c.tools, "raise_on_run", RuntimeError("Job fc-1 failed: ValueError: chain 'A' not present"))
    )
    spec = _register_cloud_tool(clean_registry, "failing-cloud-tool")

    with pytest.raises(RuntimeError, match="chain 'A'"):
        spec.function(_CloudInput(payload="x"), _CloudConfig(device="proto"))


def test_unexpected_result_type_raises(fake_proto_client, arm_stub_client, clean_registry):
    """If the server returns a non-conforming result, remote dispatch flags it."""
    arm_stub_client(lambda c: setattr(c.tools, "output_to_return", {"unexpected": "ok"}))
    spec = _register_cloud_tool(clean_registry, "type-check-tool")

    with pytest.raises(Exception, match="does not conform to _CloudOutput"):
        spec.function(_CloudInput(payload="x"), _CloudConfig(device="proto"))


# ─ log streaming ─────────────────────────────────────────────────────────────


@dataclass
class _StubLogRecord:
    """Stand-in for a streamed log record."""

    msg: str
    level: str = "info"
    stream: str = "system"
    seq: int = 0
    type: str = "record"
    update_status: bool = False


@dataclass
class _StubLogsEnd:
    """Stand-in for the terminator that ends a log stream."""

    type: str = "end"
    reason: str = "completed"
    final_seq: int = 0


def test_proto_streams_remote_logs_through_local_logger(
    fake_proto_client, arm_stub_client, caplog, monkeypatch, clean_registry
):
    """LogRecord rows from the server are replayed via ``proto_tools.proto.remote`` so the user sees live progress."""
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

    caplog.set_level(logging.DEBUG, logger="proto_tools.proto.remote")
    spec.function(_CloudInput(payload="x"), _CloudConfig(device="proto", verbose=1))

    messages = [r.message for r in caplog.records if r.name == "proto_tools.proto.remote"]
    assert "Starting ESMFold" in messages
    assert "Folded 1/3" in messages
    assert "GPU OOM averted" in messages
    assert "should not be replayed" not in messages

    # The warning record must map to Python WARNING, not INFO.
    levels = {r.message: r.levelno for r in caplog.records if r.name == "proto_tools.proto.remote"}
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


def test_proto_verbose_levels_map_to_server_filters(fake_proto_client, arm_stub_client, monkeypatch, clean_registry):
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
        spec.function(_CloudInput(payload="x"), _CloudConfig(device="proto", verbose=verbose))
        call = fake_proto_client.last_instance.tools.log_iter_calls[-1]
        assert call["level"] == expected_level, f"verbose={verbose}"
        assert call["stream"] == expected_stream, f"verbose={verbose}"


def test_proto_dispatch_opens_cloud_spinner(fake_proto_client, arm_stub_client, monkeypatch, clean_registry):
    """The cloud dispatch path opens a progress_bar with the computer↔cloud spinner so a glance shows it's a cloud run."""
    arm_stub_client(lambda c: setattr(c.tools, "output_to_return", {"result": "ok"}))
    spec = _register_cloud_tool(clean_registry, "spinner-tool")

    # Spy where the bar is now opened: both remote devices share ``remote_progress``, so the
    # call lives in utils.progress rather than in either device's module.
    import proto_tools.utils.progress as progress_mod

    captured: dict[str, Any] = {}
    real_progress_bar = progress_mod.progress_bar

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        return real_progress_bar(*args, **kwargs)

    monkeypatch.setattr(progress_mod, "progress_bar", _spy)

    spec.function(_CloudInput(payload="x"), _CloudConfig(device="proto"))

    # Emoji terminals get the "proto" pulse style (no separate prefix); non-UTF falls back to dots + [proto].
    assert (captured.get("spinner_style"), captured.get("prefix")) in {("proto", None), ("dots", "[proto]")}
    # Opens by naming the backend being waited on; the poll loop replaces it with real job status.
    assert captured.get("desc") == remote_connecting_status("proto")
    # status-only spinner: no progress bar widget, only the spinner + description
    assert captured.get("show_bar") is False


def test_proto_spinner_substatus_tracks_job_status(fake_proto_client, arm_stub_client, monkeypatch, clean_registry):
    """The cloud spinner's substatus follows job status while polling (queued → running) before the result returns."""
    import proto_tools.proto as cloud_mod

    seen: list[str] = []
    monkeypatch.setattr(cloud_mod, "set_substatus", lambda msg, *a, **k: seen.append(msg))
    monkeypatch.setattr(cloud_mod.time, "sleep", lambda _s: None)  # don't actually wait between polls

    def _seed(c):
        c.tools.output_to_return = {"result": "ok"}
        c.tools.status_sequence = ["running", "completed"]  # first poll running, then terminal

    arm_stub_client(_seed)
    spec = _register_cloud_tool(clean_registry, "status-tool")
    spec.function(_CloudInput(payload="x"), _CloudConfig(device="proto"))

    assert "running" in seen  # the running status was mapped onto the spinner substatus


def test_strip_logger_prefix():
    """The worker logger-name prefix is removed from phase markers; ordinary 'word:' messages are kept."""
    from proto_tools.proto import _strip_logger_prefix

    assert (
        _strip_logger_prefix("proto_tools.worker.esmfold.esmfold-prediction: Loading ESMFold model: v1 on cuda")
        == "Loading ESMFold model: v1 on cuda"
    )
    assert _strip_logger_prefix("proto_tools.utils.progress: Running esmfold") == "Running esmfold"
    # No dotted logger head -> leave untouched (message may legitimately start with 'word: ').
    assert _strip_logger_prefix("Done: 5 sequences") == "Done: 5 sequences"
    assert _strip_logger_prefix("Folding 1 complex(es), num_recycles=3") == "Folding 1 complex(es), num_recycles=3"


def test_proto_replay_strips_logger_prefix_on_plain_lines(
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
    caplog.set_level(logging.DEBUG, logger="proto_tools.proto.remote")
    spec.function(_CloudInput(payload="x"), _CloudConfig(device="proto", verbose=1))

    msgs = [r.message for r in caplog.records if r.name == "proto_tools.proto.remote"]
    assert "Starting worker" in msgs
    assert "proto_tools.utils.progress: Starting worker" not in msgs


def test_proto_update_status_records_drive_spinner_substatus(
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
    caplog.set_level(logging.DEBUG, logger="proto_tools.proto.remote")
    spec.function(_CloudInput(payload="x"), _CloudConfig(device="proto", verbose=1))

    # The flag rides through onto the replayed record; SpinnerFromLogsHandler (tested separately) routes it to the bar.
    by_msg = {r.message: r for r in caplog.records if r.name == "proto_tools.proto.remote"}
    assert getattr(by_msg["Loading checkpoint"], "update_status", False) is True
    assert getattr(by_msg["Folded 1/3"], "update_status", False) is False


def test_proto_log_stream_failure_does_not_break_run(fake_proto_client, arm_stub_client, caplog, clean_registry):
    """A broken log stream must not fail the run — the result is what matters."""

    def _seed(c):
        c.tools.output_to_return = {"result": "ok"}

        def _broken_logs(*args, **kwargs):
            raise ConnectionError("log stream down")
            yield  # unreachable; keep this a generator function

        c.tools.iter_job_logs = _broken_logs

    arm_stub_client(_seed)
    spec = _register_cloud_tool(clean_registry, "log-fail-tool")

    caplog.set_level(logging.DEBUG, logger="proto_tools.proto")
    result = spec.function(_CloudInput(payload="x"), _CloudConfig(device="proto"))
    assert result.result == "ok"


# ─ api_key override ──────────────────────────────────────────────────────────


def test_explicit_api_key_overrides_env(fake_proto_client, arm_stub_client, monkeypatch, clean_registry):
    """dispatch_to_proto's api_key kwarg takes precedence over PROTO_API_KEY."""
    monkeypatch.setenv("PROTO_API_KEY", "env-key")
    arm_stub_client(lambda c: setattr(c.tools, "output_to_return", {"result": "ok"}))
    _register_cloud_tool(clean_registry, "explicit-key-tool")

    from proto_tools.proto import dispatch_to_proto

    dispatch_to_proto(
        "explicit-key-tool",
        _CloudInput(payload="hi"),
        _CloudConfig(device="proto"),
        api_key="explicit-key",
    )

    client = fake_proto_client.last_instance
    assert client is not None
    assert client.kwargs == {"api_key": "explicit-key"}


# ---------------------------------------------------------------------------
# Device validation and per-device remote restrictions
# ---------------------------------------------------------------------------

VALID_DEVICE_STRINGS = [
    "cpu",
    "cuda",
    "cuda:0",
    "cuda:7",
    "cudax2",
    "cudax64",
    "cuda:0,1",
    "cuda:0,1,2",
    "cuda:1,cuda:2",
    "cuda:0,cuda:1,cuda:2",
    " cpu ",
    "cuda:0, 1",
    "proto",
    "modal",
]

INVALID_DEVICE_STRINGS = ["banana", "gpu", "cuda:-1", "0,1", "cuda:", "cudax0", "CUDA:0", "Cuda"]


@pytest.mark.parametrize("device", VALID_DEVICE_STRINGS)
def test_base_config_accepts_every_supported_device_form(device):
    """Validation is syntactic, so every documented form works without that hardware present."""
    from proto_tools.utils.base_config import BaseConfig

    assert BaseConfig(device=device).device == device


@pytest.mark.parametrize("device", INVALID_DEVICE_STRINGS)
def test_base_config_rejects_malformed_devices(device):
    """A typo used to survive construction and fail much later, at dispatch."""
    from pydantic import ValidationError

    from proto_tools.utils.base_config import BaseConfig

    with pytest.raises(ValidationError):
        BaseConfig(device=device)


# ─ device='modal' routing (delegates to proto_tools.modal) ───────────────────


def _patch_modal_dispatch(monkeypatch, dispatch):
    """Replace both Modal dispatch entry points so routing tests reach no real workspace."""
    monkeypatch.setattr("proto_tools.modal.dispatch_to_modal", dispatch)
    # The router imports both forms; the batch one carries chunks, one config each, when a tool
    # sets max_chunk_size.
    monkeypatch.setattr(
        "proto_tools.modal.dispatch_batch_to_modal",
        lambda key, inputs_list, configs: [
            dispatch(key, chunk, one) for chunk, one in zip(inputs_list, configs, strict=True)
        ],
    )


def test_device_modal_delegates_and_returns_validated_output(clean_registry, monkeypatch):
    """device='modal' routes to the Modal dispatcher and validates its result."""
    captured = {}

    def _dispatch(key, inputs, config):
        captured["key"] = key
        captured["payload"] = inputs.payload
        captured["device"] = config.device
        return _CloudOutput(result="from-modal")

    _patch_modal_dispatch(monkeypatch, _dispatch)
    spec = _register_cloud_tool(clean_registry, "modal-tool")  # uses_gpu; local impl asserts if reached

    result = spec.function(_CloudInput(payload="hi"), _CloudConfig(device="modal"))

    assert isinstance(result, _CloudOutput)
    assert result.result == "from-modal"
    assert result.success is True
    assert result.tool_id == "modal-tool"
    assert captured == {"key": "modal-tool", "payload": "hi", "device": "modal"}


def test_device_modal_no_ops_trivially_local_tools(clean_registry, monkeypatch):
    """A tool with no GPU and no standalone environment runs in-process rather than on Modal.

    ``local_cpu`` means there is nothing to accelerate and nothing to build, so a worker can
    offer the tool nothing and the round trip is pure cost. This does not reach the CPU tools
    that are actually deployed: those carry a standalone environment, which is precisely what
    makes running them remotely worthwhile.
    """

    def _dispatch_should_not_run(key, inputs, config):
        raise AssertionError("a local_cpu tool has nothing to gain from Modal and must stay local")

    _patch_modal_dispatch(monkeypatch, _dispatch_should_not_run)

    def _local_impl(inputs, config, instance=None):
        del instance
        return _CloudOutput(result=f"local:{inputs.payload}:{config.device}")

    clean_registry.register(
        key="modal-local-cpu",
        label="modal-local-cpu",
        category="test",
        input_class=_CloudInput,
        config_class=_CloudConfig,
        output_class=_CloudOutput,
        description="pure-python tool — no GPU, no standalone env",
    )(_local_impl)
    spec = clean_registry.get("modal-local-cpu")
    assert spec.local_cpu is True

    result = spec.function(_CloudInput(payload="hi"), _CloudConfig(device="modal"))

    # The device is rewritten to cpu so the tool sees where it actually ran.
    assert result.result == "local:hi:cpu"


def test_device_modal_blocks_unsupported_config(clean_registry, monkeypatch):
    """A config needing a local resource fails fast before any dispatch is attempted."""

    class _ModalUnsupportedConfig(_CloudConfig):
        def remote_unsupported_reason(self, device):
            return "needs a local database file" if device == "modal" else None

    def _dispatch_should_not_run(key, inputs, config):
        raise AssertionError("must not dispatch a config that declares itself unsupported")

    _patch_modal_dispatch(monkeypatch, _dispatch_should_not_run)
    spec = _register_cloud_tool(clean_registry, "modal-tool-3", config_class=_ModalUnsupportedConfig)

    with pytest.raises(ValueError, match="needs a local database file"):
        spec.function(_CloudInput(payload="hi"), _ModalUnsupportedConfig(device="modal"))


def test_batch_dispatch_submits_every_job_before_waiting(fake_proto_client, arm_stub_client, clean_registry):
    """All jobs are submitted first, so Proto runs them concurrently rather than one at a time.

    Waiting on each job before submitting the next would serialize the batch on the client, which
    is the whole cost fan-out exists to remove. The stub records the order of submit and get calls,
    so a regression to submit-wait-submit is visible rather than merely slower.
    """
    from proto_tools.proto import dispatch_batch_to_proto

    order: list[str] = []

    def _seed(client):
        client.tools.output_to_return = {"result": "ok"}
        original_submit, original_get = client.tools.submit, client.tools.get

        def submit(*args, **kwargs):
            order.append("submit")
            return original_submit(*args, **kwargs)

        def get(*args, **kwargs):
            order.append("get")
            return original_get(*args, **kwargs)

        client.tools.submit, client.tools.get = submit, get

    arm_stub_client(_seed)
    spec = _register_cloud_tool(clean_registry, "batch-tool")

    results = dispatch_batch_to_proto("batch-tool", [_CloudInput(payload=p) for p in ("a", "b", "c")])

    assert [r.result for r in results] == ["ok", "ok", "ok"]
    assert order[:3] == ["submit", "submit", "submit"], f"submits must all precede any wait, got {order}"
    assert "get" in order[3:]
    assert spec is not None


def test_batch_dispatch_of_nothing_contacts_no_server(fake_proto_client, arm_stub_client, clean_registry):
    """An empty batch returns immediately rather than opening a job."""
    from proto_tools.proto import dispatch_batch_to_proto

    arm_stub_client(lambda c: setattr(c.tools, "output_to_return", {"result": "unreachable"}))
    _register_cloud_tool(clean_registry, "batch-tool-empty")

    assert dispatch_batch_to_proto("batch-tool-empty", []) == []
    assert fake_proto_client.last_instance is None


# --- transport retry -------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for a requests.Response in retry tests."""

    def __init__(self, status_code: int, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.ok = status_code < 400

    def close(self) -> None:
        """No-op; the retry path closes a response before sleeping."""


@pytest.fixture
def no_sleep(monkeypatch):
    """Collapse backoff so retry tests do not actually wait."""
    from proto_tools.proto import _client

    monkeypatch.setattr(_client.time, "sleep", lambda _s: None)


def test_transient_status_is_retried_then_succeeds(no_sleep):
    """A 503 mid-poll must be re-sent, not surfaced — the wrapper's own retry would resubmit the job."""
    from proto_tools.proto._client import _request_with_retry

    attempts = {"n": 0}

    def _flaky(method, url, **kwargs):
        attempts["n"] += 1
        return _FakeResponse(503) if attempts["n"] == 1 else _FakeResponse(200)

    response = _request_with_retry(types.SimpleNamespace(request=_flaky), "GET", "http://x")
    assert attempts["n"] == 2
    assert response.status_code == 200


def test_client_error_is_not_retried(no_sleep):
    """A 422 describes the request itself, so repeating it only wastes round trips."""
    from proto_tools.proto._client import _request_with_retry

    attempts = {"n": 0}

    def _reject(method, url, **kwargs):
        attempts["n"] += 1
        return _FakeResponse(422)

    _request_with_retry(types.SimpleNamespace(request=_reject), "GET", "http://x")
    assert attempts["n"] == 1


def test_connection_error_retries_then_reraises(no_sleep):
    """Transport faults get a bounded number of attempts, then propagate unchanged."""
    import requests

    from proto_tools.proto._client import _MAX_RETRIES, _request_with_retry

    attempts = {"n": 0}

    def _down(method, url, **kwargs):
        attempts["n"] += 1
        raise requests.ConnectionError("down")

    with pytest.raises(requests.ConnectionError):
        _request_with_retry(types.SimpleNamespace(request=_down), "GET", "http://x")
    assert attempts["n"] == _MAX_RETRIES + 1


def test_retry_after_is_honored_and_capped():
    """A rate-limited caller waits as long as the server asked, but not unboundedly."""
    from proto_tools.proto._client import _RETRY_AFTER_MAX, _backoff_delay

    assert _backoff_delay(0, _FakeResponse(429, {"Retry-After": "7"})) == 7.0
    assert _backoff_delay(0, _FakeResponse(429, {"Retry-After": "99999"})) == _RETRY_AFTER_MAX
    # An HTTP-date Retry-After is not parsed; the caller falls back to its own backoff.
    assert _backoff_delay(0, _FakeResponse(429, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})) > 0


def test_submit_sends_an_idempotency_key():
    """Retrying a submit must return the job already created, which needs a key on every send."""
    from proto_tools.proto._client import _ToolsNamespace

    sent: dict[str, object] = {}

    class _SubmitResponse(_FakeResponse):
        def json(self) -> dict[str, str]:
            return {"job_id": "fc-1"}

    class _Session:
        def request(self, method: str, url: str, **kwargs: object) -> _SubmitResponse:
            sent.update(kwargs)
            return _SubmitResponse(200)

    job_id = _ToolsNamespace(_Session(), "https://example.test").submit("some-tool", inputs={}, config={})

    assert job_id == "fc-1"
    assert "Idempotency-Key" in sent["headers"]
