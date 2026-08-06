"""tests/tool_infra_tests/test_remote_device_pipeline.py.

Tests that a remote device runs the same prepared work a local one does.

``device="proto"`` and ``device="modal"`` once dispatched before dedup, the per-item cache,
and ``preprocess``, so the paid device repeated work the caller had already done and never
reused a cached item. Both now fall through the shared pipeline and dispatch at the point a
local device would execute, and the config records that preprocess ran so the worker on the
far side does not repeat it.
"""

from __future__ import annotations

import contextlib
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, Field

from proto_tools.modal import ModalDispatchError
from proto_tools.tools.tool_registry import ToolRegistry, tool
from proto_tools.utils import BaseConfig, ConfigField
from proto_tools.utils.base_config import INTERNAL_STATE_KEY
from proto_tools.utils.fanout import FailedItem, chunk_indices
from proto_tools.utils.tool_cache import ToolCache, _program_tool_cache
from proto_tools.utils.tool_io import BaseToolInput
from proto_tools.utils.tool_pool import PartialFailureError
from tests.tool_infra_tests.test_export_functionality import MockToolOutputBase


class _RemoteInput(BaseToolInput):
    """Inputs for the remote-pipeline stand-in tool."""

    items: list[str] = Field(default_factory=list, description="Items to process")


class _RemoteItem(MockToolOutputBase):
    """One processed item."""

    value: str = Field(default="", description="Processed value")


class _RemoteOutput(MockToolOutputBase):
    """Outputs for the remote-pipeline stand-in tool."""

    results: list[_RemoteItem] = Field(default_factory=list, description="One result per item")


class _RemoteConfig(BaseConfig):
    """Config whose preprocess counts its invocations so a double run is visible."""

    device: str = ConfigField(default="cpu", title="Device", description="Device to run on")

    # A ClassVar, not a field: the count is test bookkeeping and must stay out of the schema.
    preprocess_calls: ClassVar[int] = 0

    def preprocess(self, inputs: _RemoteInput) -> _RemoteInput:
        """Count invocations on the class, since each call gets its own config copy."""
        type(self).preprocess_calls += 1
        return inputs


def _register_probe(
    monkeypatch, key: str, *, uses_gpu: bool, max_chunk_size: int | None = None, stochastic: bool = False
):
    """Register a cacheable iterable tool and capture what each remote dispatch receives.

    ``uses_gpu`` decides ``local_cpu``: a test-registered tool has no ``standalone/`` directory,
    so declaring no GPU is what makes it trivially local.
    """
    seen: dict[str, Any] = {"dispatches": [], "preprocess_completed": [], "envelope": [], "seeds": []}
    _RemoteConfig.preprocess_calls = 0

    @tool(
        key=key,
        label="Remote Pipeline Probe",
        category="testing",
        input_class=_RemoteInput,
        config_class=_RemoteConfig,
        output_class=_RemoteOutput,
        description="Stand-in tool for remote-device pipeline tests",
        uses_gpu=uses_gpu,
        cacheable=True,
        stochastic=stochastic,
        iterable_input_fields=["items"],
        iterable_output_field="results",
        max_chunk_size=max_chunk_size,
    )
    def _run(inputs: _RemoteInput, config: _RemoteConfig, instance: Any = None) -> _RemoteOutput:
        return _RemoteOutput(results=[_RemoteItem(value=item) for item in inputs.items])

    def fake_dispatch(tool_key: str, inputs: _RemoteInput, config: _RemoteConfig) -> _RemoteOutput:
        seen["dispatches"].append(list(inputs.items))
        seen["seeds"].append(config.seed)
        seen["preprocess_completed"].append(config._preprocess_completed)
        seen["envelope"].append(INTERNAL_STATE_KEY in config.to_transport_dict())
        return _RemoteOutput(
            tool_id=tool_key,
            execution_time=0.0,
            success=True,
            results=[_RemoteItem(value=item) for item in inputs.items],
        )

    import proto_tools.proto as proto_module

    monkeypatch.setattr(proto_module, "dispatch_to_proto", fake_dispatch, raising=False)

    def fake_batch_dispatch(tool_key: str, inputs_list, configs):
        """Chunked fan-out path; one config per chunk, recorded as the single-call form does."""
        return [fake_dispatch(tool_key, chunk, one) for chunk, one in zip(inputs_list, configs, strict=True)]

    # Replace both dispatch entry points so no call reaches a real Modal workspace.
    monkeypatch.setattr("proto_tools.modal.dispatch_to_modal", fake_dispatch)
    monkeypatch.setattr("proto_tools.modal.dispatch_batch_to_modal", fake_batch_dispatch)

    return ToolRegistry.get(key), seen


@pytest.fixture
def remote_tool(monkeypatch):
    """A tool a remote worker can genuinely help with, so remote dispatch is not declined."""
    key = "remote-pipeline-probe"
    yield _register_probe(monkeypatch, key, uses_gpu=True)
    ToolRegistry._registry.pop(key, None)


@pytest.fixture
def trivially_local_tool(monkeypatch):
    """A local_cpu tool: no GPU and no standalone environment, so a worker offers it nothing."""
    key = "remote-pipeline-probe-local"
    spec, seen = _register_probe(monkeypatch, key, uses_gpu=False)
    assert spec.local_cpu, "fixture must produce a local_cpu tool for this to be meaningful"
    yield spec, seen
    ToolRegistry._registry.pop(key, None)


@pytest.fixture
def program_cache():
    """Give the tool an active program cache for the duration of a test."""
    token = _program_tool_cache.set(ToolCache())
    yield
    _program_tool_cache.reset(token)


@pytest.mark.parametrize("device", ["proto", "modal"])
def test_remote_dispatch_receives_deduplicated_items(remote_tool, device):
    """Duplicates collapse before dispatch, and the caller still gets one result per input."""
    spec, seen = remote_tool

    result = spec.function(_RemoteInput(items=["a", "b", "a", "b"]), _RemoteConfig(device=device))

    assert seen["dispatches"] == [["a", "b"]], "the remote should only be paid for unique items"
    assert [r.value for r in result.results] == ["a", "b", "a", "b"]


@pytest.mark.parametrize("device", ["proto", "modal"])
def test_remote_dispatch_reuses_cached_items(remote_tool, program_cache, device):
    """A repeat call skips dispatch, and a partial repeat dispatches only the new item."""
    spec, seen = remote_tool
    config = lambda: _RemoteConfig(device=device)  # noqa: E731 — a fresh config per call

    first = spec.function(_RemoteInput(items=["a", "b"]), config())
    spec.function(_RemoteInput(items=["a", "b"]), config())
    third = spec.function(_RemoteInput(items=["a", "b", "c"]), config())

    assert seen["dispatches"] == [["a", "b"], ["c"]], "cached items must not be dispatched again"
    assert [r.value for r in first.results] == ["a", "b"]
    assert [r.value for r in third.results] == ["a", "b", "c"]


@pytest.mark.parametrize("device", ["proto", "modal"])
def test_preprocess_runs_once_on_the_caller(remote_tool, device):
    """The caller preprocesses, and says so, so the worker on the far side does not repeat it."""
    spec, seen = remote_tool

    spec.function(_RemoteInput(items=["a"]), _RemoteConfig(device=device))

    assert _RemoteConfig.preprocess_calls == 1
    assert seen["preprocess_completed"] == [True]
    assert seen["envelope"] == [True], "the worker must be told preprocess already ran"


def test_worker_side_config_skips_preprocess(remote_tool):
    """A config arriving with the state set does not preprocess again, which is the worker's case."""
    spec, _ = remote_tool
    config = _RemoteConfig(device="cpu")
    config._preprocess_completed = True

    spec.function(_RemoteInput(items=["a"]), config)

    assert _RemoteConfig.preprocess_calls == 0


def test_a_plain_payload_still_preprocesses(remote_tool):
    """A caller that never ran preprocess leaves the state unset, so the worker runs it."""
    spec, _ = remote_tool
    # A config rebuilt from an ordinary dump, as an MCP-style caller would send.
    config = _RemoteConfig(**_RemoteConfig(device="cpu").model_dump(mode="json"))

    spec.function(_RemoteInput(items=["a"]), config)

    assert _RemoteConfig.preprocess_calls == 1, "skipping must be opt-in, never the default"


def test_a_short_remote_result_is_reported_clearly(remote_tool, program_cache, monkeypatch):
    """A worker returning fewer items than were sent names the cause instead of failing on a zip."""
    spec, _ = remote_tool

    def short_dispatch(tool_key, inputs, config):
        # What a worker built from a different proto-tools does: its output field is not the one
        # this version reads, so validation drops it and the list arrives empty.
        return _RemoteOutput(tool_id=tool_key, execution_time=0.0, success=True, results=[])

    monkeypatch.setattr("proto_tools.modal.dispatch_to_modal", short_dispatch)

    with pytest.raises(ValueError, match=r"returned 0 item\(s\) in 'results' for 2 sent"):
        spec.function(_RemoteInput(items=["a", "b"]), _RemoteConfig(device="modal"))


@pytest.mark.parametrize("device", ["proto", "modal"])
def test_a_trivially_local_tool_is_never_dispatched(trivially_local_tool, device):
    """No GPU and no standalone environment means a worker offers nothing, on either device."""
    spec, seen = trivially_local_tool

    result = spec.function(_RemoteInput(items=["a", "b"]), _RemoteConfig(device=device))

    assert seen["dispatches"] == [], "a remote worker has nothing to offer this tool"
    assert [r.value for r in result.results] == ["a", "b"]


def test_reusing_one_config_preprocesses_every_call(remote_tool):
    """Completion is recorded on a copy, so a config reused across calls is not marked spent.

    Recording it on the caller's own object would make the second call believe the work was
    already done and dispatch raw inputs, skipping the MSA search or selection preprocess
    performs. Nothing about that is remote, so it would silently affect local calls too.
    """
    spec, _ = remote_tool
    config = _RemoteConfig(device="cpu")

    for _ in range(3):
        spec.function(_RemoteInput(items=["a"]), config)

    assert _RemoteConfig.preprocess_calls == 3, "each call must prepare its own inputs"
    assert config._preprocess_completed is False, "the caller's config must not be marked spent"


def test_local_dispatch_is_unaffected(remote_tool):
    """A local device still executes the tool rather than taking a remote path."""
    spec, seen = remote_tool

    result = spec.function(_RemoteInput(items=["a", "b"]), _RemoteConfig(device="cpu"))

    assert seen["dispatches"] == []
    assert [r.value for r in result.results] == ["a", "b"]


# ── Fan-out across containers ───────────────────────────────────────────────


@pytest.fixture
def chunked_tool(monkeypatch):
    """A remote tool that takes at most three items per execution."""
    key = "remote-pipeline-probe-chunked"
    spec, seen = _register_probe(monkeypatch, key, uses_gpu=True, max_chunk_size=3)
    yield spec, seen
    ToolRegistry._registry.pop(key, None)


def test_a_batch_is_split_by_max_chunk_size(chunked_tool):
    """Items beyond one chunk go out as several executions, in original order."""
    spec, seen = chunked_tool

    result = spec.function(_RemoteInput(items=[f"s{n}" for n in range(8)]), _RemoteConfig(device="modal"))

    assert seen["dispatches"] == [["s0", "s1", "s2"], ["s3", "s4", "s5"], ["s6", "s7"]]
    assert [r.value for r in result.results] == [f"s{n}" for n in range(8)]


def test_a_batch_within_one_chunk_is_sent_whole(chunked_tool):
    """No splitting when the batch already fits, so small calls are unchanged."""
    spec, seen = chunked_tool

    spec.function(_RemoteInput(items=["a", "b"]), _RemoteConfig(device="modal"))

    assert seen["dispatches"] == [["a", "b"]]


def test_a_tool_without_a_chunk_size_is_never_split(remote_tool):
    """Fan-out is opt-in: an undeclared tool keeps sending the whole batch to one execution.

    The declaration is a claim about a tool's economics. One execution per item suits a fold
    taking minutes and is ruinous for work measured in milliseconds, so it is not defaulted.
    """
    spec, seen = remote_tool
    assert spec.max_chunk_size is None

    spec.function(_RemoteInput(items=[f"s{n}" for n in range(8)]), _RemoteConfig(device="modal"))

    assert seen["dispatches"] == [[f"s{n}" for n in range(8)]]


def test_chunks_that_return_the_wrong_count_are_reported(chunked_tool, monkeypatch):
    """A chunk returning fewer items than it was sent names the mismatch rather than misaligning."""
    spec, _ = chunked_tool

    def short_batch(tool_key, inputs_list, configs):
        assert len(configs) == len(inputs_list)
        return [_RemoteOutput(tool_id=tool_key, execution_time=0.0, success=True, results=[]) for _ in inputs_list]

    monkeypatch.setattr("proto_tools.modal.dispatch_batch_to_modal", short_batch)

    # Every chunk is short, so every chunk fails and nothing survives to return.
    with pytest.raises(PartialFailureError, match=r"chunk of 3 item\(s\) starting at 0 returned 0 item"):
        spec.function(_RemoteInput(items=[f"s{n}" for n in range(8)]), _RemoteConfig(device="modal"))


@pytest.fixture
def stochastic_chunked_tool(monkeypatch):
    """A stochastic remote tool that takes at most three items per execution."""
    key = "remote-pipeline-probe-stochastic"
    spec, seen = _register_probe(monkeypatch, key, uses_gpu=True, max_chunk_size=3, stochastic=True)
    yield spec, seen
    ToolRegistry._registry.pop(key, None)


def test_each_chunk_of_a_seeded_batch_gets_its_own_seed(stochastic_chunked_tool):
    """A stochastic tool seeds per execution, so chunks sharing one seed would draw identically."""
    spec, seen = stochastic_chunked_tool

    spec.function(_RemoteInput(items=[f"s{n}" for n in range(8)]), _RemoteConfig(device="modal", seed=7))

    assert len(seen["seeds"]) == 3
    assert seen["seeds"][0] == 7, "the first chunk keeps the caller's seed"
    assert len(set(seen["seeds"])) == 3, f"chunks must not share a seed, got {seen['seeds']}"


def test_an_unsplit_seeded_batch_keeps_the_callers_seed(stochastic_chunked_tool):
    """Nothing is derived when the batch fits one execution, so small calls are unchanged."""
    spec, seen = stochastic_chunked_tool

    spec.function(_RemoteInput(items=["a", "b"]), _RemoteConfig(device="modal", seed=7))

    assert seen["seeds"] == [7]


def test_an_unseeded_batch_stays_unseeded_in_every_chunk(stochastic_chunked_tool):
    """Without a seed there is nothing to derive from, and the tool must stay free-running."""
    spec, seen = stochastic_chunked_tool

    spec.function(_RemoteInput(items=[f"s{n}" for n in range(8)]), _RemoteConfig(device="modal"))

    assert seen["seeds"] == [None, None, None]


def test_a_deterministic_tool_keeps_one_seed_across_chunks(chunked_tool):
    """Re-seeding a deterministic tool would change nothing and only obscure the config."""
    spec, seen = chunked_tool

    spec.function(_RemoteInput(items=[f"s{n}" for n in range(8)]), _RemoteConfig(device="modal", seed=7))

    assert seen["seeds"] == [7, 7, 7]


def test_the_derived_seeds_are_reproducible(stochastic_chunked_tool):
    """Two identical calls must derive the same per-chunk seeds, or nothing is reproducible."""
    spec, seen = stochastic_chunked_tool
    items = _RemoteInput(items=[f"s{n}" for n in range(8)])

    spec.function(items, _RemoteConfig(device="modal", seed=7))
    first = list(seen["seeds"])
    seen["seeds"].clear()
    spec.function(items, _RemoteConfig(device="modal", seed=7))

    assert seen["seeds"] == first


def test_a_split_batch_reports_how_it_was_split(chunked_tool):
    """The caller made one call and cannot otherwise tell it became several executions."""
    spec, _ = chunked_tool

    result = spec.function(_RemoteInput(items=[f"s{n}" for n in range(8)]), _RemoteConfig(device="modal"))

    assert result.metadata["dispatch_stats"] == {
        "total_items": 8,
        "chunks": 3,
        "chunk_sizes": [3, 3, 2],
        "device": "modal",
    }


def test_an_unsplit_batch_reports_no_dispatch_stats(chunked_tool):
    """Nothing was fanned out, so there is no split to describe."""
    spec, _ = chunked_tool

    result = spec.function(_RemoteInput(items=["a", "b"]), _RemoteConfig(device="modal"))

    assert "dispatch_stats" not in (result.metadata or {})


# ── partial failure across a split batch ────────────────────────────────────


def _fail_chunks(monkeypatch, *, failing: set[int], exc: Exception | None = None, fail_times: int | None = None):
    """Make the listed chunk positions fail, optionally only for the first ``fail_times`` attempts.

    Returns a dict recording how many chunks each dispatch round was asked for, so a test can tell
    a retry of the failures apart from a retry of the whole batch.
    """
    seen: dict[str, list[int]] = {"rounds": []}
    attempts: dict[int, int] = {}
    real_error = exc if exc is not None else RuntimeError("chunk exploded")

    def batch(tool_key, inputs_list, configs):
        seen["rounds"].append(len(inputs_list))
        out: list[Any] = []
        for chunk, one in zip(inputs_list, configs, strict=True):
            position = chunk.items[0]
            index = int(position.removeprefix("s")) // 3
            attempts[index] = attempts.get(index, 0) + 1
            spent = fail_times is not None and attempts[index] > fail_times
            if index in failing and not spent:
                out.append(real_error)
                continue
            out.append(
                _RemoteOutput(
                    tool_id=tool_key,
                    execution_time=0.0,
                    success=True,
                    results=[_RemoteItem(value=f"{item}:{one.seed}") for item in chunk.items],
                )
            )
        return out

    monkeypatch.setattr("proto_tools.modal.dispatch_batch_to_modal", batch)
    return seen


def test_a_failed_chunk_no_longer_discards_the_others(chunked_tool, monkeypatch):
    """The chunks that succeeded were already billed, so losing them to a sibling is pure waste."""
    spec, _ = chunked_tool
    _fail_chunks(monkeypatch, failing={1})

    with pytest.raises(PartialFailureError) as caught:
        spec.function(_RemoteInput(items=[f"s{n}" for n in range(8)]), _RemoteConfig(device="modal"))

    error = caught.value
    assert [index for index, _ in error.succeeded] == [0, 1, 2, 6, 7], "chunks 0 and 2 must survive"
    assert [f["indices"] for f in error.failed] == [[3, 4, 5]]
    assert "1/3 chunk(s) failed" in str(error)


def test_return_partial_keeps_the_output_aligned_with_the_input(chunked_tool, monkeypatch):
    """Opting in fills each failed position with a FailedItem, so zip against the input still holds."""
    spec, _ = chunked_tool
    _fail_chunks(monkeypatch, failing={1})

    monkeypatch.setenv("PROTO_ON_PARTIAL_FAILURE", "return_partial")

    result = spec.function(_RemoteInput(items=[f"s{n}" for n in range(8)]), _RemoteConfig(device="modal"))

    values = [item.value.split(":")[0] if isinstance(item, _RemoteItem) else item.error for item in result.results]
    assert values[:3] == ["s0", "s1", "s2"]
    assert values[6:] == ["s6", "s7"]
    assert all("chunk exploded" in v for v in values[3:6]), "each failed position names the cause"
    assert any("chunk(s) failed" in e for e in result.errors)
    assert result.metadata["dispatch_stats"]["failed_items"] == 3


def test_a_transient_chunk_failure_is_retried_and_the_call_succeeds(chunked_tool, monkeypatch):
    """A dropped connection on one chunk should not cost the batch."""
    spec, _ = chunked_tool
    seen = _fail_chunks(monkeypatch, failing={1}, exc=ConnectionError("reset"), fail_times=1)

    result = spec.function(_RemoteInput(items=[f"s{n}" for n in range(8)]), _RemoteConfig(device="modal"))

    assert [item.value.split(":")[0] for item in result.results] == [f"s{n}" for n in range(8)]
    assert seen["rounds"] == [3, 1], "the retry must resend only the failed chunk, not the batch"
    assert result.metadata["dispatch_stats"]["retry_rounds"] == 1


def test_a_permanent_chunk_failure_is_not_retried(chunked_tool, monkeypatch):
    """A malformed result fails identically on a second attempt, so retrying only burns money."""
    spec, _ = chunked_tool
    seen = _fail_chunks(monkeypatch, failing={1}, exc=TypeError("does not conform"))

    with pytest.raises(PartialFailureError):
        spec.function(_RemoteInput(items=[f"s{n}" for n in range(8)]), _RemoteConfig(device="modal"))

    assert seen["rounds"] == [3], "a validation failure must not be resent"


def test_a_retried_chunk_keeps_its_derived_seed(stochastic_chunked_tool, monkeypatch):
    """Re-seeding on retry would make the same call return different results run to run."""
    spec, _ = stochastic_chunked_tool
    _fail_chunks(monkeypatch, failing={1}, exc=ConnectionError("reset"), fail_times=1)

    result = spec.function(_RemoteInput(items=[f"s{n}" for n in range(8)]), _RemoteConfig(device="modal", seed=7))

    seeds = [item.value.split(":")[1] for item in result.results]
    assert seeds[3] == seeds[4] == seeds[5], "the retried chunk keeps one seed"
    assert seeds[0] != seeds[3], "and it is still distinct from the first chunk's"


def test_the_surviving_chunks_are_cached_so_a_retry_only_repeats_the_failure(chunked_tool, program_cache, monkeypatch):
    """The whole point: work already paid for must not be paid for twice."""
    spec, seen = chunked_tool
    _fail_chunks(monkeypatch, failing={1})

    with pytest.raises(PartialFailureError):
        spec.function(_RemoteInput(items=[f"s{n}" for n in range(8)]), _RemoteConfig(device="modal"))

    # Same batch again. The five items that came back are cached, so only the failed three go out.
    seen["dispatches"].clear()
    monkeypatch.setattr("proto_tools.modal.dispatch_batch_to_modal", _unused_batch)
    result = spec.function(_RemoteInput(items=[f"s{n}" for n in range(8)]), _RemoteConfig(device="modal"))

    assert seen["dispatches"] == [["s3", "s4", "s5"]], "cached survivors must not be recomputed"
    assert len(result.results) == 8


def _unused_batch(tool_key, inputs_list, configs):
    """Stand-in that fails loudly: the retry above should fit one chunk and never batch."""
    raise AssertionError(f"batch dispatch not expected, got {len(inputs_list)} chunk(s)")


def test_reported_indices_are_the_callers_not_post_strip_positions(chunked_tool, program_cache, monkeypatch):
    """Dedup and the cache shorten the batch, so a raw dispatch index names the wrong input."""
    spec, _ = chunked_tool

    # Warm one item so cache-strip shifts every later position by one.
    monkeypatch.setattr("proto_tools.modal.dispatch_batch_to_modal", _unused_batch)
    spec.function(_RemoteInput(items=["s0"]), _RemoteConfig(device="modal"))

    _fail_chunks(monkeypatch, failing={1})
    with pytest.raises(PartialFailureError) as caught:
        spec.function(_RemoteInput(items=[f"s{n}" for n in range(8)]), _RemoteConfig(device="modal"))

    # 's0' is cached, so the dispatched batch is s1..s7 and its second chunk is s4,s5,s6.
    assert [f["indices"] for f in caught.value.failed] == [[4, 5, 6]]
    assert [index for index, _ in caught.value.succeeded] == [1, 2, 3, 7]


def test_duplicate_inputs_report_every_position_they_came_from(chunked_tool, monkeypatch):
    """One dispatched item can stand for several inputs, and all of them failed or none did."""
    spec, _ = chunked_tool
    _fail_chunks(monkeypatch, failing={1})

    # Twelve inputs, eight unique: dedup collapses them before the batch is split.
    items = [f"s{n}" for n in range(8)] + ["s0", "s3", "s4", "s7"]
    with pytest.raises(PartialFailureError) as caught:
        spec.function(_RemoteInput(items=items), _RemoteConfig(device="modal"))

    # s3 and s4 are in the failed chunk and each appears twice in the caller's list.
    assert [f["indices"] for f in caught.value.failed] == [[3, 4, 5, 9, 10]]


def test_a_stochastic_tool_preserves_successes_without_caching_them(
    stochastic_chunked_tool, program_cache, monkeypatch
):
    """Stochastic tools use the whole-call cache, so there is no correct per-item entry to write."""
    spec, _ = stochastic_chunked_tool
    rounds = _fail_chunks(monkeypatch, failing={1})

    with pytest.raises(PartialFailureError) as caught:
        spec.function(_RemoteInput(items=[f"s{n}" for n in range(8)]), _RemoteConfig(device="modal", seed=7))

    assert [index for index, _ in caught.value.succeeded] == [0, 1, 2, 6, 7], "successes still reported"

    with pytest.raises(PartialFailureError):
        spec.function(_RemoteInput(items=[f"s{n}" for n in range(8)]), _RemoteConfig(device="modal", seed=7))
    assert rounds["rounds"] == [3, 3], "nothing cached, so the second call resends all three chunks"


def test_return_partial_does_not_cache_the_failures(chunked_tool, program_cache, monkeypatch):
    """A failure is not a result. Caching one makes it permanent and never retried."""
    spec, seen = chunked_tool
    monkeypatch.setenv("PROTO_ON_PARTIAL_FAILURE", "return_partial")
    _fail_chunks(monkeypatch, failing={1})

    first = spec.function(_RemoteInput(items=[f"s{n}" for n in range(8)]), _RemoteConfig(device="modal"))
    assert [i for i, item in enumerate(first.results) if isinstance(item, FailedItem)] == [3, 4, 5]

    # Same batch again, nothing failing this time: the three failed items must go back out.
    seen["dispatches"].clear()
    monkeypatch.setattr("proto_tools.modal.dispatch_batch_to_modal", _unused_batch)
    second = spec.function(_RemoteInput(items=[f"s{n}" for n in range(8)]), _RemoteConfig(device="modal"))

    assert seen["dispatches"] == [["s3", "s4", "s5"]], "the failures must be retried, not served from cache"
    assert not any(isinstance(item, FailedItem) for item in second.results), "the retry must produce real results"


def test_return_partial_does_not_cache_a_partial_whole_output(stochastic_chunked_tool, program_cache, monkeypatch):
    """A stochastic tool uses the whole-call cache, where a partial result would stand in for a full one."""
    spec, seen = stochastic_chunked_tool
    monkeypatch.setenv("PROTO_ON_PARTIAL_FAILURE", "return_partial")
    _fail_chunks(monkeypatch, failing={1})

    items = _RemoteInput(items=[f"s{n}" for n in range(8)])
    first = spec.function(items, _RemoteConfig(device="modal", seed=7))
    assert any(isinstance(item, FailedItem) for item in first.results)

    seen["dispatches"].clear()
    rounds = _fail_chunks(monkeypatch, failing=set())
    second = spec.function(items, _RemoteConfig(device="modal", seed=7))

    assert rounds["rounds"], "a partial result must not satisfy the whole-call cache"
    assert not any(isinstance(item, FailedItem) for item in second.results)


def _failed_items_in_cache() -> list[str]:
    """Every FailedItem the active cache holds, however deeply nested.

    Walks values rather than checking a known key, so it covers the per-item entries, the output
    template, and the whole-call entry alike — including a placeholder sitting inside a cached
    output's item list.
    """
    cache = _program_tool_cache.get()
    found: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, FailedItem):
            found.append(value.error)
        elif isinstance(value, BaseModel):
            for name in type(value).model_fields:
                walk(getattr(value, name, None))
        elif isinstance(value, (list, tuple)):
            for entry in value:
                walk(entry)
        elif isinstance(value, dict):
            for entry in value.values():
                walk(entry)

    for entries in getattr(cache, "_cache", {}).values():
        for stored in entries.values():
            walk(stored)
    return found


@pytest.mark.parametrize("policy", ["raise", "return_partial"])
@pytest.mark.parametrize("tool", ["deterministic", "stochastic"])
def test_no_failure_is_ever_written_to_the_cache(
    chunked_tool, stochastic_chunked_tool, program_cache, monkeypatch, policy, tool
):
    """A cached failure is permanent: the next identical call returns it without retrying.

    Asserted over the whole cache rather than one entry, because the same mistake has already
    appeared on two independent paths — per-item entries and the whole-call entry.
    """
    spec, _ = chunked_tool if tool == "deterministic" else stochastic_chunked_tool
    monkeypatch.setenv("PROTO_ON_PARTIAL_FAILURE", policy)
    _fail_chunks(monkeypatch, failing={1})

    inputs = _RemoteInput(items=[f"s{n}" for n in range(8)])
    config = _RemoteConfig(device="modal", seed=7 if tool == "stochastic" else None)
    with contextlib.suppress(PartialFailureError):
        spec.function(inputs, config)

    assert _failed_items_in_cache() == [], "a failure must never be cached, on any path"


def test_return_partial_survives_a_warm_cache(chunked_tool, program_cache, monkeypatch):
    """Placeholders must survive reassembly when some items were already cached.

    Outputs validate on assignment, so putting a FailedItem back into a ``list[Item]`` the wrong
    way raises and takes the whole call with it — losing the survivors this mode exists to return.
    Only reached when the batch is stitched against cached items, which is why it needs a warm
    cache rather than a fresh one.
    """
    spec, _ = chunked_tool
    monkeypatch.setenv("PROTO_ON_PARTIAL_FAILURE", "return_partial")

    # Warm two items so the failing run below has to stitch against cached results.
    monkeypatch.setattr("proto_tools.modal.dispatch_batch_to_modal", _unused_batch)
    spec.function(_RemoteInput(items=["s0", "s1"]), _RemoteConfig(device="modal"))

    _fail_chunks(monkeypatch, failing={1})
    result = spec.function(_RemoteInput(items=[f"s{n}" for n in range(10)]), _RemoteConfig(device="modal"))

    assert len(result.results) == 10, "every position accounted for"
    assert any(isinstance(item, FailedItem) for item in result.results), "the failures are reported"


def test_return_partial_marks_every_position_a_failed_duplicate_came_from(chunked_tool, monkeypatch):
    """One dispatched item can stand for several inputs, and a failure must mark all of them.

    Dedup collapses duplicates before the batch is split, so a failed chunk covers fewer dispatched
    items than the caller sent. Expansion runs after the placeholders are in place, so each one has
    to land at every position its input occupied — not only the first.
    """
    spec, _ = chunked_tool
    monkeypatch.setenv("PROTO_ON_PARTIAL_FAILURE", "return_partial")
    _fail_chunks(monkeypatch, failing={1})

    # Twelve inputs, eight unique. s3/s4 sit in the failed chunk and each appears twice.
    items = [f"s{n}" for n in range(8)] + ["s0", "s3", "s4", "s7"]
    result = spec.function(_RemoteInput(items=items), _RemoteConfig(device="modal"))

    failed_at = [i for i, item in enumerate(result.results) if isinstance(item, FailedItem)]
    assert failed_at == [3, 4, 5, 9, 10], "both copies of s3 and s4 must be marked, not just the first"
    assert len(result.results) == len(items), "expansion keeps every caller position"
    survivors = [item for item in result.results if not isinstance(item, FailedItem)]
    assert [item.value.split(":")[0] for item in survivors] == ["s0", "s1", "s2", "s6", "s7", "s0", "s7"]


def test_a_total_failure_raises_even_under_return_partial(chunked_tool, monkeypatch):
    """There is nothing partial about a batch where nothing worked.

    Also the only safe answer: invariant output fields are carried from a chunk that succeeded, and
    six iterable tools declare required ones, so an all-failed output could only be fabricated
    without them. A caller opting into ``return_partial`` still handles this case.
    """
    spec, _ = chunked_tool
    monkeypatch.setenv("PROTO_ON_PARTIAL_FAILURE", "return_partial")
    _fail_chunks(monkeypatch, failing={0, 1, 2})

    with pytest.raises(PartialFailureError) as caught:
        spec.function(_RemoteInput(items=[f"s{n}" for n in range(8)]), _RemoteConfig(device="modal"))

    assert caught.value.succeeded == [], "nothing survived to hand back"
    assert sum(len(f["indices"]) for f in caught.value.failed) == 8


@pytest.mark.parametrize(
    "error",
    [
        PermissionError("cloud rejected the configured key"),
        NotImplementedError("no API key is configured"),
        ImportError("device='modal' requires the Modal SDK"),
        ModalDispatchError("the app is not deployed in this workspace"),
    ],
    ids=["bad-credentials", "no-api-key", "client-not-installed", "not-deployed"],
)
@pytest.mark.parametrize("policy", ["raise", "return_partial"])
def test_a_setup_failure_surfaces_instead_of_becoming_partial(chunked_tool, monkeypatch, error, policy):
    """Nothing about the items is wrong, so every chunk would fail identically.

    Reporting one placeholder per item would repeat the same setup problem N times and bury the
    single thing the caller has to fix. It is also not worth retrying: the credentials will not
    become valid between attempts.
    """
    spec, _ = chunked_tool
    monkeypatch.setenv("PROTO_ON_PARTIAL_FAILURE", policy)

    def cannot_dispatch(tool_key, inputs_list, configs):
        raise error

    monkeypatch.setattr("proto_tools.modal.dispatch_batch_to_modal", cannot_dispatch)

    with pytest.raises(type(error)) as caught:
        spec.function(_RemoteInput(items=[f"s{n}" for n in range(8)]), _RemoteConfig(device="modal"))

    assert not isinstance(caught.value, PartialFailureError), "a setup problem is not a partial failure"


def test_a_setup_failure_is_not_retried(chunked_tool, monkeypatch):
    """Credentials do not become valid between attempts, so a retry only delays the message."""
    spec, _ = chunked_tool
    attempts: list[int] = []

    def cannot_dispatch(tool_key, inputs_list, configs):
        attempts.append(len(inputs_list))
        raise PermissionError("cloud rejected the configured key")

    monkeypatch.setattr("proto_tools.modal.dispatch_batch_to_modal", cannot_dispatch)

    with pytest.raises(PermissionError):
        spec.function(_RemoteInput(items=[f"s{n}" for n in range(8)]), _RemoteConfig(device="modal"))

    assert attempts == [3], "one attempt only, and no per-chunk retry rounds"


def test_a_missing_asset_still_propagates_through_the_partial_machinery(chunked_tool, monkeypatch):
    """MissingAssetError is documented as always propagating; a placeholder would break that.

    The pytest skip hook catches the real type to turn an unprovisioned asset into a skip, so
    absorbing it into a FailedItem would silently stop those skips from happening.
    """
    from proto_tools.utils.tool_io import MissingAssetError

    spec, _ = chunked_tool
    monkeypatch.setenv("PROTO_ON_PARTIAL_FAILURE", "return_partial")

    def no_asset(tool_key, inputs_list, configs):
        raise MissingAssetError("esmfold", "weights")

    monkeypatch.setattr("proto_tools.modal.dispatch_batch_to_modal", no_asset)

    with pytest.raises(MissingAssetError):
        spec.function(_RemoteInput(items=[f"s{n}" for n in range(8)]), _RemoteConfig(device="modal"))


# ── cutting a batch by cost rather than count ───────────────────────────────


def _span_costs(spans, costs):
    """Total cost carried by each span."""
    return [round(sum(costs[start:stop])) for start, stop in spans]


def test_a_mixed_batch_is_cut_so_the_pieces_carry_similar_work():
    """Even counts of uneven items give pieces of wildly different work, and the slowest one wins."""
    costs = [500.0] * 20 + [50.0] * 20

    spans = chunk_indices(40, 32, costs)

    assert [stop - start for start, stop in spans] != [32, 8], "counting alone gives lopsided work"
    assert _span_costs(spans, costs) == [5500, 5500], "cost is split evenly instead"


def test_the_cut_does_not_depend_on_which_end_the_batch_is_read_from():
    """The previous attempt derived one size from a prefix, so reversing the batch changed the split."""
    costs = [500.0] * 20 + [50.0] * 20

    forward = _span_costs(chunk_indices(40, 32, costs), costs)
    backward = _span_costs(chunk_indices(40, 32, costs[::-1]), costs[::-1])

    assert forward == backward, f"same items, different order, different work per piece: {forward} vs {backward}"


def test_uniform_cost_cuts_exactly_as_counting_did():
    """Most tools declare no cost, so their behaviour must be untouched."""
    assert chunk_indices(40, 32, [1.0] * 40) == chunk_indices(40, 32, None) == [(0, 32), (32, 40)]
    assert chunk_indices(100, 7, [1.0] * 100) == chunk_indices(100, 7, None)


def test_every_item_is_dispatched_once_whatever_the_costs():
    """Coverage is the invariant a caller relies on; a gap would silently drop work."""
    import random

    rng = random.Random(0)
    for _ in range(500):
        total = rng.randint(1, 40)
        cap = rng.randint(1, 12)
        costs = [float(rng.choice([1, 5, 50, 500])) for _ in range(total)]

        spans = chunk_indices(total, cap, costs)

        covered = [i for start, stop in spans for i in range(start, stop)]
        assert covered == list(range(total)), f"coverage broke for {total=} {cap=}"
        assert all(stop - start <= cap for start, stop in spans), f"ceiling broke for {total=} {cap=}"
        # An empty span leaves coverage intact, so it needs asserting separately: it would
        # dispatch a container with nothing to do.
        assert all(start < stop for start, stop in spans), f"empty span for {total=} {cap=}"
        assert len(spans) <= total, f"more pieces than items for {total=} {cap=}"


def test_costs_that_do_not_describe_the_batch_are_refused():
    """A misaligned cost list would silently cut in the wrong places."""
    with pytest.raises(ValueError, match="describes 3 item"):
        chunk_indices(4, 2, [1.0, 1.0, 1.0])


class _CostlyInput(_RemoteInput):
    """Inputs whose items carry a real cost, as a structure predictor's do."""

    @classmethod
    def item_cost(cls, item: Any) -> float:
        """Cost is the item's length, so 'sssss' is worth five times 's'."""
        return float(len(item))


def test_a_tool_declaring_a_cost_has_its_batch_cut_by_that_cost(monkeypatch):
    """End to end: what reaches each chunk reflects cost, not a fixed count."""
    key = "remote-pipeline-probe-costly"
    spec, seen = _register_probe(monkeypatch, key, uses_gpu=True, max_chunk_size=4)
    monkeypatch.setattr(spec, "input_model", _CostlyInput, raising=False)
    try:
        # Two heavy items then six light ones, all distinct so dedup does not collapse them.
        # Counting alone would cut 4/4, ignoring that the first two dominate the work.
        items = ["x" * 20, "z" * 20, "a", "b", "c", "d", "e", "f"]

        spec.function(_CostlyInput(items=items), _RemoteConfig(device="modal"))

        sizes = [len(chunk) for chunk in seen["dispatches"]]
        assert sizes != [4, 4], "a fixed count ignores that the first two items dominate the work"
        assert sum(sizes) == len(items), "every item still dispatched exactly once"
        assert all(size <= 4 for size in sizes), "the count ceiling still binds"
    finally:
        ToolRegistry._registry.pop(key, None)


def test_isolating_an_expensive_item_may_cost_an_extra_piece():
    """Cost can cut more pieces than counting alone, and that is the point rather than a surprise.

    Peeling the heavy item out is what stops one container carrying almost all the work. Each piece
    still respects the ceiling, and pieces never outnumber items, so the extra ones are dispatches
    against warm containers rather than fresh starts.
    """
    costs = [100.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    spans = chunk_indices(6, 2, costs)

    assert len(spans) > len(chunk_indices(6, 2, None)), "the heavy item earns its own piece"
    assert spans[0] == (0, 1), "and it is alone in it"
    assert all(stop - start <= 2 for start, stop in spans), "the ceiling still binds"
    assert len(spans) <= 6, "never more pieces than items"


# ── adjusting a config for a hosted environment ─────────────────────────────


def test_a_local_database_search_is_forced_remote_only_when_hosted(monkeypatch):
    """A hosted environment cannot stage a corpus, so a search that needs one has to give way.

    a tool needing a staged corpus is out of scope for a hosted deployment — ``uniref30-2302`` is
    hundreds of gigabytes — so the remote API is the only search such a process can run.
    """
    from proto_tools.tools.sequence_alignment.mmseqs2.homology_search import Mmseqs2HomologySearchConfig
    from proto_tools.utils import run_preprocess

    spec = ToolRegistry.get("boltz2-prediction")
    example = spec.example_input()

    def searched_with(hosted: bool) -> str:
        """Run preprocess with a local search asked for, and report what it actually used."""
        monkeypatch.delenv("PROTO_IS_HOSTED_ENV", raising=False)
        if hosted:
            monkeypatch.setenv("PROTO_IS_HOSTED_ENV", "1")
        config = spec.config_model(use_msa=False, msa_search_config=Mmseqs2HomologySearchConfig(search_mode="local"))
        _, prepared = run_preprocess(config, example)
        assert config.msa_search_config.search_mode == "local", "the caller's own config is never rewritten"
        return prepared.msa_search_config.search_mode

    assert searched_with(hosted=False) == "local", "a user's own machine keeps the search they asked for"
    assert searched_with(hosted=True) == "remote", "a hosted process cannot reach a local database"


def test_a_config_with_nothing_to_adjust_is_returned_unchanged():
    """Almost every config has no setting that depends on a staged corpus."""
    config = _RemoteConfig(device="modal")

    assert config.for_hosted_env() is config, "the default hook must not copy for no reason"


@pytest.mark.parametrize(
    ("value", "hosted"),
    [("1", True), ("true", True), ("0", False), ("", False)],
)
def test_the_hosted_signal_reads_as_a_flag(monkeypatch, value, hosted):
    """Set by whichever environment hosts the tool, so an unset or zero value means a user machine."""
    from proto_tools.utils.base_config import is_hosted_env

    monkeypatch.setenv("PROTO_IS_HOSTED_ENV", value)

    assert is_hosted_env() is hosted
