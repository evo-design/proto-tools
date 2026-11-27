# Error Handling

This note covers how `proto_tools` handles tool exceptions, including capture
mode, retries, errors that always raise, partial batch failures, and GPU
out-of-memory errors.

Tool calls raise exceptions by default. Callers receive the original exception
and a traceback pointing to the call site. This applies to errors from the tool
function, errors left after the wrapper's retries, and remote dispatch errors.

To receive an output containing the error instead, enable capture mode. The
`@tool` decorator in `proto_tools/tools/tool_registry.py` then returns an output
with `success=False` and the exception recorded in `output.errors`.

## Enabling capture mode

```bash
PROTO_CAPTURE_ERRORS=1 python my_script.py
```

Set `PROTO_CAPTURE_ERRORS` to `"1"` to return tool exceptions in a `success=False`
output, except for the errors listed below that always raise. The setting is read
on each call, so tests can use `monkeypatch.setenv("PROTO_CAPTURE_ERRORS", "1")`
to enable it for a single test.

The setting applies to the whole process. There is no per-call argument.

## When are `success` / `errors` populated?

`BaseToolOutput.success` and `BaseToolOutput.errors` form a structured error contract that is only meaningful in capture mode:

| Path | `success` | `errors` |
|---|---|---|
| Tool returns normally (any mode)         | `True`  | `[]` |
| Tool raises, default mode                | call raises, no output is returned |
| Tool raises, `PROTO_CAPTURE_ERRORS=1`    | `False` | `["TypeName: msg", "<traceback>"]` |
| Tool raises `MissingAssetError`, any mode | call raises, env var ignored |
| Call matches a declared outage, any mode | call raises `OutageError` before dispatch, env var ignored |

Use these fields to inspect failures in capture mode. With the default behavior,
a failed call raises instead of returning an output, so every returned output
has `success=True` and `errors=[]`.

`BaseToolOutput.__getattr__` raises `ToolExecutionError` when you access a declared result field on a `success=False` output. That mechanism keeps working in capture mode and is harmless on the raise path (it never fires because every returned output has `success=True`).

## `MissingAssetError` always raises

`MissingAssetError` (signaled by the `ASSET_NOT_AVAILABLE` sentinel a tool's `setup.sh` prints before exiting 64, and raised in `proto_tools/utils/tool_instance.py`) **always propagates**, regardless of `PROTO_CAPTURE_ERRORS`. The pytest skip hook in `tests/conftest.py` relies on catching the real exception type to convert unprovisioned-asset failures into skips on machines that don't have gated weights / large databases.

## Upstream outages: `OutageError`

A tool declares upstream outages with `@tool(..., outages=[Outage(...)])` (`proto_tools/utils/outage.py`). An `Outage` covers the whole tool, or one mode of it when `affects` (a label such as `"include_msa"`) and `when` (a predicate over `(inputs, config)`) are set. `when` is never serialized, so an outage rebuilt from a dump keeps its label and matches nothing. A predicate that raises is logged and treated as not matching, so a broken declaration never blocks a call. A call that matches is refused in the wrapper, right after the empty-batch check and before any dispatch backend, device, or network call, with `OutageError(ValueError)`. It is raised the way a pydantic `ValidationError` is, so `PROTO_CAPTURE_ERRORS` does not turn it into `output.errors`.

`PROTO_IGNORE_OUTAGES=1` logs a warning and lets the call through, to check whether the upstream has recovered. The MCP `run_tool` runs the same check (`active_outage`) itself, since its remote branches do not go through the wrapper, and answers `{"ok": false, "error": ..., "outage": {...}}`.

The check runs at the entry point only. A Modal container skips it (`outage_checks_skipped` in `dispatch_tool_call`): the caller already checked with its override in view, and the container can neither see that override nor return a refusal except as an opaque exception Modal would retry. `OutageError` pickles, carrying the outage without its predicate.

To lift an outage, delete the `Outage` and its recovery probe's xfail marker (see `notes/testing.md`).

## Retries

The wrapper retries `ConnectionError` (and any other entry in `_RETRYABLE_EXCEPTIONS`) up to `MAX_RETRIES` times before deciding what to do with the exception. Only the **final** exception, after retries are exhausted, is subject to the raise-vs-capture decision.

`TimeoutError` is intentionally non-retryable, because hitting the timeout once means hitting it again at the same limit, so it is surfaced immediately.

## ToolPool

`ToolPool._parallel_dispatch` is independent of the `PROTO_CAPTURE_ERRORS` policy, because pool partitions call the **raw undecorated** tool function, bypassing the `@tool` wrapper entirely. Per-partition exceptions are caught by the pool's own `try: future.result() except Exception` and aggregated into `PartialFailureError`, with the original exception type preserved on `PartialFailureError.failed[i]["exception"]` and successful partitions' results preserved on `PartialFailureError.succeeded`.

A partition is one execution of a split batch exactly as a remote chunk is, so `ToolPool` honours `PROTO_ON_PARTIAL_FAILURE` too — see [A split batch that partly succeeds](#a-split-batch-that-partly-succeeds).

## How a batch is cut

**Cost** determines where to split a batch, and **count** limits how many items
each piece can hold. A tool declares cost by overriding `BaseToolInput.item_cost`. The default of `1.0` per item makes the two measures identical, so a tool that declares nothing splits exactly as it did before.

Both fan-out paths read cost through `fanout.item_costs` and pack differently, because they execute differently:

| | `chunk_indices` (remote) | `lpt_schedule` (pool) |
|---|---|---|
| Pieces | variable, contiguous | fixed — one per device |
| Why | `starmap` reuses warm containers, so one finishing early pulls the next piece | each partition runs exactly once, so balance must be settled upfront |

Remote chunks must stay contiguous because each chunk's derived seed uses its
start index and the merge concatenates chunks in order. `lpt_schedule` reorders
items, so it is used only by the local pool.

Spans are computed from the whole batch's costs rather than a running total, so the same items in a different order give pieces carrying the same work. Nothing is added to the cache key: cost is a pure function of the item, and the items and `max_chunk_size` are already in it.

## A split batch that partly succeeds

`max_chunk_size` splits one call across several executions. Some chunks may
succeed while others fail. Successful chunks are preserved because their
compute has already been billed.

**A setup failure is not a partial failure.** Missing Modal credentials, a tool that is shipped but not deployed, a client that is not installed — none of these say anything about the items being sent, so every chunk would fail identically. `fanout.is_fatal_dispatch_error` covers `ImportError`, `NotImplementedError`, `PermissionError`, `MissingAssetError`, and proto-modal's `ModalDispatchError` family (`ModalCredentialsError`, `ToolNotDeployedError`, `ToolNotShippedError`). These bypass the machinery below and surface as themselves, and they are not retried — credentials do not become valid between attempts. Both fan-out paths consult it, since `ToolPool` catches a bare `Exception` per partition and would otherwise absorb them. `MissingAssetError` is included so the pytest skip hook still sees the real exception type (see above).

`KeyboardInterrupt`, `SystemExit`, and `asyncio.CancelledError` derive from `BaseException`, so the `except Exception` clauses never see them.

**Retry first.** Only the failed chunks are resent, never the whole batch, using the wrapper's own `MAX_RETRIES` / `RETRY_DELAY` backoff. `_is_retryable_chunk_error` is deliberately narrow — transport faults and transient GPU acquisition — and excludes anything that would fail identically on a second attempt, such as a malformed result or a container whose CUDA context is already poisoned. A retried chunk keeps its own config, so a derived seed is unchanged and a stochastic result stays reproducible.

**Then the caller's policy applies**, set with `PROTO_ON_PARTIAL_FAILURE`:

| Value | Behaviour |
|---|---|
| unset (default) | `PartialFailureError`, carrying `succeeded` and `failed` — the same shape `ToolPool` has always raised |
| `return_partial` | The output, with a `FailedItem` at each failed position and the failure recorded in `errors` |

An environment variable rather than a config field: a config crosses to a worker running whatever proto-tools its image was built with, and `extra="forbid"` means a field that version lacks is rejected outright, breaking every deployed container until each is rebuilt.

`return_partial` keeps the item list the same length as the input, so `zip(inputs, results)` still lines up. Omitting the failures would shift every position after the first gap. Each `FailedItem` carries an `error` naming why its position is empty. The output model still declares `list[Item]` while the list holds `list[Item | FailedItem]` — the merge uses `model_copy(update=...)`, so nothing re-validates and no tool's output model is widened for an opt-in mode.

**A total failure raises even under `return_partial`.** There are no successful results to return. It is also the only safe answer: six iterable tools declare required output fields carried from a chunk that succeeded (`borzoi-prediction` needs `output_tracks`, `species`, `replicate`, `avg_output_tracks`), and an output fabricated without them raises `AttributeError` on access. A caller using `return_partial` must therefore still handle `PartialFailureError`, which here means not one bad item but nothing working at all — usually something systemic, such as a dead container or a poisoned worker (#100).

**Successes are cached, failures never are.** A retry of the same batch then dispatches only what failed. Caching a `FailedItem` as though it were a result would make the failure permanent, and a later identical call would hand back the failures without retrying them, so every cache write is suppressed for a partial result — per-item entries, the output template, and the whole-call entry alike. Survivors are stored separately, before the partial output is assembled. This applies to deterministic iterable tools. A stochastic tool uses the whole-call cache, so its successes are reported but not stored. `ToolPool` merges the partial output itself and returns it like any other, so the wrapper inspects what came back with `_holds_failed_items` rather than being told.

`post_process_iterable` is handed only the items that succeeded. A hook expects the tool's declared item type and may access fields a failure
placeholder does not have. Passing a placeholder could raise and lose the
partial result. Successful items are still modified in place, just as in a
batch that fully succeeded.

Indices reaching a caller are always the caller's own positions. Dedup and the per-item cache shorten the batch before dispatch, so the wrapper translates back through both, and one dispatched item can map to several positions when duplicates were collapsed onto it.

A chunk is all-or-nothing — one returning the wrong number of items is recorded as that chunk's failure — which is what makes caching a survivor safe.

## Remote dispatch / `dispatch_to_modal`

`proto_tools.modal.dispatch_to_modal` raises on remote failure, and the wrapper propagates that exception to the caller by default. Setting `PROTO_CAPTURE_ERRORS=1` packs the remote exception into a `success=False` output, identical to the local-execution capture path.

## GPU out-of-memory

OOM is hardware/config-dependent (tokens x batch x precision x VRAM), so tools do not
predict it with fixed caps. `standalone_helpers.oom` provides `is_cuda_oom`,
`release_cuda_memory`, and `oom_guard` / `raise_oom`. On a real OOM a tool frees cached GPU
memory and raises an actionable `GpuOutOfMemoryError` instead of a deep CUDA trace. The
`@tool` decorator does **not** retry OOM, since a retry would hit the same limit, but
ESMFold and ESMFold2 do their own in-tool reactive recovery (batch / sampling-step halving)
before surfacing the error.

## Files

| File | Role |
|---|---|
| `proto_tools/tools/tool_registry.py` | `_should_capture_errors()`, `PROTO_CAPTURE_ERRORS` env var, `_make_error_output_or_raise()` helper at the three exception sites in the `@tool` wrapper |
| `proto_tools/utils/tool_io.py` | `MissingAssetError`, capture-mode fields `success` and `errors`, and deferred result-field errors through `__getattr__` |
| `tests/conftest.py` | Pytest hook that catches `MissingAssetError` and converts to skip |
