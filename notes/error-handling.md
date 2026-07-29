# Error Handling

This note covers how `proto_tools` decides whether a tool exception is raised or captured, including capture mode, the retry loop, the `MissingAssetError` carve-out, pool error aggregation, and GPU out-of-memory handling.

The `@tool` decorator in `proto_tools/tools/tool_registry.py` **raises by default** when a tool function (or its retry-exhausted `_RETRYABLE_EXCEPTIONS` loop, or the `dispatch_to_proto` call) lets an exception escape. Callers see the original exception with a meaningful traceback at the call site.

Capture mode, where the exception is packed into `output.errors` with `success=False` and returned to the caller instead of raising, is opt-in.

## Toggling capture mode

```bash
PROTO_CAPTURE_ERRORS=1 python my_script.py
```

When set to `"1"`, every tool exception in the process is packed into a `success=False` output instead of raising. The variable is read **dynamically per call**, so test code can use `monkeypatch.setenv("PROTO_CAPTURE_ERRORS", "1")` to scope the change to a single test.

This is a process-wide setting, with no per-call kwarg.

## When are `success` / `errors` populated?

`BaseToolOutput.success` and `BaseToolOutput.errors` form a structured error contract that is only meaningful in capture mode:

| Path | `success` | `errors` |
|---|---|---|
| Tool returns normally (any mode)         | `True`  | `[]` |
| Tool raises, default mode                | call raises, no output is returned |
| Tool raises, `PROTO_CAPTURE_ERRORS=1`    | `False` | `["TypeName: msg", "<traceback>"]` |
| Tool raises `MissingAssetError`, any mode | call raises, env var ignored |

Treat the fields as the wire-format contract for capture mode, and don't write code that reads them under the default raise path, where they will only ever be `success=True, errors=[]`.

`BaseToolOutput.__getattr__` raises `ToolExecutionError` when you access a declared result field on a `success=False` output. That mechanism keeps working in capture mode and is harmless on the raise path (it never fires because every returned output has `success=True`).

## Carve-out: `MissingAssetError` always raises

`MissingAssetError` (signaled by `proto_resolve_asset_availability` in `standalone_helpers.sh` and raised in `proto_tools/utils/tool_instance.py`) **always propagates**, regardless of `PROTO_CAPTURE_ERRORS`. The pytest skip hook in `tests/conftest.py` relies on catching the real exception type to convert unprovisioned-asset failures into skips on machines that don't have gated weights / large databases.

## Retry loop is unchanged

The wrapper retries `ConnectionError` (and any other entry in `_RETRYABLE_EXCEPTIONS`) up to `MAX_RETRIES` times before deciding what to do with the exception. Only the **final** exception, after retries are exhausted, is subject to the raise-vs-capture decision.

`TimeoutError` is intentionally non-retryable, because hitting the timeout once means hitting it again at the same limit, so it is surfaced immediately.

## ToolPool

`ToolPool._parallel_dispatch` is independent of the `PROTO_CAPTURE_ERRORS` policy, because pool partitions call the **raw undecorated** tool function, bypassing the `@tool` wrapper entirely. Per-partition exceptions are caught by the pool's own `try: future.result() except Exception` and aggregated into `PartialFailureError`, with the original exception type preserved on `PartialFailureError.failed[i]["exception"]` and successful partitions' results preserved on `PartialFailureError.succeeded`.

A partition is one execution of a split batch exactly as a remote chunk is, so `ToolPool` honours `PROTO_ON_PARTIAL_FAILURE` too — see [A split batch that partly succeeds](#a-split-batch-that-partly-succeeds).

## How a batch is cut

Two bounds apply: **cost** decides where a cut falls, **count** caps how wide a piece gets. A tool declares cost by overriding `BaseToolInput.item_cost`; the default of `1.0` per item makes the two measures identical, so a tool that declares nothing splits exactly as it did before.

Both fan-out paths read cost through `fanout.item_costs` and pack differently, because they execute differently:

| | `chunk_indices` (remote) | `lpt_schedule` (pool) |
|---|---|---|
| Pieces | variable, contiguous | fixed — one per device |
| Why | `starmap` reuses warm containers, so one finishing early pulls the next piece | each partition runs exactly once, so balance must be settled upfront |

Contiguity is load-bearing on the remote side: a chunk's derived seed comes from its start index, and the merge concatenates chunks in order. `lpt_schedule` reorders, which is why it stays the pool's.

Spans are computed from the whole batch's costs rather than a running total, so the same items in a different order give pieces carrying the same work. Nothing is added to the cache key: cost is a pure function of the item, and the items and `max_chunk_size` are already in it.

## A split batch that partly succeeds

`max_chunk_size` splits one call across several executions, so a batch can come back part succeeded and part failed. The chunks that worked were already billed, so they are preserved rather than discarded along with the failure.

**A setup failure is not a partial failure.** Missing Modal credentials, a tool that is shipped but not deployed, a client that is not installed — none of these say anything about the items being sent, so every chunk would fail identically. `fanout.is_fatal_dispatch_error` covers `ImportError`, `NotImplementedError`, `PermissionError`, `MissingAssetError`, and proto-modal's `ModalDispatchError` family (`ModalCredentialsError`, `ToolNotDeployedError`, `ToolNotShippedError`). These bypass the machinery below and surface as themselves, and they are not retried — credentials do not become valid between attempts. Both fan-out paths consult it, since `ToolPool` catches a bare `Exception` per partition and would otherwise absorb them. `MissingAssetError` is included so the pytest skip hook still sees the real exception type (see the carve-out above).

`KeyboardInterrupt`, `SystemExit`, and `asyncio.CancelledError` derive from `BaseException`, so the `except Exception` clauses never see them.

**Retry first.** Only the failed chunks are resent, never the whole batch, using the wrapper's own `MAX_RETRIES` / `RETRY_DELAY` backoff. `_is_retryable_chunk_error` is deliberately narrow — transport faults and transient GPU acquisition — and excludes anything that would fail identically on a second attempt, such as a malformed result or a container whose CUDA context is already poisoned. A retried chunk keeps its own config, so a derived seed is unchanged and a stochastic result stays reproducible.

**Then the caller's policy applies**, set with `PROTO_ON_PARTIAL_FAILURE`:

| Value | Behaviour |
|---|---|
| unset (default) | `PartialFailureError`, carrying `succeeded` and `failed` — the same shape `ToolPool` has always raised |
| `return_partial` | The output, with a `FailedItem` at each failed position and the failure recorded in `errors` |

An environment variable rather than a config field: a config crosses to a worker running whatever proto-tools its image was built with, and `extra="forbid"` means a field that version lacks is rejected outright, breaking every deployed container until each is rebuilt.

`return_partial` keeps the item list the same length as the input, so `zip(inputs, results)` still lines up; omitting the failures would shift every position after the first gap. Each `FailedItem` carries an `error` naming why its position is empty. The output model still declares `list[Item]` while the list holds `list[Item | FailedItem]` — the merge uses `model_copy(update=...)`, so nothing re-validates and no tool's output model is widened for an opt-in mode.

**A total failure raises even under `return_partial`.** The mode salvages survivors, and there are none. It is also the only safe answer: six iterable tools declare required output fields carried from a chunk that succeeded (`borzoi-prediction` needs `output_tracks`, `species`, `replicate`, `avg_output_tracks`), and an output fabricated without them raises `AttributeError` on access. A caller using `return_partial` must therefore still handle `PartialFailureError`, which here means not one bad item but nothing working at all — usually something systemic, such as a dead container or a poisoned worker (#100).

**Successes are cached, failures never are.** A retry of the same batch then dispatches only what failed. Caching a `FailedItem` as though it were a result would make the failure permanent, and a later identical call would hand back the failures without retrying them, so every cache write is suppressed for a partial result — per-item entries, the output template, and the whole-call entry alike. Survivors are stored separately, before the partial output is assembled. This applies to deterministic iterable tools; a stochastic tool uses the whole-call cache, so its successes are reported but not stored. `ToolPool` merges its own partial output and returns it like any other, so the wrapper inspects what came back with `_holds_failed_items` rather than being told.

`post_process_iterable` is handed only the items that succeeded. A hook works on the item type its tool declares — the four embedding tools reach for `mean_embedding` to attach UMAP projections — so a placeholder would raise and destroy the partial result. Survivors are still mutated in place, exactly as in a batch that never failed.

Indices reaching a caller are always the caller's own positions. Dedup and the per-item cache shorten the batch before dispatch, so the wrapper translates back through both, and one dispatched item can map to several positions when duplicates were collapsed onto it.

A chunk is all-or-nothing — one returning the wrong number of items is recorded as that chunk's failure — which is what makes caching a survivor safe.

## Remote dispatch / `dispatch_to_proto`

`proto_tools.proto.dispatch_to_proto` raises on remote failure, and the wrapper propagates that exception to the caller by default. Setting `PROTO_CAPTURE_ERRORS=1` packs the remote exception into a `success=False` output, identical to the local-execution capture path.

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
| `proto_tools/utils/tool_io.py` | `MissingAssetError` (carve-out); `BaseToolOutput.success` / `BaseToolOutput.errors` capture-mode contract; `__getattr__` deferred raise on `success=False` outputs |
| `tests/conftest.py` | Pytest hook that catches `MissingAssetError` and converts to skip |
