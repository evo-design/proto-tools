# Error Handling

This note covers how `proto_tools` decides whether a tool exception is raised or captured, including capture mode, the retry loop, the `MissingAssetError` carve-out, pool error aggregation, and GPU out-of-memory handling.

The `@tool` decorator in `proto_tools/tools/tool_registry.py` **raises by default** when a tool function (or its retry-exhausted `_RETRYABLE_EXCEPTIONS` loop, or the cloud `_try_dispatch` hook) lets an exception escape. Callers see the original exception with a meaningful traceback at the call site.

Capture mode, where the exception is packed into `output.errors` with `success=False` and returned to the caller instead of raising, is opt-in.

## Toggling capture mode

```bash
PROTO_CAPTURE_ERRORS=1 python my_script.py
```

When set to `"1"`, every tool exception in the process is packed into a `success=False` output instead of raising. The variable is read **dynamically per call**, so test code can use `monkeypatch.setenv("PROTO_CAPTURE_ERRORS", "1")` to scope the change to a single test.

This is a process-wide knob, with no per-call kwarg.

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

`ToolPool._parallel_dispatch` is independent of the policy, because pool partitions call the **raw undecorated** tool function, bypassing the `@tool` wrapper entirely. Per-partition exceptions are caught by the pool's own `try: future.result() except Exception` and aggregated into `PartialFailureError`, with the original exception type preserved on `PartialFailureError.failed[i]["exception"]` and successful partitions' results preserved on `PartialFailureError.succeeded`.

## Cloud / `_try_dispatch`

`proto_tools.cloud._route_to_cloud` raises on remote failure, and the wrapper propagates that exception to the caller by default. Setting `PROTO_CAPTURE_ERRORS=1` packs the cloud exception into a `success=False` output, identical to the local-execution capture path.

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
