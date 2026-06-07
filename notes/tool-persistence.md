# Tool Persistence Reference

This note covers how `proto_tools` keeps model workers alive across calls,
namely the worker cache hierarchy, the dispatch resolution paths,
preprocess/dispatch worker sharing, config-change restarts, multi-GPU fan-out,
and device management. The user-facing walkthrough lives in the Tool Persistence
guide, while this is the internals reference.

Source of truth: `proto_tools/utils/tool_instance.py`,
`proto_tools/utils/persistent_worker.py`, `proto_tools/utils/tool_pool.py`,
`proto_tools/utils/device_manager.py`.

## Mental model

Every tool runs inside an isolated environment managed by `ToolInstance`. By
default each `run_*` call is **one-shot**: `dispatch()` spawns an ephemeral
subprocess, loads the model, runs inference, and tears the subprocess down when
the call returns. Nothing leaks between calls and no GPU memory is retained.

<img src="../guides/assets/tool-persistence/default.svg" alt="In the default one-shot mode, each call reloads the model and releases its GPU memory afterward." width="560">

One-shot is the safe default but pays the model load on every call. A forward
pass may take well under a second, while loading a structure-prediction or
language model takes seconds to tens of seconds. **Persistence** keeps the worker
subprocess (and its loaded model) alive between calls so the load is paid once.

There are three opt-in mechanisms, all peers of the one-shot default rather than
upgrades over it:

| Mechanism | Scope | Cleanup | Use when |
|---|---|---|---|
| `ToolInstance.persist()` | Every tool called in the block | Automatic on block exit | Batch loops, optimization passes (recommended default) |
| `ToolInstance.persist_tool(name)` | One named tool, possibly several live instances | Automatic on block exit | Multiple simultaneous workers for one tool (multi-GPU) |
| `ToolInstance.get(name)` | A single worker, manual lifetime | Manual (`shutdown()`) | Sessions that outlive any one block (notebooks) |

`ToolInstance.scope()` is a fourth, lower-level primitive, an isolated cache for
test fixtures and batch jobs (see [Cache scopes](#cache-scopes)).

## The worker cache hierarchy

Persistence is bookkeeping over a set of caches. A `ToolInstance` is a handle to
one isolated environment plus (optionally) a live persistent worker. Cached
instances are looked up by a string key (the toolkit name by default, or an
explicit `instance_name`).

Three layers exist, consulted in this order by `_lookup_instance(key)`:

1. **Auto-persist overlay** (`_auto_persist_overlay`, a `ContextVar`) is a
   per-task overlay seeded by `_auto_persist_scope` so that a tool's
   `preprocess` and its main dispatch share one worker within a single call
   (see [Preprocess/dispatch sharing](#preprocessdispatch-worker-sharing)).
2. **Active cache** (`_active_cache()`) is the scope-local cache when a
   `scope()` / `persist()` block is active (`_scope_override`), otherwise the
   process-global cache `_instances`.
3. (no further fallback). A miss means no warm worker exists.

`get()` writes into `_active_cache()`. Under `persist()`/`scope()` that is the
scope-local dict, so auto-created workers never leak into the global cache.

### dispatch() resolution paths

`ToolInstance.dispatch(toolkit, input_dict, *, instance=None, script_path=None, config=None)`
resolves a worker as follows. The `instance` kwarg is forwarded from the
`instance=` argument on every `run_*` call.

- **Path 1, `instance` is a `ToolInstance`**: use it directly. The caller owns
  the handle.
- **Path 2, `instance` is a `str`**: resolve it as a cache key via
  `_lookup_instance`. If found, reuse it. If not found but a persist context
  is active, create-and-register under that key, and if not found and no persist
  context is active, raise `ValueError`. A bare string is a reference, not a
  creation request.
- **Path 3, `instance is None`**: look up the toolkit key. If a cached
  instance exists, reuse it. Otherwise, if persist mode is active, create and
  cache one (`get(toolkit)`). Otherwise fall through.
- **Path 4, no cached instance, no persist mode**: run a one-shot ephemeral
  subprocess (`_oneshot`). For GPU devices this takes a transient device lease
  so concurrent one-shot calls do not stomp the same GPU.

## persist(): auto-cache everything in a block

```python
with ToolInstance.persist():
    for seq in sequences:
        run_esmfold(inputs, config)   # call 1 loads; calls 2+ reuse the worker
    run_esm2_score(other, cfg)        # a different tool, also cached on first use
# all auto-created workers shut down here; GPU memory released
```

`persist()` sets the `_persist_mode` `ContextVar` and opens an internal
`scope()`. Inside the block, the first dispatch of a given toolkit takes Path 3,
creates a worker, and caches it in the scope-local cache. Subsequent dispatches
reuse it. On exit, every worker created in the scope is shut down.

<img src="../guides/assets/tool-persistence/persist.svg" alt="Inside a persist() block the model loads once and many inferences reuse the warm worker, which shuts down on block exit." width="560">

`persist()` does not take a `device=` argument. The first call inside the block
establishes the configuration, and a later change is picked up by the
[config-change restart](#auto-restart-on-config-change-reload_on_change) path.

## persist_tool(): named instances and multi-GPU

`persist_tool(toolkit, *, instance_name=None, env_overrides=None)` scopes
persistence to a single tool. For most batch work `persist()` is sufficient.
Reach for `persist_tool()` when you need **more than one live worker for the
same tool at once**, most often one worker per GPU:

```python
with ToolInstance.persist_tool("esmfold", instance_name="worker_a") as a:
    with ToolInstance.persist_tool("esmfold", instance_name="worker_b"):
        out_a = run_esmfold(inputs, ESMFoldConfig(device="cuda:0"), instance=a)
        out_b = run_esmfold(inputs, ESMFoldConfig(device="cuda:1"), instance="worker_b")
```

Each named instance runs in its own subprocess. At call time, route to a
specific worker by passing the handle (or its name as a string) as `instance=`.

With no `instance_name`, `persist_tool()` claims the toolkit slot atomically. If
that slot is already taken, the new instance is **not** auto-used by bare calls.
A warning is logged and you must pass it explicitly via `instance=`.

<img src="../guides/assets/tool-persistence/named.svg" alt="persist_tool() with named instances keeps several workers for one tool alive at once, typically one per GPU." width="560">

## get() / shutdown(): manual lifetime

```python
tool = ToolInstance.get("esmfold")          # create or fetch; cached until closed
for seq in sequences:
    run_esmfold(inputs, config)             # reuses the cached worker
tool.shutdown()                             # stop worker, evict from cache
# or, without a handle:
ToolInstance.shutdown_instance("esmfold")
```

`get()` is the right tool when the unit of work is a session rather than a
block, for example keeping a model warm across many notebook cells, including
idle periods. `get(toolkit, instance_name="K")` registers under `"K"` so later
`dispatch(..., instance="K")` calls can reference it.

<img src="../guides/assets/tool-persistence/get.svg" alt="get() keeps a single worker warm across calls until it is shut down." width="560">

## Cache scopes

`ToolInstance.scope()` swaps `_active_cache()` for a fresh, isolated dict via the
`_scope_override` `ContextVar`. Workers created inside are shut down on exit and
the previous cache is restored. It is the isolation primitive that `persist()`
builds on, and is useful directly in test fixtures and self-contained batch
jobs that must not touch the global cache. Scopes are per-thread/per-task, so
concurrent callers do not collide.

## Preprocess/dispatch worker sharing

Some tools define a config `preprocess()` that **itself runs the model** before
the main inference. The canonical case is masked-language-model sampling
(`esm2-sample`, `esm3-sample`), where `preprocess` calls the embedding tool to
score positions for a model-based masking strategy and then the main sampling
call runs the same model. Without coordination that is two loads per call.

The `@tool` wrapper handles this with `ToolInstance._auto_persist_scope`. When a
tool has a standalone env **and** either a custom `preprocess` or an explicit
`instance=`, the wrapper opens a scope that:

1. creates one worker for the toolkit,
2. seeds it into the auto-persist overlay (consulted first by
   `_lookup_instance`), so the `preprocess` dispatch and the main dispatch both
   resolve to that single worker, then
3. on exit, releases it.

Outside `persist()`, the per-call worker is shut down on scope exit (correct for
one-shot). **Inside `persist()`**, the worker is instead handed to the persist
scoped cache so subsequent calls reuse it. This is what lets custom-preprocess
tools stay warm across a `persist()` block, not just within one call.

Implication for tool authors: marking a config field's `preprocess` triggers
this scope even if the preprocess does not touch the model. Keep `preprocess`
cheap, and rely on `reload_on_change` (below) rather than re-dispatching inside
it where avoidable.

## Persistent workers

A warm `ToolInstance` owns a `PersistentWorker` (`persistent_worker.py`), a
subprocess that stays alive between calls and communicates over a stdin/stdout
JSON-line protocol. Each call writes one JSON request line and reads one JSON
response line, and the model stays resident in the worker's process between
requests. Worker stderr is drained on a background thread and surfaced (bounded)
in crash diagnostics. See `notes/logging.md` for the worker logging architecture
and the `verbose` scale.

## Auto-restart on config change (`reload_on_change`)

A persistent worker reflects the configuration it was last loaded with. When a
**load-time** parameter changes between calls (`device`, `model_checkpoint`,
`model_name`, or any other field marked `reload_on_change=True` on the tool's
`Config`), the persistence layer detects the mismatch (`reload_fields()` /
`_reload_params`) and transparently restarts the worker with the new config
before running the call.

Two rules for tool authors:

- **Mark every init-affecting field** `reload_on_change=True`. If a field
  changes how the model is constructed but is not marked, a persistent worker
  will silently keep serving the stale model.
- **Standalone scripts must not check for config changes themselves.** The
  `ToolInstance` layer owns restarts. A standalone that re-inits on its own will
  double-load.

Runtime-only fields (e.g. `timeout`) set `reload_on_change=False` and do not
trigger restarts.

## ToolPool and parallel execution

`ToolPool` (`tool_pool.py`) parallelizes a single list-input tool call across
several devices. It intercepts the `@tool` call, partitions the input items
across persistent workers on different devices, runs the partitions concurrently
on a `ThreadPoolExecutor`, and reassembles results in original input order. It is
the multi-GPU counterpart to `persist()`. The user-facing walkthrough is the
Parallel Execution guide.

```python
from proto_tools.utils import ToolPool

with ToolPool(gpus="all"):              # or gpus=2, or gpus=["cuda:0", "cuda:1"]
    out = run_esmfold(ESMFoldInput(complexes=many_sequences), config)
```

### Constructor

`ToolPool(gpus="all", cpus=None)`:

- `gpus` is `"all"` (every visible GPU, default), an `int N` (the first N visible
  GPUs, and errors if N exceeds the visible count), an explicit device list
  (`["cuda:0", "cuda:2"]`, validated against the visible set), or `0` (explicit
  CPU-only mode, no GPU detection).
- `cpus` is the total CPU budget for CPU fan-out. `None` resolves to
  `min(_detect_cpus(), 4)`. `_detect_cpus()` is cgroup-aware
  (`os.sched_getaffinity`), so it respects Slurm/Kubernetes allocations.

### What gets partitioned

Only tools that declare `iterable_input_fields` are partitioned (e.g.
`["complexes"]` for ESMFold, `["sequences"]` for ESM2), and a tool with a single
indivisible input runs on one device regardless of pool size. `iterable_input_fields`
is a list (primary first): when an input carries index-parallel sibling lists (e.g.
structure-prediction `["complexes", "msas"]`), the framework slices, dedups, and
cache-keys the whole group so siblings stay aligned to the primary. A config may set
`gpus_per_instance > 1` (a worker that needs several GPUs, e.g. ColabFold), and the
pool then groups devices into slots of that size, while `gpus_per_instance == 0`
with a positive `cpus_per_instance` selects CPU fan-out mode instead.

### Cost-aware scheduling (LPT)

Items are distributed by **longest-processing-time-first (LPT)** bin-packing.
Each tool reports a per-item estimate via `BaseToolInput.item_cost()` (default
uniform `1.0`, and ESMFold overrides it with residue count). `ToolPool` sorts items
by descending cost and assigns each to the worker with the least total work, so
all devices finish at roughly the same wall-clock time. A device may set a
`max_item_cost` cap to route oversized items away from it. With uniform costs,
LPT reduces to round-robin.

<img src="../guides/assets/parallel-execution/lpt.svg" alt="Longest-processing-time-first scheduling assigns each item to the worker with the least total work so all devices finish at about the same time." width="560">

### Deduplication

Duplicate items in a batch are computed once and the result expanded back to
every position where the item appeared, so the returned list keeps the input's
length and order. This is the same dedup the runtime cache uses.

### Persistence and failure handling

Each partition runs as its own persistent worker, warm for the life of the
`with` block. Pooling composes with `persist()`, and the pool owns the workers
for the block. A partition that raises does not discard the others, so
successful results are preserved and a `PartialFailureError` reports which
partitions failed.

## Device management (`DeviceManager`)

`DeviceManager` places each persistent worker on a managed GPU and evicts the
least-recently-used worker when the pool is full. Persistence and pooling are
what drive these allocations. The device-string semantics, the allocation map,
the eviction strategies (RESTART by default, or CPU offload), the managed-device
pool, multi-per-device packing, and the `to_device()` mechanics are documented
in full in `notes/device-management.md`.

## Timeouts

Every tool call enforces a timeout, default 3600 seconds (`DEFAULT_TIMEOUT`,
overridable per call via the `timeout` config field). The timeout bounds the
**entire** wait for the worker's response — including any third-party or native
output written to the worker's stdout, not just the first byte. If a call exceeds
it, the worker subprocess is killed and a `TimeoutError` is raised. For a
**persistent** worker the timeout is per-call: a call that trips the timeout tears
down that worker, so the next call reloads the model in a fresh subprocess.
`timeout` is `reload_on_change=False` and `include_in_key=False`, so changing it
never restarts a worker on its own.

## Lifecycle and cleanup

- `tool.shutdown()` stops the worker and evicts the instance from its cache.
- `ToolInstance.shutdown_instance("key")` does the same, by cache key, with no handle
  needed.
- `ToolInstance.clear_all()` stops every cached worker and clears the cache.
- `persist()` / `persist_tool()` / `scope()` blocks shut down everything they
  created on exit.
- An `atexit` hook stops any workers still alive at interpreter shutdown, so a
  forgotten `get()` does not leak a subprocess.

## Authoring checklist

- Mark every config field that affects model initialization
  `reload_on_change=True`, and leave runtime-only fields `False`.
- Never check for config changes inside a standalone script. The
  `ToolInstance` layer restarts workers.
- Keep `preprocess` cheap, and remember it forces the preprocess/dispatch sharing
  scope.
- Outputs must stay JSON-serializable for the worker wire protocol (see
  `notes/finding-tools.md`).
- For batch APIs, prefer one shared worker (`persist()`) or a `ToolPool` over
  manual per-item `get()`/`shutdown()` churn.
