# Device Management Reference

This note covers how `proto_tools` places model workers on GPUs, evicts them
under memory pressure, and moves them between GPU and CPU through `DeviceManager`.
The user-facing walkthrough is the Device Management guide, while this is the
internals reference.

Source of truth: `proto_tools/utils/device_manager.py`,
`proto_tools/utils/device.py`, and the `to_device()` protocol in
`notes/tool-environments.md`.

## Mental model

Each tool's config carries a `device` field. `DeviceManager` is a process-wide
singleton (`DeviceManager.get_instance()`) that tracks every GPU allocation,
places new workers on free managed GPUs, and evicts the least-recently-used
worker when the pool is full. The defaults need no configuration, namely one worker per
GPU, LRU eviction, and a full restart of an evicted worker on reuse.

## Device strings

`device` resolves through `parse_device_string` (`device.py`) into a
`DeviceSpec`. The accepted forms split into two intents:

- **General**. Name a class and let DeviceManager choose the physical device,
  such as `"cpu"`, `"cuda"` (one GPU), or `"cudax2"` / `"cudax3"` / … (N GPUs). This form is
  preferred because it needs no knowledge of which GPUs are free and triggers eviction
  automatically when they are not.
- **Specific**. Name exact devices, such as `"cuda:0"`, `"cuda:0,1"` or
  `"cuda:0,cuda:1"`. DeviceManager honors the request and evicts whatever
  occupies the slot if it is taken.

Device indices are **logical**. They refer to the GPUs visible after
`CUDA_VISIBLE_DEVICES` filtering, not physical bus IDs.

<img src="../guides/assets/device-management/general-specific.svg" alt="A general request lands on any free managed GPU, while a specific request pins to the named device." width="560">

## The allocation map

DeviceManager holds an allocation per live worker in `_allocations` (keyed by
instance name). Each `DeviceAllocation` records `toolkit`, `device_ids`,
`allocated_at`, `last_used`, an `eviction_callback`, and an `allocation_type`:

- `AllocationType.PERSISTENT`. A long-lived worker. Evictable by LRU.
- `AllocationType.TRANSIENT`. A one-shot lease (see [Leases](#device-leases-for-one-shot-calls)). Non-evictable, auto-released.

`get_device_status()` returns the full picture. It reports `available_devices`,
`allocations` (name → device, ages, type), `offload_strategy`,
`allow_multiple_per_device`, and per-GPU `gpu_memory` (used/free/total GB), and it is
the canonical way to inspect placement.

```python
DeviceManager.get_instance().get_device_status()["allocations"]
# {"A": {"device_id": "cuda:0", "last_used": ..., "allocation_type": "persistent"}, ...}
```

## LRU eviction

The managed pool is finite. When every managed GPU is occupied and a new worker
needs to load, DeviceManager sorts the **PERSISTENT** allocations by `last_used`
and evicts the oldest, regardless of which GPU it sits on. Every `run_*` call
refreshes its worker's `last_used`, so "least recently used" tracks actual call
activity, not load order. TRANSIENT leases are never evicted.

<img src="../guides/assets/device-management/lru.svg" alt="With two GPUs and three instances, the least recently used instance is evicted to make room for the third." width="560">

## Eviction strategies (`offload_strategy`)

What eviction *does* to the displaced worker is set by `offload_strategy` (or
`BIO_TOOLS_OFFLOAD_STRATEGY`):

- **RESTART** (default). `_evict_allocation` terminates the worker's
  subprocess, so all its GPU memory is freed and the next call to that tool pays the
  full cold-start cost (subprocess spawn + model load).
- **CPU**. The worker's weights are moved into system RAM via the tool's
  `to_device()` and the process is kept alive. Promoting it back to GPU is a
  tensor copy, far cheaper than a reload (for ESMFold, ~9 GB, measured ~8 s back
  from CPU versus ~17 s cold).

<img src="../guides/assets/device-management/restart.svg" alt="RESTART eviction fully shuts down the displaced worker so its GPU memory is freed." width="560">

<img src="../guides/assets/device-management/cpu-offload.svg" alt="CPU offload moves the evicted model to system RAM and brings it back quickly when it is reactivated." width="560">

Not every tool can stay resident on a GPU or be offloaded to CPU, and for tools that
cannot, DeviceManager falls back to RESTART regardless of the configured
strategy. The CPU strategy is also auto-escalated to RESTART when any GPU is in
Exclusive_Process compute mode, because an evicted subprocess retains its CUDA
context and would block other processes even after offloading to CPU. Physical
movement of loaded weights is the **`to_device()` protocol**
documented in `notes/tool-environments.md`, and the move is bounded by
`DEVICE_MOVE_TIMEOUT` (200 s).

## Managed devices

By default the pool is every visible GPU. Narrow it with
`BIO_TOOLS_MANAGED_DEVICES` (set before any tool runs) or
`configure(managed_devices=[...])` at runtime. General requests then land only
on managed GPUs. Unmanaged cards are never touched by DeviceManager and stay
free for other work on the machine.

<img src="../guides/assets/device-management/managed.svg" alt="A managed pool restricts general requests to the listed GPUs and leaves the rest free for other work." width="560">

## Multiple workers per device

The default is one worker per GPU. With `allow_multiple_per_device=True` (or
`BIO_TOOLS_ALLOW_MULTI_DEVICE=true`), new allocations prefer free GPUs and then
pack onto the least-loaded cards instead of triggering eviction, so several
workers can share a card.

<img src="../guides/assets/device-management/packing.svg" alt="Four instances packed across two GPUs, two per card." width="560">

DeviceManager does not track model sizes or estimate memory, so ensuring the packed
models fit is the caller's responsibility, and overcommitting raises an
out-of-memory error.

## Device leases for one-shot calls

One-shot dispatch (`ToolInstance._oneshot`, the non-persistent default) does not
register a long-lived worker. Instead it takes a **TRANSIENT lease**
(`DeviceManager.lease(...)`) for the duration of the call so that concurrent
one-shot calls do not stamp on the same GPU. Leases are non-evictable and
auto-released when the call returns, so they never participate in LRU.

## Explicit requests are always honored

A specific `device=` request always wins. If the named GPU is occupied,
DeviceManager evicts to make room (and logs that it did) rather than redirecting
the request elsewhere. This is what makes specific requests suitable for
benchmarking and reproducing a fixed placement.

## Configuration reference

| Variable | Programmatic | Values |
|---|---|---|
| `BIO_TOOLS_MANAGED_DEVICES` | `managed_devices=[...]` | logical device IDs (`"cuda:0,cuda:1"`) |
| `BIO_TOOLS_OFFLOAD_STRATEGY` | `offload_strategy=OffloadStrategy.{RESTART,CPU}` | `"restart"` / `"cpu"` |
| `BIO_TOOLS_ALLOW_MULTI_DEVICE` | `allow_multiple_per_device=` | `"true"` / `"false"` |

```python
from proto_tools.utils.device_manager import DeviceManager, OffloadStrategy

DeviceManager.get_instance().configure(
    managed_devices=["cuda:0", "cuda:1"],
    offload_strategy=OffloadStrategy.CPU,
    allow_multiple_per_device=False,
)
```

Environment variables are read once at startup, while `configure(...)` applies at
runtime. `DeviceManager.reset_instance()` clears all state (used between
independent examples and in tests).

## Authoring tools for device management

- Implement `to_device(device)` on the model wrapper in `standalone/inference.py`
  (or `run.py`) so the tool can be CPU-offloaded. See `notes/tool-environments.md`.
  Without it, the tool falls back to RESTART.
- Mark a tool `gpu_only=True` when it cannot run on CPU. LRU eviction then
  restarts it instead of offloading, and direct `device="cpu"` is rejected.
- Set `device_count` on the `@tool` registration so allocation requests are
  validated (under-allocation errors, over-allocation warns).

For how persistence and pooling drive these allocations, see
`notes/tool-persistence.md`.
