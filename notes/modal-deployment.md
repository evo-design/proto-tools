# Modal deployment and dispatch

How `proto_tools/modal/` turns a registered tool into a deployed Modal app, and how
`device="modal"` reaches one. User-facing setup lives in
[`proto_tools/modal/README.md`](../proto_tools/modal/README.md); this note is the
developer reference for the machinery behind it.

Two separable activities share the module, usually performed by different people at
different times:

- **Deploying** a tool into a Modal workspace, one app per service, built from this
  repository's own tool definitions.
- **Dispatching** a call to an already-deployed app, which is what `device="modal"` does.

Nothing about the second requires you to have done the first. A workspace someone else
deployed into serves calls the same way.

## Why deployments build from your checkout

The image for a tool is built from the proto-tools tree the module lives in. When
proto-tools is installed editable from a clone, that tree is *yours*, so a tool you add
under `proto_tools/tools/` is deployable without vendoring anything or pinning a
revision. Add the tool, register its service, deploy.

A non-editable install has no source tree, so the image is built from the installed
package instead. Its dependencies come from the distribution metadata and the package
itself is mounted. The catalogue that shipped with the release deploys; a tool you added
locally does not exist to deploy, which needs an editable install.

`PROTO_MODAL_PROTO_TOOLS` overrides the resolution when you want to build from a
different checkout than the one you are running.

## The manifest

`manifest.py` is the single declaration of what exists. Everything else derives from it,
and a startup check fails loudly when the pieces disagree.

| Table | Holds |
|---|---|
| `APP_BUCKETS` | app name to the service classes it contains, the grouping that decides what shares an image |
| `SERVICE_TO_MODULE` | service class to the Python module defining it |
| `SERVICE_MODAL_TIMEOUTS` | per-service wall clock, in seconds |
| `GPU_SERVICES` | which services need a GPU, which `physical_device_for_service` turns into the device string a container reports |

`SERVICE_TO_APP` is inverted from `APP_BUCKETS` rather than written twice.

Grouping several services into one app is how tools sharing a model and environment
share an image and a warm container. Splitting them would mean paying for the same
weights twice.

## App entrypoints are rendered, not committed

`modal deploy` needs a module path to import. That module used to be 35 generated files
checked into the tree, with a generator and a consistency test to keep them honest.

They are now rendered into a temporary directory at deploy time and discarded. The
staleness class is removed structurally rather than tested for, and nothing is written
into a user's tree. `render_entrypoint` in `deploy.py` builds the module text from
`APP_BUCKETS`; `import_all_services()` in `registry.py` replaced the aggregate import the
deleted package used to provide.

## Images

`base_images.py` builds every image from a shared base plus the tool's own requirements.

One rule is load-bearing and was learned the hard way. Modal auto-mounts the package
containing a service class, and that mount lands at `/root`, which **precedes** `/pkg` on
`sys.path`. An image that also mounted proto-tools at `/pkg` therefore had two copies,
and imports resolved to the one nothing had been written to. Standalone overrides applied
to `/pkg` were inert. Images now ship exactly one copy, at the path imports resolve to:

```python
image = image.add_local_python_source("proto_tools", copy=True, ignore=[])
```

`ignore=[]` is deliberate. `_from_local_python_packages` defaults to including only some
files, and a standalone directory is not all Python.

## Standalone overrides

An escape hatch for when a tool's environment builds on bare metal but not inside a
container: a different wheel index, a pinned system library, an extra variable.

Each tool ships a `standalone/` directory describing how to build its isolated
environment. To replace part of it for Modal only, put the modified file beside the
service that deploys it:

```
proto_tools/modal/causal_models/evo1_deployment/
├── evo1_service.py
└── standalone_overrides/
    └── setup.sh            # only the files you are replacing
```

The overlay is partial. Files present here are copied in, everything else comes from the
tool untouched, and it is applied by the image builder:

```python
image = with_proto_tools(GPU_BASE, overrides="evo1", overrides_dir=Path(__file__).parent)
```

With no `standalone_overrides/` directory the call returns the image unchanged, so it is
safe to leave in place.

Record why each override exists at the top of the file: which upstream file it replaces,
what the deliberate difference is, and when it was last checked against upstream.

Changing an override forces a full environment rebuild, because proto-tools fingerprints
the standalone directory and any edit invalidates the cached environment. For a tool
whose environment takes half an hour to resolve, that is the whole build again.

## Fingerprinting and drift

A deployment can silently fall behind the code that describes it. `fingerprint.py`
records three hashes per tool at deploy time, written to `/_fingerprints` on the cache
volume, and the client compares them on dispatch.

| Hash | Covers | Catches |
|---|---|---|
| `schema_hash` | the tool's input, config, and output JSON schemas | a field added, removed, or retyped |
| `code_hash` | the tool's defining files plus its first-party MRO and the framework prefixes | a behaviour change with an unchanged schema |
| `env_hash` | a recursive walk of the standalone directory | a dependency or setup change |

`ALGORITHM` is a version guard. A deployment recorded under a different algorithm reports
that it cannot be compared rather than guessing, which is the correct answer and not a
failure.

`code_hash` autodetects a tool's defining files from `spec.source_file` and its
first-party MRO, so a tool that inherits behaviour from a shared base is covered without
declaring anything. The blind spot found in review was `standalone_helpers/`, which
`env_hash` now walks; for ESM2 that is 61 files that previously could change with no
effect on any hash.

Drift produces a warning rather than an error. A stale deployment still runs, which is
usually what a caller wants, and the warning names which of the three moved.

## Dispatch

`client.py` holds both forms. `dispatch_to_modal` runs one call through `.remote()`;
`dispatch_batch_to_modal` fans a batch out through `starmap`, which queues chunks and
reuses warm containers, with `return_exceptions=True` so a failed chunk does not discard
chunks that already succeeded and were already billed, and `order_outputs=True` so each
entry stays aligned with the input that produced it.

`_resolve_device` rewrites a logical device (`"proto"`, `"modal"`, or the `"cpu"` default)
into the physical device of the container that will actually run it. The caller does not
know, and should not need to know, what hardware the deployment sits on.

### The transport envelope

A config crossing a process boundary carries framework state that is not a field.
`to_transport_dict()` adds it under `_proto_internal`; `BaseConfig`'s wrap validator
strips the key before validation so `extra="forbid"` never sees it, restores what it
recognises, and **ignores what it does not**.

That last property is what makes rollout safe in both directions. A worker predating a
new key reads an unchanged envelope; a worker that understands a key streams nothing when
it is absent. It is also why live progress needed no change to any service's method
signature.

`cache_key()` dumps through `model_dump`, not `to_transport_dict`, so nothing in the
envelope can split the cache between caller and worker.

Serialization is explicit rather than a `model_serializer` override, which means a
dispatch site that forgets `to_transport_dict()` sends no state and the worker
preprocesses as it otherwise would. `tests/modal_tests/test_config_transport.py` greps
each dispatch function's source to catch that.

### Live progress

A Modal function call is an RPC, so a container's output reaches `modal app logs` and
never the caller. Progress travels beside the call on a `modal.Queue`: one persistent
queue per environment, partitioned per call, with the partition carried in the transport
envelope described above.

The performance contract is enforced by construction, not by measurement:

- `emit` runs on the tool's own thread and does one bounded-deque append of the
  **unformatted** record. Deferring `%`-formatting is the whole trick, worth +16.8%
  against +0.2% when it was measured.
- A daemon formats and batch-writes on a throttled interval, holding the container to
  roughly one network write per interval regardless of log volume.
- The queue is resolved on the drainer thread, so the tool thread does no network work
  even at setup.
- The buffer drops its oldest entry rather than applying backpressure. A lost progress
  line costs nothing; a slow `emit` would cost the run.
- The first queue error disables streaming for the rest of the run.

Streaming is off unless someone is watching, meaning an active spinner or a verbose
caller. The client's tailer exits on the end sentinels **or** on a stop event set once the
result is in hand, whichever comes first, which is what stops a deployment too old to emit
anything from leaving a thread polling.

Records replay through `proto_tools.modal.remote` carrying the `update_status` flag the
spinner keys on, so a remote line renders exactly as the same line does locally. This
mirrors the `device="proto"` path rather than introducing a second convention.

One gap worth knowing: progress opens per call, inside `dispatch_tool_call`. A container
still waking up has not reached that point, so a cold start streams nothing for its first
several seconds. The client covers the interval by naming the backend on the spinner.

### Capability guards

A config declares `remote_unsupported_reason(device)` when a setting cannot work on a
remote worker. The registry calls it once before dispatch, for whichever remote device
was chosen.

The hook takes the device because remote targets differ in what they can run. A local
checkpoint path exists on no remote machine, so both devices refuse it. A custom
checkpoint is a different matter: loading one executes the pickle inside it, which Proto's
shared service will not do with a caller's file, while a Modal workspace belongs to the
caller and has no such objection.

The name matters more than it looks. Three configs still spelled it
`cloud_unsupported_reason` after `device="cloud"` became `device="proto"`, so the registry
never called them, nothing failed, and the restrictions silently did not apply.
`tests/style_consistency_tests/test_remote_capability_hooks.py` now fails on any config
defining a differently named hook.

**The guard is client-side.** It runs in the caller's process, so a modified client or a
request crafted directly against the API bypasses it entirely. Server-side enforcement
belongs at Proto's submission path and is tracked in proto-tools-api#567.

## Hosted environments

`PROTO_IS_HOSTED_ENV` marks a process running tools for someone else rather than on a
user's own machine. `dispatch_tool_call` sets it per call rather than at import, because
the value has to be visible wherever the call actually executes.

A config adjusts itself for that case by overriding `BaseConfig.for_hosted_env`, which
`run_preprocess` calls when the flag is set. The live consumer is
`MSAStructurePredictionConfig`, which rewrites a local MSA search to the remote API
because a hosted container cannot stage `uniref30-2302`. The substitution changes results
and therefore logs a warning rather than passing silently.

## Configuration

Four environment variables, all optional, read in the environment a deploy runs from. Only
`PROTO_MODAL_SCALEDOWN_WINDOW` appears in the user-facing README; the rest are here.

| Variable | Default | Effect |
|---|---|---|
| `PROTO_MODAL_HF_SECRET` | unset | Name of a Modal secret holding `HF_TOKEN`. Set to `none` to force anonymous downloads. |
| `PROTO_MODAL_CACHE_VOLUME` | `proto-cache` | Volume holding model weights, shared by every service. |
| `PROTO_MODAL_SCALEDOWN_WINDOW` | `30` | Seconds an idle container stays alive holding its model. |
| `PROTO_MODAL_PROTO_TOOLS` | unset | Build from this proto-tools checkout instead of the installed one. |

### Gated model weights

Gated models need a HuggingFace token, and anonymous downloads are rate-limited.

With `PROTO_MODAL_HF_SECRET` unset, deploying uses the token the deploying machine already
has, resolved by `resolve_hf_token()` from `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, or the file
written by `hf auth login`. Accepting a model's licence once is therefore enough. The token is
wrapped in an anonymous `modal.Secret.from_dict({"HF_TOKEN": token})` bound to the caller's own
workspace, its use is recorded at debug level rather than printed, and a read-scoped token
suffices.

To manage the secret explicitly instead:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_...
export PROTO_MODAL_HF_SECRET=huggingface-secret
```

`app.py` resolves this once at import. Inside a container the token has already arrived
through the attached secret, so re-reading the environment there would only rewrap it; the
local-machine branch is guarded by `modal.is_local()`.

Missing this is not loud. A build with no token still succeeds for ungated weights and fails
only where a licence is required, which is how 34 of 35 warmups once ran without one.

## Spending

Every deploy costs money before you run anything, because each build ends in a real
warmup inference, on a GPU for GPU tools. Deploy the apps you need rather than all of
them, and set concurrency limits on your Modal environment so an unattended batch cannot
scale to your plan's ceiling.

Cached weights persist on a Modal volume and accrue storage cost until removed.

`SCALEDOWN_WINDOW` (30s, `PROTO_MODAL_SCALEDOWN_WINDOW`) is the direct trade. A longer
window keeps a container warm and skips cold starts on the next call; it also bills for
an idle GPU for that long. Raise it for an interactive session, leave it low for
occasional calls.

Container `timeout` in Modal is **per execution attempt**, so `modal.Retries(max_retries=3)`
multiplies the worst-case wall clock by four. Startup shares that budget.

## Testing

`tests/modal_tests/` runs entirely offline. Fakes stand in for the Modal SDK, which means
the suite pins the contracts the client depends on rather than Modal's behaviour.

That boundary has a cost worth stating: a bug in how a Modal object comes into being is
invisible to it. `modal.Queue.from_name` is lazy, so `create_if_missing=True` never
created the progress queue, every worker's lookup failed, and the feature produced nothing
while results still came back correct. The unit tests substituted a fake for the very
object whose lifecycle was broken. A real deploy found it in one run.

Deploy against a development environment before trusting anything here.
