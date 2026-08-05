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

## Worker extension points

A service method ends in `run_tool_call(run_fn, InputModel, ConfigModel, input_dict, config_dict)`,
which validates the mappings and hands off to `dispatch_tool_call`. A guard test asserts every
`@modal.method` goes through it — a method that dispatches some other way silently ignores hooks
rather than erroring.

`proto_tools/modal/hooks.py` offers two:

| | Runs | Sees |
|---|---|---|
| `register_payload_hook` | before validation | the raw `input_dict` and `config_dict`, mutable |
| `register_call_middleware` | around the call | a `CallContext` and the next step; may transform the result |

A payload hook is the only place a value can still be rewritten. Middleware follows the ASGI
shape; first registered is outermost, and `CallContext.tool_key` says which tool it wrapped.

Both are process-wide. Register during import, from whatever module defines the deployment's
entry point — registration is not synchronized. Nothing registers by default.

Two limits worth knowing: payload hooks run before the progress context opens, so a slow one is
silent on the caller's spinner; and a middleware that forgets to return raises `TypeError` rather
than returning `None` to the client.

### Getting extension code into a worker

A deployed container imports only what a service module reaches, and carries only what its image
was built with. Three variables, read where the deploy runs, cover both:

| Variable | Effect |
|---|---|
| `PROTO_MODAL_EXTRA_PACKAGES` | Requirements every service image installs. Whitespace-separated, since a specifier may contain a comma — which also means an environment marker cannot be expressed here. |
| `PROTO_MODAL_EXTRA_SOURCE` | Directories every service image carries, separated by `os.pathsep`. Each is mounted under `/root` by its own name and imports as that name, so the name must be a Python identifier. |
| `PROTO_MODAL_WORKER_PLUGINS` | Modules a worker imports before serving its first call. Comma- or whitespace-separated. |

```bash
PROTO_MODAL_EXTRA_PACKAGES="alpha>=1 beta" \
PROTO_MODAL_EXTRA_SOURCE=/path/to/extras \
PROTO_MODAL_WORKER_PLUGINS=extras.hooks \
proto-tools deploy --apps tmalign --env proto-env
```

Packages and source are added by `with_proto_tools`, below proto-tools' own layers, so editing
proto-tools does not rebuild them. The plugin list travels in `RUNTIME_ENV` instead, applied
*above* the warmup: it is runtime metadata, and renaming a module would otherwise rebuild and
re-warm all 54 images. Guard tests assert every service applies both, since the two that once
omitted `RUNTIME_ENV` loaded no plugins at all while deploying and smoke-testing green.

Plugins are imported once per process, on the first call a worker serves — not at `@modal.enter`.
So a middleware cannot observe what a service does in its enter hook, and a cold container charges
the import to its first request. The deploy-time warmup calls run functions directly rather than
through `run_tool_call`, so it never loads them.

Both failure modes are surfaced rather than deferred, because a worker serving calls with a
deployment's extensions silently absent is worse than a loud failure:

- a module that cannot be imported raises an `ImportError` naming `PROTO_MODAL_WORKER_PLUGINS`;
- a directory that does not exist, or whose name nothing could import, fails the deploy up front
  rather than inside a subprocess whose output is filtered to phase lines.

Extension code that reaches a credentialed service needs credentials in the container.
`PROTO_MODAL_SECRETS` names further Modal secrets, attached to the app rather than to each service
class — Modal gives a function its app's secrets in addition to its own, so a service declaring
secrets of its own keeps them. The HuggingFace secret is the default member of that list and needs
no entry; naming others adds to it rather than replacing it, so gated weight downloads keep
working. A name that does not exist in the workspace fails the deploy.

A mounted directory excludes `.git` and the usual build artefacts. It does *not* exclude `tests`,
which in someone else's tree may be a package their code imports. Note that `/root` precedes
site-packages, so a directory sharing a name with an installed package shadows it.

## Hosted environments

`PROTO_IS_HOSTED_ENV` marks a process running tools for someone else rather than on a
user's own machine. `dispatch_tool_call` sets it per call rather than at import, because
the value has to be visible wherever the call actually executes.

A config adjusts itself for that case by overriding `BaseConfig.for_hosted_env`, which
`run_preprocess` calls when the flag is set. The live consumer is
`MSAStructurePredictionConfig`, which rewrites a local MSA search to the remote API
because a hosted container cannot stage `uniref30-2302`. The substitution changes results
and therefore logs a warning rather than passing silently.

## Credentials

Modal accepts a credential from three sources, and `proto_tools/utils/modal_status.py`
resolves them the same way the SDK does.

| Source | Set by | Suits |
|---|---|---|
| `~/.modal.toml` | `modal setup` | A machine you work on directly. |
| `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` | The environment | A container, CI job, or agent sandbox. |
| `MODAL_CONFIG_PATH` | The environment | A host that mounts a token file rather than injecting variables. |

Environment variables take precedence over the file. Precedence is by *membership*, not
truthiness — `MODAL_TOKEN_ID=""` still shadows a readable config file and then fails to
authenticate, which is why `credentials_checked` reports `set`/`empty`/`unset` rather than a
boolean.

The distinction matters most where proto-tools does not run on a laptop. A token file written
outside the process is frequently unreadable from inside it: the file may be absent, or present
in a directory the uid cannot traverse. Those two states are indistinguishable to
`Path.exists()`, so `config_state()` separates `absent` from `unreadable` — only the first is
fixed by writing a token. For the same reason `config_path()` uses `os.path.expanduser` rather
than `Path.home()`, which raises when `HOME` is unset and the uid has no passwd entry.

`proto_tools/utils/modal_status.py` deliberately sits outside `proto_tools/modal/`. That
package builds Modal objects at import time, which is precisely what fails when Modal is
unconfigured, so a credential check living there could not run in the state it exists to
diagnose.

### `proto-tools doctor`

One command that reports whether this environment can reach Modal, and exits non-zero naming a
remedy when it cannot. It verifies the credential with `Client.hello()` rather than stopping at
`Client.from_env()`, which only checks that a token is *present* — a revoked or wrong-workspace
token passes that check and fails every real call. Four outcomes are reported apart because
each needs a different fix:

| Outcome | Meaning |
|---|---|
| `OK via <source>` | Verified against the server, naming which of the three sources was used. |
| `not found` | No credential in any source, listing what was checked. |
| `rejected` | A credential was found and the server refused it — revoked, or another workspace. |
| `unverified` | A credential was found but Modal was unreachable; a network fault, not a bad token. |

`workspace_info` in the MCP still stops at `from_env()`, so it reports `authenticated: true`
for a token the server would refuse. Verifying there would add a network roundtrip to an
agent's first orienting call, so the two surfaces differ on purpose.

## Configuration

Nine environment variables, all optional, read in the environment a deploy runs from.

| Variable | Default | Effect |
|---|---|---|
| `PROTO_MODAL_HF_SECRET` | unset | Name of a Modal secret holding `HF_TOKEN`. Set to `none` to force anonymous downloads. |
| `PROTO_MODAL_CACHE_VOLUME` | `proto-cache` | Volume holding model weights, shared by every service. |
| `PROTO_MODAL_SCALEDOWN_WINDOW` | `30` | Seconds an idle container stays alive holding its model. |
| `PROTO_MODAL_TIMEOUT_SCALE` | `1` | Multiplies every container wall tier. Values below 1 are ignored. |
| `PROTO_MODAL_PROTO_TOOLS` | unset | Build from this proto-tools checkout instead of the installed one. |
| `PROTO_MODAL_EXTRA_PACKAGES` | unset | Requirements every service image installs. |
| `PROTO_MODAL_EXTRA_SOURCE` | unset | Directories every service image carries, importable by their own names. |
| `PROTO_MODAL_WORKER_PLUGINS` | unset | Modules a worker imports before serving its first call. Also re-read inside the container, which is where the import happens. |
| `PROTO_MODAL_SECRETS` | unset | Names of further Modal secrets every service receives, alongside the HuggingFace one. |

### Container wall tiers

Each service picks a wall from `TIER_SECONDS` in `proto_tools/modal/manifest.py` rather than
carrying its own number, so the fleet runs on five understood budgets: `fast` (10 min),
`medium` (30 min), `long` (1 hour), `extended` (4 hours), `batch` (24 hours).

The tiers are deliberately generous. A wall covers the slowest input a tool accepts, under the
slowest config it accepts, on a cold container, and per-item cost varies by orders of magnitude
with sequence length. Modal also restarts the wall on every retry, so a wedged call can bill up
to four times the tier before it is killed.

`PROTO_MODAL_TIMEOUT_SCALE` lengthens every tier for a workload whose inputs exceed what the
shipped budgets assume, and is baked in at deploy time:

```bash
PROTO_MODAL_TIMEOUT_SCALE=2 proto-tools deploy --apps esmfold --env proto-env
```

It cannot shorten a wall. A value below 1 is ignored with a warning, since shortening a wall
kills work that used to complete, and the deploy-time knob would otherwise silently defeat the
floor that `tests/modal_tests/test_entrypoints.py` enforces on the tier table itself.

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
