<a href="https://modal.com"><img src="../../guides/assets/modal/modal-logo.png" alt="Modal" width="260"></a>

# Modal Set Up

`proto-tools` can run models and tools in remote containers on
[Modal](https://modal.com), a third-party serverless compute platform. Modal charges
for usage and offers new users $30 in trial credit. See
[Modal pricing](https://modal.com/pricing) for details.

After creating an account and deploying a tool, set `device="modal"` in its config
to run it in a remote Modal container. Modal can scale to multiple GPUs on demand.

This page covers first-time account setup, deploying a tool, and the configuration options
available. For a runnable walkthrough that deploys a model and calls it, see the
[Cloud Inference guide](../../guides/cloud_inference.ipynb).

## Setup

### Step 1: Create a Modal account

Create an account on [modal.com](https://modal.com).

Install Modal and authenticate on your local machine by running these commands
and following the prompts.

```bash
pip install modal
modal setup
```

Once you have authenticated, a token will be written to `~/.modal.toml`. Proto reads it to
run and deploy tools on your account. Your local Modal key authenticates you directly with
Modal and is never transmitted to Proto or stored on any system we operate.

If you have multiple people who need access to the same tools, you can create a
shared workspace. This will enable each user to access the same deployed apps and
cached model weights, which can save money.

### Step 2: Create an environment

Create a Modal *environment* to hold your apps and their storage, including cached
model weights. Each app provides services that run deployed models.

The default environment used by proto-tools is `proto-env`, but you can use any name. We recommend using the same environment everywhere to avoid redundancy.

```bash
modal environment create proto-env
```

Note that when you create environments through your Modal dashboard, you have
a little more control over various settings such as the maximum number of concurrent
GPUs you would like the apps in the environment to be able to scale to. When we
run the command above, these settings default to your Modal workspace limit.

### Step 3: Deploy a tool

Deploying a tool creates an app in your Modal account that can run it remotely.
The app remains available until you remove the deployment in your Modal dashboard.

> [!NOTE]
> Deploying a tool and storing weights on Modal costs money. The weights a tool downloads persist on a Modal storage volume so you can reuse them without needing to redownload. This accrues a minimal storage cost until you remove them. Deploying itself is a one-time cost per tool as later calls reuse the app and the cached weights.

List the tools available to deploy.

```bash
proto-tools deploy --list
```

Deploy the tools you need, one at a time.

```bash
proto-tools deploy --apps esmc --env proto-env
```

Deployment time depends on the model. Third-party downloads sometimes fail, so
retry once if a deployment fails. If the problem persists, open an issue in the
[proto-tools](https://github.com/evo-design/proto-tools/issues) repository.

### Step 4: Run a tool

Set `device="modal"` on the deployed tool's config to run it there.

```python
from proto_tools import run_esmc_embeddings, ESMCEmbeddingsInput, ESMCEmbeddingsConfig

output = run_esmc_embeddings(
    ESMCEmbeddingsInput(sequences=["MKTAYLLIGLLAIAAFSPQVLA"]),
    ESMCEmbeddingsConfig(device="modal"),
)

print(len(output.results[0].mean_embedding))
```

The tool will execute on Modal and return the result.

You are now set up to use tools on Modal!

## Other notes

### Using an MCP agent

`proto-tools` also packages an MCP server that exposes the same deployed tools
to agents like Claude or ChatGPT, allowing you to deploy to and run tools on
Modal. See [MCP Server Set Up](../mcp/README.md) to install it and register it
with an agent, or our
[documentation website](https://proto.evodesign.org/docs/tools/introduction) for
the full tool reference.

### Scale Down Window

After a call finishes, a container stays alive briefly with the model still in memory so the
next call can skip the load. This helps repetitive calls run faster and avoid start-up
overhead, but costs more as the GPU idles for a longer period of time. You can set how long
this period lasts with the environment variable `PROTO_MODAL_SCALEDOWN_WINDOW`, which defines
how many seconds the container should wait after returning output before shutting down. It
defaults to `30`:

```bash
export PROTO_MODAL_SCALEDOWN_WINDOW=300     # keep containers warm for five minutes
```

Raise it for an interactive session, leave it low for occasional calls. Other configuration
options are covered in [`notes/modal-deployment.md`](../../notes/modal-deployment.md).

### Watching what you spend

We recommend only deploying the tools you need to use (not all of them). Cached
weights persist on a Modal volume and accrue storage cost until they are removed.

Remember to remove model weights from your storage if they are no longer in use!

### Further reading

- [`guides/cloud_inference.ipynb`](../../guides/cloud_inference.ipynb) — a runnable
  walkthrough, deploying a service and then folding a structure with it.
- [`notes/modal-deployment.md`](../../notes/modal-deployment.md) — the developer
  reference: the manifest, image construction, fingerprinting and drift, the transport
  envelope, live progress, capability guards, and standalone overrides.
