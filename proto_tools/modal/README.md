<a href="https://modal.com"><img src="../../guides/assets/modal/modal-logo.png" alt="Modal" width="260"></a>

# Modal Set Up

`proto-tools` enables users to scale their tool use beyond their local machine through an
integration with [Modal](https://modal.com). Modal is a third party serverless compute
platform that allows users to execute models and tools in remote containers. Note that
using Modal costs money! You can review their pricing on their website here: [https://modal.com/pricing](https://modal.com/pricing)

After setting up an account and deploying the tools you would like to have access to,
setting `device="modal"` in a tool config will dispatch the execution of your tool to a
remote Modal container, allowing you to scale up to a large number of GPUs on demand.

This page covers first-time account setup, deploying a tool, and the configuration options
available. For a runnable walkthrough that deploys a model and calls it, see the
[Cloud Inference guide](../../guides/cloud_inference.ipynb).

## Setup

### Step 1: Create a Modal account

To begin, you will need to set up an account on [modal.com](https://modal.com).

You will then need to authenticate your account on your local machine.

```bash
pip install modal
modal setup
```

This writes a token to `~/.modal.toml`.

To check your setup at any point, run `proto-tools doctor`.

If you have multiple people who need access to the same tools, you can create a
shared workspace. This will enable each user to access the same deployed apps and
cached model weights.

### Step 2: Create an environment

Next, you will need to create a Modal *environment*. An environment is a namespace
inside your Modal account that houses apps (that will run your deployed tools)
and their storage (such as cached model weights).

Here we will call our environment `proto-env`, but you can use any name:

```bash
modal environment create proto-env
```

Note that when you create environments through your Modal dashboard, you have
a little more control over various settings such as the maximum number of concurrent
GPUs you would like the apps in the environment to be able to scale to. When we
run the command above, these settings default to your Modal workspace limit.

### Step 3: Deploy a tool

The next step is to deploy a tool you would like to use through Modal. Deploying a tool means
setting up an app through which you will be able to run inference in the future. This step
installs the environment dependencies and then downloads weights so the model is ready to run.

> [!NOTE]
> Deploying a tool and storing weights on Modal costs money. The build ends by running the tool
> once to warm it, on a GPU for GPU-backed tools, so a deploy bills compute before it returns
> anything. The weights it downloads then persist on a Modal volume and accrue storage cost
> until you remove them. Deploying is a one-time cost per tool — later calls reuse the app and
> the cached weights.

To determine which tools are available to deploy, run the following command:

```bash
proto-tools deploy --list
```

You can then deploy tools as you need them, one by one:

```bash
proto-tools deploy --apps esmc --env proto-env
```

Depending on the model, a deployment can take minutes to complete. Note that some
deployments are a little flaky due to issues with third party download links failing.
We recommend just retrying them.

A deployment that finishes has built an image, which is not quite the same as the tool
working. To check that, run it once against what you just deployed with `--test`, where
`--skip-deploy` keeps the command from rebuilding:

```bash
proto-tools deploy --apps esmc --env proto-env --test --skip-deploy
```

### Step 4: Run a tool

Now that your tool is deployed, set `device="modal"` on its config to run it there:

```python
from proto_tools import run_esmc_embeddings, ESMCEmbeddingsInput, ESMCEmbeddingsConfig

output = run_esmc_embeddings(
    ESMCEmbeddingsInput(sequences=["MKTAYLLIGLLAIAAFSPQVLA"]),
    ESMCEmbeddingsConfig(device="modal"),
)

print(len(output.results[0].mean_embedding))
```
You are now set up to use the tool on Modal!

## Other notes

### Using an MCP agent

`proto-tools` also packages an MCP server that exposes the same deployed tools
to a coding agent, allowing you to deploy to and run tools on Modal. See our
[documentation website](https://proto.evodesign.org/docs/mcp/introduction) for
more details.

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
