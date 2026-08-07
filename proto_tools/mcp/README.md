# MCP Server Set Up

`proto-tools` packages an [MCP](https://modelcontextprotocol.io) server, which exposes its
tools to agents like Claude or ChatGPT. The agent can browse the catalogue, read a tool's
schema, and run it. You can ask for a structure prediction in conversation rather than writing
the call yourself.

This page covers installing the server and registering it with an agent.

Tools run either on this machine or on remote compute, and you choose which when you register
the server.

**Locally**, `proto-tools` builds each tool's environment on first use, so no account and no
setup is needed beyond the install. Tools use whatever compute this machine has, so a tool
that needs a GPU needs one here.

**Remote tools run on your Modal account.** Connect your Modal account to run tools on remote
compute, GPU models in particular. See
[Modal Set Up](../modal/README.md) for creating an account, creating an environment, and
deploying a tool.

## Setup

### Step 1: Install the server

The server depends on `fastmcp`, which is not part of the base install:

```bash
pip install "proto-tools[mcp] @ git+https://github.com/evo-design/proto-tools.git"
```

This installs a `proto-tools-mcp` command. Check it resolves:

```bash
proto-tools-mcp --help
```

### Step 2: Connect Modal, for remote compute

Skip this step to run tools on this machine only.

The server defaults to running tools on Modal, which needs a credential here:

```bash
modal setup
```

This opens a browser, and writes a token to `~/.modal.toml` when you approve. Nothing is
pasted, and the credential is not copied into any agent configuration.

If you already use Modal for something else, `modal token new` creates an additional token
without disturbing your existing configuration.

Run `proto-tools doctor` to check the credential, the environment, and what is deployed.

### Step 3: Register the server with your agent

The agent launches the server. You do not run it yourself.

#### Claude Code

```bash
claude mcp add proto-tools --scope user -- proto-tools-mcp
```

`--scope user` registers it for every session.

Tools run on Modal by default. Add `--device local` to run them on this machine instead. The
choice becomes the session default. An agent can override it per call with `run_on`.

#### Claude Desktop

Add `proto-tools-mcp` as the command for a server named `proto-tools` in
`claude_desktop_config.json`.

#### ChatGPT

`proto-tools` is not currently available on ChatGPT since the ChatGPT app only accepts remote
MCP servers reached over HTTPS. An HTTPS based version of the MCP is in development.

### Step 4: Deploy the tools you want, if using Modal

Running locally, there is nothing to deploy. Skip this step.

A fresh Modal workspace has nothing deployed, and the agent can only run what exists. Deploy
a tool before asking for it:

```bash
proto-tools deploy --apps tmalign
```

Each deploy builds an image in your workspace and takes a few minutes. See
[Modal Set Up](../modal/README.md) for the full deploy workflow.

Ask the agent to call `workspace_info` first. It reports which workspace and environment your
calls reach, and how many tools are deployed.

## Hosted HTTPS server

**Coming soon.** A hosted version of this server, reachable over HTTPS, so an agent can use
proto-tools without a local install. Tools will still run on your own Modal account.
