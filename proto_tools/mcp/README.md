# MCP Server Set Up

`proto-tools` packages an [MCP](https://modelcontextprotocol.io) server, which exposes its
tools to an agent such as Claude Code or Claude Desktop. The agent can browse the catalogue,
read a tool's schema, and run it — so you can ask for a structure prediction in conversation
rather than writing the call yourself.

This page covers installing the server and registering it with an agent.

**Tools run on your own Modal account.** The server dispatches work to remote containers you
deploy and pay for, so set that up first: see [Modal Set Up](../modal/README.md) for creating
an account, creating an environment, and deploying a tool. Without it the server starts and
every tool call fails.

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

The install is large — the dependency tree includes RDKit, biotite and the scientific Python
stack — so expect it to take a few minutes.

### Step 2: Authenticate with Modal

The server defaults to running tools on Modal, which needs a credential on this machine:

```bash
modal setup
```

This opens a browser, and writes a token to `~/.modal.toml` when you approve. Nothing is
pasted, and the credential is not copied into any agent configuration.

If you already use Modal for something else, `modal token new` creates an additional token
without disturbing your existing configuration.

Run `proto-tools doctor` to check the credential, the environment, and what is deployed.

### Step 3: Register the server with your agent

The server speaks MCP over standard input and output, so the agent launches it — you do not
run it yourself.

**Claude Code:**

```bash
claude mcp add proto-tools --scope user -- proto-tools-mcp
```

`--scope user` registers it for every session. To remove it later:

```bash
claude mcp remove proto-tools --scope user
```

**Claude Desktop, Cursor, and other clients configured through JSON:**

```json
{
  "mcpServers": {
    "proto-tools": {
      "command": "proto-tools-mcp"
    }
  }
}
```

For Claude Desktop this goes in `claude_desktop_config.json`.

Prefer `proto-tools-mcp` over `python -m proto_tools.mcp`. The console script carries the
interpreter it was installed into, so a client that does not inherit your shell environment —
Claude Desktop is a desktop application, not a terminal — still starts the right Python.

Note that no credential appears in either configuration. The server reads `~/.modal.toml`
itself, so nothing sensitive is written into an agent's config file.

### Step 4: Deploy the tools you want

A fresh Modal workspace has nothing deployed, and the agent can only run what exists. Deploy
a tool before asking for it:

```bash
proto-tools deploy --apps tmalign
```

Each deploy builds an image in your workspace and takes a few minutes. See
[Modal Set Up](../modal/README.md) for the full deploy workflow.

Ask the agent to call `workspace_info` first — it reports which workspace and environment
your calls reach, and how many tools are deployed.

## Other notes

### Choosing where tools run

The server resolves one backend at startup, and defaults to `modal`. Pass `--device` to
change it:

```bash
claude mcp add proto-tools --scope user -- proto-tools-mcp --device local
```

`local` runs tools in the server process, which suits tools that need no GPU and no
standalone environment. `modal` dispatches to your workspace.

### Deploying from the agent

The server exposes `deploy_tool`, so an agent can deploy on your behalf. This spends money on
your Modal account and takes minutes per tool, so say what you want deployed rather than
leaving it to the agent's judgement.

### Do not run the server by hand

A stdio server waits on standard input and prints nothing, so running `proto-tools-mcp`
directly looks like it has hung. Use `--help` to confirm the entry point resolves.

## Hosted HTTPS server

**Coming soon.** A hosted version of this server, reachable over HTTPS, so an agent can use
proto-tools without a local install. Tools will still run on your own Modal account.

## Further reading

- [Modal Set Up](../modal/README.md) — accounts, environments, deploying tools, and what it
  costs.
- [Documentation website](https://proto.evodesign.org/docs/mcp/introduction) — the MCP tool
  reference, including every function the server exposes.
