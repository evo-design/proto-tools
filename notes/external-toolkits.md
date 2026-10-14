# Out-of-tree Toolkits

A package outside proto-tools can define its own toolkits: its own `@tool` functions, models, docs files, and `standalone/` environment, with the source living in that package's repository. Importing the package registers its tools, and they then run locally through the same environment system as packaged tools, whether proto-tools is installed editable or as a regular wheel.

## Layout

Mirror the packaged layout under your package:

```
mypkg/
└── tools/
    └── {category}/
        └── {toolkit}/
            ├── __init__.py
            ├── {tool_key_snake}.py
            ├── README.md
            ├── cite.bib
            ├── license.yaml
            ├── links.yaml
            ├── examples/
            │   └── example.ipynb
            └── standalone/
                ├── setup.sh
                ├── python_version.txt
                ├── requirements.txt
                └── inference.py
```

Metadata (`cite.bib`, `license.yaml`, `links.yaml`, the example notebook, and README sections) is read from the directory of the file that defines each tool, so it works wherever the package is installed. The `"{category}/{toolkit}"` identifier form matches the two directory names, so keep `@tool(category=...)` equal to the category directory.

## Writing a tool

A tool follows the same `Input -> Config -> run_*() -> Output` pattern as a packaged one (see `CONTRIBUTING.md` and the `implement-tool` skill). The run function dispatches to its own toolkit directory name:

```python
from typing import Any

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ToolInstance


@tool(
    key="mytool-score",
    label="MyTool Score",
    category="scoring",
    input_class=MyToolScoreInput,
    config_class=MyToolScoreConfig,
    output_class=MyToolScoreOutput,
    description="Score sequences with MyTool",
    uses_gpu=True,
)
def run_mytool_score(inputs: MyToolScoreInput, config: MyToolScoreConfig, instance: Any = None) -> MyToolScoreOutput:
    result = ToolInstance.dispatch("mytool", inputs.model_dump(), instance=instance, config=config)
    return MyToolScoreOutput(**result)
```

Give each tool its own config class, since registration records the tool key on it.

## Using it

Import the module that defines the tool (or a package `__init__` that imports it). From then on the tool behaves like any other: `ToolRegistry.get("mytool-score")`, calling `run_mytool_score`, and `ToolInstance.persist_tool("mytool-score")` all work. Registration happens at import, so the `proto-tools` CLI and MCP server, which only import proto-tools, list packaged tools only.

## Naming

Workers, environments, and `PROTO_<TOOLKIT>_STANDALONE_DIR` are keyed by the toolkit directory name, a namespace shared with proto-tools. An out-of-tree toolkit with a `standalone/` directory therefore cannot use the name of a packaged toolkit (e.g. `esm2`) or of another registered out-of-tree toolkit; importing it raises a `ValueError` naming both directories. Tool keys must be unique across all registered tools, as for packaged tools.

## Environments

The environment is built from your `standalone/` exactly as for a packaged toolkit (see `tool-environments.md`): `setup.sh` runs from that directory, the setup hash covers its env files, and the env lives under `PROTO_HOME/proto_tool_envs/`. Its name is `{toolkit}__ext_{hash}_env`, where the hash is of the package path defining the toolkit (e.g. `mypkg.tools.scoring.mytool`). Every install of your package (any venv, editable or not) shares one env, and two packages with a same-named toolkit never share an env.

- `standalone/shared_env.txt` may name a packaged shared env (e.g. `biohub_esm`); the toolkit then uses that env under its packaged name.
- `PROTO_<TOOLKIT>_STANDALONE_DIR` overrides your env definition the same way it does a packaged one.

## Local only

An out-of-tree toolkit with a `standalone/` environment runs only on the local machine: remote workers ship proto-tools' own toolkits, not yours. It defaults to a `local_only` reason, so `device="modal"` or `device="proto"` raises a `ValueError` before anything is sent. Pass `local_only=` to `@tool` to state your own reason. A tool without a `standalone/` directory runs in-process and follows the usual device routing.

## Packaging

Non-Python files must ship in your wheel, or a non-editable install loses `setup.sh`, `requirements.txt`, `python_version.txt`, `cite.bib`, `README.md`, and the rest. Replicate proto-tools' package-data globs in your `pyproject.toml`, and keep namespace discovery on (the default for `[tool.setuptools.packages.find]`) so `standalone/` ships without an `__init__.py`:

```toml
[tool.setuptools.packages.find]
include = ["mypkg*"]

[tool.setuptools.package-data]
"*" = ["*.sh", "*.txt", "*.csv", "*.bib", "*.yaml", "*.md", "*.ipynb", "*.json"]
```

## Not yet supported

- Binary installs through `install_binary.py` and `standalone/binary_config.py`.
- Shared envs defined by your own package.
- Discovery by the `proto-tools` CLI and MCP server.
- Docs pages: `ToolRegistry.get_docs_url` returns `None`.
- Remote devices.
