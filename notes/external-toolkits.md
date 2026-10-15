# Out-of-tree Toolkits

A package outside proto-tools can define toolkits, with `@tool` functions, models, docs files, and a `standalone/` environment, all kept in that package's repository. Importing the package registers its tools, and they then run locally through the same environment system as packaged tools, whether proto-tools is installed editable or as a regular wheel.

This guide walks through building one, using a toolkit `mytool` in the category `scoring` of a package `mypkg`.

## Step 1: Lay out the toolkit

Mirror the packaged layout under your package:

```
mypkg/
└── tools/
    └── scoring/                  # {category}
        └── mytool/               # {toolkit}
            ├── __init__.py
            ├── mytool_score.py   # {tool_key_snake}.py, one file per tool
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

- **Pick a unique toolkit name.** The toolkit directory name is shared with proto-tools' toolkits, so it cannot match a packaged toolkit (e.g. `esm2`) or another registered out-of-tree toolkit. Importing a toolkit that does raises a `ValueError` naming both directories.
- **Keep the category in sync.** `@tool(category=...)` should match the category directory, so the `"scoring/mytool"` identifier form resolves.

## Step 2: Define the environment

The environment is built from `standalone/` exactly as for a packaged toolkit (see `tool-environments.md`): `setup.sh` runs from that directory with `bash`, and the env is rebuilt whenever its env files change.

`standalone/python_version.txt`:

```
default: 3.12
```

`standalone/setup.sh`:

```bash
#!/bin/bash
set -euo pipefail
source standalone_helpers.sh

proto_install_pytorch                 # only if the tool needs PyTorch
uv pip install -r requirements.txt
```

`standalone_helpers.sh` is provided on `PATH` while `setup.sh` runs, so its helpers (`proto_install_pytorch` and the rest) work without copying anything.

### Prebuilt binaries

To install a platform-specific binary, add `standalone/binary_config.py` (a `URLS` dict keyed by `(system, machine)` and an `extract(archive_path, bin_dir)` function, see "Binary Installation" in `tool-environments.md`) and call the helper from `setup.sh`:

```bash
proto_install_binary mytool
```

It reads `binary_config.py` from the directory `setup.sh` runs in, so it works from your package exactly as from a packaged toolkit.

### Shared envs

To reuse one env across several toolkits, put its definition (`setup.sh`, `python_version.txt`, and the rest) in a `shared_envs/<name>/` directory next to your package's `tools/`, and replace each toolkit's env files with a `standalone/shared_env.txt` naming it:

```
mypkg/
├── shared_envs/
│   └── mypkg_scoring/
│       ├── setup.sh
│       ├── python_version.txt
│       └── requirements.txt
└── tools/
    └── scoring/
        ├── mytool/standalone/shared_env.txt     # contents: "mypkg_scoring"
        └── othertool/standalone/shared_env.txt  # contents: "mypkg_scoring"
```

The env is built once as `mypkg_scoring__ext_{hash}_env`, the hash being of your package (`mypkg`). A name not found in your package resolves to proto-tools' shared envs (e.g. `biohub_esm`), and a name defined in both raises a `ValueError`.

## Step 3: Write the worker script

`standalone/inference.py` runs inside the environment. It exposes `dispatch()` for persistent workers and a `__main__` block for one-shot runs:

```python
import json
import sys
from typing import Any

from standalone_helpers import get_logger

logger = get_logger(__name__)


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Score the requested sequences."""
    return {"scores": [len(seq) for seq in input_dict["sequences"]]}


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        result = dispatch(json.load(f))
    with open(sys.argv[2], "w") as f:
        json.dump(result, f)
```

A GPU tool also implements `to_device()` and `get_memory_stats()` (see `tool-environments.md`).

`standalone_helpers` is provided on `PYTHONPATH` to every worker, the same as for packaged toolkits. For that reason the names `standalone_helpers/`, `standalone_helpers.py`, and `standalone_helpers.sh` are reserved inside `standalone/`: importing a toolkit that uses one raises a `ValueError`.

## Step 4: Register the tool

A tool follows the same `Input -> Config -> run_*() -> Output` pattern as a packaged one (see `CONTRIBUTING.md` and the `implement-tool` skill). The run function dispatches to the toolkit's directory name:

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
)
def run_mytool_score(inputs: MyToolScoreInput, config: MyToolScoreConfig, instance: Any = None) -> MyToolScoreOutput:
    result = ToolInstance.dispatch("mytool", inputs.model_dump(), instance=instance, config=config)
    return MyToolScoreOutput(**result)
```

- Each tool needs a separate config class.
- Tool keys must be unique across all registered tools.
- Running on remote devices (`device="modal"` or `device="proto"`) is coming soon.

## Step 5: Add docs and metadata

Add `README.md`, `cite.bib`, `license.yaml`, `links.yaml`, and `examples/example.ipynb` as for a packaged toolkit. They are read from the directory of the file that defines each tool, so `ToolRegistry.get_citation`, `get_license`, `get_links`, `get_readme`, and the notebook lookups work wherever the package is installed. `ToolRegistry.get_docs_url` returns `None`, since the docs site only covers packaged toolkits.

## Step 6: Package it

Non-Python files must ship in your wheel, or a non-editable install loses `setup.sh`, `requirements.txt`, `python_version.txt`, the docs files, and the rest. Use proto-tools' package-data globs, plus any other extensions your toolkit ships, and keep namespace discovery on (the default for `[tool.setuptools.packages.find]`) so `standalone/` ships without an `__init__.py`:

```toml
[tool.setuptools.packages.find]
include = ["mypkg*"]

[tool.setuptools.package-data]
"*" = ["*.sh", "*.txt", "*.csv", "*.smi", "*.cif", "*.bib", "*.yaml", "*.md", "*.ipynb", "*.pkl", "*.pdb", "*.a3m", "*.fasta", "*.hmm", "*.meme", "*.json", "*.def", "py.typed"]
```

## Step 7: Import and run

Import the module that defines the tool (or a package `__init__` that imports it); `@tool` registers it at import. From then on the tool behaves like any other:

```python
import mypkg.tools.scoring.mytool.mytool_score as mytool_score
from proto_tools.utils import ToolInstance

result = mytool_score.run_mytool_score(inputs, config)   # first call builds the environment

with ToolInstance.persist_tool("mytool-score"):          # reuse one warm worker
    for batch in batches:
        mytool_score.run_mytool_score(batch, config)
```

The environment lives under `PROTO_HOME/proto_tool_envs/` as `mytool__ext_{hash}_env`, where the hash is of the package path defining the toolkit (`mypkg.tools.scoring.mytool`). Every install of your package (any venv, editable or not) shares one env, and two packages with a same-named toolkit never share an env. `PROTO_MYTOOL_STANDALONE_DIR` overrides the env definition the same way it does for a packaged toolkit.

The `proto-tools` CLI and MCP server only import proto-tools, so they list packaged tools only.

## Not yet supported

- Discovery by the `proto-tools` CLI and MCP server.
- Docs pages on the proto-tools docs site.
- Remote devices.
