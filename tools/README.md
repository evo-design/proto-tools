# Model Tools in `bio_programming/bio_tools/tools`

This document describes the model implementations in the `bio_programming/bio_tools/tools` directory.
These include language models, structure prediction models, RNA splicing models, and sequence
scoring models. Each model is implemented with support for both local execution and cloud deployment.

## Execution Modes

Models in this directory support two execution modes:
1. **Local Execution**: Using local devices with the option of isolated virtual environments managed by the [`EnvManager`](../env_manager.py)
2. **the cloud runtime Deployment**: Cloud execution using the cloud runtime

All models will support both execution modes (if possible), allowing users to choose
between local compute resources and cloud scaling based on their needs.

### Dependecy isolated local execution

Models with complex dependencies are managed using the `EnvManager` class from
`bio_programming.bio_tools.tools.infra.env_manager`. The EnvManager automatically creates and manages
isolated virtual environments for each model.

#### How it works

1. **Model Discovery**: The EnvManager scans this directory tree for any subdirectory
   that contains a `standalone/` folder.

2. **Virtual Environment Creation**: For each model, it creates a dedicated venv in
   `.venvs/{model_name}_env/`.

3. **Dependency Installation**: Each model's `standalone/setup.sh` script is executed
   within the activated venv to install dependencies.

4. **Status Tracking**: A `STATUS.txt` file in each venv tracks whether setup was
   successful or contains error details.

5. **Script Execution**: Model scripts run in their isolated environments via
   `call_standalone_script_in_venv()`.

### Model Structure

Each model that uses the EnvManager should have this structure:

```
tools/
├── masked_models/           # Masked/bidirectional language models (ESM2, ESM3)
│   └── model_name/
│       ├── __init__.py          # Model interface (local + cloud)
│       └── standalone/
│           ├── setup.sh          # Dependency installation script
│           ├── requirements.txt  # (optional) pip requirements
│           ├── run.py           # A script that can be run with the venv activated
│           └── ...              # Other model files
├── causal_models/           # Causal/autoregressive language models (Evo2, ProGen2)
│   └── model_name/
│       ├── __init__.py          # Model interface (local + cloud)
│       └── standalone/
│           ├── setup.sh          # Dependency installation script
│           ├── requirements.txt  # (optional) pip requirements
│           ├── run.py           # A script that can be run with the venv activated
│           └── ...              # Other model files
├── structure_prediction/
│   └── model_name/
│       ├── __init__.py          # Model interface (local + cloud)
│       └── standalone/
│           ├── setup.sh          # Dependency installation script
│           ├── requirements.txt  # (optional) pip requirements
│           ├── run.py           # A script that can be run with the venv activated
│           └── ...              # Other model files
├── rna_splicing/
└── sequence_scoring/
```

#### Setup Script Requirements

The `setup.sh` script should:
- Start with `#!/bin/bash`
- Use simple commands like `pip install -r requirements.txt`
- Exit with code 0 on success, non-zero on failure
- The script runs with the venv already activated

### Tools Requiring System Binaries

Some tools wrap external C/C++ binaries (e.g., BLAST+, MMseqs2, MAFFT) that can't
be installed via pip. These use the same `standalone/` pattern but additionally include
a `binary_config.py` file that tells the shared installer how to download and extract
the binary for the current platform.

#### How it works

1. The tool's `setup.sh` calls `python ../../infra/install_binary.py <tool_name>`
2. `infra/install_binary.py` discovers the tool's `standalone/binary_config.py`
3. It reads the platform-specific download URL from `binary_config.URLS`
4. Downloads the archive and calls `binary_config.extract()` to install binaries
5. Binaries are placed in the venv's `bin/` directory alongside the venv Python
6. The tool's `run.py` finds binaries via `Path(sys.executable).parent / "binary_name"`

#### Adding a new binary tool

1. Create `standalone/binary_config.py` with two exports:

```python
# my_tool/standalone/binary_config.py
from pathlib import Path

URLS = {
    ("Darwin", "arm64"):  "https://example.com/my_tool-macos-arm64.tar.gz",
    ("Darwin", "x86_64"): "https://example.com/my_tool-macos-x64.tar.gz",
    ("Linux", "x86_64"):  "https://example.com/my_tool-linux-x64.tar.gz",
}

def extract(archive_path: Path, bin_dir: Path) -> None:
    """Extract binaries from the downloaded archive into bin_dir."""
    import tarfile, stat
    with tarfile.open(archive_path, "r:gz") as tar:
        # Extract and copy the relevant binaries to bin_dir
        ...
```

2. Add the download step to `standalone/setup.sh`:

```bash
#!/bin/bash
set -euo pipefail
pip install uv
uv pip install -r requirements.txt
python "$(dirname "$0")/../../infra/install_binary.py" my_tool
```

3. In `standalone/run.py`, find binaries relative to the venv Python:

```python
from pathlib import Path
import sys

my_binary = str(Path(sys.executable).parent / "my_tool")
```

Supported platforms: macOS ARM64, macOS x86_64, Linux x86_64.

### Usage Example

```python
from bio_programming.bio_tools.tools.infra.env_manager import EnvManager

# Create/validate environment for a model
env_manager = EnvManager("boltz2")

# Run a script in the model's environment
result = env_manager.call_standalone_script_in_venv(
    script_path=Path("path/to/script.py"),
    input_dict={"param": "value"},
    device="cuda:0"
)
```

## the cloud runtime Deployment (Cloud Execution)

Models can also be deployed on the cloud runtime for cloud execution. This is useful for:
- Models requiring specific GPU types not available locally
- Scaling to handle multiple concurrent requests
- Offloading compute to reduce local resource usage
the cloud runtime deployment configurations are typically defined in the model's `__init__.py`
file alongside the local execution interface.
