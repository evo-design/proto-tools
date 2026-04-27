# Linux x86_64 Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-97%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-46-brightgreen) ![Failed](https://img.shields.io/badge/failed-1-red) ![Skipped](https://img.shields.io/badge/skipped-3-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Linux Linux 5.15.0-164-generic |
| **Architecture** | x86_64 |
| **Hostname** | `GPU71E4` |
| **Python** | 3.14.4 |
| **RAM** | 1007.4 GB |
| **GPU** | 1x NVIDIA H100 80GB HBM3 |
| **CUDA** | 12.2 |
| **Mamba Env** | `proto-tools` |

## Git

- **Commit**: `509a7f884e75`
- **Branch**: `feat/skip-foundation-env-on-capable-hosts`
- **Dirty**: Yes

## Environment Variables

### Parent Process Environment

```
BLASTDB=/common_datasets/external/databases/blast
BROWSER=/home/bviggiano/.vscode-server/cli/servers/Stable-10c8e557c8b9f9ed0a87f61f1c9a44bde731c409/server/bin/helpers/browser.sh
BUNDLED_DEBUGPY_PATH=/home/bviggiano/.vscode-server/extensions/ms-python.debugpy-2025.18.0/bundled/libs/debugpy
CLAUDECODE=1
CLAUDE_CODE_ENTRYPOINT=cli
CLAUDE_CODE_EXECPATH=/home/bviggiano/.local/share/claude/versions/2.1.119
CLAUDE_CODE_SSE_PORT=27569
COLORTERM=truecolor
CONDA_DEFAULT_ENV=proto-tools
CONDA_EXE=/home/bviggiano/miniforge3/bin/conda
CONDA_PREFIX=/home/bviggiano/miniforge3/envs/proto-tools
CONDA_PREFIX_1=/home/bviggiano/miniforge3
CONDA_PREFIX_2=/home/bviggiano/miniforge3/envs/proto-tools
CONDA_PREFIX_3=/home/bviggiano/miniforge3
CONDA_PROMPT_MODIFIER=(proto-tools) 
CONDA_PYTHON_EXE=/home/bviggiano/miniforge3/bin/python
CONDA_SHLVL=4
COREPACK_ENABLE_AUTO_PIN=0
DISABLE_PANDERA_IMPORT_WARNING=True
GIT_EDITOR=true
GK_GL_ADDR=http://127.0.0.1:38473
GK_GL_PATH=/tmp/gitkraken/gitlens/gitlens-ipc-server-2958225-38473.json
HOME=/home/bviggiano
LANG=C.UTF-8
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64
LESSCLOSE=/usr/bin/lesspipe %s %s
LESSOPEN=| /usr/bin/lesspipe %s
LOGNAME=bviggiano
LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.a...
MAMBA_EXE=/home/bviggiano/miniforge3/bin/mamba
MAMBA_ROOT_PREFIX=/home/bviggiano/.local/share/mamba
MOTD_SHOWN=pam
NoDefaultCurrentDirectoryInExePath=1
OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta
PATH=/home/bviggiano/.local/bin:/home/bviggiano/bin:/usr/local/cuda/bin:/home/bviggiano/miniforge3/envs/proto-tools/bin:/home/bviggiano/miniforge3/condabin:/home/bviggiano/.vscode-server/data/User/globalSt...
PROTO_ALPHAFOLD3_WEIGHTS_DIR=/large_storage/hielab/brk/models/af3_weights
PROTO_HOME=/large_storage/hielab/bviggiano/proto_cache
PWD=/home/bviggiano/main/codebases/evo-design/proto-tools
PYDEVD_DISABLE_FILE_VALIDATION=1
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.3
PYTHONSTARTUP=/home/bviggiano/.vscode-server/data/User/workspaceStorage/606b48bbf88196b292aff26cc9961831/ms-python.python/pythonrc.py
PYTHON_BASIC_REPL=1
RCLONE_CONFIG=/large_storage/rclone/etc/rclone.conf
RDBASE=/home/bviggiano/miniforge3/envs/proto-tools/lib/python3.14/site-packages/rdkit
SHELL=/bin/bash
SHLVL=4
SLURM_JOB_ID=2222952
TERM=xterm-256color
TERM_PROGRAM=tmux
TERM_PROGRAM_VERSION=3.2a
TMUX=/tmp/tmux-10249/default,2595237,0
TMUX_PANE=%0
USER=bviggiano
VSCODE_DEBUGPY_ADAPTER_ENDPOINTS=/home/bviggiano/.vscode-server/extensions/ms-python.debugpy-2025.18.0/.noConfigDebugAdapterEndpoints/endpoint-8103c33f2ddb9245.txt
VSCODE_GIT_IPC_HANDLE=/tmp/vscode-git-2e22fcc75c.sock
VSCODE_IPC_HOOK_CLI=/tmp/vscode-ipc-ce76d828-a694-4aec-83cb-c7c78a2c3b3a.sock
VSCODE_PYTHON_AUTOACTIVATE_GUARD=1
XDG_DATA_DIRS=/usr/local/share:/usr/share:/var/lib/snapd/desktop
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
XML_CATALOG_FILES=file:///home/bviggiano/miniforge3/etc/xml/catalog file:///etc/xml/catalog file:///home/bviggiano/miniforge3/etc/xml/catalog file:///etc/xml/catalog
_=/home/bviggiano/miniforge3/envs/proto-tools/bin/pytest
_CE_CONDA=
_CE_M=
_CONDA_EXE=/home/bviggiano/miniforge3/bin/conda
_CONDA_ROOT=/home/bviggiano/miniforge3
```

### Subprocess Environment (passed to tools)

```
CONDA_PREFIX=/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/proteinmpnn_env
DETECTED_COMPUTE_PLATFORM=cuda
DETECTED_CUDA_VERSION=12
DETECTED_DRIVER_VERSION=535
HF_HOME=/large_storage/hielab/bviggiano/proto_cache/proto_model_cache/huggingface
HOME=/home/bviggiano
LANG=C.UTF-8
LD_LIBRARY_PATH=/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/proteinmpnn_env/cuda_env/lib:/usr/local/cuda/lib64:/home/bviggiano/miniforge3/envs/proto-tools/lib
LOGNAME=bviggiano
PATH=/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/proteinmpnn_env/bin:/usr/local/cuda/bin:/home/bviggiano/.local/bin:/home/bviggiano/bin:/home/bviggiano/miniforge3/envs/proto-tools/bin:/home...
PIP_CACHE_DIR=/large_storage/hielab/bviggiano/proto_cache/pip_cache
PIP_DEFAULT_TIMEOUT=300
PROTO_ALPHAFOLD3_WEIGHTS_DIR=/large_storage/hielab/brk/models/af3_weights
PROTO_HOME=/large_storage/hielab/bviggiano/proto_cache
RECOMMENDED_JAX_SPEC=jax[cuda12]>=0.4.20,<1
RECOMMENDED_JAX_VARIANT=cuda12
RECOMMENDED_TORCH_INDEX=https://download.pytorch.org/whl/cu126
RECOMMENDED_TORCH_SPEC=torch>=2.4,<3
SHELL=/bin/bash
TORCH_CUDA_ARCH_LIST=9.0
TORCH_HOME=/large_storage/hielab/bviggiano/proto_cache/proto_model_cache/torch
USER=bviggiano
UV_CACHE_DIR=/large_storage/hielab/bviggiano/proto_cache/uv_cache
UV_HTTP_TIMEOUT=300
VIRTUAL_ENV=/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/proteinmpnn_env
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Results by Category

### Causal Models (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `evo1-sample` | yes | ✅ | 216.6s | `509a7f8` ✱ | ✅ Pass |
| `evo2-sample` | yes | ✅ | 259.5s | `509a7f8` ✱ | ✅ Pass |
| `progen2-sample` | yes | ✅ | 89.8s | `509a7f8` ✱ | ✅ Pass |
| `progen3-sample` | yes | ✅ | 32.1s | `509a7f8` ✱ | ✅ Pass |

### Gene Annotation (5/5)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `blast-create-db` | no | ✅ | 44.1s | `509a7f8` ✱ | ✅ Pass |
| `crispr-tracr` | no | ✅ | 181.5s | `509a7f8` ✱ | ✅ Pass |
| `minced-crispr` | no | ✅ | 20.4s | `509a7f8` ✱ | ✅ Pass |
| `mmseqs-clustering` | no | ✅ | 23.6s | `509a7f8` ✱ | ✅ Pass |
| `pyhmmer-hmmscan` | no | ✅ | 16.4s | `509a7f8` ✱ | ✅ Pass |

### Inverse Folding (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `esm-if1-sample` | yes | ✅ | 60.2s | `509a7f8` ✱ | ✅ Pass |
| `fampnn-pack` | yes | ✅ | 155.7s | `509a7f8` ✱ | ✅ Pass |
| `ligandmpnn-sample` | yes | ✅ | 118.9s | `509a7f8` ✱ | ✅ Pass |
| `proteinmpnn-sample` | yes | ✅ | 73.8s | `509a7f8` ✱ | ✅ Pass |

### Masked Models (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `ablang-embedding` | yes | ✅ | 54.5s | `509a7f8` ✱ | ✅ Pass |
| `esm2-embedding` | yes | ✅ | 58.3s | `509a7f8` ✱ | ✅ Pass |
| `esm3-embedding` | yes | ✅ | 71.8s | `509a7f8` ✱ | ✅ Pass |
| `esmc-embedding` | yes | ✅ | 8.9s | `509a7f8` ✱ | ✅ Pass |

### Mutagenesis (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `random-nucleotide-sample` | no | - | 0.0s | `509a7f8` ✱ | ✅ Pass |
| `random-protein-sample` | no | - | 0.0s | `509a7f8` ✱ | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `orfipy-prediction` | no | ✅ | 21.3s | `509a7f8` ✱ | ✅ Pass |
| `prodigal-prediction` | no | ✅ | 19.4s | `509a7f8` ✱ | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `splice-transformer-prediction` | yes | ✅ | 52.9s | `509a7f8` ✱ | ✅ Pass |

### Sequence Alignment (2/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `colabfold-search` | no | ✅ | 1640.7s | `509a7f8` ✱ | ✅ Pass |
| `mafft-align` | no | ✅ | 132.6s | `509a7f8` ✱ | ✅ Pass |
| `mmseqs2-homology-search` | no | - | 0.0s | `509a7f8` ✱ | ❌ Fail |

### Sequence Scoring (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphagenome-predict-intervals` | yes | ✅ | 1776.8s | `509a7f8` ✱ | ✅ Pass |
| `borzoi-ensemble` | yes | ✅ | 118.2s | `509a7f8` ✱ | ✅ Pass |
| `enformer-prediction` | yes | ✅ | 44.2s | `509a7f8` ✱ | ✅ Pass |
| `segmasker-score` | no | ✅ | 27.6s | `509a7f8` ✱ | ✅ Pass |

### Structure Alignment (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `tmalign-alignment` | no | ✅ | 25.6s | `509a7f8` ✱ | ✅ Pass |
| `usalign-alignment` | no | ✅ | 36.4s | `509a7f8` ✱ | ✅ Pass |

### Structure Design (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `rfdiffusion3-design` | yes | ✅ | 165.3s | `509a7f8` ✱ | ✅ Pass |

### Structure Dynamics (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `bioemu-sample` | yes | ✅ | 124.6s | `509a7f8` ✱ | ✅ Pass |

### Structure Prediction (7/7)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphafold2-binder` | yes | ✅ | 246.6s | `509a7f8` ✱ | ✅ Pass |
| `alphafold3-prediction` | yes | ✅ | 453.6s | `509a7f8` ✱ | ✅ Pass |
| `boltz2-prediction` | yes | ✅ | 103.3s | `509a7f8` ✱ | ✅ Pass |
| `chai1-prediction` | yes | ✅ | 366.2s | `509a7f8` ✱ | ✅ Pass |
| `esmfold-prediction` | yes | ✅ | 86.1s | `509a7f8` ✱ | ✅ Pass |
| `protenix-prediction` | yes | ✅ | 355.1s | `509a7f8` ✱ | ✅ Pass |
| `viennarna-prediction` | no | ✅ | 15.4s | `509a7f8` ✱ | ✅ Pass |

### Structure Scoring (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `dssp-secondary-structure` | no | ✅ | 24.5s | `509a7f8` ✱ | ✅ Pass |
| `pdockq2` | no | - | 0.0s | `509a7f8` ✱ | ✅ Pass |
| `pyrosetta-energy` | no | ✅ | 60.0s | `509a7f8` ✱ | ✅ Pass |
| `structure-metrics` | no | - | 0.3s | `509a7f8` ✱ | ✅ Pass |

### Testing (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `mock-cli-multi-gpu-tool-run` | yes | - | - | `509a7f8` ✱ | ⏭️ Skip |
| `mock-cli-tool-run` | yes | ✅ | 14.9s | `509a7f8` ✱ | ✅ Pass |
| `mock-jax-multi-gpu-tool-run` | yes | - | - | `509a7f8` ✱ | ⏭️ Skip |
| `mock-jax-tool-run` | yes | ✅ | 48.9s | `509a7f8` ✱ | ✅ Pass |
| `mock-pytorch-multi-gpu-tool-run` | yes | - | - | `509a7f8` ✱ | ⏭️ Skip |
| `mock-pytorch-tool-run` | yes | ✅ | 148.3s | `509a7f8` ✱ | ✅ Pass |

## Failure Details

### ❌ `mmseqs2-homology-search`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mmseqs2-homology-search]`

```
tests/tool_infra_tests/test_env_report.py:74: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool mmseqs2-homology-search failed: ['Dataset \'uniref30-2302\' not provisioned on disk: expected /large_storage/hielab/bviggiano/proto_cache/proto_model_cache/databases/uniref30_2302.\nProvision with:\n  bash proto_tools/tools/sequence_alignment/colabfold_search/setup_databases.sh \\\n    "/large_storage/hielab/bviggiano/proto_cache/proto_model_cache/databases/uniref30_2302" uniref30_2302 colabfold_envdb_202108 1\n(see proto_tools/tools/sequence_alignment/colabfold_search/README.md → Local Database Setup)', 'Traceback (most recent call last):\n  File "/large_storage/hielab/bviggiano/codebases/evo-design/proto-tools/proto_tools/tools/tool_registry.py", line 566, in _wrapper_body\n    result = func(inputs, config, instance)\n  File "/large_storage/hielab/bviggiano/codebases/evo-design/proto-tools/proto_tools/tools/sequence_alignment/mmseqs2_homology_search/mmseqs2_homology_search.py", line 441, in run_mmseqs2_homology_search\n    _check_dataset_provisioned(dataset_name, entry, cache_dir, require_idx_pad=config.use_gpu)\n    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/large_storage/hielab/bviggiano/codebases/evo-design/proto-tools/proto_tools/tools/sequence_alignment/mmseqs2_homology_search/mmseqs2_homology_search.py", line 598, in _check_dataset_provisioned\n    raise FileNotFoundError(\n    ...<5 lines>...\n    )\nFileNotFoundError: Dataset \'uniref30-2302\' not provisioned on disk: expected /large_storage/hielab/bviggiano/proto_cache/proto_model_cache/databases/uniref30_2302.\nProvision with:\n  bash proto_tools/tools/sequence_alignment/colabfold_search/setup_databases.sh \\\n    "/large_storage/hielab/bviggiano/proto_cache/proto_model_cache/databases/uniref30_2302" uniref30_2302 colabfold_envdb_202108 1\n(see proto_tools/tools/sequence_alignment/colabfold_search/README.md → Local Database Setup)\n']
E   assert False
E    +  where False = Mmseqs2HomologySearchOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success
```

---
*Generated at 2026-04-26 20:41:16 by `pytest --env-report`*

<!-- env-report-data
[
  {
    "tool_key": "random-nucleotide-sample",
    "category": "mutagenesis",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[random-nucleotide-sample]",
    "status": "passed",
    "duration_seconds": 0.01,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "esm3-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm3-embedding]",
    "status": "passed",
    "duration_seconds": 71.81,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/evolutionaryscale_esm_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "bioemu-sample",
    "category": "structure_dynamics",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[bioemu-sample]",
    "status": "passed",
    "duration_seconds": 124.56,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/bioemu_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "mock-jax-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-jax-tool-run]",
    "status": "passed",
    "duration_seconds": 48.87,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mock_jax_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "ablang-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ablang-embedding]",
    "status": "passed",
    "duration_seconds": 54.5,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/ablang_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "protenix-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[protenix-prediction]",
    "status": "passed",
    "duration_seconds": 355.05,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/protenix_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "progen3-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen3-sample]",
    "status": "passed",
    "duration_seconds": 32.13,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/progen3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "mock-cli-multi-gpu-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-cli-multi-gpu-tool-run]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/large_storage/hielab/bviggiano/codebases/evo-design/proto-tools/tests/tool_infra_tests/test_env_report.py', 68, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "progen2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen2-sample]",
    "status": "passed",
    "duration_seconds": 89.81,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/progen2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "esmc-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esmc-embedding]",
    "status": "passed",
    "duration_seconds": 8.9,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/evolutionaryscale_esm_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "segmasker-score",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[segmasker-score]",
    "status": "passed",
    "duration_seconds": 27.59,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/segmasker_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "pdockq2",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pdockq2]",
    "status": "passed",
    "duration_seconds": 0.0,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "esm-if1-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm-if1-sample]",
    "status": "passed",
    "duration_seconds": 60.2,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/esm_if1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "orfipy-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[orfipy-prediction]",
    "status": "passed",
    "duration_seconds": 21.27,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/orfipy_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "mafft-align",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mafft-align]",
    "status": "passed",
    "duration_seconds": 132.62,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mafft_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "pyrosetta-energy",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pyrosetta-energy]",
    "status": "passed",
    "duration_seconds": 59.99,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/pyrosetta_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "crispr-tracr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[crispr-tracr]",
    "status": "passed",
    "duration_seconds": 181.47,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/crispr_tracr_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "mmseqs2-homology-search",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mmseqs2-homology-search]",
    "status": "failed",
    "duration_seconds": 0.02,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "tests/tool_infra_tests/test_env_report.py:74: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool mmseqs2-homology-search failed: ['Dataset \\'uniref30-2302\\' not provisioned on disk: expected /large_storage/hielab/bviggiano/proto_cache/proto_model_cache/databases/uniref30_2302.\\nProvision with:\\n  bash proto_tools/tools/sequence_alignment/colabfold_search/setup_databases.sh \\\\\\n    \"/large_storage/hielab/bviggiano/proto_cache/proto_model_cache/databases/uniref30_2302\" uniref30_2302 colabfold_envdb_202108 1\\n(see proto_tools/tools/sequence_alignment/colabfold_search/README.md \u2192 Local Database Setup)', 'Traceback (most recent call last):\\n  File \"/large_storage/hielab/bviggiano/codebases/evo-design/proto-tools/proto_tools/tools/tool_registry.py\", line 566, in _wrapper_body\\n    result = func(inputs, config, instance)\\n  File \"/large_storage/hielab/bviggiano/codebases/evo-design/proto-tools/proto_tools/tools/sequence_alignment/mmseqs2_homology_search/mmseqs2_homology_search.py\", line 441, in run_mmseqs2_homology_search\\n    _check_dataset_provisioned(dataset_name, entry, cache_dir, require_idx_pad=config.use_gpu)\\n    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/large_storage/hielab/bviggiano/codebases/evo-design/proto-tools/proto_tools/tools/sequence_alignment/mmseqs2_homology_search/mmseqs2_homology_search.py\", line 598, in _check_dataset_provisioned\\n    raise FileNotFoundError(\\n    ...<5 lines>...\\n    )\\nFileNotFoundError: Dataset \\'uniref30-2302\\' not provisioned on disk: expected /large_storage/hielab/bviggiano/proto_cache/proto_model_cache/databases/uniref30_2302.\\nProvision with:\\n  bash proto_tools/tools/sequence_alignment/colabfold_search/setup_databases.sh \\\\\\n    \"/large_storage/hielab/bviggiano/proto_cache/proto_model_cache/databases/uniref30_2302\" uniref30_2302 colabfold_envdb_202108 1\\n(see proto_tools/tools/sequence_alignment/colabfold_search/README.md \u2192 Local Database Setup)\\n']\nE   assert False\nE    +  where False = Mmseqs2HomologySearchOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success",
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "enformer-prediction",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[enformer-prediction]",
    "status": "passed",
    "duration_seconds": 44.18,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/enformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "mock-cli-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-cli-tool-run]",
    "status": "passed",
    "duration_seconds": 14.95,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mock_cli_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "mock-pytorch-multi-gpu-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-pytorch-multi-gpu-tool-run]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/large_storage/hielab/bviggiano/codebases/evo-design/proto-tools/tests/tool_infra_tests/test_env_report.py', 68, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "chai1-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[chai1-prediction]",
    "status": "passed",
    "duration_seconds": 366.18,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/chai1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "tmalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[tmalign-alignment]",
    "status": "passed",
    "duration_seconds": 25.57,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/tmalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "boltz2-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[boltz2-prediction]",
    "status": "passed",
    "duration_seconds": 103.28,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/boltz2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "alphafold3-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold3-prediction]",
    "status": "passed",
    "duration_seconds": 453.57,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphafold3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "alphafold2-binder",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold2-binder]",
    "status": "passed",
    "duration_seconds": 246.6,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphafold2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "mock-jax-multi-gpu-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-jax-multi-gpu-tool-run]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/large_storage/hielab/bviggiano/codebases/evo-design/proto-tools/tests/tool_infra_tests/test_env_report.py', 68, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "esm2-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm2-embedding]",
    "status": "passed",
    "duration_seconds": 58.32,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/esm2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "rfdiffusion3-design",
    "category": "structure_design",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[rfdiffusion3-design]",
    "status": "passed",
    "duration_seconds": 165.31,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/rfdiffusion3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "minced-crispr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[minced-crispr]",
    "status": "passed",
    "duration_seconds": 20.43,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/minced_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "random-protein-sample",
    "category": "mutagenesis",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[random-protein-sample]",
    "status": "passed",
    "duration_seconds": 0.0,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "prodigal-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[prodigal-prediction]",
    "status": "passed",
    "duration_seconds": 19.45,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/prodigal_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "alphagenome-predict-intervals",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphagenome-predict-intervals]",
    "status": "passed",
    "duration_seconds": 1776.78,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphagenome_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "mock-pytorch-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-pytorch-tool-run]",
    "status": "passed",
    "duration_seconds": 148.29,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mock_pytorch_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "ligandmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ligandmpnn-sample]",
    "status": "passed",
    "duration_seconds": 118.92,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/ligandmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "mmseqs-clustering",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mmseqs-clustering]",
    "status": "passed",
    "duration_seconds": 23.58,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mmseqs_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "evo2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo2-sample]",
    "status": "passed",
    "duration_seconds": 259.46,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/evo2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "esmfold-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esmfold-prediction]",
    "status": "passed",
    "duration_seconds": 86.08,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/esmfold_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "usalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[usalign-alignment]",
    "status": "passed",
    "duration_seconds": 36.42,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/usalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "structure-metrics",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[structure-metrics]",
    "status": "passed",
    "duration_seconds": 0.28,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "dssp-secondary-structure",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[dssp-secondary-structure]",
    "status": "passed",
    "duration_seconds": 24.46,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/dssp_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "blast-create-db",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[blast-create-db]",
    "status": "passed",
    "duration_seconds": 44.09,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/blast_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "borzoi-ensemble",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[borzoi-ensemble]",
    "status": "passed",
    "duration_seconds": 118.17,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/borzoi_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "colabfold-search",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[colabfold-search]",
    "status": "passed",
    "duration_seconds": 1640.73,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/colabfold_search_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "splice-transformer-prediction",
    "category": "rna_splicing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[splice-transformer-prediction]",
    "status": "passed",
    "duration_seconds": 52.93,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/splice_transformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "viennarna-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[viennarna-prediction]",
    "status": "passed",
    "duration_seconds": 15.43,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/viennarna_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "pyhmmer-hmmscan",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pyhmmer-hmmscan]",
    "status": "passed",
    "duration_seconds": 16.39,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/pyhmmer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "fampnn-pack",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[fampnn-pack]",
    "status": "passed",
    "duration_seconds": 155.66,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/fampnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "evo1-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo1-sample]",
    "status": "passed",
    "duration_seconds": 216.63,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/evo1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  },
  {
    "tool_key": "proteinmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[proteinmpnn-sample]",
    "status": "passed",
    "duration_seconds": 73.77,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/proteinmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509a7f884e75",
    "git_dirty": true
  }
]
-->