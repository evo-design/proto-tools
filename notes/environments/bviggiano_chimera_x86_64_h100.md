# Linux x86_64 Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-100%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-43-brightgreen) ![Failed](https://img.shields.io/badge/failed-0-red) ![Skipped](https://img.shields.io/badge/skipped-3-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Linux Linux 5.15.0-171-generic |
| **Architecture** | x86_64 |
| **Hostname** | `GPUC960` |
| **Python** | 3.12.13 |
| **RAM** | 1007.4 GB |
| **GPU** | 1x NVIDIA H100 80GB HBM3 |
| **CUDA** | 12.2 |
| **Mamba Env** | `proto-tools` |

## Git

- **Commit**: `509ed1a709d6`
- **Branch**: `fix/pyrosetta-energy-seed-repro`
- **Dirty**: Yes

## Environment Variables

### Parent Process Environment

```
BLASTDB=/common_datasets/external/databases/blast
BROWSER=/home/bviggiano/.cursor-server/cli/servers/Stable-63715ffc1807793ce209e935e5c3ab9b79fddc80/server/bin/helpers/browser.sh
BUNDLED_DEBUGPY_PATH=/home/bviggiano/.cursor-server/extensions/ms-python.debugpy-2025.18.0-linux-x64/bundled/libs/debugpy
CLAUDECODE=1
CLAUDE_CODE_ENTRYPOINT=cli
CLAUDE_CODE_EXECPATH=/home/bviggiano/.local/share/claude/versions/2.1.104
CLAUDE_CODE_SSE_PORT=38338
COLORTERM=truecolor
CONDA_DEFAULT_ENV=proto-tools
CONDA_EXE=/home/bviggiano/miniforge3/bin/conda
CONDA_PREFIX=/home/bviggiano/miniforge3/envs/proto-tools
CONDA_PREFIX_1=/home/bviggiano/miniforge3
CONDA_PROMPT_MODIFIER=(proto-tools) 
CONDA_PYTHON_EXE=/home/bviggiano/miniforge3/bin/python
CONDA_SHLVL=2
COREPACK_ENABLE_AUTO_PIN=0
DISABLE_PANDERA_IMPORT_WARNING=True
GIT_EDITOR=true
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
OVSX_REGISTRY_URL=https://open-vsx.org
PATH=/home/bviggiano/.local/bin:/home/bviggiano/bin:/usr/local/cuda/bin:/home/bviggiano/miniforge3/envs/proto-tools/bin:/home/bviggiano/miniforge3/condabin:/home/bviggiano/.cursor-server/cli/servers/Stable...
PROTO_HOME=/large_storage/hielab/bviggiano/proto_cache
PWD=/home/bviggiano/main/codebases/proto-bio/proto-tools
PYDEVD_DISABLE_FILE_VALIDATION=1
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
RCLONE_CONFIG=/large_storage/rclone/etc/rclone.conf
RDBASE=/home/bviggiano/miniforge3/envs/proto-tools/lib/python3.12/site-packages/rdkit
SHELL=/bin/bash
SHLVL=3
SLURM_JOB_ID=2007380
TERM=xterm-256color
TERM_PROGRAM=tmux
TERM_PROGRAM_VERSION=3.2a
TMUX=/tmp/tmux-10249/default,2369338,0
TMUX_PANE=%0
USER=bviggiano
VSCODE_DEBUGPY_ADAPTER_ENDPOINTS=/home/bviggiano/.cursor-server/extensions/ms-python.debugpy-2025.18.0-linux-x64/.noConfigDebugAdapterEndpoints/endpoint-8103c33f2ddb9245.txt
VSCODE_GIT_IPC_HANDLE=/tmp/vscode-git-671fabc7c4.sock
VSCODE_IPC_HOOK_CLI=/tmp/vscode-ipc-cf6ec16f-ad05-4c7c-9d50-e7f0b09e50c4.sock
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
CONDA_PREFIX=/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/viennarna_env
DETECTED_COMPUTE_PLATFORM=cuda
DETECTED_CUDA_VERSION=12
DETECTED_DRIVER_VERSION=535
HF_HOME=/large_storage/hielab/bviggiano/proto_cache/proto_model_cache/huggingface
HOME=/home/bviggiano
LANG=C.UTF-8
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/home/bviggiano/miniforge3/envs/proto-tools/lib
LOGNAME=bviggiano
PATH=/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/viennarna_env/bin:/home/bviggiano/.local/bin:/home/bviggiano/bin:/usr/local/cuda/bin:/home/bviggiano/miniforge3/envs/proto-tools/bin:/home/b...
PIP_DEFAULT_TIMEOUT=300
PROTO_HOME=/large_storage/hielab/bviggiano/proto_cache
RECOMMENDED_JAX_SPEC=jax[cuda12]>=0.4.20,<1
RECOMMENDED_JAX_VARIANT=cuda12
RECOMMENDED_TORCH_INDEX=https://download.pytorch.org/whl/cu126
RECOMMENDED_TORCH_SPEC=torch>=2.4,<3
SHELL=/bin/bash
TORCH_CUDA_ARCH_LIST=9.0
TORCH_HOME=/large_storage/hielab/bviggiano/proto_cache/proto_model_cache/torch
USER=bviggiano
UV_HTTP_TIMEOUT=300
VIRTUAL_ENV=/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/viennarna_env
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Results by Category

### Causal Models (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `evo1-sample` | yes | ✅ | 332.1s | `509ed1a` ✱ | ✅ Pass |
| `evo2-sample` | yes | ✅ | 368.5s | `509ed1a` ✱ | ✅ Pass |
| `progen2-sample` | yes | ✅ | 229.6s | `509ed1a` ✱ | ✅ Pass |
| `progen3-sample` | yes | ✅ | 28.9s | `509ed1a` ✱ | ✅ Pass |

### Gene Annotation (5/5)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `blast-create-db` | no | ✅ | 40.1s | `509ed1a` ✱ | ✅ Pass |
| `crispr-tracr` | no | ✅ | 149.8s | `509ed1a` ✱ | ✅ Pass |
| `minced-crispr` | no | ✅ | 22.0s | `509ed1a` ✱ | ✅ Pass |
| `mmseqs-clustering` | no | ✅ | 31.5s | `509ed1a` ✱ | ✅ Pass |
| `pyhmmer-hmmscan` | no | ✅ | 22.6s | `509ed1a` ✱ | ✅ Pass |

### Inverse Folding (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `esm-if1-sample` | yes | ✅ | 127.3s | `509ed1a` ✱ | ✅ Pass |
| `fampnn-pack` | yes | ✅ | 189.4s | `509ed1a` ✱ | ✅ Pass |
| `ligandmpnn-sample` | yes | ✅ | 238.1s | `509ed1a` ✱ | ✅ Pass |
| `proteinmpnn-sample` | yes | ✅ | 92.6s | `509ed1a` ✱ | ✅ Pass |

### Masked Models (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `ablang-embedding` | yes | ✅ | 140.9s | `509ed1a` ✱ | ✅ Pass |
| `esm2-embedding` | yes | ✅ | 123.6s | `509ed1a` ✱ | ✅ Pass |
| `esm3-embedding` | yes | ✅ | 128.1s | `509ed1a` ✱ | ✅ Pass |

### Mutagenesis (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `random-nucleotide-sample` | no | - | 0.0s | `509ed1a` ✱ | ✅ Pass |
| `random-protein-sample` | no | - | 0.0s | `509ed1a` ✱ | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `orfipy-prediction` | no | ✅ | 24.2s | `509ed1a` ✱ | ✅ Pass |
| `prodigal-prediction` | no | ✅ | 15.1s | `509ed1a` ✱ | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `splice-transformer-prediction` | yes | ✅ | 97.3s | `509ed1a` ✱ | ✅ Pass |

### Sequence Alignment (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `colabfold-search` | no | ✅ | 112.0s | `509ed1a` ✱ | ✅ Pass |
| `mafft-align` | no | ✅ | 27.7s | `509ed1a` ✱ | ✅ Pass |

### Sequence Scoring (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphagenome-predict-intervals` | yes | ✅ | 276.1s | `509ed1a` ✱ | ✅ Pass |
| `borzoi-ensemble` | yes | ✅ | 160.9s | `509ed1a` ✱ | ✅ Pass |
| `enformer-prediction` | yes | ✅ | 104.7s | `509ed1a` ✱ | ✅ Pass |
| `segmasker-score` | no | ✅ | 43.9s | `509ed1a` ✱ | ✅ Pass |

### Structure Alignment (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `tmalign-alignment` | no | ✅ | 26.6s | `509ed1a` ✱ | ✅ Pass |
| `usalign-alignment` | no | ✅ | 38.9s | `509ed1a` ✱ | ✅ Pass |

### Structure Design (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `rfdiffusion3-design` | yes | ✅ | 249.3s | `509ed1a` ✱ | ✅ Pass |

### Structure Dynamics (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `bioemu-sample` | yes | ✅ | 257.1s | `509ed1a` ✱ | ✅ Pass |

### Structure Prediction (7/7)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphafold2-prediction` | yes | ✅ | 106.8s | `509ed1a` ✱ | ✅ Pass |
| `alphafold3-prediction` | yes | ✅ | 71.9s | `509ed1a` ✱ | ✅ Pass |
| `boltz2-prediction` | yes | ✅ | 164.7s | `509ed1a` ✱ | ✅ Pass |
| `chai1-prediction` | yes | ✅ | 366.6s | `509ed1a` ✱ | ✅ Pass |
| `esmfold-prediction` | yes | ✅ | 116.7s | `509ed1a` ✱ | ✅ Pass |
| `protenix-prediction` | yes | ✅ | 407.8s | `509ed1a` ✱ | ✅ Pass |
| `viennarna-prediction` | no | ✅ | 15.2s | `509ed1a` ✱ | ✅ Pass |

### Structure Scoring (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `pyrosetta-energy` | no | ✅ | 69.2s | `509ed1a` ✱ | ✅ Pass |
| `structure-metrics` | no | ✅ | 27.7s | `509ed1a` ✱ | ✅ Pass |

### Testing (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `mock-cli-multi-gpu-tool-run` | yes | - | - | `509ed1a` ✱ | ⏭️ Skip |
| `mock-cli-tool-run` | yes | ✅ | 20.1s | `509ed1a` ✱ | ✅ Pass |
| `mock-jax-multi-gpu-tool-run` | yes | - | - | `509ed1a` ✱ | ⏭️ Skip |
| `mock-jax-tool-run` | yes | ✅ | 61.9s | `509ed1a` ✱ | ✅ Pass |
| `mock-pytorch-multi-gpu-tool-run` | yes | - | - | `509ed1a` ✱ | ⏭️ Skip |
| `mock-pytorch-tool-run` | yes | ✅ | 132.4s | `509ed1a` ✱ | ✅ Pass |

---
*Generated at 2026-04-12 00:32:47 by `pytest --env-report`*

<!-- env-report-data
[
  {
    "tool_name": "ablang-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ablang-embedding]",
    "status": "passed",
    "duration_seconds": 140.9,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/ablang_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "alphafold2-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold2-prediction]",
    "status": "passed",
    "duration_seconds": 106.79,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphafold2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "alphafold3-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold3-prediction]",
    "status": "passed",
    "duration_seconds": 71.94,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphafold3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "alphagenome-predict-intervals",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphagenome-predict-intervals]",
    "status": "passed",
    "duration_seconds": 276.06,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphagenome_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "bioemu-sample",
    "category": "structure_dynamics",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[bioemu-sample]",
    "status": "passed",
    "duration_seconds": 257.06,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/bioemu_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "blast-create-db",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[blast-create-db]",
    "status": "passed",
    "duration_seconds": 40.14,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/blast_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "boltz2-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[boltz2-prediction]",
    "status": "passed",
    "duration_seconds": 164.69,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/boltz2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "borzoi-ensemble",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[borzoi-ensemble]",
    "status": "passed",
    "duration_seconds": 160.91,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/borzoi_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "chai1-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[chai1-prediction]",
    "status": "passed",
    "duration_seconds": 366.6,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/chai1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "colabfold-search",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[colabfold-search]",
    "status": "passed",
    "duration_seconds": 112.01,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/colabfold_search_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "crispr-tracr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[crispr-tracr]",
    "status": "passed",
    "duration_seconds": 149.77,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/crispr_tracr_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "enformer-prediction",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[enformer-prediction]",
    "status": "passed",
    "duration_seconds": 104.65,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/enformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "esm-if1-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm-if1-sample]",
    "status": "passed",
    "duration_seconds": 127.32,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/esm_if1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "esm2-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm2-embedding]",
    "status": "passed",
    "duration_seconds": 123.56,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/esm2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "esm3-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm3-embedding]",
    "status": "passed",
    "duration_seconds": 128.13,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/esm3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "esmfold-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esmfold-prediction]",
    "status": "passed",
    "duration_seconds": 116.67,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/esmfold_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "evo1-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo1-sample]",
    "status": "passed",
    "duration_seconds": 332.08,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/evo1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "evo2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo2-sample]",
    "status": "passed",
    "duration_seconds": 368.53,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/evo2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "fampnn-pack",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[fampnn-pack]",
    "status": "passed",
    "duration_seconds": 189.4,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/fampnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "ligandmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ligandmpnn-sample]",
    "status": "passed",
    "duration_seconds": 238.08,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/ligandmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "mafft-align",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mafft-align]",
    "status": "passed",
    "duration_seconds": 27.67,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mafft_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "minced-crispr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[minced-crispr]",
    "status": "passed",
    "duration_seconds": 22.01,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/minced_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "mmseqs-clustering",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mmseqs-clustering]",
    "status": "passed",
    "duration_seconds": 31.51,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mmseqs_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "mock-cli-multi-gpu-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-cli-multi-gpu-tool-run]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 72, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "mock-cli-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-cli-tool-run]",
    "status": "passed",
    "duration_seconds": 20.11,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mock_cli_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "mock-jax-multi-gpu-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-jax-multi-gpu-tool-run]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 72, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "mock-jax-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-jax-tool-run]",
    "status": "passed",
    "duration_seconds": 61.9,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mock_jax_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "mock-pytorch-multi-gpu-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-pytorch-multi-gpu-tool-run]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 72, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "mock-pytorch-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-pytorch-tool-run]",
    "status": "passed",
    "duration_seconds": 132.43,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mock_pytorch_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "orfipy-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[orfipy-prediction]",
    "status": "passed",
    "duration_seconds": 24.23,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/orfipy_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "prodigal-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[prodigal-prediction]",
    "status": "passed",
    "duration_seconds": 15.11,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/prodigal_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "progen2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen2-sample]",
    "status": "passed",
    "duration_seconds": 229.61,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/progen2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "progen3-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen3-sample]",
    "status": "passed",
    "duration_seconds": 28.95,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/progen3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "proteinmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[proteinmpnn-sample]",
    "status": "passed",
    "duration_seconds": 92.56,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/proteinmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "protenix-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[protenix-prediction]",
    "status": "passed",
    "duration_seconds": 407.77,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/protenix_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "pyhmmer-hmmscan",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pyhmmer-hmmscan]",
    "status": "passed",
    "duration_seconds": 22.65,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/pyhmmer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "pyrosetta-energy",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pyrosetta-energy]",
    "status": "passed",
    "duration_seconds": 69.2,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/pyrosetta_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "random-nucleotide-sample",
    "category": "mutagenesis",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[random-nucleotide-sample]",
    "status": "passed",
    "duration_seconds": 0.0,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "random-protein-sample",
    "category": "mutagenesis",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[random-protein-sample]",
    "status": "passed",
    "duration_seconds": 0.0,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "rfdiffusion3-design",
    "category": "structure_design",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[rfdiffusion3-design]",
    "status": "passed",
    "duration_seconds": 249.27,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/rfdiffusion3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "segmasker-score",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[segmasker-score]",
    "status": "passed",
    "duration_seconds": 43.88,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/segmasker_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "splice-transformer-prediction",
    "category": "rna_splicing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[splice-transformer-prediction]",
    "status": "passed",
    "duration_seconds": 97.28,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/splice_transformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "structure-metrics",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[structure-metrics]",
    "status": "passed",
    "duration_seconds": 27.72,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/structure_metrics_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "tmalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[tmalign-alignment]",
    "status": "passed",
    "duration_seconds": 26.63,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/tmalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "usalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[usalign-alignment]",
    "status": "passed",
    "duration_seconds": 38.94,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/usalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  },
  {
    "tool_name": "viennarna-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[viennarna-prediction]",
    "status": "passed",
    "duration_seconds": 15.23,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/viennarna_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "509ed1a709d6",
    "git_dirty": true
  }
]
-->