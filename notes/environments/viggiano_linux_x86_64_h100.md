# Linux x86_64 Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-100%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-28-brightgreen) ![Failed](https://img.shields.io/badge/failed-0-brightgreen) ![Skipped](https://img.shields.io/badge/skipped-1-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Linux Linux 5.15.0-1086-nvidia |
| **Architecture** | x86_64 |
| **Hostname** | `ashleylab-h100` |
| **Python** | 3.12.12 |
| **RAM** | 2015.6 GB |
| **GPU** | 8× NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3 |
| **CUDA** | 12.2 |
| **Conda Env** | `bio_tools` |

## Git

- **Commit**: `e61a3b246272`
- **Branch**: `bv/deps_iso_pass`
- **Dirty**: No

## Environment Variables

### Parent Process Environment

```
BROWSER=/home/viggiano/.cursor-server/cli/servers/Stable-32cfbe848b35d9eb320980195985450f244b3030/server/bin/helpers/browser.sh
BUNDLED_DEBUGPY_PATH=/home/viggiano/.cursor-server/extensions/ms-python.debugpy-2025.18.0-linux-x64/bundled/libs/debugpy
CLAUDE_CODE_SSE_PORT=38774
COLORTERM=truecolor
CONDA_DEFAULT_ENV=bio_tools
CONDA_EXE=/home/viggiano/miniconda3/bin/conda
CONDA_PREFIX=/projects/viggiano/envs/bio_tools
CONDA_PREFIX_1=/home/viggiano/miniconda3
CONDA_PREFIX_2=/projects/viggiano/envs/provada-env
CONDA_PREFIX_3=/home/viggiano/miniconda3
CONDA_PROMPT_MODIFIER=(bio_tools) 
CONDA_PYTHON_EXE=/home/viggiano/miniconda3/bin/python
CONDA_SHLVL=4
DISABLE_PANDERA_IMPORT_WARNING=True
ENABLE_IDE_INTEGRATION=true
HOME=/home/viggiano
LANG=en_US.UTF-8
LESSCLOSE=/usr/bin/lesspipe %s %s
LESSOPEN=| /usr/bin/lesspipe %s
LOGNAME=viggiano
LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.a...
MIG_PARTED_CHECKPOINT_FILE=/var/lib/nvidia-mig-manager/checkpoint.json
MIG_PARTED_CONFIG_FILE=/etc/nvidia-mig-manager/config.yaml
MIG_PARTED_HOOKS_FILE=/etc/nvidia-mig-manager/hooks.yaml
MOTD_SHOWN=pam
PATH=/home/viggiano/.local/bin:/projects/viggiano/envs/bio_tools/bin:/usr/local/cuda/bin:/opt/bin:/projects/viggiano/envs/bio_tools/bin:/home/viggiano/miniconda3/condabin:/home/viggiano/.cursor-server/cli/...
PWD=/raid/projects/viggiano/codebases/bio-programming-tools
PYDEVD_DISABLE_FILE_VALIDATION=1
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
RDBASE=/projects/viggiano/envs/bio_tools/lib/python3.12/site-packages/rdkit
SHELL=/bin/bash
SHLVL=3
TERM=tmux-256color
TERM_PROGRAM=tmux
TERM_PROGRAM_VERSION=3.2a
TMUX=/tmp/tmux-1013/default,909045,2
TMUX_PANE=%2
USER=viggiano
VSCODE_DEBUGPY_ADAPTER_ENDPOINTS=/home/viggiano/.cursor-server/extensions/ms-python.debugpy-2025.18.0-linux-x64/.noConfigDebugAdapterEndpoints/endpoint-3538d7243f1c2085.txt
VSCODE_GIT_IPC_HANDLE=/run/user/1013/vscode-git-db937a60ce.sock
VSCODE_IPC_HOOK_CLI=/run/user/1013/vscode-ipc-9051c67a-aa46-4029-b3ea-57cda09c9dbe.sock
XDG_DATA_DIRS=/usr/local/share:/usr/share:/var/lib/snapd/desktop
XDG_RUNTIME_DIR=/run/user/1013
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
_=/projects/viggiano/envs/bio_tools/bin/pytest
```

### Subprocess Environment (passed to tools)

```
CONDA_DEFAULT_ENV=bio_tools
CONDA_PREFIX=/projects/viggiano/envs/bio_tools
CONDA_SHLVL=4
CUDA_VISIBLE_DEVICES=0
DETECTED_COMPUTE_PLATFORM=cuda
DETECTED_CUDA_VERSION=12
DETECTED_DRIVER_VERSION=535
HOME=/home/viggiano
LANG=en_US.UTF-8
LD_LIBRARY_PATH=/raid/projects/viggiano/codebases/bio-programming-tools/tool_envs/splice_transformer_env/lib/python3.12/site-packages/nvidia/cublas/lib:/raid/projects/viggiano/codebases/bio-programming-tools/tool_env...
LOGNAME=viggiano
PATH=/raid/projects/viggiano/codebases/bio-programming-tools/tool_envs/splice_transformer_env/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RECOMMENDED_JAX_SPEC=jax[cuda12]>=0.4.20,<1
RECOMMENDED_JAX_VARIANT=cuda12
RECOMMENDED_TORCH_SPEC=torch>=2.4,<3
SHELL=/bin/bash
TORCH_CUDA_ARCH_LIST=9.0
TORCH_HOME=/raid/projects/viggiano/codebases/bio-programming-tools/tool_envs/splice_transformer_env/cache/torch
USER=viggiano
```

## Results by Category

### Causal Models (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `evo1` | yes | ✅ | 486.3s | ✅ Pass |
| `evo2` | yes | ✅ | 111.9s | ✅ Pass |
| `progen2` | yes | ✅ | 59.5s | ✅ Pass |

### Gene Annotation (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `blast` | no | ✅ | 47.1s | ✅ Pass |
| `minced` | no | ✅ | 6.2s | ✅ Pass |
| `mmseqs` | no | ✅ | 8.4s | ✅ Pass |
| `pyhmmer` | no | ✅ | 6.3s | ✅ Pass |

### Inverse Folding (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `ligandmpnn` | yes | ✅ | 104.0s | ✅ Pass |
| `proteinmpnn` | yes | ✅ | 33.3s | ✅ Pass |

### Masked Models (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `esm2` | yes | ✅ | 18.9s | ✅ Pass |
| `esm3` | yes | ✅ | 18.6s | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `orfipy` | no | ✅ | 6.5s | ✅ Pass |
| `prodigal` | no | ✅ | 6.1s | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `splice_transformer` | yes | ✅ | 19.1s | ✅ Pass |

### Sequence Alignment (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `mafft` | no | ✅ | 8.7s | ✅ Pass |

### Sequence Scoring (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `borzoi` | yes | ✅ | 43.7s | ✅ Pass |
| `enformer` | yes | ✅ | 31.0s | ✅ Pass |

### Structure Design (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `rfdiffusion3` | yes | ✅ | 75.4s | ✅ Pass |

### Structure Dynamics (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `bioemu` | no | — | 0.0s | ✅ Pass |

### Structure Prediction (5/5, 1 skipped)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `alphafold3` | yes | — | — | ⏭️ Skip (Chimera only) |
| `boltz2` | yes | ✅ | 142.3s | ✅ Pass |
| `chai1` | yes | ✅ | 200.8s | ✅ Pass |
| `esmfold` | yes | ✅ | 46.8s | ✅ Pass |
| `protenix` | yes | ✅ | 310.2s | ✅ Pass |
| `viennarna` | no | ✅ | 7.0s | ✅ Pass |

### Unknown (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `alphagenome` | yes | ✅ | 137.1s | ✅ Pass |
| `crispr_tracr` | no | ✅ | 87.9s | ✅ Pass |
| `local_colabfold_search` | no | — | 59.7s | ✅ Pass |
| `structure_metrics` | no | ✅ | 6.7s | ✅ Pass |

---
*Generated at 2026-02-21 17:12:52 by `pytest --env-report`*