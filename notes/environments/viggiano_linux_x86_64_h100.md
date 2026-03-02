# Linux x86_64 Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-100%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-33-brightgreen) ![Failed](https://img.shields.io/badge/failed-0-red) ![Skipped](https://img.shields.io/badge/skipped-0-lightgrey)

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
| **Conda Env** | `bio-tools` |

## Git

- **Commit**: `5c8918c9d60d`
- **Branch**: `device_management`
- **Dirty**: Yes

## Environment Variables

### Parent Process Environment

```
CONDA_DEFAULT_ENV=bio-tools
CONDA_EXE=/home/viggiano/miniconda3/bin/conda
CONDA_PREFIX=/projects/viggiano/envs/bio-tools
CONDA_PREFIX_1=/home/viggiano/miniconda3
CONDA_PREFIX_2=/projects/viggiano/envs/bio_tools
CONDA_PREFIX_3=/home/viggiano/miniconda3
CONDA_PREFIX_4=/projects/viggiano/envs/bio_tools
CONDA_PREFIX_5=/home/viggiano/miniconda3
CONDA_PREFIX_6=/projects/viggiano/envs/bio_tools
CONDA_PROMPT_MODIFIER=(bio-tools) 
CONDA_PYTHON_EXE=/home/viggiano/miniconda3/bin/python
CONDA_SHLVL=7
DISABLE_PANDERA_IMPORT_WARNING=True
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
OLDPWD=/home/viggiano/main/codebases
PATH=/home/viggiano/.local/bin:/usr/local/cuda/bin:/opt/bin:/home/viggiano/.local/bin:/home/viggiano/.cargo/bin:/projects/viggiano/envs/bio-tools/bin:/home/viggiano/miniconda3/condabin:/usr/local/cuda/bin:...
PWD=/home/viggiano/main/codebases/bio-programming-tools
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
RDBASE=/projects/viggiano/envs/bio-tools/lib/python3.12/site-packages/rdkit
SHELL=/bin/bash
SHLVL=2
TERM=tmux-256color
TERM_PROGRAM=tmux
TERM_PROGRAM_VERSION=3.2a
TMUX=/tmp/tmux-1013/default,152215,0
TMUX_PANE=%0
USER=viggiano
XDG_DATA_DIRS=/usr/local/share:/usr/share:/var/lib/snapd/desktop
XDG_RUNTIME_DIR=/run/user/1013
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
_=/projects/viggiano/envs/bio-tools/bin/pytest
```

### Subprocess Environment (passed to tools)

```
CONDA_PREFIX=/raid/projects/viggiano/codebases/bio-programming-tools/tool_envs/splice_transformer_env
DETECTED_COMPUTE_PLATFORM=cuda
DETECTED_CUDA_VERSION=12
DETECTED_DRIVER_VERSION=535
HOME=/home/viggiano
LANG=en_US.UTF-8
LD_LIBRARY_PATH=/projects/viggiano/envs/bio-tools/lib
LOGNAME=viggiano
PATH=/raid/projects/viggiano/codebases/bio-programming-tools/tool_envs/splice_transformer_env/bin:/usr/local/cuda/bin:/home/viggiano/.local/bin:/opt/bin:/home/viggiano/.cargo/bin:/projects/viggiano/envs/bi...
RECOMMENDED_JAX_SPEC=jax[cuda12]>=0.4.20,<1
RECOMMENDED_JAX_VARIANT=cuda12
RECOMMENDED_TORCH_SPEC=torch>=2.4,<3
SHELL=/bin/bash
TORCH_CUDA_ARCH_LIST=9.0
TORCH_HOME=/raid/projects/viggiano/codebases/bio-programming-tools/tool_envs/splice_transformer_env/cache/torch
USER=viggiano
VIRTUAL_ENV=/raid/projects/viggiano/codebases/bio-programming-tools/tool_envs/splice_transformer_env
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Results by Category

### Causal Models (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `evo1` | yes | ✅ | 102.8s | ✅ Pass |
| `evo2` | yes | ✅ | 86.1s | ✅ Pass |
| `progen2` | yes | ✅ | 39.5s | ✅ Pass |

### Gene Annotation (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `blast` | no | ✅ | 28.7s | ✅ Pass |
| `minced` | no | ✅ | 6.2s | ✅ Pass |
| `mmseqs` | no | ✅ | 8.2s | ✅ Pass |
| `pyhmmer` | no | ✅ | 5.8s | ✅ Pass |

### Inverse Folding (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `ligandmpnn` | yes | ✅ | 36.7s | ✅ Pass |
| `proteinmpnn` | yes | ✅ | 34.0s | ✅ Pass |

### Masked Models (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `esm2` | yes | ✅ | 20.2s | ✅ Pass |
| `esm3` | yes | ✅ | 20.3s | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `orfipy` | no | ✅ | 6.4s | ✅ Pass |
| `prodigal` | no | ✅ | 5.8s | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `splice_transformer` | yes | ✅ | 17.5s | ✅ Pass |

### Sequence Alignment (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `mafft` | no | ✅ | 8.7s | ✅ Pass |

### Sequence Scoring (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `borzoi` | yes | ✅ | 37.1s | ✅ Pass |
| `enformer` | yes | ✅ | 18.5s | ✅ Pass |

### Structure Alignment (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `tmalign` | no | ✅ | 14.4s | ✅ Pass |
| `tmalign` | no | ✅ | 0.0s | ✅ Pass |
| `usalign` | no | ✅ | 25.0s | ✅ Pass |
| `usalign` | no | ✅ | 0.0s | ✅ Pass |

### Structure Design (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `rfdiffusion3` | yes | ✅ | 43.9s | ✅ Pass |

### Structure Dynamics (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `bioemu` | yes | ✅ | 34.6s | ✅ Pass |

### Structure Prediction (6/6)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `alphafold2` | yes | ✅ | 92.2s | ✅ Pass |
| `boltz2` | yes | ✅ | 88.2s | ✅ Pass |
| `chai1` | yes | ✅ | 268.2s | ✅ Pass |
| `esmfold` | yes | ✅ | 29.2s | ✅ Pass |
| `protenix` | yes | ✅ | 302.1s | ✅ Pass |
| `viennarna` | no | ✅ | 5.8s | ✅ Pass |

### Unknown (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `alphagenome` | yes | ✅ | 134.6s | ✅ Pass |
| `crispr_tracr` | no | ✅ | 86.8s | ✅ Pass |
| `local_colabfold_search` | no | — | 37.0s | ✅ Pass |
| `structure_metrics` | no | ✅ | 6.4s | ✅ Pass |

---
*Generated at 2026-03-01 15:58:02 by `pytest --env-report`*