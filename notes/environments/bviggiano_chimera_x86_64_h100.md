# Linux x86_64 Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-100%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-29-brightgreen) ![Failed](https://img.shields.io/badge/failed-0-lightgrey) ![Skipped](https://img.shields.io/badge/skipped-0-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Linux Linux 5.15.0-164-generic |
| **Architecture** | x86_64 |
| **Hostname** | `GPU71E4` |
| **Python** | 3.12.12 |
| **RAM** | 1007.4 GB |
| **GPU** | 1× NVIDIA H100 80GB HBM3 |
| **CUDA** | 12.2 |
| **Mamba Env** | `bio_tools` |

## Git

- **Commit**: `e35a0bd6e3c5`
- **Branch**: `bv/deps_iso_pass`
- **Dirty**: No

## Environment Variables

### Parent Process Environment

```
BLASTDB=/common_datasets/external/databases/blast
CONDA_DEFAULT_ENV=bio_tools
CONDA_EXE=/home/bviggiano/miniforge3/bin/conda
CONDA_PREFIX=/home/bviggiano/miniforge3/envs/bio_tools
CONDA_PREFIX_1=/home/bviggiano/miniforge3
CONDA_PROMPT_MODIFIER=(bio_tools) 
CONDA_PYTHON_EXE=/home/bviggiano/miniforge3/bin/python
CONDA_SHLVL=2
DISABLE_PANDERA_IMPORT_WARNING=True
HOME=/home/bviggiano
LANG=C.UTF-8
LD_LIBRARY_PATH=/usr/local/cuda/lib64
LESSCLOSE=/usr/bin/lesspipe %s %s
LESSOPEN=| /usr/bin/lesspipe %s
LOGNAME=bviggiano
LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.a...
MAMBA_EXE=/home/bviggiano/miniforge3/bin/mamba
MAMBA_ROOT_PREFIX=/home/bviggiano/.local/share/mamba
MOTD_SHOWN=pam
OLDPWD=/home/bviggiano/main/codebases/bio-programming
PATH=/home/bviggiano/.local/bin:/home/bviggiano/bin:/home/bviggiano/miniforge3/envs/bio_tools/bin:/home/bviggiano/miniforge3/condabin:/usr/local/ncbi/sra-tools/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/...
PWD=/home/bviggiano/main/codebases/bio-programming/bio-programming-tools
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
RDBASE=/home/bviggiano/miniforge3/envs/bio_tools/lib/python3.12/site-packages/rdkit
SHELL=/bin/bash
SHLVL=1
SLURM_JOB_ID=1708377
TERM=xterm-256color
USER=bviggiano
XDG_DATA_DIRS=/usr/local/share:/usr/share:/var/lib/snapd/desktop
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
XML_CATALOG_FILES=file:///home/bviggiano/miniforge3/etc/xml/catalog file:///etc/xml/catalog
_=/home/bviggiano/miniforge3/envs/bio_tools/bin/pytest
_CE_CONDA=
_CE_M=
_CONDA_EXE=/home/bviggiano/miniforge3/bin/conda
_CONDA_ROOT=/home/bviggiano/miniforge3
```

### Subprocess Environment (passed to tools)

```
CONDA_DEFAULT_ENV=bio_tools
CONDA_PREFIX=/home/bviggiano/miniforge3/envs/bio_tools
CONDA_SHLVL=2
CUDA_VISIBLE_DEVICES=0
DETECTED_COMPUTE_PLATFORM=cuda
DETECTED_CUDA_VERSION=12
DETECTED_DRIVER_VERSION=535
HOME=/home/bviggiano
LANG=C.UTF-8
LD_LIBRARY_PATH=/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/splice_transformer_env/lib/python3.1/site-packages/nvidia/cufft/lib:/large_storage/hielab/bviggiano/codebases/...
LOGNAME=bviggiano
PATH=/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/splice_transformer_env/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RECOMMENDED_JAX_SPEC=jax[cuda12]>=0.4.20,<1
RECOMMENDED_JAX_VARIANT=cuda12
RECOMMENDED_TORCH_SPEC=torch>=2.4,<3
SHELL=/bin/bash
TORCH_HOME=/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/splice_transformer_env/cache/torch
USER=bviggiano
```

## Results by Category

### Causal Models (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `evo1` | yes | ✅ | 117.5s | ✅ Pass |
| `evo2` | yes | ✅ | 230.4s | ✅ Pass |
| `progen2` | yes | ✅ | 79.9s | ✅ Pass |

### Gene Annotation (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `blast` | no | ✅ | 53.0s | ✅ Pass |
| `minced` | no | ✅ | 20.8s | ✅ Pass |
| `mmseqs` | no | ✅ | 23.1s | ✅ Pass |
| `pyhmmer` | no | ✅ | 21.7s | ✅ Pass |

### Inverse Folding (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `ligandmpnn` | yes | ✅ | 104.1s | ✅ Pass |
| `proteinmpnn` | yes | ✅ | 41.2s | ✅ Pass |

### Masked Models (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `esm2` | yes | ✅ | 60.1s | ✅ Pass |
| `esm3` | yes | ✅ | 59.4s | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `orfipy` | no | ✅ | 24.1s | ✅ Pass |
| `prodigal` | no | ✅ | 20.5s | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `splice_transformer` | yes | ✅ | 49.5s | ✅ Pass |

### Sequence Alignment (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `mafft` | no | ✅ | 25.5s | ✅ Pass |

### Sequence Scoring (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `borzoi` | yes | ✅ | 78.5s | ✅ Pass |
| `enformer` | yes | ✅ | 55.3s | ✅ Pass |

### Structure Design (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `rfdiffusion3` | yes | ✅ | 130.7s | ✅ Pass |

### Structure Dynamics (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `bioemu` | no | — | 0.0s | ✅ Pass |

### Structure Prediction (6/6)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `alphafold3` | yes | — | 61.7s | ✅ Pass |
| `boltz2` | yes | ✅ | 153.2s | ✅ Pass |
| `chai1` | yes | ✅ | 227.5s | ✅ Pass |
| `esmfold` | yes | ✅ | 70.6s | ✅ Pass |
| `protenix` | yes | ✅ | 434.9s | ✅ Pass |
| `viennarna` | no | ✅ | 23.1s | ✅ Pass |

### Unknown (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `alphagenome` | yes | ✅ | 230.5s | ✅ Pass |
| `crispr_tracr` | no | ✅ | 158.3s | ✅ Pass |
| `local_colabfold_search` | no | — | 84.3s | ✅ Pass |
| `structure_metrics` | no | ✅ | 22.2s | ✅ Pass |

---
*Generated at 2026-02-20 23:36:47 by `pytest --env-report`*