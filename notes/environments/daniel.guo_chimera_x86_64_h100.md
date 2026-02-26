# Chimera Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-100%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-34-brightgreen) ![Failed](https://img.shields.io/badge/failed-0-red) ![Skipped](https://img.shields.io/badge/skipped-0-lightgrey)

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
| **Conda Env** | `bio-programming` |

## Git

- **Commit**: `46bc099e9f0e`
- **Branch**: `dguo/sherlock-fixes`
- **Dirty**: Yes

## Environment Variables

### Parent Process Environment

```
BLASTDB=/common_datasets/external/databases/blast
CONDA_DEFAULT_ENV=bio-programming
CONDA_EXE=/home/daniel.guo/miniconda/bin/conda
CONDA_PREFIX=/home/daniel.guo/miniconda/envs/bio-programming
CONDA_PREFIX_1=/home/daniel.guo/miniconda
CONDA_PROMPT_MODIFIER=(bio-programming) 
CONDA_PYTHON_EXE=/home/daniel.guo/miniconda/bin/python
CONDA_SHLVL=2
CUDA_HOME=/home/daniel.guo/miniconda
CUDA_VISIBLE_DEVICES=0
CUDNN_INCLUDE_DIR=/home/daniel.guo/miniconda/lib/python3.11/site-packages/nvidia/cudnn/include
CUDNN_LIBRARY_DIR=/home/daniel.guo/miniconda/lib/python3.11/site-packages/nvidia/cudnn/lib
DISABLE_PANDERA_IMPORT_WARNING=True
GPU_DEVICE_ORDINAL=0
HOME=/home/daniel.guo
HYDRA_BOOTSTRAP=slurm
HYDRA_LAUNCHER_EXTRA_ARGS=--external-launcher
I_MPI_HYDRA_BOOTSTRAP=slurm
I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS=--external-launcher
LANG=en_US.UTF-8
LESSCLOSE=/usr/bin/lesspipe %s %s
LESSOPEN=| /usr/bin/lesspipe %s
LOGNAME=daniel.guo
LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.a...
MOTD_SHOWN=pam
OLDPWD=/home/daniel.guo
OMPI_MCA_plm_slurm_args=--external-launcher
PATH=/home/daniel.guo/.local/bin:/home/daniel.guo/miniconda/envs/bio-programming/bin:/home/daniel.guo/miniconda/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/g...
PRTE_MCA_plm_slurm_args=--external-launcher
PWD=/home/daniel.guo/bio-programming/bio-programming-tools
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
RDBASE=/home/daniel.guo/miniconda/envs/bio-programming/lib/python3.12/site-packages/rdkit
ROCR_VISIBLE_DEVICES=0
SHELL=/bin/bash
SHLVL=3
SLURMD_DEBUG=2
SLURMD_NODENAME=GPU71E4
SLURM_CLUSTER_NAME=arc-slurm
SLURM_CONF=/etc/slurm/slurm.conf
SLURM_CPUS_ON_NODE=8
SLURM_GPUS=1
SLURM_GPUS_ON_NODE=1
SLURM_GTIDS=0
SLURM_JOBID=1717712
SLURM_JOB_ACCOUNT=hielab
SLURM_JOB_CPUS_PER_NODE=8
SLURM_JOB_END_TIME=1772124166
SLURM_JOB_GID=10004
SLURM_JOB_GPUS=2
SLURM_JOB_ID=1717712
SLURM_JOB_NAME=1_sh_gpu
SLURM_JOB_NODELIST=GPU71E4
SLURM_JOB_NUM_NODES=1
SLURM_JOB_PARTITION=evo_gpu_priority
SLURM_JOB_QOS=normal
SLURM_JOB_START_TIME=1772080966
SLURM_JOB_UID=10085
SLURM_JOB_USER=daniel.guo
SLURM_LAUNCH_NODE_IPADDR=172.18.140.10
SLURM_LOCALID=0
SLURM_MPI_TYPE=pmix
SLURM_NNODES=1
SLURM_NODEID=0
SLURM_NODELIST=GPU71E4
SLURM_OOM_KILL_STEP=0
SLURM_PMIXP_ABORT_AGENT_PORT=42505
SLURM_PMIX_MAPPING_SERV=(vector,(0,1,1))
SLURM_PRIO_PROCESS=0
SLURM_PROCID=0
SLURM_PTY_PORT=39029
SLURM_PTY_WIN_COL=191
SLURM_PTY_WIN_ROW=53
SLURM_SRUN_COMM_HOST=172.18.140.10
SLURM_SRUN_COMM_PORT=35079
SLURM_STEPID=4294967290
SLURM_STEP_ID=4294967290
SLURM_STEP_LAUNCHER_PORT=35079
SLURM_STEP_NODELIST=GPU71E4
SLURM_STEP_NUM_NODES=1
SLURM_STEP_NUM_TASKS=1
SLURM_STEP_TASKS_PER_NODE=1
SLURM_SUBMIT_DIR=/home/daniel.guo
SLURM_SUBMIT_HOST=arc-slurm
SLURM_TASKS_PER_NODE=8
SLURM_TASK_PID=3920348
SLURM_TOPOLOGY_ADDR=GPU71E4
SLURM_TOPOLOGY_ADDR_PATTERN=node
SRUN_DEBUG=3
TERM=xterm-256color
TERM_PROGRAM=WarpTerminal
TMPDIR=/tmp
USER=daniel.guo
XDG_DATA_DIRS=/usr/local/share:/usr/share:/var/lib/snapd/desktop
XDG_RUNTIME_DIR=/run/user/10085
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
ZES_ENABLE_SYSMAN=1
ZE_AFFINITY_MASK=0
_=/home/daniel.guo/miniconda/envs/bio-programming/bin/pytest
_CE_CONDA=
_CE_M=
```

### Subprocess Environment (passed to tools)

```
CONDA_DEFAULT_ENV=bio-programming
CONDA_PREFIX=/home/daniel.guo/miniconda/envs/bio-programming
CONDA_SHLVL=2
CUDA_VISIBLE_DEVICES=0
DETECTED_COMPUTE_PLATFORM=cuda
DETECTED_CUDA_VERSION=12
DETECTED_DRIVER_VERSION=535
HOME=/home/daniel.guo
LANG=en_US.UTF-8
LOGNAME=daniel.guo
PATH=/home/daniel.guo/bio-programming/bio-programming-tools/tool_envs/splice_transformer_env/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RECOMMENDED_JAX_SPEC=jax[cuda12]>=0.4.20,<1
RECOMMENDED_JAX_VARIANT=cuda12
RECOMMENDED_TORCH_SPEC=torch>=2.4,<3
SHELL=/bin/bash
TMPDIR=/tmp
TORCH_CUDA_ARCH_LIST=9.0
TORCH_HOME=/home/daniel.guo/bio-programming/bio-programming-tools/tool_envs/splice_transformer_env/cache/torch
USER=daniel.guo
```

## Results by Category

### Causal Models (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `evo1` | yes | ✅ | 246.3s | ✅ Pass |
| `evo2` | yes | ✅ | 255.0s | ✅ Pass |
| `progen2` | yes | ✅ | 106.6s | ✅ Pass |

### Gene Annotation (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `blast` | no | ✅ | 61.1s | ✅ Pass |
| `minced` | no | ✅ | 21.4s | ✅ Pass |
| `mmseqs` | no | ✅ | 24.1s | ✅ Pass |
| `pyhmmer` | no | ✅ | 23.6s | ✅ Pass |

### Inverse Folding (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `ligandmpnn` | yes | ✅ | 76.2s | ✅ Pass |
| `proteinmpnn` | yes | ✅ | 40.4s | ✅ Pass |

### Masked Models (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `esm2` | yes | ✅ | 50.5s | ✅ Pass |
| `esm3` | yes | ✅ | 52.0s | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `orfipy` | no | ✅ | 22.5s | ✅ Pass |
| `prodigal` | no | ✅ | 30.5s | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `splice_transformer` | yes | ✅ | 39.7s | ✅ Pass |

### Sequence Alignment (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `mafft` | no | ✅ | 27.2s | ✅ Pass |

### Sequence Scoring (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `borzoi` | yes | ✅ | 78.3s | ✅ Pass |
| `enformer` | yes | ✅ | 43.3s | ✅ Pass |

### Structure Alignment (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `tmalign` | no | ✅ | 28.8s | ✅ Pass |
| `tmalign` | no | ✅ | 0.1s | ✅ Pass |
| `usalign` | no | ✅ | 38.6s | ✅ Pass |
| `usalign` | no | ✅ | 0.1s | ✅ Pass |

### Structure Design (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `rfdiffusion3` | yes | ✅ | 270.9s | ✅ Pass |

### Structure Dynamics (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `bioemu` | yes | ✅ | 76.6s | ✅ Pass |

### Structure Prediction (7/7)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `alphafold2` | yes | ✅ | 845.9s | ✅ Pass |
| `alphafold3` | yes | — | 58.7s | ✅ Pass |
| `boltz2` | yes | ✅ | 119.8s | ✅ Pass |
| `chai1` | yes | ✅ | 363.1s | ✅ Pass |
| `esmfold` | yes | ✅ | 64.9s | ✅ Pass |
| `protenix` | yes | ✅ | 365.4s | ✅ Pass |
| `viennarna` | no | ✅ | 20.6s | ✅ Pass |

### Unknown (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `alphagenome` | yes | ✅ | 179.3s | ✅ Pass |
| `crispr_tracr` | no | ✅ | 238.9s | ✅ Pass |
| `local_colabfold_search` | no | — | 68.9s | ✅ Pass |
| `structure_metrics` | no | ✅ | 23.1s | ✅ Pass |

---
*Generated at 2026-02-25 22:01:59 by `pytest --env-report`*