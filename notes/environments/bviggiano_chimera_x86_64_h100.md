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
| **Mamba Env** | `bio-tools` |

## Git

- **Commit**: `5c8918c9d60d`
- **Branch**: `device_management`
- **Dirty**: No

## Environment Variables

### Parent Process Environment

```
ADDR2LINE=x86_64-conda-linux-gnu-addr2line
AR=x86_64-conda-linux-gnu-ar
AS=x86_64-conda-linux-gnu-as
BLASTDB=/common_datasets/external/databases/blast
BUILD=x86_64-conda-linux-gnu
CC=/home/bviggiano/miniforge3/envs/bio-tools/bin/x86_64-conda-linux-gnu-cc
CC_FOR_BUILD=/home/bviggiano/miniforge3/envs/bio-tools/bin/x86_64-conda-linux-gnu-cc
CFLAGS=-march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem /home/bviggiano/miniforge3/envs/bio-tools/include  -I/home/bviggiano/minifo...
CMAKE_ARGS=-DCMAKE_AR=/home/bviggiano/miniforge3/envs/bio-tools/bin/x86_64-conda-linux-gnu-ar -DCMAKE_CXX_COMPILER_AR=/home/bviggiano/miniforge3/envs/bio-tools/bin/x86_64-conda-linux-gnu-gcc-ar -DCMAKE_C_COMPILE...
CMAKE_PREFIX_PATH=/home/bviggiano/miniforge3/envs/bio-tools:/home/bviggiano/miniforge3/envs/bio-tools/x86_64-conda-linux-gnu/sysroot/usr
CONDA_BUILD_CROSS_COMPILATION=
CONDA_BUILD_SYSROOT=/home/bviggiano/miniforge3/envs/bio-tools/x86_64-conda-linux-gnu/sysroot
CONDA_DEFAULT_ENV=bio-tools
CONDA_EXE=/home/bviggiano/miniforge3/bin/conda
CONDA_PREFIX=/home/bviggiano/miniforge3/envs/bio-tools
CONDA_PREFIX_1=/home/bviggiano/miniforge3
CONDA_PREFIX_2=/home/bviggiano/miniforge3/envs/bio_tools
CONDA_PROMPT_MODIFIER=(bio-tools) 
CONDA_PYTHON_EXE=/home/bviggiano/miniforge3/bin/python
CONDA_SHLVL=3
CONDA_TOOLCHAIN_BUILD=x86_64-conda-linux-gnu
CONDA_TOOLCHAIN_HOST=x86_64-conda-linux-gnu
CPP=/home/bviggiano/miniforge3/envs/bio-tools/bin/x86_64-conda-linux-gnu-cpp
CPPFLAGS=-DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem /home/bviggiano/miniforge3/envs/bio-tools/include  -I/home/bviggiano/miniforge3/envs/bio-tools/targets/x86_64-linux/include
CPP_FOR_BUILD=/home/bviggiano/miniforge3/envs/bio-tools/bin/x86_64-conda-linux-gnu-cpp
CUDA_HOME=/home/bviggiano/miniforge3/envs/bio-tools
CUDA_VISIBLE_DEVICES=0
CXX=/home/bviggiano/miniforge3/envs/bio-tools/bin/x86_64-conda-linux-gnu-c++
CXXFILT=x86_64-conda-linux-gnu-c++filt
CXXFLAGS=-fvisibility-inlines-hidden -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem /home/bviggiano/miniforge3/e...
CXX_FOR_BUILD=/home/bviggiano/miniforge3/envs/bio-tools/bin/x86_64-conda-linux-gnu-c++
DEBUG_CFLAGS=-march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-all -fno-plt -Og -g -Wall -Wextra -fvar-tracking-assignments -ffunction-sections -pipe -isystem /home/bviggiano/miniforge3/envs/bi...
DEBUG_CPPFLAGS=-D_DEBUG -D_FORTIFY_SOURCE=2 -Og -isystem /home/bviggiano/miniforge3/envs/bio-tools/include
DEBUG_CXXFLAGS=-fvisibility-inlines-hidden -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-all -fno-plt -Og -g -Wall -Wextra -fvar-tracking-assignments -ffunction-sections -p...
DISABLE_PANDERA_IMPORT_WARNING=True
ELFEDIT=x86_64-conda-linux-gnu-elfedit
GCC=/home/bviggiano/miniforge3/envs/bio-tools/bin/x86_64-conda-linux-gnu-gcc
GCC_AR=/home/bviggiano/miniforge3/envs/bio-tools/bin/x86_64-conda-linux-gnu-gcc-ar
GCC_NM=/home/bviggiano/miniforge3/envs/bio-tools/bin/x86_64-conda-linux-gnu-gcc-nm
GCC_RANLIB=/home/bviggiano/miniforge3/envs/bio-tools/bin/x86_64-conda-linux-gnu-gcc-ranlib
GPROF=x86_64-conda-linux-gnu-gprof
GPU_DEVICE_ORDINAL=0
GXX=/home/bviggiano/miniforge3/envs/bio-tools/bin/x86_64-conda-linux-gnu-g++
HOME=/home/bviggiano
HOST=x86_64-conda-linux-gnu
HYDRA_BOOTSTRAP=slurm
HYDRA_LAUNCHER_EXTRA_ARGS=--external-launcher
I_MPI_HYDRA_BOOTSTRAP=slurm
I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS=--external-launcher
LANG=en_US.UTF-8
LD=x86_64-conda-linux-gnu-ld
LDFLAGS=-Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,--allow-shlib-undefined -Wl,-rpath,/home/bviggiano/miniforge3/envs/bio-tools/lib -Wl,-rp...
LDFLAGS_LD=-O2 --sort-common --as-needed -z relro -z now --disable-new-dtags --gc-sections --allow-shlib-undefined -rpath /home/bviggiano/miniforge3/envs/bio-tools/lib -rpath-link /home/bviggiano/miniforge3/envs...
LESSCLOSE=/usr/bin/lesspipe %s %s
LESSOPEN=| /usr/bin/lesspipe %s
LOGNAME=bviggiano
LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.a...
MAMBA_EXE=/home/bviggiano/miniforge3/bin/mamba
MAMBA_ROOT_PREFIX=/home/bviggiano/.local/share/mamba
MESON_ARGS=-Dbuildtype=release
MOTD_SHOWN=pam
NM=x86_64-conda-linux-gnu-nm
NVCC_PREPEND_FLAGS= -ccbin=/home/bviggiano/miniforge3/envs/bio-tools/bin/x86_64-conda-linux-gnu-c++
OBJCOPY=x86_64-conda-linux-gnu-objcopy
OBJDUMP=x86_64-conda-linux-gnu-objdump
OLDPWD=/home/bviggiano/main/codebases/bio-programming
OMPI_MCA_plm_slurm_args=--external-launcher
PATH=/home/bviggiano/.local/bin:/home/bviggiano/bin:/home/bviggiano/.local/bin:/home/bviggiano/bin:/home/bviggiano/miniforge3/envs/bio-tools/bin:/home/bviggiano/miniforge3/condabin:/usr/local/sbin:/usr/loc...
PRTE_MCA_plm_slurm_args=--external-launcher
PWD=/home/bviggiano/main/codebases/bio-programming/bio-programming-tools
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
RANLIB=x86_64-conda-linux-gnu-ranlib
RDBASE=/home/bviggiano/miniforge3/envs/bio-tools/lib/python3.12/site-packages/rdkit
READELF=x86_64-conda-linux-gnu-readelf
ROCR_VISIBLE_DEVICES=0
SHELL=/bin/bash
SHLVL=5
SIZE=x86_64-conda-linux-gnu-size
SLURMD_DEBUG=2
SLURMD_NODENAME=GPU71E4
SLURM_CLUSTER_NAME=arc-slurm
SLURM_CONF=/etc/slurm/slurm.conf
SLURM_CPUS_ON_NODE=8
SLURM_GPUS=1
SLURM_GPUS_ON_NODE=1
SLURM_GTIDS=0
SLURM_JOBID=1731344
SLURM_JOB_ACCOUNT=hielab
SLURM_JOB_CPUS_PER_NODE=8
SLURM_JOB_END_TIME=1772429802
SLURM_JOB_GID=10004
SLURM_JOB_GPUS=1
SLURM_JOB_ID=1731344
SLURM_JOB_NAME=1_sh_gpu
SLURM_JOB_NODELIST=GPU71E4
SLURM_JOB_NUM_NODES=1
SLURM_JOB_PARTITION=evo_gpu_priority
SLURM_JOB_QOS=normal
SLURM_JOB_START_TIME=1772386602
SLURM_JOB_UID=10249
SLURM_JOB_USER=bviggiano
SLURM_LAUNCH_NODE_IPADDR=172.18.140.10
SLURM_LOCALID=0
SLURM_MPI_TYPE=pmix
SLURM_NNODES=1
SLURM_NODEID=0
SLURM_NODELIST=GPU71E4
SLURM_OOM_KILL_STEP=0
SLURM_PMIXP_ABORT_AGENT_PORT=40839
SLURM_PMIX_MAPPING_SERV=(vector,(0,1,1))
SLURM_PRIO_PROCESS=0
SLURM_PROCID=0
SLURM_PTY_PORT=39933
SLURM_PTY_WIN_COL=209
SLURM_PTY_WIN_ROW=43
SLURM_SRUN_COMM_HOST=172.18.140.10
SLURM_SRUN_COMM_PORT=37845
SLURM_STEPID=4294967290
SLURM_STEP_ID=4294967290
SLURM_STEP_LAUNCHER_PORT=37845
SLURM_STEP_NODELIST=GPU71E4
SLURM_STEP_NUM_NODES=1
SLURM_STEP_NUM_TASKS=1
SLURM_STEP_TASKS_PER_NODE=1
SLURM_SUBMIT_DIR=/home/bviggiano
SLURM_SUBMIT_HOST=arc-slurm
SLURM_TASKS_PER_NODE=8
SLURM_TASK_PID=2937593
SLURM_TOPOLOGY_ADDR=GPU71E4
SLURM_TOPOLOGY_ADDR_PATTERN=node
SRUN_DEBUG=3
STRINGS=x86_64-conda-linux-gnu-strings
STRIP=x86_64-conda-linux-gnu-strip
TERM=screen
TERM_PROGRAM=tmux
TERM_PROGRAM_VERSION=3.2a
TMPDIR=/tmp
TMUX=/tmp/tmux-10249/default,1122679,5
TMUX_PANE=%5
USER=bviggiano
XDG_DATA_DIRS=/usr/local/share:/usr/share:/var/lib/snapd/desktop
XDG_RUNTIME_DIR=/run/user/10249
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
XML_CATALOG_FILES=file:///home/bviggiano/miniforge3/etc/xml/catalog file:///etc/xml/catalog file:///home/bviggiano/miniforge3/etc/xml/catalog file:///etc/xml/catalog file:///home/bviggiano/miniforge3/etc/xml/catalog fi...
ZES_ENABLE_SYSMAN=1
ZE_AFFINITY_MASK=0
_=/home/bviggiano/miniforge3/envs/bio-tools/bin/pytest
_CE_CONDA=
_CE_M=
_CONDA_EXE=/home/bviggiano/miniforge3/bin/conda
_CONDA_ROOT=/home/bviggiano/miniforge3
build_alias=x86_64-conda-linux-gnu
host_alias=x86_64-conda-linux-gnu
```

### Subprocess Environment (passed to tools)

```
CONDA_PREFIX=/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/splice_transformer_env
CUDA_VISIBLE_DEVICES=0
DETECTED_COMPUTE_PLATFORM=cuda
DETECTED_CUDA_VERSION=12
DETECTED_DRIVER_VERSION=535
HOME=/home/bviggiano
LANG=en_US.UTF-8
LD_LIBRARY_PATH=/home/bviggiano/miniforge3/envs/bio-tools/lib
LOGNAME=bviggiano
PATH=/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/splice_transformer_env/bin:/usr/local/cuda/bin:/home/bviggiano/.local/bin:/home/bviggiano/bin:/home/bviggiano/...
RECOMMENDED_JAX_SPEC=jax[cuda12]>=0.4.20,<1
RECOMMENDED_JAX_VARIANT=cuda12
RECOMMENDED_TORCH_SPEC=torch>=2.4,<3
SHELL=/bin/bash
TMPDIR=/tmp
TORCH_CUDA_ARCH_LIST=9.0
TORCH_HOME=/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/splice_transformer_env/cache/torch
USER=bviggiano
VIRTUAL_ENV=/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/splice_transformer_env
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Results by Category

### Causal Models (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `evo1` | yes | ✅ | 310.6s | ✅ Pass |
| `evo2` | yes | ✅ | 322.5s | ✅ Pass |
| `progen2` | yes | ✅ | 192.3s | ✅ Pass |

### Gene Annotation (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `blast` | no | ✅ | 56.0s | ✅ Pass |
| `minced` | no | ✅ | 21.3s | ✅ Pass |
| `mmseqs` | no | ✅ | 26.0s | ✅ Pass |
| `pyhmmer` | no | ✅ | 23.9s | ✅ Pass |

### Inverse Folding (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `ligandmpnn` | yes | ✅ | 185.6s | ✅ Pass |
| `proteinmpnn` | yes | ✅ | 65.7s | ✅ Pass |

### Masked Models (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `esm2` | yes | ✅ | 100.7s | ✅ Pass |
| `esm3` | yes | ✅ | 110.0s | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `orfipy` | no | ✅ | 25.3s | ✅ Pass |
| `prodigal` | no | ✅ | 17.5s | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `splice_transformer` | yes | ✅ | 85.5s | ✅ Pass |

### Sequence Alignment (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `mafft` | no | ✅ | 29.0s | ✅ Pass |

### Sequence Scoring (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `borzoi` | yes | ✅ | 119.7s | ✅ Pass |
| `enformer` | yes | ✅ | 100.7s | ✅ Pass |

### Structure Alignment (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `tmalign` | no | ✅ | 25.9s | ✅ Pass |
| `tmalign` | no | ✅ | 0.1s | ✅ Pass |
| `usalign` | no | ✅ | 35.5s | ✅ Pass |
| `usalign` | no | ✅ | 0.1s | ✅ Pass |

### Structure Design (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `rfdiffusion3` | yes | ✅ | 221.8s | ✅ Pass |

### Structure Dynamics (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `bioemu` | yes | ✅ | 133.8s | ✅ Pass |

### Structure Prediction (7/7)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `alphafold2` | yes | ✅ | 121.2s | ✅ Pass |
| `alphafold3` | yes | ✅ | 142.3s | ✅ Pass |
| `boltz2` | yes | ✅ | 189.0s | ✅ Pass |
| `chai1` | yes | ✅ | 506.1s | ✅ Pass |
| `esmfold` | yes | ✅ | 126.0s | ✅ Pass |
| `protenix` | yes | ✅ | 448.3s | ✅ Pass |
| `viennarna` | no | ✅ | 27.7s | ✅ Pass |

### Unknown (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `alphagenome` | yes | ✅ | 265.6s | ✅ Pass |
| `crispr_tracr` | no | ✅ | 148.6s | ✅ Pass |
| `local_colabfold_search` | no | — | 127.2s | ✅ Pass |
| `structure_metrics` | no | ✅ | 34.0s | ✅ Pass |

---
*Generated at 2026-03-01 16:52:27 by `pytest --env-report`*