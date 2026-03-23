# Chimera Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-97%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-38-brightgreen) ![Failed](https://img.shields.io/badge/failed-0-red) ![Skipped](https://img.shields.io/badge/skipped-1-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Linux Linux 5.15.0-171-generic |
| **Architecture** | x86_64 |
| **Hostname** | `GPUC960` |
| **Python** | 3.12.12 |
| **RAM** | 1007.4 GB |
| **GPU** | 1× NVIDIA H100 80GB HBM3 |
| **CUDA** | 12.2 |
| **Mamba Env** | `bio-tools` |

## Git

- **Commit**: `ccb2a62f10bc`
- **Branch**: `bv/checkpoint_locations`
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
CLAUDECODE=1
CLAUDE_CODE_ENTRYPOINT=cli
CMAKE_ARGS=-DCMAKE_AR=/home/bviggiano/miniforge3/envs/bio-tools/bin/x86_64-conda-linux-gnu-ar -DCMAKE_CXX_COMPILER_AR=/home/bviggiano/miniforge3/envs/bio-tools/bin/x86_64-conda-linux-gnu-gcc-ar -DCMAKE_C_COMPILE...
CMAKE_PREFIX_PATH=/home/bviggiano/miniforge3/envs/bio-tools:/home/bviggiano/miniforge3/envs/bio-tools/x86_64-conda-linux-gnu/sysroot/usr
CONDA_BUILD_CROSS_COMPILATION=
CONDA_BUILD_SYSROOT=/home/bviggiano/miniforge3/envs/bio-tools/x86_64-conda-linux-gnu/sysroot
CONDA_DEFAULT_ENV=bio-tools
CONDA_EXE=/home/bviggiano/miniforge3/bin/conda
CONDA_PREFIX=/home/bviggiano/miniforge3/envs/bio-tools
CONDA_PREFIX_1=/home/bviggiano/miniforge3
CONDA_PROMPT_MODIFIER=(bio-tools) 
CONDA_PYTHON_EXE=/home/bviggiano/miniforge3/bin/python
CONDA_SHLVL=2
CONDA_TOOLCHAIN_BUILD=x86_64-conda-linux-gnu
CONDA_TOOLCHAIN_HOST=x86_64-conda-linux-gnu
COREPACK_ENABLE_AUTO_PIN=0
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
GIT_EDITOR=true
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
NoDefaultCurrentDirectoryInExePath=1
OBJCOPY=x86_64-conda-linux-gnu-objcopy
OBJDUMP=x86_64-conda-linux-gnu-objdump
OLDPWD=/home/bviggiano/main
OMPI_MCA_plm_slurm_args=--external-launcher
OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta
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
SHLVL=6
SIZE=x86_64-conda-linux-gnu-size
SLURMD_DEBUG=2
SLURMD_NODENAME=GPUC960
SLURM_CLUSTER_NAME=arc-slurm
SLURM_CONF=/etc/slurm/slurm.conf
SLURM_CPUS_ON_NODE=8
SLURM_GPUS=1
SLURM_GPUS_ON_NODE=1
SLURM_GTIDS=0
SLURM_JOBID=1884900
SLURM_JOB_ACCOUNT=hielab
SLURM_JOB_CPUS_PER_NODE=8
SLURM_JOB_END_TIME=1774317568
SLURM_JOB_GID=10004
SLURM_JOB_GPUS=0
SLURM_JOB_ID=1884900
SLURM_JOB_NAME=1_sh_gpu
SLURM_JOB_NODELIST=GPUC960
SLURM_JOB_NUM_NODES=1
SLURM_JOB_PARTITION=evo_gpu_priority
SLURM_JOB_QOS=normal
SLURM_JOB_START_TIME=1774274368
SLURM_JOB_UID=10249
SLURM_JOB_USER=bviggiano
SLURM_LAUNCH_NODE_IPADDR=172.18.140.10
SLURM_LOCALID=0
SLURM_MPI_TYPE=pmix
SLURM_NNODES=1
SLURM_NODEID=0
SLURM_NODELIST=GPUC960
SLURM_OOM_KILL_STEP=0
SLURM_PMIXP_ABORT_AGENT_PORT=45581
SLURM_PMIX_MAPPING_SERV=(vector,(0,1,1))
SLURM_PRIO_PROCESS=0
SLURM_PROCID=0
SLURM_PTY_PORT=34811
SLURM_PTY_WIN_COL=212
SLURM_PTY_WIN_ROW=54
SLURM_SRUN_COMM_HOST=172.18.140.10
SLURM_SRUN_COMM_PORT=40119
SLURM_STEPID=4294967290
SLURM_STEP_ID=4294967290
SLURM_STEP_LAUNCHER_PORT=40119
SLURM_STEP_NODELIST=GPUC960
SLURM_STEP_NUM_NODES=1
SLURM_STEP_NUM_TASKS=1
SLURM_STEP_TASKS_PER_NODE=1
SLURM_SUBMIT_DIR=/home/bviggiano
SLURM_SUBMIT_HOST=arc-slurm
SLURM_TASKS_PER_NODE=8
SLURM_TASK_PID=3031605
SLURM_TOPOLOGY_ADDR=GPUC960
SLURM_TOPOLOGY_ADDR_PATTERN=node
SRUN_DEBUG=3
STRINGS=x86_64-conda-linux-gnu-strings
STRIP=x86_64-conda-linux-gnu-strip
TERM=xterm-256color
TERM_PROGRAM=tmux
TERM_PROGRAM_VERSION=3.2a
TMPDIR=/tmp
TMUX=/tmp/tmux-10249/default,1505081,0
TMUX_PANE=%0
USER=bviggiano
XDG_DATA_DIRS=/usr/local/share:/usr/share:/var/lib/snapd/desktop
XDG_RUNTIME_DIR=/run/user/10249
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
XML_CATALOG_FILES=file:///home/bviggiano/miniforge3/etc/xml/catalog file:///etc/xml/catalog file:///home/bviggiano/miniforge3/etc/xml/catalog file:///etc/xml/catalog file:///home/bviggiano/miniforge3/etc/xml/catalog fi...
ZES_ENABLE_SYSMAN=1
ZE_AFFINITY_MASK=0
_=/home/bviggiano/miniforge3/envs/bio-tools/bin/python3
_CE_CONDA=
_CE_M=
_CONDA_EXE=/home/bviggiano/miniforge3/bin/conda
_CONDA_ROOT=/home/bviggiano/miniforge3
build_alias=x86_64-conda-linux-gnu
host_alias=x86_64-conda-linux-gnu
```

### Subprocess Environment (passed to tools)

```
BPT_MODEL_CACHE=/home/bviggiano/main/models
CONDA_PREFIX=/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/mock_pytorch_tool_env
CUDA_VISIBLE_DEVICES=0
DETECTED_COMPUTE_PLATFORM=cuda
DETECTED_CUDA_VERSION=12
DETECTED_DRIVER_VERSION=535
HF_HOME=/home/bviggiano/main/models/huggingface
HOME=/home/bviggiano
LANG=en_US.UTF-8
LD_LIBRARY_PATH=/home/bviggiano/miniforge3/envs/bio-tools/lib
LOGNAME=bviggiano
PATH=/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/mock_pytorch_tool_env/bin:/usr/local/cuda/bin:/home/bviggiano/.local/bin:/home/bviggiano/bin:/home/bviggiano/m...
PIP_DEFAULT_TIMEOUT=300
RECOMMENDED_JAX_SPEC=jax[cuda12]>=0.4.20,<1
RECOMMENDED_JAX_VARIANT=cuda12
RECOMMENDED_TORCH_INDEX=https://download.pytorch.org/whl/cu126
RECOMMENDED_TORCH_SPEC=torch>=2.4,<3
SHELL=/bin/bash
TMPDIR=/tmp
TORCH_CUDA_ARCH_LIST=9.0
TORCH_HOME=/home/bviggiano/main/models/torch
USER=bviggiano
UV_HTTP_TIMEOUT=300
VIRTUAL_ENV=/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/mock_pytorch_tool_env
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Results by Category

### Causal Models (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `evo1` | yes | ✅ | 299.7s | ✅ Pass |
| `evo2` | yes | ✅ | 334.2s | ✅ Pass |
| `progen2` | yes | ✅ | 207.7s | ✅ Pass |

### Gene Annotation (5/5)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `blast` | no | ✅ | 50.3s | ✅ Pass |
| `crispr_tracr` | no | ✅ | 151.5s | ✅ Pass |
| `minced` | no | ✅ | 20.4s | ✅ Pass |
| `mmseqs` | no | ✅ | 25.9s | ✅ Pass |
| `pyhmmer` | no | ✅ | 24.1s | ✅ Pass |

### Inverse Folding (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `esm_if1` | yes | ✅ | 128.4s | ✅ Pass |
| `fampnn` | yes | ✅ | 254.1s | ✅ Pass |
| `ligandmpnn` | yes | ✅ | 209.7s | ✅ Pass |
| `proteinmpnn` | yes | ✅ | 95.0s | ✅ Pass |

### Masked Models (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `esm2` | yes | ✅ | 122.9s | ✅ Pass |
| `esm3` | yes | ✅ | 171.0s | ✅ Pass |

### Mock (1/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `mock_tools_env_report` | yes | — | 213.4s | ✅ Pass |
| `mock_tools_env_report` | yes | — | — | ⏭️ Skip |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `orfipy` | no | ✅ | 25.0s | ✅ Pass |
| `prodigal` | no | ✅ | 15.6s | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `splice_transformer` | yes | ✅ | 93.6s | ✅ Pass |

### Sequence Alignment (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `local_colabfold_search` | no | — | 141.8s | ✅ Pass |
| `mafft` | no | ✅ | 25.2s | ✅ Pass |

### Sequence Scoring (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `alphagenome` | yes | ✅ | 297.7s | ✅ Pass |
| `borzoi` | yes | ✅ | 114.6s | ✅ Pass |
| `enformer` | yes | ✅ | 99.5s | ✅ Pass |
| `segmasker` | no | ✅ | 33.6s | ✅ Pass |

### Structure Alignment (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `tmalign` | no | ✅ | 24.2s | ✅ Pass |
| `tmalign` | no | ✅ | 0.1s | ✅ Pass |
| `usalign` | no | ✅ | 32.6s | ✅ Pass |
| `usalign` | no | ✅ | 0.1s | ✅ Pass |

### Structure Design (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `rfdiffusion3` | yes | ✅ | 218.4s | ✅ Pass |

### Structure Dynamics (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `bioemu` | yes | ✅ | 250.7s | ✅ Pass |

### Structure Prediction (8/8)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `alphafold2` | yes | ✅ | 111.2s | ✅ Pass |
| `alphafold3` | yes | ✅ | 82.5s | ✅ Pass |
| `boltz2` | yes | ✅ | 196.7s | ✅ Pass |
| `chai1` | yes | ✅ | 482.1s | ✅ Pass |
| `esmfold` | yes | ✅ | 119.8s | ✅ Pass |
| `protenix` | yes | ✅ | 438.4s | ✅ Pass |
| `structure_metrics` | no | ✅ | 25.1s | ✅ Pass |
| `viennarna` | no | ✅ | 17.5s | ✅ Pass |

---
*Generated at 2026-03-23 11:57:03 by `pytest --env-report`*