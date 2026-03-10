# Linux x86_64 Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-100%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-43-brightgreen) ![Failed](https://img.shields.io/badge/failed-0-red) ![Skipped](https://img.shields.io/badge/skipped-0-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Linux Linux 5.15.0-164-generic |
| **Architecture** | x86_64 |
| **Hostname** | `GPU71E4` |
| **Python** | 3.12.12 |
| **RAM** | 1007.4 GB |
| **GPU** | 2× NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3 |
| **CUDA** | 12.2 |
| **Mamba Env** | `bio-tools` |

## Git

- **Commit**: `8921085f2979`
- **Branch**: `bv/parallel`
- **Dirty**: Yes

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
CONDA_PREFIX_2=/home/bviggiano/miniforge3/envs/bio-tools
CONDA_PREFIX_3=/home/bviggiano/miniforge3
CONDA_PROMPT_MODIFIER=(bio-tools) 
CONDA_PYTHON_EXE=/home/bviggiano/miniforge3/bin/python
CONDA_SHLVL=4
CONDA_TOOLCHAIN_BUILD=x86_64-conda-linux-gnu
CONDA_TOOLCHAIN_HOST=x86_64-conda-linux-gnu
CPP=/home/bviggiano/miniforge3/envs/bio-tools/bin/x86_64-conda-linux-gnu-cpp
CPPFLAGS=-DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem /home/bviggiano/miniforge3/envs/bio-tools/include  -I/home/bviggiano/miniforge3/envs/bio-tools/targets/x86_64-linux/include
CPP_FOR_BUILD=/home/bviggiano/miniforge3/envs/bio-tools/bin/x86_64-conda-linux-gnu-cpp
CUDA_HOME=/home/bviggiano/miniforge3/envs/bio-tools
CUDA_HOME_BACKUP=/home/bviggiano/miniforge3/envs/bio-tools
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
GXX=/home/bviggiano/miniforge3/envs/bio-tools/bin/x86_64-conda-linux-gnu-g++
HOME=/home/bviggiano
HOST=x86_64-conda-linux-gnu
LANG=C.UTF-8
LD=x86_64-conda-linux-gnu-ld
LDFLAGS=-Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,--allow-shlib-undefined -Wl,-rpath,/home/bviggiano/miniforge3/envs/bio-tools/lib -Wl,-rp...
LDFLAGS_LD=-O2 --sort-common --as-needed -z relro -z now --disable-new-dtags --gc-sections --allow-shlib-undefined -rpath /home/bviggiano/miniforge3/envs/bio-tools/lib -rpath-link /home/bviggiano/miniforge3/envs...
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64
LESSCLOSE=/usr/bin/lesspipe %s %s
LESSOPEN=| /usr/bin/lesspipe %s
LOGNAME=bviggiano
LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.a...
MAMBA_EXE=/home/bviggiano/miniforge3/bin/mamba
MAMBA_ROOT_PREFIX=/home/bviggiano/.local/share/mamba
MESON_ARGS=-Dbuildtype=release
MOTD_SHOWN=pam
NM=x86_64-conda-linux-gnu-nm
NVCC_PREPEND_FLAGS= -ccbin=/home/bviggiano/miniforge3/envs/bio-tools/bin/x86_64-conda-linux-gnu-c++ -ccbin=/home/bviggiano/miniforge3/envs/bio-tools/bin/x86_64-conda-linux-gnu-c++
NVCC_PREPEND_FLAGS_BACKUP= -ccbin=/home/bviggiano/miniforge3/envs/bio-tools/bin/x86_64-conda-linux-gnu-c++
OBJCOPY=x86_64-conda-linux-gnu-objcopy
OBJDUMP=x86_64-conda-linux-gnu-objdump
OLDPWD=/home/bviggiano/main/codebases
PATH=/home/bviggiano/.local/bin:/home/bviggiano/bin:/usr/local/cuda/bin:/home/bviggiano/.local/bin:/home/bviggiano/bin:/home/bviggiano/miniforge3/envs/bio-tools/bin:/home/bviggiano/miniforge3/condabin:/usr...
PWD=/home/bviggiano/main/codebases/bio-programming/bio-programming-tools
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
RANLIB=x86_64-conda-linux-gnu-ranlib
RCLONE_CONFIG=/large_storage/rclone/etc/rclone.conf
RDBASE=/home/bviggiano/miniforge3/envs/bio-tools/lib/python3.12/site-packages/rdkit
READELF=x86_64-conda-linux-gnu-readelf
SHELL=/bin/bash
SHLVL=2
SIZE=x86_64-conda-linux-gnu-size
SLURM_JOB_ID=1763207
STRINGS=x86_64-conda-linux-gnu-strings
STRIP=x86_64-conda-linux-gnu-strip
TERM=xterm-256color
TERM_PROGRAM=tmux
TERM_PROGRAM_VERSION=3.2a
TMUX=/tmp/tmux-10249/default,1388272,0
TMUX_PANE=%0
USER=bviggiano
XDG_DATA_DIRS=/usr/local/share:/usr/share:/var/lib/snapd/desktop
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
XML_CATALOG_FILES=file:///home/bviggiano/miniforge3/etc/xml/catalog file:///etc/xml/catalog file:///home/bviggiano/miniforge3/envs/bio-tools/etc/xml/catalog file:///etc/xml/catalog file:///home/bviggiano/miniforge3/etc...
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
CONDA_PREFIX=/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/viennarna_env
DETECTED_COMPUTE_PLATFORM=cuda
DETECTED_CUDA_VERSION=12
DETECTED_DRIVER_VERSION=535
HOME=/home/bviggiano
LANG=C.UTF-8
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/home/bviggiano/miniforge3/envs/bio-tools/lib
LOGNAME=bviggiano
PATH=/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/viennarna_env/bin:/home/bviggiano/.local/bin:/home/bviggiano/bin:/usr/local/cuda/bin:/home/bviggiano/miniforge...
RECOMMENDED_JAX_SPEC=jax[cuda12]>=0.4.20,<1
RECOMMENDED_JAX_VARIANT=cuda12
RECOMMENDED_TORCH_SPEC=torch>=2.4,<2.7
SHELL=/bin/bash
TORCH_CUDA_ARCH_LIST=9.0
TORCH_HOME=/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/viennarna_env/cache/torch
USER=bviggiano
VIRTUAL_ENV=/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/viennarna_env
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Results by Category

### Causal Models (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `evo1` | yes | ✅ | 300.2s | ✅ Pass |
| `evo2` | yes | ✅ | 305.9s | ✅ Pass |
| `evo2` | yes | ✅ | 34.5s | ✅ Pass |
| `progen2` | yes | ✅ | 176.2s | ✅ Pass |

### Gene Annotation (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `blast` | no | ✅ | 56.5s | ✅ Pass |
| `minced` | no | ✅ | 22.7s | ✅ Pass |
| `mmseqs` | no | ✅ | 28.7s | ✅ Pass |
| `pyhmmer` | no | ✅ | 27.8s | ✅ Pass |

### Inverse Folding (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `ligandmpnn` | yes | ✅ | 164.9s | ✅ Pass |
| `proteinmpnn` | yes | ✅ | 102.5s | ✅ Pass |

### Masked Models (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `esm2` | yes | ✅ | 102.0s | ✅ Pass |
| `esm3` | yes | ✅ | 204.6s | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `orfipy` | no | ✅ | 26.7s | ✅ Pass |
| `prodigal` | no | ✅ | 17.6s | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `splice_transformer` | yes | ✅ | 96.4s | ✅ Pass |

### Sequence Alignment (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `mafft` | no | ✅ | 28.1s | ✅ Pass |

### Sequence Scoring (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `borzoi` | yes | ✅ | 138.5s | ✅ Pass |
| `enformer` | yes | ✅ | 117.8s | ✅ Pass |

### Structure Alignment (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `tmalign` | no | ✅ | 25.6s | ✅ Pass |
| `tmalign` | no | ✅ | 0.1s | ✅ Pass |
| `usalign` | no | ✅ | 35.7s | ✅ Pass |
| `usalign` | no | ✅ | 0.1s | ✅ Pass |

### Structure Design (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `rfdiffusion3` | yes | ✅ | 148.8s | ✅ Pass |

### Structure Dynamics (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `bioemu` | yes | ✅ | 149.3s | ✅ Pass |

### Structure Prediction (15/15)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `alphafold2` | yes | ✅ | 231.3s | ✅ Pass |
| `alphafold3` | yes | ✅ | 82.2s | ✅ Pass |
| `boltz2` | yes | ✅ | 194.2s | ✅ Pass |
| `chai1` | yes | ✅ | 282.5s | ✅ Pass |
| `esmfold` | yes | ✅ | 118.5s | ✅ Pass |
| `protenix` | yes | ✅ | 536.5s | ✅ Pass |
| `protenix` | yes | ✅ | 176.5s | ✅ Pass |
| `protenix` | yes | ✅ | 177.9s | ✅ Pass |
| `protenix` | yes | ✅ | 285.7s | ✅ Pass |
| `protenix` | yes | ✅ | 68.5s | ✅ Pass |
| `protenix` | yes | ✅ | 256.3s | ✅ Pass |
| `protenix` | yes | ✅ | 530.1s | ✅ Pass |
| `protenix` | yes | ✅ | 62.7s | ✅ Pass |
| `protenix` | yes | ✅ | 110.0s | ✅ Pass |
| `viennarna` | no | ✅ | 17.3s | ✅ Pass |

### Unknown (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `alphagenome` | yes | ✅ | 355.9s | ✅ Pass |
| `crispr_tracr` | no | ✅ | 161.1s | ✅ Pass |
| `local_colabfold_search` | no | — | 133.9s | ✅ Pass |
| `structure_metrics` | no | ✅ | 38.3s | ✅ Pass |

---
*Generated at 2026-03-08 17:31:05 by `pytest --env-report`*