# Linux x86_64 Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-100%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-33-brightgreen) ![Failed](https://img.shields.io/badge/failed-0-red) ![Skipped](https://img.shields.io/badge/skipped-0-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Linux Linux 3.10.0-1160.139.1.el7.tuxcare.els4.x86_64 |
| **Architecture** | x86_64 |
| **Hostname** | `sh04-15n01.int` |
| **Python** | 3.12.12 |
| **RAM** | 2015.1 GB |
| **GPU** | 1× NVIDIA H100 80GB HBM3 |
| **CUDA** | 12.4 |
| **Conda Env** | `bio-programming` |

## Git

- **Commit**: `f947e4790e75`
- **Branch**: `main`
- **Dirty**: Yes

## Environment Variables

### Parent Process Environment

```
APPTAINER_APPNAME=
APPTAINER_BIND=/bashrc_custom
APPTAINER_CACHEDIR=/oak/stanford/groups/brianhie/danguo/.apptainer
APPTAINER_COMMAND=exec
APPTAINER_CONTAINER=/home/groups/brianhie/brianhie/simg/pytorch_latest.sif
APPTAINER_ENVIRONMENT=/.singularity.d/env/91-environment.sh
APPTAINER_NAME=pytorch_latest.sif
BASH_ENV=/share/software/user/open/lmod/lmod/init/bash
BASH_FUNC__cache_cmd()=() {  local cache=$HOME/.cache/sh/;
 local cmd=${1:-};
 local cachelife=${2:-};
 local entry exp;
 shift 2;
 [[ -d ${cache} ]] || /bin/mkdir -p "${cache}";
 read -r prm_hash dash <<< "$(/bin/md5sum <<...
BASH_FUNC_bc()=() {  timeout 3600 bc "$@"
}
BASH_FUNC_ml()=() {  eval "$($LMOD_DIR/ml_cmd "$@")"
}
BASH_FUNC_module()=() {  if [ -z "${LMOD_SH_DBG_ON+x}" ]; then
 case "$-" in 
 *v*x*)
 __lmod_sh_dbg='vx'
 ;;
 *v*)
 __lmod_sh_dbg='v'
 ;;
 *x*)
 __lmod_sh_dbg='x'
 ;;
 esac;
 fi;
 if [ -n "${__lmod_sh_dbg:-}" ]; then
 ...
BASH_FUNC_sacct()=() {  _cache_cmd /usr/bin/sacct 60 "$@"
}
BASH_FUNC_scontrol()=() {  [[ $1 == "show" ]] && _cache_cmd /usr/bin/scontrol 120 "$@" || /usr/bin/scontrol "$@"
}
BASH_FUNC_sh_jobs()=() {  _cache_cmd $SRCC_PATH/sh_jobs 90 "$@"
}
BASH_FUNC_sh_jobwait()=() {  _cache_cmd $SRCC_PATH/sh_jobwait 300 "$@"
}
BASH_FUNC_sh_next_downtime()=() {  _cache_cmd $SRCC_PATH/sh_next_downtime 3600 "$@"
}
BASH_FUNC_sh_part()=() {  _cache_cmd $SRCC_PATH/sh_part 30 "$@"
}
BASH_FUNC_sh_quota()=() {  _cache_cmd $SRCC_PATH/sh_quota 180 "$@"
}
BASH_FUNC_sh_status()=() {  _cache_cmd $SRCC_PATH/sh_status 3600 "$@"
}
BASH_FUNC_sh_usage()=() {  _cache_cmd $SRCC_PATH/sh_usage 90 "$@"
}
BASH_FUNC_sinfo()=() {  _cache_cmd /usr/bin/sinfo 60 "$@"
}
BASH_FUNC_sleep()=() {  timeout 14400 sleep "$@"
}
BASH_FUNC_squeue()=() {  _cache_cmd /usr/bin/squeue 20 "$@"
}
BASH_FUNC_sstat()=() {  _cache_cmd /usr/bin/sstat 10 "$@"
}
BASH_FUNC_sudo()=() {  $SRCC_PATH/sudo
}
BLIS_NUM_THREADS=8
CC=gcc
CONDA_DEFAULT_ENV=bio-programming
CONDA_EXE=/home/users/danguo/miniforge3/bin/conda
CONDA_PKGS_DIRS=/home/groups/brianhie/danguo/.conda/pkgs
CONDA_PREFIX=/home/users/danguo/miniforge3/envs/bio-programming
CONDA_PREFIX_1=/home/users/danguo/miniforge3
CONDA_PROMPT_MODIFIER=(bio-programming) 
CONDA_PYTHON_EXE=/home/users/danguo/miniforge3/bin/python
CONDA_SHLVL=2
CPATH=/share/software/user/open/nodejs/25.3.0/include:/share/software/user/open/gcc/14.2.0/include
CPP=cpp
CUDA_VISIBLE_DEVICES=0
CXX=c++
DISABLE_PANDERA_IMPORT_WARNING=True
F77=gfortran
F90=gfortran
FC=gfortran
GOTO_NUM_THREADS=8
GROUP=brianhie
GROUP_HOME=/home/groups/brianhie
GROUP_SCRATCH=/scratch/groups/brianhie
HF_HOME=/oak/stanford/groups/brianhie/danguo/.cache/huggingface
HISTCONTROL=ignoreboth:erasedups
HISTSIZE=1000
HOME=/home/users/danguo
HOSTNAME=sh04-15n01.int
HYDRA_BOOTSTRAP=slurm
HYDRA_LAUNCHER_EXTRA_ARGS=--external-launcher
INFOPATH=/share/software/user/open/nodejs/25.3.0/share/info
I_MPI_HYDRA_BOOTSTRAP=slurm
I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS=--external-launcher
KRB5CCNAME=FILE:/tmp/krb5cc_369751_3VmI5ZFIrY
LANG=en_US.UTF-8
LC_CTYPE=C.UTF-8
LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs
LESSOPEN=||/usr/bin/lesspipe.sh %s
LIBRARY_PATH=/share/software/user/open/nodejs/25.3.0/lib:/share/software/user/open/gcc/14.2.0/lib64:/share/software/user/open/gcc/14.2.0/lib
LMOD_ADMIN_FILE=/share/software/modules/admin.list
LMOD_AVAIL_STYLE=categories
LMOD_CMD=/share/software/user/open/lmod/9.0.2/libexec/lmod
LMOD_COLORIZE=yes
LMOD_DIR=/share/software/user/open/lmod/9.0.2/libexec
LMOD_FULL_SETTARG_SUPPORT=no
LMOD_MODULERCFILE=/share/software/modules/.modulerc.lua
LMOD_PACKAGE_PATH=/share/software/modules/
LMOD_PKG=/share/software/user/open/lmod/9.0.2
LMOD_PREPEND_BLOCK=normal
LMOD_RC=/share/software/modules/lmodrc.lua
LMOD_REDIRECT=yes
LMOD_ROOT=/share/software/user/open/lmod
LMOD_SETTARG_CMD=:
LMOD_SETTARG_FULL_SUPPORT=no
LMOD_SYSHOST=sherlock
LMOD_SYSTEM_DEFAULT_MODULES=devel,math
LMOD_VERSION=9.0.2
LMOD_arch=x86_64
LMOD_sys=Linux
LOADEDMODULES=devel:math:gcc/14.2.0:nodejs/25.3.0:claude-code/2.1.38
LOCAL_SCRATCH=/lscratch/danguo
LOGNAME=danguo
LS_COLORS=rs=0:di=38;5;27:ln=38;5;51:mh=44;38;5;15:pi=40;38;5;11:so=38;5;13:do=38;5;5:bd=48;5;232;38;5;11:cd=48;5;232;38;5;3:or=48;5;232;38;5;9:mi=05;48;5;232;38;5;15:su=48;5;196;38;5;15:sg=48;5;11;38;5;16:ca=4...
L_SCRATCH=/lscratch/danguo
L_SCRATCH_JOB=/lscratch/danguo/17261351
L_SCRATCH_USER=/lscratch/danguo
MAIL=/var/spool/mail/danguo
MANPATH=/share/software/user/open/lmod/lmod/share/man:/share/software/user/open/nodejs/25.3.0/share/man:/share/software/user/open/gcc/14.2.0/share/man:/usr/local/share/man:/usr/share/man
MKL_NUM_THREADS=8
MODULEPATH=/share/software/modules/math:/share/software/modules/devel:/share/software/modules/categories
MODULEPATH_ROOT=/share/software/modules
MODULESHOME=/share/software/user/open/lmod/9.0.2
NVIDIA_DRIVER_CAPABILITIES=compute,utility
NVIDIA_VISIBLE_DEVICES=all
OAK=/oak/stanford/groups/brianhie
OLDPWD=/home/groups/brianhie/danguo
OMPI_MCA_orte_precondition_transports=0107632700000000-0107632700000000
OMPI_MCA_plm_slurm_args=--external-launcher
OMP_NUM_THREADS=8
OPENBLAS_NUM_THREADS=8
PATH=/home/users/danguo/miniforge3/envs/bio-programming/bin:/home/users/danguo/.local/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PMIX_BFROP_BUFFER_TYPE=PMIX_BFROP_BUFFER_NON_DESC
PMIX_GDS_MODULE=shmem2,hash
PMIX_HOSTNAME=sh04-15n01
PMIX_NAMESPACE=slurm.pmix.17261351.0
PMIX_RANK=0
PMIX_SECURITY_MODE=native
PMIX_SERVER_TMPDIR=/var/spool/slurmd/pmix.17261351.0/
PMIX_SERVER_URI2=pmix-server.20960;tcp4://127.0.0.1:37757
PMIX_SERVER_URI21=pmix-server.20960;tcp4://127.0.0.1:37757
PMIX_SERVER_URI3=pmix-server.20960;tcp4://127.0.0.1:37757
PMIX_SERVER_URI4=pmix-server.20960;tcp4://127.0.0.1:37757
PMIX_SERVER_URI41=pmix-server.20960;tcp4://127.0.0.1:37757
PMIX_SYSTEM_TMPDIR=/tmp
PMIX_VERSION=5.0.3
PROMPT_COMMAND=RET=$?;/bin/logger -t user_audit "username=$USER pid=$$ cmd=\"$(history 1 | /bin/sed "s/^[ ]*[0-9]\+[ ]*//" )\" newpwd=$PWD ret=$RET" 2>/dev/null
PRTE_MCA_plm_slurm_args=--external-launcher
PS1=(bio-programming) (apptainer) \u@\h:\w$ 
PWD=/home/groups/brianhie/danguo/bio-programming/bio-programming-tools
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
PYTHONPYCACHEPREFIX=/tmp
PYTHONUSERBASE=/home/groups/brianhie/danguo/.local
PYTORCH_VERSION=2.2.1
RDBASE=/home/users/danguo/miniforge3/envs/bio-programming/lib/python3.12/site-packages/rdkit
SCRATCH=/scratch/users/danguo
SHELL=/bin/bash
SHERLOCK=2
SHLVL=3
SH_DEF_TIMEOUT=30
SH_INFO_TIMEOUT=30
SH_TIMEOUT_CMD=timeout -s9
SH_USER_ENV_INITD=1
SINGULARITY_BIND=/bashrc_custom
SINGULARITY_CONTAINER=/home/groups/brianhie/brianhie/simg/pytorch_latest.sif
SINGULARITY_ENVIRONMENT=/.singularity.d/env/91-environment.sh
SINGULARITY_NAME=pytorch_latest.sif
SLURMD_DEBUG=2
SLURMD_NODENAME=sh04-15n01
SLURM_CLUSTER_NAME=sherlock
SLURM_CONF=/var/spool/slurmd/conf-cache/slurm.conf
SLURM_CPUS_ON_NODE=8
SLURM_CPUS_PER_TASK=8
SLURM_CPU_BIND=quiet,mask_cpu:0x000003FC00000000
SLURM_CPU_BIND_LIST=0x000003FC00000000
SLURM_CPU_BIND_TYPE=mask_cpu:
SLURM_CPU_BIND_VERBOSE=quiet
SLURM_DISTRIBUTION=block
SLURM_GPUS=1
SLURM_GPUS_ON_NODE=1
SLURM_GTIDS=0
SLURM_JOBID=17261351
SLURM_JOB_ACCOUNT=brianhie
SLURM_JOB_CPUS_PER_NODE=8
SLURM_JOB_END_TIME=1772203535
SLURM_JOB_GID=21859
SLURM_JOB_GROUP=brianhie
SLURM_JOB_ID=17261351
SLURM_JOB_NAME=bash
SLURM_JOB_NODELIST=sh04-15n01
SLURM_JOB_NUM_NODES=1
SLURM_JOB_PARTITION=brianhie
SLURM_JOB_QOS=normal
SLURM_JOB_START_TIME=1772160331
SLURM_JOB_UID=369751
SLURM_JOB_USER=danguo
SLURM_LAUNCH_NODE_IPADDR=10.18.0.62
SLURM_LOCALID=0
SLURM_MEM_PER_CPU=30720
SLURM_MPI_TYPE=pmix
SLURM_NNODES=1
SLURM_NODEID=0
SLURM_NODELIST=sh04-15n01
SLURM_NPROCS=1
SLURM_NTASKS=1
SLURM_OOM_KILL_STEP=0
SLURM_PMIXP_ABORT_AGENT_PORT=32782
SLURM_PMIX_MAPPING_SERV=(vector,(0,1,1))
SLURM_PRIO_PROCESS=0
SLURM_PROCID=0
SLURM_PTY_PORT=33346
SLURM_PTY_WIN_COL=327
SLURM_PTY_WIN_ROW=95
SLURM_SCRIPT_CONTEXT=prolog_task
SLURM_SRUN_COMM_HOST=10.18.0.62
SLURM_SRUN_COMM_PORT=36967
SLURM_STEPID=0
SLURM_STEPMGR=sh04-15n01
SLURM_STEP_GPUS=4
SLURM_STEP_ID=0
SLURM_STEP_LAUNCHER_PORT=36967
SLURM_STEP_NODELIST=sh04-15n01
SLURM_STEP_NUM_NODES=1
SLURM_STEP_NUM_TASKS=1
SLURM_STEP_TASKS_PER_NODE=1
SLURM_SUBMIT_DIR=/home/groups/brianhie/danguo
SLURM_SUBMIT_HOST=sh02-ln02.stanford.edu
SLURM_TASKS_PER_NODE=1
SLURM_TASK_PID=20981
SLURM_TOPOLOGY_ADDR=sh04.sh04-isw-15.sh04-15n01
SLURM_TOPOLOGY_ADDR_PATTERN=switch.switch.node
SLURM_TRES_PER_TASK=cpu=8
SLURM_UMASK=0022
SRCC_PATH=/share/software/user/srcc/bin
SRUN_CPUS_PER_TASK=8
SRUN_DEBUG=3
TERM=xterm-256color
TERM_PROGRAM=WarpTerminal
TMOUT=86400
TMPDIR=/tmp
TORCH_HOME=/oak/stanford/groups/brianhie/danguo/.cache/torch
USER=danguo
USER_PATH=/home/users/danguo/.local/bin:/home/users/danguo/.local/bin:/share/software/user/srcc/bin:/share/software/user/open/claude-code/2.1.38/bin:/share/software/user/open/nodejs/25.3.0/bin:/share/software/u...
XDG_CACHE_HOME=/tmp
XDG_RUNTIME_DIR=/tmp
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
XML_CATALOG_FILES=file:///home/users/danguo/miniforge3/etc/xml/catalog file:///etc/xml/catalog file:///home/users/danguo/miniforge3/etc/xml/catalog file:///etc/xml/catalog file:///home/users/danguo/miniforge3/etc/xml/c...
_=/home/users/danguo/miniforge3/envs/bio-programming/bin/pytest
_CE_CONDA=
_CE_M=
_CONDA_EXE=/home/users/danguo/miniforge3/bin/conda
_CONDA_ROOT=/home/users/danguo/miniforge3
_LMFILES_=/share/software/modules/categories/devel.lua:/share/software/modules/categories/math.lua:/share/software/modules/devel/gcc/14.2.0.lua:/share/software/modules/devel/nodejs/25.3.0.lua:/share/software/mo...
_ModuleTable001_=X01vZHVsZVRhYmxlXyA9IHsKTVR2ZXJzaW9uID0gMywKY19yZWJ1aWxkVGltZSA9IGZhbHNlLApjX3Nob3J0VGltZSA9IGZhbHNlLApkZXB0aFQgPSB7fSwKZmFtaWx5ID0ge30sCm1UID0gewpbImNsYXVkZS1jb2RlIl0gPSB7CmRlcFQgPSB7CmRlcEEgPSB7CnsK...
_ModuleTable002_=dmUiLAp1c2VyTmFtZSA9ICJjbGF1ZGUtY29kZSIsCndWID0gIjAwMDAwMDAwMi4wMDAwMDAwMDEuMDAwMDAwMDM4Lip6ZmluYWwiLAp9LApkZXZlbCA9IHsKYWN0aW9uQSA9IHsKW1twcmVwZW5kX3BhdGgoIk1PRFVMRVBBVEgiLCIvc2hhcmUvc29mdHdhcmUvbW9k...
_ModuleTable003_=L3NoYXJlL3NvZnR3YXJlL21vZHVsZXMvZGV2ZWwvZ2NjLzE0LjIuMC5sdWEiLApmdWxsTmFtZSA9ICJnY2MvMTQuMi4wIiwKbG9hZE9yZGVyID0gMywKcHJvcFQgPSB7fSwKcmVmX2NvdW50ID0gMSwKc3RhY2tEZXB0aCA9IDIsCnN0YXR1cyA9ICJhY3RpdmUiLAp1...
_ModuleTable004_=LApwcm9wVCA9IHsKbG1vZCA9IHsKc3RpY2t5ID0gMSwKfSwKfSwKc3RhY2tEZXB0aCA9IDAsCnN0YXR1cyA9ICJhY3RpdmUiLAp1c2VyTmFtZSA9ICJtYXRoIiwKd1YgPSAiTS4qemZpbmFsIiwKfSwKbm9kZWpzID0gewpkZXBUID0gewpkZXBBID0gewp7CnNuID0g...
_ModuleTable005_=ZSIsCnVzZXJOYW1lID0gIm5vZGVqcy8yNS4zLjAiLAp3ViA9ICIwMDAwMDAwMjUuMDAwMDAwMDAzLip6ZmluYWwiLAp9LAp9LAptcGF0aEEgPSB7CiIvc2hhcmUvc29mdHdhcmUvbW9kdWxlcy9tYXRoIiwgIi9zaGFyZS9zb2Z0d2FyZS9tb2R1bGVzL2RldmVsIiwg...
_ModuleTable_Sz_=5
__Init_Default_Modules=1
__LMOD_REF_COUNT_CPATH=/share/software/user/open/nodejs/25.3.0/include:1;/share/software/user/open/gcc/14.2.0/include:1
__LMOD_REF_COUNT_INFOPATH=/share/software/user/open/nodejs/25.3.0/share/info:1
__LMOD_REF_COUNT_LD_LIBRARY_PATH=/share/software/user/open/nodejs/25.3.0/lib:1;/share/software/user/open/gcc/14.2.0/lib64:1;/share/software/user/open/gcc/14.2.0/lib/gcc/x86_64-pc-linux-gnu:1;/share/software/user/open/gcc/14.2.0/lib:1
__LMOD_REF_COUNT_LIBRARY_PATH=/share/software/user/open/nodejs/25.3.0/lib:1;/share/software/user/open/gcc/14.2.0/lib64:1;/share/software/user/open/gcc/14.2.0/lib:1
__LMOD_REF_COUNT_MANPATH=/share/software/user/open/nodejs/25.3.0/share/man:1;/share/software/user/open/gcc/14.2.0/share/man:1;/share/software/user/open/lmod/lmod/share/man:1;/usr/local/share/man:1;/usr/share/man:1
__LMOD_REF_COUNT_MODULEPATH=/share/software/modules/math:1;/share/software/modules/devel:1;/share/software/modules/categories:1
__LMOD_REF_COUNT_PATH=/share/software/user/open/claude-code/2.1.38/bin:1;/share/software/user/open/nodejs/25.3.0/bin:1;/share/software/user/open/gcc/14.2.0/bin:1;/home/users/danguo/.local/bin:3;/share/software/user/srcc/bi...
__LMOD_STACK_CC=false
__LMOD_STACK_CPP=false
__LMOD_STACK_CXX=false
__LMOD_STACK_F77=false
__LMOD_STACK_F90=false
__LMOD_STACK_FC=false
```

### Subprocess Environment (passed to tools)

```
CONDA_PREFIX=/home/groups/brianhie/danguo/bio-programming/bio-programming-tools/tool_envs/splice_transformer_env
CUDA_VISIBLE_DEVICES=0
DETECTED_COMPUTE_PLATFORM=cuda
DETECTED_CUDA_VERSION=12
DETECTED_DRIVER_VERSION=550
HOME=/home/users/danguo
LANG=en_US.UTF-8
LC_CTYPE=C.UTF-8
LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs:/home/users/danguo/miniforge3/envs/bio-programming/lib
LOGNAME=danguo
PATH=/home/groups/brianhie/danguo/bio-programming/bio-programming-tools/tool_envs/splice_transformer_env/bin:/usr/local/cuda/bin:/home/users/danguo/miniforge3/envs/bio-programming/bin:/home/users/danguo/.l...
RECOMMENDED_JAX_SPEC=jax[cuda12]>=0.4.20,<1
RECOMMENDED_JAX_VARIANT=cuda12
RECOMMENDED_TORCH_SPEC=torch>=2.5,<3
SHELL=/bin/bash
TMPDIR=/tmp
TORCH_CUDA_ARCH_LIST=9.0
TORCH_HOME=/home/groups/brianhie/danguo/bio-programming/bio-programming-tools/tool_envs/splice_transformer_env/cache/torch
USER=danguo
VIRTUAL_ENV=/home/groups/brianhie/danguo/bio-programming/bio-programming-tools/tool_envs/splice_transformer_env
XDG_CACHE_HOME=/tmp
```

## Results by Category

### Causal Models (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `evo1` | yes | ✅ | 641.6s | ✅ Pass |
| `evo2` | yes | ✅ | 844.6s | ✅ Pass |
| `progen2` | yes | ✅ | 581.1s | ✅ Pass |

### Gene Annotation (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `blast` | no | ✅ | 112.8s | ✅ Pass |
| `minced` | no | ✅ | 49.5s | ✅ Pass |
| `mmseqs` | no | ✅ | 89.4s | ✅ Pass |
| `pyhmmer` | no | ✅ | 73.5s | ✅ Pass |

### Inverse Folding (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `ligandmpnn` | yes | ✅ | 776.0s | ✅ Pass |
| `proteinmpnn` | yes | ✅ | 235.1s | ✅ Pass |

### Masked Models (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `esm2` | yes | ✅ | 383.3s | ✅ Pass |
| `esm3` | yes | ✅ | 577.9s | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `orfipy` | no | ✅ | 87.8s | ✅ Pass |
| `prodigal` | no | ✅ | 42.4s | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `splice_transformer` | yes | ✅ | 323.0s | ✅ Pass |

### Sequence Alignment (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `mafft` | no | ✅ | 62.0s | ✅ Pass |

### Sequence Scoring (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `borzoi` | yes | ✅ | 477.3s | ✅ Pass |
| `enformer` | yes | ✅ | 403.2s | ✅ Pass |

### Structure Alignment (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `tmalign` | no | ✅ | 48.2s | ✅ Pass |
| `tmalign` | no | ✅ | 0.1s | ✅ Pass |
| `usalign` | no | ✅ | 57.0s | ✅ Pass |
| `usalign` | no | ✅ | 0.2s | ✅ Pass |

### Structure Design (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `rfdiffusion3` | yes | ✅ | 819.2s | ✅ Pass |

### Structure Dynamics (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `bioemu` | yes | ✅ | 1189.6s | ✅ Pass |

### Structure Prediction (6/6)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `alphafold2` | yes | ✅ | 373.1s | ✅ Pass |
| `boltz2` | yes | ✅ | 607.3s | ✅ Pass |
| `chai1` | yes | ✅ | 1054.1s | ✅ Pass |
| `esmfold` | yes | ✅ | 476.9s | ✅ Pass |
| `protenix` | yes | ✅ | 1055.1s | ✅ Pass |
| `viennarna` | no | ✅ | 61.8s | ✅ Pass |

### Unknown (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `alphagenome` | yes | ✅ | 694.0s | ✅ Pass |
| `crispr_tracr` | no | ✅ | 454.1s | ✅ Pass |
| `local_colabfold_search` | no | — | 382.1s | ✅ Pass |
| `structure_metrics` | no | ✅ | 83.8s | ✅ Pass |

---
*Generated at 2026-02-27 06:08:16 by `pytest --env-report`*