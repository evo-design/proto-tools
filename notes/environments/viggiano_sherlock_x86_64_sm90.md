# Linux x86_64 Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-98%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-51-brightgreen) ![Failed](https://img.shields.io/badge/failed-1-red) ![Skipped](https://img.shields.io/badge/skipped-3-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Linux Linux 3.10.0-1160.139.1.el7.tuxcare.els4.x86_64 |
| **Architecture** | x86_64 |
| **Hostname** | `sh04-16n09.int` |
| **Python** | 3.10.20 |
| **RAM** | 2015.1 GB |
| **GPU** | 1x NVIDIA H200 |
| **CUDA** | 12.4 |
| **Conda Env** | `proto-tools` |

## Git

- **Commit**: `4c6e45c77832`
- **Branch**: `fix/sherlock-germinal-minimal`
- **Dirty**: No

## Environment Variables

### Parent Process Environment

```
AI_AGENT=claude-code_2-1-132_agent
APPTAINER_APPNAME=
APPTAINER_BIND=/bashrc_custom,/share/software/user/open
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
BASH_FUNC_ml%%=() {  eval "$($LMOD_DIR/ml_cmd "$@")"
}
BASH_FUNC_ml()=() {  eval "$($LMOD_DIR/ml_cmd "$@")"
}
BASH_FUNC_module%%=() {  if [ -z "${LMOD_SH_DBG_ON+x}" ]; then
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
BLIS_NUM_THREADS=4
CC=gcc
CLAUDECODE=1
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
CLAUDE_CODE_ENTRYPOINT=cli
CLAUDE_CODE_EXECPATH=/share/software/user/open/claude-code/2.1.132/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe
COMMON_DATASETS=/oak/stanford/datasets/common
CONDA_DEFAULT_ENV=proto-tools
CONDA_EXE=/home/users/viggiano/miniconda3/bin/conda
CONDA_PREFIX=/scratch/users/viggiano/envs/proto-tools
CONDA_PREFIX_1=/home/users/viggiano/miniconda3
CONDA_PREFIX_2=/scratch/users/viggiano/envs/proto-tools
CONDA_PREFIX_3=/home/users/viggiano/miniconda3
CONDA_PROMPT_MODIFIER=(proto-tools) 
CONDA_PYTHON_EXE=/home/users/viggiano/miniconda3/bin/python
CONDA_SHLVL=4
COREPACK_ENABLE_AUTO_PIN=0
CPATH=/share/software/user/open/nodejs/20.20.0/include:/share/software/user/open/gcc/14.2.0/include
CPP=cpp
CUDA_VISIBLE_DEVICES=0
CXX=c++
DISABLE_INSTALLATION_CHECKS=1
DISABLE_PANDERA_IMPORT_WARNING=True
F77=gfortran
F90=gfortran
FC=gfortran
GIT_EDITOR=true
GIT_SSL_CAINFO=/etc/ssl/certs/ca-certificates.crt
GOTO_NUM_THREADS=4
GROUP=euan
GROUP_HOME=/home/groups/euan
GROUP_SCRATCH=/scratch/groups/euan
HISTCONTROL=ignoreboth:erasedups
HISTSIZE=1000
HOME=/home/users/viggiano
HOSTNAME=sh04-16n09.int
HYDRA_BOOTSTRAP=slurm
HYDRA_LAUNCHER_EXTRA_ARGS=--external-launcher
INFOPATH=/share/software/user/open/nodejs/20.20.0/share/info
I_MPI_HYDRA_BOOTSTRAP=slurm
I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS=--external-launcher
KRB5CCNAME=FILE:/tmp/krb5cc_389221_FmRhf6oxqp
LANG=en_US.UTF-8
LC_CTYPE=C.UTF-8
LD_LIBRARY_PATH=/share/software/user/open/expat/2.2.3/lib:/share/software/user/open/curl/8.4.0/lib:/share/software/user/open/gcc/14.2.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs
LESSOPEN=||/usr/bin/lesspipe.sh %s
LIBRARY_PATH=/share/software/user/open/nodejs/20.20.0/lib:/share/software/user/open/gcc/14.2.0/lib64:/share/software/user/open/gcc/14.2.0/lib
LMOD_ADMIN_FILE=/share/software/modules/admin.list
LMOD_AVAIL_STYLE=categories
LMOD_CMD=/share/software/user/open/lmod/lmod/libexec/lmod
LMOD_COLORIZE=yes
LMOD_DIR=/share/software/user/open/lmod/lmod/libexec
LMOD_FULL_SETTARG_SUPPORT=no
LMOD_MODULERCFILE=/share/software/modules/.modulerc.lua
LMOD_PACKAGE_PATH=/share/software/modules/
LMOD_PKG=/share/software/user/open/lmod/lmod
LMOD_PREPEND_BLOCK=normal
LMOD_RC=/share/software/modules/lmodrc.lua
LMOD_REDIRECT=yes
LMOD_ROOT=/share/software/user/open/lmod
LMOD_SETTARG_CMD=:
LMOD_SETTARG_FULL_SUPPORT=no
LMOD_SYSHOST=sherlock
LMOD_SYSTEM_DEFAULT_MODULES=devel,math
LMOD_VERSION=9.2
LMOD_arch=x86_64
LMOD_sys=Linux
LOADEDMODULES=devel:math:claude-code/2.1.132:gcc/14.2.0:nodejs/20.20.0
LOCAL_SCRATCH=/lscratch/viggiano
LOGNAME=viggiano
LS_COLORS=su=00:sg=00:ca=00:or=40;31;01
L_SCRATCH=/lscratch/viggiano
L_SCRATCH_JOB=/lscratch/viggiano/24471977
L_SCRATCH_USER=/lscratch/viggiano
MAIL=/var/spool/mail/viggiano
MANPATH=/share/software/user/open/nodejs/20.20.0/share/man:/share/software/user/open/gcc/14.2.0/share/man:/share/software/user/open/lmod/lmod/share/man:/usr/local/share/man:/usr/share/man
MKL_NUM_THREADS=4
MODULEPATH=/share/software/modules/math:/share/software/modules/devel:/share/software/modules/categories
MODULEPATH_ROOT=/share/software/modules
MODULESHOME=/share/software/user/open/lmod/lmod
NVIDIA_DRIVER_CAPABILITIES=compute,utility
NVIDIA_VISIBLE_DEVICES=all
NoDefaultCurrentDirectoryInExePath=1
OAK=/oak/stanford/groups/euan
OLDPWD=/home/users/viggiano
OMPI_MCA_orte_precondition_transports=017569a900000000-017569a900000000
OMPI_MCA_plm_slurm_args=--external-launcher
OMP_NUM_THREADS=4
OPENBLAS_NUM_THREADS=4
PATH=/home/users/viggiano/.local/bin:/share/software/user/open/git/2.45.1/bin:/share/software/user/open/nodejs/25.3.0/bin:/share/software/user/open/claude-code/2.1.132/bin:/home/users/viggiano/.npm-global/...
PMIX_BFROP_BUFFER_TYPE=PMIX_BFROP_BUFFER_NON_DESC
PMIX_GDS_MODULE=shmem2,hash
PMIX_HOSTNAME=sh04-16n09
PMIX_NAMESPACE=slurm.pmix.24471977.0
PMIX_RANK=0
PMIX_SECURITY_MODE=native
PMIX_SERVER_TMPDIR=/var/spool/slurmd/pmix.24471977.0/
PMIX_SERVER_URI2=pmix-server.26410;tcp4://127.0.0.1:38247
PMIX_SERVER_URI21=pmix-server.26410;tcp4://127.0.0.1:38247
PMIX_SERVER_URI3=pmix-server.26410;tcp4://127.0.0.1:38247
PMIX_SERVER_URI4=pmix-server.26410;tcp4://127.0.0.1:38247
PMIX_SERVER_URI41=pmix-server.26410;tcp4://127.0.0.1:38247
PMIX_SYSTEM_TMPDIR=/tmp
PMIX_VERSION=5.0.3
PROMPT_COMMAND=RET=$?;/bin/logger -t user_audit "username=$USER pid=$$ cmd=\"$(history 1 | /bin/sed "s/^[ ]*[0-9]\+[ ]*//" )\" newpwd=$PWD ret=$RET" 2>/dev/null
PROTO_HOME=/oak/stanford/groups/euan/projects/viggiano/.proto
PROTO_MODEL_CACHE=/scratch/users/viggiano/model_weights/bio-programming-tools
PRTE_MCA_plm_slurm_args=--external-launcher
PWD=/home/users/viggiano/proto-tools
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.3
PYTHONPYCACHEPREFIX=/tmp
PYTORCH_VERSION=2.2.1
SCRATCH=/scratch/users/viggiano
SHELL=/bin/bash
SHERLOCK=2
SHLVL=3
SH_DEF_TIMEOUT=30
SH_INFO_TIMEOUT=30
SH_TIMEOUT_CMD=timeout -s9
SH_USER_ENV_INITD=1
SINGULARITY_BIND=/bashrc_custom,/share/software/user/open
SINGULARITY_CONTAINER=/home/groups/brianhie/brianhie/simg/pytorch_latest.sif
SINGULARITY_ENVIRONMENT=/.singularity.d/env/91-environment.sh
SINGULARITY_NAME=pytorch_latest.sif
SLURMD_DEBUG=2
SLURMD_NODENAME=sh04-16n09
SLURM_CLUSTER_NAME=sherlock
SLURM_CONF=/var/spool/slurmd/conf-cache/slurm.conf
SLURM_CPUS_ON_NODE=4
SLURM_CPUS_PER_TASK=4
SLURM_CPU_BIND=quiet,mask_cpu:0x0000000F00000000
SLURM_CPU_BIND_LIST=0x0000000F00000000
SLURM_CPU_BIND_TYPE=mask_cpu:
SLURM_CPU_BIND_VERBOSE=quiet
SLURM_DISTRIBUTION=block
SLURM_GPUS=1
SLURM_GPUS_ON_NODE=1
SLURM_GTIDS=0
SLURM_JOBID=24471977
SLURM_JOB_ACCOUNT=euan
SLURM_JOB_CPUS_PER_NODE=4
SLURM_JOB_END_TIME=1778348057
SLURM_JOB_GID=11886
SLURM_JOB_GROUP=euan
SLURM_JOB_ID=24471977
SLURM_JOB_NAME=.ptshell-launcher.xNr4q4.sh
SLURM_JOB_NODELIST=sh04-16n09
SLURM_JOB_NUM_NODES=1
SLURM_JOB_PARTITION=brianhie
SLURM_JOB_QOS=normal
SLURM_JOB_START_TIME=1778304857
SLURM_JOB_UID=389221
SLURM_JOB_USER=viggiano
SLURM_LAUNCH_NODE_IPADDR=10.18.0.63
SLURM_LOCALID=0
SLURM_MEM_PER_CPU=10240
SLURM_MPI_TYPE=pmix
SLURM_NNODES=1
SLURM_NODEID=0
SLURM_NODELIST=sh04-16n09
SLURM_NPROCS=1
SLURM_NTASKS=1
SLURM_OOM_KILL_STEP=0
SLURM_PMIXP_ABORT_AGENT_PORT=35689
SLURM_PMIX_MAPPING_SERV=(vector,(0,1,1))
SLURM_PRIO_PROCESS=0
SLURM_PROCID=0
SLURM_PTY_PORT=39900
SLURM_PTY_WIN_COL=212
SLURM_PTY_WIN_ROW=59
SLURM_SCRIPT_CONTEXT=prolog_task
SLURM_SRUN_COMM_HOST=10.18.0.63
SLURM_SRUN_COMM_PORT=43212
SLURM_STEPID=0
SLURM_STEPMGR=sh04-16n09
SLURM_STEP_GPUS=4
SLURM_STEP_ID=0
SLURM_STEP_LAUNCHER_PORT=43212
SLURM_STEP_NODELIST=sh04-16n09
SLURM_STEP_NUM_NODES=1
SLURM_STEP_NUM_TASKS=1
SLURM_STEP_TASKS_PER_NODE=1
SLURM_SUBMIT_DIR=/home/users/viggiano
SLURM_SUBMIT_HOST=sh02-ln03.stanford.edu
SLURM_TASKS_PER_NODE=1
SLURM_TASK_PID=26431
SLURM_TOPOLOGY_ADDR=sh04.sh04-isw-16.sh04-16n09
SLURM_TOPOLOGY_ADDR_PATTERN=switch.switch.node
SLURM_TRES_PER_TASK=cpu=4
SLURM_UMASK=0022
SRCC_PATH=/share/software/user/srcc/bin
SRUN_CPUS_PER_TASK=4
SRUN_DEBUG=3
TERM=screen
TMOUT=86400
TMPDIR=/tmp
TMUX=/tmp/tmux-389221/default,254337,0
TMUX_LAUNCHED_SHERLOCK=1
TMUX_PANE=%0
USER=viggiano
USER_PATH=/share/software/user/open/nodejs/20.20.0/bin:/share/software/user/open/gcc/14.2.0/bin:/share/software/user/open/claude-code/2.1.132/bin:/home/users/viggiano/.local/bin:/home/users/viggiano/.npm-global...
XDG_CACHE_HOME=/tmp
XDG_RUNTIME_DIR=/tmp
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
_=/usr/bin/nohup
_CE_CONDA=
_CE_M=
_LMFILES_=/share/software/modules/categories/devel.lua:/share/software/modules/categories/math.lua:/share/software/modules/devel/claude-code/2.1.132.lua:/share/software/modules/devel/gcc/14.2.0.lua:/share/softw...
_ModuleTable001_=X01vZHVsZVRhYmxlXyA9IHsKTVR2ZXJzaW9uID0gMywKY19yZWJ1aWxkVGltZSA9IGZhbHNlLApjX3Nob3J0VGltZSA9IGZhbHNlLApkZXB0aFQgPSB7fSwKZmFtaWx5ID0ge30sCm1UID0gewpbImNsYXVkZS1jb2RlIl0gPSB7CmRlcFQgPSB7CmRlcEEgPSB7CnsK...
_ModuleTable002_=dGl2ZSIsCnVzZXJOYW1lID0gImNsYXVkZS1jb2RlIiwKd1YgPSAiMDAwMDAwMDAyLjAwMDAwMDAwMS4wMDAwMDAxMzIuKnpmaW5hbCIsCn0sCmRldmVsID0gewphY3Rpb25BID0gewpbW3ByZXBlbmRfcGF0aCgiTU9EVUxFUEFUSCIsIi9zaGFyZS9zb2Z0d2FyZS9t...
_ModuleTable003_=ICIvc2hhcmUvc29mdHdhcmUvbW9kdWxlcy9kZXZlbC9nY2MvMTQuMi4wLmx1YSIsCmZ1bGxOYW1lID0gImdjYy8xNC4yLjAiLApsb2FkT3JkZXIgPSA0LApwcm9wVCA9IHt9LApyZWZfY291bnQgPSAxLApzdGFja0RlcHRoID0gMSwKc3RhdHVzID0gImFjdGl2ZSIs...
_ModuleTable004_=IDIsCnByb3BUID0gewpsbW9kID0gewpzdGlja3kgPSAxLAp9LAp9LApzdGFja0RlcHRoID0gMCwKc3RhdHVzID0gImFjdGl2ZSIsCnVzZXJOYW1lID0gIm1hdGgiLAp3ViA9ICJNLip6ZmluYWwiLAp9LApub2RlanMgPSB7CmRlcFQgPSB7CmRlcEEgPSB7CnsKc24g...
_ModuleTable005_=ZSA9ICJub2RlanMvMjAuMjAuMCIsCndWID0gIjAwMDAwMDAyMC4wMDAwMDAwMjAuKnpmaW5hbCIsCn0sCn0sCm1wYXRoQSA9IHsKIi9zaGFyZS9zb2Z0d2FyZS9tb2R1bGVzL21hdGgiLCAiL3NoYXJlL3NvZnR3YXJlL21vZHVsZXMvZGV2ZWwiLCAiL3NoYXJlL3Nv...
_ModuleTable_Sz_=5
__Init_Default_Modules=1
__LMOD_REF_COUNT_CPATH=/share/software/user/open/nodejs/20.20.0/include:1;/share/software/user/open/gcc/14.2.0/include:1
__LMOD_REF_COUNT_INFOPATH=/share/software/user/open/nodejs/20.20.0/share/info:1
__LMOD_REF_COUNT_LD_LIBRARY_PATH=/share/software/user/open/nodejs/20.20.0/lib:1;/share/software/user/open/gcc/14.2.0/lib64:1;/share/software/user/open/gcc/14.2.0/lib/gcc/x86_64-pc-linux-gnu:1;/share/software/user/open/gcc/14.2.0/lib:...
__LMOD_REF_COUNT_LIBRARY_PATH=/share/software/user/open/nodejs/20.20.0/lib:1;/share/software/user/open/gcc/14.2.0/lib64:1;/share/software/user/open/gcc/14.2.0/lib:1
__LMOD_REF_COUNT_MANPATH=/share/software/user/open/nodejs/20.20.0/share/man:1;/share/software/user/open/gcc/14.2.0/share/man:1;/share/software/user/open/lmod/lmod/share/man:1;/usr/local/share/man:1;/usr/share/man:1
__LMOD_REF_COUNT_MODULEPATH=/share/software/modules/math:1;/share/software/modules/devel:1;/share/software/modules/categories:1
__LMOD_REF_COUNT_PATH=/share/software/user/open/nodejs/20.20.0/bin:1;/share/software/user/open/gcc/14.2.0/bin:1;/share/software/user/open/claude-code/2.1.132/bin:1;/home/users/viggiano/.local/bin:1;/home/users/viggiano/.np...
__LMOD_STACK_CC=false
__LMOD_STACK_CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=false
__LMOD_STACK_CPP=false
__LMOD_STACK_CXX=false
__LMOD_STACK_DISABLE_INSTALLATION_CHECKS=false
__LMOD_STACK_F77=false
__LMOD_STACK_F90=false
__LMOD_STACK_FC=false
```

### Subprocess Environment (passed to tools)

```
CONDA_PREFIX=/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/viennarna_env
CUDA_VISIBLE_DEVICES=0
DETECTED_COMPUTE_PLATFORM=cuda
DETECTED_CUDA_VERSION=12
DETECTED_DRIVER_VERSION=550
HF_HOME=/scratch/users/viggiano/model_weights/bio-programming-tools/huggingface
HOME=/home/users/viggiano
LANG=en_US.UTF-8
LC_CTYPE=C.UTF-8
LD_LIBRARY_PATH=/share/software/user/open/expat/2.2.3/lib:/share/software/user/open/curl/8.4.0/lib:/share/software/user/open/gcc/14.2.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs:/scratc...
LOGNAME=viggiano
PATH=/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/viennarna_env/bin:/home/users/viggiano/.local/bin:/share/software/user/open/git/2.45.1/bin:/share/software/user/open/nodejs/25.3.0/bi...
PIP_CACHE_DIR=/oak/stanford/groups/euan/projects/viggiano/.proto/pip_cache
PIP_DEFAULT_TIMEOUT=300
PROTO_HOME=/oak/stanford/groups/euan/projects/viggiano/.proto
PROTO_MODEL_CACHE=/scratch/users/viggiano/model_weights/bio-programming-tools
RECOMMENDED_JAX_SPEC=jax[cuda12]>=0.4.20,<1
RECOMMENDED_JAX_VARIANT=cuda12
RECOMMENDED_TORCH_INDEX=https://download.pytorch.org/whl/cu126
RECOMMENDED_TORCH_SPEC=torch>=2.5,<3
SHELL=/bin/bash
TMPDIR=/tmp
TORCH_CUDA_ARCH_LIST=9.0
TORCH_HOME=/scratch/users/viggiano/model_weights/bio-programming-tools/torch
USER=viggiano
UV_CACHE_DIR=/oak/stanford/groups/euan/projects/viggiano/.proto/uv_cache
UV_HTTP_TIMEOUT=300
VIRTUAL_ENV=/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/viennarna_env
XDG_CACHE_HOME=/tmp
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Results by Category

### Binder Design (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `germinal-design` | yes | ✅ | 1445.7s | `4c6e45c` | ✅ Pass |

### Causal Models (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `evo1-sample` | yes | ✅ | 808.8s | `4c6e45c` | ✅ Pass |
| `evo2-sample` | yes | ✅ | 408.0s | `4c6e45c` | ✅ Pass |
| `progen2-sample` | yes | ✅ | 365.6s | `4c6e45c` | ✅ Pass |
| `progen3-sample` | yes | ✅ | 630.6s | `4c6e45c` | ✅ Pass |

### Gene Annotation (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `crispr-tracr-rna` | no | ✅ | 255.5s | `4c6e45c` | ✅ Pass |
| `minced-crispr` | no | ✅ | 15.4s | `4c6e45c` | ✅ Pass |
| `promoter-calculator` | no | ✅ | 27.4s | `4c6e45c` | ✅ Pass |
| `pyhmmer-hmmscan` | no | ✅ | 20.2s | `4c6e45c` | ✅ Pass |

### Inverse Folding (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `esm-if1-sample` | yes | ✅ | 145.2s | `4c6e45c` | ✅ Pass |
| `fampnn-pack` | yes | ✅ | 263.1s | `4c6e45c` | ✅ Pass |
| `ligandmpnn-sample` | yes | ✅ | 267.4s | `4c6e45c` | ✅ Pass |
| `proteinmpnn-gradient` | yes | ✅ | 154.0s | `4c6e45c` | ✅ Pass |

### Masked Models (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `ablang-embedding` | yes | ✅ | 786.6s | `4c6e45c` | ✅ Pass |
| `esm2-embedding` | yes | ✅ | 94.5s | `4c6e45c` | ✅ Pass |
| `esm3-embedding` | yes | ✅ | 98.3s | `4c6e45c` | ✅ Pass |
| `esmc-embedding` | yes | ✅ | 7.8s | `4c6e45c` | ✅ Pass |

### Mutagenesis (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `random-nucleotide-sample` | no | - | 0.0s | `4c6e45c` | ✅ Pass |
| `random-protein-sample` | no | - | 0.0s | `4c6e45c` | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `orfipy-prediction` | no | ✅ | 17.8s | `4c6e45c` | ✅ Pass |
| `prodigal-prediction` | no | ✅ | 16.7s | `4c6e45c` | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `splice-transformer-prediction` | yes | ✅ | 230.0s | `4c6e45c` | ✅ Pass |

### Sequence Alignment (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `blast-create-db` | no | ✅ | 26.9s | `4c6e45c` | ✅ Pass |
| `colabfold-search` | no | ✅ | 37.4s | `4c6e45c` | ✅ Pass |
| `mafft-align` | no | ✅ | 22.5s | `4c6e45c` | ✅ Pass |
| `mmseqs2-clustering` | no | ✅ | 36.1s | `4c6e45c` | ✅ Pass |

### Sequence Scoring (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphagenome-predict-intervals` | yes | ✅ | 1055.5s | `4c6e45c` | ✅ Pass |
| `borzoi-ensemble` | yes | ✅ | 332.0s | `4c6e45c` | ✅ Pass |
| `enformer-prediction` | yes | ✅ | 142.3s | `4c6e45c` | ✅ Pass |
| `segmasker-score` | no | ✅ | 32.4s | `4c6e45c` | ✅ Pass |

### Structure Alignment (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `foldmason-msa` | no | - | 1.3s | `4c6e45c` | ✅ Pass |
| `foldseek-cluster` | no | ✅ | 16.9s | `4c6e45c` | ✅ Pass |
| `tmalign-alignment` | no | ✅ | 41.8s | `4c6e45c` | ✅ Pass |
| `usalign-alignment` | no | ✅ | 35.5s | `4c6e45c` | ✅ Pass |

### Structure Design (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `bindcraft-design` | yes | ✅ | 347.8s | `4c6e45c` | ✅ Pass |
| `rfdiffusion3-design` | yes | ✅ | 593.7s | `4c6e45c` | ✅ Pass |

### Structure Dynamics (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `bioemu-sample` | yes | ✅ | 374.6s | `4c6e45c` | ✅ Pass |

### Structure Prediction (6/7)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphafold2-binder` | yes | ✅ | 325.7s | `4c6e45c` | ✅ Pass |
| `alphafold3-prediction` | yes | ✅ | 10.1s | `4c6e45c` | ❌ Fail |
| `boltz2-prediction` | yes | ✅ | 498.1s | `4c6e45c` | ✅ Pass |
| `chai1-prediction` | yes | ✅ | 541.2s | `4c6e45c` | ✅ Pass |
| `esmfold-gradient` | yes | ✅ | 94.3s | `4c6e45c` | ✅ Pass |
| `protenix-prediction` | yes | ✅ | 685.7s | `4c6e45c` | ✅ Pass |
| `viennarna-prediction` | no | ✅ | 29.6s | `4c6e45c` | ✅ Pass |

### Structure Scoring (5/5)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `dssp-secondary-structure` | no | ✅ | 32.2s | `4c6e45c` | ✅ Pass |
| `ipsae-scoring` | no | ✅ | 22.9s | `4c6e45c` | ✅ Pass |
| `pdockq2` | no | - | 0.1s | `4c6e45c` | ✅ Pass |
| `pyrosetta-energy` | no | ✅ | 522.2s | `4c6e45c` | ✅ Pass |
| `structure-metrics` | no | - | 2.1s | `4c6e45c` | ✅ Pass |

### Testing (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `mock-cli-multi-gpu-tool-run` | yes | - | - | `4c6e45c` | ⏭️ Skip |
| `mock-cli-tool-run` | yes | ✅ | 12.2s | `4c6e45c` | ✅ Pass |
| `mock-jax-multi-gpu-tool-run` | yes | - | - | `4c6e45c` | ⏭️ Skip |
| `mock-jax-tool-run` | yes | ✅ | 398.4s | `4c6e45c` | ✅ Pass |
| `mock-pytorch-multi-gpu-tool-run` | yes | - | - | `4c6e45c` | ⏭️ Skip |
| `mock-pytorch-tool-run` | yes | ✅ | 169.1s | `4c6e45c` | ✅ Pass |

## Failure Details

### ❌ `alphafold3-prediction`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold3-prediction]`

```
tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool alphafold3-prediction failed: ["MissingAssetError: alphafold3: weights not provisioned\nLicense / access: https://github.com/google-deepmind/alphafold3#obtaining-model-parameters\nProvisioning steps:\n  AlphaFold3 weights are gated by DeepMind's Terms of Use and are NOT\n  automatically downloaded. To obtain access:\n    1. Request access via DeepMind's form (link above).\n    2. After approval (2-3 business days), download the weights archive\n       from the link DeepMind emails you.\n    3. Place af3.bin (or af3.bin.zst) in the resolved directory above,\n       OR point PROTO_ALPHAFOLD3_WEIGHTS_DIR at the directory containing it.\n  See notes/storage.md for PROTO_MODEL_CACHE / PROTO_HOME rules.", 'Traceback (most recent call last):\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/tools/tool_registry.py", line 601, in _wrapper_body\n    result = func(inputs, config, instance)\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/tools/structure_prediction/alphafold3/alphafold3.py", line 327, in run_alphafold3\n    output_data = ToolInstance.dispatch(\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/utils/tool_instance.py", line 414, in dispatch\n    return cached.run(\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/utils/tool_instance.py", line 777, in run\n    return self._run_persistent(\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/utils/tool_instance.py", line 1028, in _run_persistent\n    self._ensure_env()\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/utils/tool_instance.py", line 745, in _ensure_env\n    self._create_env()\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/utils/tool_instance.py", line 2186, in _create_env\n    raise MissingAssetError(toolkit, asset_kind, tail)\nproto_tools.utils.tool_io.MissingAssetError: alphafold3: weights not provisioned\nLicense / access: https://github.com/google-deepmind/alphafold3#obtaining-model-parameters\nProvisioning steps:\n  AlphaFold3 weights are gated by DeepMind\'s Terms of Use and are NOT\n  automatically downloaded. To obtain access:\n    1. Request access via DeepMind\'s form (link above).\n    2. After approval (2-3 business days), download the weights archive\n       from the link DeepMind emails you.\n    3. Place af3.bin (or af3.bin.zst) in the resolved directory above,\n       OR point PROTO_ALPHAFOLD3_WEIGHTS_DIR at the directory containing it.\n  See notes/storage.md for PROTO_MODEL_CACHE / PROTO_HOME rules.\n']
E   assert False
E    +  where False = <[ToolExecutionError('alphafold3-prediction: cannot read field \'structures\' — tool failed: MissingAssetError: alphafold3: weights not provisioned\nLicense / access: https://github.com/google-deepmind/alphafold3#obtaining-model-parameters\nProvisioning steps:\n  AlphaFold3 weights are gated by DeepMind\'s Terms of Use and are NOT\n  automatically downloaded. To obtain access:\n    1. Request access via DeepMind\'s form (link above).\n    2. After approval (2-3 business days), download the weights archive\n       from the link DeepMind emails you.\n    3. Place af3.bin (or af3.bin.zst) in the resolved directory above,\n       OR point PROTO_ALPHAFOLD3_WEIGHTS_DIR at the directory containing it.\n  See notes/storage.md for PROTO_MODEL_CACHE / PROTO_HOME rules. | Traceback (most recent call last):\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/tools/tool_registry.py", line 601, in _wrapper_body\n    result = func(inputs, config, instance)\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/tools/structure_prediction/alphafold3/alphafold3.py", line 327, in run_alphafold3\n    output_data = ToolInstance.dispatch(\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/utils/tool_instance.py", line 414, in dispatch\n    return cached.run(\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/utils/tool_instance.py", line 777, in run\n    return self._run_persistent(\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/utils/tool_instance.py", line 1028, in _run_persistent\n    self._ensure_env()\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/utils/tool_instance.py", line 745, in _ensure_env\n    self._create_env()\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/utils/tool_instance.py", line 2186, in _create_env\n    raise MissingAssetError(toolkit, asset_kind, tail)\nproto_tools.utils.tool_io.MissingAssetError: alphafold3: weights not provisioned\nLicense / access: https://github.com/google-deepmind/alphafold3#obtaining-model-parameters\nProvisioning steps:\n  AlphaFold3 weights are gated by DeepMind\'s Terms of Use and are NOT\n  automatically downloaded. To obtain access:\n    1. Request access via DeepMind\'s form (link above).\n    2. After approval (2-3 business days), download the weights archive\n       from the link DeepMind emails you.\n    3. Place af3.bin (or af3.bin.zst) in the resolved directory above,\n       OR point PROTO_ALPHAFOLD3_WEIGHTS_DIR at the directory containing it.\n  See notes/storage.md for PROTO_MODEL_CACHE / PROTO_HOME rules.\n') raised in repr()] AlphaFold3Output object at 0x7f53077f92b0>.success
```

---
*Generated at 2026-05-09 03:08:02 by `pytest --env-report`*

<!-- env-report-data
[
  {
    "tool_key": "ablang-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ablang-embedding]",
    "status": "passed",
    "duration_seconds": 786.56,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/ablang_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "alphafold2-binder",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold2-binder]",
    "status": "passed",
    "duration_seconds": 325.71,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/alphafold2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "alphafold3-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold3-prediction]",
    "status": "failed",
    "duration_seconds": 10.08,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/alphafold3_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool alphafold3-prediction failed: [\"MissingAssetError: alphafold3: weights not provisioned\\nLicense / access: https://github.com/google-deepmind/alphafold3#obtaining-model-parameters\\nProvisioning steps:\\n  AlphaFold3 weights are gated by DeepMind's Terms of Use and are NOT\\n  automatically downloaded. To obtain access:\\n    1. Request access via DeepMind's form (link above).\\n    2. After approval (2-3 business days), download the weights archive\\n       from the link DeepMind emails you.\\n    3. Place af3.bin (or af3.bin.zst) in the resolved directory above,\\n       OR point PROTO_ALPHAFOLD3_WEIGHTS_DIR at the directory containing it.\\n  See notes/storage.md for PROTO_MODEL_CACHE / PROTO_HOME rules.\", 'Traceback (most recent call last):\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/tools/tool_registry.py\", line 601, in _wrapper_body\\n    result = func(inputs, config, instance)\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/tools/structure_prediction/alphafold3/alphafold3.py\", line 327, in run_alphafold3\\n    output_data = ToolInstance.dispatch(\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/utils/tool_instance.py\", line 414, in dispatch\\n    return cached.run(\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/utils/tool_instance.py\", line 777, in run\\n    return self._run_persistent(\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/utils/tool_instance.py\", line 1028, in _run_persistent\\n    self._ensure_env()\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/utils/tool_instance.py\", line 745, in _ensure_env\\n    self._create_env()\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/utils/tool_instance.py\", line 2186, in _create_env\\n    raise MissingAssetError(toolkit, asset_kind, tail)\\nproto_tools.utils.tool_io.MissingAssetError: alphafold3: weights not provisioned\\nLicense / access: https://github.com/google-deepmind/alphafold3#obtaining-model-parameters\\nProvisioning steps:\\n  AlphaFold3 weights are gated by DeepMind\\'s Terms of Use and are NOT\\n  automatically downloaded. To obtain access:\\n    1. Request access via DeepMind\\'s form (link above).\\n    2. After approval (2-3 business days), download the weights archive\\n       from the link DeepMind emails you.\\n    3. Place af3.bin (or af3.bin.zst) in the resolved directory above,\\n       OR point PROTO_ALPHAFOLD3_WEIGHTS_DIR at the directory containing it.\\n  See notes/storage.md for PROTO_MODEL_CACHE / PROTO_HOME rules.\\n']\nE   assert False\nE    +  where False = <[ToolExecutionError('alphafold3-prediction: cannot read field \\'structures\\' \u2014 tool failed: MissingAssetError: alphafold3: weights not provisioned\\nLicense / access: https://github.com/google-deepmind/alphafold3#obtaining-model-parameters\\nProvisioning steps:\\n  AlphaFold3 weights are gated by DeepMind\\'s Terms of Use and are NOT\\n  automatically downloaded. To obtain access:\\n    1. Request access via DeepMind\\'s form (link above).\\n    2. After approval (2-3 business days), download the weights archive\\n       from the link DeepMind emails you.\\n    3. Place af3.bin (or af3.bin.zst) in the resolved directory above,\\n       OR point PROTO_ALPHAFOLD3_WEIGHTS_DIR at the directory containing it.\\n  See notes/storage.md for PROTO_MODEL_CACHE / PROTO_HOME rules. | Traceback (most recent call last):\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/tools/tool_registry.py\", line 601, in _wrapper_body\\n    result = func(inputs, config, instance)\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/tools/structure_prediction/alphafold3/alphafold3.py\", line 327, in run_alphafold3\\n    output_data = ToolInstance.dispatch(\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/utils/tool_instance.py\", line 414, in dispatch\\n    return cached.run(\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/utils/tool_instance.py\", line 777, in run\\n    return self._run_persistent(\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/utils/tool_instance.py\", line 1028, in _run_persistent\\n    self._ensure_env()\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/utils/tool_instance.py\", line 745, in _ensure_env\\n    self._create_env()\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/proto_tools/utils/tool_instance.py\", line 2186, in _create_env\\n    raise MissingAssetError(toolkit, asset_kind, tail)\\nproto_tools.utils.tool_io.MissingAssetError: alphafold3: weights not provisioned\\nLicense / access: https://github.com/google-deepmind/alphafold3#obtaining-model-parameters\\nProvisioning steps:\\n  AlphaFold3 weights are gated by DeepMind\\'s Terms of Use and are NOT\\n  automatically downloaded. To obtain access:\\n    1. Request access via DeepMind\\'s form (link above).\\n    2. After approval (2-3 business days), download the weights archive\\n       from the link DeepMind emails you.\\n    3. Place af3.bin (or af3.bin.zst) in the resolved directory above,\\n       OR point PROTO_ALPHAFOLD3_WEIGHTS_DIR at the directory containing it.\\n  See notes/storage.md for PROTO_MODEL_CACHE / PROTO_HOME rules.\\n') raised in repr()] AlphaFold3Output object at 0x7f53077f92b0>.success",
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "alphagenome-predict-intervals",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphagenome-predict-intervals]",
    "status": "passed",
    "duration_seconds": 1055.45,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/alphagenome_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "bindcraft-design",
    "category": "structure_design",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[bindcraft-design]",
    "status": "passed",
    "duration_seconds": 347.76,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/bindcraft_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "bioemu-sample",
    "category": "structure_dynamics",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[bioemu-sample]",
    "status": "passed",
    "duration_seconds": 374.56,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/bioemu_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "blast-create-db",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[blast-create-db]",
    "status": "passed",
    "duration_seconds": 26.87,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/blast_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "boltz2-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[boltz2-prediction]",
    "status": "passed",
    "duration_seconds": 498.15,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/boltz2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "borzoi-ensemble",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[borzoi-ensemble]",
    "status": "passed",
    "duration_seconds": 332.04,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/borzoi_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "chai1-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[chai1-prediction]",
    "status": "passed",
    "duration_seconds": 541.21,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/chai1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "colabfold-search",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[colabfold-search]",
    "status": "passed",
    "duration_seconds": 37.4,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/colabfold_search_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "crispr-tracr-rna",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[crispr-tracr-rna]",
    "status": "passed",
    "duration_seconds": 255.54,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/crispr_tracr_rna_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "dssp-secondary-structure",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[dssp-secondary-structure]",
    "status": "passed",
    "duration_seconds": 32.2,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/dssp_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "enformer-prediction",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[enformer-prediction]",
    "status": "passed",
    "duration_seconds": 142.29,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/enformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "esm-if1-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm-if1-sample]",
    "status": "passed",
    "duration_seconds": 145.21,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/esm_if1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "esm2-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm2-embedding]",
    "status": "passed",
    "duration_seconds": 94.45,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/esm2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "esm3-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm3-embedding]",
    "status": "passed",
    "duration_seconds": 98.3,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/evolutionaryscale_esm_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "esmc-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esmc-embedding]",
    "status": "passed",
    "duration_seconds": 7.83,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/evolutionaryscale_esm_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "esmfold-gradient",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esmfold-gradient]",
    "status": "passed",
    "duration_seconds": 94.35,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/esmfold_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "evo1-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo1-sample]",
    "status": "passed",
    "duration_seconds": 808.8,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/evo1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "evo2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo2-sample]",
    "status": "passed",
    "duration_seconds": 407.96,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/evo2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "fampnn-pack",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[fampnn-pack]",
    "status": "passed",
    "duration_seconds": 263.1,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/fampnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "foldmason-msa",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[foldmason-msa]",
    "status": "passed",
    "duration_seconds": 1.33,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "foldseek-cluster",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[foldseek-cluster]",
    "status": "passed",
    "duration_seconds": 16.87,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/foldseek_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "germinal-design",
    "category": "binder_design",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[germinal-design]",
    "status": "passed",
    "duration_seconds": 1445.68,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/germinal_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "ipsae-scoring",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ipsae-scoring]",
    "status": "passed",
    "duration_seconds": 22.85,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/ipsae_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "ligandmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ligandmpnn-sample]",
    "status": "passed",
    "duration_seconds": 267.37,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/ligandmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "mafft-align",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mafft-align]",
    "status": "passed",
    "duration_seconds": 22.53,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/mafft_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "minced-crispr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[minced-crispr]",
    "status": "passed",
    "duration_seconds": 15.38,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/minced_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "mmseqs2-clustering",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mmseqs2-clustering]",
    "status": "passed",
    "duration_seconds": 36.06,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/mmseqs2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
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
    "error_message": "('/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/tests/tool_infra_tests/test_env_report.py', 72, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "mock-cli-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-cli-tool-run]",
    "status": "passed",
    "duration_seconds": 12.16,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/mock_cli_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
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
    "error_message": "('/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/tests/tool_infra_tests/test_env_report.py', 72, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "mock-jax-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-jax-tool-run]",
    "status": "passed",
    "duration_seconds": 398.45,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/mock_jax_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
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
    "error_message": "('/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/tests/tool_infra_tests/test_env_report.py', 72, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "mock-pytorch-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-pytorch-tool-run]",
    "status": "passed",
    "duration_seconds": 169.14,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/mock_pytorch_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "orfipy-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[orfipy-prediction]",
    "status": "passed",
    "duration_seconds": 17.81,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/orfipy_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "pdockq2",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pdockq2]",
    "status": "passed",
    "duration_seconds": 0.13,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "prodigal-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[prodigal-prediction]",
    "status": "passed",
    "duration_seconds": 16.74,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/prodigal_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "progen2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen2-sample]",
    "status": "passed",
    "duration_seconds": 365.64,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/progen2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "progen3-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen3-sample]",
    "status": "passed",
    "duration_seconds": 630.62,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/progen3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "promoter-calculator",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[promoter-calculator]",
    "status": "passed",
    "duration_seconds": 27.41,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/promoter_calculator_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "proteinmpnn-gradient",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[proteinmpnn-gradient]",
    "status": "passed",
    "duration_seconds": 154.04,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/proteinmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "protenix-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[protenix-prediction]",
    "status": "passed",
    "duration_seconds": 685.74,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/protenix_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "pyhmmer-hmmscan",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pyhmmer-hmmscan]",
    "status": "passed",
    "duration_seconds": 20.21,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/pyhmmer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "pyrosetta-energy",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pyrosetta-energy]",
    "status": "passed",
    "duration_seconds": 522.25,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/pyrosetta_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "random-nucleotide-sample",
    "category": "mutagenesis",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[random-nucleotide-sample]",
    "status": "passed",
    "duration_seconds": 0.04,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
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
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "rfdiffusion3-design",
    "category": "structure_design",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[rfdiffusion3-design]",
    "status": "passed",
    "duration_seconds": 593.68,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/rfdiffusion3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "segmasker-score",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[segmasker-score]",
    "status": "passed",
    "duration_seconds": 32.4,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/segmasker_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "splice-transformer-prediction",
    "category": "rna_splicing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[splice-transformer-prediction]",
    "status": "passed",
    "duration_seconds": 230.03,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/splice_transformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "structure-metrics",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[structure-metrics]",
    "status": "passed",
    "duration_seconds": 2.06,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "tmalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[tmalign-alignment]",
    "status": "passed",
    "duration_seconds": 41.83,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/tmalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "usalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[usalign-alignment]",
    "status": "passed",
    "duration_seconds": 35.53,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/usalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  },
  {
    "tool_key": "viennarna-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[viennarna-prediction]",
    "status": "passed",
    "duration_seconds": 29.55,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/viennarna_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "4c6e45c77832",
    "git_dirty": false
  }
]
-->