# Linux x86_64 Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-100%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-14-brightgreen) ![Failed](https://img.shields.io/badge/failed-0-brightgreen) ![Skipped](https://img.shields.io/badge/skipped-0-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Linux Linux 3.10.0-1160.139.1.el7.tuxcare.els4.x86_64 |
| **Architecture** | x86_64 |
| **Hostname** | `sh04-15n01.int` |
| **Python** | 3.12.13 |
| **RAM** | 2015.1 GB |
| **GPU** | 1× NVIDIA H100 80GB HBM3 |
| **CUDA** | 12.4 |
| **Conda Env** | `bio-tools` |

## Git

- **Commit**: `d9f27a9d35b6`
- **Branch**: `bv/env_stability`
- **Dirty**: No

## Environment Variables

### Parent Process Environment

```
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
BLIS_NUM_THREADS=8
CC=gcc
CLAUDECODE=1
CLAUDE_CODE_ENTRYPOINT=cli
COMMON_DATASETS=/oak/stanford/datasets/common
CONDA_DEFAULT_ENV=bio-tools
CONDA_EXE=/home/users/viggiano/miniconda3/bin/conda
CONDA_PREFIX=/home/groups/euan/viggiano/envs/bio-tools
CONDA_PREFIX_1=/home/users/viggiano/miniconda3
CONDA_PROMPT_MODIFIER=(bio-tools) 
CONDA_PYTHON_EXE=/home/users/viggiano/miniconda3/bin/python
CONDA_SHLVL=2
COREPACK_ENABLE_AUTO_PIN=0
CPATH=/share/software/user/open/nodejs/20.20.0/include:/share/software/user/open/gcc/14.2.0/include
CPP=cpp
CUDA_VISIBLE_DEVICES=0
CXX=c++
DISABLE_PANDERA_IMPORT_WARNING=True
F77=gfortran
F90=gfortran
FC=gfortran
GIT_EDITOR=true
GOTO_NUM_THREADS=8
GROUP=euan
GROUP_HOME=/home/groups/euan
GROUP_SCRATCH=/scratch/groups/euan
HISTCONTROL=ignoreboth:erasedups
HISTSIZE=1000
HOME=/home/users/viggiano
HOSTNAME=sh04-15n01.int
HYDRA_BOOTSTRAP=slurm
HYDRA_LAUNCHER_EXTRA_ARGS=--external-launcher
INFOPATH=/share/software/user/open/nodejs/20.20.0/share/info
I_MPI_HYDRA_BOOTSTRAP=slurm
I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS=--external-launcher
KRB5CCNAME=FILE:/tmp/krb5cc_389221_JZZHYso4EB
LANG=en_US.UTF-8
LC_CTYPE=C.UTF-8
LD_LIBRARY_PATH=/share/software/user/open/gcc/14.2.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs
LESSOPEN=||/usr/bin/lesspipe.sh %s
LIBRARY_PATH=/share/software/user/open/nodejs/20.20.0/lib:/share/software/user/open/gcc/14.2.0/lib64:/share/software/user/open/gcc/14.2.0/lib
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
LOADEDMODULES=devel:math:gcc/14.2.0:nodejs/20.20.0
LOCAL_SCRATCH=/lscratch/viggiano
LOGNAME=viggiano
LS_COLORS=su=00:sg=00:ca=00:or=40;31;01
L_SCRATCH=/lscratch/viggiano
L_SCRATCH_JOB=/lscratch/viggiano/19263436
L_SCRATCH_USER=/lscratch/viggiano
MAIL=/var/spool/mail/viggiano
MANPATH=/share/software/user/open/nodejs/20.20.0/share/man:/share/software/user/open/gcc/14.2.0/share/man:/share/software/user/open/lmod/lmod/share/man:/usr/local/share/man:/usr/share/man
MKL_NUM_THREADS=8
MODULEPATH=/share/software/modules/math:/share/software/modules/devel:/share/software/modules/categories
MODULEPATH_ROOT=/share/software/modules
MODULESHOME=/share/software/user/open/lmod/9.0.2
NVIDIA_DRIVER_CAPABILITIES=compute,utility
NVIDIA_VISIBLE_DEVICES=all
NoDefaultCurrentDirectoryInExePath=1
OAK=/oak/stanford/groups/euan
OLDPWD=/home/users/viggiano/oak_main/codebases
OMPI_MCA_orte_precondition_transports=0125efcc00000000-0125efcc00000000
OMPI_MCA_plm_slurm_args=--external-launcher
OMP_NUM_THREADS=8
OPENBLAS_NUM_THREADS=8
OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta
PATH=/home/users/viggiano/.local/bin:/share/software/user/open/nodejs/25.3.0/bin:/share/software/user/open/claude-code/2.1.69/bin:/home/users/viggiano/.npm-global/bin:/home/groups/euan/viggiano/envs/bio-to...
PMIX_BFROP_BUFFER_TYPE=PMIX_BFROP_BUFFER_NON_DESC
PMIX_GDS_MODULE=shmem2,hash
PMIX_HOSTNAME=sh04-15n01
PMIX_NAMESPACE=slurm.pmix.19263436.0
PMIX_RANK=0
PMIX_SECURITY_MODE=native
PMIX_SERVER_TMPDIR=/var/spool/slurmd/pmix.19263436.0/
PMIX_SERVER_URI2=pmix-server.9359;tcp4://127.0.0.1:42506
PMIX_SERVER_URI21=pmix-server.9359;tcp4://127.0.0.1:42506
PMIX_SERVER_URI3=pmix-server.9359;tcp4://127.0.0.1:42506
PMIX_SERVER_URI4=pmix-server.9359;tcp4://127.0.0.1:42506
PMIX_SERVER_URI41=pmix-server.9359;tcp4://127.0.0.1:42506
PMIX_SYSTEM_TMPDIR=/tmp
PMIX_VERSION=5.0.3
PROMPT_COMMAND=RET=$?;/bin/logger -t user_audit "username=$USER pid=$$ cmd=\"$(history 1 | /bin/sed "s/^[ ]*[0-9]\+[ ]*//" )\" newpwd=$PWD ret=$RET" 2>/dev/null
PRTE_MCA_plm_slurm_args=--external-launcher
PWD=/home/users/viggiano/oak_main/codebases/bio-programming-tools
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
PYTHONPYCACHEPREFIX=/tmp
PYTORCH_VERSION=2.2.1
RDBASE=/home/groups/euan/viggiano/envs/bio-tools/lib/python3.12/site-packages/rdkit
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
SLURMD_NODENAME=sh04-15n01
SLURM_CLUSTER_NAME=sherlock
SLURM_CONF=/var/spool/slurmd/conf-cache/slurm.conf
SLURM_CPUS_ON_NODE=8
SLURM_CPUS_PER_TASK=8
SLURM_CPU_BIND=quiet,mask_cpu:0x0000000000011E1C
SLURM_CPU_BIND_LIST=0x0000000000011E1C
SLURM_CPU_BIND_TYPE=mask_cpu:
SLURM_CPU_BIND_VERBOSE=quiet
SLURM_DISTRIBUTION=block
SLURM_GPUS=1
SLURM_GPUS_ON_NODE=1
SLURM_GTIDS=0
SLURM_JOBID=19263436
SLURM_JOB_ACCOUNT=euan
SLURM_JOB_CPUS_PER_NODE=8
SLURM_JOB_END_TIME=1774238226
SLURM_JOB_GID=11886
SLURM_JOB_GROUP=euan
SLURM_JOB_ID=19263436
SLURM_JOB_NAME=.ptshell-launcher.prjLqr.sh
SLURM_JOB_NODELIST=sh04-15n01
SLURM_JOB_NUM_NODES=1
SLURM_JOB_PARTITION=brianhie
SLURM_JOB_QOS=normal
SLURM_JOB_START_TIME=1774195022
SLURM_JOB_UID=389221
SLURM_JOB_USER=viggiano
SLURM_LAUNCH_NODE_IPADDR=10.18.0.64
SLURM_LOCALID=0
SLURM_MEM_PER_CPU=30720
SLURM_MPI_TYPE=pmix
SLURM_NNODES=1
SLURM_NODEID=0
SLURM_NODELIST=sh04-15n01
SLURM_NPROCS=1
SLURM_NTASKS=1
SLURM_OOM_KILL_STEP=0
SLURM_PMIXP_ABORT_AGENT_PORT=37194
SLURM_PMIX_MAPPING_SERV=(vector,(0,1,1))
SLURM_PRIO_PROCESS=0
SLURM_PROCID=0
SLURM_PTY_PORT=41402
SLURM_PTY_WIN_COL=212
SLURM_PTY_WIN_ROW=54
SLURM_SCRIPT_CONTEXT=prolog_task
SLURM_SRUN_COMM_HOST=10.18.0.64
SLURM_SRUN_COMM_PORT=43865
SLURM_STEPID=0
SLURM_STEPMGR=sh04-15n01
SLURM_STEP_GPUS=3
SLURM_STEP_ID=0
SLURM_STEP_LAUNCHER_PORT=43865
SLURM_STEP_NODELIST=sh04-15n01
SLURM_STEP_NUM_NODES=1
SLURM_STEP_NUM_TASKS=1
SLURM_STEP_TASKS_PER_NODE=1
SLURM_SUBMIT_DIR=/home/users/viggiano
SLURM_SUBMIT_HOST=sh02-ln04.stanford.edu
SLURM_TASKS_PER_NODE=1
SLURM_TASK_PID=9397
SLURM_TOPOLOGY_ADDR=sh04.sh04-isw-15.sh04-15n01
SLURM_TOPOLOGY_ADDR_PATTERN=switch.switch.node
SLURM_TRES_PER_TASK=cpu=8
SLURM_UMASK=0022
SRCC_PATH=/share/software/user/srcc/bin
SRUN_CPUS_PER_TASK=8
SRUN_DEBUG=3
TERM=screen
TMOUT=86400
TMPDIR=/tmp
TMUX=/tmp/tmux-389221/default,178411,0
TMUX_LAUNCHED_SHERLOCK=1
TMUX_PANE=%0
USER=viggiano
USER_PATH=/share/software/user/open/nodejs/20.20.0/bin:/share/software/user/open/gcc/14.2.0/bin:/home/users/viggiano/.local/bin:/home/users/viggiano/.npm-global/bin:/home/users/viggiano/miniconda3/bin:/home/use...
XDG_CACHE_HOME=/tmp
XDG_RUNTIME_DIR=/tmp
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
_=/home/groups/euan/viggiano/envs/bio-tools/bin/python
_CE_CONDA=
_CE_M=
_LMFILES_=/share/software/modules/categories/devel.lua:/share/software/modules/categories/math.lua:/share/software/modules/devel/gcc/14.2.0.lua:/share/software/modules/devel/nodejs/20.20.0.lua
_ModuleTable001_=X01vZHVsZVRhYmxlXyA9IHsKTVR2ZXJzaW9uID0gMywKY19yZWJ1aWxkVGltZSA9IGZhbHNlLApjX3Nob3J0VGltZSA9IGZhbHNlLApkZXB0aFQgPSB7fSwKZmFtaWx5ID0ge30sCm1UID0gewpkZXZlbCA9IHsKYWN0aW9uQSA9IHsKW1twcmVwZW5kX3BhdGgoIk1P...
_ModuleTable002_=Ik0uKnpmaW5hbCIsCn0sCmdjYyA9IHsKZm4gPSAiL3NoYXJlL3NvZnR3YXJlL21vZHVsZXMvZGV2ZWwvZ2NjLzE0LjIuMC5sdWEiLApmdWxsTmFtZSA9ICJnY2MvMTQuMi4wIiwKbG9hZE9yZGVyID0gMywKcHJvcFQgPSB7fSwKcmVmX2NvdW50ID0gMSwKc3RhY2tE...
_ModuleTable003_=bGxOYW1lID0gIm1hdGgiLApsb2FkT3JkZXIgPSAyLApwcm9wVCA9IHsKbG1vZCA9IHsKc3RpY2t5ID0gMSwKfSwKfSwKc3RhY2tEZXB0aCA9IDAsCnN0YXR1cyA9ICJhY3RpdmUiLAp1c2VyTmFtZSA9ICJtYXRoIiwKd1YgPSAiTS4qemZpbmFsIiwKfSwKbm9kZWpz...
_ModuleTable004_=LApzdGF0dXMgPSAiYWN0aXZlIiwKdXNlck5hbWUgPSAibm9kZWpzLzIwLjIwLjAiLAp3ViA9ICIwMDAwMDAwMjAuMDAwMDAwMDIwLip6ZmluYWwiLAp9LAp9LAptcGF0aEEgPSB7CiIvc2hhcmUvc29mdHdhcmUvbW9kdWxlcy9tYXRoIiwgIi9zaGFyZS9zb2Z0d2Fy...
_ModuleTable_Sz_=4
__Init_Default_Modules=1
__LMOD_REF_COUNT_CPATH=/share/software/user/open/nodejs/20.20.0/include:1;/share/software/user/open/gcc/14.2.0/include:1
__LMOD_REF_COUNT_INFOPATH=/share/software/user/open/nodejs/20.20.0/share/info:1
__LMOD_REF_COUNT_LD_LIBRARY_PATH=/share/software/user/open/nodejs/20.20.0/lib:1;/share/software/user/open/gcc/14.2.0/lib64:1;/share/software/user/open/gcc/14.2.0/lib/gcc/x86_64-pc-linux-gnu:1;/share/software/user/open/gcc/14.2.0/lib:...
__LMOD_REF_COUNT_LIBRARY_PATH=/share/software/user/open/nodejs/20.20.0/lib:1;/share/software/user/open/gcc/14.2.0/lib64:1;/share/software/user/open/gcc/14.2.0/lib:1
__LMOD_REF_COUNT_MANPATH=/share/software/user/open/nodejs/20.20.0/share/man:1;/share/software/user/open/gcc/14.2.0/share/man:1;/share/software/user/open/lmod/lmod/share/man:1;/usr/local/share/man:1;/usr/share/man:1
__LMOD_REF_COUNT_MODULEPATH=/share/software/modules/math:1;/share/software/modules/devel:1;/share/software/modules/categories:1
__LMOD_REF_COUNT_PATH=/share/software/user/open/nodejs/20.20.0/bin:1;/share/software/user/open/gcc/14.2.0/bin:1;/home/users/viggiano/.local/bin:1;/home/users/viggiano/.npm-global/bin:1;/home/users/viggiano/miniconda3/bin:1...
__LMOD_STACK_CC=false
__LMOD_STACK_CPP=false
__LMOD_STACK_CXX=false
__LMOD_STACK_F77=false
__LMOD_STACK_F90=false
__LMOD_STACK_FC=false
```

### Subprocess Environment (passed to tools)

```
CONDA_PREFIX=/oak/stanford/groups/euan/projects/viggiano/codebases/bio-programming-tools/tool_envs/esm3_env
CUDA_VISIBLE_DEVICES=0
DETECTED_COMPUTE_PLATFORM=cuda
DETECTED_CUDA_VERSION=12
DETECTED_DRIVER_VERSION=550
HOME=/home/users/viggiano
LANG=en_US.UTF-8
LC_CTYPE=C.UTF-8
LD_LIBRARY_PATH=/share/software/user/open/gcc/14.2.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs:/home/groups/euan/viggiano/envs/bio-tools/lib
LOGNAME=viggiano
PATH=/oak/stanford/groups/euan/projects/viggiano/codebases/bio-programming-tools/tool_envs/esm3_env/bin:/usr/local/cuda/bin:/home/users/viggiano/.local/bin:/share/software/user/open/nodejs/25.3.0/bin:/shar...
RECOMMENDED_JAX_SPEC=jax[cuda12]>=0.4.20,<1
RECOMMENDED_JAX_VARIANT=cuda12
RECOMMENDED_TORCH_INDEX=https://download.pytorch.org/whl/cu126
RECOMMENDED_TORCH_SPEC=torch>=2.5,<3
SHELL=/bin/bash
TMPDIR=/tmp
TORCH_CUDA_ARCH_LIST=9.0
TORCH_HOME=/oak/stanford/groups/euan/projects/viggiano/codebases/bio-programming-tools/tool_envs/esm3_env/cache/torch
USER=viggiano
VIRTUAL_ENV=/oak/stanford/groups/euan/projects/viggiano/codebases/bio-programming-tools/tool_envs/esm3_env
XDG_CACHE_HOME=/tmp
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Results by Category

### Causal Models (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `evo1` | yes | ✅ | 593.9s | ✅ Pass |
| `evo2` | yes | ✅ | 397.4s | ✅ Pass |
| `progen2` | yes | ✅ | 208.8s | ✅ Pass |

### Gene Annotation (5/5)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `blast` | no | ✅ | 35.7s | ✅ Pass |
| `crispr_tracr` | no | ✅ | 164.5s | ✅ Pass |
| `minced` | no | ✅ | 17.9s | ✅ Pass |
| `mmseqs` | no | ✅ | 24.7s | ✅ Pass |
| `pyhmmer` | no | ✅ | 21.0s | ✅ Pass |

### Inverse Folding (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `esm_if1` | yes | ✅ | 169.1s | ✅ Pass |
| `fampnn` | yes | ✅ | 173.2s | ✅ Pass |
| `ligandmpnn` | yes | ✅ | 212.2s | ✅ Pass |
| `proteinmpnn` | yes | ✅ | 114.7s | ✅ Pass |

### Masked Models (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `esm2` | yes | ✅ | 132.7s | ✅ Pass |
| `esm3` | yes | ✅ | 120.4s | ✅ Pass |

---
*Generated at 2026-03-22 19:29:06 by `pytest --env-report` (ESM3 verified passing after HF re-auth on 2026-03-22)*