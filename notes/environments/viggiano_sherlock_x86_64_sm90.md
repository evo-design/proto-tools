# Linux x86_64 Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-95%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-45-brightgreen) ![Failed](https://img.shields.io/badge/failed-2-red) ![Skipped](https://img.shields.io/badge/skipped-3-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Linux Linux 3.10.0-1160.139.1.el7.tuxcare.els4.x86_64 |
| **Architecture** | x86_64 |
| **Hostname** | `sh04-10n05.int` |
| **Python** | 3.10.20 |
| **RAM** | 2015.1 GB |
| **GPU** | 1x NVIDIA H200 |
| **CUDA** | 12.4 |
| **Conda Env** | `proto-tools` |

## Git

- **Commit**: `0830e1163353`
- **Branch**: `fix/test-failures-2026-04-27`
- **Dirty**: Yes

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
BLIS_NUM_THREADS=4
CC=gcc
CLAUDECODE=1
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
CLAUDE_CODE_ENTRYPOINT=cli
CLAUDE_CODE_EXECPATH=/share/software/user/open/claude-code/2.1.114/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe
COMMON_DATASETS=/oak/stanford/datasets/common
CONDA_DEFAULT_ENV=proto-tools
CONDA_EXE=/home/users/viggiano/miniconda3/bin/conda
CONDA_PREFIX=/scratch/users/viggiano/envs/proto-tools
CONDA_PREFIX_1=/home/users/viggiano/miniconda3
CONDA_PROMPT_MODIFIER=(proto-tools) 
CONDA_PYTHON_EXE=/home/users/viggiano/miniconda3/bin/python
CONDA_SHLVL=2
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
HOSTNAME=sh04-10n05.int
HYDRA_BOOTSTRAP=slurm
HYDRA_LAUNCHER_EXTRA_ARGS=--external-launcher
INFOPATH=/share/software/user/open/nodejs/20.20.0/share/info
I_MPI_HYDRA_BOOTSTRAP=slurm
I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS=--external-launcher
KRB5CCNAME=FILE:/tmp/krb5cc_389221_jgtUWwC87G
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
LOADEDMODULES=devel:math:claude-code/2.1.81:gcc/14.2.0:nodejs/20.20.0
LOCAL_SCRATCH=/lscratch/viggiano
LOGNAME=viggiano
LS_COLORS=su=00:sg=00:ca=00:or=40;31;01
L_SCRATCH=/lscratch/viggiano
L_SCRATCH_JOB=/lscratch/viggiano/22902023
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
OMPI_MCA_orte_precondition_transports=015d750700000000-015d750700000000
OMPI_MCA_plm_slurm_args=--external-launcher
OMP_NUM_THREADS=4
OPENBLAS_NUM_THREADS=4
OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta
PATH=/home/users/viggiano/.local/bin:/share/software/user/open/git/2.45.1/bin:/share/software/user/open/nodejs/25.3.0/bin:/share/software/user/open/claude-code/2.1.114/bin:/home/users/viggiano/.npm-global/...
PMIX_BFROP_BUFFER_TYPE=PMIX_BFROP_BUFFER_NON_DESC
PMIX_GDS_MODULE=shmem2,hash
PMIX_HOSTNAME=sh04-10n05
PMIX_NAMESPACE=slurm.pmix.22902023.0
PMIX_RANK=0
PMIX_SECURITY_MODE=native
PMIX_SERVER_TMPDIR=/var/spool/slurmd/pmix.22902023.0/
PMIX_SERVER_URI2=pmix-server.25459;tcp4://127.0.0.1:46630
PMIX_SERVER_URI21=pmix-server.25459;tcp4://127.0.0.1:46630
PMIX_SERVER_URI3=pmix-server.25459;tcp4://127.0.0.1:46630
PMIX_SERVER_URI4=pmix-server.25459;tcp4://127.0.0.1:46630
PMIX_SERVER_URI41=pmix-server.25459;tcp4://127.0.0.1:46630
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
RDBASE=/scratch/users/viggiano/envs/proto-tools/lib/python3.10/site-packages/rdkit
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
SLURMD_NODENAME=sh04-10n05
SLURM_CLUSTER_NAME=sherlock
SLURM_CONF=/var/spool/slurmd/conf-cache/slurm.conf
SLURM_CPUS_ON_NODE=4
SLURM_CPUS_PER_TASK=4
SLURM_CPU_BIND=quiet,mask_cpu:0xC000500000000000
SLURM_CPU_BIND_LIST=0xC000500000000000
SLURM_CPU_BIND_TYPE=mask_cpu:
SLURM_CPU_BIND_VERBOSE=quiet
SLURM_DISTRIBUTION=block
SLURM_GPUS=1
SLURM_GPUS_ON_NODE=1
SLURM_GTIDS=0
SLURM_JOBID=22902023
SLURM_JOB_ACCOUNT=euan
SLURM_JOB_CPUS_PER_NODE=4
SLURM_JOB_END_TIME=1777355287
SLURM_JOB_GID=11886
SLURM_JOB_GROUP=euan
SLURM_JOB_ID=22902023
SLURM_JOB_NAME=.ptshell-launcher.sE2pWq.sh
SLURM_JOB_NODELIST=sh04-10n05
SLURM_JOB_NUM_NODES=1
SLURM_JOB_PARTITION=brianhie
SLURM_JOB_QOS=normal
SLURM_JOB_START_TIME=1777312080
SLURM_JOB_UID=389221
SLURM_JOB_USER=viggiano
SLURM_LAUNCH_NODE_IPADDR=10.19.0.68
SLURM_LOCALID=0
SLURM_MEM_PER_CPU=10240
SLURM_MPI_TYPE=pmix
SLURM_NNODES=1
SLURM_NODEID=0
SLURM_NODELIST=sh04-10n05
SLURM_NPROCS=1
SLURM_NTASKS=1
SLURM_OOM_KILL_STEP=0
SLURM_PMIXP_ABORT_AGENT_PORT=38761
SLURM_PMIX_MAPPING_SERV=(vector,(0,1,1))
SLURM_PRIO_PROCESS=0
SLURM_PROCID=0
SLURM_PTY_PORT=45418
SLURM_PTY_WIN_COL=236
SLURM_PTY_WIN_ROW=57
SLURM_SCRIPT_CONTEXT=prolog_task
SLURM_SRUN_COMM_HOST=10.19.0.68
SLURM_SRUN_COMM_PORT=37564
SLURM_STEPID=0
SLURM_STEPMGR=sh04-10n05
SLURM_STEP_GPUS=7
SLURM_STEP_ID=0
SLURM_STEP_LAUNCHER_PORT=37564
SLURM_STEP_NODELIST=sh04-10n05
SLURM_STEP_NUM_NODES=1
SLURM_STEP_NUM_TASKS=1
SLURM_STEP_TASKS_PER_NODE=1
SLURM_SUBMIT_DIR=/home/users/viggiano
SLURM_SUBMIT_HOST=sh03-ln08.stanford.edu
SLURM_TASKS_PER_NODE=1
SLURM_TASK_PID=25501
SLURM_TOPOLOGY_ADDR=sh04.sh04-isw-10.sh04-10n05
SLURM_TOPOLOGY_ADDR_PATTERN=switch.switch.node
SLURM_TRES_PER_TASK=cpu=4
SLURM_UMASK=0022
SRCC_PATH=/share/software/user/srcc/bin
SRUN_CPUS_PER_TASK=4
SRUN_DEBUG=3
TERM=screen
TMOUT=86400
TMPDIR=/tmp
TMUX=/tmp/tmux-389221/default,61303,0
TMUX_LAUNCHED_SHERLOCK=1
TMUX_PANE=%0
USER=viggiano
USER_PATH=/share/software/user/open/nodejs/20.20.0/bin:/share/software/user/open/gcc/14.2.0/bin:/share/software/user/open/claude-code/2.1.81/bin:/home/users/viggiano/.local/bin:/home/users/viggiano/.npm-global/...
XDG_CACHE_HOME=/tmp
XDG_RUNTIME_DIR=/tmp
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
_=/scratch/users/viggiano/envs/proto-tools/bin/python3
_CE_CONDA=
_CE_M=
_LMFILES_=/share/software/modules/categories/devel.lua:/share/software/modules/categories/math.lua:/share/software/modules/devel/claude-code/2.1.81.lua:/share/software/modules/devel/gcc/14.2.0.lua:/share/softwa...
_ModuleTable001_=X01vZHVsZVRhYmxlXyA9IHsKTVR2ZXJzaW9uID0gMywKY19yZWJ1aWxkVGltZSA9IGZhbHNlLApjX3Nob3J0VGltZSA9IGZhbHNlLApkZXB0aFQgPSB7fSwKZmFtaWx5ID0ge30sCm1UID0gewpbImNsYXVkZS1jb2RlIl0gPSB7CmRlcFQgPSB7CmRlcEEgPSB7CnsK...
_ModuleTable002_=dmUiLAp1c2VyTmFtZSA9ICJjbGF1ZGUtY29kZSIsCndWID0gIl4wMDAwMDAwMi4wMDAwMDAwMDEuMDAwMDAwMDgxLip6ZmluYWwiLAp9LApkZXZlbCA9IHsKYWN0aW9uQSA9IHsKW1twcmVwZW5kX3BhdGgoIk1PRFVMRVBBVEgiLCIvc2hhcmUvc29mdHdhcmUvbW9k...
_ModuleTable003_=L3NoYXJlL3NvZnR3YXJlL21vZHVsZXMvZGV2ZWwvZ2NjLzE0LjIuMC5sdWEiLApmdWxsTmFtZSA9ICJnY2MvMTQuMi4wIiwKbG9hZE9yZGVyID0gNCwKcHJvcFQgPSB7fSwKcmVmX2NvdW50ID0gMSwKc3RhY2tEZXB0aCA9IDEsCnN0YXR1cyA9ICJhY3RpdmUiLAp1...
_ModuleTable004_=LApwcm9wVCA9IHsKbG1vZCA9IHsKc3RpY2t5ID0gMSwKfSwKfSwKc3RhY2tEZXB0aCA9IDAsCnN0YXR1cyA9ICJhY3RpdmUiLAp1c2VyTmFtZSA9ICJtYXRoIiwKd1YgPSAiTS4qemZpbmFsIiwKfSwKbm9kZWpzID0gewpkZXBUID0gewpkZXBBID0gewp7CnNuID0g...
_ModuleTable005_=PSAibm9kZWpzLzIwLjIwLjAiLAp3ViA9ICIwMDAwMDAwMjAuMDAwMDAwMDIwLip6ZmluYWwiLAp9LAp9LAptcGF0aEEgPSB7CiIvc2hhcmUvc29mdHdhcmUvbW9kdWxlcy9tYXRoIiwgIi9zaGFyZS9zb2Z0d2FyZS9tb2R1bGVzL2RldmVsIiwgIi9zaGFyZS9zb2Z0...
_ModuleTable_Sz_=5
__Init_Default_Modules=1
__LMOD_REF_COUNT_CPATH=/share/software/user/open/nodejs/20.20.0/include:1;/share/software/user/open/gcc/14.2.0/include:1
__LMOD_REF_COUNT_INFOPATH=/share/software/user/open/nodejs/20.20.0/share/info:1
__LMOD_REF_COUNT_LD_LIBRARY_PATH=/share/software/user/open/nodejs/20.20.0/lib:1;/share/software/user/open/gcc/14.2.0/lib64:1;/share/software/user/open/gcc/14.2.0/lib/gcc/x86_64-pc-linux-gnu:1;/share/software/user/open/gcc/14.2.0/lib:...
__LMOD_REF_COUNT_LIBRARY_PATH=/share/software/user/open/nodejs/20.20.0/lib:1;/share/software/user/open/gcc/14.2.0/lib64:1;/share/software/user/open/gcc/14.2.0/lib:1
__LMOD_REF_COUNT_MANPATH=/share/software/user/open/nodejs/20.20.0/share/man:1;/share/software/user/open/gcc/14.2.0/share/man:1;/share/software/user/open/lmod/lmod/share/man:1;/usr/local/share/man:1;/usr/share/man:1
__LMOD_REF_COUNT_MODULEPATH=/share/software/modules/math:1;/share/software/modules/devel:1;/share/software/modules/categories:1
__LMOD_REF_COUNT_PATH=/share/software/user/open/nodejs/20.20.0/bin:1;/share/software/user/open/gcc/14.2.0/bin:1;/share/software/user/open/claude-code/2.1.81/bin:1;/home/users/viggiano/.local/bin:1;/home/users/viggiano/.npm...
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
CONDA_PREFIX=/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/mmseqs2_homology_search_env
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
PATH=/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/mmseqs2_homology_search_env/bin:/home/users/viggiano/.local/bin:/share/software/user/open/git/2.45.1/bin:/share/software/user/open/no...
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
VIRTUAL_ENV=/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/mmseqs2_homology_search_env
XDG_CACHE_HOME=/tmp
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Results by Category

### Causal Models (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `evo1-sample` | yes | ✅ | 475.1s | `72cee05` | ✅ Pass |
| `evo2-sample` | yes | ✅ | 1499.5s | `72cee05` | ✅ Pass |
| `progen2-sample` | yes | ✅ | 338.1s | `72cee05` | ✅ Pass |
| `progen3-sample` | yes | ✅ | 506.3s | `72cee05` | ✅ Pass |

### Gene Annotation (5/5)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `blast-create-db` | no | ✅ | 31.6s | `72cee05` | ✅ Pass |
| `crispr-tracr` | no | ✅ | 193.2s | `72cee05` | ✅ Pass |
| `minced-crispr` | no | ✅ | 12.7s | `72cee05` | ✅ Pass |
| `mmseqs-clustering` | no | ✅ | 15.5s | `72cee05` | ✅ Pass |
| `pyhmmer-hmmscan` | no | ✅ | 23.5s | `72cee05` | ✅ Pass |

### Inverse Folding (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `esm-if1-sample` | yes | ✅ | 109.3s | `72cee05` | ✅ Pass |
| `fampnn-pack` | yes | ✅ | 254.3s | `72cee05` | ✅ Pass |
| `ligandmpnn-sample` | yes | ✅ | 318.5s | `72cee05` | ✅ Pass |
| `proteinmpnn-sample` | yes | ✅ | 284.8s | `72cee05` | ✅ Pass |

### Masked Models (3/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `ablang-embedding` | yes | ✅ | 356.4s | `72cee05` | ❌ Fail |
| `esm2-embedding` | yes | ✅ | 203.0s | `72cee05` | ✅ Pass |
| `esm3-embedding` | yes | ✅ | 162.7s | `72cee05` | ✅ Pass |
| `esmc-embedding` | yes | ✅ | 32.4s | `72cee05` | ✅ Pass |

### Mutagenesis (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `random-nucleotide-sample` | no | - | 0.0s | `72cee05` | ✅ Pass |
| `random-protein-sample` | no | - | 0.0s | `72cee05` | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `orfipy-prediction` | no | ✅ | 14.4s | `72cee05` | ✅ Pass |
| `prodigal-prediction` | no | ✅ | 15.6s | `72cee05` | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `splice-transformer-prediction` | yes | ✅ | 90.4s | `72cee05` | ✅ Pass |

### Sequence Alignment (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `colabfold-search` | no | ✅ | 36.4s | `72cee05` | ✅ Pass |
| `mafft-align` | no | ✅ | 17.1s | `72cee05` | ✅ Pass |
| `mmseqs2-homology-search` | no | ✅ | 123.3s | `0830e11` ✱ | ✅ Pass |

### Sequence Scoring (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphagenome-predict-intervals` | yes | ✅ | 1196.7s | `72cee05` | ✅ Pass |
| `borzoi-ensemble` | yes | ✅ | 156.7s | `72cee05` | ✅ Pass |
| `enformer-prediction` | yes | ✅ | 103.0s | `72cee05` | ✅ Pass |
| `segmasker-score` | no | ✅ | 24.2s | `72cee05` | ✅ Pass |

### Structure Alignment (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `tmalign-alignment` | no | ✅ | 21.5s | `72cee05` | ✅ Pass |
| `usalign-alignment` | no | ✅ | 51.2s | `72cee05` | ✅ Pass |

### Structure Design (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `rfdiffusion3-design` | yes | ✅ | 437.3s | `72cee05` | ✅ Pass |

### Structure Dynamics (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `bioemu-sample` | yes | ✅ | 394.0s | `72cee05` | ✅ Pass |

### Structure Prediction (6/7)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphafold2-binder` | yes | ✅ | 440.1s | `72cee05` | ✅ Pass |
| `alphafold3-prediction` | yes | ✅ | 95.9s | `72cee05` | ❌ Fail |
| `boltz2-prediction` | yes | ✅ | 339.2s | `72cee05` | ✅ Pass |
| `chai1-prediction` | yes | ✅ | 926.0s | `72cee05` | ✅ Pass |
| `esmfold-prediction` | yes | ✅ | 225.3s | `72cee05` | ✅ Pass |
| `protenix-prediction` | yes | ✅ | 516.2s | `72cee05` | ✅ Pass |
| `viennarna-prediction` | no | ✅ | 26.3s | `72cee05` | ✅ Pass |

### Structure Scoring (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `dssp-secondary-structure` | no | ✅ | 24.4s | `72cee05` | ✅ Pass |
| `pdockq2` | no | - | 0.0s | `72cee05` | ✅ Pass |
| `pyrosetta-energy` | no | ✅ | 182.1s | `72cee05` | ✅ Pass |
| `structure-metrics` | no | - | 1.0s | `72cee05` | ✅ Pass |

### Testing (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `mock-cli-multi-gpu-tool-run` | yes | - | - | `72cee05` | ⏭️ Skip |
| `mock-cli-tool-run` | yes | ✅ | 10.6s | `72cee05` | ✅ Pass |
| `mock-jax-multi-gpu-tool-run` | yes | - | - | `72cee05` | ⏭️ Skip |
| `mock-jax-tool-run` | yes | ✅ | 45.0s | `72cee05` | ✅ Pass |
| `mock-pytorch-multi-gpu-tool-run` | yes | - | - | `72cee05` | ⏭️ Skip |
| `mock-pytorch-tool-run` | yes | ✅ | 81.0s | `72cee05` | ✅ Pass |

## Failure Details

### ❌ `ablang-embedding`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ablang-embedding]`

```
tests/tool_infra_tests/test_env_report.py:74: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool ablang-embedding failed: ["Command '['/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/ablang_env/bin/python', '/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/tools/masked_models/ablang/standalone/inference.py', '/tmp/tmpaaswplzy/input.json', '/tmp/tmpaaswplzy/output.json']' returned non-zero exit status 1.", 'Traceback (most recent call last):\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py", line 566, in _wrapper_body\n    result = func(inputs, config, instance)\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/tools/masked_models/ablang/ablang_embeddings.py", line 116, in run_ablang_embeddings\n    outputs = ToolInstance.dispatch(\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 422, in dispatch\n    return cls._oneshot(\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 461, in _oneshot\n    return inst._run_oneshot(\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 1215, in _run_oneshot\n    subprocess.run(\n  File "/scratch/users/viggiano/envs/proto-tools/lib/python3.10/subprocess.py", line 526, in run\n    raise CalledProcessError(retcode, process.args,\nsubprocess.CalledProcessError: Command \'[\'/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/ablang_env/bin/python\', \'/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/tools/masked_models/ablang/standalone/inference.py\', \'/tmp/tmpaaswplzy/input.json\', \'/tmp/tmpaaswplzy/output.json\']\' returned non-zero exit status 1.\n']
E   assert False
E    +  where False = AbLangEmbeddingsOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success
```

### ❌ `alphafold3-prediction`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold3-prediction]`

```
tests/tool_infra_tests/test_env_report.py:74: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool alphafold3-prediction failed: ["'alphafold3' may not be compatible with your system. setup.sh failed (exit 1).\n      of\n        CMake.\n        Update the VERSION argument <min> value.  Or, use the <min>...<max>\n      syntax\n        to tell CMake that the project requires at least <min> but has been\n      updated\n        to work with policies introduced by <max> or earlier.\n      *** CMake build failed\n      hint: This usually indicates a problem with the package or the build\n      environment.", 'Traceback (most recent call last):\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py", line 566, in _wrapper_body\n    result = func(inputs, config, instance)\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/alphafold3/alphafold3.py", line 269, in run_alphafold3\n    output_data = ToolInstance.dispatch(\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 404, in dispatch\n    return cached.run(\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 766, in run\n    return self._run_persistent(\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 1014, in _run_persistent\n    self._ensure_env()\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 734, in _ensure_env\n    self._create_env()\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 2012, in _create_env\n    raise RuntimeError(\nRuntimeError: \'alphafold3\' may not be compatible with your system. setup.sh failed (exit 1).\n      of\n        CMake.\n        Update the VERSION argument <min> value.  Or, use the <min>...<max>\n      syntax\n        to tell CMake that the project requires at least <min> but has been\n      updated\n        to work with policies introduced by <max> or earlier.\n      *** CMake build failed\n      hint: This usually indicates a problem with the package or the build\n      environment.\n']
E   assert False
E    +  where False = <[ToolExecutionError('Attempt to access field of tool output after failure:       environment.\n\nError Messages:\n\'alphafold3\' may not be compatible with your system. setup.sh failed (exit 1).\n      of\n        CMake.\n        Update the VERSION argument <min> value.  Or, use the <min>...<max>\n      syntax\n        to tell CMake that the project requires at least <min> but has been\n      updated\n        to work with policies introduced by <max> or earlier.\n      *** CMake build failed\n      hint: This usually indicates a problem with the package or the build\n      environment.\nTraceback (most recent call last):\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py", line 566, in _wrapper_body\n    result = func(inputs, config, instance)\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/alphafold3/alphafold3.py", line 269, in run_alphafold3\n    output_data = ToolInstance.dispatch(\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 404, in dispatch\n    retu.../codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 766, in run\n    return self._run_persistent(\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 1014, in _run_persistent\n    self._ensure_env()\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 734, in _ensure_env\n    self._create_env()\n  File "/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 2012, in _create_env\n    raise RuntimeError(\nRuntimeError: \'alphafold3\' may not be compatible with your system. setup.sh failed (exit 1).\n      of\n        CMake.\n        Update the VERSION argument <min> value.  Or, use the <min>...<max>\n      syntax\n        to tell CMake that the project requires at least <min> but has been\n      updated\n        to work with policies introduced by <max> or earlier.\n      *** CMake build failed\n      hint: This usually indicates a problem with the package or the build\n      environment.\n') raised in repr()] AlphaFold3Output object at 0x7fad9f1fcae0>.success
```

---
*Generated at 2026-04-27 16:50:52 by `pytest --env-report`*

<!-- env-report-data
[
  {
    "tool_key": "evo2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo2-sample]",
    "status": "passed",
    "duration_seconds": 1499.48,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/evo2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "ablang-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ablang-embedding]",
    "status": "failed",
    "duration_seconds": 356.4,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/ablang_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:74: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool ablang-embedding failed: [\"Command '['/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/ablang_env/bin/python', '/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/tools/masked_models/ablang/standalone/inference.py', '/tmp/tmpaaswplzy/input.json', '/tmp/tmpaaswplzy/output.json']' returned non-zero exit status 1.\", 'Traceback (most recent call last):\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py\", line 566, in _wrapper_body\\n    result = func(inputs, config, instance)\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/tools/masked_models/ablang/ablang_embeddings.py\", line 116, in run_ablang_embeddings\\n    outputs = ToolInstance.dispatch(\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 422, in dispatch\\n    return cls._oneshot(\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 461, in _oneshot\\n    return inst._run_oneshot(\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 1215, in _run_oneshot\\n    subprocess.run(\\n  File \"/scratch/users/viggiano/envs/proto-tools/lib/python3.10/subprocess.py\", line 526, in run\\n    raise CalledProcessError(retcode, process.args,\\nsubprocess.CalledProcessError: Command \\'[\\'/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/ablang_env/bin/python\\', \\'/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/tools/masked_models/ablang/standalone/inference.py\\', \\'/tmp/tmpaaswplzy/input.json\\', \\'/tmp/tmpaaswplzy/output.json\\']\\' returned non-zero exit status 1.\\n']\nE   assert False\nE    +  where False = AbLangEmbeddingsOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success",
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "mmseqs-clustering",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mmseqs-clustering]",
    "status": "passed",
    "duration_seconds": 15.5,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/mmseqs_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "pyhmmer-hmmscan",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pyhmmer-hmmscan]",
    "status": "passed",
    "duration_seconds": 23.52,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/pyhmmer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "mmseqs2-homology-search",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mmseqs2-homology-search]",
    "status": "passed",
    "duration_seconds": 123.35,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/mmseqs2_homology_search_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "0830e1163353",
    "git_dirty": true
  },
  {
    "tool_key": "esm3-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm3-embedding]",
    "status": "passed",
    "duration_seconds": 162.65,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/evolutionaryscale_esm_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "ligandmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ligandmpnn-sample]",
    "status": "passed",
    "duration_seconds": 318.5,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/ligandmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
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
    "error_message": "('/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 68, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "progen2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen2-sample]",
    "status": "passed",
    "duration_seconds": 338.1,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/progen2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "minced-crispr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[minced-crispr]",
    "status": "passed",
    "duration_seconds": 12.74,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/minced_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "esmc-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esmc-embedding]",
    "status": "passed",
    "duration_seconds": 32.4,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/evolutionaryscale_esm_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "enformer-prediction",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[enformer-prediction]",
    "status": "passed",
    "duration_seconds": 102.95,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/enformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "borzoi-ensemble",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[borzoi-ensemble]",
    "status": "passed",
    "duration_seconds": 156.74,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/borzoi_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
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
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "alphafold2-binder",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold2-binder]",
    "status": "passed",
    "duration_seconds": 440.15,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/alphafold2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "progen3-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen3-sample]",
    "status": "passed",
    "duration_seconds": 506.28,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/progen3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "fampnn-pack",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[fampnn-pack]",
    "status": "passed",
    "duration_seconds": 254.33,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/fampnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "evo1-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo1-sample]",
    "status": "passed",
    "duration_seconds": 475.07,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/evo1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "blast-create-db",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[blast-create-db]",
    "status": "passed",
    "duration_seconds": 31.65,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/blast_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "pyrosetta-energy",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pyrosetta-energy]",
    "status": "passed",
    "duration_seconds": 182.07,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/pyrosetta_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "esm-if1-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm-if1-sample]",
    "status": "passed",
    "duration_seconds": 109.31,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/esm_if1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "crispr-tracr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[crispr-tracr]",
    "status": "passed",
    "duration_seconds": 193.25,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/crispr_tracr_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "protenix-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[protenix-prediction]",
    "status": "passed",
    "duration_seconds": 516.21,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/protenix_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "chai1-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[chai1-prediction]",
    "status": "passed",
    "duration_seconds": 926.05,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/chai1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "usalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[usalign-alignment]",
    "status": "passed",
    "duration_seconds": 51.18,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/usalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "structure-metrics",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[structure-metrics]",
    "status": "passed",
    "duration_seconds": 0.98,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "esm2-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm2-embedding]",
    "status": "passed",
    "duration_seconds": 203.0,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/esm2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "mock-pytorch-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-pytorch-tool-run]",
    "status": "passed",
    "duration_seconds": 81.04,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/mock_pytorch_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "proteinmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[proteinmpnn-sample]",
    "status": "passed",
    "duration_seconds": 284.81,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/proteinmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "mafft-align",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mafft-align]",
    "status": "passed",
    "duration_seconds": 17.15,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/mafft_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "alphagenome-predict-intervals",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphagenome-predict-intervals]",
    "status": "passed",
    "duration_seconds": 1196.74,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/alphagenome_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
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
    "error_message": "('/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 68, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "mock-jax-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-jax-tool-run]",
    "status": "passed",
    "duration_seconds": 44.95,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/mock_jax_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "alphafold3-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold3-prediction]",
    "status": "failed",
    "duration_seconds": 95.86,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/alphafold3_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:74: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool alphafold3-prediction failed: [\"'alphafold3' may not be compatible with your system. setup.sh failed (exit 1).\\n      of\\n        CMake.\\n        Update the VERSION argument <min> value.  Or, use the <min>...<max>\\n      syntax\\n        to tell CMake that the project requires at least <min> but has been\\n      updated\\n        to work with policies introduced by <max> or earlier.\\n      *** CMake build failed\\n      hint: This usually indicates a problem with the package or the build\\n      environment.\", 'Traceback (most recent call last):\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py\", line 566, in _wrapper_body\\n    result = func(inputs, config, instance)\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/alphafold3/alphafold3.py\", line 269, in run_alphafold3\\n    output_data = ToolInstance.dispatch(\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 404, in dispatch\\n    return cached.run(\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 766, in run\\n    return self._run_persistent(\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 1014, in _run_persistent\\n    self._ensure_env()\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 734, in _ensure_env\\n    self._create_env()\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 2012, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'alphafold3\\' may not be compatible with your system. setup.sh failed (exit 1).\\n      of\\n        CMake.\\n        Update the VERSION argument <min> value.  Or, use the <min>...<max>\\n      syntax\\n        to tell CMake that the project requires at least <min> but has been\\n      updated\\n        to work with policies introduced by <max> or earlier.\\n      *** CMake build failed\\n      hint: This usually indicates a problem with the package or the build\\n      environment.\\n']\nE   assert False\nE    +  where False = <[ToolExecutionError('Attempt to access field of tool output after failure:       environment.\\n\\nError Messages:\\n\\'alphafold3\\' may not be compatible with your system. setup.sh failed (exit 1).\\n      of\\n        CMake.\\n        Update the VERSION argument <min> value.  Or, use the <min>...<max>\\n      syntax\\n        to tell CMake that the project requires at least <min> but has been\\n      updated\\n        to work with policies introduced by <max> or earlier.\\n      *** CMake build failed\\n      hint: This usually indicates a problem with the package or the build\\n      environment.\\nTraceback (most recent call last):\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py\", line 566, in _wrapper_body\\n    result = func(inputs, config, instance)\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/alphafold3/alphafold3.py\", line 269, in run_alphafold3\\n    output_data = ToolInstance.dispatch(\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 404, in dispatch\\n    retu.../codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 766, in run\\n    return self._run_persistent(\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 1014, in _run_persistent\\n    self._ensure_env()\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 734, in _ensure_env\\n    self._create_env()\\n  File \"/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 2012, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'alphafold3\\' may not be compatible with your system. setup.sh failed (exit 1).\\n      of\\n        CMake.\\n        Update the VERSION argument <min> value.  Or, use the <min>...<max>\\n      syntax\\n        to tell CMake that the project requires at least <min> but has been\\n      updated\\n        to work with policies introduced by <max> or earlier.\\n      *** CMake build failed\\n      hint: This usually indicates a problem with the package or the build\\n      environment.\\n') raised in repr()] AlphaFold3Output object at 0x7fad9f1fcae0>.success",
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "esmfold-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esmfold-prediction]",
    "status": "passed",
    "duration_seconds": 225.26,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/esmfold_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "bioemu-sample",
    "category": "structure_dynamics",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[bioemu-sample]",
    "status": "passed",
    "duration_seconds": 393.98,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/bioemu_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "pdockq2",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pdockq2]",
    "status": "passed",
    "duration_seconds": 0.01,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "random-protein-sample",
    "category": "mutagenesis",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[random-protein-sample]",
    "status": "passed",
    "duration_seconds": 0.01,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "tmalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[tmalign-alignment]",
    "status": "passed",
    "duration_seconds": 21.54,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/tmalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "viennarna-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[viennarna-prediction]",
    "status": "passed",
    "duration_seconds": 26.29,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/viennarna_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "boltz2-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[boltz2-prediction]",
    "status": "passed",
    "duration_seconds": 339.16,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/boltz2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "mock-cli-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-cli-tool-run]",
    "status": "passed",
    "duration_seconds": 10.58,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/mock_cli_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "rfdiffusion3-design",
    "category": "structure_design",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[rfdiffusion3-design]",
    "status": "passed",
    "duration_seconds": 437.27,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/rfdiffusion3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "segmasker-score",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[segmasker-score]",
    "status": "passed",
    "duration_seconds": 24.16,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/segmasker_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "prodigal-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[prodigal-prediction]",
    "status": "passed",
    "duration_seconds": 15.61,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/prodigal_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "splice-transformer-prediction",
    "category": "rna_splicing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[splice-transformer-prediction]",
    "status": "passed",
    "duration_seconds": 90.4,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/splice_transformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
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
    "error_message": "('/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 68, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "colabfold-search",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[colabfold-search]",
    "status": "passed",
    "duration_seconds": 36.43,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/colabfold_search_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "dssp-secondary-structure",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[dssp-secondary-structure]",
    "status": "passed",
    "duration_seconds": 24.43,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/dssp_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  },
  {
    "tool_key": "orfipy-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[orfipy-prediction]",
    "status": "passed",
    "duration_seconds": 14.38,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/orfipy_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cee0539ffe",
    "git_dirty": false
  }
]
-->