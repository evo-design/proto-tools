# Linux x86_64 Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-97%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-37-brightgreen) ![Failed](https://img.shields.io/badge/failed-0-red) ![Skipped](https://img.shields.io/badge/skipped-1-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Linux Linux 3.10.0-1160.139.1.el7.tuxcare.els4.x86_64 |
| **Architecture** | x86_64 |
| **Hostname** | `sh04-16n05.int` |
| **Python** | 3.12.13 |
| **RAM** | 2015.1 GB |
| **GPU** | 1× NVIDIA H200 |
| **CUDA** | 12.4 |
| **Conda Env** | `bio-tools` |

## Git

- **Commit**: `74365507622d`
- **Branch**: `bv/package-prep`
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
CONDA_BACKUP_ADDR2LINE=/home/groups/euan/viggiano/envs/proto-language/bin/x86_64-conda-linux-gnu-addr2line
CONDA_BACKUP_AR=/home/groups/euan/viggiano/envs/proto-language/bin/x86_64-conda-linux-gnu-ar
CONDA_BACKUP_AS=/home/groups/euan/viggiano/envs/proto-language/bin/x86_64-conda-linux-gnu-as
CONDA_BACKUP_CXXFILT=/home/groups/euan/viggiano/envs/proto-language/bin/x86_64-conda-linux-gnu-c++filt
CONDA_BACKUP_DWP=/home/groups/euan/viggiano/envs/proto-language/bin/x86_64-conda-linux-gnu-dwp
CONDA_BACKUP_ELFEDIT=/home/groups/euan/viggiano/envs/proto-language/bin/x86_64-conda-linux-gnu-elfedit
CONDA_BACKUP_GPROF=/home/groups/euan/viggiano/envs/proto-language/bin/x86_64-conda-linux-gnu-gprof
CONDA_BACKUP_LD=/home/groups/euan/viggiano/envs/proto-language/bin/x86_64-conda-linux-gnu-ld
CONDA_BACKUP_LD_GOLD=/home/groups/euan/viggiano/envs/proto-language/bin/x86_64-conda-linux-gnu-ld.gold
CONDA_BACKUP_NM=/home/groups/euan/viggiano/envs/proto-language/bin/x86_64-conda-linux-gnu-nm
CONDA_BACKUP_OBJCOPY=/home/groups/euan/viggiano/envs/proto-language/bin/x86_64-conda-linux-gnu-objcopy
CONDA_BACKUP_OBJDUMP=/home/groups/euan/viggiano/envs/proto-language/bin/x86_64-conda-linux-gnu-objdump
CONDA_BACKUP_RANLIB=/home/groups/euan/viggiano/envs/proto-language/bin/x86_64-conda-linux-gnu-ranlib
CONDA_BACKUP_READELF=/home/groups/euan/viggiano/envs/proto-language/bin/x86_64-conda-linux-gnu-readelf
CONDA_BACKUP_SIZE=/home/groups/euan/viggiano/envs/proto-language/bin/x86_64-conda-linux-gnu-size
CONDA_BACKUP_STRINGS=/home/groups/euan/viggiano/envs/proto-language/bin/x86_64-conda-linux-gnu-strings
CONDA_BACKUP_STRIP=/home/groups/euan/viggiano/envs/proto-language/bin/x86_64-conda-linux-gnu-strip
CONDA_DEFAULT_ENV=bio-tools
CONDA_EXE=/home/users/viggiano/miniconda3/bin/conda
CONDA_PREFIX=/home/groups/euan/viggiano/envs/bio-tools
CONDA_PREFIX_1=/home/users/viggiano/miniconda3
CONDA_PREFIX_2=/home/groups/euan/viggiano/envs/proto-language
CONDA_PREFIX_3=/home/users/viggiano/miniconda3
CONDA_PROMPT_MODIFIER=(bio-tools) 
CONDA_PYTHON_EXE=/home/users/viggiano/miniconda3/bin/python
CONDA_SHLVL=4
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
HOSTNAME=sh04-16n05.int
HYDRA_BOOTSTRAP=slurm
HYDRA_LAUNCHER_EXTRA_ARGS=--external-launcher
INFOPATH=/share/software/user/open/nodejs/20.20.0/share/info
I_MPI_HYDRA_BOOTSTRAP=slurm
I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS=--external-launcher
KRB5CCNAME=FILE:/tmp/krb5cc_389221_1Fj6s29aTz
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
L_SCRATCH_JOB=/lscratch/viggiano/19710274
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
OMPI_MCA_orte_precondition_transports=012cc14200000000-012cc14200000000
OMPI_MCA_plm_slurm_args=--external-launcher
OMP_NUM_THREADS=8
OPENBLAS_NUM_THREADS=8
OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta
PATH=/home/users/viggiano/.local/bin:/share/software/user/open/nodejs/25.3.0/bin:/share/software/user/open/claude-code/2.1.81/bin:/home/users/viggiano/.npm-global/bin:/home/groups/euan/viggiano/envs/bio-to...
PMIX_BFROP_BUFFER_TYPE=PMIX_BFROP_BUFFER_NON_DESC
PMIX_GDS_MODULE=shmem2,hash
PMIX_HOSTNAME=sh04-16n05
PMIX_NAMESPACE=slurm.pmix.19710274.0
PMIX_RANK=0
PMIX_SECURITY_MODE=native
PMIX_SERVER_TMPDIR=/var/spool/slurmd/pmix.19710274.0/
PMIX_SERVER_URI2=pmix-server.10649;tcp4://127.0.0.1:35172
PMIX_SERVER_URI21=pmix-server.10649;tcp4://127.0.0.1:35172
PMIX_SERVER_URI3=pmix-server.10649;tcp4://127.0.0.1:35172
PMIX_SERVER_URI4=pmix-server.10649;tcp4://127.0.0.1:35172
PMIX_SERVER_URI41=pmix-server.10649;tcp4://127.0.0.1:35172
PMIX_SYSTEM_TMPDIR=/tmp
PMIX_VERSION=5.0.3
PROMPT_COMMAND=RET=$?;/bin/logger -t user_audit "username=$USER pid=$$ cmd=\"$(history 1 | /bin/sed "s/^[ ]*[0-9]\+[ ]*//" )\" newpwd=$PWD ret=$RET" 2>/dev/null
PROTO_HOME=/oak/stanford/groups/euan/projects/viggiano/.proto
PROTO_MODEL_CACHE=/scratch/users/viggiano/model_weights/proto-tools
PRTE_MCA_plm_slurm_args=--external-launcher
PWD=/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools
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
SLURMD_NODENAME=sh04-16n05
SLURM_CLUSTER_NAME=sherlock
SLURM_CONF=/var/spool/slurmd/conf-cache/slurm.conf
SLURM_CPUS_ON_NODE=8
SLURM_CPUS_PER_TASK=8
SLURM_CPU_BIND=quiet,mask_cpu:0x000000E0000F0080
SLURM_CPU_BIND_LIST=0x000000E0000F0080
SLURM_CPU_BIND_TYPE=mask_cpu:
SLURM_CPU_BIND_VERBOSE=quiet
SLURM_DISTRIBUTION=block
SLURM_GPUS=1
SLURM_GPUS_ON_NODE=1
SLURM_GTIDS=0
SLURM_JOBID=19710274
SLURM_JOB_ACCOUNT=euan
SLURM_JOB_CPUS_PER_NODE=8
SLURM_JOB_END_TIME=1774664663
SLURM_JOB_GID=11886
SLURM_JOB_GROUP=euan
SLURM_JOB_ID=19710274
SLURM_JOB_NAME=.ptshell-launcher.FCfOue.sh
SLURM_JOB_NODELIST=sh04-16n05
SLURM_JOB_NUM_NODES=1
SLURM_JOB_PARTITION=brianhie
SLURM_JOB_QOS=normal
SLURM_JOB_START_TIME=1774621460
SLURM_JOB_UID=389221
SLURM_JOB_USER=viggiano
SLURM_LAUNCH_NODE_IPADDR=10.19.0.61
SLURM_LOCALID=0
SLURM_MEM_PER_CPU=30720
SLURM_MPI_TYPE=pmix
SLURM_NNODES=1
SLURM_NODEID=0
SLURM_NODELIST=sh04-16n05
SLURM_NPROCS=1
SLURM_NTASKS=1
SLURM_OOM_KILL_STEP=0
SLURM_PMIXP_ABORT_AGENT_PORT=41217
SLURM_PMIX_MAPPING_SERV=(vector,(0,1,1))
SLURM_PRIO_PROCESS=0
SLURM_PROCID=0
SLURM_PTY_PORT=34492
SLURM_PTY_WIN_COL=212
SLURM_PTY_WIN_ROW=59
SLURM_SCRIPT_CONTEXT=prolog_task
SLURM_SRUN_COMM_HOST=10.19.0.61
SLURM_SRUN_COMM_PORT=45942
SLURM_STEPID=0
SLURM_STEPMGR=sh04-16n05
SLURM_STEP_GPUS=1
SLURM_STEP_ID=0
SLURM_STEP_LAUNCHER_PORT=45942
SLURM_STEP_NODELIST=sh04-16n05
SLURM_STEP_NUM_NODES=1
SLURM_STEP_NUM_TASKS=1
SLURM_STEP_TASKS_PER_NODE=1
SLURM_SUBMIT_DIR=/home/users/viggiano
SLURM_SUBMIT_HOST=sh03-ln01.stanford.edu
SLURM_TASKS_PER_NODE=1
SLURM_TASK_PID=10696
SLURM_TOPOLOGY_ADDR=sh04.sh04-isw-16.sh04-16n05
SLURM_TOPOLOGY_ADDR_PATTERN=switch.switch.node
SLURM_TRES_PER_TASK=cpu=8
SLURM_UMASK=0022
SRCC_PATH=/share/software/user/srcc/bin
SRUN_CPUS_PER_TASK=8
SRUN_DEBUG=3
TERM=screen
TMOUT=86400
TMPDIR=/tmp
TMUX=/tmp/tmux-389221/default,20159,0
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
CONDA_PREFIX=/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/tool_envs/mock_pytorch_tool_env
CUDA_VISIBLE_DEVICES=0
DETECTED_COMPUTE_PLATFORM=cuda
DETECTED_CUDA_VERSION=12
DETECTED_DRIVER_VERSION=550
HF_HOME=/scratch/users/viggiano/model_weights/proto-tools/huggingface
HOME=/home/users/viggiano
LANG=en_US.UTF-8
LC_CTYPE=C.UTF-8
LD_LIBRARY_PATH=/share/software/user/open/gcc/14.2.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs:/home/groups/euan/viggiano/envs/bio-tools/lib
LOGNAME=viggiano
PATH=/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/tool_envs/mock_pytorch_tool_env/bin:/usr/local/cuda/bin:/home/users/viggiano/.local/bin:/share/software/user/open/...
PIP_DEFAULT_TIMEOUT=300
PROTO_HOME=/oak/stanford/groups/euan/projects/viggiano/.proto
PROTO_MODEL_CACHE=/scratch/users/viggiano/model_weights/proto-tools
RECOMMENDED_JAX_SPEC=jax[cuda12]>=0.4.20,<1
RECOMMENDED_JAX_VARIANT=cuda12
RECOMMENDED_TORCH_INDEX=https://download.pytorch.org/whl/cu126
RECOMMENDED_TORCH_SPEC=torch>=2.5,<3
SHELL=/bin/bash
TMPDIR=/tmp
TORCH_CUDA_ARCH_LIST=9.0
TORCH_HOME=/scratch/users/viggiano/model_weights/proto-tools/torch
USER=viggiano
UV_HTTP_TIMEOUT=300
VIRTUAL_ENV=/oak/stanford/groups/euan/projects/viggiano/codebases/proto-bio/proto-tools/tool_envs/mock_pytorch_tool_env
XDG_CACHE_HOME=/tmp
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Results by Category

### Causal Models (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `evo1` | yes | ✅ | 200.8s | ✅ Pass |
| `evo2` | yes | ✅ | 232.6s | ✅ Pass |
| `progen2` | yes | ✅ | 159.7s | ✅ Pass |

### Gene Annotation (5/5)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `blast` | no | ✅ | 31.2s | ✅ Pass |
| `crispr_tracr` | no | ✅ | 160.0s | ✅ Pass |
| `minced` | no | ✅ | 24.7s | ✅ Pass |
| `mmseqs` | no | ✅ | 21.5s | ✅ Pass |
| `pyhmmer` | no | ✅ | 24.0s | ✅ Pass |

### Inverse Folding (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `esm_if1` | yes | ✅ | 106.7s | ✅ Pass |
| `fampnn` | yes | ✅ | 153.5s | ✅ Pass |
| `ligandmpnn` | yes | ✅ | 211.2s | ✅ Pass |
| `proteinmpnn` | yes | ✅ | 78.1s | ✅ Pass |

### Masked Models (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `esm2` | yes | ✅ | 101.1s | ✅ Pass |
| `esm3` | yes | ✅ | 109.7s | ✅ Pass |

### Mock (1/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `mock_tools_env_report` | yes | — | 104.9s | ✅ Pass |
| `mock_tools_env_report` | yes | — | — | ⏭️ Skip |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `orfipy` | no | ✅ | 20.4s | ✅ Pass |
| `prodigal` | no | ✅ | 13.2s | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `splice_transformer` | yes | ✅ | 95.1s | ✅ Pass |

### Sequence Alignment (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `local_colabfold_search` | no | — | 120.6s | ✅ Pass |
| `mafft` | no | ✅ | 18.3s | ✅ Pass |

### Sequence Scoring (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `alphagenome` | yes | ✅ | 248.7s | ✅ Pass |
| `borzoi` | yes | ✅ | 97.0s | ✅ Pass |
| `enformer` | yes | ✅ | 90.5s | ✅ Pass |
| `segmasker` | no | ✅ | 27.5s | ✅ Pass |

### Structure Alignment (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `tmalign` | no | ✅ | 25.4s | ✅ Pass |
| `tmalign` | no | ✅ | 0.1s | ✅ Pass |
| `usalign` | no | ✅ | 33.8s | ✅ Pass |
| `usalign` | no | ✅ | 0.1s | ✅ Pass |

### Structure Design (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `rfdiffusion3` | yes | ✅ | 225.2s | ✅ Pass |

### Structure Dynamics (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `bioemu` | yes | ✅ | 281.3s | ✅ Pass |

### Structure Prediction (7/7)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `alphafold2` | yes | ✅ | 116.1s | ✅ Pass |
| `boltz2` | yes | ✅ | 195.3s | ✅ Pass |
| `chai1` | yes | ✅ | 362.7s | ✅ Pass |
| `esmfold` | yes | ✅ | 160.0s | ✅ Pass |
| `protenix` | yes | ✅ | 428.6s | ✅ Pass |
| `structure_metrics` | no | ✅ | 25.2s | ✅ Pass |
| `viennarna` | no | ✅ | 25.5s | ✅ Pass |

---
*Generated at 2026-03-27 13:16:38 by `pytest --env-report`*