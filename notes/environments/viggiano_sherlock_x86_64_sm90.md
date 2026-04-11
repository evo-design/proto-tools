# Linux x86_64 Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-100%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-40-brightgreen) ![Failed](https://img.shields.io/badge/failed-0-red) ![Skipped](https://img.shields.io/badge/skipped-4-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Linux Linux 3.10.0-1160.139.1.el7.tuxcare.els4.x86_64 |
| **Architecture** | x86_64 |
| **Hostname** | `sh04-10n09.int` |
| **Python** | 3.12.13 |
| **RAM** | 2015.1 GB |
| **GPU** | 1× NVIDIA H200 |
| **CUDA** | 12.4 |
| **Conda Env** | `proto-tools` |

## Git

- **Commit**: `9a15ca08a125`
- **Branch**: `feat/pyrosetta-scoring-tools`
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
CLAUDE_CODE_ENTRYPOINT=cli
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
DISABLE_PANDERA_IMPORT_WARNING=True
F77=gfortran
F90=gfortran
FC=gfortran
GIT_EDITOR=true
GOTO_NUM_THREADS=4
GROUP=euan
GROUP_HOME=/home/groups/euan
GROUP_SCRATCH=/scratch/groups/euan
HISTCONTROL=ignoreboth:erasedups
HISTSIZE=1000
HOME=/home/users/viggiano
HOSTNAME=sh04-10n09.int
HYDRA_BOOTSTRAP=slurm
HYDRA_LAUNCHER_EXTRA_ARGS=--external-launcher
INFOPATH=/share/software/user/open/nodejs/20.20.0/share/info
I_MPI_HYDRA_BOOTSTRAP=slurm
I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS=--external-launcher
KRB5CCNAME=FILE:/tmp/krb5cc_389221_nnaykBSDbo
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
L_SCRATCH_JOB=/lscratch/viggiano/20193581
L_SCRATCH_USER=/lscratch/viggiano
MAIL=/var/spool/mail/viggiano
MANPATH=/share/software/user/open/nodejs/20.20.0/share/man:/share/software/user/open/gcc/14.2.0/share/man:/share/software/user/open/lmod/lmod/share/man:/usr/local/share/man:/usr/share/man
MKL_NUM_THREADS=4
MODULEPATH=/share/software/modules/math:/share/software/modules/devel:/share/software/modules/categories
MODULEPATH_ROOT=/share/software/modules
MODULESHOME=/share/software/user/open/lmod/9.0.2
NVIDIA_DRIVER_CAPABILITIES=compute,utility
NVIDIA_VISIBLE_DEVICES=all
NoDefaultCurrentDirectoryInExePath=1
OAK=/oak/stanford/groups/euan
OLDPWD=/home/users/viggiano/oak_main/codebases/proto-bio
OMPI_MCA_orte_precondition_transports=0134212d00000000-0134212d00000000
OMPI_MCA_plm_slurm_args=--external-launcher
OMP_NUM_THREADS=4
OPENBLAS_NUM_THREADS=4
OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta
PATH=/home/users/viggiano/.local/bin:/share/software/user/open/nodejs/25.3.0/bin:/share/software/user/open/claude-code/2.1.81/bin:/home/users/viggiano/.npm-global/bin:/scratch/users/viggiano/envs/proto-too...
PMIX_BFROP_BUFFER_TYPE=PMIX_BFROP_BUFFER_NON_DESC
PMIX_GDS_MODULE=shmem2,hash
PMIX_HOSTNAME=sh04-10n09
PMIX_NAMESPACE=slurm.pmix.20193581.0
PMIX_RANK=0
PMIX_SECURITY_MODE=native
PMIX_SERVER_TMPDIR=/var/spool/slurmd/pmix.20193581.0/
PMIX_SERVER_URI2=pmix-server.35133;tcp4://127.0.0.1:43189
PMIX_SERVER_URI21=pmix-server.35133;tcp4://127.0.0.1:43189
PMIX_SERVER_URI3=pmix-server.35133;tcp4://127.0.0.1:43189
PMIX_SERVER_URI4=pmix-server.35133;tcp4://127.0.0.1:43189
PMIX_SERVER_URI41=pmix-server.35133;tcp4://127.0.0.1:43189
PMIX_SYSTEM_TMPDIR=/tmp
PMIX_VERSION=5.0.3
PROMPT_COMMAND=RET=$?;/bin/logger -t user_audit "username=$USER pid=$$ cmd=\"$(history 1 | /bin/sed "s/^[ ]*[0-9]\+[ ]*//" )\" newpwd=$PWD ret=$RET" 2>/dev/null
PROTO_HOME=/oak/stanford/groups/euan/projects/viggiano/.proto
PROTO_MODEL_CACHE=/scratch/users/viggiano/model_weights/bio-programming-tools
PRTE_MCA_plm_slurm_args=--external-launcher
PWD=/home/users/viggiano/oak_main/codebases/evo-design/proto-tools
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
PYTHONPYCACHEPREFIX=/tmp
PYTORCH_VERSION=2.2.1
RDBASE=/scratch/users/viggiano/envs/proto-tools/lib/python3.12/site-packages/rdkit
SCRATCH=/scratch/users/viggiano
SHELL=/bin/bash
SHERLOCK=2
SHLVL=4
SH_DEF_TIMEOUT=30
SH_INFO_TIMEOUT=30
SH_TIMEOUT_CMD=timeout -s9
SH_USER_ENV_INITD=1
SINGULARITY_BIND=/bashrc_custom,/share/software/user/open
SINGULARITY_CONTAINER=/home/groups/brianhie/brianhie/simg/pytorch_latest.sif
SINGULARITY_ENVIRONMENT=/.singularity.d/env/91-environment.sh
SINGULARITY_NAME=pytorch_latest.sif
SLURMD_DEBUG=2
SLURMD_NODENAME=sh04-10n09
SLURM_CLUSTER_NAME=sherlock
SLURM_CONF=/var/spool/slurmd/conf-cache/slurm.conf
SLURM_CPUS_ON_NODE=4
SLURM_CPUS_PER_TASK=4
SLURM_CPU_BIND=quiet,mask_cpu:0x000F000000000000
SLURM_CPU_BIND_LIST=0x000F000000000000
SLURM_CPU_BIND_TYPE=mask_cpu:
SLURM_CPU_BIND_VERBOSE=quiet
SLURM_DISTRIBUTION=block
SLURM_GPUS=1
SLURM_GPUS_ON_NODE=1
SLURM_GTIDS=0
SLURM_JOBID=20193581
SLURM_JOB_ACCOUNT=euan
SLURM_JOB_CPUS_PER_NODE=4
SLURM_JOB_END_TIME=1775102362
SLURM_JOB_GID=11886
SLURM_JOB_GROUP=euan
SLURM_JOB_ID=20193581
SLURM_JOB_NAME=.ptshell-launcher.jNiGqE.sh
SLURM_JOB_NODELIST=sh04-10n09
SLURM_JOB_NUM_NODES=1
SLURM_JOB_PARTITION=brianhie
SLURM_JOB_QOS=normal
SLURM_JOB_START_TIME=1775059159
SLURM_JOB_UID=389221
SLURM_JOB_USER=viggiano
SLURM_LAUNCH_NODE_IPADDR=10.20.0.64
SLURM_LOCALID=0
SLURM_MEM_PER_CPU=10240
SLURM_MPI_TYPE=pmix
SLURM_NNODES=1
SLURM_NODEID=0
SLURM_NODELIST=sh04-10n09
SLURM_NPROCS=1
SLURM_NTASKS=1
SLURM_OOM_KILL_STEP=0
SLURM_PMIXP_ABORT_AGENT_PORT=34875
SLURM_PMIX_MAPPING_SERV=(vector,(0,1,1))
SLURM_PRIO_PROCESS=0
SLURM_PROCID=0
SLURM_PTY_PORT=40125
SLURM_PTY_WIN_COL=236
SLURM_PTY_WIN_ROW=59
SLURM_SCRIPT_CONTEXT=prolog_task
SLURM_SRUN_COMM_HOST=10.20.0.64
SLURM_SRUN_COMM_PORT=46480
SLURM_STEPID=0
SLURM_STEPMGR=sh04-10n09
SLURM_STEP_GPUS=5
SLURM_STEP_ID=0
SLURM_STEP_LAUNCHER_PORT=46480
SLURM_STEP_NODELIST=sh04-10n09
SLURM_STEP_NUM_NODES=1
SLURM_STEP_NUM_TASKS=1
SLURM_STEP_TASKS_PER_NODE=1
SLURM_SUBMIT_DIR=/home/users/viggiano
SLURM_SUBMIT_HOST=sh04-ln04.stanford.edu
SLURM_TASKS_PER_NODE=1
SLURM_TASK_PID=35169
SLURM_TOPOLOGY_ADDR=sh04.sh04-isw-10.sh04-10n09
SLURM_TOPOLOGY_ADDR_PATTERN=switch.switch.node
SLURM_TRES_PER_TASK=cpu=4
SLURM_UMASK=0022
SRCC_PATH=/share/software/user/srcc/bin
SRUN_CPUS_PER_TASK=4
SRUN_DEBUG=3
TERM=screen
TMOUT=86400
TMPDIR=/tmp
TMUX=/tmp/tmux-389221/default,26913,0
TMUX_LAUNCHED_SHERLOCK=1
TMUX_PANE=%0
USER=viggiano
USER_PATH=/share/software/user/open/nodejs/20.20.0/bin:/share/software/user/open/gcc/14.2.0/bin:/home/users/viggiano/.local/bin:/home/users/viggiano/.npm-global/bin:/home/users/viggiano/miniconda3/bin:/home/use...
XDG_CACHE_HOME=/tmp
XDG_RUNTIME_DIR=/tmp
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
_=/scratch/users/viggiano/envs/proto-tools/bin/python3
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
CONDA_PREFIX=/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/pyrosetta_env
CUDA_VISIBLE_DEVICES=0
DETECTED_COMPUTE_PLATFORM=cuda
DETECTED_CUDA_VERSION=12
DETECTED_DRIVER_VERSION=550
HF_HOME=/scratch/users/viggiano/model_weights/bio-programming-tools/huggingface
HOME=/home/users/viggiano
LANG=en_US.UTF-8
LC_CTYPE=C.UTF-8
LD_LIBRARY_PATH=/share/software/user/open/gcc/14.2.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs:/scratch/users/viggiano/envs/proto-tools/lib
LOGNAME=viggiano
PATH=/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/pyrosetta_env/bin:/home/users/viggiano/.local/bin:/share/software/user/open/nodejs/25.3.0/bin:/share/software/user/open/claude-code/2...
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
UV_HTTP_TIMEOUT=300
VIRTUAL_ENV=/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/pyrosetta_env
XDG_CACHE_HOME=/tmp
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Results by Category

### Causal Models (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `evo1-sample` | yes | ✅ | 503.7s | `9a15ca0` | ✅ Pass |
| `evo2-sample` | yes | ✅ | 320.0s | `9a15ca0` | ✅ Pass |
| `progen2-sample` | yes | ✅ | 284.7s | `9a15ca0` | ✅ Pass |

### Gene Annotation (5/5)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `blast-create-db` | no | ✅ | 38.3s | `9a15ca0` | ✅ Pass |
| `crispr-tracr` | no | ✅ | 151.0s | `9a15ca0` | ✅ Pass |
| `minced-crispr` | no | ✅ | 18.4s | `9a15ca0` | ✅ Pass |
| `mmseqs-clustering` | no | ✅ | 20.9s | `9a15ca0` | ✅ Pass |
| `pyhmmer-hmmscan` | no | ✅ | 33.7s | `9a15ca0` | ✅ Pass |

### Inverse Folding (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `esm-if1-sample` | yes | ✅ | 117.9s | `9a15ca0` | ✅ Pass |
| `fampnn-pack` | yes | ✅ | 169.2s | `9a15ca0` | ✅ Pass |
| `ligandmpnn-sample` | yes | ✅ | 203.8s | `9a15ca0` | ✅ Pass |
| `proteinmpnn-sample` | yes | ✅ | 67.4s | `9a15ca0` | ✅ Pass |

### Masked Models (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `esm2-embedding` | yes | ✅ | 115.6s | `9a15ca0` | ✅ Pass |
| `esm3-embedding` | yes | ✅ | 127.8s | `9a15ca0` | ✅ Pass |

### Mutagenesis (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `random-nucleotide-sample` | no | - | 0.1s | `9a15ca0` | ✅ Pass |
| `random-protein-sample` | no | - | 0.0s | `9a15ca0` | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `orfipy-prediction` | no | ✅ | 28.2s | `9a15ca0` | ✅ Pass |
| `prodigal-prediction` | no | ✅ | 12.9s | `9a15ca0` | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `splice-transformer-prediction` | yes | ✅ | 95.3s | `9a15ca0` | ✅ Pass |

### Sequence Alignment (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `colabfold-search` | no | ✅ | 108.6s | `9a15ca0` | ✅ Pass |
| `mafft-align` | no | ✅ | 19.6s | `9a15ca0` | ✅ Pass |

### Sequence Scoring (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphagenome-predict-intervals` | yes | ✅ | 812.3s | `9a15ca0` | ✅ Pass |
| `borzoi-ensemble` | yes | ✅ | 173.0s | `9a15ca0` | ✅ Pass |
| `enformer-prediction` | yes | ✅ | 105.1s | `9a15ca0` | ✅ Pass |
| `segmasker-score` | no | ✅ | 28.2s | `9a15ca0` | ✅ Pass |

### Structure Alignment (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `tmalign-alignment` | no | ✅ | 21.7s | `9a15ca0` | ✅ Pass |
| `usalign-alignment` | no | ✅ | 34.3s | `9a15ca0` | ✅ Pass |

### Structure Design (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `rfdiffusion3-design` | yes | ✅ | 225.9s | `9a15ca0` | ✅ Pass |

### Structure Dynamics (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `bioemu-sample` | yes | ✅ | 329.4s | `9a15ca0` | ✅ Pass |

### Structure Prediction (7/7)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphafold2-prediction` | yes | ✅ | 664.2s | `9a15ca0` | ✅ Pass |
| `alphafold3-prediction` | yes | - | - | `9a15ca0` | ⏭️ Skip |
| `boltz2-prediction` | yes | ✅ | 195.3s | `9a15ca0` | ✅ Pass |
| `chai1-prediction` | yes | ✅ | 354.7s | `9a15ca0` | ✅ Pass |
| `esmfold-prediction` | yes | ✅ | 187.0s | `9a15ca0` | ✅ Pass |
| `protenix-prediction` | yes | ✅ | 397.0s | `9a15ca0` | ✅ Pass |
| `structure-metrics` | no | ✅ | 22.0s | `9a15ca0` | ✅ Pass |
| `viennarna-prediction` | no | ✅ | 12.6s | `9a15ca0` | ✅ Pass |

### Structure Scoring (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `pyrosetta-energy` | no | ✅ | 471.1s | `9a15ca0` ✱ | ✅ Pass |

### Testing (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `mock-cli-multi-gpu-tool-run` | yes | - | - | `9a15ca0` | ⏭️ Skip |
| `mock-cli-tool-run` | yes | ✅ | 10.0s | `9a15ca0` | ✅ Pass |
| `mock-jax-multi-gpu-tool-run` | yes | - | - | `9a15ca0` | ⏭️ Skip |
| `mock-jax-tool-run` | yes | ✅ | 46.6s | `9a15ca0` | ✅ Pass |
| `mock-pytorch-multi-gpu-tool-run` | yes | - | - | `9a15ca0` | ⏭️ Skip |
| `mock-pytorch-tool-run` | yes | ✅ | 94.3s | `9a15ca0` | ✅ Pass |

---
*Generated at 2026-04-01 09:18:50 by `pytest --env-report`*

<!-- env-report-data
[
  {
    "tool_name": "alphafold2-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold2-prediction]",
    "status": "passed",
    "duration_seconds": 664.21,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/alphafold2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "alphafold3-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold3-prediction]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/tests/tool_infra_tests/test_env_report.py', 93, 'Skipped: --env-report: requires Chimera cluster')",
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "alphagenome-predict-intervals",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphagenome-predict-intervals]",
    "status": "passed",
    "duration_seconds": 812.26,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/alphagenome_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "bioemu-sample",
    "category": "structure_dynamics",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[bioemu-sample]",
    "status": "passed",
    "duration_seconds": 329.36,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/bioemu_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "blast-create-db",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[blast-create-db]",
    "status": "passed",
    "duration_seconds": 38.33,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/blast_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "boltz2-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[boltz2-prediction]",
    "status": "passed",
    "duration_seconds": 195.29,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/boltz2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "borzoi-ensemble",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[borzoi-ensemble]",
    "status": "passed",
    "duration_seconds": 173.04,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/borzoi_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "chai1-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[chai1-prediction]",
    "status": "passed",
    "duration_seconds": 354.74,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/chai1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "colabfold-search",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[colabfold-search]",
    "status": "passed",
    "duration_seconds": 108.64,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/colabfold_search_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "crispr-tracr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[crispr-tracr]",
    "status": "passed",
    "duration_seconds": 150.98,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/crispr_tracr_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "enformer-prediction",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[enformer-prediction]",
    "status": "passed",
    "duration_seconds": 105.08,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/enformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "esm-if1-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm-if1-sample]",
    "status": "passed",
    "duration_seconds": 117.87,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/esm_if1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "esm2-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm2-embedding]",
    "status": "passed",
    "duration_seconds": 115.64,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/esm2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "esm3-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm3-embedding]",
    "status": "passed",
    "duration_seconds": 127.79,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/esm3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "esmfold-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esmfold-prediction]",
    "status": "passed",
    "duration_seconds": 187.03,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/esmfold_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "evo1-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo1-sample]",
    "status": "passed",
    "duration_seconds": 503.67,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/evo1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "evo2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo2-sample]",
    "status": "passed",
    "duration_seconds": 320.01,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/evo2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "fampnn-pack",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[fampnn-pack]",
    "status": "passed",
    "duration_seconds": 169.16,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/fampnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "ligandmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ligandmpnn-sample]",
    "status": "passed",
    "duration_seconds": 203.84,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/ligandmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "mafft-align",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mafft-align]",
    "status": "passed",
    "duration_seconds": 19.59,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/mafft_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "minced-crispr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[minced-crispr]",
    "status": "passed",
    "duration_seconds": 18.42,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/minced_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "mmseqs-clustering",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mmseqs-clustering]",
    "status": "passed",
    "duration_seconds": 20.95,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/mmseqs_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "mock-cli-multi-gpu-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-cli-multi-gpu-tool-run]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/tests/tool_infra_tests/test_env_report.py', 93, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "mock-cli-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-cli-tool-run]",
    "status": "passed",
    "duration_seconds": 9.98,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/mock_cli_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "mock-jax-multi-gpu-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-jax-multi-gpu-tool-run]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/tests/tool_infra_tests/test_env_report.py', 93, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "mock-jax-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-jax-tool-run]",
    "status": "passed",
    "duration_seconds": 46.58,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/mock_jax_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "mock-pytorch-multi-gpu-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-pytorch-multi-gpu-tool-run]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/oak/stanford/groups/euan/projects/viggiano/codebases/evo-design/proto-tools/tests/tool_infra_tests/test_env_report.py', 93, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "mock-pytorch-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-pytorch-tool-run]",
    "status": "passed",
    "duration_seconds": 94.27,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/mock_pytorch_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "orfipy-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[orfipy-prediction]",
    "status": "passed",
    "duration_seconds": 28.21,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/orfipy_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "prodigal-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[prodigal-prediction]",
    "status": "passed",
    "duration_seconds": 12.9,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/prodigal_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "progen2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen2-sample]",
    "status": "passed",
    "duration_seconds": 284.68,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/progen2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "proteinmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[proteinmpnn-sample]",
    "status": "passed",
    "duration_seconds": 67.42,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/proteinmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "protenix-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[protenix-prediction]",
    "status": "passed",
    "duration_seconds": 396.99,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/protenix_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "pyhmmer-hmmscan",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pyhmmer-hmmscan]",
    "status": "passed",
    "duration_seconds": 33.71,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/pyhmmer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "pyrosetta-energy",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pyrosetta-energy]",
    "status": "passed",
    "duration_seconds": 471.06,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/pyrosetta_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": true
  },
  {
    "tool_name": "random-nucleotide-sample",
    "category": "mutagenesis",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[random-nucleotide-sample]",
    "status": "passed",
    "duration_seconds": 0.08,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "random-protein-sample",
    "category": "mutagenesis",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[random-protein-sample]",
    "status": "passed",
    "duration_seconds": 0.0,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "rfdiffusion3-design",
    "category": "structure_design",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[rfdiffusion3-design]",
    "status": "passed",
    "duration_seconds": 225.89,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/rfdiffusion3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "segmasker-score",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[segmasker-score]",
    "status": "passed",
    "duration_seconds": 28.2,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/segmasker_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "splice-transformer-prediction",
    "category": "rna_splicing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[splice-transformer-prediction]",
    "status": "passed",
    "duration_seconds": 95.33,
    "uses_gpu": true,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/splice_transformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "structure-metrics",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[structure-metrics]",
    "status": "passed",
    "duration_seconds": 22.03,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/structure_metrics_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "tmalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[tmalign-alignment]",
    "status": "passed",
    "duration_seconds": 21.66,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/tmalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "usalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[usalign-alignment]",
    "status": "passed",
    "duration_seconds": 34.33,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/usalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  },
  {
    "tool_name": "viennarna-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[viennarna-prediction]",
    "status": "passed",
    "duration_seconds": 12.58,
    "uses_gpu": false,
    "env_path": "/oak/stanford/groups/euan/projects/viggiano/.proto/proto_tool_envs/viennarna_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "9a15ca08a125",
    "git_dirty": false
  }
]
-->
