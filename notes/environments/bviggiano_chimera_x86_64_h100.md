# Chimera Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-100%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-41-brightgreen) ![Failed](https://img.shields.io/badge/failed-0-red) ![Skipped](https://img.shields.io/badge/skipped-3-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Linux Linux 5.15.0-171-generic |
| **Architecture** | x86_64 |
| **Hostname** | `GPUC960` |
| **Python** | 3.12.13 |
| **RAM** | 1007.4 GB |
| **GPU** | 1× NVIDIA H100 80GB HBM3 |
| **CUDA** | 12.2 |
| **Mamba Env** | `proto-tools` |

## Git

- **Commit**: `eb044b500f21`
- **Branch**: `main`
- **Dirty**: No

## Environment Variables

### Parent Process Environment

```
BLASTDB=/common_datasets/external/databases/blast
CONDA_DEFAULT_ENV=proto-tools
CONDA_EXE=/home/bviggiano/miniforge3/bin/conda
CONDA_PREFIX=/home/bviggiano/miniforge3/envs/proto-tools
CONDA_PREFIX_1=/home/bviggiano/miniforge3
CONDA_PROMPT_MODIFIER=(proto-tools) 
CONDA_PYTHON_EXE=/home/bviggiano/miniforge3/bin/python
CONDA_SHLVL=2
CUDA_VISIBLE_DEVICES=0
DISABLE_PANDERA_IMPORT_WARNING=True
GPU_DEVICE_ORDINAL=0
HOME=/home/bviggiano
HYDRA_BOOTSTRAP=slurm
HYDRA_LAUNCHER_EXTRA_ARGS=--external-launcher
I_MPI_HYDRA_BOOTSTRAP=slurm
I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS=--external-launcher
LANG=en_US.UTF-8
LESSCLOSE=/usr/bin/lesspipe %s %s
LESSOPEN=| /usr/bin/lesspipe %s
LOGNAME=bviggiano
LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.a...
MAMBA_EXE=/home/bviggiano/miniforge3/bin/mamba
MAMBA_ROOT_PREFIX=/home/bviggiano/.local/share/mamba
MOTD_SHOWN=pam
OLDPWD=/home/bviggiano
OMPI_MCA_plm_slurm_args=--external-launcher
PATH=/home/bviggiano/.local/bin:/home/bviggiano/bin:/home/bviggiano/.local/bin:/home/bviggiano/bin:/home/bviggiano/miniforge3/envs/proto-tools/bin:/home/bviggiano/miniforge3/condabin:/usr/local/sbin:/usr/l...
PROTO_HOME=/large_storage/hielab/bviggiano/proto_cache
PRTE_MCA_plm_slurm_args=--external-launcher
PWD=/home/bviggiano/proto-tools
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
RDBASE=/home/bviggiano/miniforge3/envs/proto-tools/lib/python3.12/site-packages/rdkit
ROCR_VISIBLE_DEVICES=0
SHELL=/bin/bash
SHLVL=5
SLURMD_DEBUG=2
SLURMD_NODENAME=GPUC960
SLURM_CLUSTER_NAME=arc-slurm
SLURM_CONF=/etc/slurm/slurm.conf
SLURM_CPUS_ON_NODE=8
SLURM_GPUS=1
SLURM_GPUS_ON_NODE=1
SLURM_GTIDS=0
SLURM_JOBID=1990260
SLURM_JOB_ACCOUNT=hielab
SLURM_JOB_CPUS_PER_NODE=8
SLURM_JOB_END_TIME=1775842532
SLURM_JOB_GID=10004
SLURM_JOB_GPUS=0
SLURM_JOB_ID=1990260
SLURM_JOB_NAME=1_sh_gpu
SLURM_JOB_NODELIST=GPUC960
SLURM_JOB_NUM_NODES=1
SLURM_JOB_PARTITION=evo_gpu_priority
SLURM_JOB_QOS=normal
SLURM_JOB_START_TIME=1775799332
SLURM_JOB_UID=10249
SLURM_JOB_USER=bviggiano
SLURM_LAUNCH_NODE_IPADDR=172.18.140.10
SLURM_LOCALID=0
SLURM_MPI_TYPE=pmix
SLURM_NNODES=1
SLURM_NODEID=0
SLURM_NODELIST=GPUC960
SLURM_OOM_KILL_STEP=0
SLURM_PMIXP_ABORT_AGENT_PORT=34539
SLURM_PMIX_MAPPING_SERV=(vector,(0,1,1))
SLURM_PRIO_PROCESS=0
SLURM_PROCID=0
SLURM_PTY_PORT=44229
SLURM_PTY_WIN_COL=236
SLURM_PTY_WIN_ROW=64
SLURM_SRUN_COMM_HOST=172.18.140.10
SLURM_SRUN_COMM_PORT=33099
SLURM_STEPID=4294967290
SLURM_STEP_ID=4294967290
SLURM_STEP_LAUNCHER_PORT=33099
SLURM_STEP_NODELIST=GPUC960
SLURM_STEP_NUM_NODES=1
SLURM_STEP_NUM_TASKS=1
SLURM_STEP_TASKS_PER_NODE=1
SLURM_SUBMIT_DIR=/home/bviggiano
SLURM_SUBMIT_HOST=arc-slurm
SLURM_TASKS_PER_NODE=8
SLURM_TASK_PID=711598
SLURM_TOPOLOGY_ADDR=GPUC960
SLURM_TOPOLOGY_ADDR_PATTERN=node
SRUN_DEBUG=3
TERM=xterm-256color
TERM_PROGRAM=tmux
TERM_PROGRAM_VERSION=3.2a
TMPDIR=/tmp
TMUX=/tmp/tmux-10249/default,1274489,7
TMUX_PANE=%7
USER=bviggiano
XDG_DATA_DIRS=/usr/local/share:/usr/share:/var/lib/snapd/desktop
XDG_RUNTIME_DIR=/run/user/10249
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
XML_CATALOG_FILES=file:///home/bviggiano/miniforge3/etc/xml/catalog file:///etc/xml/catalog file:///home/bviggiano/miniforge3/etc/xml/catalog file:///etc/xml/catalog file:///home/bviggiano/miniforge3/etc/xml/catalog fi...
ZES_ENABLE_SYSMAN=1
ZE_AFFINITY_MASK=0
_=/home/bviggiano/miniforge3/envs/proto-tools/bin/pytest
_CE_CONDA=
_CE_M=
_CONDA_EXE=/home/bviggiano/miniforge3/bin/conda
_CONDA_ROOT=/home/bviggiano/miniforge3
```

### Subprocess Environment (passed to tools)

```
CONDA_PREFIX=/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/viennarna_env
CUDA_VISIBLE_DEVICES=0
DETECTED_COMPUTE_PLATFORM=cuda
DETECTED_CUDA_VERSION=12
DETECTED_DRIVER_VERSION=535
HF_HOME=/large_storage/hielab/bviggiano/proto_cache/proto_model_cache/huggingface
HOME=/home/bviggiano
LANG=en_US.UTF-8
LD_LIBRARY_PATH=/home/bviggiano/miniforge3/envs/proto-tools/lib
LOGNAME=bviggiano
PATH=/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/viennarna_env/bin:/home/bviggiano/.local/bin:/home/bviggiano/bin:/home/bviggiano/miniforge3/envs/proto-tools/bin:/home/bviggiano/miniforge3/...
PIP_DEFAULT_TIMEOUT=300
PROTO_HOME=/large_storage/hielab/bviggiano/proto_cache
RECOMMENDED_JAX_SPEC=jax[cuda12]>=0.4.20,<1
RECOMMENDED_JAX_VARIANT=cuda12
RECOMMENDED_TORCH_INDEX=https://download.pytorch.org/whl/cu126
RECOMMENDED_TORCH_SPEC=torch>=2.4,<3
SHELL=/bin/bash
TMPDIR=/tmp
TORCH_CUDA_ARCH_LIST=9.0
TORCH_HOME=/large_storage/hielab/bviggiano/proto_cache/proto_model_cache/torch
USER=bviggiano
UV_HTTP_TIMEOUT=300
VIRTUAL_ENV=/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/viennarna_env
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Results by Category

### Causal Models (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `evo1-sample` | yes | ✅ | 326.9s | `eb044b5` | ✅ Pass |
| `evo2-sample` | yes | ✅ | 338.6s | `eb044b5` | ✅ Pass |
| `progen2-sample` | yes | ✅ | 201.6s | `eb044b5` | ✅ Pass |
| `progen3-sample` | yes | ✅ | 31.2s | `eb044b5` | ✅ Pass |

### Gene Annotation (5/5)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `blast-create-db` | no | ✅ | 41.0s | `eb044b5` | ✅ Pass |
| `crispr-tracr` | no | ✅ | 171.1s | `eb044b5` | ✅ Pass |
| `minced-crispr` | no | ✅ | 20.3s | `eb044b5` | ✅ Pass |
| `mmseqs-clustering` | no | ✅ | 26.9s | `eb044b5` | ✅ Pass |
| `pyhmmer-hmmscan` | no | ✅ | 24.7s | `eb044b5` | ✅ Pass |

### Inverse Folding (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `esm-if1-sample` | yes | ✅ | 159.4s | `eb044b5` | ✅ Pass |
| `fampnn-pack` | yes | ✅ | 191.4s | `eb044b5` | ✅ Pass |
| `ligandmpnn-sample` | yes | ✅ | 231.0s | `eb044b5` | ✅ Pass |
| `proteinmpnn-sample` | yes | ✅ | 95.6s | `eb044b5` | ✅ Pass |

### Masked Models (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `esm2-embedding` | yes | ✅ | 162.1s | `eb044b5` | ✅ Pass |
| `esm3-embedding` | yes | ✅ | 151.5s | `eb044b5` | ✅ Pass |

### Mutagenesis (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `random-nucleotide-sample` | no | - | 0.0s | `eb044b5` | ✅ Pass |
| `random-protein-sample` | no | - | 0.0s | `eb044b5` | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `orfipy-prediction` | no | ✅ | 23.9s | `eb044b5` | ✅ Pass |
| `prodigal-prediction` | no | ✅ | 15.8s | `eb044b5` | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `splice-transformer-prediction` | yes | ✅ | 113.8s | `eb044b5` | ✅ Pass |

### Sequence Alignment (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `colabfold-search` | no | ✅ | 127.1s | `eb044b5` | ✅ Pass |
| `mafft-align` | no | ✅ | 25.3s | `eb044b5` | ✅ Pass |

### Sequence Scoring (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphagenome-predict-intervals` | yes | ✅ | 847.9s | `eb044b5` | ✅ Pass |
| `borzoi-ensemble` | yes | ✅ | 177.7s | `eb044b5` | ✅ Pass |
| `enformer-prediction` | yes | ✅ | 137.5s | `eb044b5` | ✅ Pass |
| `segmasker-score` | no | ✅ | 31.1s | `eb044b5` | ✅ Pass |

### Structure Alignment (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `tmalign-alignment` | no | ✅ | 25.8s | `eb044b5` | ✅ Pass |
| `usalign-alignment` | no | ✅ | 38.3s | `eb044b5` | ✅ Pass |

### Structure Design (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `rfdiffusion3-design` | yes | ✅ | 272.0s | `eb044b5` | ✅ Pass |

### Structure Dynamics (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `bioemu-sample` | yes | ✅ | 297.2s | `eb044b5` | ✅ Pass |

### Structure Prediction (8/8)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphafold2-prediction` | yes | ✅ | 110.4s | `eb044b5` | ✅ Pass |
| `alphafold3-prediction` | yes | ✅ | 75.8s | `eb044b5` | ✅ Pass |
| `boltz2-prediction` | yes | ✅ | 194.6s | `eb044b5` | ✅ Pass |
| `chai1-prediction` | yes | ✅ | 391.3s | `eb044b5` | ✅ Pass |
| `esmfold-prediction` | yes | ✅ | 139.8s | `eb044b5` | ✅ Pass |
| `protenix-prediction` | yes | ✅ | 421.3s | `eb044b5` | ✅ Pass |
| `structure-metrics` | no | ✅ | 26.1s | `eb044b5` | ✅ Pass |
| `viennarna-prediction` | no | ✅ | 16.0s | `eb044b5` | ✅ Pass |

### Structure Scoring (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `pyrosetta-energy` | no | ✅ | 231.5s | `07a60f6` | ✅ Pass |

### Testing (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `mock-cli-multi-gpu-tool-run` | yes | - | - | `eb044b5` | ⏭️ Skip |
| `mock-cli-tool-run` | yes | ✅ | 13.4s | `eb044b5` | ✅ Pass |
| `mock-jax-multi-gpu-tool-run` | yes | - | - | `eb044b5` | ⏭️ Skip |
| `mock-jax-tool-run` | yes | ✅ | 59.1s | `eb044b5` | ✅ Pass |
| `mock-pytorch-multi-gpu-tool-run` | yes | - | - | `eb044b5` | ⏭️ Skip |
| `mock-pytorch-tool-run` | yes | ✅ | 144.3s | `eb044b5` | ✅ Pass |

---
*Generated at 2026-04-10 00:52:29 by `pytest --env-report`*

<!-- env-report-data
[
  {
    "tool_name": "alphafold2-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold2-prediction]",
    "status": "passed",
    "duration_seconds": 110.37,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphafold2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "alphafold3-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold3-prediction]",
    "status": "passed",
    "duration_seconds": 75.79,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphafold3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "alphagenome-predict-intervals",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphagenome-predict-intervals]",
    "status": "passed",
    "duration_seconds": 847.87,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphagenome_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "bioemu-sample",
    "category": "structure_dynamics",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[bioemu-sample]",
    "status": "passed",
    "duration_seconds": 297.24,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/bioemu_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "blast-create-db",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[blast-create-db]",
    "status": "passed",
    "duration_seconds": 40.98,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/blast_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "boltz2-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[boltz2-prediction]",
    "status": "passed",
    "duration_seconds": 194.58,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/boltz2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "borzoi-ensemble",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[borzoi-ensemble]",
    "status": "passed",
    "duration_seconds": 177.68,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/borzoi_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "chai1-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[chai1-prediction]",
    "status": "passed",
    "duration_seconds": 391.31,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/chai1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "colabfold-search",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[colabfold-search]",
    "status": "passed",
    "duration_seconds": 127.07,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/colabfold_search_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "crispr-tracr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[crispr-tracr]",
    "status": "passed",
    "duration_seconds": 171.08,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/crispr_tracr_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "enformer-prediction",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[enformer-prediction]",
    "status": "passed",
    "duration_seconds": 137.49,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/enformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "esm-if1-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm-if1-sample]",
    "status": "passed",
    "duration_seconds": 159.37,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/esm_if1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "esm2-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm2-embedding]",
    "status": "passed",
    "duration_seconds": 162.11,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/esm2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "esm3-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm3-embedding]",
    "status": "passed",
    "duration_seconds": 151.49,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/esm3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "esmfold-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esmfold-prediction]",
    "status": "passed",
    "duration_seconds": 139.81,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/esmfold_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "evo1-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo1-sample]",
    "status": "passed",
    "duration_seconds": 326.87,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/evo1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "evo2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo2-sample]",
    "status": "passed",
    "duration_seconds": 338.6,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/evo2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "fampnn-pack",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[fampnn-pack]",
    "status": "passed",
    "duration_seconds": 191.44,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/fampnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "ligandmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ligandmpnn-sample]",
    "status": "passed",
    "duration_seconds": 231.01,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/ligandmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "mafft-align",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mafft-align]",
    "status": "passed",
    "duration_seconds": 25.3,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mafft_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "minced-crispr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[minced-crispr]",
    "status": "passed",
    "duration_seconds": 20.28,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/minced_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "mmseqs-clustering",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mmseqs-clustering]",
    "status": "passed",
    "duration_seconds": 26.92,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mmseqs_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "mock-cli-multi-gpu-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-cli-multi-gpu-tool-run]",
    "status": "skipped",
    "duration_seconds": 0.01,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 72, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "mock-cli-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-cli-tool-run]",
    "status": "passed",
    "duration_seconds": 13.36,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mock_cli_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
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
    "error_message": "('/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 72, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "mock-jax-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-jax-tool-run]",
    "status": "passed",
    "duration_seconds": 59.13,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mock_jax_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
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
    "error_message": "('/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 72, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "mock-pytorch-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-pytorch-tool-run]",
    "status": "passed",
    "duration_seconds": 144.35,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mock_pytorch_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "orfipy-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[orfipy-prediction]",
    "status": "passed",
    "duration_seconds": 23.91,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/orfipy_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "prodigal-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[prodigal-prediction]",
    "status": "passed",
    "duration_seconds": 15.8,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/prodigal_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "progen2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen2-sample]",
    "status": "passed",
    "duration_seconds": 201.59,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/progen2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "progen3-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen3-sample]",
    "status": "passed",
    "duration_seconds": 31.18,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/progen3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "proteinmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[proteinmpnn-sample]",
    "status": "passed",
    "duration_seconds": 95.57,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/proteinmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "protenix-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[protenix-prediction]",
    "status": "passed",
    "duration_seconds": 421.3,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/protenix_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "pyhmmer-hmmscan",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pyhmmer-hmmscan]",
    "status": "passed",
    "duration_seconds": 24.67,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/pyhmmer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "random-nucleotide-sample",
    "category": "mutagenesis",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[random-nucleotide-sample]",
    "status": "passed",
    "duration_seconds": 0.01,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "eb044b500f21",
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
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "rfdiffusion3-design",
    "category": "structure_design",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[rfdiffusion3-design]",
    "status": "passed",
    "duration_seconds": 271.97,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/rfdiffusion3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "segmasker-score",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[segmasker-score]",
    "status": "passed",
    "duration_seconds": 31.14,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/segmasker_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "splice-transformer-prediction",
    "category": "rna_splicing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[splice-transformer-prediction]",
    "status": "passed",
    "duration_seconds": 113.82,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/splice_transformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "structure-metrics",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[structure-metrics]",
    "status": "passed",
    "duration_seconds": 26.11,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/structure_metrics_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "tmalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[tmalign-alignment]",
    "status": "passed",
    "duration_seconds": 25.8,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/tmalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "usalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[usalign-alignment]",
    "status": "passed",
    "duration_seconds": 38.34,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/usalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  },
  {
    "tool_name": "viennarna-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[viennarna-prediction]",
    "status": "passed",
    "duration_seconds": 16.02,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/viennarna_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "eb044b500f21",
    "git_dirty": false
  }
]
-->
