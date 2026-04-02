# Linux x86_64 Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-100%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-42-brightgreen) ![Failed](https://img.shields.io/badge/failed-0-red) ![Skipped](https://img.shields.io/badge/skipped-1-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Linux Linux 5.15.0-1086-nvidia |
| **Architecture** | x86_64 |
| **Hostname** | `ashleylab-h100` |
| **Python** | 3.12.13 |
| **RAM** | 2015.6 GB |
| **GPU** | 8× NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3 |
| **CUDA** | 12.2 |
| **Conda Env** | `proto-tools` |

## Git

- **Commit**: `22a5c75334db`
- **Branch**: `fix/bioemu-protobuf-conflict`
- **Dirty**: Yes

## Environment Variables

### Parent Process Environment

```
CLAUDECODE=1
CLAUDE_CODE_ENTRYPOINT=cli
CONDA_DEFAULT_ENV=proto-tools
CONDA_EXE=/home/viggiano/miniconda3/bin/conda
CONDA_PREFIX=/projects/viggiano/envs/proto-tools
CONDA_PREFIX_1=/home/viggiano/miniconda3
CONDA_PROMPT_MODIFIER=(proto-tools)
CONDA_PYTHON_EXE=/home/viggiano/miniconda3/bin/python
CONDA_SHLVL=2
COREPACK_ENABLE_AUTO_PIN=0
DISABLE_PANDERA_IMPORT_WARNING=True
GIT_EDITOR=true
HOME=/home/viggiano
LANG=en_US.UTF-8
LESSCLOSE=/usr/bin/lesspipe %s %s
LESSOPEN=| /usr/bin/lesspipe %s
LOGNAME=viggiano
LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.a...
MIG_PARTED_CHECKPOINT_FILE=/var/lib/nvidia-mig-manager/checkpoint.json
MIG_PARTED_CONFIG_FILE=/etc/nvidia-mig-manager/config.yaml
MIG_PARTED_HOOKS_FILE=/etc/nvidia-mig-manager/hooks.yaml
MOTD_SHOWN=pam
NoDefaultCurrentDirectoryInExePath=1
OLDPWD=/home/viggiano/main/codebases/proto-bio
OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta
PATH=/home/viggiano/.local/bin:/usr/local/cuda/bin:/opt/bin:/home/viggiano/.local/bin:/home/viggiano/.cargo/bin:/projects/viggiano/envs/proto-tools/bin:/home/viggiano/miniconda3/condabin:/usr/local/cuda/bi...
PROTO_HOME=/raid/projects/viggiano/codebases/proto-bio
PWD=/home/viggiano/main/codebases/proto-bio/proto-tools
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
RDBASE=/projects/viggiano/envs/proto-tools/lib/python3.12/site-packages/rdkit
SHELL=/bin/bash
SHLVL=3
TERM=tmux-256color
TERMINFO_DIRS=/home/viggiano/.terminfo:/home/viggiano/.terminfo:
TERM_PROGRAM=tmux
TERM_PROGRAM_VERSION=3.2a
TMUX=/tmp/tmux-1013/default,1879133,0
TMUX_PANE=%0
USER=viggiano
XDG_DATA_DIRS=/usr/local/share:/usr/share:/var/lib/snapd/desktop
XDG_RUNTIME_DIR=/run/user/1013
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
_=/projects/viggiano/envs/proto-tools/bin/python3
```

### Subprocess Environment (passed to tools)

```
CONDA_PREFIX=/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/viennarna_env
DETECTED_COMPUTE_PLATFORM=cuda
DETECTED_CUDA_VERSION=12
DETECTED_DRIVER_VERSION=535
HF_HOME=/raid/projects/viggiano/codebases/proto-bio/proto_model_cache/huggingface
HOME=/home/viggiano
LANG=en_US.UTF-8
LD_LIBRARY_PATH=/projects/viggiano/envs/proto-tools/lib
LOGNAME=viggiano
PATH=/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/viennarna_env/bin:/home/viggiano/.local/bin:/usr/local/cuda/bin:/opt/bin:/home/viggiano/.cargo/bin:/projects/viggiano/envs/proto-tools/bin:/...
PIP_DEFAULT_TIMEOUT=300
PROTO_HOME=/raid/projects/viggiano/codebases/proto-bio
RECOMMENDED_JAX_SPEC=jax[cuda12]>=0.4.20,<1
RECOMMENDED_JAX_VARIANT=cuda12
RECOMMENDED_TORCH_INDEX=https://download.pytorch.org/whl/cu126
RECOMMENDED_TORCH_SPEC=torch>=2.4,<3
SHELL=/bin/bash
TORCH_CUDA_ARCH_LIST=9.0
TORCH_HOME=/raid/projects/viggiano/codebases/proto-bio/proto_model_cache/torch
USER=viggiano
UV_HTTP_TIMEOUT=300
VIRTUAL_ENV=/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/viennarna_env
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Results by Category

### Causal Models (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `evo1-sample` | yes | ✅ | 94.8s | `22a5c75` ✱ | ✅ Pass |
| `evo2-sample` | yes | ✅ | 80.7s | `22a5c75` ✱ | ✅ Pass |
| `progen2-sample` | yes | ✅ | 32.6s | `22a5c75` ✱ | ✅ Pass |

### Gene Annotation (5/5)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `blast-create-db` | no | ✅ | 19.7s | `22a5c75` ✱ | ✅ Pass |
| `crispr-tracr` | no | ✅ | 87.5s | `22a5c75` ✱ | ✅ Pass |
| `minced-crispr` | no | ✅ | 5.7s | `22a5c75` ✱ | ✅ Pass |
| `mmseqs-clustering` | no | ✅ | 8.0s | `22a5c75` ✱ | ✅ Pass |
| `pyhmmer-hmmscan` | no | ✅ | 12.3s | `22a5c75` ✱ | ✅ Pass |

### Inverse Folding (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `esm-if1-sample` | yes | ✅ | 22.7s | `22a5c75` ✱ | ✅ Pass |
| `fampnn-pack` | yes | ✅ | 43.9s | `22a5c75` ✱ | ✅ Pass |
| `ligandmpnn-sample` | yes | ✅ | 42.3s | `22a5c75` ✱ | ✅ Pass |
| `proteinmpnn-sample` | yes | ✅ | 32.6s | `22a5c75` ✱ | ✅ Pass |

### Masked Models (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `esm2-embedding` | yes | ✅ | 19.5s | `22a5c75` ✱ | ✅ Pass |
| `esm3-embedding` | yes | ✅ | 19.8s | `22a5c75` ✱ | ✅ Pass |

### Mutagenesis (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `random-nucleotide-sample` | no | — | 0.0s | `22a5c75` ✱ | ✅ Pass |
| `random-protein-sample` | no | — | 0.0s | `22a5c75` ✱ | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `orfipy-prediction` | no | ✅ | 6.1s | `22a5c75` ✱ | ✅ Pass |
| `prodigal-prediction` | no | ✅ | 5.5s | `22a5c75` ✱ | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `splice-transformer-prediction` | yes | ✅ | 16.0s | `22a5c75` ✱ | ✅ Pass |

### Sequence Alignment (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `colabfold-search` | no | ✅ | 9.8s | `22a5c75` ✱ | ✅ Pass |
| `mafft-align` | no | ✅ | 8.4s | `22a5c75` ✱ | ✅ Pass |

### Sequence Scoring (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphagenome-predict-intervals` | yes | ✅ | 524.5s | `22a5c75` ✱ | ✅ Pass |
| `borzoi-ensemble` | yes | ✅ | 66.4s | `22a5c75` ✱ | ✅ Pass |
| `enformer-prediction` | yes | ✅ | 17.7s | `22a5c75` ✱ | ✅ Pass |
| `segmasker-score` | no | ✅ | 19.4s | `22a5c75` ✱ | ✅ Pass |

### Structure Alignment (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `tmalign-alignment` | no | ✅ | 14.1s | `22a5c75` ✱ | ✅ Pass |
| `usalign-alignment` | no | ✅ | 24.6s | `22a5c75` ✱ | ✅ Pass |

### Structure Design (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `rfdiffusion3-design` | yes | ✅ | 61.5s | `22a5c75` ✱ | ✅ Pass |

### Structure Dynamics (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `bioemu-sample` | yes | ✅ | 35.3s | `22a5c75` ✱ | ✅ Pass |

### Structure Prediction (7/7)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphafold2-prediction` | yes | ✅ | 40.9s | `22a5c75` ✱ | ✅ Pass |
| `alphafold3-prediction` | yes | — | — | `22a5c75` ✱ | ⏭️ Skip |
| `boltz2-prediction` | yes | ✅ | 53.2s | `22a5c75` ✱ | ✅ Pass |
| `chai1-prediction` | yes | ✅ | 134.7s | `22a5c75` ✱ | ✅ Pass |
| `esmfold-prediction` | yes | ✅ | 27.6s | `22a5c75` ✱ | ✅ Pass |
| `protenix-prediction` | yes | ✅ | 236.7s | `22a5c75` ✱ | ✅ Pass |
| `structure-metrics` | no | ✅ | 6.2s | `22a5c75` ✱ | ✅ Pass |
| `viennarna-prediction` | no | ✅ | 5.4s | `22a5c75` ✱ | ✅ Pass |

### Testing (6/6)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `mock-cli-multi-gpu-tool-run` | yes | ✅ | 4.8s | `22a5c75` ✱ | ✅ Pass |
| `mock-cli-tool-run` | yes | ✅ | 4.7s | `22a5c75` ✱ | ✅ Pass |
| `mock-jax-multi-gpu-tool-run` | yes | ✅ | 24.4s | `22a5c75` ✱ | ✅ Pass |
| `mock-jax-tool-run` | yes | ✅ | 23.7s | `22a5c75` ✱ | ✅ Pass |
| `mock-pytorch-multi-gpu-tool-run` | yes | ✅ | 22.7s | `22a5c75` ✱ | ✅ Pass |
| `mock-pytorch-tool-run` | yes | ✅ | 21.4s | `22a5c75` ✱ | ✅ Pass |

---
*Generated at 2026-04-01 00:04:42 by `pytest --env-report`*

<!-- env-report-data
[
  {
    "tool_name": "alphafold2-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold2-prediction]",
    "status": "passed",
    "duration_seconds": 40.87,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/alphafold2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
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
    "error_message": "('/raid/projects/viggiano/codebases/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 96, 'Skipped: --env-report: requires Chimera cluster')",
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "alphagenome-predict-intervals",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphagenome-predict-intervals]",
    "status": "passed",
    "duration_seconds": 524.48,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/alphagenome_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "bioemu-sample",
    "category": "structure_dynamics",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[bioemu-sample]",
    "status": "passed",
    "duration_seconds": 35.32,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/bioemu_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "blast-create-db",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[blast-create-db]",
    "status": "passed",
    "duration_seconds": 19.72,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/blast_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "boltz2-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[boltz2-prediction]",
    "status": "passed",
    "duration_seconds": 53.16,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/boltz2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "borzoi-ensemble",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[borzoi-ensemble]",
    "status": "passed",
    "duration_seconds": 66.44,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/borzoi_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "chai1-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[chai1-prediction]",
    "status": "passed",
    "duration_seconds": 134.71,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/chai1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "colabfold-search",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[colabfold-search]",
    "status": "passed",
    "duration_seconds": 9.79,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/colabfold_search_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "crispr-tracr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[crispr-tracr]",
    "status": "passed",
    "duration_seconds": 87.55,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/crispr_tracr_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "enformer-prediction",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[enformer-prediction]",
    "status": "passed",
    "duration_seconds": 17.74,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/enformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "esm-if1-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm-if1-sample]",
    "status": "passed",
    "duration_seconds": 22.73,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/esm_if1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "esm2-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm2-embedding]",
    "status": "passed",
    "duration_seconds": 19.53,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/esm2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "esm3-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm3-embedding]",
    "status": "passed",
    "duration_seconds": 19.8,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/esm3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "esmfold-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esmfold-prediction]",
    "status": "passed",
    "duration_seconds": 27.63,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/esmfold_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "evo1-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo1-sample]",
    "status": "passed",
    "duration_seconds": 94.77,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/evo1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "evo2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo2-sample]",
    "status": "passed",
    "duration_seconds": 80.67,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/evo2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "fampnn-pack",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[fampnn-pack]",
    "status": "passed",
    "duration_seconds": 43.92,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/fampnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "ligandmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ligandmpnn-sample]",
    "status": "passed",
    "duration_seconds": 42.29,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/ligandmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "mafft-align",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mafft-align]",
    "status": "passed",
    "duration_seconds": 8.4,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/mafft_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "minced-crispr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[minced-crispr]",
    "status": "passed",
    "duration_seconds": 5.65,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/minced_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "mmseqs-clustering",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mmseqs-clustering]",
    "status": "passed",
    "duration_seconds": 8.03,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/mmseqs_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "mock-cli-multi-gpu-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-cli-multi-gpu-tool-run]",
    "status": "passed",
    "duration_seconds": 4.85,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/mock_cli_multi_gpu_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "mock-cli-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-cli-tool-run]",
    "status": "passed",
    "duration_seconds": 4.72,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/mock_cli_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "mock-jax-multi-gpu-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-jax-multi-gpu-tool-run]",
    "status": "passed",
    "duration_seconds": 24.43,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/mock_jax_multi_gpu_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "mock-jax-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-jax-tool-run]",
    "status": "passed",
    "duration_seconds": 23.7,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/mock_jax_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "mock-pytorch-multi-gpu-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-pytorch-multi-gpu-tool-run]",
    "status": "passed",
    "duration_seconds": 22.73,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/mock_pytorch_multi_gpu_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "mock-pytorch-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-pytorch-tool-run]",
    "status": "passed",
    "duration_seconds": 21.39,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/mock_pytorch_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "orfipy-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[orfipy-prediction]",
    "status": "passed",
    "duration_seconds": 6.09,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/orfipy_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "prodigal-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[prodigal-prediction]",
    "status": "passed",
    "duration_seconds": 5.46,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/prodigal_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "progen2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen2-sample]",
    "status": "passed",
    "duration_seconds": 32.56,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/progen2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "proteinmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[proteinmpnn-sample]",
    "status": "passed",
    "duration_seconds": 32.58,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/proteinmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "protenix-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[protenix-prediction]",
    "status": "passed",
    "duration_seconds": 236.68,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/protenix_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "pyhmmer-hmmscan",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pyhmmer-hmmscan]",
    "status": "passed",
    "duration_seconds": 12.32,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/pyhmmer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "random-nucleotide-sample",
    "category": "mutagenesis",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[random-nucleotide-sample]",
    "status": "passed",
    "duration_seconds": 0.0,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
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
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "rfdiffusion3-design",
    "category": "structure_design",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[rfdiffusion3-design]",
    "status": "passed",
    "duration_seconds": 61.47,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/rfdiffusion3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "segmasker-score",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[segmasker-score]",
    "status": "passed",
    "duration_seconds": 19.42,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/segmasker_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "splice-transformer-prediction",
    "category": "rna_splicing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[splice-transformer-prediction]",
    "status": "passed",
    "duration_seconds": 16.02,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/splice_transformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "structure-metrics",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[structure-metrics]",
    "status": "passed",
    "duration_seconds": 6.2,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/structure_metrics_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "tmalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[tmalign-alignment]",
    "status": "passed",
    "duration_seconds": 14.06,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/tmalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "usalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[usalign-alignment]",
    "status": "passed",
    "duration_seconds": 24.59,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/usalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  },
  {
    "tool_name": "viennarna-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[viennarna-prediction]",
    "status": "passed",
    "duration_seconds": 5.39,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/viennarna_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "22a5c75334db",
    "git_dirty": true
  }
]
-->
