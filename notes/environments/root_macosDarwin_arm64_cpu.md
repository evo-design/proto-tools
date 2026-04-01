# Darwin arm64 Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-100%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-16-brightgreen) ![Failed](https://img.shields.io/badge/failed-0-red) ![Skipped](https://img.shields.io/badge/skipped-27-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Darwin Darwin 25.3.0 |
| **Architecture** | arm64 |
| **Hostname** | `spock-2.local` |
| **Python** | 3.10.13 |
| **RAM** | 64.0 GB |
| **GPU** | None |
| **Conda Env** | `base` |

## Git

- **Commit**: `d2604fc6b288`
- **Branch**: `fix/exclude-env-reports-from-precommit`
- **Dirty**: Yes

## Environment Variables

### Parent Process Environment

```
BUNDLED_DEBUGPY_PATH=/Users/bviggiano/.cursor/extensions/ms-python.debugpy-2025.18.0-darwin-arm64/bundled/libs/debugpy
CLAUDECODE=1
CLAUDE_CODE_ENTRYPOINT=cli
CLAUDE_CODE_SSE_PORT=17417
CLICOLOR=1
COLORTERM=truecolor
COMMAND_MODE=unix2003
CONDA_DEFAULT_ENV=base
CONDA_EXE=/Users/bviggiano/miniconda3/bin/conda
CONDA_PREFIX=/Users/bviggiano/miniconda3
CONDA_PROMPT_MODIFIER=
CONDA_PYTHON_EXE=/Users/bviggiano/miniconda3/bin/python
CONDA_SHLVL=1
COREPACK_ENABLE_AUTO_PIN=0
CURSOR_TRACE_ID=1164fa62cd30487aae65c1a2f22596e9
DISABLE_PANDERA_IMPORT_WARNING=True
DISPLAY=/private/tmp/com.apple.launchd.MKGNb39xy3/org.xquartz:0
GIT_EDITOR=true
GK_GL_ADDR=http://127.0.0.1:54679
GK_GL_PATH=/var/folders/6f/gqwcqlqn3sxdz7rlzjxk7pgh0000gn/T/gitkraken/gitlens/gitlens-ipc-server-90674-54679.json
HOME=/Users/bviggiano
HOMEBREW_CELLAR=/opt/homebrew/Cellar
HOMEBREW_PREFIX=/opt/homebrew
HOMEBREW_REPOSITORY=/opt/homebrew
INFOPATH=/opt/homebrew/share/info:
LANG=en_US.UTF-8
LOGNAME=bviggiano
LSCOLORS=ExFxBxDxCxegedabagacad
MACH_PORT_RENDEZVOUS_PEER_VALDATION=0
MallocNanoZone=0
NoDefaultCurrentDirectoryInExePath=1
OLDPWD=/Users/bviggiano/Projects/proto-bio/proto-tools
OSLogRateLimit=64
OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta
PATH=/Users/bviggiano/.local/bin:/Users/bviggiano/.juliaup/bin:/Users/bviggiano/miniconda3/bin:/Users/bviggiano/miniconda3/condabin:/Library/Frameworks/Python.framework/Versions/3.10/bin:/opt/homebrew/bin:...
PWD=/Users/bviggiano/Projects/proto-bio/proto-tools
PYDEVD_DISABLE_FILE_VALIDATION=1
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
RDBASE=/Users/bviggiano/miniconda3/lib/python3.10/site-packages/rdkit
SHELL=/bin/zsh
SHLVL=2
STARSHIP_SHELL=zsh
TERM=xterm-256color
TERM_PROGRAM=vscode
TERM_PROGRAM_VERSION=2.6.22
TMPDIR=/var/folders/6f/gqwcqlqn3sxdz7rlzjxk7pgh0000gn/T/
USER=bviggiano
USER_ZDOTDIR=/Users/bviggiano
VSCODE_DEBUGPY_ADAPTER_ENDPOINTS=/Users/bviggiano/.cursor/extensions/ms-python.debugpy-2025.18.0-darwin-arm64/.noConfigDebugAdapterEndpoints/endpoint-4a8ea84ff27e749c.txt
VSCODE_GIT_IPC_HANDLE=/var/folders/6f/gqwcqlqn3sxdz7rlzjxk7pgh0000gn/T/vscode-git-d21be58b9b.sock
VSCODE_INJECTION=1
VSCODE_PROFILE_INITIALIZED=1
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
XML_CATALOG_FILES=file:///Users/bviggiano/miniconda3/etc/xml/catalog file:///etc/xml/catalog
XPC_FLAGS=0x0
XPC_SERVICE_NAME=0
ZDOTDIR=/Users/bviggiano
_=/Users/bviggiano/miniconda3/bin/python
__CFBundleIdentifier=com.todesktop.230313mzl4w4u92
__CF_USER_TEXT_ENCODING=0x1F5:0x0:0x0
```

### Subprocess Environment (passed to tools)

```
CONDA_PREFIX=/Users/bviggiano/.proto/proto_tool_envs/viennarna_env
DETECTED_COMPUTE_PLATFORM=cpu
HF_HOME=/Users/bviggiano/.proto/proto_model_cache/huggingface
HOME=/Users/bviggiano
LANG=en_US.UTF-8
LD_LIBRARY_PATH=/Users/bviggiano/miniconda3/lib
LOGNAME=bviggiano
PATH=/Users/bviggiano/.proto/proto_tool_envs/viennarna_env/bin:/Users/bviggiano/.local/bin:/Users/bviggiano/.juliaup/bin:/Users/bviggiano/miniconda3/bin:/Users/bviggiano/miniconda3/condabin:/Library/Framew...
PIP_DEFAULT_TIMEOUT=300
PROTO_HOME=/Users/bviggiano/.proto
RECOMMENDED_JAX_SPEC=jax
RECOMMENDED_TORCH_INDEX=https://download.pytorch.org/whl/cpu
RECOMMENDED_TORCH_SPEC=torch
SHELL=/bin/zsh
TMPDIR=/var/folders/6f/gqwcqlqn3sxdz7rlzjxk7pgh0000gn/T/
TORCH_HOME=/Users/bviggiano/.proto/proto_model_cache/torch
USER=bviggiano
UV_HTTP_TIMEOUT=300
VIRTUAL_ENV=/Users/bviggiano/.proto/proto_tool_envs/viennarna_env
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Results by Category

### Causal Models (0/0)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `evo1-sample` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |
| `evo2-sample` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |
| `progen2-sample` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |

### Gene Annotation (5/5)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `blast-create-db` | no | ✅ | 57.0s | `d2604fc` ✱ | ✅ Pass |
| `crispr-tracr` | no | ✅ | 249.2s | `d2604fc` ✱ | ✅ Pass |
| `minced-crispr` | no | ✅ | 14.2s | `d2604fc` ✱ | ✅ Pass |
| `mmseqs-clustering` | no | ✅ | 17.6s | `d2604fc` ✱ | ✅ Pass |
| `pyhmmer-hmmscan` | no | ✅ | 18.2s | `d2604fc` ✱ | ✅ Pass |

### Inverse Folding (0/0)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `esm-if1-sample` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |
| `fampnn-pack` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |
| `ligandmpnn-sample` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |
| `proteinmpnn-sample` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |

### Masked Models (0/0)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `esm2-embedding` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |
| `esm3-embedding` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |

### Mutagenesis (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `random-nucleotide-sample` | no | - | 0.0s | `d2604fc` ✱ | ✅ Pass |
| `random-protein-sample` | no | - | 0.0s | `d2604fc` ✱ | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `orfipy-prediction` | no | ✅ | 18.0s | `d2604fc` ✱ | ✅ Pass |
| `prodigal-prediction` | no | ✅ | 15.6s | `d2604fc` ✱ | ✅ Pass |

### Rna Splicing (0/0)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `splice-transformer-prediction` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |

### Sequence Alignment (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `colabfold-search` | no | ✅ | 29.1s | `d2604fc` ✱ | ✅ Pass |
| `mafft-align` | no | ✅ | 22.1s | `d2604fc` ✱ | ✅ Pass |

### Sequence Scoring (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphagenome-predict-intervals` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |
| `borzoi-ensemble` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |
| `enformer-prediction` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |
| `segmasker-score` | no | ✅ | 29.2s | `d2604fc` ✱ | ✅ Pass |

### Structure Alignment (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `tmalign-alignment` | no | ✅ | 20.0s | `d2604fc` ✱ | ✅ Pass |
| `usalign-alignment` | no | ✅ | 23.8s | `d2604fc` ✱ | ✅ Pass |

### Structure Design (0/0)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `rfdiffusion3-design` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |

### Structure Dynamics (0/0)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `bioemu-sample` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |

### Structure Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphafold2-prediction` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |
| `alphafold3-prediction` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |
| `boltz2-prediction` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |
| `chai1-prediction` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |
| `esmfold-prediction` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |
| `protenix-prediction` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |
| `structure-metrics` | no | ✅ | 25.7s | `d2604fc` ✱ | ✅ Pass |
| `viennarna-prediction` | no | ✅ | 17.8s | `d2604fc` ✱ | ✅ Pass |

### Testing (0/0)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `mock-cli-multi-gpu-tool-run` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |
| `mock-cli-tool-run` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |
| `mock-jax-multi-gpu-tool-run` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |
| `mock-jax-tool-run` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |
| `mock-pytorch-multi-gpu-tool-run` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |
| `mock-pytorch-tool-run` | yes | - | - | `d2604fc` ✱ | ⏭️ Skip |

---
*Generated at 2026-04-01 09:54:33 by `pytest --env-report`*

<!-- env-report-data
[
  {
    "tool_name": "alphafold2-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold2-prediction]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: GPU not available')",
    "git_commit": "d2604fc6b288",
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
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: requires Chimera cluster')",
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "alphagenome-predict-intervals",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphagenome-predict-intervals]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: GPU not available')",
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "bioemu-sample",
    "category": "structure_dynamics",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[bioemu-sample]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: GPU not available')",
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "blast-create-db",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[blast-create-db]",
    "status": "passed",
    "duration_seconds": 56.99,
    "uses_gpu": false,
    "env_path": "/Users/bviggiano/.proto/proto_tool_envs/blast_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "boltz2-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[boltz2-prediction]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: GPU not available')",
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "borzoi-ensemble",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[borzoi-ensemble]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: GPU not available')",
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "chai1-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[chai1-prediction]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: GPU not available')",
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "colabfold-search",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[colabfold-search]",
    "status": "passed",
    "duration_seconds": 29.1,
    "uses_gpu": false,
    "env_path": "/Users/bviggiano/.proto/proto_tool_envs/colabfold_search_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "crispr-tracr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[crispr-tracr]",
    "status": "passed",
    "duration_seconds": 249.25,
    "uses_gpu": false,
    "env_path": "/Users/bviggiano/.proto/proto_tool_envs/crispr_tracr_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "enformer-prediction",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[enformer-prediction]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: GPU not available')",
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "esm-if1-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm-if1-sample]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: GPU not available')",
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "esm2-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm2-embedding]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: GPU not available')",
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "esm3-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm3-embedding]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: GPU not available')",
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "esmfold-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esmfold-prediction]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: GPU not available')",
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "evo1-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo1-sample]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: GPU not available')",
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "evo2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo2-sample]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: GPU not available')",
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "fampnn-pack",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[fampnn-pack]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: GPU not available')",
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "ligandmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ligandmpnn-sample]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: GPU not available')",
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "mafft-align",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mafft-align]",
    "status": "passed",
    "duration_seconds": 22.08,
    "uses_gpu": false,
    "env_path": "/Users/bviggiano/.proto/proto_tool_envs/mafft_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "minced-crispr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[minced-crispr]",
    "status": "passed",
    "duration_seconds": 14.23,
    "uses_gpu": false,
    "env_path": "/Users/bviggiano/.proto/proto_tool_envs/minced_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "mmseqs-clustering",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mmseqs-clustering]",
    "status": "passed",
    "duration_seconds": 17.6,
    "uses_gpu": false,
    "env_path": "/Users/bviggiano/.proto/proto_tool_envs/mmseqs_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "d2604fc6b288",
    "git_dirty": true
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
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: GPU not available')",
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "mock-cli-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-cli-tool-run]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: GPU not available')",
    "git_commit": "d2604fc6b288",
    "git_dirty": true
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
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: GPU not available')",
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "mock-jax-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-jax-tool-run]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: GPU not available')",
    "git_commit": "d2604fc6b288",
    "git_dirty": true
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
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: GPU not available')",
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "mock-pytorch-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-pytorch-tool-run]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: GPU not available')",
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "orfipy-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[orfipy-prediction]",
    "status": "passed",
    "duration_seconds": 17.99,
    "uses_gpu": false,
    "env_path": "/Users/bviggiano/.proto/proto_tool_envs/orfipy_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "prodigal-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[prodigal-prediction]",
    "status": "passed",
    "duration_seconds": 15.57,
    "uses_gpu": false,
    "env_path": "/Users/bviggiano/.proto/proto_tool_envs/prodigal_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "progen2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen2-sample]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: GPU not available')",
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "proteinmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[proteinmpnn-sample]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: GPU not available')",
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "protenix-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[protenix-prediction]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: GPU not available')",
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "pyhmmer-hmmscan",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pyhmmer-hmmscan]",
    "status": "passed",
    "duration_seconds": 18.16,
    "uses_gpu": false,
    "env_path": "/Users/bviggiano/.proto/proto_tool_envs/pyhmmer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "d2604fc6b288",
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
    "git_commit": "d2604fc6b288",
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
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "rfdiffusion3-design",
    "category": "structure_design",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[rfdiffusion3-design]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: GPU not available')",
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "segmasker-score",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[segmasker-score]",
    "status": "passed",
    "duration_seconds": 29.22,
    "uses_gpu": false,
    "env_path": "/Users/bviggiano/.proto/proto_tool_envs/segmasker_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "splice-transformer-prediction",
    "category": "rna_splicing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[splice-transformer-prediction]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/Users/bviggiano/Projects/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 92, 'Skipped: --env-report: GPU not available')",
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "structure-metrics",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[structure-metrics]",
    "status": "passed",
    "duration_seconds": 25.68,
    "uses_gpu": false,
    "env_path": "/Users/bviggiano/.proto/proto_tool_envs/structure_metrics_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "tmalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[tmalign-alignment]",
    "status": "passed",
    "duration_seconds": 20.01,
    "uses_gpu": false,
    "env_path": "/Users/bviggiano/.proto/proto_tool_envs/tmalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "usalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[usalign-alignment]",
    "status": "passed",
    "duration_seconds": 23.79,
    "uses_gpu": false,
    "env_path": "/Users/bviggiano/.proto/proto_tool_envs/usalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  },
  {
    "tool_name": "viennarna-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[viennarna-prediction]",
    "status": "passed",
    "duration_seconds": 17.84,
    "uses_gpu": false,
    "env_path": "/Users/bviggiano/.proto/proto_tool_envs/viennarna_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "d2604fc6b288",
    "git_dirty": true
  }
]
-->