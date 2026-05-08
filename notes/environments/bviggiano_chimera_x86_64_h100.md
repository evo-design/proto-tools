# Linux x86_64 Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-96%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-50-brightgreen) ![Failed](https://img.shields.io/badge/failed-2-red) ![Skipped](https://img.shields.io/badge/skipped-3-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Linux Linux 5.15.0-176-generic |
| **Architecture** | x86_64 |
| **Hostname** | `GPU71E4` |
| **Python** | 3.14.4 |
| **RAM** | 1007.4 GB |
| **GPU** | 1x NVIDIA H100 80GB HBM3 |
| **CUDA** | 12.6 |
| **Mamba Env** | `proto-tools` |

## Git

- **Commit**: `ecb1d6060109`
- **Branch**: `chore/refresh-env-report-proteinmpnn`
- **Dirty**: No

## Environment Variables

### Parent Process Environment

```
AI_AGENT=claude-code_2-1-133_agent
BLASTDB=/common_datasets/external/databases/blast
BROWSER=/home/bviggiano/.vscode-server/cli/servers/Stable-8b640eef5a6c6089c029249d48efa5c99adf7d51/server/bin/helpers/browser.sh
BUNDLED_DEBUGPY_PATH=/home/bviggiano/.vscode-server/extensions/ms-python.debugpy-2026.6.0-linux-x64/bundled/libs/debugpy
CLAUDECODE=1
CLAUDE_CODE_ENTRYPOINT=cli
CLAUDE_CODE_EXECPATH=/home/bviggiano/.local/share/claude/versions/2.1.133
CLAUDE_CODE_SSE_PORT=23985
CLAUDE_EFFORT=xhigh
COLORTERM=truecolor
CONDA_DEFAULT_ENV=proto-tools
CONDA_EXE=/home/bviggiano/miniforge3/bin/conda
CONDA_PREFIX=/home/bviggiano/miniforge3/envs/proto-tools
CONDA_PREFIX_1=/home/bviggiano/miniforge3
CONDA_PROMPT_MODIFIER=(proto-tools) 
CONDA_PYTHON_EXE=/home/bviggiano/miniforge3/bin/python
CONDA_SHLVL=2
COREPACK_ENABLE_AUTO_PIN=0
DISABLE_PANDERA_IMPORT_WARNING=True
GIT_EDITOR=true
GK_GL_ADDR=http://127.0.0.1:39849
GK_GL_PATH=/tmp/gitkraken/gitlens/gitlens-ipc-server-1327380-39849.json
HOME=/home/bviggiano
LANG=C.UTF-8
LD_LIBRARY_PATH=/usr/local/cuda/lib64
LESSCLOSE=/usr/bin/lesspipe %s %s
LESSOPEN=| /usr/bin/lesspipe %s
LOGNAME=bviggiano
LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.a...
MAMBA_EXE=/home/bviggiano/miniforge3/bin/mamba
MAMBA_ROOT_PREFIX=/home/bviggiano/.local/share/mamba
MOTD_SHOWN=pam
NoDefaultCurrentDirectoryInExePath=1
PATH=/home/bviggiano/miniforge3/envs/proto-tools/bin:/home/bviggiano/miniforge3/condabin:/home/bviggiano/.vscode-server/data/User/globalStorage/github.copilot-chat/debugCommand:/home/bviggiano/.vscode-serv...
PROTO_ALPHAFOLD3_WEIGHTS_DIR=/large_storage/hielab/brk/models/af3_weights
PROTO_HOME=/large_storage/hielab/bviggiano/proto_cache
PWD=/home/bviggiano/main/codebases/evo-design/proto-tools
PYDEVD_DISABLE_FILE_VALIDATION=1
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.3
PYTHONSTARTUP=/home/bviggiano/.vscode-server/data/User/workspaceStorage/606b48bbf88196b292aff26cc9961831/ms-python.python/pythonrc.py
PYTHON_BASIC_REPL=1
RCLONE_CONFIG=/large_storage/rclone/etc/rclone.conf
SHELL=/bin/bash
SHLVL=2
SLURM_JOB_ID=2290692
TERM=xterm-256color
TERM_PROGRAM=vscode
TERM_PROGRAM_VERSION=1.119.0
USER=bviggiano
VSCODE_DEBUGPY_ADAPTER_ENDPOINTS=/home/bviggiano/.vscode-server/extensions/ms-python.debugpy-2026.6.0-linux-x64/.noConfigDebugAdapterEndpoints/endpoint-1e8e4f814d451fc8.txt
VSCODE_GIT_IPC_HANDLE=/tmp/vscode-git-2e22fcc75c.sock
VSCODE_IPC_HOOK_CLI=/tmp/vscode-ipc-68eb0521-3803-49f9-9783-12095e63840d.sock
VSCODE_PYTHON_AUTOACTIVATE_GUARD=1
XDG_DATA_DIRS=/usr/local/share:/usr/share:/var/lib/snapd/desktop
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
XML_CATALOG_FILES=file:///home/bviggiano/miniforge3/etc/xml/catalog file:///etc/xml/catalog
_=/home/bviggiano/miniforge3/envs/proto-tools/bin/pytest
_CE_CONDA=
_CE_M=
_CONDA_EXE=/home/bviggiano/miniforge3/bin/conda
_CONDA_ROOT=/home/bviggiano/miniforge3
```

### Subprocess Environment (passed to tools)

```
CONDA_PREFIX=/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/proteinmpnn_env
DETECTED_COMPUTE_PLATFORM=cuda
DETECTED_CUDA_VERSION=12
DETECTED_DRIVER_VERSION=560
HF_HOME=/large_storage/hielab/bviggiano/proto_cache/proto_model_cache/huggingface
HOME=/home/bviggiano
LANG=C.UTF-8
LD_LIBRARY_PATH=/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/proteinmpnn_env/cuda_env/lib:/usr/local/cuda/lib64:/home/bviggiano/miniforge3/envs/proto-tools/lib
LOGNAME=bviggiano
PATH=/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/proteinmpnn_env/bin:/usr/local/cuda/bin:/home/bviggiano/miniforge3/envs/proto-tools/bin:/home/bviggiano/miniforge3/condabin:/home/bviggiano/...
PIP_CACHE_DIR=/large_storage/hielab/bviggiano/proto_cache/pip_cache
PIP_DEFAULT_TIMEOUT=300
PROTO_ALPHAFOLD3_WEIGHTS_DIR=/large_storage/hielab/brk/models/af3_weights
PROTO_HOME=/large_storage/hielab/bviggiano/proto_cache
RECOMMENDED_JAX_SPEC=jax[cuda12]>=0.4.20,<1
RECOMMENDED_JAX_VARIANT=cuda12
RECOMMENDED_TORCH_INDEX=https://download.pytorch.org/whl/cu126
RECOMMENDED_TORCH_SPEC=torch>=2.5,<3
SHELL=/bin/bash
TORCH_CUDA_ARCH_LIST=9.0
TORCH_HOME=/large_storage/hielab/bviggiano/proto_cache/proto_model_cache/torch
USER=bviggiano
UV_CACHE_DIR=/large_storage/hielab/bviggiano/proto_cache/uv_cache
UV_HTTP_TIMEOUT=300
VIRTUAL_ENV=/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/proteinmpnn_env
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Results by Category

### Binder Design (0/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `germinal-design` | yes | ✅ | 3766.7s | `c2fad93` | ❌ Fail |

### Causal Models (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `evo1-sample` | yes | ✅ | 215.8s | `c2fad93` | ✅ Pass |
| `evo2-sample` | yes | ✅ | 280.8s | `c2fad93` | ✅ Pass |
| `progen2-sample` | yes | ✅ | 112.5s | `c2fad93` | ✅ Pass |
| `progen3-sample` | yes | ✅ | 51.6s | `c2fad93` | ✅ Pass |

### Gene Annotation (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `crispr-tracr-rna` | no | ✅ | 199.5s | `c2fad93` | ✅ Pass |
| `minced-crispr` | no | ✅ | 17.9s | `c2fad93` | ✅ Pass |
| `promoter-calculator` | no | ✅ | 22.9s | `c2fad93` | ✅ Pass |
| `pyhmmer-hmmscan` | no | ✅ | 18.7s | `c2fad93` | ✅ Pass |

### Inverse Folding (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `esm-if1-sample` | yes | ✅ | 65.3s | `c2fad93` | ✅ Pass |
| `fampnn-pack` | yes | ✅ | 97.0s | `c2fad93` | ✅ Pass |
| `ligandmpnn-sample` | yes | ✅ | 157.8s | `c2fad93` | ✅ Pass |
| `proteinmpnn-gradient` | yes | ✅ | 112.2s | `ecb1d60` | ✅ Pass |

### Masked Models (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `ablang-embedding` | yes | ✅ | 94.3s | `c2fad93` | ✅ Pass |
| `esm2-embedding` | yes | ✅ | 53.2s | `c2fad93` | ✅ Pass |
| `esm3-embedding` | yes | ✅ | 10.7s | `c2fad93` | ✅ Pass |
| `esmc-embedding` | yes | ✅ | 62.6s | `c2fad93` | ✅ Pass |

### Mutagenesis (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `random-nucleotide-sample` | no | - | 0.0s | `c2fad93` | ✅ Pass |
| `random-protein-sample` | no | - | 0.0s | `c2fad93` | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `orfipy-prediction` | no | ✅ | 19.5s | `c2fad93` | ✅ Pass |
| `prodigal-prediction` | no | ✅ | 18.1s | `c2fad93` | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `splice-transformer-prediction` | yes | ✅ | 64.4s | `c2fad93` | ✅ Pass |

### Sequence Alignment (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `blast-create-db` | no | ✅ | 32.5s | `c2fad93` | ✅ Pass |
| `colabfold-search` | no | ✅ | 46.2s | `c2fad93` | ✅ Pass |
| `mafft-align` | no | ✅ | 26.8s | `c2fad93` | ✅ Pass |
| `mmseqs2-clustering` | no | ✅ | 47.5s | `c2fad93` | ✅ Pass |

### Sequence Scoring (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphagenome-predict-intervals` | yes | ✅ | 253.1s | `c2fad93` | ✅ Pass |
| `borzoi-ensemble` | yes | ✅ | 118.0s | `c2fad93` | ✅ Pass |
| `enformer-prediction` | yes | ✅ | 46.0s | `c2fad93` | ✅ Pass |
| `segmasker-score` | no | ✅ | 36.6s | `c2fad93` | ✅ Pass |

### Structure Alignment (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `foldmason-msa` | no | - | 1.8s | `e347546` ✱ | ✅ Pass |
| `foldseek-cluster` | no | ✅ | 30.7s | `e347546` ✱ | ✅ Pass |
| `tmalign-alignment` | no | ✅ | 27.3s | `c2fad93` | ✅ Pass |
| `usalign-alignment` | no | ✅ | 36.5s | `c2fad93` | ✅ Pass |

### Structure Design (1/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `bindcraft-design` | yes | ✅ | 2709.7s | `c2fad93` | ❌ Fail |
| `rfdiffusion3-design` | yes | ✅ | 175.4s | `c2fad93` | ✅ Pass |

### Structure Dynamics (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `bioemu-sample` | yes | ✅ | 142.1s | `c2fad93` | ✅ Pass |

### Structure Prediction (7/7)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphafold2-binder` | yes | ✅ | 229.9s | `c2fad93` | ✅ Pass |
| `alphafold3-prediction` | yes | ✅ | 435.9s | `c2fad93` | ✅ Pass |
| `boltz2-prediction` | yes | ✅ | 154.9s | `c2fad93` | ✅ Pass |
| `chai1-prediction` | yes | ✅ | 245.9s | `c2fad93` | ✅ Pass |
| `esmfold-gradient` | yes | ✅ | 66.8s | `c2fad93` | ✅ Pass |
| `protenix-prediction` | yes | ✅ | 455.6s | `c2fad93` | ✅ Pass |
| `viennarna-prediction` | no | ✅ | 16.7s | `c2fad93` | ✅ Pass |

### Structure Scoring (5/5)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `dssp-secondary-structure` | no | ✅ | 26.4s | `c2fad93` | ✅ Pass |
| `ipsae-scoring` | no | ✅ | 17.8s | `c2fad93` | ✅ Pass |
| `pdockq2` | no | - | 0.0s | `c2fad93` | ✅ Pass |
| `pyrosetta-energy` | no | ✅ | 163.6s | `c2fad93` | ✅ Pass |
| `structure-metrics` | no | - | 1.2s | `c2fad93` | ✅ Pass |

### Testing (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `mock-cli-multi-gpu-tool-run` | yes | - | - | `c2fad93` | ⏭️ Skip |
| `mock-cli-tool-run` | yes | ✅ | 15.1s | `c2fad93` | ✅ Pass |
| `mock-jax-multi-gpu-tool-run` | yes | - | - | `c2fad93` | ⏭️ Skip |
| `mock-jax-tool-run` | yes | ✅ | 54.8s | `c2fad93` | ✅ Pass |
| `mock-pytorch-multi-gpu-tool-run` | yes | - | - | `c2fad93` | ⏭️ Skip |
| `mock-pytorch-tool-run` | yes | ✅ | 72.9s | `c2fad93` | ✅ Pass |

## Failure Details

### ❌ `bindcraft-design`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[bindcraft-design]`

```
tests/tool_infra_tests/test_env_report.py:79: in test_tool_env_report
    assert_metrics_in_spec(result)
tests/tool_infra_tests/_metric_helpers.py:62: in assert_metrics_in_spec
    metrics.validate_against_spec()
proto_tools/utils/tool_io.py:348: in validate_against_spec
    _check_scalar_in_spec(name, value, value_spec)
proto_tools/utils/tool_io.py:377: in _check_scalar_in_spec
    raise AssertionError(f"Metric {name!r}={value} above declared max {max_v}")
E   AssertionError: Metric 'interface_hydrophobicity'=47.22 above declared max 1.0
```

### ❌ `germinal-design`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[germinal-design]`

```
tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool germinal-design failed: ['TimeoutError: germinal: timed out after 3600s', 'Traceback (most recent call last):\n  File "/large_storage/hielab/bviggiano/codebases/evo-design/proto-tools/proto_tools/tools/tool_registry.py", line 588, in _wrapper_body\n    result = func(inputs, config, instance)\n  File "/large_storage/hielab/bviggiano/codebases/evo-design/proto-tools/proto_tools/tools/binder_design/germinal/germinal_design.py", line 609, in run_germinal_design\n    output_data = ToolInstance.dispatch("germinal", input_data, instance=instance, config=config)\n  File "/large_storage/hielab/bviggiano/codebases/evo-design/proto-tools/proto_tools/utils/tool_instance.py", line 432, in dispatch\n    return cls._oneshot(\n           ~~~~~~~~~~~~^\n        toolkit,\n        ^^^^^^^^\n    ...<3 lines>...\n        timeout=timeout,\n        ^^^^^^^^^^^^^^^^\n    )\n    ^\n  File "/large_storage/hielab/bviggiano/codebases/evo-design/proto-tools/proto_tools/utils/tool_instance.py", line 471, in _oneshot\n    return inst._run_oneshot(\n           ~~~~~~~~~~~~~~~~~^\n        leased_input,\n        ^^^^^^^^^^^^^\n    ...<2 lines>...\n        timeout=timeout,\n        ^^^^^^^^^^^^^^^^\n    )\n    ^\n  File "/large_storage/hielab/bviggiano/codebases/evo-design/proto-tools/proto_tools/utils/tool_instance.py", line 1275, in _run_oneshot\n    raise TimeoutError(f"{self.toolkit}: timed out after {effective_timeout}s") from None\nTimeoutError: germinal: timed out after 3600s\n']
E   assert False
E    +  where False = GerminalOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, designs, pipeline_stats, num_accepted, num_designs).success
```

---
*Generated at 2026-05-08 10:07:27 by `pytest --env-report`*

<!-- env-report-data
[
  {
    "tool_key": "usalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[usalign-alignment]",
    "status": "passed",
    "duration_seconds": 36.55,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/usalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "structure-metrics",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[structure-metrics]",
    "status": "passed",
    "duration_seconds": 1.22,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "mock-pytorch-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-pytorch-tool-run]",
    "status": "passed",
    "duration_seconds": 72.92,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mock_pytorch_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "enformer-prediction",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[enformer-prediction]",
    "status": "passed",
    "duration_seconds": 46.03,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/enformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
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
    "error_message": "('/large_storage/hielab/bviggiano/codebases/evo-design/proto-tools/tests/tool_infra_tests/test_env_report.py', 72, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "promoter-calculator",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[promoter-calculator]",
    "status": "passed",
    "duration_seconds": 22.95,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/promoter_calculator_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "minced-crispr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[minced-crispr]",
    "status": "passed",
    "duration_seconds": 17.86,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/minced_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "alphafold3-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold3-prediction]",
    "status": "passed",
    "duration_seconds": 435.9,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphafold3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "mmseqs2-clustering",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mmseqs2-clustering]",
    "status": "passed",
    "duration_seconds": 47.45,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mmseqs2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "viennarna-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[viennarna-prediction]",
    "status": "passed",
    "duration_seconds": 16.72,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/viennarna_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "ablang-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ablang-embedding]",
    "status": "passed",
    "duration_seconds": 94.31,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/ablang_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
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
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "ipsae-scoring",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ipsae-scoring]",
    "status": "passed",
    "duration_seconds": 17.84,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/ipsae_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "chai1-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[chai1-prediction]",
    "status": "passed",
    "duration_seconds": 245.89,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/chai1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "fampnn-pack",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[fampnn-pack]",
    "status": "passed",
    "duration_seconds": 97.0,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/fampnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "evo1-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo1-sample]",
    "status": "passed",
    "duration_seconds": 215.8,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/evo1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "alphafold2-binder",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold2-binder]",
    "status": "passed",
    "duration_seconds": 229.93,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphafold2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "pyhmmer-hmmscan",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pyhmmer-hmmscan]",
    "status": "passed",
    "duration_seconds": 18.73,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/pyhmmer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "ligandmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ligandmpnn-sample]",
    "status": "passed",
    "duration_seconds": 157.77,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/ligandmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "esm2-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm2-embedding]",
    "status": "passed",
    "duration_seconds": 53.18,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/esm2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "crispr-tracr-rna",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[crispr-tracr-rna]",
    "status": "passed",
    "duration_seconds": 199.51,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/crispr_tracr_rna_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
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
    "error_message": "('/large_storage/hielab/bviggiano/codebases/evo-design/proto-tools/tests/tool_infra_tests/test_env_report.py', 72, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "bindcraft-design",
    "category": "structure_design",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[bindcraft-design]",
    "status": "failed",
    "duration_seconds": 2709.68,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/bindcraft_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:79: in test_tool_env_report\n    assert_metrics_in_spec(result)\ntests/tool_infra_tests/_metric_helpers.py:62: in assert_metrics_in_spec\n    metrics.validate_against_spec()\nproto_tools/utils/tool_io.py:348: in validate_against_spec\n    _check_scalar_in_spec(name, value, value_spec)\nproto_tools/utils/tool_io.py:377: in _check_scalar_in_spec\n    raise AssertionError(f\"Metric {name!r}={value} above declared max {max_v}\")\nE   AssertionError: Metric 'interface_hydrophobicity'=47.22 above declared max 1.0",
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "random-nucleotide-sample",
    "category": "mutagenesis",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[random-nucleotide-sample]",
    "status": "passed",
    "duration_seconds": 0.0,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "evo2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo2-sample]",
    "status": "passed",
    "duration_seconds": 280.84,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/evo2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "protenix-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[protenix-prediction]",
    "status": "passed",
    "duration_seconds": 455.65,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/protenix_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "pyrosetta-energy",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pyrosetta-energy]",
    "status": "passed",
    "duration_seconds": 163.61,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/pyrosetta_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "blast-create-db",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[blast-create-db]",
    "status": "passed",
    "duration_seconds": 32.54,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/blast_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "borzoi-ensemble",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[borzoi-ensemble]",
    "status": "passed",
    "duration_seconds": 118.0,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/borzoi_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
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
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "splice-transformer-prediction",
    "category": "rna_splicing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[splice-transformer-prediction]",
    "status": "passed",
    "duration_seconds": 64.44,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/splice_transformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "orfipy-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[orfipy-prediction]",
    "status": "passed",
    "duration_seconds": 19.46,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/orfipy_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "esmfold-gradient",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esmfold-gradient]",
    "status": "passed",
    "duration_seconds": 66.75,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/esmfold_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "segmasker-score",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[segmasker-score]",
    "status": "passed",
    "duration_seconds": 36.57,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/segmasker_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "germinal-design",
    "category": "binder_design",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[germinal-design]",
    "status": "failed",
    "duration_seconds": 3766.69,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/germinal_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool germinal-design failed: ['TimeoutError: germinal: timed out after 3600s', 'Traceback (most recent call last):\\n  File \"/large_storage/hielab/bviggiano/codebases/evo-design/proto-tools/proto_tools/tools/tool_registry.py\", line 588, in _wrapper_body\\n    result = func(inputs, config, instance)\\n  File \"/large_storage/hielab/bviggiano/codebases/evo-design/proto-tools/proto_tools/tools/binder_design/germinal/germinal_design.py\", line 609, in run_germinal_design\\n    output_data = ToolInstance.dispatch(\"germinal\", input_data, instance=instance, config=config)\\n  File \"/large_storage/hielab/bviggiano/codebases/evo-design/proto-tools/proto_tools/utils/tool_instance.py\", line 432, in dispatch\\n    return cls._oneshot(\\n           ~~~~~~~~~~~~^\\n        toolkit,\\n        ^^^^^^^^\\n    ...<3 lines>...\\n        timeout=timeout,\\n        ^^^^^^^^^^^^^^^^\\n    )\\n    ^\\n  File \"/large_storage/hielab/bviggiano/codebases/evo-design/proto-tools/proto_tools/utils/tool_instance.py\", line 471, in _oneshot\\n    return inst._run_oneshot(\\n           ~~~~~~~~~~~~~~~~~^\\n        leased_input,\\n        ^^^^^^^^^^^^^\\n    ...<2 lines>...\\n        timeout=timeout,\\n        ^^^^^^^^^^^^^^^^\\n    )\\n    ^\\n  File \"/large_storage/hielab/bviggiano/codebases/evo-design/proto-tools/proto_tools/utils/tool_instance.py\", line 1275, in _run_oneshot\\n    raise TimeoutError(f\"{self.toolkit}: timed out after {effective_timeout}s\") from None\\nTimeoutError: germinal: timed out after 3600s\\n']\nE   assert False\nE    +  where False = GerminalOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, designs, pipeline_stats, num_accepted, num_designs).success",
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "proteinmpnn-gradient",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[proteinmpnn-gradient]",
    "status": "passed",
    "duration_seconds": 112.22,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/proteinmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ecb1d6060109",
    "git_dirty": false
  },
  {
    "tool_key": "mock-cli-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-cli-tool-run]",
    "status": "passed",
    "duration_seconds": 15.13,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mock_cli_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "dssp-secondary-structure",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[dssp-secondary-structure]",
    "status": "passed",
    "duration_seconds": 26.44,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/dssp_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "esm-if1-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm-if1-sample]",
    "status": "passed",
    "duration_seconds": 65.31,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/esm_if1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "boltz2-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[boltz2-prediction]",
    "status": "passed",
    "duration_seconds": 154.93,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/boltz2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "tmalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[tmalign-alignment]",
    "status": "passed",
    "duration_seconds": 27.26,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/tmalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "bioemu-sample",
    "category": "structure_dynamics",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[bioemu-sample]",
    "status": "passed",
    "duration_seconds": 142.07,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/bioemu_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "colabfold-search",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[colabfold-search]",
    "status": "passed",
    "duration_seconds": 46.17,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/colabfold_search_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "esmc-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esmc-embedding]",
    "status": "passed",
    "duration_seconds": 62.58,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/evolutionaryscale_esm_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "progen2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen2-sample]",
    "status": "passed",
    "duration_seconds": 112.48,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/progen2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "progen3-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen3-sample]",
    "status": "passed",
    "duration_seconds": 51.64,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/progen3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
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
    "error_message": "('/large_storage/hielab/bviggiano/codebases/evo-design/proto-tools/tests/tool_infra_tests/test_env_report.py', 72, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "foldmason-msa",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[foldmason-msa]",
    "status": "passed",
    "duration_seconds": 1.79,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "e347546d1f64",
    "git_dirty": true
  },
  {
    "tool_key": "mock-jax-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-jax-tool-run]",
    "status": "passed",
    "duration_seconds": 54.78,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mock_jax_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "mafft-align",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mafft-align]",
    "status": "passed",
    "duration_seconds": 26.78,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mafft_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "prodigal-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[prodigal-prediction]",
    "status": "passed",
    "duration_seconds": 18.06,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/prodigal_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "alphagenome-predict-intervals",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphagenome-predict-intervals]",
    "status": "passed",
    "duration_seconds": 253.14,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphagenome_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "foldseek-cluster",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[foldseek-cluster]",
    "status": "passed",
    "duration_seconds": 30.66,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/foldseek_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "e347546d1f64",
    "git_dirty": true
  },
  {
    "tool_key": "rfdiffusion3-design",
    "category": "structure_design",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[rfdiffusion3-design]",
    "status": "passed",
    "duration_seconds": 175.42,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/rfdiffusion3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  },
  {
    "tool_key": "esm3-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm3-embedding]",
    "status": "passed",
    "duration_seconds": 10.67,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/evolutionaryscale_esm_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c2fad93fa119",
    "git_dirty": false
  }
]
-->