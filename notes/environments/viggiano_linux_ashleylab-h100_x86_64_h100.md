# Linux x86_64 Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-95%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-45-brightgreen) ![Failed](https://img.shields.io/badge/failed-2-red) ![Skipped](https://img.shields.io/badge/skipped-1-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Linux Linux 5.15.0-1086-nvidia |
| **Architecture** | x86_64 |
| **Hostname** | `ashleylab-h100` |
| **Python** | 3.12.13 |
| **RAM** | 2015.6 GB |
| **GPU** | 8x NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3 |
| **CUDA** | 12.2 |
| **Conda Env** | `proto-tools` |

## Git

- **Commit**: `c0b536273d62`
- **Branch**: `feat/auto-persist-preprocess-scope`
- **Dirty**: No

## Environment Variables

### Parent Process Environment

```
BROWSER=/home/viggiano/.cursor-server/cli/servers/Stable-63715ffc1807793ce209e935e5c3ab9b79fddc80/server/bin/helpers/browser.sh
BUNDLED_DEBUGPY_PATH=/home/viggiano/.cursor-server/extensions/ms-python.debugpy-2025.18.0-linux-x64/bundled/libs/debugpy
CLAUDE_CODE_SSE_PORT=32633
COLORTERM=truecolor
CONDA_DEFAULT_ENV=proto-tools
CONDA_EXE=/home/viggiano/miniconda3/bin/conda
CONDA_PREFIX=/projects/viggiano/envs/proto-tools
CONDA_PREFIX_1=/home/viggiano/miniconda3
CONDA_PREFIX_2=/projects/viggiano/envs/proto-tools
CONDA_PREFIX_3=/home/viggiano/miniconda3
CONDA_PROMPT_MODIFIER=(proto-tools) 
CONDA_PYTHON_EXE=/home/viggiano/miniconda3/bin/python
CONDA_SHLVL=4
DISABLE_PANDERA_IMPORT_WARNING=True
GK_GL_ADDR=http://127.0.0.1:42177
GK_GL_PATH=/tmp/gitkraken/gitlens/gitlens-ipc-server-448565-42177.json
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
OVSX_REGISTRY_URL=https://open-vsx.org
PATH=/home/viggiano/.local/bin:/usr/local/cuda/bin:/opt/bin:/projects/viggiano/envs/proto-tools/bin:/home/viggiano/miniconda3/condabin:/home/viggiano/.cursor-server/cli/servers/Stable-d8673fb56ba50fda33ad7...
PROTO_HOME=/raid/projects/viggiano/codebases/proto-bio
PWD=/home/viggiano/main/codebases/proto-bio/proto-tools
PYDEVD_DISABLE_FILE_VALIDATION=1
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
RDBASE=/projects/viggiano/envs/proto-tools/lib/python3.12/site-packages/rdkit
SHELL=/bin/bash
SHLVL=3
TERM=tmux-256color
TERMINFO_DIRS=/home/viggiano/.terminfo:/home/viggiano/.terminfo:
TERM_PROGRAM=tmux
TERM_PROGRAM_VERSION=3.2a
TMUX=/tmp/tmux-1013/default,615045,2
TMUX_PANE=%2
USER=viggiano
VSCODE_DEBUGPY_ADAPTER_ENDPOINTS=/home/viggiano/.cursor-server/extensions/ms-python.debugpy-2025.18.0-linux-x64/.noConfigDebugAdapterEndpoints/endpoint-a9156f262dc46f33.txt
VSCODE_GIT_IPC_HANDLE=/run/user/1013/vscode-git-b2c13778b0.sock
VSCODE_IPC_HOOK_CLI=/run/user/1013/vscode-ipc-6c9ade80-79d6-4163-a09f-08a45c368743.sock
XDG_DATA_DIRS=/usr/local/share:/usr/share:/var/lib/snapd/desktop
XDG_RUNTIME_DIR=/run/user/1013
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
_=/projects/viggiano/envs/proto-tools/bin/pytest
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
PATH=/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/viennarna_env/bin:/home/viggiano/.local/bin:/usr/local/cuda/bin:/opt/bin:/projects/viggiano/envs/proto-tools/bin:/home/viggiano/miniconda3/c...
PIP_CACHE_DIR=/raid/projects/viggiano/codebases/proto-bio/pip_cache
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
UV_CACHE_DIR=/raid/projects/viggiano/codebases/proto-bio/uv_cache
UV_HTTP_TIMEOUT=300
VIRTUAL_ENV=/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/viennarna_env
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Results by Category

### Causal Models (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `evo1-sample` | yes | ✅ | 97.0s | `c0b5362` | ✅ Pass |
| `evo2-sample` | yes | ✅ | 91.0s | `c0b5362` | ✅ Pass |
| `progen2-sample` | yes | ✅ | 50.4s | `c0b5362` | ✅ Pass |
| `progen3-sample` | yes | ✅ | 23.9s | `c0b5362` | ✅ Pass |

### Gene Annotation (5/5)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `blast-create-db` | no | ✅ | 20.1s | `c0b5362` | ✅ Pass |
| `crispr-tracr` | no | ✅ | 95.6s | `c0b5362` | ✅ Pass |
| `minced-crispr` | no | ✅ | 5.8s | `c0b5362` | ✅ Pass |
| `mmseqs-clustering` | no | ✅ | 8.7s | `c0b5362` | ✅ Pass |
| `pyhmmer-hmmscan` | no | ✅ | 6.0s | `c0b5362` | ✅ Pass |

### Inverse Folding (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `esm-if1-sample` | yes | ✅ | 22.1s | `c0b5362` | ✅ Pass |
| `fampnn-pack` | yes | ✅ | 43.1s | `c0b5362` | ✅ Pass |
| `ligandmpnn-sample` | yes | ✅ | 42.2s | `c0b5362` | ✅ Pass |
| `proteinmpnn-sample` | yes | ✅ | 43.9s | `c0b5362` | ✅ Pass |

### Masked Models (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `ablang-embedding` | yes | ✅ | 501.7s | `c0b5362` | ✅ Pass |
| `esm2-embedding` | yes | ✅ | 18.4s | `c0b5362` | ✅ Pass |
| `esm3-embedding` | yes | ✅ | 18.9s | `c0b5362` | ✅ Pass |
| `esmc-embedding` | yes | ✅ | 21.2s | `c0b5362` | ✅ Pass |

### Mutagenesis (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `random-nucleotide-sample` | no | - | 0.0s | `c0b5362` | ✅ Pass |
| `random-protein-sample` | no | - | 0.0s | `c0b5362` | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `orfipy-prediction` | no | ✅ | 11.2s | `c0b5362` | ✅ Pass |
| `prodigal-prediction` | no | ✅ | 5.9s | `c0b5362` | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `splice-transformer-prediction` | yes | ✅ | 15.9s | `c0b5362` | ✅ Pass |

### Sequence Alignment (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `colabfold-search` | no | ✅ | 29.6s | `c0b5362` | ✅ Pass |
| `mafft-align` | no | ✅ | 8.4s | `c0b5362` | ✅ Pass |

### Sequence Scoring (3/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphagenome-predict-intervals` | yes | ✅ | 101.2s | `c0b5362` | ❌ Fail |
| `borzoi-ensemble` | yes | ✅ | 83.2s | `c0b5362` | ✅ Pass |
| `enformer-prediction` | yes | ✅ | 18.2s | `c0b5362` | ✅ Pass |
| `segmasker-score` | no | ✅ | 19.9s | `c0b5362` | ✅ Pass |

### Structure Alignment (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `tmalign-alignment` | no | ✅ | 21.9s | `c0b5362` | ✅ Pass |
| `usalign-alignment` | no | ✅ | 25.7s | `c0b5362` | ✅ Pass |

### Structure Design (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `rfdiffusion3-design` | yes | ✅ | 68.3s | `c0b5362` | ✅ Pass |

### Structure Dynamics (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `bioemu-sample` | yes | ✅ | 52.6s | `c0b5362` | ✅ Pass |

### Structure Prediction (5/6)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphafold2-binder` | yes | ✅ | 156.1s | `c0b5362` | ❌ Fail |
| `alphafold3-prediction` | yes | - | - | `c0b5362` | ⏭️ Skip |
| `boltz2-prediction` | yes | ✅ | 54.6s | `c0b5362` | ✅ Pass |
| `chai1-prediction` | yes | ✅ | 282.0s | `c0b5362` | ✅ Pass |
| `esmfold-prediction` | yes | ✅ | 28.2s | `c0b5362` | ✅ Pass |
| `protenix-prediction` | yes | ✅ | 248.4s | `c0b5362` | ✅ Pass |
| `viennarna-prediction` | no | ✅ | 6.2s | `c0b5362` | ✅ Pass |

### Structure Scoring (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `pdockq2` | no | - | 0.0s | `c0b5362` | ✅ Pass |
| `pyrosetta-energy` | no | ✅ | 54.3s | `c0b5362` | ✅ Pass |
| `structure-metrics` | no | ✅ | 6.5s | `c0b5362` | ✅ Pass |

### Testing (6/6)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `mock-cli-multi-gpu-tool-run` | yes | ✅ | 4.9s | `c0b5362` | ✅ Pass |
| `mock-cli-tool-run` | yes | ✅ | 4.8s | `c0b5362` | ✅ Pass |
| `mock-jax-multi-gpu-tool-run` | yes | ✅ | 27.5s | `c0b5362` | ✅ Pass |
| `mock-jax-tool-run` | yes | ✅ | 22.3s | `c0b5362` | ✅ Pass |
| `mock-pytorch-multi-gpu-tool-run` | yes | ✅ | 21.5s | `c0b5362` | ✅ Pass |
| `mock-pytorch-tool-run` | yes | ✅ | 20.6s | `c0b5362` | ✅ Pass |

## Failure Details

### ❌ `alphafold2-binder`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold2-binder]`

```
tests/tool_infra_tests/test_env_report.py:80: in test_tool_env_report
    assert_metrics_in_spec(result)
tests/tool_infra_tests/_metric_helpers.py:62: in assert_metrics_in_spec
    metrics.validate_against_spec()
proto_tools/utils/tool_io.py:310: in validate_against_spec
    _check_list_in_spec(name, value, value_spec)
proto_tools/utils/tool_io.py:357: in _check_list_in_spec
    raise AssertionError(
E   AssertionError: Metric 'pae' has type float but spec declares 'list[list[float]]'
```

### ❌ `alphagenome-predict-intervals`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphagenome-predict-intervals]`

```
tests/tool_infra_tests/test_env_report.py:79: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool alphagenome-predict-intervals failed: ["Command '['/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/alphagenome_env/bin/python', '/raid/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_scoring/alphagenome/standalone/inference.py', '/tmp/tmpnwmqim4r/input.json', '/tmp/tmpnwmqim4r/output.json']' returned non-zero exit status 1.", 'Traceback (most recent call last):\n  File "/raid/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py", line 497, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/raid/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_scoring/alphagenome/alphagenome_predict_intervals.py", line 148, in run_alphagenome_predict_intervals\n    dispatch_result = ToolInstance.dispatch(\n                      ^^^^^^^^^^^^^^^^^^^^^^\n  File "/raid/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 325, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/raid/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 364, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/raid/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 1077, in _run_oneshot\n    subprocess.run(\n  File "/projects/viggiano/envs/proto-tools/lib/python3.12/subprocess.py", line 571, in run\n    raise CalledProcessError(retcode, process.args,\nsubprocess.CalledProcessError: Command \'[\'/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/alphagenome_env/bin/python\', \'/raid/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_scoring/alphagenome/standalone/inference.py\', \'/tmp/tmpnwmqim4r/input.json\', \'/tmp/tmpnwmqim4r/output.json\']\' returned non-zero exit status 1.\n']
E   assert False
E    +  where False = AlphaGenomePredictIntervalsOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success
```

---
*Generated at 2026-04-22 00:12:15 by `pytest --env-report`*

<!-- env-report-data
[
  {
    "tool_key": "ablang-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ablang-embedding]",
    "status": "passed",
    "duration_seconds": 501.73,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/ablang_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "alphafold2-binder",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold2-binder]",
    "status": "failed",
    "duration_seconds": 156.07,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/alphafold2_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:80: in test_tool_env_report\n    assert_metrics_in_spec(result)\ntests/tool_infra_tests/_metric_helpers.py:62: in assert_metrics_in_spec\n    metrics.validate_against_spec()\nproto_tools/utils/tool_io.py:310: in validate_against_spec\n    _check_list_in_spec(name, value, value_spec)\nproto_tools/utils/tool_io.py:357: in _check_list_in_spec\n    raise AssertionError(\nE   AssertionError: Metric 'pae' has type float but spec declares 'list[list[float]]'",
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "alphafold3-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold3-prediction]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/raid/projects/viggiano/codebases/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 73, 'Skipped: --env-report: requires Chimera cluster')",
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "alphagenome-predict-intervals",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphagenome-predict-intervals]",
    "status": "failed",
    "duration_seconds": 101.16,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/alphagenome_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:79: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool alphagenome-predict-intervals failed: [\"Command '['/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/alphagenome_env/bin/python', '/raid/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_scoring/alphagenome/standalone/inference.py', '/tmp/tmpnwmqim4r/input.json', '/tmp/tmpnwmqim4r/output.json']' returned non-zero exit status 1.\", 'Traceback (most recent call last):\\n  File \"/raid/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py\", line 497, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/raid/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_scoring/alphagenome/alphagenome_predict_intervals.py\", line 148, in run_alphagenome_predict_intervals\\n    dispatch_result = ToolInstance.dispatch(\\n                      ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/raid/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 325, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/raid/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 364, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/raid/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 1077, in _run_oneshot\\n    subprocess.run(\\n  File \"/projects/viggiano/envs/proto-tools/lib/python3.12/subprocess.py\", line 571, in run\\n    raise CalledProcessError(retcode, process.args,\\nsubprocess.CalledProcessError: Command \\'[\\'/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/alphagenome_env/bin/python\\', \\'/raid/projects/viggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_scoring/alphagenome/standalone/inference.py\\', \\'/tmp/tmpnwmqim4r/input.json\\', \\'/tmp/tmpnwmqim4r/output.json\\']\\' returned non-zero exit status 1.\\n']\nE   assert False\nE    +  where False = AlphaGenomePredictIntervalsOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success",
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "bioemu-sample",
    "category": "structure_dynamics",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[bioemu-sample]",
    "status": "passed",
    "duration_seconds": 52.64,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/bioemu_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "blast-create-db",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[blast-create-db]",
    "status": "passed",
    "duration_seconds": 20.09,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/blast_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "boltz2-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[boltz2-prediction]",
    "status": "passed",
    "duration_seconds": 54.6,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/boltz2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "borzoi-ensemble",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[borzoi-ensemble]",
    "status": "passed",
    "duration_seconds": 83.21,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/borzoi_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "chai1-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[chai1-prediction]",
    "status": "passed",
    "duration_seconds": 282.02,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/chai1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "colabfold-search",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[colabfold-search]",
    "status": "passed",
    "duration_seconds": 29.59,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/colabfold_search_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "crispr-tracr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[crispr-tracr]",
    "status": "passed",
    "duration_seconds": 95.59,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/crispr_tracr_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "enformer-prediction",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[enformer-prediction]",
    "status": "passed",
    "duration_seconds": 18.24,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/enformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "esm-if1-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm-if1-sample]",
    "status": "passed",
    "duration_seconds": 22.07,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/esm_if1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "esm2-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm2-embedding]",
    "status": "passed",
    "duration_seconds": 18.45,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/esm2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "esm3-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm3-embedding]",
    "status": "passed",
    "duration_seconds": 18.87,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/evolutionaryscale_esm_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "esmc-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esmc-embedding]",
    "status": "passed",
    "duration_seconds": 21.24,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/evolutionaryscale_esm_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "esmfold-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esmfold-prediction]",
    "status": "passed",
    "duration_seconds": 28.2,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/esmfold_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "evo1-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo1-sample]",
    "status": "passed",
    "duration_seconds": 96.97,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/evo1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "evo2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo2-sample]",
    "status": "passed",
    "duration_seconds": 91.0,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/evo2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "fampnn-pack",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[fampnn-pack]",
    "status": "passed",
    "duration_seconds": 43.08,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/fampnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "ligandmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ligandmpnn-sample]",
    "status": "passed",
    "duration_seconds": 42.23,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/ligandmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "mafft-align",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mafft-align]",
    "status": "passed",
    "duration_seconds": 8.44,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/mafft_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "minced-crispr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[minced-crispr]",
    "status": "passed",
    "duration_seconds": 5.79,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/minced_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "mmseqs-clustering",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mmseqs-clustering]",
    "status": "passed",
    "duration_seconds": 8.74,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/mmseqs_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "mock-cli-multi-gpu-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-cli-multi-gpu-tool-run]",
    "status": "passed",
    "duration_seconds": 4.92,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/mock_cli_multi_gpu_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "mock-cli-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-cli-tool-run]",
    "status": "passed",
    "duration_seconds": 4.78,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/mock_cli_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "mock-jax-multi-gpu-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-jax-multi-gpu-tool-run]",
    "status": "passed",
    "duration_seconds": 27.5,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/mock_jax_multi_gpu_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "mock-jax-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-jax-tool-run]",
    "status": "passed",
    "duration_seconds": 22.28,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/mock_jax_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "mock-pytorch-multi-gpu-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-pytorch-multi-gpu-tool-run]",
    "status": "passed",
    "duration_seconds": 21.47,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/mock_pytorch_multi_gpu_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "mock-pytorch-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-pytorch-tool-run]",
    "status": "passed",
    "duration_seconds": 20.57,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/mock_pytorch_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "orfipy-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[orfipy-prediction]",
    "status": "passed",
    "duration_seconds": 11.17,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/orfipy_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "pdockq2",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pdockq2]",
    "status": "passed",
    "duration_seconds": 0.0,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "prodigal-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[prodigal-prediction]",
    "status": "passed",
    "duration_seconds": 5.87,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/prodigal_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "progen2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen2-sample]",
    "status": "passed",
    "duration_seconds": 50.44,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/progen2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "progen3-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen3-sample]",
    "status": "passed",
    "duration_seconds": 23.9,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/progen3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "proteinmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[proteinmpnn-sample]",
    "status": "passed",
    "duration_seconds": 43.86,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/proteinmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "protenix-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[protenix-prediction]",
    "status": "passed",
    "duration_seconds": 248.39,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/protenix_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "pyhmmer-hmmscan",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pyhmmer-hmmscan]",
    "status": "passed",
    "duration_seconds": 6.03,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/pyhmmer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "pyrosetta-energy",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pyrosetta-energy]",
    "status": "passed",
    "duration_seconds": 54.26,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/pyrosetta_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
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
    "git_commit": "c0b536273d62",
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
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "rfdiffusion3-design",
    "category": "structure_design",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[rfdiffusion3-design]",
    "status": "passed",
    "duration_seconds": 68.27,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/rfdiffusion3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "segmasker-score",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[segmasker-score]",
    "status": "passed",
    "duration_seconds": 19.91,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/segmasker_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "splice-transformer-prediction",
    "category": "rna_splicing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[splice-transformer-prediction]",
    "status": "passed",
    "duration_seconds": 15.89,
    "uses_gpu": true,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/splice_transformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "structure-metrics",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[structure-metrics]",
    "status": "passed",
    "duration_seconds": 6.45,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/structure_metrics_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "tmalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[tmalign-alignment]",
    "status": "passed",
    "duration_seconds": 21.89,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/tmalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "usalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[usalign-alignment]",
    "status": "passed",
    "duration_seconds": 25.72,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/usalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "viennarna-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[viennarna-prediction]",
    "status": "passed",
    "duration_seconds": 6.2,
    "uses_gpu": false,
    "env_path": "/raid/projects/viggiano/codebases/proto-bio/proto_tool_envs/viennarna_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  }
]
-->