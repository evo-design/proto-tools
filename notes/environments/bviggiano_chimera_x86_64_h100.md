# Linux x86_64 Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-95%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-43-brightgreen) ![Failed](https://img.shields.io/badge/failed-2-red) ![Skipped](https://img.shields.io/badge/skipped-3-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Linux Linux 5.15.0-171-generic |
| **Architecture** | x86_64 |
| **Hostname** | `GPUC960` |
| **Python** | 3.14.4 |
| **RAM** | 1007.4 GB |
| **GPU** | 1x NVIDIA H100 80GB HBM3 |
| **CUDA** | 12.2 |
| **Mamba Env** | `proto-tools` |

## Git

- **Commit**: `c0b536273d62`
- **Branch**: `feat/auto-persist-preprocess-scope`
- **Dirty**: No

## Environment Variables

### Parent Process Environment

```
BLASTDB=/common_datasets/external/databases/blast
BROWSER=/home/bviggiano/.cursor-server/cli/servers/Stable-d8673fb56ba50fda33ad78382000b519bb8acb70/server/bin/helpers/browser.sh
BUNDLED_DEBUGPY_PATH=/home/bviggiano/.cursor-server/extensions/ms-python.debugpy-2025.18.0-linux-x64/bundled/libs/debugpy
CLAUDE_CODE_SSE_PORT=62782
COLORTERM=truecolor
CONDA_DEFAULT_ENV=proto-tools
CONDA_EXE=/home/bviggiano/miniforge3/bin/conda
CONDA_PREFIX=/home/bviggiano/miniforge3/envs/proto-tools
CONDA_PREFIX_1=/home/bviggiano/miniforge3
CONDA_PREFIX_2=/home/bviggiano/miniforge3/envs/proto-tools
CONDA_PREFIX_3=/home/bviggiano/miniforge3
CONDA_PROMPT_MODIFIER=(proto-tools) 
CONDA_PYTHON_EXE=/home/bviggiano/miniforge3/bin/python
CONDA_SHLVL=4
DISABLE_PANDERA_IMPORT_WARNING=True
HOME=/home/bviggiano
LANG=C.UTF-8
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64
LESSCLOSE=/usr/bin/lesspipe %s %s
LESSOPEN=| /usr/bin/lesspipe %s
LOGNAME=bviggiano
LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.a...
MAMBA_EXE=/home/bviggiano/miniforge3/bin/mamba
MAMBA_ROOT_PREFIX=/home/bviggiano/.local/share/mamba
MOTD_SHOWN=pam
OVSX_REGISTRY_URL=https://open-vsx.org
PATH=/home/bviggiano/.local/bin:/home/bviggiano/bin:/usr/local/cuda/bin:/home/bviggiano/miniforge3/envs/proto-tools/bin:/home/bviggiano/miniforge3/condabin:/home/bviggiano/.cursor-server/cli/servers/Stable...
PROTO_HOME=/large_storage/hielab/bviggiano/proto_cache
PWD=/home/bviggiano/main/codebases/proto-bio/proto-tools
PYDEVD_DISABLE_FILE_VALIDATION=1
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.3
RCLONE_CONFIG=/large_storage/rclone/etc/rclone.conf
RDBASE=/home/bviggiano/miniforge3/envs/proto-tools/lib/python3.14/site-packages/rdkit
SHELL=/bin/bash
SHLVL=3
SLURM_JOB_ID=2206052
TERM=xterm-256color
TERM_PROGRAM=tmux
TERM_PROGRAM_VERSION=3.2a
TMUX=/tmp/tmux-10249/default,2511031,0
TMUX_PANE=%0
USER=bviggiano
VSCODE_DEBUGPY_ADAPTER_ENDPOINTS=/home/bviggiano/.cursor-server/extensions/ms-python.debugpy-2025.18.0-linux-x64/.noConfigDebugAdapterEndpoints/endpoint-8103c33f2ddb9245.txt
VSCODE_GIT_IPC_HANDLE=/tmp/vscode-git-0f6dc9dc9f.sock
VSCODE_IPC_HOOK_CLI=/tmp/vscode-ipc-43505172-0d1a-43c2-bfdb-929c2fd6384b.sock
XDG_DATA_DIRS=/usr/local/share:/usr/share:/var/lib/snapd/desktop
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
XML_CATALOG_FILES=file:///home/bviggiano/miniforge3/etc/xml/catalog file:///etc/xml/catalog file:///home/bviggiano/miniforge3/etc/xml/catalog file:///etc/xml/catalog
_=/home/bviggiano/miniforge3/envs/proto-tools/bin/pytest
_CE_CONDA=
_CE_M=
_CONDA_EXE=/home/bviggiano/miniforge3/bin/conda
_CONDA_ROOT=/home/bviggiano/miniforge3
```

### Subprocess Environment (passed to tools)

```
CONDA_PREFIX=/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/pyhmmer_env
DETECTED_COMPUTE_PLATFORM=cuda
DETECTED_CUDA_VERSION=12
DETECTED_DRIVER_VERSION=535
HF_HOME=/large_storage/hielab/bviggiano/proto_cache/proto_model_cache/huggingface
HOME=/home/bviggiano
LANG=C.UTF-8
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/home/bviggiano/miniforge3/envs/proto-tools/lib
LOGNAME=bviggiano
PATH=/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/pyhmmer_env/bin:/home/bviggiano/.local/bin:/home/bviggiano/bin:/usr/local/cuda/bin:/home/bviggiano/miniforge3/envs/proto-tools/bin:/home/bvi...
PIP_CACHE_DIR=/large_storage/hielab/bviggiano/proto_cache/pip_cache
PIP_DEFAULT_TIMEOUT=300
PROTO_HOME=/large_storage/hielab/bviggiano/proto_cache
RECOMMENDED_JAX_SPEC=jax[cuda12]>=0.4.20,<1
RECOMMENDED_JAX_VARIANT=cuda12
RECOMMENDED_TORCH_INDEX=https://download.pytorch.org/whl/cu126
RECOMMENDED_TORCH_SPEC=torch>=2.4,<3
SHELL=/bin/bash
TORCH_CUDA_ARCH_LIST=9.0
TORCH_HOME=/large_storage/hielab/bviggiano/proto_cache/proto_model_cache/torch
USER=bviggiano
UV_CACHE_DIR=/large_storage/hielab/bviggiano/proto_cache/uv_cache
UV_HTTP_TIMEOUT=300
VIRTUAL_ENV=/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/pyhmmer_env
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Results by Category

### Causal Models (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `evo1-sample` | yes | ✅ | 206.2s | `c0b5362` | ✅ Pass |
| `evo2-sample` | yes | ✅ | 242.9s | `c0b5362` | ✅ Pass |
| `progen2-sample` | yes | ✅ | 69.9s | `c0b5362` | ✅ Pass |
| `progen3-sample` | yes | ✅ | 28.5s | `c0b5362` | ✅ Pass |

### Gene Annotation (5/5)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `blast-create-db` | no | ✅ | 32.1s | `c0b5362` | ✅ Pass |
| `crispr-tracr` | no | ✅ | 139.9s | `c0b5362` | ✅ Pass |
| `minced-crispr` | no | ✅ | 16.0s | `c0b5362` | ✅ Pass |
| `mmseqs-clustering` | no | ✅ | 19.2s | `c0b5362` | ✅ Pass |
| `pyhmmer-hmmscan` | no | ✅ | 15.6s | `c0b5362` | ✅ Pass |

### Inverse Folding (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `esm-if1-sample` | yes | ✅ | 49.9s | `c0b5362` | ✅ Pass |
| `fampnn-pack` | yes | ✅ | 158.7s | `c0b5362` | ✅ Pass |
| `ligandmpnn-sample` | yes | ✅ | 102.6s | `c0b5362` | ✅ Pass |
| `proteinmpnn-sample` | yes | ✅ | 444.4s | `c0b5362` | ✅ Pass |

### Masked Models (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `ablang-embedding` | yes | ✅ | 167.1s | `c0b5362` | ✅ Pass |
| `esm2-embedding` | yes | ✅ | 44.1s | `c0b5362` | ✅ Pass |
| `esm3-embedding` | yes | ✅ | 51.6s | `c0b5362` | ✅ Pass |
| `esmc-embedding` | yes | ✅ | 8.2s | `c0b5362` | ✅ Pass |

### Mutagenesis (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `random-nucleotide-sample` | no | - | 0.0s | `c0b5362` | ✅ Pass |
| `random-protein-sample` | no | - | 0.0s | `c0b5362` | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `orfipy-prediction` | no | ✅ | 17.4s | `c0b5362` | ✅ Pass |
| `prodigal-prediction` | no | ✅ | 14.7s | `c0b5362` | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `splice-transformer-prediction` | yes | ✅ | 33.2s | `c0b5362` | ✅ Pass |

### Sequence Alignment (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `colabfold-search` | no | ✅ | 40.5s | `c0b5362` | ✅ Pass |
| `mafft-align` | no | ✅ | 20.0s | `c0b5362` | ✅ Pass |

### Sequence Scoring (3/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphagenome-predict-intervals` | yes | ✅ | 125.3s | `c0b5362` | ❌ Fail |
| `borzoi-ensemble` | yes | ✅ | 95.2s | `c0b5362` | ✅ Pass |
| `enformer-prediction` | yes | ✅ | 38.4s | `c0b5362` | ✅ Pass |
| `segmasker-score` | no | ✅ | 32.2s | `c0b5362` | ✅ Pass |

### Structure Alignment (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `tmalign-alignment` | no | ✅ | 112.8s | `c0b5362` | ✅ Pass |
| `usalign-alignment` | no | ✅ | 37.1s | `c0b5362` | ✅ Pass |

### Structure Design (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `rfdiffusion3-design` | yes | ✅ | 153.6s | `c0b5362` | ✅ Pass |

### Structure Dynamics (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `bioemu-sample` | yes | ✅ | 116.8s | `c0b5362` | ✅ Pass |

### Structure Prediction (6/7)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphafold2-binder` | yes | ✅ | 207.0s | `c0b5362` | ❌ Fail |
| `alphafold3-prediction` | yes | ✅ | 70.8s | `c0b5362` | ✅ Pass |
| `boltz2-prediction` | yes | ✅ | 106.2s | `c0b5362` | ✅ Pass |
| `chai1-prediction` | yes | ✅ | 200.5s | `c0b5362` | ✅ Pass |
| `esmfold-prediction` | yes | ✅ | 58.0s | `c0b5362` | ✅ Pass |
| `protenix-prediction` | yes | ✅ | 454.8s | `c0b5362` | ✅ Pass |
| `viennarna-prediction` | no | ✅ | 14.7s | `c0b5362` | ✅ Pass |

### Structure Scoring (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `pdockq2` | no | - | 0.0s | `c0b5362` | ✅ Pass |
| `pyrosetta-energy` | no | ✅ | 51.0s | `c0b5362` | ✅ Pass |
| `structure-metrics` | no | ✅ | 18.5s | `c0b5362` | ✅ Pass |

### Testing (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `mock-cli-multi-gpu-tool-run` | yes | - | - | `c0b5362` | ⏭️ Skip |
| `mock-cli-tool-run` | yes | ✅ | 13.1s | `c0b5362` | ✅ Pass |
| `mock-jax-multi-gpu-tool-run` | yes | - | - | `c0b5362` | ⏭️ Skip |
| `mock-jax-tool-run` | yes | ✅ | 43.8s | `c0b5362` | ✅ Pass |
| `mock-pytorch-multi-gpu-tool-run` | yes | - | - | `c0b5362` | ⏭️ Skip |
| `mock-pytorch-tool-run` | yes | ✅ | 69.4s | `c0b5362` | ✅ Pass |

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
E   AssertionError: Tool alphagenome-predict-intervals failed: ["Command '['/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphagenome_env/bin/python', '/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_scoring/alphagenome/standalone/inference.py', '/tmp/tmpfkbkffye/input.json', '/tmp/tmpfkbkffye/output.json']' returned non-zero exit status 1.", 'Traceback (most recent call last):\n  File "/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py", line 497, in _wrapper_body\n    result = func(inputs, config, instance)\n  File "/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_scoring/alphagenome/alphagenome_predict_intervals.py", line 148, in run_alphagenome_predict_intervals\n    dispatch_result = ToolInstance.dispatch(\n        "alphagenome",\n    ...<17 lines>...\n        config=config,\n    )\n  File "/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 325, in dispatch\n    return cls._oneshot(\n           ~~~~~~~~~~~~^\n        toolkit,\n        ^^^^^^^^\n    ...<3 lines>...\n        timeout=timeout,\n        ^^^^^^^^^^^^^^^^\n    )\n    ^\n  File "/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 364, in _oneshot\n    return inst._run_oneshot(\n           ~~~~~~~~~~~~~~~~~^\n        leased_input,\n        ^^^^^^^^^^^^^\n    ...<2 lines>...\n        timeout=timeout,\n        ^^^^^^^^^^^^^^^^\n    )\n    ^\n  File "/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 1077, in _run_oneshot\n    subprocess.run(\n    ~~~~~~~~~~~~~~^\n        [python_exe, str(sp), str(input_path), str(output_path)],\n        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n    ...<5 lines>...\n        stderr=None if verbose else subprocess.PIPE,\n        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n    )\n    ^\n  File "/home/bviggiano/miniforge3/envs/proto-tools/lib/python3.14/subprocess.py", line 578, in run\n    raise CalledProcessError(retcode, process.args,\n                             output=stdout, stderr=stderr)\nsubprocess.CalledProcessError: Command \'[\'/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphagenome_env/bin/python\', \'/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_scoring/alphagenome/standalone/inference.py\', \'/tmp/tmpfkbkffye/input.json\', \'/tmp/tmpfkbkffye/output.json\']\' returned non-zero exit status 1.\n']
E   assert False
E    +  where False = AlphaGenomePredictIntervalsOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success
```

---
*Generated at 2026-04-22 00:51:04 by `pytest --env-report`*

<!-- env-report-data
[
  {
    "tool_key": "mafft-align",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mafft-align]",
    "status": "passed",
    "duration_seconds": 20.04,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mafft_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "prodigal-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[prodigal-prediction]",
    "status": "passed",
    "duration_seconds": 14.65,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/prodigal_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "rfdiffusion3-design",
    "category": "structure_design",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[rfdiffusion3-design]",
    "status": "passed",
    "duration_seconds": 153.59,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/rfdiffusion3_env",
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
    "duration_seconds": 444.35,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/proteinmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
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
    "error_message": "('/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 73, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "esm2-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm2-embedding]",
    "status": "passed",
    "duration_seconds": 44.09,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/esm2_env",
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
    "duration_seconds": 43.78,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mock_jax_tool_env",
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
    "duration_seconds": 14.69,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/viennarna_env",
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
    "duration_seconds": 57.99,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/esmfold_env",
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
    "duration_seconds": 32.22,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/segmasker_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "alphafold3-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold3-prediction]",
    "status": "passed",
    "duration_seconds": 70.8,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphafold3_env",
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
    "duration_seconds": 17.43,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/orfipy_env",
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
    "duration_seconds": 15.99,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/minced_env",
    "env_status": "success",
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
    "tool_key": "mock-cli-multi-gpu-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-cli-multi-gpu-tool-run]",
    "status": "skipped",
    "duration_seconds": 0.0,
    "uses_gpu": true,
    "env_path": null,
    "env_status": "not_found",
    "error_message": "('/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 73, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "colabfold-search",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[colabfold-search]",
    "status": "passed",
    "duration_seconds": 40.46,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/colabfold_search_env",
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
    "duration_seconds": 69.39,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mock_pytorch_tool_env",
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
    "duration_seconds": 13.08,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mock_cli_tool_env",
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
    "duration_seconds": 139.95,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/crispr_tracr_env",
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
    "duration_seconds": 106.24,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/boltz2_env",
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
    "duration_seconds": 112.78,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/tmalign_env",
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
    "duration_seconds": 454.76,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/protenix_env",
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
    "duration_seconds": 207.0,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphafold2_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:80: in test_tool_env_report\n    assert_metrics_in_spec(result)\ntests/tool_infra_tests/_metric_helpers.py:62: in assert_metrics_in_spec\n    metrics.validate_against_spec()\nproto_tools/utils/tool_io.py:310: in validate_against_spec\n    _check_list_in_spec(name, value, value_spec)\nproto_tools/utils/tool_io.py:357: in _check_list_in_spec\n    raise AssertionError(\nE   AssertionError: Metric 'pae' has type float but spec declares 'list[list[float]]'",
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "esm3-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm3-embedding]",
    "status": "passed",
    "duration_seconds": 51.62,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/evolutionaryscale_esm_env",
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
    "duration_seconds": 242.86,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/evo2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "ablang-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ablang-embedding]",
    "status": "passed",
    "duration_seconds": 167.1,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/ablang_env",
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
    "duration_seconds": 0.01,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "ligandmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ligandmpnn-sample]",
    "status": "passed",
    "duration_seconds": 102.57,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/ligandmpnn_env",
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
    "duration_seconds": 8.24,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/evolutionaryscale_esm_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "alphagenome-predict-intervals",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphagenome-predict-intervals]",
    "status": "failed",
    "duration_seconds": 125.26,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphagenome_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:79: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool alphagenome-predict-intervals failed: [\"Command '['/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphagenome_env/bin/python', '/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_scoring/alphagenome/standalone/inference.py', '/tmp/tmpfkbkffye/input.json', '/tmp/tmpfkbkffye/output.json']' returned non-zero exit status 1.\", 'Traceback (most recent call last):\\n  File \"/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py\", line 497, in _wrapper_body\\n    result = func(inputs, config, instance)\\n  File \"/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_scoring/alphagenome/alphagenome_predict_intervals.py\", line 148, in run_alphagenome_predict_intervals\\n    dispatch_result = ToolInstance.dispatch(\\n        \"alphagenome\",\\n    ...<17 lines>...\\n        config=config,\\n    )\\n  File \"/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 325, in dispatch\\n    return cls._oneshot(\\n           ~~~~~~~~~~~~^\\n        toolkit,\\n        ^^^^^^^^\\n    ...<3 lines>...\\n        timeout=timeout,\\n        ^^^^^^^^^^^^^^^^\\n    )\\n    ^\\n  File \"/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 364, in _oneshot\\n    return inst._run_oneshot(\\n           ~~~~~~~~~~~~~~~~~^\\n        leased_input,\\n        ^^^^^^^^^^^^^\\n    ...<2 lines>...\\n        timeout=timeout,\\n        ^^^^^^^^^^^^^^^^\\n    )\\n    ^\\n  File \"/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 1077, in _run_oneshot\\n    subprocess.run(\\n    ~~~~~~~~~~~~~~^\\n        [python_exe, str(sp), str(input_path), str(output_path)],\\n        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n    ...<5 lines>...\\n        stderr=None if verbose else subprocess.PIPE,\\n        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n    )\\n    ^\\n  File \"/home/bviggiano/miniforge3/envs/proto-tools/lib/python3.14/subprocess.py\", line 578, in run\\n    raise CalledProcessError(retcode, process.args,\\n                             output=stdout, stderr=stderr)\\nsubprocess.CalledProcessError: Command \\'[\\'/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphagenome_env/bin/python\\', \\'/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_scoring/alphagenome/standalone/inference.py\\', \\'/tmp/tmpfkbkffye/input.json\\', \\'/tmp/tmpfkbkffye/output.json\\']\\' returned non-zero exit status 1.\\n']\nE   assert False\nE    +  where False = AlphaGenomePredictIntervalsOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success",
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "progen2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen2-sample]",
    "status": "passed",
    "duration_seconds": 69.89,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/progen2_env",
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
    "duration_seconds": 206.19,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/evo1_env",
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
    "duration_seconds": 158.66,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/fampnn_env",
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
    "duration_seconds": 19.17,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mmseqs_env",
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
    "duration_seconds": 28.48,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/progen3_env",
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
    "duration_seconds": 18.5,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/structure_metrics_env",
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
    "duration_seconds": 37.12,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/usalign_env",
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
    "duration_seconds": 50.99,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/pyrosetta_env",
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
    "duration_seconds": 32.09,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/blast_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
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
    "error_message": "('/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 73, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "c0b536273d62",
    "git_dirty": false
  },
  {
    "tool_key": "bioemu-sample",
    "category": "structure_dynamics",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[bioemu-sample]",
    "status": "passed",
    "duration_seconds": 116.84,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/bioemu_env",
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
    "duration_seconds": 33.25,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/splice_transformer_env",
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
    "duration_seconds": 49.93,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/esm_if1_env",
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
    "duration_seconds": 95.25,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/borzoi_env",
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
    "duration_seconds": 200.46,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/chai1_env",
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
    "duration_seconds": 38.43,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/enformer_env",
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
    "duration_seconds": 15.58,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/pyhmmer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "c0b536273d62",
    "git_dirty": false
  }
]
-->