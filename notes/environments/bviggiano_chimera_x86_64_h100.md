# Linux x86_64 Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-97%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-44-brightgreen) ![Failed](https://img.shields.io/badge/failed-1-red) ![Skipped](https://img.shields.io/badge/skipped-3-lightgrey)

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

- **Commit**: `97ce3e1c5f47`
- **Branch**: `chore/af3-notebook-and-env-report`
- **Dirty**: Yes

## Environment Variables

### Parent Process Environment

```
BLASTDB=/common_datasets/external/databases/blast
BROWSER=/home/bviggiano/.vscode-server/cli/servers/Stable-10c8e557c8b9f9ed0a87f61f1c9a44bde731c409/server/bin/helpers/browser.sh
CLAUDECODE=1
CLAUDE_CODE_ENTRYPOINT=cli
CLAUDE_CODE_EXECPATH=/home/bviggiano/.local/share/claude/versions/2.1.119
CLAUDE_CODE_SSE_PORT=20642
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
OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta
PATH=/home/bviggiano/miniforge3/envs/proto-tools/bin:/home/bviggiano/miniforge3/condabin:/home/bviggiano/.vscode-server/data/User/globalStorage/github.copilot-chat/debugCommand:/home/bviggiano/.vscode-serv...
PROTO_ALPHAFOLD3_WEIGHTS_DIR=/large_storage/hielab/brk/models/af3_weights
PROTO_HOME=/large_storage/hielab/bviggiano/proto_cache
PWD=/home/bviggiano/main/codebases/proto-bio/proto-tools
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.3
RCLONE_CONFIG=/large_storage/rclone/etc/rclone.conf
RDBASE=/home/bviggiano/miniforge3/envs/proto-tools/lib/python3.14/site-packages/rdkit
SHELL=/bin/bash
SHLVL=3
SLURM_JOB_ID=2222565
TERM=xterm-256color
TERM_PROGRAM=vscode
TERM_PROGRAM_VERSION=1.117.0
USER=bviggiano
VSCODE_GIT_IPC_HANDLE=/tmp/vscode-git-d439488ff3.sock
VSCODE_IPC_HOOK_CLI=/tmp/vscode-ipc-0fc5976c-1f9b-4d27-8acc-ec620d5a687a.sock
VSCODE_PYTHON_AUTOACTIVATE_GUARD=1
XDG_DATA_DIRS=/usr/local/share:/usr/share:/var/lib/snapd/desktop
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
XML_CATALOG_FILES=file:///home/bviggiano/miniforge3/etc/xml/catalog file:///etc/xml/catalog
_=/home/bviggiano/miniforge3/envs/proto-tools/bin/python
_CE_CONDA=
_CE_M=
_CONDA_EXE=/home/bviggiano/miniforge3/bin/conda
_CONDA_ROOT=/home/bviggiano/miniforge3
```

### Subprocess Environment (passed to tools)

```
CONDA_PREFIX=/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphafold3_env
CUDA_ROOT=/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphafold3_env/cuda_env
DETECTED_COMPUTE_PLATFORM=cuda
DETECTED_CUDA_VERSION=12
DETECTED_DRIVER_VERSION=535
HF_HOME=/large_storage/hielab/bviggiano/proto_cache/proto_model_cache/huggingface
HOME=/home/bviggiano
LANG=C.UTF-8
LD_LIBRARY_PATH=/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphafold3_env/cuda_env/lib:/usr/local/cuda/lib64:/home/bviggiano/miniforge3/envs/proto-tools/lib
LOGNAME=bviggiano
PATH=/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphafold3_env/bin:/usr/local/cuda/bin:/home/bviggiano/miniforge3/envs/proto-tools/bin:/home/bviggiano/miniforge3/condabin:/home/bviggiano/....
PIP_CACHE_DIR=/large_storage/hielab/bviggiano/proto_cache/pip_cache
PIP_DEFAULT_TIMEOUT=300
PROTO_ALPHAFOLD3_WEIGHTS_DIR=/large_storage/hielab/brk/models/af3_weights
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
VIRTUAL_ENV=/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphafold3_env
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Results by Category

### Causal Models (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `evo1-sample` | yes | ✅ | 292.4s | `ba634cb` | ✅ Pass |
| `evo2-sample` | yes | ✅ | 537.0s | `ba634cb` | ✅ Pass |
| `progen2-sample` | yes | ✅ | 99.7s | `ba634cb` | ✅ Pass |
| `progen3-sample` | yes | ✅ | 32.6s | `ba634cb` | ✅ Pass |

### Gene Annotation (5/5)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `blast-create-db` | no | ✅ | 35.4s | `ba634cb` | ✅ Pass |
| `crispr-tracr` | no | ✅ | 421.4s | `ba634cb` | ✅ Pass |
| `minced-crispr` | no | ✅ | 39.3s | `ba634cb` | ✅ Pass |
| `mmseqs-clustering` | no | ✅ | 34.0s | `ba634cb` | ✅ Pass |
| `pyhmmer-hmmscan` | no | ✅ | 19.9s | `ba634cb` | ✅ Pass |

### Inverse Folding (3/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `esm-if1-sample` | yes | ✅ | 170.5s | `ba634cb` | ✅ Pass |
| `fampnn-pack` | yes | ✅ | 207.2s | `ba634cb` | ✅ Pass |
| `ligandmpnn-sample` | yes | ✅ | 157.7s | `ba634cb` | ✅ Pass |
| `proteinmpnn-sample` | yes | ✅ | 66.6s | `ba634cb` | ❌ Fail |

### Masked Models (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `ablang-embedding` | yes | ✅ | 154.0s | `ba634cb` | ✅ Pass |
| `esm2-embedding` | yes | ✅ | 77.7s | `ba634cb` | ✅ Pass |
| `esm3-embedding` | yes | ✅ | 10.2s | `ba634cb` | ✅ Pass |
| `esmc-embedding` | yes | ✅ | 112.6s | `ba634cb` | ✅ Pass |

### Mutagenesis (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `random-nucleotide-sample` | no | - | 0.0s | `ba634cb` | ✅ Pass |
| `random-protein-sample` | no | - | 0.0s | `ba634cb` | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `orfipy-prediction` | no | ✅ | 52.6s | `ba634cb` | ✅ Pass |
| `prodigal-prediction` | no | ✅ | 18.4s | `ba634cb` | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `splice-transformer-prediction` | yes | ✅ | 50.3s | `ba634cb` | ✅ Pass |

### Sequence Alignment (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `colabfold-search` | no | ✅ | 130.1s | `ba634cb` | ✅ Pass |
| `mafft-align` | no | ✅ | 25.2s | `ba634cb` | ✅ Pass |

### Sequence Scoring (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphagenome-predict-intervals` | yes | ✅ | 428.8s | `ba634cb` | ✅ Pass |
| `borzoi-ensemble` | yes | ✅ | 99.5s | `ba634cb` | ✅ Pass |
| `enformer-prediction` | yes | ✅ | 75.7s | `ba634cb` | ✅ Pass |
| `segmasker-score` | no | ✅ | 30.8s | `ba634cb` | ✅ Pass |

### Structure Alignment (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `tmalign-alignment` | no | ✅ | 49.1s | `ba634cb` | ✅ Pass |
| `usalign-alignment` | no | ✅ | 41.9s | `ba634cb` | ✅ Pass |

### Structure Design (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `rfdiffusion3-design` | yes | ✅ | 205.8s | `ba634cb` | ✅ Pass |

### Structure Dynamics (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `bioemu-sample` | yes | ✅ | 139.2s | `ba634cb` | ✅ Pass |

### Structure Prediction (7/7)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphafold2-binder` | yes | ✅ | 223.4s | `ba634cb` | ✅ Pass |
| `alphafold3-prediction` | yes | ✅ | 473.7s | `97ce3e1` ✱ | ✅ Pass |
| `boltz2-prediction` | yes | ✅ | 106.8s | `ba634cb` | ✅ Pass |
| `chai1-prediction` | yes | ✅ | 294.7s | `ba634cb` | ✅ Pass |
| `esmfold-prediction` | yes | ✅ | 68.2s | `ba634cb` | ✅ Pass |
| `protenix-prediction` | yes | ✅ | 423.8s | `ba634cb` | ✅ Pass |
| `viennarna-prediction` | no | ✅ | 92.5s | `ba634cb` | ✅ Pass |

### Structure Scoring (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `pdockq2` | no | - | 0.0s | `ba634cb` | ✅ Pass |
| `pyrosetta-energy` | no | ✅ | 60.4s | `ba634cb` | ✅ Pass |
| `structure-metrics` | no | ✅ | 23.4s | `ba634cb` | ✅ Pass |

### Testing (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `mock-cli-multi-gpu-tool-run` | yes | - | - | `ba634cb` | ⏭️ Skip |
| `mock-cli-tool-run` | yes | ✅ | 16.6s | `ba634cb` | ✅ Pass |
| `mock-jax-multi-gpu-tool-run` | yes | - | - | `ba634cb` | ⏭️ Skip |
| `mock-jax-tool-run` | yes | ✅ | 58.6s | `ba634cb` | ✅ Pass |
| `mock-pytorch-multi-gpu-tool-run` | yes | - | - | `ba634cb` | ⏭️ Skip |
| `mock-pytorch-tool-run` | yes | ✅ | 114.2s | `ba634cb` | ✅ Pass |

## Failure Details

### ❌ `proteinmpnn-sample`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[proteinmpnn-sample]`

```
tests/tool_infra_tests/test_env_report.py:79: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool proteinmpnn-sample failed: ["Command '['/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/proteinmpnn_env/bin/python', '/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/inverse_folding/proteinmpnn/standalone/inference.py', '/tmp/tmphama5kuy/input.json', '/tmp/tmphama5kuy/output.json']' returned non-zero exit status 1.", 'Traceback (most recent call last):\n  File "/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py", line 554, in _wrapper_body\n    result = func(inputs, config, instance)\n  File "/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/inverse_folding/proteinmpnn/proteinmpnn_sample.py", line 152, in run_proteinmpnn_sample\n    result = ToolInstance.dispatch(\n        "proteinmpnn",\n    ...<2 lines>...\n        config=config,\n    )\n  File "/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 422, in dispatch\n    return cls._oneshot(\n           ~~~~~~~~~~~~^\n        toolkit,\n        ^^^^^^^^\n    ...<3 lines>...\n        timeout=timeout,\n        ^^^^^^^^^^^^^^^^\n    )\n    ^\n  File "/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 461, in _oneshot\n    return inst._run_oneshot(\n           ~~~~~~~~~~~~~~~~~^\n        leased_input,\n        ^^^^^^^^^^^^^\n    ...<2 lines>...\n        timeout=timeout,\n        ^^^^^^^^^^^^^^^^\n    )\n    ^\n  File "/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 1215, in _run_oneshot\n    subprocess.run(\n    ~~~~~~~~~~~~~~^\n        [python_exe, str(sp), str(input_path), str(output_path)],\n        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n    ...<5 lines>...\n        stderr=None if verbose else subprocess.PIPE,\n        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n    )\n    ^\n  File "/home/bviggiano/miniforge3/envs/proto-tools/lib/python3.14/subprocess.py", line 578, in run\n    raise CalledProcessError(retcode, process.args,\n                             output=stdout, stderr=stderr)\nsubprocess.CalledProcessError: Command \'[\'/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/proteinmpnn_env/bin/python\', \'/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/inverse_folding/proteinmpnn/standalone/inference.py\', \'/tmp/tmphama5kuy/input.json\', \'/tmp/tmphama5kuy/output.json\']\' returned non-zero exit status 1.\n']
E   assert False
E    +  where False = <[ToolExecutionError('Attempt to access field of tool output after failure: subprocess.CalledProcessError: Command \'[\'/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/proteinmpnn_env/bin/python\', \'/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/inverse_folding/proteinmpnn/standalone/inference.py\', \'/tmp/tmphama5kuy/input.json\', \'/tmp/tmphama5kuy/output.json\']\' returned non-zero exit status 1.\n\nError Messages:\nCommand \'[\'/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/proteinmpnn_env/bin/python\', \'/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/inverse_folding/proteinmpnn/standalone/inference.py\', \'/tmp/tmphama5kuy/input.json\', \'/tmp/tmphama5kuy/output.json\']\' returned non-zero exit status 1.\nTraceback (most recent call last):\n  File "/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py", line 554, in _wrapper_body\n    result = func(inputs, config, instance)\n  File "/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/inverse_folding/proteinmpnn/proteinmpnn_sample.py", line 152, in run_prote...   ...<2 lines>...\n        timeout=timeout,\n        ^^^^^^^^^^^^^^^^\n    )\n    ^\n  File "/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 1215, in _run_oneshot\n    subprocess.run(\n    ~~~~~~~~~~~~~~^\n        [python_exe, str(sp), str(input_path), str(output_path)],\n        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n    ...<5 lines>...\n        stderr=None if verbose else subprocess.PIPE,\n        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n    )\n    ^\n  File "/home/bviggiano/miniforge3/envs/proto-tools/lib/python3.14/subprocess.py", line 578, in run\n    raise CalledProcessError(retcode, process.args,\n                             output=stdout, stderr=stderr)\nsubprocess.CalledProcessError: Command \'[\'/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/proteinmpnn_env/bin/python\', \'/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/inverse_folding/proteinmpnn/standalone/inference.py\', \'/tmp/tmphama5kuy/input.json\', \'/tmp/tmphama5kuy/output.json\']\' returned non-zero exit status 1.\n') raised in repr()] InverseFoldingOutput object at 0x7fe10a17aa50>.success
```

---
*Generated at 2026-04-25 18:30:09 by `pytest --env-report`*

<!-- env-report-data
[
  {
    "tool_key": "mock-jax-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-jax-tool-run]",
    "status": "passed",
    "duration_seconds": 58.59,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mock_jax_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "segmasker-score",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[segmasker-score]",
    "status": "passed",
    "duration_seconds": 30.77,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/segmasker_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "fampnn-pack",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[fampnn-pack]",
    "status": "passed",
    "duration_seconds": 207.25,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/fampnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "evo1-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo1-sample]",
    "status": "passed",
    "duration_seconds": 292.42,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/evo1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "esm-if1-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm-if1-sample]",
    "status": "passed",
    "duration_seconds": 170.54,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/esm_if1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "evo2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo2-sample]",
    "status": "passed",
    "duration_seconds": 537.04,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/evo2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "viennarna-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[viennarna-prediction]",
    "status": "passed",
    "duration_seconds": 92.52,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/viennarna_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "alphagenome-predict-intervals",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphagenome-predict-intervals]",
    "status": "passed",
    "duration_seconds": 428.77,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphagenome_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "protenix-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[protenix-prediction]",
    "status": "passed",
    "duration_seconds": 423.84,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/protenix_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "enformer-prediction",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[enformer-prediction]",
    "status": "passed",
    "duration_seconds": 75.68,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/enformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "progen3-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen3-sample]",
    "status": "passed",
    "duration_seconds": 32.62,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/progen3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "splice-transformer-prediction",
    "category": "rna_splicing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[splice-transformer-prediction]",
    "status": "passed",
    "duration_seconds": 50.29,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/splice_transformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "alphafold3-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold3-prediction]",
    "status": "passed",
    "duration_seconds": 473.73,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphafold3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "97ce3e1c5f47",
    "git_dirty": true
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
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "progen2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen2-sample]",
    "status": "passed",
    "duration_seconds": 99.68,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/progen2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "esmc-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esmc-embedding]",
    "status": "passed",
    "duration_seconds": 112.58,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/evolutionaryscale_esm_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
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
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "orfipy-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[orfipy-prediction]",
    "status": "passed",
    "duration_seconds": 52.58,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/orfipy_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "minced-crispr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[minced-crispr]",
    "status": "passed",
    "duration_seconds": 39.29,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/minced_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "tmalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[tmalign-alignment]",
    "status": "passed",
    "duration_seconds": 49.06,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/tmalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "esm3-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm3-embedding]",
    "status": "passed",
    "duration_seconds": 10.23,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/evolutionaryscale_esm_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "crispr-tracr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[crispr-tracr]",
    "status": "passed",
    "duration_seconds": 421.36,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/crispr_tracr_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "boltz2-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[boltz2-prediction]",
    "status": "passed",
    "duration_seconds": 106.84,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/boltz2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "colabfold-search",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[colabfold-search]",
    "status": "passed",
    "duration_seconds": 130.05,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/colabfold_search_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "mock-pytorch-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-pytorch-tool-run]",
    "status": "passed",
    "duration_seconds": 114.21,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mock_pytorch_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "mock-cli-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-cli-tool-run]",
    "status": "passed",
    "duration_seconds": 16.62,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mock_cli_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "random-nucleotide-sample",
    "category": "mutagenesis",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[random-nucleotide-sample]",
    "status": "passed",
    "duration_seconds": 0.01,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "alphafold2-binder",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold2-binder]",
    "status": "passed",
    "duration_seconds": 223.44,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/alphafold2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "ablang-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ablang-embedding]",
    "status": "passed",
    "duration_seconds": 153.99,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/ablang_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "mmseqs-clustering",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mmseqs-clustering]",
    "status": "passed",
    "duration_seconds": 33.95,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mmseqs_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
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
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "rfdiffusion3-design",
    "category": "structure_design",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[rfdiffusion3-design]",
    "status": "passed",
    "duration_seconds": 205.78,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/rfdiffusion3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "ligandmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ligandmpnn-sample]",
    "status": "passed",
    "duration_seconds": 157.68,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/ligandmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "prodigal-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[prodigal-prediction]",
    "status": "passed",
    "duration_seconds": 18.42,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/prodigal_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "mafft-align",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mafft-align]",
    "status": "passed",
    "duration_seconds": 25.19,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/mafft_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "pyhmmer-hmmscan",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pyhmmer-hmmscan]",
    "status": "passed",
    "duration_seconds": 19.89,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/pyhmmer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "bioemu-sample",
    "category": "structure_dynamics",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[bioemu-sample]",
    "status": "passed",
    "duration_seconds": 139.23,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/bioemu_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "esmfold-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esmfold-prediction]",
    "status": "passed",
    "duration_seconds": 68.17,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/esmfold_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "pyrosetta-energy",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pyrosetta-energy]",
    "status": "passed",
    "duration_seconds": 60.39,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/pyrosetta_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
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
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "structure-metrics",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[structure-metrics]",
    "status": "passed",
    "duration_seconds": 23.44,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/structure_metrics_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "usalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[usalign-alignment]",
    "status": "passed",
    "duration_seconds": 41.86,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/usalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "borzoi-ensemble",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[borzoi-ensemble]",
    "status": "passed",
    "duration_seconds": 99.52,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/borzoi_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "blast-create-db",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[blast-create-db]",
    "status": "passed",
    "duration_seconds": 35.43,
    "uses_gpu": false,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/blast_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "chai1-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[chai1-prediction]",
    "status": "passed",
    "duration_seconds": 294.71,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/chai1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "proteinmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[proteinmpnn-sample]",
    "status": "failed",
    "duration_seconds": 66.63,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/proteinmpnn_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:79: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool proteinmpnn-sample failed: [\"Command '['/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/proteinmpnn_env/bin/python', '/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/inverse_folding/proteinmpnn/standalone/inference.py', '/tmp/tmphama5kuy/input.json', '/tmp/tmphama5kuy/output.json']' returned non-zero exit status 1.\", 'Traceback (most recent call last):\\n  File \"/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py\", line 554, in _wrapper_body\\n    result = func(inputs, config, instance)\\n  File \"/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/inverse_folding/proteinmpnn/proteinmpnn_sample.py\", line 152, in run_proteinmpnn_sample\\n    result = ToolInstance.dispatch(\\n        \"proteinmpnn\",\\n    ...<2 lines>...\\n        config=config,\\n    )\\n  File \"/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 422, in dispatch\\n    return cls._oneshot(\\n           ~~~~~~~~~~~~^\\n        toolkit,\\n        ^^^^^^^^\\n    ...<3 lines>...\\n        timeout=timeout,\\n        ^^^^^^^^^^^^^^^^\\n    )\\n    ^\\n  File \"/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 461, in _oneshot\\n    return inst._run_oneshot(\\n           ~~~~~~~~~~~~~~~~~^\\n        leased_input,\\n        ^^^^^^^^^^^^^\\n    ...<2 lines>...\\n        timeout=timeout,\\n        ^^^^^^^^^^^^^^^^\\n    )\\n    ^\\n  File \"/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 1215, in _run_oneshot\\n    subprocess.run(\\n    ~~~~~~~~~~~~~~^\\n        [python_exe, str(sp), str(input_path), str(output_path)],\\n        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n    ...<5 lines>...\\n        stderr=None if verbose else subprocess.PIPE,\\n        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n    )\\n    ^\\n  File \"/home/bviggiano/miniforge3/envs/proto-tools/lib/python3.14/subprocess.py\", line 578, in run\\n    raise CalledProcessError(retcode, process.args,\\n                             output=stdout, stderr=stderr)\\nsubprocess.CalledProcessError: Command \\'[\\'/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/proteinmpnn_env/bin/python\\', \\'/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/inverse_folding/proteinmpnn/standalone/inference.py\\', \\'/tmp/tmphama5kuy/input.json\\', \\'/tmp/tmphama5kuy/output.json\\']\\' returned non-zero exit status 1.\\n']\nE   assert False\nE    +  where False = <[ToolExecutionError('Attempt to access field of tool output after failure: subprocess.CalledProcessError: Command \\'[\\'/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/proteinmpnn_env/bin/python\\', \\'/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/inverse_folding/proteinmpnn/standalone/inference.py\\', \\'/tmp/tmphama5kuy/input.json\\', \\'/tmp/tmphama5kuy/output.json\\']\\' returned non-zero exit status 1.\\n\\nError Messages:\\nCommand \\'[\\'/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/proteinmpnn_env/bin/python\\', \\'/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/inverse_folding/proteinmpnn/standalone/inference.py\\', \\'/tmp/tmphama5kuy/input.json\\', \\'/tmp/tmphama5kuy/output.json\\']\\' returned non-zero exit status 1.\\nTraceback (most recent call last):\\n  File \"/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py\", line 554, in _wrapper_body\\n    result = func(inputs, config, instance)\\n  File \"/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/inverse_folding/proteinmpnn/proteinmpnn_sample.py\", line 152, in run_prote...   ...<2 lines>...\\n        timeout=timeout,\\n        ^^^^^^^^^^^^^^^^\\n    )\\n    ^\\n  File \"/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 1215, in _run_oneshot\\n    subprocess.run(\\n    ~~~~~~~~~~~~~~^\\n        [python_exe, str(sp), str(input_path), str(output_path)],\\n        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n    ...<5 lines>...\\n        stderr=None if verbose else subprocess.PIPE,\\n        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n    )\\n    ^\\n  File \"/home/bviggiano/miniforge3/envs/proto-tools/lib/python3.14/subprocess.py\", line 578, in run\\n    raise CalledProcessError(retcode, process.args,\\n                             output=stdout, stderr=stderr)\\nsubprocess.CalledProcessError: Command \\'[\\'/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/proteinmpnn_env/bin/python\\', \\'/large_storage/hielab/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/inverse_folding/proteinmpnn/standalone/inference.py\\', \\'/tmp/tmphama5kuy/input.json\\', \\'/tmp/tmphama5kuy/output.json\\']\\' returned non-zero exit status 1.\\n') raised in repr()] InverseFoldingOutput object at 0x7fe10a17aa50>.success",
    "git_commit": "ba634cb14378",
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
    "git_commit": "ba634cb14378",
    "git_dirty": false
  },
  {
    "tool_key": "esm2-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm2-embedding]",
    "status": "passed",
    "duration_seconds": 77.74,
    "uses_gpu": true,
    "env_path": "/large_storage/hielab/bviggiano/proto_cache/proto_tool_envs/esm2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "ba634cb14378",
    "git_dirty": false
  }
]
-->