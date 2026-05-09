# DGX Spark Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-76%25-yellow) ![Passed](https://img.shields.io/badge/passed-40-brightgreen) ![Failed](https://img.shields.io/badge/failed-12-red) ![Skipped](https://img.shields.io/badge/skipped-3-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Linux Linux 6.17.0-1014-nvidia |
| **Architecture** | aarch64 |
| **Hostname** | `spark-c5f6` |
| **Python** | 3.12.13 |
| **RAM** | 121.7 GB |
| **GPU** | 1x NVIDIA GB10 |
| **CUDA** | 13.0 |
| **Conda Env** | `proto-tools` |

## Git

- **Commit**: `03b48cb13207`
- **Branch**: `main`
- **Dirty**: No

## Environment Variables

### Parent Process Environment

```
BROWSER=/home/bviggiano/.vscode-server/cli/servers/Stable-8b640eef5a6c6089c029249d48efa5c99adf7d51/server/bin/helpers/browser.sh
CLAUDE_CODE_SSE_PORT=38481
COLORTERM=truecolor
CONDA_DEFAULT_ENV=proto-tools
CONDA_EXE=/home/bviggiano/miniconda3/bin/conda
CONDA_PREFIX=/home/bviggiano/miniconda3/envs/proto-tools
CONDA_PREFIX_1=/home/bviggiano/miniconda3
CONDA_PREFIX_2=/home/bviggiano/miniconda3/envs/proto-tools
CONDA_PREFIX_3=/home/bviggiano/miniconda3
CONDA_PROMPT_MODIFIER=(proto-tools) 
CONDA_PYTHON_EXE=/home/bviggiano/miniconda3/bin/python
CONDA_SHLVL=4
DEBUGINFOD_URLS=https://debuginfod.ubuntu.com 
DISABLE_PANDERA_IMPORT_WARNING=True
HOME=/home/bviggiano
LANG=en_US.utf8
LESSCLOSE=/usr/bin/lesspipe %s %s
LESSOPEN=| /usr/bin/lesspipe %s
LOGNAME=bviggiano
LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=00:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.arj=...
OLDPWD=/home/bviggiano/.vscode-server
PATH=/home/bviggiano/.local/bin:/home/bviggiano/.local/bin:/home/bviggiano/.local/bin:/usr/local/cuda/bin:/opt/bin:/home/bviggiano/.local/bin:/home/bviggiano/miniconda3/envs/proto-tools/bin:/home/bviggiano...
PWD=/home/bviggiano/codebases/proto/proto-tools
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.3
SHELL=/bin/bash
SHLVL=3
TERM=tmux-256color
TERM_PROGRAM=tmux
TERM_PROGRAM_VERSION=3.4
TMUX=/tmp/tmux-1003/default,51725,0
TMUX_PANE=%0
USER=bviggiano
VSCODE_GIT_IPC_HANDLE=/run/user/1003/vscode-git-7100935534.sock
VSCODE_IPC_HOOK_CLI=/run/user/1003/vscode-ipc-2156346e-ea31-47bf-b523-5d1f7b5dfdd6.sock
VSCODE_PYTHON_AUTOACTIVATE_GUARD=1
XDG_DATA_DIRS=/usr/share/gnome:/usr/local/share:/usr/share:/var/lib/snapd/desktop
XDG_RUNTIME_DIR=/run/user/1003
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
_=/home/bviggiano/miniconda3/envs/proto-tools/bin/pytest
_CE_CONDA=
_CE_M=
_CONDA_EXE=/home/bviggiano/miniconda3/bin/conda
_CONDA_ROOT=/home/bviggiano/miniconda3
```

### Subprocess Environment (passed to tools)

```
CONDA_PREFIX=/home/bviggiano/.proto/proto_tool_envs/ipsae_env
DETECTED_COMPUTE_PLATFORM=cuda
DETECTED_CUDA_VERSION=13
DETECTED_DRIVER_VERSION=580
HF_HOME=/home/bviggiano/.proto/proto_model_cache/huggingface
HOME=/home/bviggiano
LANG=en_US.utf8
LD_LIBRARY_PATH=/home/bviggiano/miniconda3/envs/proto-tools/lib
LOGNAME=bviggiano
PATH=/home/bviggiano/.proto/proto_tool_envs/ipsae_env/bin:/home/bviggiano/.local/bin:/usr/local/cuda/bin:/opt/bin:/home/bviggiano/miniconda3/envs/proto-tools/bin:/home/bviggiano/miniconda3/condabin:/home/b...
PIP_CACHE_DIR=/home/bviggiano/.proto/pip_cache
PIP_DEFAULT_TIMEOUT=300
PROTO_HOME=/home/bviggiano/.proto
RECOMMENDED_JAX_SPEC=jax[cuda13]>=0.4.20,<1
RECOMMENDED_JAX_VARIANT=cuda13
RECOMMENDED_TORCH_INDEX=https://download.pytorch.org/whl/cu128
RECOMMENDED_TORCH_SPEC=torch>=2.8,<3
SHELL=/bin/bash
TORCH_CUDA_ARCH_LIST=12.0
TORCH_HOME=/home/bviggiano/.proto/proto_model_cache/torch
USER=bviggiano
UV_CACHE_DIR=/home/bviggiano/.proto/uv_cache
UV_HTTP_TIMEOUT=300
VIRTUAL_ENV=/home/bviggiano/.proto/proto_tool_envs/ipsae_env
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Results by Category

### Binder Design (0/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `germinal-design` | yes | ✅ | 48.1s | `03b48cb` | ❌ Fail |

### Causal Models (0/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `evo1-sample` | yes | ✅ | 65.7s | `03b48cb` | ❌ Fail |
| `evo2-sample` | yes | ✅ | 1.9s | `03b48cb` | ❌ Fail |
| `progen2-sample` | yes | ✅ | 2.0s | `03b48cb` | ❌ Fail |
| `progen3-sample` | yes | ✅ | 0.0s | `03b48cb` | ❌ Fail |

### Gene Annotation (3/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `crispr-tracr-rna` | no | ✅ | 46.0s | `03b48cb` | ❌ Fail |
| `minced-crispr` | no | ✅ | 3.2s | `03b48cb` | ✅ Pass |
| `promoter-calculator` | no | ✅ | 4.6s | `03b48cb` | ✅ Pass |
| `pyhmmer-hmmscan` | no | ✅ | 2.9s | `03b48cb` | ✅ Pass |

### Inverse Folding (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `esm-if1-sample` | yes | ✅ | 15.9s | `03b48cb` | ✅ Pass |
| `fampnn-pack` | yes | ✅ | 56.0s | `03b48cb` | ✅ Pass |
| `ligandmpnn-sample` | yes | ✅ | 25.6s | `03b48cb` | ✅ Pass |
| `proteinmpnn-gradient` | yes | ✅ | 28.9s | `03b48cb` | ✅ Pass |

### Masked Models (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `ablang-embedding` | yes | ✅ | 11.8s | `03b48cb` | ✅ Pass |
| `esm2-embedding` | yes | ✅ | 38.2s | `03b48cb` | ✅ Pass |
| `esm3-embedding` | yes | ✅ | 7.3s | `03b48cb` | ✅ Pass |
| `esmc-embedding` | yes | ✅ | 15.8s | `03b48cb` | ✅ Pass |

### Mutagenesis (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `random-nucleotide-sample` | no | - | 0.0s | `03b48cb` | ✅ Pass |
| `random-protein-sample` | no | - | 0.0s | `03b48cb` | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `orfipy-prediction` | no | ✅ | 10.5s | `03b48cb` | ✅ Pass |
| `prodigal-prediction` | no | ✅ | 2.7s | `03b48cb` | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `splice-transformer-prediction` | yes | ✅ | 16.9s | `03b48cb` | ✅ Pass |

### Sequence Alignment (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `blast-create-db` | no | ✅ | 80.9s | `03b48cb` | ✅ Pass |
| `colabfold-search` | no | ✅ | 15.0s | `03b48cb` | ✅ Pass |
| `mafft-align` | no | ✅ | 9.0s | `03b48cb` | ✅ Pass |
| `mmseqs2-clustering` | no | ✅ | 16.0s | `03b48cb` | ✅ Pass |

### Sequence Scoring (3/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphagenome-predict-intervals` | yes | ✅ | 80.5s | `03b48cb` | ✅ Pass |
| `borzoi-ensemble` | yes | ✅ | 47.3s | `03b48cb` | ✅ Pass |
| `enformer-prediction` | yes | ✅ | 18.7s | `03b48cb` | ✅ Pass |
| `segmasker-score` | no | ✅ | 51.9s | `03b48cb` | ❌ Fail |

### Structure Alignment (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `foldmason-msa` | no | - | 1.3s | `03b48cb` | ✅ Pass |
| `foldseek-cluster` | no | ✅ | 5.8s | `03b48cb` | ✅ Pass |
| `tmalign-alignment` | no | ✅ | 8.7s | `03b48cb` | ✅ Pass |
| `usalign-alignment` | no | ✅ | 20.4s | `03b48cb` | ✅ Pass |

### Structure Design (1/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `bindcraft-design` | yes | ✅ | 71.9s | `03b48cb` | ❌ Fail |
| `rfdiffusion3-design` | yes | ✅ | 1181.0s | `03b48cb` | ✅ Pass |

### Structure Dynamics (0/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `bioemu-sample` | yes | ✅ | 47.5s | `03b48cb` | ❌ Fail |

### Structure Prediction (4/7)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphafold2-binder` | yes | ✅ | 114.6s | `03b48cb` | ✅ Pass |
| `alphafold3-prediction` | yes | ✅ | 4.3s | `03b48cb` | ❌ Fail |
| `boltz2-prediction` | yes | ✅ | 57.5s | `03b48cb` | ❌ Fail |
| `chai1-prediction` | yes | ✅ | 2.0s | `03b48cb` | ❌ Fail |
| `esmfold-gradient` | yes | ✅ | 45.7s | `03b48cb` | ✅ Pass |
| `protenix-prediction` | yes | ✅ | 180.5s | `03b48cb` | ✅ Pass |
| `viennarna-prediction` | no | ✅ | 2.6s | `03b48cb` | ✅ Pass |

### Structure Scoring (5/5)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `dssp-secondary-structure` | no | ✅ | 6.2s | `03b48cb` | ✅ Pass |
| `ipsae-scoring` | no | ✅ | 3.1s | `03b48cb` | ✅ Pass |
| `pdockq2` | no | - | 0.0s | `03b48cb` | ✅ Pass |
| `pyrosetta-energy` | no | ✅ | 13.8s | `03b48cb` | ✅ Pass |
| `structure-metrics` | no | - | 0.6s | `03b48cb` | ✅ Pass |

### Testing (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `mock-cli-multi-gpu-tool-run` | yes | - | - | `03b48cb` | ⏭️ Skip |
| `mock-cli-tool-run` | yes | ✅ | 2.2s | `03b48cb` | ✅ Pass |
| `mock-jax-multi-gpu-tool-run` | yes | - | - | `03b48cb` | ⏭️ Skip |
| `mock-jax-tool-run` | yes | ✅ | 26.6s | `03b48cb` | ✅ Pass |
| `mock-pytorch-multi-gpu-tool-run` | yes | - | - | `03b48cb` | ⏭️ Skip |
| `mock-pytorch-tool-run` | yes | ✅ | 19.5s | `03b48cb` | ✅ Pass |

## Failure Details

### ❌ `progen3-sample`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen3-sample]`

```
tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool progen3-sample failed: ['RuntimeError: progen3: previously failed setup; last error: bash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\nERROR: progen3 setup: not supported on aarch64 (flash-attn has no aarch64 wheels)', 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 588, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/causal_models/progen3/progen3_sample.py", line 191, in run_progen3_sample\n    result = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 432, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 471, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1201, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 701, in _ensure_env\n    raise RuntimeError(f"{self.toolkit}: previously failed setup; last error: {tail or \'<no stderr>\'}")\nRuntimeError: progen3: previously failed setup; last error: bash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\nERROR: progen3 setup: not supported on aarch64 (flash-attn has no aarch64 wheels)\n']
E   assert False
E    +  where False = CausalModelSampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success
```

### ❌ `bioemu-sample`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[bioemu-sample]`

```
tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool bioemu-sample failed: ['RuntimeError: bioemu worker error for request f38ebd03: Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py", line 298, in run_bioemu_batch\n    result = _model(\n             ^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py", line 81, in __call__\n    from bioemu.steering import log_physicality\nModuleNotFoundError: No module named \'bioemu.steering\'\n\nThe above exception was the direct cause of the following exception:\n\nTraceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/_worker_bootstrap.py", line 318, in main\n    result = dispatch(input_dict)\n             ^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py", line 336, in dispatch\n    return run_bioemu_batch(input_dict)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py", line 316, in run_bioemu_batch\n    raise RuntimeError(f"bioemu: sequence {seq_idx + 1}/{len(sequences)} failed: {e}") from e\nRuntimeError: bioemu: sequence 1/1 failed: No module named \'bioemu.steering\'\n; process exit=None; last stderr: [worker] ready (script=/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py)', 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 588, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/bioemu_sample.py", line 312, in run_bioemu\n    output = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 414, in dispatch\n    return cached.run(\n           ^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 777, in run\n    return self._run_persistent(\n           ^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1159, in _run_persistent\n    result = self._worker.send(input_dict, timeout=effective_timeout)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/persistent_worker.py", line 767, in send\n    raise RuntimeError(\nRuntimeError: bioemu worker error for request f38ebd03: Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py", line 298, in run_bioemu_batch\n    result = _model(\n             ^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py", line 81, in __call__\n    from bioemu.steering import log_physicality\nModuleNotFoundError: No module named \'bioemu.steering\'\n\nThe above exception was the direct cause of the following exception:\n\nTraceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/_worker_bootstrap.py", line 318, in main\n    result = dispatch(input_dict)\n             ^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py", line 336, in dispatch\n    return run_bioemu_batch(input_dict)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py", line 316, in run_bioemu_batch\n    raise RuntimeError(f"bioemu: sequence {seq_idx + 1}/{len(sequences)} failed: {e}") from e\nRuntimeError: bioemu: sequence 1/1 failed: No module named \'bioemu.steering\'\n; process exit=None; last stderr: [worker] ready (script=/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py)\n']
E   assert False
E    +  where False = BioEmuOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success
```

### ❌ `evo2-sample`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo2-sample]`

```
tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool evo2-sample failed: ['RuntimeError: evo2: setup.sh failed (exit 1): bash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\nERROR: evo2 setup: not supported on aarch64 (transformer-engine and flash-attn have no aarch64 wheels)', 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 588, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/causal_models/evo2/evo2_sample.py", line 262, in run_evo2_sample\n    result = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 432, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 471, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1201, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 745, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 2188, in _create_env\n    raise RuntimeError(f"{self.toolkit}: setup.sh failed (exit {returncode}): {tail or \'<no stderr>\'}")\nRuntimeError: evo2: setup.sh failed (exit 1): bash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\nERROR: evo2 setup: not supported on aarch64 (transformer-engine and flash-attn have no aarch64 wheels)\n']
E   assert False
E    +  where False = Evo2SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, logits, kv_caches).success
```

### ❌ `progen2-sample`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen2-sample]`

```
tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool progen2-sample failed: ['RuntimeError: progen2: setup.sh failed (exit 1): bash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\nERROR: progen2 setup: not supported on aarch64 (torch==2.2.2 pin has no aarch64 CUDA wheel)', 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 588, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/causal_models/progen2/progen2_sample.py", line 225, in run_progen2_sample\n    result = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 432, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 471, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1201, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 745, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 2188, in _create_env\n    raise RuntimeError(f"{self.toolkit}: setup.sh failed (exit {returncode}): {tail or \'<no stderr>\'}")\nRuntimeError: progen2: setup.sh failed (exit 1): bash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\nERROR: progen2 setup: not supported on aarch64 (torch==2.2.2 pin has no aarch64 CUDA wheel)\n']
E   assert False
E    +  where False = ProGen2SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, logits).success
```

### ❌ `evo1-sample`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo1-sample]`

```
tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool evo1-sample failed: ['RuntimeError: evo1: setup.sh failed (exit 1):       on `psutil`, but doesn\'t declare it as a build dependency. If\n      `flash-attn` is a first-party package, consider adding `psutil`\n      to its `build-system.requires`. Otherwise, either add it to your\n      `pyproject.toml` under:\n      [tool.uv.extra-build-dependencies]\n      flash-attn = ["psutil"]\n      or `uv pip install psutil` into the environment and re-run with\n      `--no-build-isolation`.\n  help: `flash-attn` (v2.8.3) was included because `evo-model` (v0.5) depends\n        on `stripedhyena` (v0.2.2) which depends on `flash-attn`', 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 588, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/causal_models/evo1/evo1_sample.py", line 177, in run_evo1_sample\n    result = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 432, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 471, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1201, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 745, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 2188, in _create_env\n    raise RuntimeError(f"{self.toolkit}: setup.sh failed (exit {returncode}): {tail or \'<no stderr>\'}")\nRuntimeError: evo1: setup.sh failed (exit 1):       on `psutil`, but doesn\'t declare it as a build dependency. If\n      `flash-attn` is a first-party package, consider adding `psutil`\n      to its `build-system.requires`. Otherwise, either add it to your\n      `pyproject.toml` under:\n      [tool.uv.extra-build-dependencies]\n      flash-attn = ["psutil"]\n      or `uv pip install psutil` into the environment and re-run with\n      `--no-build-isolation`.\n  help: `flash-attn` (v2.8.3) was included because `evo-model` (v0.5) depends\n        on `stripedhyena` (v0.2.2) which depends on `flash-attn`\n']
E   assert False
E    +  where False = Evo1SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, scores).success
```

### ❌ `boltz2-prediction`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[boltz2-prediction]`

```
tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool boltz2-prediction failed: ['RuntimeError: boltz2 worker error for request 57d46e20: Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py", line 129, in __call__\n    subprocess.run(\n  File "/home/bviggiano/.proto/proto_tool_envs/boltz2_env/lib/python3.12/subprocess.py", line 571, in run\n    raise CalledProcessError(retcode, process.args,\nsubprocess.CalledProcessError: Command \'[\'/home/bviggiano/.proto/proto_tool_envs/boltz2_env/bin/boltz\', \'predict\', \'/tmp/tmp81b3s_ag/boltz2_input.yaml\', \'--out_dir=/tmp/tmp81b3s_ag/boltz2_output\', \'--recycling_steps=3\', \'--diffusion_samples=1\', \'--sampling_steps=200\', \'--step_scale=1.5\', \'--max_msa_seqs=8192\', \'--output_format=mmcif\', \'--devices=1\', \'--cache=/home/bviggiano/.proto/proto_model_cache/boltz2\', \'--num_workers=4\']\' returned non-zero exit status 1.\n\nThe above exception was the direct cause of the following exception:\n\nTraceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/_worker_bootstrap.py", line 318, in main\n    result = dispatch(input_dict)\n             ^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py", line 225, in dispatch\n    return _model(\n           ^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py", line 140, in __call__\n    raise RuntimeError(f"boltz2: failed (exit {e.returncode}): {stderr_tail}") from e\nRuntimeError: boltz2: failed (exit 1): <no stderr>\n; process exit=None; last stderr:               value[i] = reducer::combine((*acc)[i], value[i]); |             } |           } |           if (final_output) { |             set_results_to_output<output_vec_size>(value, base_offsets); |           } else { |             *acc = value; |           } |         } |       } |     } |     return value; |   } | }; | extern "C" | __launch_bounds__(512, 4) | __global__ void reduction_prod_kernel(ReduceJitOp r){ |   r.run(); | } | nvrtc: error: invalid value for --gpu-architecture (-arch)', 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 588, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/boltz2.py", line 343, in run_boltz2\n    run_boltz2_on_complex(\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/boltz2.py", line 435, in run_boltz2_on_complex\n    output_data = ToolInstance.dispatch(\n                  ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 414, in dispatch\n    return cached.run(\n           ^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 777, in run\n    return self._run_persistent(\n           ^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1159, in _run_persistent\n    result = self._worker.send(input_dict, timeout=effective_timeout)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/persistent_worker.py", line 767, in send\n    raise RuntimeError(\nRuntimeError: boltz2 worker error for request 57d46e20: Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py", line 129, in __call__\n    subprocess.run(\n  File "/home/bviggiano/.proto/proto_tool_envs/boltz2_env/lib/python3.12/subprocess.py", line 571, in run\n    raise CalledProcessError(retcode, process.args,\nsubprocess.CalledProcessError: Command \'[\'/home/bviggiano/.proto/proto_tool_envs/boltz2_env/bin/boltz\', \'predict\', \'/tmp/tmp81b3s_ag/boltz2_input.yaml\', \'--out_dir=/tmp/tmp81b3s_ag/boltz2_output\', \'--recycling_steps=3\', \'--diffusion_samples=1\', \'--sampling_steps=200\', \'--step_scale=1.5\', \'--max_msa_seqs=8192\', \'--output_format=mmcif\', \'--devices=1\', \'--cache=/home/bviggiano/.proto/proto_model_cache/boltz2\', \'--num_workers=4\']\' returned non-zero exit status 1.\n\nThe above exception was the direct cause of the following exception:\n\nTraceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/_worker_bootstrap.py", line 318, in main\n    result = dispatch(input_dict)\n             ^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py", line 225, in dispatch\n    return _model(\n           ^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py", line 140, in __call__\n    raise RuntimeError(f"boltz2: failed (exit {e.returncode}): {stderr_tail}") from e\nRuntimeError: boltz2: failed (exit 1): <no stderr>\n; process exit=None; last stderr:               value[i] = reducer::combine((*acc)[i], value[i]); |             } |           } |           if (final_output) { |             set_results_to_output<output_vec_size>(value, base_offsets); |           } else { |             *acc = value; |           } |         } |       } |     } |     return value; |   } | }; | extern "C" | __launch_bounds__(512, 4) | __global__ void reduction_prod_kernel(ReduceJitOp r){ |   r.run(); | } | nvrtc: error: invalid value for --gpu-architecture (-arch)\n']
E   assert False
E    +  where False = <[ToolExecutionError('boltz2-prediction: cannot read field \'structures\' — tool failed: RuntimeError: boltz2 worker error for request 57d46e20: Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py", line 129, in __call__\n    subprocess.run(\n  File "/home/bviggiano/.proto/proto_tool_envs/boltz2_env/lib/python3.12/subprocess.py", line 571, in run\n    raise CalledProcessError(retcode, process.args,\nsubprocess.CalledProcessError: Command \'[\'/home/bviggiano/.proto/proto_tool_envs/boltz2_env/bin/boltz\', \'predict\', \'/tmp/tmp81b3s_ag/boltz2_input.yaml\', \'--out_dir=/tmp/tmp81b3s_ag/boltz2_output\', \'--recycling_steps=3\', \'--diffusion_samples=1\', \'--sampling_steps=200\', \'--step_scale=1.5\', \'--max_msa_seqs=8192\', \'--output_format=mmcif\', \'--devices=1\', \'--cache=/home/bviggiano/.proto/proto_model_cache/boltz2\', \'--num_workers=4\']\' returned non-zero exit status 1.\n\nThe above exception was the direct cause of the following exception:\n\nTraceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/_worker_bootstra...s/proto_tools/utils/_worker_bootstrap.py", line 318, in main\n    result = dispatch(input_dict)\n             ^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py", line 225, in dispatch\n    return _model(\n           ^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py", line 140, in __call__\n    raise RuntimeError(f"boltz2: failed (exit {e.returncode}): {stderr_tail}") from e\nRuntimeError: boltz2: failed (exit 1): <no stderr>\n; process exit=None; last stderr:               value[i] = reducer::combine((*acc)[i], value[i]); |             } |           } |           if (final_output) { |             set_results_to_output<output_vec_size>(value, base_offsets); |           } else { |             *acc = value; |           } |         } |       } |     } |     return value; |   } | }; | extern "C" | __launch_bounds__(512, 4) | __global__ void reduction_prod_kernel(ReduceJitOp r){ |   r.run(); | } | nvrtc: error: invalid value for --gpu-architecture (-arch)\n') raised in repr()] Boltz2Output object at 0xe8955d91dfe0>.success
```

### ❌ `germinal-design`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[germinal-design]`

```
tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool germinal-design failed: ['RuntimeError: germinal: setup.sh failed (exit 1):       and you require torch==2.6.*, we can conclude that your requirements\n      are unsatisfiable.\n      hint: `torch` was found on https://download.pytorch.org/whl/cu128, but\n      not at the requested version (torch==2.6.*). A compatible version may\n      be available on a subsequent index (e.g., https://pypi.org/simple).\n      By default, uv will only consider versions that are published on the\n      first index that contains a given package, to avoid dependency confusion\n      attacks. If all indexes are equally trusted, use `--index-strategy\n      unsafe-best-match` to consider all versions from all indexes, regardless\n      of the order in which they were defined.', 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 588, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/binder_design/germinal/germinal_design.py", line 613, in run_germinal_design\n    output_data = ToolInstance.dispatch("germinal", input_data, instance=instance, config=config)\n                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 432, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 471, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1201, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 745, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 2188, in _create_env\n    raise RuntimeError(f"{self.toolkit}: setup.sh failed (exit {returncode}): {tail or \'<no stderr>\'}")\nRuntimeError: germinal: setup.sh failed (exit 1):       and you require torch==2.6.*, we can conclude that your requirements\n      are unsatisfiable.\n      hint: `torch` was found on https://download.pytorch.org/whl/cu128, but\n      not at the requested version (torch==2.6.*). A compatible version may\n      be available on a subsequent index (e.g., https://pypi.org/simple).\n      By default, uv will only consider versions that are published on the\n      first index that contains a given package, to avoid dependency confusion\n      attacks. If all indexes are equally trusted, use `--index-strategy\n      unsafe-best-match` to consider all versions from all indexes, regardless\n      of the order in which they were defined.\n']
E   assert False
E    +  where False = GerminalOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, designs, pipeline_stats, num_accepted, num_designs).success
```

### ❌ `segmasker-score`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[segmasker-score]`

```
tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool segmasker-score failed: ["CalledProcessError: Command '['/home/bviggiano/.proto/proto_tool_envs/segmasker_env/bin/python', '/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_scoring/segmasker/standalone/run.py', '/tmp/tmp8e7q_z2v/input.json', '/tmp/tmp8e7q_z2v/output.json']' returned non-zero exit status 1.", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 588, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_scoring/segmasker/segmasker.py", line 228, in run_segmasker\n    output_data = ToolInstance.dispatch(\n                  ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 432, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 478, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1287, in _run_oneshot\n    raise subprocess.CalledProcessError(rc, cmd, stderr=tail)\nsubprocess.CalledProcessError: Command \'[\'/home/bviggiano/.proto/proto_tool_envs/segmasker_env/bin/python\', \'/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_scoring/segmasker/standalone/run.py\', \'/tmp/tmp8e7q_z2v/input.json\', \'/tmp/tmp8e7q_z2v/output.json\']\' returned non-zero exit status 1.\n']
E   assert False
E    +  where False = SegmaskerOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, results).success
```

### ❌ `alphafold3-prediction`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold3-prediction]`

```
tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool alphafold3-prediction failed: ["MissingAssetError: alphafold3: weights not provisioned\nLicense / access: https://github.com/google-deepmind/alphafold3#obtaining-model-parameters\nProvisioning steps:\n  AlphaFold3 weights are gated by DeepMind's Terms of Use and are NOT\n  automatically downloaded. To obtain access:\n    1. Request access via DeepMind's form (link above).\n    2. After approval (2-3 business days), download the weights archive\n       from the link DeepMind emails you.\n    3. Place af3.bin (or af3.bin.zst) in the resolved directory above,\n       OR point PROTO_ALPHAFOLD3_WEIGHTS_DIR at the directory containing it.\n  See notes/storage.md for PROTO_MODEL_CACHE / PROTO_HOME rules.", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 588, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/alphafold3/alphafold3.py", line 325, in run_alphafold3\n    output_data = ToolInstance.dispatch(\n                  ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 414, in dispatch\n    return cached.run(\n           ^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 777, in run\n    return self._run_persistent(\n           ^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1028, in _run_persistent\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 745, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 2186, in _create_env\n    raise MissingAssetError(toolkit, asset_kind, tail)\nproto_tools.utils.tool_io.MissingAssetError: alphafold3: weights not provisioned\nLicense / access: https://github.com/google-deepmind/alphafold3#obtaining-model-parameters\nProvisioning steps:\n  AlphaFold3 weights are gated by DeepMind\'s Terms of Use and are NOT\n  automatically downloaded. To obtain access:\n    1. Request access via DeepMind\'s form (link above).\n    2. After approval (2-3 business days), download the weights archive\n       from the link DeepMind emails you.\n    3. Place af3.bin (or af3.bin.zst) in the resolved directory above,\n       OR point PROTO_ALPHAFOLD3_WEIGHTS_DIR at the directory containing it.\n  See notes/storage.md for PROTO_MODEL_CACHE / PROTO_HOME rules.\n']
E   assert False
E    +  where False = <[ToolExecutionError('alphafold3-prediction: cannot read field \'structures\' — tool failed: MissingAssetError: alphafold3: weights not provisioned\nLicense / access: https://github.com/google-deepmind/alphafold3#obtaining-model-parameters\nProvisioning steps:\n  AlphaFold3 weights are gated by DeepMind\'s Terms of Use and are NOT\n  automatically downloaded. To obtain access:\n    1. Request access via DeepMind\'s form (link above).\n    2. After approval (2-3 business days), download the weights archive\n       from the link DeepMind emails you.\n    3. Place af3.bin (or af3.bin.zst) in the resolved directory above,\n       OR point PROTO_ALPHAFOLD3_WEIGHTS_DIR at the directory containing it.\n  See notes/storage.md for PROTO_MODEL_CACHE / PROTO_HOME rules. | Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 588, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/alphafold3/alphafold3.py", line 325, in run_alphafold3\n    output_data = ToolInstance.dispa.../codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1028, in _run_persistent\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 745, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 2186, in _create_env\n    raise MissingAssetError(toolkit, asset_kind, tail)\nproto_tools.utils.tool_io.MissingAssetError: alphafold3: weights not provisioned\nLicense / access: https://github.com/google-deepmind/alphafold3#obtaining-model-parameters\nProvisioning steps:\n  AlphaFold3 weights are gated by DeepMind\'s Terms of Use and are NOT\n  automatically downloaded. To obtain access:\n    1. Request access via DeepMind\'s form (link above).\n    2. After approval (2-3 business days), download the weights archive\n       from the link DeepMind emails you.\n    3. Place af3.bin (or af3.bin.zst) in the resolved directory above,\n       OR point PROTO_ALPHAFOLD3_WEIGHTS_DIR at the directory containing it.\n  See notes/storage.md for PROTO_MODEL_CACHE / PROTO_HOME rules.\n') raised in repr()] AlphaFold3Output object at 0xe8955d5fc410>.success
```

### ❌ `crispr-tracr-rna`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[crispr-tracr-rna]`

```
tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool crispr-tracr-rna failed: ["RuntimeError: crispr_tracr_rna: setup.sh failed (exit 1): Cloning CRISPRidentify into CRISPRtracrRNA tools directory...\nCloning into '/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_rna_env/CRISPRtracrRNA/tools/CRISPRidentify/CRISPRidentify'...\nCloning CRISPRcasIdentifier into CRISPRtracrRNA tools directory...\nCloning into '/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_rna_env/CRISPRtracrRNA/tools/CRISPRcasIdentifier/CRISPRcasIdentifier'...\nDownloading CRISPRcasIdentifier HMM/ML models from Google Drive...\nCreating isolated conda environment (Python 3.8 + scikit-learn 0.22)...\nCRISPRidentify's pickled models require sklearn 0.22 (incompatible with 3.12).\nUsing /home/bviggiano/.proto/proto_tool_envs/crispr_tracr_rna_env/conda_deps to avoid polluting base env...\nERROR: CRISPRtracrRNA requires x86_64 bioconda packages (vmatch, etc.)\n       that are not available on Linux aarch64.", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 588, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/gene_annotation/crispr_tracr_rna/crispr_tracr_rna.py", line 451, in run_crispr_tracr_rna\n    output_data = ToolInstance.dispatch(\n                  ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 432, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 478, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1201, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 745, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 2188, in _create_env\n    raise RuntimeError(f"{self.toolkit}: setup.sh failed (exit {returncode}): {tail or \'<no stderr>\'}")\nRuntimeError: crispr_tracr_rna: setup.sh failed (exit 1): Cloning CRISPRidentify into CRISPRtracrRNA tools directory...\nCloning into \'/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_rna_env/CRISPRtracrRNA/tools/CRISPRidentify/CRISPRidentify\'...\nCloning CRISPRcasIdentifier into CRISPRtracrRNA tools directory...\nCloning into \'/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_rna_env/CRISPRtracrRNA/tools/CRISPRcasIdentifier/CRISPRcasIdentifier\'...\nDownloading CRISPRcasIdentifier HMM/ML models from Google Drive...\nCreating isolated conda environment (Python 3.8 + scikit-learn 0.22)...\nCRISPRidentify\'s pickled models require sklearn 0.22 (incompatible with 3.12).\nUsing /home/bviggiano/.proto/proto_tool_envs/crispr_tracr_rna_env/conda_deps to avoid polluting base env...\nERROR: CRISPRtracrRNA requires x86_64 bioconda packages (vmatch, etc.)\n       that are not available on Linux aarch64.\n']
E   assert False
E    +  where False = CrisprTracrRNAOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, results).success
```

### ❌ `bindcraft-design`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[bindcraft-design]`

```
tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool bindcraft-design failed: ["CalledProcessError: Command '['/home/bviggiano/.proto/proto_tool_envs/bindcraft_env/bin/python', '/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_design/bindcraft/standalone/inference.py', '/tmp/tmprggla74f/input.json', '/tmp/tmprggla74f/output.json']' returned non-zero exit status 1.", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 588, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_design/bindcraft/bindcraft_design.py", line 930, in run_bindcraft_design\n    result = ToolInstance.dispatch("bindcraft", payload, instance=instance, config=config)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 432, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 471, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1287, in _run_oneshot\n    raise subprocess.CalledProcessError(rc, cmd, stderr=tail)\nsubprocess.CalledProcessError: Command \'[\'/home/bviggiano/.proto/proto_tool_envs/bindcraft_env/bin/python\', \'/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_design/bindcraft/standalone/inference.py\', \'/tmp/tmprggla74f/input.json\', \'/tmp/tmprggla74f/output.json\']\' returned non-zero exit status 1.\n']
E   assert False
E    +  where False = BindCraftOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, designs, n_trajectories_run, n_designs_accepted).success
```

### ❌ `chai1-prediction`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[chai1-prediction]`

```
tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool chai1-prediction failed: ['RuntimeError: chai1: setup.sh failed (exit 1): bash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\nERROR: chai1 setup: not supported on aarch64 (chai_lab==0.6.1 pins torch<2.7 lacks sm_121 + ships x86_64-only TorchScript ESM2)', 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 588, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py", line 321, in run_chai1\n    run_chai1_on_complex(comp=comp, config=config, msas=inputs.msas, instance=instance)\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py", line 409, in run_chai1_on_complex\n    result = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 414, in dispatch\n    return cached.run(\n           ^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 777, in run\n    return self._run_persistent(\n           ^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1028, in _run_persistent\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 745, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 2188, in _create_env\n    raise RuntimeError(f"{self.toolkit}: setup.sh failed (exit {returncode}): {tail or \'<no stderr>\'}")\nRuntimeError: chai1: setup.sh failed (exit 1): bash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\nERROR: chai1 setup: not supported on aarch64 (chai_lab==0.6.1 pins torch<2.7 lacks sm_121 + ships x86_64-only TorchScript ESM2)\n']
E   assert False
E    +  where False = <[ToolExecutionError('chai1-prediction: cannot read field \'structures\' — tool failed: RuntimeError: chai1: setup.sh failed (exit 1): bash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\nERROR: chai1 setup: not supported on aarch64 (chai_lab==0.6.1 pins torch<2.7 lacks sm_121 + ships x86_64-only TorchScript ESM2) | Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 588, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py", line 321, in run_chai1\n    run_chai1_on_complex(comp=comp, config=config, msas=inputs.msas, instance=instance)\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py", line 409, in run_chai1_on_complex\n    result = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 414, in dispatch\n    return cached.run(\n           ^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 777, in run\n    return self._run_persistent(\n           ^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1028, in _run_persistent\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 745, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 2188, in _create_env\n    raise RuntimeError(f"{self.toolkit}: setup.sh failed (exit {returncode}): {tail or \'<no stderr>\'}")\nRuntimeError: chai1: setup.sh failed (exit 1): bash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\nERROR: chai1 setup: not supported on aarch64 (chai_lab==0.6.1 pins torch<2.7 lacks sm_121 + ships x86_64-only TorchScript ESM2)\n') raised in repr()] Chai1Output object at 0xe8955d5fe6c0>.success
```

---
*Generated at 2026-05-08 15:56:26 by `pytest --env-report`*

<!-- env-report-data
[
  {
    "tool_key": "progen3-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen3-sample]",
    "status": "failed",
    "duration_seconds": 0.04,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/progen3_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool progen3-sample failed: ['RuntimeError: progen3: previously failed setup; last error: bash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\\nERROR: progen3 setup: not supported on aarch64 (flash-attn has no aarch64 wheels)', 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 588, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/causal_models/progen3/progen3_sample.py\", line 191, in run_progen3_sample\\n    result = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 432, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 471, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1201, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 701, in _ensure_env\\n    raise RuntimeError(f\"{self.toolkit}: previously failed setup; last error: {tail or \\'<no stderr>\\'}\")\\nRuntimeError: progen3: previously failed setup; last error: bash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\\nERROR: progen3 setup: not supported on aarch64 (flash-attn has no aarch64 wheels)\\n']\nE   assert False\nE    +  where False = CausalModelSampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success",
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "bioemu-sample",
    "category": "structure_dynamics",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[bioemu-sample]",
    "status": "failed",
    "duration_seconds": 47.52,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/bioemu_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool bioemu-sample failed: ['RuntimeError: bioemu worker error for request f38ebd03: Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py\", line 298, in run_bioemu_batch\\n    result = _model(\\n             ^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py\", line 81, in __call__\\n    from bioemu.steering import log_physicality\\nModuleNotFoundError: No module named \\'bioemu.steering\\'\\n\\nThe above exception was the direct cause of the following exception:\\n\\nTraceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/_worker_bootstrap.py\", line 318, in main\\n    result = dispatch(input_dict)\\n             ^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py\", line 336, in dispatch\\n    return run_bioemu_batch(input_dict)\\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py\", line 316, in run_bioemu_batch\\n    raise RuntimeError(f\"bioemu: sequence {seq_idx + 1}/{len(sequences)} failed: {e}\") from e\\nRuntimeError: bioemu: sequence 1/1 failed: No module named \\'bioemu.steering\\'\\n; process exit=None; last stderr: [worker] ready (script=/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py)', 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 588, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/bioemu_sample.py\", line 312, in run_bioemu\\n    output = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 414, in dispatch\\n    return cached.run(\\n           ^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 777, in run\\n    return self._run_persistent(\\n           ^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1159, in _run_persistent\\n    result = self._worker.send(input_dict, timeout=effective_timeout)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/persistent_worker.py\", line 767, in send\\n    raise RuntimeError(\\nRuntimeError: bioemu worker error for request f38ebd03: Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py\", line 298, in run_bioemu_batch\\n    result = _model(\\n             ^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py\", line 81, in __call__\\n    from bioemu.steering import log_physicality\\nModuleNotFoundError: No module named \\'bioemu.steering\\'\\n\\nThe above exception was the direct cause of the following exception:\\n\\nTraceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/_worker_bootstrap.py\", line 318, in main\\n    result = dispatch(input_dict)\\n             ^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py\", line 336, in dispatch\\n    return run_bioemu_batch(input_dict)\\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py\", line 316, in run_bioemu_batch\\n    raise RuntimeError(f\"bioemu: sequence {seq_idx + 1}/{len(sequences)} failed: {e}\") from e\\nRuntimeError: bioemu: sequence 1/1 failed: No module named \\'bioemu.steering\\'\\n; process exit=None; last stderr: [worker] ready (script=/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py)\\n']\nE   assert False\nE    +  where False = BioEmuOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success",
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "promoter-calculator",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[promoter-calculator]",
    "status": "passed",
    "duration_seconds": 4.57,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/promoter_calculator_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "evo2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo2-sample]",
    "status": "failed",
    "duration_seconds": 1.93,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/evo2_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool evo2-sample failed: ['RuntimeError: evo2: setup.sh failed (exit 1): bash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\\nERROR: evo2 setup: not supported on aarch64 (transformer-engine and flash-attn have no aarch64 wheels)', 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 588, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/causal_models/evo2/evo2_sample.py\", line 262, in run_evo2_sample\\n    result = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 432, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 471, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1201, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 745, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 2188, in _create_env\\n    raise RuntimeError(f\"{self.toolkit}: setup.sh failed (exit {returncode}): {tail or \\'<no stderr>\\'}\")\\nRuntimeError: evo2: setup.sh failed (exit 1): bash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\\nERROR: evo2 setup: not supported on aarch64 (transformer-engine and flash-attn have no aarch64 wheels)\\n']\nE   assert False\nE    +  where False = Evo2SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, logits, kv_caches).success",
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "esmc-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esmc-embedding]",
    "status": "passed",
    "duration_seconds": 15.75,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/evolutionaryscale_esm_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "progen2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen2-sample]",
    "status": "failed",
    "duration_seconds": 1.96,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/progen2_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool progen2-sample failed: ['RuntimeError: progen2: setup.sh failed (exit 1): bash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\\nERROR: progen2 setup: not supported on aarch64 (torch==2.2.2 pin has no aarch64 CUDA wheel)', 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 588, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/causal_models/progen2/progen2_sample.py\", line 225, in run_progen2_sample\\n    result = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 432, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 471, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1201, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 745, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 2188, in _create_env\\n    raise RuntimeError(f\"{self.toolkit}: setup.sh failed (exit {returncode}): {tail or \\'<no stderr>\\'}\")\\nRuntimeError: progen2: setup.sh failed (exit 1): bash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\\nERROR: progen2 setup: not supported on aarch64 (torch==2.2.2 pin has no aarch64 CUDA wheel)\\n']\nE   assert False\nE    +  where False = ProGen2SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, logits).success",
    "git_commit": "03b48cb13207",
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
    "git_commit": "03b48cb13207",
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
    "error_message": "('/home/bviggiano/codebases/proto/proto-tools/tests/tool_infra_tests/test_env_report.py', 72, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "mock-jax-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-jax-tool-run]",
    "status": "passed",
    "duration_seconds": 26.64,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/mock_jax_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "fampnn-pack",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[fampnn-pack]",
    "status": "passed",
    "duration_seconds": 56.0,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/fampnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "evo1-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo1-sample]",
    "status": "failed",
    "duration_seconds": 65.69,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/evo1_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool evo1-sample failed: ['RuntimeError: evo1: setup.sh failed (exit 1):       on `psutil`, but doesn\\'t declare it as a build dependency. If\\n      `flash-attn` is a first-party package, consider adding `psutil`\\n      to its `build-system.requires`. Otherwise, either add it to your\\n      `pyproject.toml` under:\\n      [tool.uv.extra-build-dependencies]\\n      flash-attn = [\"psutil\"]\\n      or `uv pip install psutil` into the environment and re-run with\\n      `--no-build-isolation`.\\n  help: `flash-attn` (v2.8.3) was included because `evo-model` (v0.5) depends\\n        on `stripedhyena` (v0.2.2) which depends on `flash-attn`', 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 588, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/causal_models/evo1/evo1_sample.py\", line 177, in run_evo1_sample\\n    result = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 432, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 471, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1201, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 745, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 2188, in _create_env\\n    raise RuntimeError(f\"{self.toolkit}: setup.sh failed (exit {returncode}): {tail or \\'<no stderr>\\'}\")\\nRuntimeError: evo1: setup.sh failed (exit 1):       on `psutil`, but doesn\\'t declare it as a build dependency. If\\n      `flash-attn` is a first-party package, consider adding `psutil`\\n      to its `build-system.requires`. Otherwise, either add it to your\\n      `pyproject.toml` under:\\n      [tool.uv.extra-build-dependencies]\\n      flash-attn = [\"psutil\"]\\n      or `uv pip install psutil` into the environment and re-run with\\n      `--no-build-isolation`.\\n  help: `flash-attn` (v2.8.3) was included because `evo-model` (v0.5) depends\\n        on `stripedhyena` (v0.2.2) which depends on `flash-attn`\\n']\nE   assert False\nE    +  where False = Evo1SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, scores).success",
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "esm3-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm3-embedding]",
    "status": "passed",
    "duration_seconds": 7.27,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/evolutionaryscale_esm_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "esmfold-gradient",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esmfold-gradient]",
    "status": "passed",
    "duration_seconds": 45.67,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/esmfold_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "enformer-prediction",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[enformer-prediction]",
    "status": "passed",
    "duration_seconds": 18.68,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/enformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "protenix-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[protenix-prediction]",
    "status": "passed",
    "duration_seconds": 180.47,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/protenix_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "boltz2-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[boltz2-prediction]",
    "status": "failed",
    "duration_seconds": 57.46,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/boltz2_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool boltz2-prediction failed: ['RuntimeError: boltz2 worker error for request 57d46e20: Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\", line 129, in __call__\\n    subprocess.run(\\n  File \"/home/bviggiano/.proto/proto_tool_envs/boltz2_env/lib/python3.12/subprocess.py\", line 571, in run\\n    raise CalledProcessError(retcode, process.args,\\nsubprocess.CalledProcessError: Command \\'[\\'/home/bviggiano/.proto/proto_tool_envs/boltz2_env/bin/boltz\\', \\'predict\\', \\'/tmp/tmp81b3s_ag/boltz2_input.yaml\\', \\'--out_dir=/tmp/tmp81b3s_ag/boltz2_output\\', \\'--recycling_steps=3\\', \\'--diffusion_samples=1\\', \\'--sampling_steps=200\\', \\'--step_scale=1.5\\', \\'--max_msa_seqs=8192\\', \\'--output_format=mmcif\\', \\'--devices=1\\', \\'--cache=/home/bviggiano/.proto/proto_model_cache/boltz2\\', \\'--num_workers=4\\']\\' returned non-zero exit status 1.\\n\\nThe above exception was the direct cause of the following exception:\\n\\nTraceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/_worker_bootstrap.py\", line 318, in main\\n    result = dispatch(input_dict)\\n             ^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\", line 225, in dispatch\\n    return _model(\\n           ^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\", line 140, in __call__\\n    raise RuntimeError(f\"boltz2: failed (exit {e.returncode}): {stderr_tail}\") from e\\nRuntimeError: boltz2: failed (exit 1): <no stderr>\\n; process exit=None; last stderr:               value[i] = reducer::combine((*acc)[i], value[i]); |             } |           } |           if (final_output) { |             set_results_to_output<output_vec_size>(value, base_offsets); |           } else { |             *acc = value; |           } |         } |       } |     } |     return value; |   } | }; | extern \"C\" | __launch_bounds__(512, 4) | __global__ void reduction_prod_kernel(ReduceJitOp r){ |   r.run(); | } | nvrtc: error: invalid value for --gpu-architecture (-arch)', 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 588, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/boltz2.py\", line 343, in run_boltz2\\n    run_boltz2_on_complex(\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/boltz2.py\", line 435, in run_boltz2_on_complex\\n    output_data = ToolInstance.dispatch(\\n                  ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 414, in dispatch\\n    return cached.run(\\n           ^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 777, in run\\n    return self._run_persistent(\\n           ^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1159, in _run_persistent\\n    result = self._worker.send(input_dict, timeout=effective_timeout)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/persistent_worker.py\", line 767, in send\\n    raise RuntimeError(\\nRuntimeError: boltz2 worker error for request 57d46e20: Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\", line 129, in __call__\\n    subprocess.run(\\n  File \"/home/bviggiano/.proto/proto_tool_envs/boltz2_env/lib/python3.12/subprocess.py\", line 571, in run\\n    raise CalledProcessError(retcode, process.args,\\nsubprocess.CalledProcessError: Command \\'[\\'/home/bviggiano/.proto/proto_tool_envs/boltz2_env/bin/boltz\\', \\'predict\\', \\'/tmp/tmp81b3s_ag/boltz2_input.yaml\\', \\'--out_dir=/tmp/tmp81b3s_ag/boltz2_output\\', \\'--recycling_steps=3\\', \\'--diffusion_samples=1\\', \\'--sampling_steps=200\\', \\'--step_scale=1.5\\', \\'--max_msa_seqs=8192\\', \\'--output_format=mmcif\\', \\'--devices=1\\', \\'--cache=/home/bviggiano/.proto/proto_model_cache/boltz2\\', \\'--num_workers=4\\']\\' returned non-zero exit status 1.\\n\\nThe above exception was the direct cause of the following exception:\\n\\nTraceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/_worker_bootstrap.py\", line 318, in main\\n    result = dispatch(input_dict)\\n             ^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\", line 225, in dispatch\\n    return _model(\\n           ^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\", line 140, in __call__\\n    raise RuntimeError(f\"boltz2: failed (exit {e.returncode}): {stderr_tail}\") from e\\nRuntimeError: boltz2: failed (exit 1): <no stderr>\\n; process exit=None; last stderr:               value[i] = reducer::combine((*acc)[i], value[i]); |             } |           } |           if (final_output) { |             set_results_to_output<output_vec_size>(value, base_offsets); |           } else { |             *acc = value; |           } |         } |       } |     } |     return value; |   } | }; | extern \"C\" | __launch_bounds__(512, 4) | __global__ void reduction_prod_kernel(ReduceJitOp r){ |   r.run(); | } | nvrtc: error: invalid value for --gpu-architecture (-arch)\\n']\nE   assert False\nE    +  where False = <[ToolExecutionError('boltz2-prediction: cannot read field \\'structures\\' \u2014 tool failed: RuntimeError: boltz2 worker error for request 57d46e20: Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\", line 129, in __call__\\n    subprocess.run(\\n  File \"/home/bviggiano/.proto/proto_tool_envs/boltz2_env/lib/python3.12/subprocess.py\", line 571, in run\\n    raise CalledProcessError(retcode, process.args,\\nsubprocess.CalledProcessError: Command \\'[\\'/home/bviggiano/.proto/proto_tool_envs/boltz2_env/bin/boltz\\', \\'predict\\', \\'/tmp/tmp81b3s_ag/boltz2_input.yaml\\', \\'--out_dir=/tmp/tmp81b3s_ag/boltz2_output\\', \\'--recycling_steps=3\\', \\'--diffusion_samples=1\\', \\'--sampling_steps=200\\', \\'--step_scale=1.5\\', \\'--max_msa_seqs=8192\\', \\'--output_format=mmcif\\', \\'--devices=1\\', \\'--cache=/home/bviggiano/.proto/proto_model_cache/boltz2\\', \\'--num_workers=4\\']\\' returned non-zero exit status 1.\\n\\nThe above exception was the direct cause of the following exception:\\n\\nTraceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/_worker_bootstra...s/proto_tools/utils/_worker_bootstrap.py\", line 318, in main\\n    result = dispatch(input_dict)\\n             ^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\", line 225, in dispatch\\n    return _model(\\n           ^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\", line 140, in __call__\\n    raise RuntimeError(f\"boltz2: failed (exit {e.returncode}): {stderr_tail}\") from e\\nRuntimeError: boltz2: failed (exit 1): <no stderr>\\n; process exit=None; last stderr:               value[i] = reducer::combine((*acc)[i], value[i]); |             } |           } |           if (final_output) { |             set_results_to_output<output_vec_size>(value, base_offsets); |           } else { |             *acc = value; |           } |         } |       } |     } |     return value; |   } | }; | extern \"C\" | __launch_bounds__(512, 4) | __global__ void reduction_prod_kernel(ReduceJitOp r){ |   r.run(); | } | nvrtc: error: invalid value for --gpu-architecture (-arch)\\n') raised in repr()] Boltz2Output object at 0xe8955d91dfe0>.success",
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "germinal-design",
    "category": "binder_design",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[germinal-design]",
    "status": "failed",
    "duration_seconds": 48.08,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/germinal_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool germinal-design failed: ['RuntimeError: germinal: setup.sh failed (exit 1):       and you require torch==2.6.*, we can conclude that your requirements\\n      are unsatisfiable.\\n      hint: `torch` was found on https://download.pytorch.org/whl/cu128, but\\n      not at the requested version (torch==2.6.*). A compatible version may\\n      be available on a subsequent index (e.g., https://pypi.org/simple).\\n      By default, uv will only consider versions that are published on the\\n      first index that contains a given package, to avoid dependency confusion\\n      attacks. If all indexes are equally trusted, use `--index-strategy\\n      unsafe-best-match` to consider all versions from all indexes, regardless\\n      of the order in which they were defined.', 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 588, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/binder_design/germinal/germinal_design.py\", line 613, in run_germinal_design\\n    output_data = ToolInstance.dispatch(\"germinal\", input_data, instance=instance, config=config)\\n                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 432, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 471, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1201, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 745, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 2188, in _create_env\\n    raise RuntimeError(f\"{self.toolkit}: setup.sh failed (exit {returncode}): {tail or \\'<no stderr>\\'}\")\\nRuntimeError: germinal: setup.sh failed (exit 1):       and you require torch==2.6.*, we can conclude that your requirements\\n      are unsatisfiable.\\n      hint: `torch` was found on https://download.pytorch.org/whl/cu128, but\\n      not at the requested version (torch==2.6.*). A compatible version may\\n      be available on a subsequent index (e.g., https://pypi.org/simple).\\n      By default, uv will only consider versions that are published on the\\n      first index that contains a given package, to avoid dependency confusion\\n      attacks. If all indexes are equally trusted, use `--index-strategy\\n      unsafe-best-match` to consider all versions from all indexes, regardless\\n      of the order in which they were defined.\\n']\nE   assert False\nE    +  where False = GerminalOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, designs, pipeline_stats, num_accepted, num_designs).success",
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "foldseek-cluster",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[foldseek-cluster]",
    "status": "passed",
    "duration_seconds": 5.85,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/foldseek_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "tmalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[tmalign-alignment]",
    "status": "passed",
    "duration_seconds": 8.65,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/tmalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "mock-cli-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-cli-tool-run]",
    "status": "passed",
    "duration_seconds": 2.18,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/mock_cli_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "foldmason-msa",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[foldmason-msa]",
    "status": "passed",
    "duration_seconds": 1.28,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "esm-if1-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm-if1-sample]",
    "status": "passed",
    "duration_seconds": 15.92,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/esm_if1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
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
    "error_message": "('/home/bviggiano/codebases/proto/proto-tools/tests/tool_infra_tests/test_env_report.py', 72, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "orfipy-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[orfipy-prediction]",
    "status": "passed",
    "duration_seconds": 10.52,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/orfipy_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "colabfold-search",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[colabfold-search]",
    "status": "passed",
    "duration_seconds": 15.03,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/colabfold_search_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "segmasker-score",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[segmasker-score]",
    "status": "failed",
    "duration_seconds": 51.91,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/segmasker_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool segmasker-score failed: [\"CalledProcessError: Command '['/home/bviggiano/.proto/proto_tool_envs/segmasker_env/bin/python', '/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_scoring/segmasker/standalone/run.py', '/tmp/tmp8e7q_z2v/input.json', '/tmp/tmp8e7q_z2v/output.json']' returned non-zero exit status 1.\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 588, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_scoring/segmasker/segmasker.py\", line 228, in run_segmasker\\n    output_data = ToolInstance.dispatch(\\n                  ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 432, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 478, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1287, in _run_oneshot\\n    raise subprocess.CalledProcessError(rc, cmd, stderr=tail)\\nsubprocess.CalledProcessError: Command \\'[\\'/home/bviggiano/.proto/proto_tool_envs/segmasker_env/bin/python\\', \\'/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_scoring/segmasker/standalone/run.py\\', \\'/tmp/tmp8e7q_z2v/input.json\\', \\'/tmp/tmp8e7q_z2v/output.json\\']\\' returned non-zero exit status 1.\\n']\nE   assert False\nE    +  where False = SegmaskerOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, results).success",
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "viennarna-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[viennarna-prediction]",
    "status": "passed",
    "duration_seconds": 2.63,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/viennarna_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "mmseqs2-clustering",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mmseqs2-clustering]",
    "status": "passed",
    "duration_seconds": 16.04,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/mmseqs2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "mafft-align",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mafft-align]",
    "status": "passed",
    "duration_seconds": 8.97,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/mafft_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "ablang-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ablang-embedding]",
    "status": "passed",
    "duration_seconds": 11.79,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/ablang_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "ligandmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ligandmpnn-sample]",
    "status": "passed",
    "duration_seconds": 25.58,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/ligandmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "blast-create-db",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[blast-create-db]",
    "status": "passed",
    "duration_seconds": 80.88,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/blast_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
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
    "error_message": "('/home/bviggiano/codebases/proto/proto-tools/tests/tool_infra_tests/test_env_report.py', 72, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "minced-crispr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[minced-crispr]",
    "status": "passed",
    "duration_seconds": 3.18,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/minced_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "borzoi-ensemble",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[borzoi-ensemble]",
    "status": "passed",
    "duration_seconds": 47.27,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/borzoi_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "alphagenome-predict-intervals",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphagenome-predict-intervals]",
    "status": "passed",
    "duration_seconds": 80.51,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/alphagenome_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "esm2-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm2-embedding]",
    "status": "passed",
    "duration_seconds": 38.24,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/esm2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "alphafold2-binder",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold2-binder]",
    "status": "passed",
    "duration_seconds": 114.56,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/alphafold2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "pyhmmer-hmmscan",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pyhmmer-hmmscan]",
    "status": "passed",
    "duration_seconds": 2.9,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/pyhmmer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "prodigal-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[prodigal-prediction]",
    "status": "passed",
    "duration_seconds": 2.69,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/prodigal_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "pyrosetta-energy",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pyrosetta-energy]",
    "status": "passed",
    "duration_seconds": 13.79,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/pyrosetta_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "rfdiffusion3-design",
    "category": "structure_design",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[rfdiffusion3-design]",
    "status": "passed",
    "duration_seconds": 1181.04,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/rfdiffusion3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "alphafold3-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold3-prediction]",
    "status": "failed",
    "duration_seconds": 4.29,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/alphafold3_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool alphafold3-prediction failed: [\"MissingAssetError: alphafold3: weights not provisioned\\nLicense / access: https://github.com/google-deepmind/alphafold3#obtaining-model-parameters\\nProvisioning steps:\\n  AlphaFold3 weights are gated by DeepMind's Terms of Use and are NOT\\n  automatically downloaded. To obtain access:\\n    1. Request access via DeepMind's form (link above).\\n    2. After approval (2-3 business days), download the weights archive\\n       from the link DeepMind emails you.\\n    3. Place af3.bin (or af3.bin.zst) in the resolved directory above,\\n       OR point PROTO_ALPHAFOLD3_WEIGHTS_DIR at the directory containing it.\\n  See notes/storage.md for PROTO_MODEL_CACHE / PROTO_HOME rules.\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 588, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/alphafold3/alphafold3.py\", line 325, in run_alphafold3\\n    output_data = ToolInstance.dispatch(\\n                  ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 414, in dispatch\\n    return cached.run(\\n           ^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 777, in run\\n    return self._run_persistent(\\n           ^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1028, in _run_persistent\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 745, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 2186, in _create_env\\n    raise MissingAssetError(toolkit, asset_kind, tail)\\nproto_tools.utils.tool_io.MissingAssetError: alphafold3: weights not provisioned\\nLicense / access: https://github.com/google-deepmind/alphafold3#obtaining-model-parameters\\nProvisioning steps:\\n  AlphaFold3 weights are gated by DeepMind\\'s Terms of Use and are NOT\\n  automatically downloaded. To obtain access:\\n    1. Request access via DeepMind\\'s form (link above).\\n    2. After approval (2-3 business days), download the weights archive\\n       from the link DeepMind emails you.\\n    3. Place af3.bin (or af3.bin.zst) in the resolved directory above,\\n       OR point PROTO_ALPHAFOLD3_WEIGHTS_DIR at the directory containing it.\\n  See notes/storage.md for PROTO_MODEL_CACHE / PROTO_HOME rules.\\n']\nE   assert False\nE    +  where False = <[ToolExecutionError('alphafold3-prediction: cannot read field \\'structures\\' \u2014 tool failed: MissingAssetError: alphafold3: weights not provisioned\\nLicense / access: https://github.com/google-deepmind/alphafold3#obtaining-model-parameters\\nProvisioning steps:\\n  AlphaFold3 weights are gated by DeepMind\\'s Terms of Use and are NOT\\n  automatically downloaded. To obtain access:\\n    1. Request access via DeepMind\\'s form (link above).\\n    2. After approval (2-3 business days), download the weights archive\\n       from the link DeepMind emails you.\\n    3. Place af3.bin (or af3.bin.zst) in the resolved directory above,\\n       OR point PROTO_ALPHAFOLD3_WEIGHTS_DIR at the directory containing it.\\n  See notes/storage.md for PROTO_MODEL_CACHE / PROTO_HOME rules. | Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 588, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/alphafold3/alphafold3.py\", line 325, in run_alphafold3\\n    output_data = ToolInstance.dispa.../codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1028, in _run_persistent\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 745, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 2186, in _create_env\\n    raise MissingAssetError(toolkit, asset_kind, tail)\\nproto_tools.utils.tool_io.MissingAssetError: alphafold3: weights not provisioned\\nLicense / access: https://github.com/google-deepmind/alphafold3#obtaining-model-parameters\\nProvisioning steps:\\n  AlphaFold3 weights are gated by DeepMind\\'s Terms of Use and are NOT\\n  automatically downloaded. To obtain access:\\n    1. Request access via DeepMind\\'s form (link above).\\n    2. After approval (2-3 business days), download the weights archive\\n       from the link DeepMind emails you.\\n    3. Place af3.bin (or af3.bin.zst) in the resolved directory above,\\n       OR point PROTO_ALPHAFOLD3_WEIGHTS_DIR at the directory containing it.\\n  See notes/storage.md for PROTO_MODEL_CACHE / PROTO_HOME rules.\\n') raised in repr()] AlphaFold3Output object at 0xe8955d5fc410>.success",
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "crispr-tracr-rna",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[crispr-tracr-rna]",
    "status": "failed",
    "duration_seconds": 46.03,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_rna_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool crispr-tracr-rna failed: [\"RuntimeError: crispr_tracr_rna: setup.sh failed (exit 1): Cloning CRISPRidentify into CRISPRtracrRNA tools directory...\\nCloning into '/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_rna_env/CRISPRtracrRNA/tools/CRISPRidentify/CRISPRidentify'...\\nCloning CRISPRcasIdentifier into CRISPRtracrRNA tools directory...\\nCloning into '/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_rna_env/CRISPRtracrRNA/tools/CRISPRcasIdentifier/CRISPRcasIdentifier'...\\nDownloading CRISPRcasIdentifier HMM/ML models from Google Drive...\\nCreating isolated conda environment (Python 3.8 + scikit-learn 0.22)...\\nCRISPRidentify's pickled models require sklearn 0.22 (incompatible with 3.12).\\nUsing /home/bviggiano/.proto/proto_tool_envs/crispr_tracr_rna_env/conda_deps to avoid polluting base env...\\nERROR: CRISPRtracrRNA requires x86_64 bioconda packages (vmatch, etc.)\\n       that are not available on Linux aarch64.\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 588, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/gene_annotation/crispr_tracr_rna/crispr_tracr_rna.py\", line 451, in run_crispr_tracr_rna\\n    output_data = ToolInstance.dispatch(\\n                  ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 432, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 478, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1201, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 745, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 2188, in _create_env\\n    raise RuntimeError(f\"{self.toolkit}: setup.sh failed (exit {returncode}): {tail or \\'<no stderr>\\'}\")\\nRuntimeError: crispr_tracr_rna: setup.sh failed (exit 1): Cloning CRISPRidentify into CRISPRtracrRNA tools directory...\\nCloning into \\'/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_rna_env/CRISPRtracrRNA/tools/CRISPRidentify/CRISPRidentify\\'...\\nCloning CRISPRcasIdentifier into CRISPRtracrRNA tools directory...\\nCloning into \\'/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_rna_env/CRISPRtracrRNA/tools/CRISPRcasIdentifier/CRISPRcasIdentifier\\'...\\nDownloading CRISPRcasIdentifier HMM/ML models from Google Drive...\\nCreating isolated conda environment (Python 3.8 + scikit-learn 0.22)...\\nCRISPRidentify\\'s pickled models require sklearn 0.22 (incompatible with 3.12).\\nUsing /home/bviggiano/.proto/proto_tool_envs/crispr_tracr_rna_env/conda_deps to avoid polluting base env...\\nERROR: CRISPRtracrRNA requires x86_64 bioconda packages (vmatch, etc.)\\n       that are not available on Linux aarch64.\\n']\nE   assert False\nE    +  where False = CrisprTracrRNAOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, results).success",
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "bindcraft-design",
    "category": "structure_design",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[bindcraft-design]",
    "status": "failed",
    "duration_seconds": 71.9,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/bindcraft_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool bindcraft-design failed: [\"CalledProcessError: Command '['/home/bviggiano/.proto/proto_tool_envs/bindcraft_env/bin/python', '/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_design/bindcraft/standalone/inference.py', '/tmp/tmprggla74f/input.json', '/tmp/tmprggla74f/output.json']' returned non-zero exit status 1.\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 588, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_design/bindcraft/bindcraft_design.py\", line 930, in run_bindcraft_design\\n    result = ToolInstance.dispatch(\"bindcraft\", payload, instance=instance, config=config)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 432, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 471, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1287, in _run_oneshot\\n    raise subprocess.CalledProcessError(rc, cmd, stderr=tail)\\nsubprocess.CalledProcessError: Command \\'[\\'/home/bviggiano/.proto/proto_tool_envs/bindcraft_env/bin/python\\', \\'/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_design/bindcraft/standalone/inference.py\\', \\'/tmp/tmprggla74f/input.json\\', \\'/tmp/tmprggla74f/output.json\\']\\' returned non-zero exit status 1.\\n']\nE   assert False\nE    +  where False = BindCraftOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, designs, n_trajectories_run, n_designs_accepted).success",
    "git_commit": "03b48cb13207",
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
    "git_commit": "03b48cb13207",
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
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "dssp-secondary-structure",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[dssp-secondary-structure]",
    "status": "passed",
    "duration_seconds": 6.24,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/dssp_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "proteinmpnn-gradient",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[proteinmpnn-gradient]",
    "status": "passed",
    "duration_seconds": 28.88,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/proteinmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "splice-transformer-prediction",
    "category": "rna_splicing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[splice-transformer-prediction]",
    "status": "passed",
    "duration_seconds": 16.89,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/splice_transformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "mock-pytorch-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-pytorch-tool-run]",
    "status": "passed",
    "duration_seconds": 19.54,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/mock_pytorch_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "chai1-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[chai1-prediction]",
    "status": "failed",
    "duration_seconds": 2.05,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/chai1_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool chai1-prediction failed: ['RuntimeError: chai1: setup.sh failed (exit 1): bash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\\nERROR: chai1 setup: not supported on aarch64 (chai_lab==0.6.1 pins torch<2.7 lacks sm_121 + ships x86_64-only TorchScript ESM2)', 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 588, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py\", line 321, in run_chai1\\n    run_chai1_on_complex(comp=comp, config=config, msas=inputs.msas, instance=instance)\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py\", line 409, in run_chai1_on_complex\\n    result = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 414, in dispatch\\n    return cached.run(\\n           ^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 777, in run\\n    return self._run_persistent(\\n           ^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1028, in _run_persistent\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 745, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 2188, in _create_env\\n    raise RuntimeError(f\"{self.toolkit}: setup.sh failed (exit {returncode}): {tail or \\'<no stderr>\\'}\")\\nRuntimeError: chai1: setup.sh failed (exit 1): bash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\\nERROR: chai1 setup: not supported on aarch64 (chai_lab==0.6.1 pins torch<2.7 lacks sm_121 + ships x86_64-only TorchScript ESM2)\\n']\nE   assert False\nE    +  where False = <[ToolExecutionError('chai1-prediction: cannot read field \\'structures\\' \u2014 tool failed: RuntimeError: chai1: setup.sh failed (exit 1): bash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\\nERROR: chai1 setup: not supported on aarch64 (chai_lab==0.6.1 pins torch<2.7 lacks sm_121 + ships x86_64-only TorchScript ESM2) | Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 588, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py\", line 321, in run_chai1\\n    run_chai1_on_complex(comp=comp, config=config, msas=inputs.msas, instance=instance)\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py\", line 409, in run_chai1_on_complex\\n    result = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 414, in dispatch\\n    return cached.run(\\n           ^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 777, in run\\n    return self._run_persistent(\\n           ^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1028, in _run_persistent\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 745, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 2188, in _create_env\\n    raise RuntimeError(f\"{self.toolkit}: setup.sh failed (exit {returncode}): {tail or \\'<no stderr>\\'}\")\\nRuntimeError: chai1: setup.sh failed (exit 1): bash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\\nERROR: chai1 setup: not supported on aarch64 (chai_lab==0.6.1 pins torch<2.7 lacks sm_121 + ships x86_64-only TorchScript ESM2)\\n') raised in repr()] Chai1Output object at 0xe8955d5fe6c0>.success",
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "structure-metrics",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[structure-metrics]",
    "status": "passed",
    "duration_seconds": 0.58,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "usalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[usalign-alignment]",
    "status": "passed",
    "duration_seconds": 20.45,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/usalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  },
  {
    "tool_key": "ipsae-scoring",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ipsae-scoring]",
    "status": "passed",
    "duration_seconds": 3.12,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/ipsae_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "03b48cb13207",
    "git_dirty": false
  }
]
-->