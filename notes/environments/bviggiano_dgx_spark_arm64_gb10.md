# DGX Spark Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-78%25-yellow) ![Passed](https://img.shields.io/badge/passed-37-brightgreen) ![Failed](https://img.shields.io/badge/failed-10-red) ![Skipped](https://img.shields.io/badge/skipped-3-lightgrey)

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

- **Commit**: `111bab7e0784`
- **Branch**: `fix/ablang-prefetch-and-validate`
- **Dirty**: Yes

## Environment Variables

### Parent Process Environment

```
CLAUDECODE=1
CLAUDE_CODE_ENTRYPOINT=cli
CLAUDE_CODE_EXECPATH=/home/bviggiano/.local/share/claude/versions/2.1.119
CONDA_DEFAULT_ENV=proto-tools
CONDA_EXE=/home/bviggiano/miniconda3/bin/conda
CONDA_PREFIX=/home/bviggiano/miniconda3/envs/proto-tools
CONDA_PREFIX_1=/home/bviggiano/miniconda3
CONDA_PROMPT_MODIFIER=(proto-tools) 
CONDA_PYTHON_EXE=/home/bviggiano/miniconda3/bin/python
CONDA_SHLVL=2
COREPACK_ENABLE_AUTO_PIN=0
DEBUGINFOD_URLS=https://debuginfod.ubuntu.com 
DISABLE_PANDERA_IMPORT_WARNING=True
GIT_EDITOR=true
HOME=/home/bviggiano
LANG=en_US.utf8
LESSCLOSE=/usr/bin/lesspipe %s %s
LESSOPEN=| /usr/bin/lesspipe %s
LOGNAME=bviggiano
LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=00:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.arj=...
NoDefaultCurrentDirectoryInExePath=1
OLDPWD=/home/bviggiano/codebases/proto/proto-tools
OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta
PATH=/home/bviggiano/.local/bin:/home/bviggiano/.local/bin:/home/bviggiano/miniconda3/envs/proto-tools/bin:/home/bviggiano/miniconda3/condabin:/home/bviggiano/.local/bin:/usr/local/cuda/bin:/opt/bin:/usr/l...
PWD=/home/bviggiano/codebases/proto/proto-tools
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.3
SHELL=/bin/bash
SHLVL=2
TERM=xterm-256color
USER=bviggiano
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
CONDA_PREFIX=/home/bviggiano/.proto/proto_tool_envs/ablang_env
DETECTED_COMPUTE_PLATFORM=cuda
DETECTED_CUDA_VERSION=13
DETECTED_DRIVER_VERSION=580
HF_HOME=/home/bviggiano/.proto/proto_model_cache/huggingface
HOME=/home/bviggiano
LANG=en_US.utf8
LD_LIBRARY_PATH=/home/bviggiano/miniconda3/envs/proto-tools/lib
LOGNAME=bviggiano
PATH=/home/bviggiano/.proto/proto_tool_envs/ablang_env/bin:/usr/local/cuda/bin:/home/bviggiano/.local/bin:/home/bviggiano/miniconda3/envs/proto-tools/bin:/home/bviggiano/miniconda3/condabin:/opt/bin:/usr/l...
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
VIRTUAL_ENV=/home/bviggiano/.proto/proto_tool_envs/ablang_env
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Results by Category

### Causal Models (0/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `evo1-sample` | yes | ✅ | 50.0s | `72cc6d6` | ❌ Fail |
| `evo2-sample` | yes | ✅ | 2.1s | `72cc6d6` | ❌ Fail |
| `progen2-sample` | yes | ✅ | 2.1s | `72cc6d6` | ❌ Fail |
| `progen3-sample` | yes | ✅ | 3.6s | `72cc6d6` | ❌ Fail |

### Gene Annotation (4/5)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `blast-create-db` | no | ✅ | 86.0s | `72cc6d6` | ✅ Pass |
| `crispr-tracr` | no | ✅ | 24.5s | `72cc6d6` | ❌ Fail |
| `minced-crispr` | no | ✅ | 3.1s | `72cc6d6` | ✅ Pass |
| `mmseqs-clustering` | no | ✅ | 4.4s | `72cc6d6` | ✅ Pass |
| `pyhmmer-hmmscan` | no | ✅ | 2.9s | `72cc6d6` | ✅ Pass |

### Inverse Folding (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `esm-if1-sample` | yes | ✅ | 16.5s | `72cc6d6` | ✅ Pass |
| `fampnn-pack` | yes | ✅ | 57.9s | `72cc6d6` | ✅ Pass |
| `ligandmpnn-sample` | yes | ✅ | 51.1s | `72cc6d6` | ✅ Pass |
| `proteinmpnn-sample` | yes | ✅ | 29.7s | `72cc6d6` | ✅ Pass |

### Masked Models (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `ablang-embedding` | yes | ✅ | 272.4s | `111bab7` ✱ | ✅ Pass |
| `esm2-embedding` | yes | ✅ | 26.4s | `72cc6d6` | ✅ Pass |
| `esm3-embedding` | yes | ✅ | 6.5s | `72cc6d6` | ✅ Pass |
| `esmc-embedding` | yes | ✅ | 29.7s | `72cc6d6` | ✅ Pass |

### Mutagenesis (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `random-nucleotide-sample` | no | - | 0.1s | `72cc6d6` | ✅ Pass |
| `random-protein-sample` | no | - | 0.0s | `72cc6d6` | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `orfipy-prediction` | no | ✅ | 3.3s | `72cc6d6` | ✅ Pass |
| `prodigal-prediction` | no | ✅ | 2.8s | `72cc6d6` | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `splice-transformer-prediction` | yes | ✅ | 9.3s | `72cc6d6` | ✅ Pass |

### Sequence Alignment (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `colabfold-search` | no | ✅ | 12.5s | `72cc6d6` | ✅ Pass |
| `mafft-align` | no | ✅ | 8.8s | `72cc6d6` | ✅ Pass |
| `mmseqs2-homology-search` | no | ✅ | 245.8s | `72cc6d6` | ✅ Pass |

### Sequence Scoring (3/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphagenome-predict-intervals` | yes | ✅ | 109.8s | `72cc6d6` | ✅ Pass |
| `borzoi-ensemble` | yes | ✅ | 78.4s | `72cc6d6` | ✅ Pass |
| `enformer-prediction` | yes | ✅ | 20.6s | `72cc6d6` | ✅ Pass |
| `segmasker-score` | no | ✅ | 64.8s | `72cc6d6` | ❌ Fail |

### Structure Alignment (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `tmalign-alignment` | no | ✅ | 7.8s | `72cc6d6` | ✅ Pass |
| `usalign-alignment` | no | ✅ | 13.0s | `72cc6d6` | ✅ Pass |

### Structure Design (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `rfdiffusion3-design` | yes | ✅ | 1192.2s | `72cc6d6` | ✅ Pass |

### Structure Dynamics (0/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `bioemu-sample` | yes | ✅ | 31.9s | `72cc6d6` | ❌ Fail |

### Structure Prediction (4/7)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphafold2-binder` | yes | ✅ | 131.8s | `72cc6d6` | ✅ Pass |
| `alphafold3-prediction` | yes | ✅ | 202.3s | `72cc6d6` | ❌ Fail |
| `boltz2-prediction` | yes | ✅ | 52.4s | `72cc6d6` | ❌ Fail |
| `chai1-prediction` | yes | ✅ | 1.9s | `72cc6d6` | ❌ Fail |
| `esmfold-prediction` | yes | ✅ | 37.7s | `72cc6d6` | ✅ Pass |
| `protenix-prediction` | yes | ✅ | 184.6s | `72cc6d6` | ✅ Pass |
| `viennarna-prediction` | no | ✅ | 2.8s | `72cc6d6` | ✅ Pass |

### Structure Scoring (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `dssp-secondary-structure` | no | ✅ | 8.6s | `72cc6d6` | ✅ Pass |
| `pdockq2` | no | - | 0.1s | `72cc6d6` | ✅ Pass |
| `pyrosetta-energy` | no | ✅ | 7.8s | `72cc6d6` | ✅ Pass |
| `structure-metrics` | no | - | 0.1s | `72cc6d6` | ✅ Pass |

### Testing (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `mock-cli-multi-gpu-tool-run` | yes | - | - | `72cc6d6` | ⏭️ Skip |
| `mock-cli-tool-run` | yes | ✅ | 2.2s | `72cc6d6` | ✅ Pass |
| `mock-jax-multi-gpu-tool-run` | yes | - | - | `72cc6d6` | ⏭️ Skip |
| `mock-jax-tool-run` | yes | ✅ | 9.3s | `72cc6d6` | ✅ Pass |
| `mock-pytorch-multi-gpu-tool-run` | yes | - | - | `72cc6d6` | ⏭️ Skip |
| `mock-pytorch-tool-run` | yes | ✅ | 18.7s | `72cc6d6` | ✅ Pass |

## Failure Details

### ❌ `alphafold3-prediction`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold3-prediction]`

```
tests/tool_infra_tests/test_env_report.py:74: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool alphafold3-prediction failed: ["'alphafold3' may not be compatible with your system. setup.sh failed (exit 1).\n     directory, OR point PROTO_ALPHAFOLD3_WEIGHTS_DIR at the\n     directory containing it.\nSee notes/storage.md for PROTO_MODEL_CACHE / PROTO_HOME rules.\n============================================================\nERROR: No AlphaFold3 weights (*.bin / *.bin.zst) found in:\n  /home/bviggiano/.proto/proto_model_cache/alphafold3\nFix: download DeepMind-licensed weights and either place\naf3.bin.zst in the directory above, or set\nPROTO_ALPHAFOLD3_WEIGHTS_DIR=/abs/path/to/weights/dir.\n============================================================", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 566, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/alphafold3/alphafold3.py", line 269, in run_alphafold3\n    output_data = ToolInstance.dispatch(\n                  ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 404, in dispatch\n    return cached.run(\n           ^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 766, in run\n    return self._run_persistent(\n           ^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1014, in _run_persistent\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 734, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 2012, in _create_env\n    raise RuntimeError(\nRuntimeError: \'alphafold3\' may not be compatible with your system. setup.sh failed (exit 1).\n     directory, OR point PROTO_ALPHAFOLD3_WEIGHTS_DIR at the\n     directory containing it.\nSee notes/storage.md for PROTO_MODEL_CACHE / PROTO_HOME rules.\n============================================================\nERROR: No AlphaFold3 weights (*.bin / *.bin.zst) found in:\n  /home/bviggiano/.proto/proto_model_cache/alphafold3\nFix: download DeepMind-licensed weights and either place\naf3.bin.zst in the directory above, or set\nPROTO_ALPHAFOLD3_WEIGHTS_DIR=/abs/path/to/weights/dir.\n============================================================\n']
E   assert False
E    +  where False = <[ToolExecutionError('Attempt to access field of tool output after failure: ============================================================\n\nError Messages:\n\'alphafold3\' may not be compatible with your system. setup.sh failed (exit 1).\n     directory, OR point PROTO_ALPHAFOLD3_WEIGHTS_DIR at the\n     directory containing it.\nSee notes/storage.md for PROTO_MODEL_CACHE / PROTO_HOME rules.\n============================================================\nERROR: No AlphaFold3 weights (*.bin / *.bin.zst) found in:\n  /home/bviggiano/.proto/proto_model_cache/alphafold3\nFix: download DeepMind-licensed weights and either place\naf3.bin.zst in the directory above, or set\nPROTO_ALPHAFOLD3_WEIGHTS_DIR=/abs/path/to/weights/dir.\n============================================================\nTraceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 566, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/alphafold3/alphafold3.py", line 269, in run_alphafold3\n    output_data...turn self._run_persistent(\n           ^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1014, in _run_persistent\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 734, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 2012, in _create_env\n    raise RuntimeError(\nRuntimeError: \'alphafold3\' may not be compatible with your system. setup.sh failed (exit 1).\n     directory, OR point PROTO_ALPHAFOLD3_WEIGHTS_DIR at the\n     directory containing it.\nSee notes/storage.md for PROTO_MODEL_CACHE / PROTO_HOME rules.\n============================================================\nERROR: No AlphaFold3 weights (*.bin / *.bin.zst) found in:\n  /home/bviggiano/.proto/proto_model_cache/alphafold3\nFix: download DeepMind-licensed weights and either place\naf3.bin.zst in the directory above, or set\nPROTO_ALPHAFOLD3_WEIGHTS_DIR=/abs/path/to/weights/dir.\n============================================================\n') raised in repr()] AlphaFold3Output object at 0xf0e976827110>.success
```

### ❌ `progen3-sample`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen3-sample]`

```
tests/tool_infra_tests/test_env_report.py:74: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool progen3-sample failed: ["'progen3' may not be compatible with your system. setup.sh failed (exit 1).\nbash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\nERROR: ProGen3 is not supported on aarch64.\nProGen3 requires flash-attn which has no aarch64 wheels.", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 566, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/causal_models/progen3/progen3_sample.py", line 226, in run_progen3_sample\n    result = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 422, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 461, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1173, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 734, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 2012, in _create_env\n    raise RuntimeError(\nRuntimeError: \'progen3\' may not be compatible with your system. setup.sh failed (exit 1).\nbash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\nERROR: ProGen3 is not supported on aarch64.\nProGen3 requires flash-attn which has no aarch64 wheels.\n']
E   assert False
E    +  where False = CausalModelSampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success
```

### ❌ `progen2-sample`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen2-sample]`

```
tests/tool_infra_tests/test_env_report.py:74: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool progen2-sample failed: ["'progen2' may not be compatible with your system. setup.sh failed (exit 1).\nbash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\nERROR: ProGen2 is not supported on aarch64.\nProGen2 pins torch==2.2.2 which has no aarch64 CUDA wheel available.", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 566, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/causal_models/progen2/progen2_sample.py", line 256, in run_progen2_sample\n    result = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 422, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 461, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1173, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 734, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 2012, in _create_env\n    raise RuntimeError(\nRuntimeError: \'progen2\' may not be compatible with your system. setup.sh failed (exit 1).\nbash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\nERROR: ProGen2 is not supported on aarch64.\nProGen2 pins torch==2.2.2 which has no aarch64 CUDA wheel available.\n']
E   assert False
E    +  where False = ProGen2SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, logits).success
```

### ❌ `crispr-tracr`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[crispr-tracr]`

```
tests/tool_infra_tests/test_env_report.py:74: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool crispr-tracr failed: ["'crispr_tracr' may not be compatible with your system. setup.sh failed (exit 1).\nCloning into '/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA'...\nCloning CRISPRidentify into CRISPRtracrRNA tools directory...\nCloning into '/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA/tools/CRISPRidentify/CRISPRidentify'...\nCloning CRISPRcasIdentifier into CRISPRtracrRNA tools directory...\nCloning into '/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA/tools/CRISPRcasIdentifier/CRISPRcasIdentifier'...\nCreating isolated conda environment (Python 3.8 + scikit-learn 0.22)...\nCRISPRidentify's pickled models require sklearn 0.22 (incompatible with 3.12).\nUsing /home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/conda_deps to avoid polluting base env...\nERROR: CRISPRtracrRNA requires x86_64 bioconda packages (vmatch, etc.)\n       that are not available on Linux aarch64.", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 566, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/gene_annotation/crispr_tracr/crispr_tracr.py", line 214, in run_crispr_tracr\n    output_data = ToolInstance.dispatch(\n                  ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 422, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 468, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1173, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 734, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 2012, in _create_env\n    raise RuntimeError(\nRuntimeError: \'crispr_tracr\' may not be compatible with your system. setup.sh failed (exit 1).\nCloning into \'/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA\'...\nCloning CRISPRidentify into CRISPRtracrRNA tools directory...\nCloning into \'/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA/tools/CRISPRidentify/CRISPRidentify\'...\nCloning CRISPRcasIdentifier into CRISPRtracrRNA tools directory...\nCloning into \'/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA/tools/CRISPRcasIdentifier/CRISPRcasIdentifier\'...\nCreating isolated conda environment (Python 3.8 + scikit-learn 0.22)...\nCRISPRidentify\'s pickled models require sklearn 0.22 (incompatible with 3.12).\nUsing /home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/conda_deps to avoid polluting base env...\nERROR: CRISPRtracrRNA requires x86_64 bioconda packages (vmatch, etc.)\n       that are not available on Linux aarch64.\n']
E   assert False
E    +  where False = CrisprTracrOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, predictions).success
```

### ❌ `bioemu-sample`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[bioemu-sample]`

```
tests/tool_infra_tests/test_env_report.py:74: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool bioemu-sample failed: ['Worker for bioemu returned an error:\nTraceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/_worker_bootstrap.py", line 274, in main\n    result = dispatch(input_dict)\n             ^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py", line 189, in dispatch\n    return run_bioemu_batch(input_dict)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py", line 163, in run_bioemu_batch\n    result = model(\n             ^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py", line 61, in __call__\n    bioemu_sample(\n  File "/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/bioemu/utils.py", line 59, in with_stackprint\n    return func(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context\n    return func(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/bioemu/sample.py", line 180, in main\n    batch = generate_batch(\n            ^^^^^^^^^^^^^^^\n  File "/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/bioemu/sample.py", line 286, in generate_batch\n    context_chemgraph = get_context_chemgraph(\n                        ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/bioemu/sample.py", line 222, in get_context_chemgraph\n    single_embeds_file, pair_embeds_file = get_colabfold_embeds(\n                                           ^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/bioemu/get_embeds.py", line 169, in get_colabfold_embeds\n    colabfold_bin_dir = ensure_colabfold_install()\n                        ^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/bioemu/get_embeds.py", line 82, in ensure_colabfold_install\n    assert result.returncode == 0, (\n           ^^^^^^^^^^^^^^^^^^^^^^\nAssertionError: Something went wrong during colabfold install:\nbash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\n+ echo \'Setting up colabfold...\'\nSetting up colabfold...\n+ BASE_PYTHON=/home/bviggiano/.proto/proto_tool_envs/bioemu_env/bin/python\n+ VENV_FOLDER=/home/bviggiano/.bioemu_colabfold\n+ /home/bviggiano/.proto/proto_tool_envs/bioemu_env/bin/python -m venv --without-pip /home/bviggiano/.bioemu_colabfold\n+ /home/bviggiano/.proto/proto_tool_envs/bioemu_env/bin/python -m uv pip install --python /home/bviggiano/.bioemu_colabfold/bin/python \'colabfold[alphafold-minus-jax]==1.5.4\'\nUsing Python 3.11.15 environment at: /home/bviggiano/.bioemu_colabfold\n  × No solution found when resolving dependencies:\n  ╰─▶ Because only the following versions of tensorflow-cpu{sys_platform !=\n      \'darwin\'} are available:\n          tensorflow-cpu{sys_platform != \'darwin\'}<=2.12.1\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.13.0\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.13.1\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.14.0\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.14.1\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.15.0\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.15.0.post1\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.15.1\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.16.1\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.16.2\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.17.0\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.17.1\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.18.0\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.18.1\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.19.0\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.19.1\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.20.0\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.21.0\n      and tensorflow-cpu{sys_platform != \'darwin\'}>=2.12.1 has no wheels\n      with a matching platform tag (e.g., `manylinux_2_39_aarch64`), we can\n      conclude that tensorflow-cpu{sys_platform != \'darwin\'}>=2.12.1 cannot\n      be used.\n      And because colabfold==1.5.4 depends on tensorflow-cpu{sys_platform !=\n      \'darwin\'}>=2.12.1 and you require colabfold[alphafold-minus-jax]==1.5.4,\n      we can conclude that your requirements are unsatisfiable.\n\n      hint: Pre-releases are available for `tensorflow-cpu` in the requested\n      range (e.g., 2.21.0rc1), but pre-releases weren\'t enabled (try:\n      `--prerelease=allow`)\n\n      hint: Wheels are available for `tensorflow-cpu` (v2.21.0) on the\n      following platforms: `manylinux_2_27_x86_64`, `win_amd64`\n\nPlease check the colabfold install log saved in /home/bviggiano/.bioemu_colabfold/install_log.txt.\n', 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 566, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/bioemu_sample.py", line 263, in run_bioemu\n    output = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 404, in dispatch\n    return cached.run(\n           ^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 766, in run\n    return self._run_persistent(\n           ^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1142, in _run_persistent\n    result = self._worker.send(input_dict, timeout=effective_timeout)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/persistent_worker.py", line 574, in send\n    raise RuntimeError(f"Worker for {self.toolkit} returned an error:\\n{response[\'error\']}")\nRuntimeError: Worker for bioemu returned an error:\nTraceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/_worker_bootstrap.py", line 274, in main\n    result = dispatch(input_dict)\n             ^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py", line 189, in dispatch\n    return run_bioemu_batch(input_dict)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py", line 163, in run_bioemu_batch\n    result = model(\n             ^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py", line 61, in __call__\n    bioemu_sample(\n  File "/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/bioemu/utils.py", line 59, in with_stackprint\n    return func(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context\n    return func(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/bioemu/sample.py", line 180, in main\n    batch = generate_batch(\n            ^^^^^^^^^^^^^^^\n  File "/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/bioemu/sample.py", line 286, in generate_batch\n    context_chemgraph = get_context_chemgraph(\n                        ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/bioemu/sample.py", line 222, in get_context_chemgraph\n    single_embeds_file, pair_embeds_file = get_colabfold_embeds(\n                                           ^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/bioemu/get_embeds.py", line 169, in get_colabfold_embeds\n    colabfold_bin_dir = ensure_colabfold_install()\n                        ^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/bioemu/get_embeds.py", line 82, in ensure_colabfold_install\n    assert result.returncode == 0, (\n           ^^^^^^^^^^^^^^^^^^^^^^\nAssertionError: Something went wrong during colabfold install:\nbash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\n+ echo \'Setting up colabfold...\'\nSetting up colabfold...\n+ BASE_PYTHON=/home/bviggiano/.proto/proto_tool_envs/bioemu_env/bin/python\n+ VENV_FOLDER=/home/bviggiano/.bioemu_colabfold\n+ /home/bviggiano/.proto/proto_tool_envs/bioemu_env/bin/python -m venv --without-pip /home/bviggiano/.bioemu_colabfold\n+ /home/bviggiano/.proto/proto_tool_envs/bioemu_env/bin/python -m uv pip install --python /home/bviggiano/.bioemu_colabfold/bin/python \'colabfold[alphafold-minus-jax]==1.5.4\'\nUsing Python 3.11.15 environment at: /home/bviggiano/.bioemu_colabfold\n  × No solution found when resolving dependencies:\n  ╰─▶ Because only the following versions of tensorflow-cpu{sys_platform !=\n      \'darwin\'} are available:\n          tensorflow-cpu{sys_platform != \'darwin\'}<=2.12.1\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.13.0\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.13.1\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.14.0\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.14.1\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.15.0\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.15.0.post1\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.15.1\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.16.1\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.16.2\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.17.0\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.17.1\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.18.0\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.18.1\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.19.0\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.19.1\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.20.0\n          tensorflow-cpu{sys_platform != \'darwin\'}==2.21.0\n      and tensorflow-cpu{sys_platform != \'darwin\'}>=2.12.1 has no wheels\n      with a matching platform tag (e.g., `manylinux_2_39_aarch64`), we can\n      conclude that tensorflow-cpu{sys_platform != \'darwin\'}>=2.12.1 cannot\n      be used.\n      And because colabfold==1.5.4 depends on tensorflow-cpu{sys_platform !=\n      \'darwin\'}>=2.12.1 and you require colabfold[alphafold-minus-jax]==1.5.4,\n      we can conclude that your requirements are unsatisfiable.\n\n      hint: Pre-releases are available for `tensorflow-cpu` in the requested\n      range (e.g., 2.21.0rc1), but pre-releases weren\'t enabled (try:\n      `--prerelease=allow`)\n\n      hint: Wheels are available for `tensorflow-cpu` (v2.21.0) on the\n      following platforms: `manylinux_2_27_x86_64`, `win_amd64`\n\nPlease check the colabfold install log saved in /home/bviggiano/.bioemu_colabfold/install_log.txt.\n\n']
E   assert False
E    +  where False = BioEmuOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success
```

### ❌ `chai1-prediction`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[chai1-prediction]`

```
tests/tool_infra_tests/test_env_report.py:74: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool chai1-prediction failed: ["'chai1' may not be compatible with your system. setup.sh failed (exit 1).\nbash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\nERROR: Chai is not supported on aarch64.\nchai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its\npre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 566, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py", line 302, in run_chai1\n    run_chai1_on_complex(comp=comp, config=config, msas=inputs.msas, instance=instance)\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py", line 388, in run_chai1_on_complex\n    result = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 404, in dispatch\n    return cached.run(\n           ^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 766, in run\n    return self._run_persistent(\n           ^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1014, in _run_persistent\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 734, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 2012, in _create_env\n    raise RuntimeError(\nRuntimeError: \'chai1\' may not be compatible with your system. setup.sh failed (exit 1).\nbash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\nERROR: Chai is not supported on aarch64.\nchai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its\npre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\n']
E   assert False
E    +  where False = <[ToolExecutionError('Attempt to access field of tool output after failure: pre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\n\nError Messages:\n\'chai1\' may not be compatible with your system. setup.sh failed (exit 1).\nbash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\nERROR: Chai is not supported on aarch64.\nchai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its\npre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\nTraceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 566, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py", line 302, in run_chai1\n    run_chai1_on_complex(comp=comp, config=config, msas=inputs.msas, instance=instance)\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py", line 388, in run_chai1_on_complex\n    result = ToolInst...codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 404, in dispatch\n    return cached.run(\n           ^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 766, in run\n    return self._run_persistent(\n           ^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1014, in _run_persistent\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 734, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 2012, in _create_env\n    raise RuntimeError(\nRuntimeError: \'chai1\' may not be compatible with your system. setup.sh failed (exit 1).\nbash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\nERROR: Chai is not supported on aarch64.\nchai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its\npre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\n') raised in repr()] Chai1Output object at 0xf0e934d6fc00>.success
```

### ❌ `boltz2-prediction`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[boltz2-prediction]`

```
tests/tool_infra_tests/test_env_report.py:74: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool boltz2-prediction failed: ['Worker for boltz2 returned an error:\nTraceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/_worker_bootstrap.py", line 274, in main\n    result = dispatch(input_dict)\n             ^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py", line 201, in dispatch\n    return _model(\n           ^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py", line 118, in __call__\n    subprocess.run(\n  File "/home/bviggiano/.proto/proto_tool_envs/boltz2_env/lib/python3.12/subprocess.py", line 571, in run\n    raise CalledProcessError(retcode, process.args,\nsubprocess.CalledProcessError: Command \'[\'/home/bviggiano/.proto/proto_tool_envs/boltz2_env/bin/boltz\', \'predict\', \'/tmp/tmp_avx4moy/boltz2_input.yaml\', \'--out_dir=/tmp/tmp_avx4moy/boltz2_output\', \'--recycling_steps=10\', \'--diffusion_samples=25\', \'--sampling_steps=200\', \'--output_format=mmcif\', \'--devices=1\', \'--cache=/home/bviggiano/.proto/proto_model_cache/boltz2\', \'--num_workers=4\']\' returned non-zero exit status 1.\n', 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 566, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/boltz2.py", line 309, in run_boltz2\n    run_boltz2_on_complex(\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/boltz2.py", line 397, in run_boltz2_on_complex\n    output_data = ToolInstance.dispatch(\n                  ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 404, in dispatch\n    return cached.run(\n           ^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 766, in run\n    return self._run_persistent(\n           ^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1142, in _run_persistent\n    result = self._worker.send(input_dict, timeout=effective_timeout)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/persistent_worker.py", line 574, in send\n    raise RuntimeError(f"Worker for {self.toolkit} returned an error:\\n{response[\'error\']}")\nRuntimeError: Worker for boltz2 returned an error:\nTraceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/_worker_bootstrap.py", line 274, in main\n    result = dispatch(input_dict)\n             ^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py", line 201, in dispatch\n    return _model(\n           ^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py", line 118, in __call__\n    subprocess.run(\n  File "/home/bviggiano/.proto/proto_tool_envs/boltz2_env/lib/python3.12/subprocess.py", line 571, in run\n    raise CalledProcessError(retcode, process.args,\nsubprocess.CalledProcessError: Command \'[\'/home/bviggiano/.proto/proto_tool_envs/boltz2_env/bin/boltz\', \'predict\', \'/tmp/tmp_avx4moy/boltz2_input.yaml\', \'--out_dir=/tmp/tmp_avx4moy/boltz2_output\', \'--recycling_steps=10\', \'--diffusion_samples=25\', \'--sampling_steps=200\', \'--output_format=mmcif\', \'--devices=1\', \'--cache=/home/bviggiano/.proto/proto_model_cache/boltz2\', \'--num_workers=4\']\' returned non-zero exit status 1.\n\n']
E   assert False
E    +  where False = <[ToolExecutionError('Attempt to access field of tool output after failure: \n\nError Messages:\nWorker for boltz2 returned an error:\nTraceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/_worker_bootstrap.py", line 274, in main\n    result = dispatch(input_dict)\n             ^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py", line 201, in dispatch\n    return _model(\n           ^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py", line 118, in __call__\n    subprocess.run(\n  File "/home/bviggiano/.proto/proto_tool_envs/boltz2_env/lib/python3.12/subprocess.py", line 571, in run\n    raise CalledProcessError(retcode, process.args,\nsubprocess.CalledProcessError: Command \'[\'/home/bviggiano/.proto/proto_tool_envs/boltz2_env/bin/boltz\', \'predict\', \'/tmp/tmp_avx4moy/boltz2_input.yaml\', \'--out_dir=/tmp/tmp_avx4moy/boltz2_output\', \'--recycling_steps=10\', \'--diffusion_samples=25\', \'--sampling_steps=200\', \'--output_format=mmcif\', \'--devices=1\', ..."/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/_worker_bootstrap.py", line 274, in main\n    result = dispatch(input_dict)\n             ^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py", line 201, in dispatch\n    return _model(\n           ^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py", line 118, in __call__\n    subprocess.run(\n  File "/home/bviggiano/.proto/proto_tool_envs/boltz2_env/lib/python3.12/subprocess.py", line 571, in run\n    raise CalledProcessError(retcode, process.args,\nsubprocess.CalledProcessError: Command \'[\'/home/bviggiano/.proto/proto_tool_envs/boltz2_env/bin/boltz\', \'predict\', \'/tmp/tmp_avx4moy/boltz2_input.yaml\', \'--out_dir=/tmp/tmp_avx4moy/boltz2_output\', \'--recycling_steps=10\', \'--diffusion_samples=25\', \'--sampling_steps=200\', \'--output_format=mmcif\', \'--devices=1\', \'--cache=/home/bviggiano/.proto/proto_model_cache/boltz2\', \'--num_workers=4\']\' returned non-zero exit status 1.\n\n') raised in repr()] Boltz2Output object at 0xf0e934db0dc0>.success
```

### ❌ `evo2-sample`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo2-sample]`

```
tests/tool_infra_tests/test_env_report.py:74: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool evo2-sample failed: ["'evo2' may not be compatible with your system. setup.sh failed (exit 1).\nbash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\nERROR: Evo2 is not supported on aarch64.\nEvo2 requires transformer-engine and flash-attn which only provide x86_64 pre-built wheels.", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 566, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/causal_models/evo2/evo2_sample.py", line 259, in run_evo2_sample\n    result = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 422, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 461, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1173, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 734, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 2012, in _create_env\n    raise RuntimeError(\nRuntimeError: \'evo2\' may not be compatible with your system. setup.sh failed (exit 1).\nbash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\nERROR: Evo2 is not supported on aarch64.\nEvo2 requires transformer-engine and flash-attn which only provide x86_64 pre-built wheels.\n']
E   assert False
E    +  where False = Evo2SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, logits, kv_caches).success
```

### ❌ `evo1-sample`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo1-sample]`

```
tests/tool_infra_tests/test_env_report.py:74: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool evo1-sample failed: ['\'evo1\' may not be compatible with your system. setup.sh failed (exit 1).\n      on `psutil`, but doesn\'t declare it as a build dependency. If\n      `flash-attn` is a first-party package, consider adding `psutil`\n      to its `build-system.requires`. Otherwise, either add it to your\n      `pyproject.toml` under:\n      [tool.uv.extra-build-dependencies]\n      flash-attn = ["psutil"]\n      or `uv pip install psutil` into the environment and re-run with\n      `--no-build-isolation`.\n  help: `flash-attn` (v2.8.3) was included because `evo-model` (v0.5) depends\n        on `stripedhyena` (v0.2.2) which depends on `flash-attn`', 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 566, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/causal_models/evo1/evo1_sample.py", line 151, in run_evo1_sample\n    result = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 422, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 461, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1173, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 734, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 2012, in _create_env\n    raise RuntimeError(\nRuntimeError: \'evo1\' may not be compatible with your system. setup.sh failed (exit 1).\n      on `psutil`, but doesn\'t declare it as a build dependency. If\n      `flash-attn` is a first-party package, consider adding `psutil`\n      to its `build-system.requires`. Otherwise, either add it to your\n      `pyproject.toml` under:\n      [tool.uv.extra-build-dependencies]\n      flash-attn = ["psutil"]\n      or `uv pip install psutil` into the environment and re-run with\n      `--no-build-isolation`.\n  help: `flash-attn` (v2.8.3) was included because `evo-model` (v0.5) depends\n        on `stripedhyena` (v0.2.2) which depends on `flash-attn`\n']
E   assert False
E    +  where False = Evo1SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, scores).success
```

### ❌ `segmasker-score`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[segmasker-score]`

```
tests/tool_infra_tests/test_env_report.py:74: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool segmasker-score failed: ["Command '['/home/bviggiano/.proto/proto_tool_envs/segmasker_env/bin/python', '/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_scoring/segmasker/standalone/run.py', '/tmp/tmpbyg9b7ll/input.json', '/tmp/tmpbyg9b7ll/output.json']' returned non-zero exit status 1.", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 566, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_scoring/segmasker/segmasker.py", line 234, in run_segmasker\n    output_data = ToolInstance.dispatch(\n                  ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 422, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 468, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1215, in _run_oneshot\n    subprocess.run(\n  File "/home/bviggiano/miniconda3/envs/proto-tools/lib/python3.12/subprocess.py", line 571, in run\n    raise CalledProcessError(retcode, process.args,\nsubprocess.CalledProcessError: Command \'[\'/home/bviggiano/.proto/proto_tool_envs/segmasker_env/bin/python\', \'/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_scoring/segmasker/standalone/run.py\', \'/tmp/tmpbyg9b7ll/input.json\', \'/tmp/tmpbyg9b7ll/output.json\']\' returned non-zero exit status 1.\n']
E   assert False
E    +  where False = SegmaskerOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, results).success
```

---
*Generated at 2026-04-29 11:54:38 by `pytest --env-report`*

<!-- env-report-data
[
  {
    "tool_key": "alphafold3-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold3-prediction]",
    "status": "failed",
    "duration_seconds": 202.32,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/alphafold3_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:74: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool alphafold3-prediction failed: [\"'alphafold3' may not be compatible with your system. setup.sh failed (exit 1).\\n     directory, OR point PROTO_ALPHAFOLD3_WEIGHTS_DIR at the\\n     directory containing it.\\nSee notes/storage.md for PROTO_MODEL_CACHE / PROTO_HOME rules.\\n============================================================\\nERROR: No AlphaFold3 weights (*.bin / *.bin.zst) found in:\\n  /home/bviggiano/.proto/proto_model_cache/alphafold3\\nFix: download DeepMind-licensed weights and either place\\naf3.bin.zst in the directory above, or set\\nPROTO_ALPHAFOLD3_WEIGHTS_DIR=/abs/path/to/weights/dir.\\n============================================================\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 566, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/alphafold3/alphafold3.py\", line 269, in run_alphafold3\\n    output_data = ToolInstance.dispatch(\\n                  ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 404, in dispatch\\n    return cached.run(\\n           ^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 766, in run\\n    return self._run_persistent(\\n           ^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1014, in _run_persistent\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 734, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 2012, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'alphafold3\\' may not be compatible with your system. setup.sh failed (exit 1).\\n     directory, OR point PROTO_ALPHAFOLD3_WEIGHTS_DIR at the\\n     directory containing it.\\nSee notes/storage.md for PROTO_MODEL_CACHE / PROTO_HOME rules.\\n============================================================\\nERROR: No AlphaFold3 weights (*.bin / *.bin.zst) found in:\\n  /home/bviggiano/.proto/proto_model_cache/alphafold3\\nFix: download DeepMind-licensed weights and either place\\naf3.bin.zst in the directory above, or set\\nPROTO_ALPHAFOLD3_WEIGHTS_DIR=/abs/path/to/weights/dir.\\n============================================================\\n']\nE   assert False\nE    +  where False = <[ToolExecutionError('Attempt to access field of tool output after failure: ============================================================\\n\\nError Messages:\\n\\'alphafold3\\' may not be compatible with your system. setup.sh failed (exit 1).\\n     directory, OR point PROTO_ALPHAFOLD3_WEIGHTS_DIR at the\\n     directory containing it.\\nSee notes/storage.md for PROTO_MODEL_CACHE / PROTO_HOME rules.\\n============================================================\\nERROR: No AlphaFold3 weights (*.bin / *.bin.zst) found in:\\n  /home/bviggiano/.proto/proto_model_cache/alphafold3\\nFix: download DeepMind-licensed weights and either place\\naf3.bin.zst in the directory above, or set\\nPROTO_ALPHAFOLD3_WEIGHTS_DIR=/abs/path/to/weights/dir.\\n============================================================\\nTraceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 566, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/alphafold3/alphafold3.py\", line 269, in run_alphafold3\\n    output_data...turn self._run_persistent(\\n           ^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1014, in _run_persistent\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 734, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 2012, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'alphafold3\\' may not be compatible with your system. setup.sh failed (exit 1).\\n     directory, OR point PROTO_ALPHAFOLD3_WEIGHTS_DIR at the\\n     directory containing it.\\nSee notes/storage.md for PROTO_MODEL_CACHE / PROTO_HOME rules.\\n============================================================\\nERROR: No AlphaFold3 weights (*.bin / *.bin.zst) found in:\\n  /home/bviggiano/.proto/proto_model_cache/alphafold3\\nFix: download DeepMind-licensed weights and either place\\naf3.bin.zst in the directory above, or set\\nPROTO_ALPHAFOLD3_WEIGHTS_DIR=/abs/path/to/weights/dir.\\n============================================================\\n') raised in repr()] AlphaFold3Output object at 0xf0e976827110>.success",
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "blast-create-db",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[blast-create-db]",
    "status": "passed",
    "duration_seconds": 85.96,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/blast_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "splice-transformer-prediction",
    "category": "rna_splicing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[splice-transformer-prediction]",
    "status": "passed",
    "duration_seconds": 9.28,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/splice_transformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "colabfold-search",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[colabfold-search]",
    "status": "passed",
    "duration_seconds": 12.46,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/colabfold_search_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "borzoi-ensemble",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[borzoi-ensemble]",
    "status": "passed",
    "duration_seconds": 78.41,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/borzoi_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
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
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "viennarna-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[viennarna-prediction]",
    "status": "passed",
    "duration_seconds": 2.8,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/viennarna_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "mock-pytorch-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-pytorch-tool-run]",
    "status": "passed",
    "duration_seconds": 18.66,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/mock_pytorch_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "usalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[usalign-alignment]",
    "status": "passed",
    "duration_seconds": 13.0,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/usalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "structure-metrics",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[structure-metrics]",
    "status": "passed",
    "duration_seconds": 0.1,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "pyhmmer-hmmscan",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pyhmmer-hmmscan]",
    "status": "passed",
    "duration_seconds": 2.94,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/pyhmmer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "esm-if1-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm-if1-sample]",
    "status": "passed",
    "duration_seconds": 16.48,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/esm_if1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
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
    "error_message": "('/home/bviggiano/codebases/proto/proto-tools/tests/tool_infra_tests/test_env_report.py', 68, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "prodigal-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[prodigal-prediction]",
    "status": "passed",
    "duration_seconds": 2.81,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/prodigal_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "mafft-align",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mafft-align]",
    "status": "passed",
    "duration_seconds": 8.81,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/mafft_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "dssp-secondary-structure",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[dssp-secondary-structure]",
    "status": "passed",
    "duration_seconds": 8.64,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/dssp_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "rfdiffusion3-design",
    "category": "structure_design",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[rfdiffusion3-design]",
    "status": "passed",
    "duration_seconds": 1192.22,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/rfdiffusion3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "progen3-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen3-sample]",
    "status": "failed",
    "duration_seconds": 3.63,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/progen3_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:74: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool progen3-sample failed: [\"'progen3' may not be compatible with your system. setup.sh failed (exit 1).\\nbash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\\nERROR: ProGen3 is not supported on aarch64.\\nProGen3 requires flash-attn which has no aarch64 wheels.\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 566, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/causal_models/progen3/progen3_sample.py\", line 226, in run_progen3_sample\\n    result = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 422, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 461, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1173, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 734, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 2012, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'progen3\\' may not be compatible with your system. setup.sh failed (exit 1).\\nbash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\\nERROR: ProGen3 is not supported on aarch64.\\nProGen3 requires flash-attn which has no aarch64 wheels.\\n']\nE   assert False\nE    +  where False = CausalModelSampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success",
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "ligandmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ligandmpnn-sample]",
    "status": "passed",
    "duration_seconds": 51.1,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/ligandmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "esmfold-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esmfold-prediction]",
    "status": "passed",
    "duration_seconds": 37.69,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/esmfold_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "esmc-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esmc-embedding]",
    "status": "passed",
    "duration_seconds": 29.69,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/evolutionaryscale_esm_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "progen2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen2-sample]",
    "status": "failed",
    "duration_seconds": 2.09,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/progen2_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:74: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool progen2-sample failed: [\"'progen2' may not be compatible with your system. setup.sh failed (exit 1).\\nbash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\\nERROR: ProGen2 is not supported on aarch64.\\nProGen2 pins torch==2.2.2 which has no aarch64 CUDA wheel available.\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 566, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/causal_models/progen2/progen2_sample.py\", line 256, in run_progen2_sample\\n    result = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 422, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 461, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1173, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 734, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 2012, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'progen2\\' may not be compatible with your system. setup.sh failed (exit 1).\\nbash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\\nERROR: ProGen2 is not supported on aarch64.\\nProGen2 pins torch==2.2.2 which has no aarch64 CUDA wheel available.\\n']\nE   assert False\nE    +  where False = ProGen2SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, logits).success",
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "mmseqs-clustering",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mmseqs-clustering]",
    "status": "passed",
    "duration_seconds": 4.42,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/mmseqs_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "alphafold2-binder",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold2-binder]",
    "status": "passed",
    "duration_seconds": 131.85,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/alphafold2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "esm3-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm3-embedding]",
    "status": "passed",
    "duration_seconds": 6.49,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/evolutionaryscale_esm_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "proteinmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[proteinmpnn-sample]",
    "status": "passed",
    "duration_seconds": 29.73,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/proteinmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "crispr-tracr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[crispr-tracr]",
    "status": "failed",
    "duration_seconds": 24.53,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:74: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool crispr-tracr failed: [\"'crispr_tracr' may not be compatible with your system. setup.sh failed (exit 1).\\nCloning into '/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA'...\\nCloning CRISPRidentify into CRISPRtracrRNA tools directory...\\nCloning into '/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA/tools/CRISPRidentify/CRISPRidentify'...\\nCloning CRISPRcasIdentifier into CRISPRtracrRNA tools directory...\\nCloning into '/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA/tools/CRISPRcasIdentifier/CRISPRcasIdentifier'...\\nCreating isolated conda environment (Python 3.8 + scikit-learn 0.22)...\\nCRISPRidentify's pickled models require sklearn 0.22 (incompatible with 3.12).\\nUsing /home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/conda_deps to avoid polluting base env...\\nERROR: CRISPRtracrRNA requires x86_64 bioconda packages (vmatch, etc.)\\n       that are not available on Linux aarch64.\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 566, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/gene_annotation/crispr_tracr/crispr_tracr.py\", line 214, in run_crispr_tracr\\n    output_data = ToolInstance.dispatch(\\n                  ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 422, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 468, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1173, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 734, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 2012, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'crispr_tracr\\' may not be compatible with your system. setup.sh failed (exit 1).\\nCloning into \\'/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA\\'...\\nCloning CRISPRidentify into CRISPRtracrRNA tools directory...\\nCloning into \\'/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA/tools/CRISPRidentify/CRISPRidentify\\'...\\nCloning CRISPRcasIdentifier into CRISPRtracrRNA tools directory...\\nCloning into \\'/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA/tools/CRISPRcasIdentifier/CRISPRcasIdentifier\\'...\\nCreating isolated conda environment (Python 3.8 + scikit-learn 0.22)...\\nCRISPRidentify\\'s pickled models require sklearn 0.22 (incompatible with 3.12).\\nUsing /home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/conda_deps to avoid polluting base env...\\nERROR: CRISPRtracrRNA requires x86_64 bioconda packages (vmatch, etc.)\\n       that are not available on Linux aarch64.\\n']\nE   assert False\nE    +  where False = CrisprTracrOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, predictions).success",
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "mmseqs2-homology-search",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mmseqs2-homology-search]",
    "status": "passed",
    "duration_seconds": 245.77,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/mmseqs2_homology_search_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "alphagenome-predict-intervals",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphagenome-predict-intervals]",
    "status": "passed",
    "duration_seconds": 109.78,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/alphagenome_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "bioemu-sample",
    "category": "structure_dynamics",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[bioemu-sample]",
    "status": "failed",
    "duration_seconds": 31.93,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/bioemu_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:74: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool bioemu-sample failed: ['Worker for bioemu returned an error:\\nTraceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/_worker_bootstrap.py\", line 274, in main\\n    result = dispatch(input_dict)\\n             ^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py\", line 189, in dispatch\\n    return run_bioemu_batch(input_dict)\\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py\", line 163, in run_bioemu_batch\\n    result = model(\\n             ^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py\", line 61, in __call__\\n    bioemu_sample(\\n  File \"/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/bioemu/utils.py\", line 59, in with_stackprint\\n    return func(*args, **kwargs)\\n           ^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/torch/utils/_contextlib.py\", line 124, in decorate_context\\n    return func(*args, **kwargs)\\n           ^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/bioemu/sample.py\", line 180, in main\\n    batch = generate_batch(\\n            ^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/bioemu/sample.py\", line 286, in generate_batch\\n    context_chemgraph = get_context_chemgraph(\\n                        ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/bioemu/sample.py\", line 222, in get_context_chemgraph\\n    single_embeds_file, pair_embeds_file = get_colabfold_embeds(\\n                                           ^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/bioemu/get_embeds.py\", line 169, in get_colabfold_embeds\\n    colabfold_bin_dir = ensure_colabfold_install()\\n                        ^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/bioemu/get_embeds.py\", line 82, in ensure_colabfold_install\\n    assert result.returncode == 0, (\\n           ^^^^^^^^^^^^^^^^^^^^^^\\nAssertionError: Something went wrong during colabfold install:\\nbash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\\n+ echo \\'Setting up colabfold...\\'\\nSetting up colabfold...\\n+ BASE_PYTHON=/home/bviggiano/.proto/proto_tool_envs/bioemu_env/bin/python\\n+ VENV_FOLDER=/home/bviggiano/.bioemu_colabfold\\n+ /home/bviggiano/.proto/proto_tool_envs/bioemu_env/bin/python -m venv --without-pip /home/bviggiano/.bioemu_colabfold\\n+ /home/bviggiano/.proto/proto_tool_envs/bioemu_env/bin/python -m uv pip install --python /home/bviggiano/.bioemu_colabfold/bin/python \\'colabfold[alphafold-minus-jax]==1.5.4\\'\\nUsing Python 3.11.15 environment at: /home/bviggiano/.bioemu_colabfold\\n  \u00d7 No solution found when resolving dependencies:\\n  \u2570\u2500\u25b6 Because only the following versions of tensorflow-cpu{sys_platform !=\\n      \\'darwin\\'} are available:\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}<=2.12.1\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.13.0\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.13.1\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.14.0\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.14.1\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.15.0\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.15.0.post1\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.15.1\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.16.1\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.16.2\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.17.0\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.17.1\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.18.0\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.18.1\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.19.0\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.19.1\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.20.0\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.21.0\\n      and tensorflow-cpu{sys_platform != \\'darwin\\'}>=2.12.1 has no wheels\\n      with a matching platform tag (e.g., `manylinux_2_39_aarch64`), we can\\n      conclude that tensorflow-cpu{sys_platform != \\'darwin\\'}>=2.12.1 cannot\\n      be used.\\n      And because colabfold==1.5.4 depends on tensorflow-cpu{sys_platform !=\\n      \\'darwin\\'}>=2.12.1 and you require colabfold[alphafold-minus-jax]==1.5.4,\\n      we can conclude that your requirements are unsatisfiable.\\n\\n      hint: Pre-releases are available for `tensorflow-cpu` in the requested\\n      range (e.g., 2.21.0rc1), but pre-releases weren\\'t enabled (try:\\n      `--prerelease=allow`)\\n\\n      hint: Wheels are available for `tensorflow-cpu` (v2.21.0) on the\\n      following platforms: `manylinux_2_27_x86_64`, `win_amd64`\\n\\nPlease check the colabfold install log saved in /home/bviggiano/.bioemu_colabfold/install_log.txt.\\n', 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 566, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/bioemu_sample.py\", line 263, in run_bioemu\\n    output = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 404, in dispatch\\n    return cached.run(\\n           ^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 766, in run\\n    return self._run_persistent(\\n           ^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1142, in _run_persistent\\n    result = self._worker.send(input_dict, timeout=effective_timeout)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/persistent_worker.py\", line 574, in send\\n    raise RuntimeError(f\"Worker for {self.toolkit} returned an error:\\\\n{response[\\'error\\']}\")\\nRuntimeError: Worker for bioemu returned an error:\\nTraceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/_worker_bootstrap.py\", line 274, in main\\n    result = dispatch(input_dict)\\n             ^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py\", line 189, in dispatch\\n    return run_bioemu_batch(input_dict)\\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py\", line 163, in run_bioemu_batch\\n    result = model(\\n             ^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py\", line 61, in __call__\\n    bioemu_sample(\\n  File \"/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/bioemu/utils.py\", line 59, in with_stackprint\\n    return func(*args, **kwargs)\\n           ^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/torch/utils/_contextlib.py\", line 124, in decorate_context\\n    return func(*args, **kwargs)\\n           ^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/bioemu/sample.py\", line 180, in main\\n    batch = generate_batch(\\n            ^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/bioemu/sample.py\", line 286, in generate_batch\\n    context_chemgraph = get_context_chemgraph(\\n                        ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/bioemu/sample.py\", line 222, in get_context_chemgraph\\n    single_embeds_file, pair_embeds_file = get_colabfold_embeds(\\n                                           ^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/bioemu/get_embeds.py\", line 169, in get_colabfold_embeds\\n    colabfold_bin_dir = ensure_colabfold_install()\\n                        ^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/.proto/proto_tool_envs/bioemu_env/lib/python3.11/site-packages/bioemu/get_embeds.py\", line 82, in ensure_colabfold_install\\n    assert result.returncode == 0, (\\n           ^^^^^^^^^^^^^^^^^^^^^^\\nAssertionError: Something went wrong during colabfold install:\\nbash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\\n+ echo \\'Setting up colabfold...\\'\\nSetting up colabfold...\\n+ BASE_PYTHON=/home/bviggiano/.proto/proto_tool_envs/bioemu_env/bin/python\\n+ VENV_FOLDER=/home/bviggiano/.bioemu_colabfold\\n+ /home/bviggiano/.proto/proto_tool_envs/bioemu_env/bin/python -m venv --without-pip /home/bviggiano/.bioemu_colabfold\\n+ /home/bviggiano/.proto/proto_tool_envs/bioemu_env/bin/python -m uv pip install --python /home/bviggiano/.bioemu_colabfold/bin/python \\'colabfold[alphafold-minus-jax]==1.5.4\\'\\nUsing Python 3.11.15 environment at: /home/bviggiano/.bioemu_colabfold\\n  \u00d7 No solution found when resolving dependencies:\\n  \u2570\u2500\u25b6 Because only the following versions of tensorflow-cpu{sys_platform !=\\n      \\'darwin\\'} are available:\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}<=2.12.1\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.13.0\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.13.1\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.14.0\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.14.1\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.15.0\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.15.0.post1\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.15.1\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.16.1\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.16.2\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.17.0\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.17.1\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.18.0\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.18.1\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.19.0\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.19.1\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.20.0\\n          tensorflow-cpu{sys_platform != \\'darwin\\'}==2.21.0\\n      and tensorflow-cpu{sys_platform != \\'darwin\\'}>=2.12.1 has no wheels\\n      with a matching platform tag (e.g., `manylinux_2_39_aarch64`), we can\\n      conclude that tensorflow-cpu{sys_platform != \\'darwin\\'}>=2.12.1 cannot\\n      be used.\\n      And because colabfold==1.5.4 depends on tensorflow-cpu{sys_platform !=\\n      \\'darwin\\'}>=2.12.1 and you require colabfold[alphafold-minus-jax]==1.5.4,\\n      we can conclude that your requirements are unsatisfiable.\\n\\n      hint: Pre-releases are available for `tensorflow-cpu` in the requested\\n      range (e.g., 2.21.0rc1), but pre-releases weren\\'t enabled (try:\\n      `--prerelease=allow`)\\n\\n      hint: Wheels are available for `tensorflow-cpu` (v2.21.0) on the\\n      following platforms: `manylinux_2_27_x86_64`, `win_amd64`\\n\\nPlease check the colabfold install log saved in /home/bviggiano/.bioemu_colabfold/install_log.txt.\\n\\n']\nE   assert False\nE    +  where False = BioEmuOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success",
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "mock-cli-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-cli-tool-run]",
    "status": "passed",
    "duration_seconds": 2.24,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/mock_cli_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "random-nucleotide-sample",
    "category": "mutagenesis",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[random-nucleotide-sample]",
    "status": "passed",
    "duration_seconds": 0.07,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
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
    "error_message": "('/home/bviggiano/codebases/proto/proto-tools/tests/tool_infra_tests/test_env_report.py', 68, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "chai1-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[chai1-prediction]",
    "status": "failed",
    "duration_seconds": 1.9,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/chai1_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:74: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool chai1-prediction failed: [\"'chai1' may not be compatible with your system. setup.sh failed (exit 1).\\nbash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\\nERROR: Chai is not supported on aarch64.\\nchai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its\\npre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 566, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py\", line 302, in run_chai1\\n    run_chai1_on_complex(comp=comp, config=config, msas=inputs.msas, instance=instance)\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py\", line 388, in run_chai1_on_complex\\n    result = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 404, in dispatch\\n    return cached.run(\\n           ^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 766, in run\\n    return self._run_persistent(\\n           ^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1014, in _run_persistent\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 734, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 2012, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'chai1\\' may not be compatible with your system. setup.sh failed (exit 1).\\nbash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\\nERROR: Chai is not supported on aarch64.\\nchai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its\\npre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\\n']\nE   assert False\nE    +  where False = <[ToolExecutionError('Attempt to access field of tool output after failure: pre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\\n\\nError Messages:\\n\\'chai1\\' may not be compatible with your system. setup.sh failed (exit 1).\\nbash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\\nERROR: Chai is not supported on aarch64.\\nchai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its\\npre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\\nTraceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 566, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py\", line 302, in run_chai1\\n    run_chai1_on_complex(comp=comp, config=config, msas=inputs.msas, instance=instance)\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py\", line 388, in run_chai1_on_complex\\n    result = ToolInst...codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 404, in dispatch\\n    return cached.run(\\n           ^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 766, in run\\n    return self._run_persistent(\\n           ^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1014, in _run_persistent\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 734, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 2012, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'chai1\\' may not be compatible with your system. setup.sh failed (exit 1).\\nbash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\\nERROR: Chai is not supported on aarch64.\\nchai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its\\npre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\\n') raised in repr()] Chai1Output object at 0xf0e934d6fc00>.success",
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "tmalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[tmalign-alignment]",
    "status": "passed",
    "duration_seconds": 7.81,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/tmalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "boltz2-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[boltz2-prediction]",
    "status": "failed",
    "duration_seconds": 52.41,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/boltz2_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:74: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool boltz2-prediction failed: ['Worker for boltz2 returned an error:\\nTraceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/_worker_bootstrap.py\", line 274, in main\\n    result = dispatch(input_dict)\\n             ^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\", line 201, in dispatch\\n    return _model(\\n           ^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\", line 118, in __call__\\n    subprocess.run(\\n  File \"/home/bviggiano/.proto/proto_tool_envs/boltz2_env/lib/python3.12/subprocess.py\", line 571, in run\\n    raise CalledProcessError(retcode, process.args,\\nsubprocess.CalledProcessError: Command \\'[\\'/home/bviggiano/.proto/proto_tool_envs/boltz2_env/bin/boltz\\', \\'predict\\', \\'/tmp/tmp_avx4moy/boltz2_input.yaml\\', \\'--out_dir=/tmp/tmp_avx4moy/boltz2_output\\', \\'--recycling_steps=10\\', \\'--diffusion_samples=25\\', \\'--sampling_steps=200\\', \\'--output_format=mmcif\\', \\'--devices=1\\', \\'--cache=/home/bviggiano/.proto/proto_model_cache/boltz2\\', \\'--num_workers=4\\']\\' returned non-zero exit status 1.\\n', 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 566, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/boltz2.py\", line 309, in run_boltz2\\n    run_boltz2_on_complex(\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/boltz2.py\", line 397, in run_boltz2_on_complex\\n    output_data = ToolInstance.dispatch(\\n                  ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 404, in dispatch\\n    return cached.run(\\n           ^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 766, in run\\n    return self._run_persistent(\\n           ^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1142, in _run_persistent\\n    result = self._worker.send(input_dict, timeout=effective_timeout)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/persistent_worker.py\", line 574, in send\\n    raise RuntimeError(f\"Worker for {self.toolkit} returned an error:\\\\n{response[\\'error\\']}\")\\nRuntimeError: Worker for boltz2 returned an error:\\nTraceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/_worker_bootstrap.py\", line 274, in main\\n    result = dispatch(input_dict)\\n             ^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\", line 201, in dispatch\\n    return _model(\\n           ^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\", line 118, in __call__\\n    subprocess.run(\\n  File \"/home/bviggiano/.proto/proto_tool_envs/boltz2_env/lib/python3.12/subprocess.py\", line 571, in run\\n    raise CalledProcessError(retcode, process.args,\\nsubprocess.CalledProcessError: Command \\'[\\'/home/bviggiano/.proto/proto_tool_envs/boltz2_env/bin/boltz\\', \\'predict\\', \\'/tmp/tmp_avx4moy/boltz2_input.yaml\\', \\'--out_dir=/tmp/tmp_avx4moy/boltz2_output\\', \\'--recycling_steps=10\\', \\'--diffusion_samples=25\\', \\'--sampling_steps=200\\', \\'--output_format=mmcif\\', \\'--devices=1\\', \\'--cache=/home/bviggiano/.proto/proto_model_cache/boltz2\\', \\'--num_workers=4\\']\\' returned non-zero exit status 1.\\n\\n']\nE   assert False\nE    +  where False = <[ToolExecutionError('Attempt to access field of tool output after failure: \\n\\nError Messages:\\nWorker for boltz2 returned an error:\\nTraceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/_worker_bootstrap.py\", line 274, in main\\n    result = dispatch(input_dict)\\n             ^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\", line 201, in dispatch\\n    return _model(\\n           ^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\", line 118, in __call__\\n    subprocess.run(\\n  File \"/home/bviggiano/.proto/proto_tool_envs/boltz2_env/lib/python3.12/subprocess.py\", line 571, in run\\n    raise CalledProcessError(retcode, process.args,\\nsubprocess.CalledProcessError: Command \\'[\\'/home/bviggiano/.proto/proto_tool_envs/boltz2_env/bin/boltz\\', \\'predict\\', \\'/tmp/tmp_avx4moy/boltz2_input.yaml\\', \\'--out_dir=/tmp/tmp_avx4moy/boltz2_output\\', \\'--recycling_steps=10\\', \\'--diffusion_samples=25\\', \\'--sampling_steps=200\\', \\'--output_format=mmcif\\', \\'--devices=1\\', ...\"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/_worker_bootstrap.py\", line 274, in main\\n    result = dispatch(input_dict)\\n             ^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\", line 201, in dispatch\\n    return _model(\\n           ^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\", line 118, in __call__\\n    subprocess.run(\\n  File \"/home/bviggiano/.proto/proto_tool_envs/boltz2_env/lib/python3.12/subprocess.py\", line 571, in run\\n    raise CalledProcessError(retcode, process.args,\\nsubprocess.CalledProcessError: Command \\'[\\'/home/bviggiano/.proto/proto_tool_envs/boltz2_env/bin/boltz\\', \\'predict\\', \\'/tmp/tmp_avx4moy/boltz2_input.yaml\\', \\'--out_dir=/tmp/tmp_avx4moy/boltz2_output\\', \\'--recycling_steps=10\\', \\'--diffusion_samples=25\\', \\'--sampling_steps=200\\', \\'--output_format=mmcif\\', \\'--devices=1\\', \\'--cache=/home/bviggiano/.proto/proto_model_cache/boltz2\\', \\'--num_workers=4\\']\\' returned non-zero exit status 1.\\n\\n') raised in repr()] Boltz2Output object at 0xf0e934db0dc0>.success",
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "orfipy-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[orfipy-prediction]",
    "status": "passed",
    "duration_seconds": 3.27,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/orfipy_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "pdockq2",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pdockq2]",
    "status": "passed",
    "duration_seconds": 0.1,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "enformer-prediction",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[enformer-prediction]",
    "status": "passed",
    "duration_seconds": 20.64,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/enformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "pyrosetta-energy",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pyrosetta-energy]",
    "status": "passed",
    "duration_seconds": 7.79,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/pyrosetta_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
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
    "error_message": "('/home/bviggiano/codebases/proto/proto-tools/tests/tool_infra_tests/test_env_report.py', 68, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "protenix-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[protenix-prediction]",
    "status": "passed",
    "duration_seconds": 184.58,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/protenix_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "evo2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo2-sample]",
    "status": "failed",
    "duration_seconds": 2.07,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/evo2_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:74: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool evo2-sample failed: [\"'evo2' may not be compatible with your system. setup.sh failed (exit 1).\\nbash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\\nERROR: Evo2 is not supported on aarch64.\\nEvo2 requires transformer-engine and flash-attn which only provide x86_64 pre-built wheels.\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 566, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/causal_models/evo2/evo2_sample.py\", line 259, in run_evo2_sample\\n    result = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 422, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 461, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1173, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 734, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 2012, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'evo2\\' may not be compatible with your system. setup.sh failed (exit 1).\\nbash: /home/bviggiano/miniconda3/envs/proto-tools/lib/libtinfo.so.6: no version information available (required by bash)\\nERROR: Evo2 is not supported on aarch64.\\nEvo2 requires transformer-engine and flash-attn which only provide x86_64 pre-built wheels.\\n']\nE   assert False\nE    +  where False = Evo2SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, logits, kv_caches).success",
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "minced-crispr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[minced-crispr]",
    "status": "passed",
    "duration_seconds": 3.1,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/minced_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "evo1-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo1-sample]",
    "status": "failed",
    "duration_seconds": 50.02,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/evo1_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:74: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool evo1-sample failed: ['\\'evo1\\' may not be compatible with your system. setup.sh failed (exit 1).\\n      on `psutil`, but doesn\\'t declare it as a build dependency. If\\n      `flash-attn` is a first-party package, consider adding `psutil`\\n      to its `build-system.requires`. Otherwise, either add it to your\\n      `pyproject.toml` under:\\n      [tool.uv.extra-build-dependencies]\\n      flash-attn = [\"psutil\"]\\n      or `uv pip install psutil` into the environment and re-run with\\n      `--no-build-isolation`.\\n  help: `flash-attn` (v2.8.3) was included because `evo-model` (v0.5) depends\\n        on `stripedhyena` (v0.2.2) which depends on `flash-attn`', 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 566, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/causal_models/evo1/evo1_sample.py\", line 151, in run_evo1_sample\\n    result = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 422, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 461, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1173, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 734, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 2012, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'evo1\\' may not be compatible with your system. setup.sh failed (exit 1).\\n      on `psutil`, but doesn\\'t declare it as a build dependency. If\\n      `flash-attn` is a first-party package, consider adding `psutil`\\n      to its `build-system.requires`. Otherwise, either add it to your\\n      `pyproject.toml` under:\\n      [tool.uv.extra-build-dependencies]\\n      flash-attn = [\"psutil\"]\\n      or `uv pip install psutil` into the environment and re-run with\\n      `--no-build-isolation`.\\n  help: `flash-attn` (v2.8.3) was included because `evo-model` (v0.5) depends\\n        on `stripedhyena` (v0.2.2) which depends on `flash-attn`\\n']\nE   assert False\nE    +  where False = Evo1SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, scores).success",
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "segmasker-score",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[segmasker-score]",
    "status": "failed",
    "duration_seconds": 64.8,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/segmasker_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:74: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool segmasker-score failed: [\"Command '['/home/bviggiano/.proto/proto_tool_envs/segmasker_env/bin/python', '/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_scoring/segmasker/standalone/run.py', '/tmp/tmpbyg9b7ll/input.json', '/tmp/tmpbyg9b7ll/output.json']' returned non-zero exit status 1.\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 566, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_scoring/segmasker/segmasker.py\", line 234, in run_segmasker\\n    output_data = ToolInstance.dispatch(\\n                  ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 422, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 468, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1215, in _run_oneshot\\n    subprocess.run(\\n  File \"/home/bviggiano/miniconda3/envs/proto-tools/lib/python3.12/subprocess.py\", line 571, in run\\n    raise CalledProcessError(retcode, process.args,\\nsubprocess.CalledProcessError: Command \\'[\\'/home/bviggiano/.proto/proto_tool_envs/segmasker_env/bin/python\\', \\'/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_scoring/segmasker/standalone/run.py\\', \\'/tmp/tmpbyg9b7ll/input.json\\', \\'/tmp/tmpbyg9b7ll/output.json\\']\\' returned non-zero exit status 1.\\n']\nE   assert False\nE    +  where False = SegmaskerOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, results).success",
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "fampnn-pack",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[fampnn-pack]",
    "status": "passed",
    "duration_seconds": 57.94,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/fampnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "ablang-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ablang-embedding]",
    "status": "passed",
    "duration_seconds": 272.38,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/ablang_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "111bab7e0784",
    "git_dirty": true
  },
  {
    "tool_key": "mock-jax-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-jax-tool-run]",
    "status": "passed",
    "duration_seconds": 9.28,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/mock_jax_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  },
  {
    "tool_key": "esm2-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm2-embedding]",
    "status": "passed",
    "duration_seconds": 26.45,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/esm2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "72cc6d679fc3",
    "git_dirty": false
  }
]
-->