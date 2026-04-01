# DGX Spark Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-71%25-yellow) ![Passed](https://img.shields.io/badge/passed-28-brightgreen) ![Failed](https://img.shields.io/badge/failed-11-red) ![Skipped](https://img.shields.io/badge/skipped-4-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Linux Linux 6.11.0-1016-nvidia |
| **Architecture** | aarch64 |
| **Hostname** | `spark-3b18` |
| **Python** | 3.12.13 |
| **RAM** | 119.7 GB |
| **GPU** | 1× NVIDIA GB10 |
| **CUDA** | 13.0 |
| **Conda Env** | `proto-tools` |

## Git

- **Commit**: `f36a85cdf12b`
- **Branch**: `refactor/env-report-autodiscovery`
- **Dirty**: Yes

## Environment Variables

### Parent Process Environment

```
CLAUDECODE=1
CLAUDE_CODE_ENTRYPOINT=cli
CONDA_DEFAULT_ENV=proto-tools
CONDA_EXE=/home/bviggiano/miniconda3/bin/conda
CONDA_PREFIX=/home/bviggiano/miniconda3/envs/proto-tools
CONDA_PREFIX_1=/home/bviggiano/miniconda3
CONDA_PREFIX_2=/home/bviggiano/miniconda3/envs/proto-tools
CONDA_PREFIX_3=/home/bviggiano/miniconda3
CONDA_PREFIX_4=/home/bviggiano/miniconda3/envs/proto-tools
CONDA_PREFIX_5=/home/bviggiano/miniconda3
CONDA_PROMPT_MODIFIER=(proto-tools) 
CONDA_PYTHON_EXE=/home/bviggiano/miniconda3/bin/python
CONDA_SHLVL=6
COREPACK_ENABLE_AUTO_PIN=0
DEBUGINFOD_URLS=https://debuginfod.ubuntu.com 
DISABLE_PANDERA_IMPORT_WARNING=True
GIT_EDITOR=true
GSETTINGS_SCHEMA_DIR=/home/bviggiano/miniconda3/envs/proto-tools/share/glib-2.0/schemas
GSETTINGS_SCHEMA_DIR_CONDA_BACKUP=
HOME=/home/bviggiano
LANG=en_US.utf8
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64:
LESSCLOSE=/usr/bin/lesspipe %s %s
LESSOPEN=| /usr/bin/lesspipe %s
LOGNAME=bviggiano
LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=00:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.arj=...
NoDefaultCurrentDirectoryInExePath=1
OLDPWD=/home/bviggiano/codebases/proto-bio
OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta
PATH=/home/bviggiano/.local/bin:/home/bviggiano/.local/bin:/usr/local/cuda/bin:/usr/local/cuda/bin:/opt/bin:/home/bviggiano/.local/bin:/usr/local/cuda/bin:/home/bviggiano/.local/bin:/usr/local/cuda/bin:/ho...
PROTO_HOME=/home/bviggiano/codebases/proto-bio
PWD=/home/bviggiano/codebases/proto-bio/proto-tools
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
RDBASE=/home/bviggiano/miniconda3/envs/proto-tools/lib/python3.12/site-packages/rdkit
SHELL=/bin/bash
SHLVL=3
TERM=tmux-256color
TERM_PROGRAM=tmux
TERM_PROGRAM_VERSION=3.4
TMUX=/tmp/tmux-1001/default,3233358,0
TMUX_PANE=%0
USER=bviggiano
XDG_DATA_DIRS=/usr/share/gnome:/usr/local/share:/usr/share:/var/lib/snapd/desktop
XDG_RUNTIME_DIR=/run/user/1001
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
CONDA_PREFIX=/home/bviggiano/codebases/proto-bio/proto_tool_envs/viennarna_env
DETECTED_COMPUTE_PLATFORM=cuda
DETECTED_CUDA_VERSION=13
DETECTED_DRIVER_VERSION=580
HF_HOME=/home/bviggiano/codebases/proto-bio/proto_model_cache/huggingface
HOME=/home/bviggiano
LANG=en_US.utf8
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/home/bviggiano/miniconda3/envs/proto-tools/lib
LOGNAME=bviggiano
PATH=/home/bviggiano/codebases/proto-bio/proto_tool_envs/viennarna_env/bin:/home/bviggiano/.local/bin:/usr/local/cuda/bin:/opt/bin:/home/bviggiano/miniconda3/envs/proto-tools/bin:/home/bviggiano/miniconda3...
PIP_DEFAULT_TIMEOUT=300
PROTO_HOME=/home/bviggiano/codebases/proto-bio
RECOMMENDED_JAX_SPEC=jax[cuda13]>=0.4.20,<1
RECOMMENDED_JAX_VARIANT=cuda13
RECOMMENDED_TORCH_INDEX=https://download.pytorch.org/whl/cu128
RECOMMENDED_TORCH_SPEC=torch>=2.8,<3
SHELL=/bin/bash
TORCH_CUDA_ARCH_LIST=12.0
TORCH_HOME=/home/bviggiano/codebases/proto-bio/proto_model_cache/torch
USER=bviggiano
UV_HTTP_TIMEOUT=300
VIRTUAL_ENV=/home/bviggiano/codebases/proto-bio/proto_tool_envs/viennarna_env
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Results by Category

### Causal Models (0/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `evo1-sample` | yes | ✅ | 123.2s | `f36a85c` ✱ | ❌ Fail |
| `evo2-sample` | yes | ✅ | 8.0s | `f36a85c` ✱ | ❌ Fail |
| `progen2-sample` | yes | ✅ | 7.8s | `f36a85c` ✱ | ❌ Fail |

### Gene Annotation (4/5)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `blast-create-db` | no | ✅ | 77.3s | `f36a85c` ✱ | ✅ Pass |
| `crispr-tracr` | no | ✅ | 30.1s | `f36a85c` ✱ | ❌ Fail |
| `minced-crispr` | no | ✅ | 11.3s | `f36a85c` ✱ | ✅ Pass |
| `mmseqs-clustering` | no | ✅ | 14.3s | `f36a85c` ✱ | ✅ Pass |
| `pyhmmer-hmmscan` | no | ✅ | 24.7s | `f36a85c` ✱ | ✅ Pass |

### Inverse Folding (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `esm-if1-sample` | yes | ✅ | 202.6s | `f36a85c` ✱ | ✅ Pass |
| `fampnn-pack` | yes | ✅ | 127.8s | `f36a85c` ✱ | ✅ Pass |
| `ligandmpnn-sample` | yes | ✅ | 68.4s | `f36a85c` ✱ | ✅ Pass |
| `proteinmpnn-sample` | yes | ✅ | 64.0s | `f36a85c` ✱ | ✅ Pass |

### Masked Models (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `esm2-embedding` | yes | ✅ | 84.2s | `f36a85c` ✱ | ✅ Pass |
| `esm3-embedding` | yes | ✅ | 117.2s | `f36a85c` ✱ | ✅ Pass |

### Mutagenesis (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `random-nucleotide-sample` | no | — | 0.0s | `f36a85c` ✱ | ✅ Pass |
| `random-protein-sample` | no | — | 0.0s | `f36a85c` ✱ | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `orfipy-prediction` | no | ✅ | 12.1s | `f36a85c` ✱ | ✅ Pass |
| `prodigal-prediction` | no | ✅ | 11.5s | `f36a85c` ✱ | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `splice-transformer-prediction` | yes | ✅ | 39.7s | `f36a85c` ✱ | ✅ Pass |

### Sequence Alignment (1/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `colabfold-search` | no | ✅ | 26.3s | `f36a85c` ✱ | ✅ Pass |
| `mafft-align` | no | ✅ | 24.2s | `f36a85c` ✱ | ❌ Fail |

### Sequence Scoring (3/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphagenome-predict-intervals` | yes | ✅ | 1434.1s | `f36a85c` | ✅ Pass |
| `borzoi-ensemble` | yes | ✅ | 238.1s | `f36a85c` ✱ | ✅ Pass |
| `enformer-prediction` | yes | ✅ | 57.8s | `f36a85c` ✱ | ✅ Pass |
| `segmasker-score` | no | ✅ | 114.9s | `f36a85c` ✱ | ❌ Fail |

### Structure Alignment (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `tmalign-alignment` | no | ✅ | 41.7s | `f36a85c` ✱ | ✅ Pass |
| `usalign-alignment` | no | ✅ | 78.9s | `f36a85c` ✱ | ✅ Pass |

### Structure Design (0/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `rfdiffusion3-design` | yes | ✅ | 3922.6s | `f36a85c` ✱ | ❌ Fail |

### Structure Dynamics (0/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `bioemu-sample` | yes | ✅ | 139.0s | `f36a85c` | ❌ Fail |

### Structure Prediction (4/7)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphafold2-prediction` | yes | ✅ | 298.5s | `f36a85c` | ✅ Pass |
| `alphafold3-prediction` | yes | — | — | `f36a85c` | ⏭️ Skip |
| `boltz2-prediction` | yes | ✅ | 270.6s | `f36a85c` ✱ | ❌ Fail |
| `chai1-prediction` | yes | ✅ | 7.8s | `f36a85c` ✱ | ❌ Fail |
| `esmfold-prediction` | yes | ✅ | 220.6s | `f36a85c` ✱ | ✅ Pass |
| `protenix-prediction` | yes | ✅ | 1075.6s | `f36a85c` ✱ | ❌ Fail |
| `structure-metrics` | no | ✅ | 14.6s | `f36a85c` ✱ | ✅ Pass |
| `viennarna-prediction` | no | ✅ | 11.9s | `f36a85c` ✱ | ✅ Pass |

### Testing (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `mock-cli-multi-gpu-tool-run` | yes | — | — | `f36a85c` ✱ | ⏭️ Skip |
| `mock-cli-tool-run` | yes | ✅ | 7.6s | `f36a85c` ✱ | ✅ Pass |
| `mock-jax-multi-gpu-tool-run` | yes | — | — | `f36a85c` ✱ | ⏭️ Skip |
| `mock-jax-tool-run` | yes | ✅ | 39.4s | `f36a85c` ✱ | ✅ Pass |
| `mock-pytorch-multi-gpu-tool-run` | yes | — | — | `f36a85c` ✱ | ⏭️ Skip |
| `mock-pytorch-tool-run` | yes | ✅ | 41.2s | `f36a85c` ✱ | ✅ Pass |

## Failure Details

### ❌ `bioemu-sample`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[bioemu-sample]`

```
tests/tool_infra_tests/test_env_report.py:119: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool bioemu-sample failed: ["'bioemu' may not be compatible with your system. setup.sh failed (exit 1).\n      conclude that tensorflow-cpu{sys_platform != 'darwin'}>=2.12.1 cannot\n      be used.\n      And because colabfold==1.5.4 depends on tensorflow-cpu{sys_platform !=\n      'darwin'}>=2.12.1 and you require colabfold[alphafold-minus-jax]==1.5.4,\n      we can conclude that your requirements are unsatisfiable.\n      hint: Pre-releases are available for `tensorflow-cpu` in the requested\n      range (e.g., 2.21.0rc1), but pre-releases weren't enabled (try:\n      `--prerelease=allow`)\n      hint: Wheels are available for `tensorflow-cpu` (v2.21.0) on the\n      following platforms: `manylinux_2_27_x86_64`, `win_amd64`", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py", line 428, in wrapper\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_dynamics/bioemu/bioemu_sample.py", line 266, in run_bioemu\n    output = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 254, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 293, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 968, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 504, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 1404, in _create_env\n    raise RuntimeError(\nRuntimeError: \'bioemu\' may not be compatible with your system. setup.sh failed (exit 1).\n      conclude that tensorflow-cpu{sys_platform != \'darwin\'}>=2.12.1 cannot\n      be used.\n      And because colabfold==1.5.4 depends on tensorflow-cpu{sys_platform !=\n      \'darwin\'}>=2.12.1 and you require colabfold[alphafold-minus-jax]==1.5.4,\n      we can conclude that your requirements are unsatisfiable.\n      hint: Pre-releases are available for `tensorflow-cpu` in the requested\n      range (e.g., 2.21.0rc1), but pre-releases weren\'t enabled (try:\n      `--prerelease=allow`)\n      hint: Wheels are available for `tensorflow-cpu` (v2.21.0) on the\n      following platforms: `manylinux_2_27_x86_64`, `win_amd64`\n']
E   assert False
E    +  where False = BioEmuOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success
```

### ❌ `boltz2-prediction`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[boltz2-prediction]`

```
tests/tool_infra_tests/test_env_report.py:119: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool boltz2-prediction failed: ["Command '['/home/bviggiano/codebases/proto-bio/proto_tool_envs/boltz2_env/bin/python', '/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py', '/tmp/tmpx1yy8k0q/input.json', '/tmp/tmpx1yy8k0q/output.json']' returned non-zero exit status 1.", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py", line 428, in wrapper\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/boltz2/boltz2.py", line 274, in run_boltz2\n    run_boltz2_on_complex(\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/boltz2/boltz2.py", line 382, in run_boltz2_on_complex\n    output_data = ToolInstance.dispatch(\n                  ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 254, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 293, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 999, in _run_oneshot\n    subprocess.run(\n  File "/home/bviggiano/miniconda3/envs/proto-tools/lib/python3.12/subprocess.py", line 571, in run\n    raise CalledProcessError(retcode, process.args,\nsubprocess.CalledProcessError: Command \'[\'/home/bviggiano/codebases/proto-bio/proto_tool_envs/boltz2_env/bin/python\', \'/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\', \'/tmp/tmpx1yy8k0q/input.json\', \'/tmp/tmpx1yy8k0q/output.json\']\' returned non-zero exit status 1.\n']
E   assert False
E    +  where False = <[ToolExecutionError('Attempt to access field of tool output after failure: subprocess.CalledProcessError: Command \'[\'/home/bviggiano/codebases/proto-bio/proto_tool_envs/boltz2_env/bin/python\', \'/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\', \'/tmp/tmpx1yy8k0q/input.json\', \'/tmp/tmpx1yy8k0q/output.json\']\' returned non-zero exit status 1.\n\nError Messages:\nCommand \'[\'/home/bviggiano/codebases/proto-bio/proto_tool_envs/boltz2_env/bin/python\', \'/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\', \'/tmp/tmpx1yy8k0q/input.json\', \'/tmp/tmpx1yy8k0q/output.json\']\' returned non-zero exit status 1.\nTraceback (most recent call last):\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py", line 428, in wrapper\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/boltz2/boltz2.py", line 274, in run_boltz2\n    run_boltz2_on_complex(\n  File "/home/bviggiano/codebase...boltz2/boltz2.py", line 382, in run_boltz2_on_complex\n    output_data = ToolInstance.dispatch(\n                  ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 254, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 293, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 999, in _run_oneshot\n    subprocess.run(\n  File "/home/bviggiano/miniconda3/envs/proto-tools/lib/python3.12/subprocess.py", line 571, in run\n    raise CalledProcessError(retcode, process.args,\nsubprocess.CalledProcessError: Command \'[\'/home/bviggiano/codebases/proto-bio/proto_tool_envs/boltz2_env/bin/python\', \'/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\', \'/tmp/tmpx1yy8k0q/input.json\', \'/tmp/tmpx1yy8k0q/output.json\']\' returned non-zero exit status 1.\n') raised in repr()] StructurePredictionOutput object at 0xe68f8a65ba70>.success
```

### ❌ `chai1-prediction`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[chai1-prediction]`

```
tests/tool_infra_tests/test_env_report.py:119: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool chai1-prediction failed: ["'chai1' may not be compatible with your system. setup.sh failed (exit 1).\nERROR: Chai is not supported on aarch64.\nchai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its\npre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py", line 428, in wrapper\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py", line 290, in run_chai1\n    results.append(run_chai1_on_complex(comp=comp, config=config, msas=inputs.msas, instance=instance))\n                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py", line 377, in run_chai1_on_complex\n    result = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 254, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 293, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 968, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 504, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 1404, in _create_env\n    raise RuntimeError(\nRuntimeError: \'chai1\' may not be compatible with your system. setup.sh failed (exit 1).\nERROR: Chai is not supported on aarch64.\nchai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its\npre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\n']
E   assert False
E    +  where False = <[ToolExecutionError('Attempt to access field of tool output after failure: pre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\n\nError Messages:\n\'chai1\' may not be compatible with your system. setup.sh failed (exit 1).\nERROR: Chai is not supported on aarch64.\nchai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its\npre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\nTraceback (most recent call last):\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py", line 428, in wrapper\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py", line 290, in run_chai1\n    results.append(run_chai1_on_complex(comp=comp, config=config, msas=inputs.msas, instance=instance))\n                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py", line 377, in run_chai1_on_complex\n    result = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 254, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 293, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 968, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 504, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 1404, in _create_env\n    raise RuntimeError(\nRuntimeError: \'chai1\' may not be compatible with your system. setup.sh failed (exit 1).\nERROR: Chai is not supported on aarch64.\nchai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its\npre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\n') raised in repr()] StructurePredictionOutput object at 0xe68f8a65b9d0>.success
```

### ❌ `crispr-tracr`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[crispr-tracr]`

```
tests/tool_infra_tests/test_env_report.py:119: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool crispr-tracr failed: ["'crispr_tracr' may not be compatible with your system. setup.sh failed (exit 1).\nCloning into '/home/bviggiano/codebases/proto-bio/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA'...\nCloning CRISPRidentify into CRISPRtracrRNA tools directory...\nCloning into '/home/bviggiano/codebases/proto-bio/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA/tools/CRISPRidentify/CRISPRidentify'...\nCloning CRISPRcasIdentifier into CRISPRtracrRNA tools directory...\nCloning into '/home/bviggiano/codebases/proto-bio/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA/tools/CRISPRcasIdentifier/CRISPRcasIdentifier'...\nCreating isolated conda environment (Python 3.8 + scikit-learn 0.22)...\nCRISPRidentify's pickled models require sklearn 0.22 (incompatible with 3.12).\nUsing /home/bviggiano/codebases/proto-bio/proto_tool_envs/crispr_tracr_env/conda_deps to avoid polluting base env...\nERROR: CRISPRtracrRNA requires x86_64 bioconda packages (vmatch, etc.)\n       that are not available on Linux aarch64.", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py", line 428, in wrapper\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/gene_annotation/crispr_tracr/crispr_tracr.py", line 220, in run_crispr_tracr\n    output_data = ToolInstance.dispatch(\n                  ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 254, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 300, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 968, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 504, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 1404, in _create_env\n    raise RuntimeError(\nRuntimeError: \'crispr_tracr\' may not be compatible with your system. setup.sh failed (exit 1).\nCloning into \'/home/bviggiano/codebases/proto-bio/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA\'...\nCloning CRISPRidentify into CRISPRtracrRNA tools directory...\nCloning into \'/home/bviggiano/codebases/proto-bio/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA/tools/CRISPRidentify/CRISPRidentify\'...\nCloning CRISPRcasIdentifier into CRISPRtracrRNA tools directory...\nCloning into \'/home/bviggiano/codebases/proto-bio/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA/tools/CRISPRcasIdentifier/CRISPRcasIdentifier\'...\nCreating isolated conda environment (Python 3.8 + scikit-learn 0.22)...\nCRISPRidentify\'s pickled models require sklearn 0.22 (incompatible with 3.12).\nUsing /home/bviggiano/codebases/proto-bio/proto_tool_envs/crispr_tracr_env/conda_deps to avoid polluting base env...\nERROR: CRISPRtracrRNA requires x86_64 bioconda packages (vmatch, etc.)\n       that are not available on Linux aarch64.\n']
E   assert False
E    +  where False = CrisprTracrOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, predictions).success
```

### ❌ `evo1-sample`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo1-sample]`

```
tests/tool_infra_tests/test_env_report.py:119: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool evo1-sample failed: ['\'evo1\' may not be compatible with your system. setup.sh failed (exit 1).\n      on `psutil`, but doesn\'t declare it as a build dependency. If\n      `flash-attn` is a first-party package, consider adding `psutil`\n      to its `build-system.requires`. Otherwise, either add it to your\n      `pyproject.toml` under:\n      [tool.uv.extra-build-dependencies]\n      flash-attn = ["psutil"]\n      or `uv pip install psutil` into the environment and re-run with\n      `--no-build-isolation`.\n  help: `flash-attn` (v2.8.3) was included because `evo-model` (v0.5) depends\n        on `stripedhyena` (v0.2.2) which depends on `flash-attn`', 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py", line 428, in wrapper\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/causal_models/evo1/evo1_sample.py", line 219, in run_evo1_sample\n    result = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 254, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 293, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 968, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 504, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 1404, in _create_env\n    raise RuntimeError(\nRuntimeError: \'evo1\' may not be compatible with your system. setup.sh failed (exit 1).\n      on `psutil`, but doesn\'t declare it as a build dependency. If\n      `flash-attn` is a first-party package, consider adding `psutil`\n      to its `build-system.requires`. Otherwise, either add it to your\n      `pyproject.toml` under:\n      [tool.uv.extra-build-dependencies]\n      flash-attn = ["psutil"]\n      or `uv pip install psutil` into the environment and re-run with\n      `--no-build-isolation`.\n  help: `flash-attn` (v2.8.3) was included because `evo-model` (v0.5) depends\n        on `stripedhyena` (v0.2.2) which depends on `flash-attn`\n']
E   assert False
E    +  where False = Evo1SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, scores).success
```

### ❌ `evo2-sample`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo2-sample]`

```
tests/tool_infra_tests/test_env_report.py:119: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool evo2-sample failed: ["'evo2' may not be compatible with your system. setup.sh failed (exit 1).\nERROR: Evo2 is not supported on aarch64.\nEvo2 requires transformer-engine and flash-attn which only provide x86_64 pre-built wheels.", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py", line 428, in wrapper\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/causal_models/evo2/evo2_sample.py", line 436, in run_evo2_sample\n    result = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 254, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 293, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 968, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 504, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 1404, in _create_env\n    raise RuntimeError(\nRuntimeError: \'evo2\' may not be compatible with your system. setup.sh failed (exit 1).\nERROR: Evo2 is not supported on aarch64.\nEvo2 requires transformer-engine and flash-attn which only provide x86_64 pre-built wheels.\n']
E   assert False
E    +  where False = Evo2SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, logits, kv_caches).success
```

### ❌ `mafft-align`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mafft-align]`

```
tests/tool_infra_tests/test_env_report.py:119: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool mafft-align failed: ['\'mafft\' may not be compatible with your system. setup.sh failed (exit 1).\n    install_binary(sys.argv[1])\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/install_binary.py", line 188, in install_binary\n    config.extract(archive_path, bin_dir)\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_alignment/mafft/standalone/binary_config.py", line 165, in extract\n    _extract_from_source(archive_path, bin_dir)\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_alignment/mafft/standalone/binary_config.py", line 134, in _extract_from_source\n    subprocess.check_call(\n  File "/home/bviggiano/codebases/proto-bio/proto_tool_envs/mafft_env/lib/python3.12/subprocess.py", line 413, in check_call\n    raise CalledProcessError(retcode, cmd)\nsubprocess.CalledProcessError: Command \'[\'make\', \'-j4\', \'PREFIX=/home/bviggiano/codebases/proto-bio/proto_tool_envs/mafft_env\', \'BINDIR=/home/bviggiano/codebases/proto-bio/proto_tool_envs/mafft_env/bin\', \'LIBDIR=/home/bviggiano/codebases/proto-bio/proto_tool_envs/mafft_env/libexec/mafft\']\' returned non-zero exit status 2.', 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py", line 428, in wrapper\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_alignment/mafft/mafft.py", line 199, in run_mafft_align\n    output_data = ToolInstance.dispatch(\n                  ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 254, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 300, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 968, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 504, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 1404, in _create_env\n    raise RuntimeError(\nRuntimeError: \'mafft\' may not be compatible with your system. setup.sh failed (exit 1).\n    install_binary(sys.argv[1])\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/install_binary.py", line 188, in install_binary\n    config.extract(archive_path, bin_dir)\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_alignment/mafft/standalone/binary_config.py", line 165, in extract\n    _extract_from_source(archive_path, bin_dir)\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_alignment/mafft/standalone/binary_config.py", line 134, in _extract_from_source\n    subprocess.check_call(\n  File "/home/bviggiano/codebases/proto-bio/proto_tool_envs/mafft_env/lib/python3.12/subprocess.py", line 413, in check_call\n    raise CalledProcessError(retcode, cmd)\nsubprocess.CalledProcessError: Command \'[\'make\', \'-j4\', \'PREFIX=/home/bviggiano/codebases/proto-bio/proto_tool_envs/mafft_env\', \'BINDIR=/home/bviggiano/codebases/proto-bio/proto_tool_envs/mafft_env/bin\', \'LIBDIR=/home/bviggiano/codebases/proto-bio/proto_tool_envs/mafft_env/libexec/mafft\']\' returned non-zero exit status 2.\n']
E   assert False
E    +  where False = MafftOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success
```

### ❌ `progen2-sample`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen2-sample]`

```
tests/tool_infra_tests/test_env_report.py:119: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool progen2-sample failed: ["'progen2' may not be compatible with your system. setup.sh failed (exit 1).\nERROR: ProGen2 is not supported on aarch64.\nProGen2 pins torch==2.2.2 which has no aarch64 CUDA wheel available.", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py", line 428, in wrapper\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/causal_models/progen2/progen2_sample.py", line 379, in run_progen2_sample\n    result = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 254, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 293, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 968, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 504, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 1404, in _create_env\n    raise RuntimeError(\nRuntimeError: \'progen2\' may not be compatible with your system. setup.sh failed (exit 1).\nERROR: ProGen2 is not supported on aarch64.\nProGen2 pins torch==2.2.2 which has no aarch64 CUDA wheel available.\n']
E   assert False
E    +  where False = ProGen2SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, logits).success
```

### ❌ `protenix-prediction`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[protenix-prediction]`

```
tests/tool_infra_tests/test_env_report.py:119: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool protenix-prediction failed: ["Command '['/home/bviggiano/codebases/proto-bio/proto_tool_envs/protenix_env/bin/python', '/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/protenix/standalone/inference.py', '/tmp/tmpfsxhwxj2/input.json', '/tmp/tmpfsxhwxj2/output.json']' returned non-zero exit status 1.", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py", line 428, in wrapper\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/protenix/protenix.py", line 392, in run_protenix\n    output_data = ToolInstance.dispatch(\n                  ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 254, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 293, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 999, in _run_oneshot\n    subprocess.run(\n  File "/home/bviggiano/miniconda3/envs/proto-tools/lib/python3.12/subprocess.py", line 571, in run\n    raise CalledProcessError(retcode, process.args,\nsubprocess.CalledProcessError: Command \'[\'/home/bviggiano/codebases/proto-bio/proto_tool_envs/protenix_env/bin/python\', \'/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/protenix/standalone/inference.py\', \'/tmp/tmpfsxhwxj2/input.json\', \'/tmp/tmpfsxhwxj2/output.json\']\' returned non-zero exit status 1.\n']
E   assert False
E    +  where False = <[ToolExecutionError('Attempt to access field of tool output after failure: subprocess.CalledProcessError: Command \'[\'/home/bviggiano/codebases/proto-bio/proto_tool_envs/protenix_env/bin/python\', \'/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/protenix/standalone/inference.py\', \'/tmp/tmpfsxhwxj2/input.json\', \'/tmp/tmpfsxhwxj2/output.json\']\' returned non-zero exit status 1.\n\nError Messages:\nCommand \'[\'/home/bviggiano/codebases/proto-bio/proto_tool_envs/protenix_env/bin/python\', \'/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/protenix/standalone/inference.py\', \'/tmp/tmpfsxhwxj2/input.json\', \'/tmp/tmpfsxhwxj2/output.json\']\' returned non-zero exit status 1.\nTraceback (most recent call last):\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py", line 428, in wrapper\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/protenix/protenix.py", line 392, in run_protenix\n    output_data = ToolInstance.dispatch(\n                  ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 254, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 293, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 999, in _run_oneshot\n    subprocess.run(\n  File "/home/bviggiano/miniconda3/envs/proto-tools/lib/python3.12/subprocess.py", line 571, in run\n    raise CalledProcessError(retcode, process.args,\nsubprocess.CalledProcessError: Command \'[\'/home/bviggiano/codebases/proto-bio/proto_tool_envs/protenix_env/bin/python\', \'/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/protenix/standalone/inference.py\', \'/tmp/tmpfsxhwxj2/input.json\', \'/tmp/tmpfsxhwxj2/output.json\']\' returned non-zero exit status 1.\n') raised in repr()] StructurePredictionOutput object at 0xe68f8991a4e0>.success
```

### ❌ `rfdiffusion3-design`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[rfdiffusion3-design]`

```
tests/tool_infra_tests/test_env_report.py:119: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool rfdiffusion3-design failed: ['Tool rfdiffusion3 timed out after 3600s', 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py", line 428, in wrapper\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_design/rfdiffusion3/rfdiffusion3_sample.py", line 598, in run_rfdiffusion3\n    output_data = ToolInstance.dispatch(\n                  ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 254, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 293, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 1016, in _run_oneshot\n    raise TimeoutError(\nTimeoutError: Tool rfdiffusion3 timed out after 3600s\n']
E   assert False
E    +  where False = RFdiffusion3Output(output_structures=[0 structures]).success
```

### ❌ `segmasker-score`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[segmasker-score]`

```
tests/tool_infra_tests/test_env_report.py:119: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool segmasker-score failed: ["Command '['/home/bviggiano/codebases/proto-bio/proto_tool_envs/segmasker_env/bin/python', '/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_scoring/segmasker/standalone/run.py', '/tmp/tmp144b8nhh/input.json', '/tmp/tmp144b8nhh/output.json']' returned non-zero exit status 1.", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py", line 428, in wrapper\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_scoring/segmasker/segmasker.py", line 257, in run_segmasker\n    output_data = ToolInstance.dispatch(\n                  ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 254, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 300, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py", line 999, in _run_oneshot\n    subprocess.run(\n  File "/home/bviggiano/miniconda3/envs/proto-tools/lib/python3.12/subprocess.py", line 571, in run\n    raise CalledProcessError(retcode, process.args,\nsubprocess.CalledProcessError: Command \'[\'/home/bviggiano/codebases/proto-bio/proto_tool_envs/segmasker_env/bin/python\', \'/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_scoring/segmasker/standalone/run.py\', \'/tmp/tmp144b8nhh/input.json\', \'/tmp/tmp144b8nhh/output.json\']\' returned non-zero exit status 1.\n']
E   assert False
E    +  where False = SegmaskerOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, results_df).success
```

---
*Generated at 2026-03-31 18:31:19 by `pytest --env-report`*

<!-- env-report-data
[
  {
    "tool_name": "alphafold2-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold2-prediction]",
    "status": "passed",
    "duration_seconds": 298.47,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/alphafold2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
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
    "error_message": "('/home/bviggiano/codebases/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 96, 'Skipped: --env-report: requires Chimera cluster')",
    "git_commit": "f36a85cdf12b",
    "git_dirty": false
  },
  {
    "tool_name": "alphagenome-predict-intervals",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphagenome-predict-intervals]",
    "status": "passed",
    "duration_seconds": 1434.12,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/alphagenome_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
    "git_dirty": false
  },
  {
    "tool_name": "bioemu-sample",
    "category": "structure_dynamics",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[bioemu-sample]",
    "status": "failed",
    "duration_seconds": 138.99,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/bioemu_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:119: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool bioemu-sample failed: [\"'bioemu' may not be compatible with your system. setup.sh failed (exit 1).\\n      conclude that tensorflow-cpu{sys_platform != 'darwin'}>=2.12.1 cannot\\n      be used.\\n      And because colabfold==1.5.4 depends on tensorflow-cpu{sys_platform !=\\n      'darwin'}>=2.12.1 and you require colabfold[alphafold-minus-jax]==1.5.4,\\n      we can conclude that your requirements are unsatisfiable.\\n      hint: Pre-releases are available for `tensorflow-cpu` in the requested\\n      range (e.g., 2.21.0rc1), but pre-releases weren't enabled (try:\\n      `--prerelease=allow`)\\n      hint: Wheels are available for `tensorflow-cpu` (v2.21.0) on the\\n      following platforms: `manylinux_2_27_x86_64`, `win_amd64`\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py\", line 428, in wrapper\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_dynamics/bioemu/bioemu_sample.py\", line 266, in run_bioemu\\n    output = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 254, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 293, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 968, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 504, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 1404, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'bioemu\\' may not be compatible with your system. setup.sh failed (exit 1).\\n      conclude that tensorflow-cpu{sys_platform != \\'darwin\\'}>=2.12.1 cannot\\n      be used.\\n      And because colabfold==1.5.4 depends on tensorflow-cpu{sys_platform !=\\n      \\'darwin\\'}>=2.12.1 and you require colabfold[alphafold-minus-jax]==1.5.4,\\n      we can conclude that your requirements are unsatisfiable.\\n      hint: Pre-releases are available for `tensorflow-cpu` in the requested\\n      range (e.g., 2.21.0rc1), but pre-releases weren\\'t enabled (try:\\n      `--prerelease=allow`)\\n      hint: Wheels are available for `tensorflow-cpu` (v2.21.0) on the\\n      following platforms: `manylinux_2_27_x86_64`, `win_amd64`\\n']\nE   assert False\nE    +  where False = BioEmuOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success",
    "git_commit": "f36a85cdf12b",
    "git_dirty": false
  },
  {
    "tool_name": "blast-create-db",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[blast-create-db]",
    "status": "passed",
    "duration_seconds": 77.33,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/blast_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "boltz2-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[boltz2-prediction]",
    "status": "failed",
    "duration_seconds": 270.6,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/boltz2_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:119: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool boltz2-prediction failed: [\"Command '['/home/bviggiano/codebases/proto-bio/proto_tool_envs/boltz2_env/bin/python', '/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py', '/tmp/tmpx1yy8k0q/input.json', '/tmp/tmpx1yy8k0q/output.json']' returned non-zero exit status 1.\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py\", line 428, in wrapper\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/boltz2/boltz2.py\", line 274, in run_boltz2\\n    run_boltz2_on_complex(\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/boltz2/boltz2.py\", line 382, in run_boltz2_on_complex\\n    output_data = ToolInstance.dispatch(\\n                  ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 254, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 293, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 999, in _run_oneshot\\n    subprocess.run(\\n  File \"/home/bviggiano/miniconda3/envs/proto-tools/lib/python3.12/subprocess.py\", line 571, in run\\n    raise CalledProcessError(retcode, process.args,\\nsubprocess.CalledProcessError: Command \\'[\\'/home/bviggiano/codebases/proto-bio/proto_tool_envs/boltz2_env/bin/python\\', \\'/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\\', \\'/tmp/tmpx1yy8k0q/input.json\\', \\'/tmp/tmpx1yy8k0q/output.json\\']\\' returned non-zero exit status 1.\\n']\nE   assert False\nE    +  where False = <[ToolExecutionError('Attempt to access field of tool output after failure: subprocess.CalledProcessError: Command \\'[\\'/home/bviggiano/codebases/proto-bio/proto_tool_envs/boltz2_env/bin/python\\', \\'/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\\', \\'/tmp/tmpx1yy8k0q/input.json\\', \\'/tmp/tmpx1yy8k0q/output.json\\']\\' returned non-zero exit status 1.\\n\\nError Messages:\\nCommand \\'[\\'/home/bviggiano/codebases/proto-bio/proto_tool_envs/boltz2_env/bin/python\\', \\'/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\\', \\'/tmp/tmpx1yy8k0q/input.json\\', \\'/tmp/tmpx1yy8k0q/output.json\\']\\' returned non-zero exit status 1.\\nTraceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py\", line 428, in wrapper\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/boltz2/boltz2.py\", line 274, in run_boltz2\\n    run_boltz2_on_complex(\\n  File \"/home/bviggiano/codebase...boltz2/boltz2.py\", line 382, in run_boltz2_on_complex\\n    output_data = ToolInstance.dispatch(\\n                  ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 254, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 293, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 999, in _run_oneshot\\n    subprocess.run(\\n  File \"/home/bviggiano/miniconda3/envs/proto-tools/lib/python3.12/subprocess.py\", line 571, in run\\n    raise CalledProcessError(retcode, process.args,\\nsubprocess.CalledProcessError: Command \\'[\\'/home/bviggiano/codebases/proto-bio/proto_tool_envs/boltz2_env/bin/python\\', \\'/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\\', \\'/tmp/tmpx1yy8k0q/input.json\\', \\'/tmp/tmpx1yy8k0q/output.json\\']\\' returned non-zero exit status 1.\\n') raised in repr()] StructurePredictionOutput object at 0xe68f8a65ba70>.success",
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "borzoi-ensemble",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[borzoi-ensemble]",
    "status": "passed",
    "duration_seconds": 238.12,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/borzoi_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "chai1-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[chai1-prediction]",
    "status": "failed",
    "duration_seconds": 7.76,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/chai1_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:119: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool chai1-prediction failed: [\"'chai1' may not be compatible with your system. setup.sh failed (exit 1).\\nERROR: Chai is not supported on aarch64.\\nchai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its\\npre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py\", line 428, in wrapper\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py\", line 290, in run_chai1\\n    results.append(run_chai1_on_complex(comp=comp, config=config, msas=inputs.msas, instance=instance))\\n                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py\", line 377, in run_chai1_on_complex\\n    result = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 254, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 293, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 968, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 504, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 1404, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'chai1\\' may not be compatible with your system. setup.sh failed (exit 1).\\nERROR: Chai is not supported on aarch64.\\nchai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its\\npre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\\n']\nE   assert False\nE    +  where False = <[ToolExecutionError('Attempt to access field of tool output after failure: pre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\\n\\nError Messages:\\n\\'chai1\\' may not be compatible with your system. setup.sh failed (exit 1).\\nERROR: Chai is not supported on aarch64.\\nchai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its\\npre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\\nTraceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py\", line 428, in wrapper\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py\", line 290, in run_chai1\\n    results.append(run_chai1_on_complex(comp=comp, config=config, msas=inputs.msas, instance=instance))\\n                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py\", line 377, in run_chai1_on_complex\\n    result = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 254, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 293, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 968, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 504, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 1404, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'chai1\\' may not be compatible with your system. setup.sh failed (exit 1).\\nERROR: Chai is not supported on aarch64.\\nchai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its\\npre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\\n') raised in repr()] StructurePredictionOutput object at 0xe68f8a65b9d0>.success",
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "colabfold-search",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[colabfold-search]",
    "status": "passed",
    "duration_seconds": 26.27,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/colabfold_search_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "crispr-tracr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[crispr-tracr]",
    "status": "failed",
    "duration_seconds": 30.05,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/crispr_tracr_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:119: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool crispr-tracr failed: [\"'crispr_tracr' may not be compatible with your system. setup.sh failed (exit 1).\\nCloning into '/home/bviggiano/codebases/proto-bio/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA'...\\nCloning CRISPRidentify into CRISPRtracrRNA tools directory...\\nCloning into '/home/bviggiano/codebases/proto-bio/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA/tools/CRISPRidentify/CRISPRidentify'...\\nCloning CRISPRcasIdentifier into CRISPRtracrRNA tools directory...\\nCloning into '/home/bviggiano/codebases/proto-bio/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA/tools/CRISPRcasIdentifier/CRISPRcasIdentifier'...\\nCreating isolated conda environment (Python 3.8 + scikit-learn 0.22)...\\nCRISPRidentify's pickled models require sklearn 0.22 (incompatible with 3.12).\\nUsing /home/bviggiano/codebases/proto-bio/proto_tool_envs/crispr_tracr_env/conda_deps to avoid polluting base env...\\nERROR: CRISPRtracrRNA requires x86_64 bioconda packages (vmatch, etc.)\\n       that are not available on Linux aarch64.\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py\", line 428, in wrapper\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/gene_annotation/crispr_tracr/crispr_tracr.py\", line 220, in run_crispr_tracr\\n    output_data = ToolInstance.dispatch(\\n                  ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 254, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 300, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 968, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 504, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 1404, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'crispr_tracr\\' may not be compatible with your system. setup.sh failed (exit 1).\\nCloning into \\'/home/bviggiano/codebases/proto-bio/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA\\'...\\nCloning CRISPRidentify into CRISPRtracrRNA tools directory...\\nCloning into \\'/home/bviggiano/codebases/proto-bio/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA/tools/CRISPRidentify/CRISPRidentify\\'...\\nCloning CRISPRcasIdentifier into CRISPRtracrRNA tools directory...\\nCloning into \\'/home/bviggiano/codebases/proto-bio/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA/tools/CRISPRcasIdentifier/CRISPRcasIdentifier\\'...\\nCreating isolated conda environment (Python 3.8 + scikit-learn 0.22)...\\nCRISPRidentify\\'s pickled models require sklearn 0.22 (incompatible with 3.12).\\nUsing /home/bviggiano/codebases/proto-bio/proto_tool_envs/crispr_tracr_env/conda_deps to avoid polluting base env...\\nERROR: CRISPRtracrRNA requires x86_64 bioconda packages (vmatch, etc.)\\n       that are not available on Linux aarch64.\\n']\nE   assert False\nE    +  where False = CrisprTracrOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, predictions).success",
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "enformer-prediction",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[enformer-prediction]",
    "status": "passed",
    "duration_seconds": 57.84,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/enformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "esm-if1-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm-if1-sample]",
    "status": "passed",
    "duration_seconds": 202.55,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/esm_if1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "esm2-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm2-embedding]",
    "status": "passed",
    "duration_seconds": 84.17,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/esm2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "esm3-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm3-embedding]",
    "status": "passed",
    "duration_seconds": 117.18,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/esm3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "esmfold-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esmfold-prediction]",
    "status": "passed",
    "duration_seconds": 220.56,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/esmfold_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "evo1-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo1-sample]",
    "status": "failed",
    "duration_seconds": 123.16,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/evo1_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:119: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool evo1-sample failed: ['\\'evo1\\' may not be compatible with your system. setup.sh failed (exit 1).\\n      on `psutil`, but doesn\\'t declare it as a build dependency. If\\n      `flash-attn` is a first-party package, consider adding `psutil`\\n      to its `build-system.requires`. Otherwise, either add it to your\\n      `pyproject.toml` under:\\n      [tool.uv.extra-build-dependencies]\\n      flash-attn = [\"psutil\"]\\n      or `uv pip install psutil` into the environment and re-run with\\n      `--no-build-isolation`.\\n  help: `flash-attn` (v2.8.3) was included because `evo-model` (v0.5) depends\\n        on `stripedhyena` (v0.2.2) which depends on `flash-attn`', 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py\", line 428, in wrapper\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/causal_models/evo1/evo1_sample.py\", line 219, in run_evo1_sample\\n    result = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 254, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 293, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 968, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 504, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 1404, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'evo1\\' may not be compatible with your system. setup.sh failed (exit 1).\\n      on `psutil`, but doesn\\'t declare it as a build dependency. If\\n      `flash-attn` is a first-party package, consider adding `psutil`\\n      to its `build-system.requires`. Otherwise, either add it to your\\n      `pyproject.toml` under:\\n      [tool.uv.extra-build-dependencies]\\n      flash-attn = [\"psutil\"]\\n      or `uv pip install psutil` into the environment and re-run with\\n      `--no-build-isolation`.\\n  help: `flash-attn` (v2.8.3) was included because `evo-model` (v0.5) depends\\n        on `stripedhyena` (v0.2.2) which depends on `flash-attn`\\n']\nE   assert False\nE    +  where False = Evo1SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, scores).success",
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "evo2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo2-sample]",
    "status": "failed",
    "duration_seconds": 8.02,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/evo2_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:119: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool evo2-sample failed: [\"'evo2' may not be compatible with your system. setup.sh failed (exit 1).\\nERROR: Evo2 is not supported on aarch64.\\nEvo2 requires transformer-engine and flash-attn which only provide x86_64 pre-built wheels.\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py\", line 428, in wrapper\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/causal_models/evo2/evo2_sample.py\", line 436, in run_evo2_sample\\n    result = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 254, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 293, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 968, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 504, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 1404, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'evo2\\' may not be compatible with your system. setup.sh failed (exit 1).\\nERROR: Evo2 is not supported on aarch64.\\nEvo2 requires transformer-engine and flash-attn which only provide x86_64 pre-built wheels.\\n']\nE   assert False\nE    +  where False = Evo2SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, logits, kv_caches).success",
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "fampnn-pack",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[fampnn-pack]",
    "status": "passed",
    "duration_seconds": 127.82,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/fampnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "ligandmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ligandmpnn-sample]",
    "status": "passed",
    "duration_seconds": 68.44,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/ligandmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "mafft-align",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mafft-align]",
    "status": "failed",
    "duration_seconds": 24.25,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/mafft_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:119: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool mafft-align failed: ['\\'mafft\\' may not be compatible with your system. setup.sh failed (exit 1).\\n    install_binary(sys.argv[1])\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/install_binary.py\", line 188, in install_binary\\n    config.extract(archive_path, bin_dir)\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_alignment/mafft/standalone/binary_config.py\", line 165, in extract\\n    _extract_from_source(archive_path, bin_dir)\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_alignment/mafft/standalone/binary_config.py\", line 134, in _extract_from_source\\n    subprocess.check_call(\\n  File \"/home/bviggiano/codebases/proto-bio/proto_tool_envs/mafft_env/lib/python3.12/subprocess.py\", line 413, in check_call\\n    raise CalledProcessError(retcode, cmd)\\nsubprocess.CalledProcessError: Command \\'[\\'make\\', \\'-j4\\', \\'PREFIX=/home/bviggiano/codebases/proto-bio/proto_tool_envs/mafft_env\\', \\'BINDIR=/home/bviggiano/codebases/proto-bio/proto_tool_envs/mafft_env/bin\\', \\'LIBDIR=/home/bviggiano/codebases/proto-bio/proto_tool_envs/mafft_env/libexec/mafft\\']\\' returned non-zero exit status 2.', 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py\", line 428, in wrapper\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_alignment/mafft/mafft.py\", line 199, in run_mafft_align\\n    output_data = ToolInstance.dispatch(\\n                  ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 254, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 300, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 968, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 504, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 1404, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'mafft\\' may not be compatible with your system. setup.sh failed (exit 1).\\n    install_binary(sys.argv[1])\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/install_binary.py\", line 188, in install_binary\\n    config.extract(archive_path, bin_dir)\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_alignment/mafft/standalone/binary_config.py\", line 165, in extract\\n    _extract_from_source(archive_path, bin_dir)\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_alignment/mafft/standalone/binary_config.py\", line 134, in _extract_from_source\\n    subprocess.check_call(\\n  File \"/home/bviggiano/codebases/proto-bio/proto_tool_envs/mafft_env/lib/python3.12/subprocess.py\", line 413, in check_call\\n    raise CalledProcessError(retcode, cmd)\\nsubprocess.CalledProcessError: Command \\'[\\'make\\', \\'-j4\\', \\'PREFIX=/home/bviggiano/codebases/proto-bio/proto_tool_envs/mafft_env\\', \\'BINDIR=/home/bviggiano/codebases/proto-bio/proto_tool_envs/mafft_env/bin\\', \\'LIBDIR=/home/bviggiano/codebases/proto-bio/proto_tool_envs/mafft_env/libexec/mafft\\']\\' returned non-zero exit status 2.\\n']\nE   assert False\nE    +  where False = MafftOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success",
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "minced-crispr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[minced-crispr]",
    "status": "passed",
    "duration_seconds": 11.31,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/minced_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "mmseqs-clustering",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mmseqs-clustering]",
    "status": "passed",
    "duration_seconds": 14.29,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/mmseqs_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
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
    "error_message": "('/home/bviggiano/codebases/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 96, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "mock-cli-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-cli-tool-run]",
    "status": "passed",
    "duration_seconds": 7.63,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/mock_cli_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
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
    "error_message": "('/home/bviggiano/codebases/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 96, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "mock-jax-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-jax-tool-run]",
    "status": "passed",
    "duration_seconds": 39.4,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/mock_jax_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
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
    "error_message": "('/home/bviggiano/codebases/proto-bio/proto-tools/tests/tool_infra_tests/test_env_report.py', 96, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "mock-pytorch-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-pytorch-tool-run]",
    "status": "passed",
    "duration_seconds": 41.24,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/mock_pytorch_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "orfipy-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[orfipy-prediction]",
    "status": "passed",
    "duration_seconds": 12.07,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/orfipy_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "prodigal-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[prodigal-prediction]",
    "status": "passed",
    "duration_seconds": 11.47,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/prodigal_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "progen2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen2-sample]",
    "status": "failed",
    "duration_seconds": 7.83,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/progen2_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:119: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool progen2-sample failed: [\"'progen2' may not be compatible with your system. setup.sh failed (exit 1).\\nERROR: ProGen2 is not supported on aarch64.\\nProGen2 pins torch==2.2.2 which has no aarch64 CUDA wheel available.\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py\", line 428, in wrapper\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/causal_models/progen2/progen2_sample.py\", line 379, in run_progen2_sample\\n    result = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 254, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 293, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 968, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 504, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 1404, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'progen2\\' may not be compatible with your system. setup.sh failed (exit 1).\\nERROR: ProGen2 is not supported on aarch64.\\nProGen2 pins torch==2.2.2 which has no aarch64 CUDA wheel available.\\n']\nE   assert False\nE    +  where False = ProGen2SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, logits).success",
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "proteinmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[proteinmpnn-sample]",
    "status": "passed",
    "duration_seconds": 64.03,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/proteinmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "protenix-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[protenix-prediction]",
    "status": "failed",
    "duration_seconds": 1075.62,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/protenix_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:119: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool protenix-prediction failed: [\"Command '['/home/bviggiano/codebases/proto-bio/proto_tool_envs/protenix_env/bin/python', '/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/protenix/standalone/inference.py', '/tmp/tmpfsxhwxj2/input.json', '/tmp/tmpfsxhwxj2/output.json']' returned non-zero exit status 1.\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py\", line 428, in wrapper\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/protenix/protenix.py\", line 392, in run_protenix\\n    output_data = ToolInstance.dispatch(\\n                  ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 254, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 293, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 999, in _run_oneshot\\n    subprocess.run(\\n  File \"/home/bviggiano/miniconda3/envs/proto-tools/lib/python3.12/subprocess.py\", line 571, in run\\n    raise CalledProcessError(retcode, process.args,\\nsubprocess.CalledProcessError: Command \\'[\\'/home/bviggiano/codebases/proto-bio/proto_tool_envs/protenix_env/bin/python\\', \\'/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/protenix/standalone/inference.py\\', \\'/tmp/tmpfsxhwxj2/input.json\\', \\'/tmp/tmpfsxhwxj2/output.json\\']\\' returned non-zero exit status 1.\\n']\nE   assert False\nE    +  where False = <[ToolExecutionError('Attempt to access field of tool output after failure: subprocess.CalledProcessError: Command \\'[\\'/home/bviggiano/codebases/proto-bio/proto_tool_envs/protenix_env/bin/python\\', \\'/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/protenix/standalone/inference.py\\', \\'/tmp/tmpfsxhwxj2/input.json\\', \\'/tmp/tmpfsxhwxj2/output.json\\']\\' returned non-zero exit status 1.\\n\\nError Messages:\\nCommand \\'[\\'/home/bviggiano/codebases/proto-bio/proto_tool_envs/protenix_env/bin/python\\', \\'/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/protenix/standalone/inference.py\\', \\'/tmp/tmpfsxhwxj2/input.json\\', \\'/tmp/tmpfsxhwxj2/output.json\\']\\' returned non-zero exit status 1.\\nTraceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py\", line 428, in wrapper\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/protenix/protenix.py\", line 392, in run_protenix\\n    output_data = ToolInstance.dispatch(\\n                  ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 254, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 293, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 999, in _run_oneshot\\n    subprocess.run(\\n  File \"/home/bviggiano/miniconda3/envs/proto-tools/lib/python3.12/subprocess.py\", line 571, in run\\n    raise CalledProcessError(retcode, process.args,\\nsubprocess.CalledProcessError: Command \\'[\\'/home/bviggiano/codebases/proto-bio/proto_tool_envs/protenix_env/bin/python\\', \\'/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_prediction/protenix/standalone/inference.py\\', \\'/tmp/tmpfsxhwxj2/input.json\\', \\'/tmp/tmpfsxhwxj2/output.json\\']\\' returned non-zero exit status 1.\\n') raised in repr()] StructurePredictionOutput object at 0xe68f8991a4e0>.success",
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "pyhmmer-hmmscan",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pyhmmer-hmmscan]",
    "status": "passed",
    "duration_seconds": 24.66,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/pyhmmer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "random-nucleotide-sample",
    "category": "mutagenesis",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[random-nucleotide-sample]",
    "status": "passed",
    "duration_seconds": 0.03,
    "uses_gpu": false,
    "env_path": null,
    "env_status": "not_found",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
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
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "rfdiffusion3-design",
    "category": "structure_design",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[rfdiffusion3-design]",
    "status": "failed",
    "duration_seconds": 3922.58,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/rfdiffusion3_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:119: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool rfdiffusion3-design failed: ['Tool rfdiffusion3 timed out after 3600s', 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py\", line 428, in wrapper\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/structure_design/rfdiffusion3/rfdiffusion3_sample.py\", line 598, in run_rfdiffusion3\\n    output_data = ToolInstance.dispatch(\\n                  ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 254, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 293, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 1016, in _run_oneshot\\n    raise TimeoutError(\\nTimeoutError: Tool rfdiffusion3 timed out after 3600s\\n']\nE   assert False\nE    +  where False = RFdiffusion3Output(output_structures=[0 structures]).success",
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "segmasker-score",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[segmasker-score]",
    "status": "failed",
    "duration_seconds": 114.89,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/segmasker_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:119: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool segmasker-score failed: [\"Command '['/home/bviggiano/codebases/proto-bio/proto_tool_envs/segmasker_env/bin/python', '/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_scoring/segmasker/standalone/run.py', '/tmp/tmp144b8nhh/input.json', '/tmp/tmp144b8nhh/output.json']' returned non-zero exit status 1.\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/tool_registry.py\", line 428, in wrapper\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_scoring/segmasker/segmasker.py\", line 257, in run_segmasker\\n    output_data = ToolInstance.dispatch(\\n                  ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 254, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 300, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/utils/tool_instance.py\", line 999, in _run_oneshot\\n    subprocess.run(\\n  File \"/home/bviggiano/miniconda3/envs/proto-tools/lib/python3.12/subprocess.py\", line 571, in run\\n    raise CalledProcessError(retcode, process.args,\\nsubprocess.CalledProcessError: Command \\'[\\'/home/bviggiano/codebases/proto-bio/proto_tool_envs/segmasker_env/bin/python\\', \\'/home/bviggiano/codebases/proto-bio/proto-tools/proto_tools/tools/sequence_scoring/segmasker/standalone/run.py\\', \\'/tmp/tmp144b8nhh/input.json\\', \\'/tmp/tmp144b8nhh/output.json\\']\\' returned non-zero exit status 1.\\n']\nE   assert False\nE    +  where False = SegmaskerOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, results_df).success",
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "splice-transformer-prediction",
    "category": "rna_splicing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[splice-transformer-prediction]",
    "status": "passed",
    "duration_seconds": 39.73,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/splice_transformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "structure-metrics",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[structure-metrics]",
    "status": "passed",
    "duration_seconds": 14.64,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/structure_metrics_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "tmalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[tmalign-alignment]",
    "status": "passed",
    "duration_seconds": 41.69,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/tmalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "usalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[usalign-alignment]",
    "status": "passed",
    "duration_seconds": 78.89,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/usalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  },
  {
    "tool_name": "viennarna-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[viennarna-prediction]",
    "status": "passed",
    "duration_seconds": 11.88,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/codebases/proto-bio/proto_tool_envs/viennarna_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "f36a85cdf12b",
    "git_dirty": true
  }
]
-->