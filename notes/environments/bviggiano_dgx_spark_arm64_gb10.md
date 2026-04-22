# DGX Spark Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-71%25-yellow) ![Passed](https://img.shields.io/badge/passed-30-brightgreen) ![Failed](https://img.shields.io/badge/failed-12-red) ![Skipped](https://img.shields.io/badge/skipped-4-lightgrey)

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

- **Commit**: `16714ef3b882`
- **Branch**: `fix/rfdiffusion3-spark-regression`
- **Dirty**: Yes

## Environment Variables

### Parent Process Environment

```
CLAUDECODE=1
CLAUDE_CODE_ENTRYPOINT=cli
CLAUDE_CODE_EXECPATH=/home/bviggiano/.local/share/claude/versions/2.1.104
CONDA_DEFAULT_ENV=proto-tools
CONDA_EXE=/home/bviggiano/miniconda3/bin/conda
CONDA_PREFIX=/home/bviggiano/miniconda3/envs/proto-tools
CONDA_PREFIX_1=/home/bviggiano/miniconda3
CONDA_PREFIX_2=/home/bviggiano/miniconda3/envs/proto-tools
CONDA_PREFIX_3=/home/bviggiano/miniconda3
CONDA_PROMPT_MODIFIER=(proto-tools) 
CONDA_PYTHON_EXE=/home/bviggiano/miniconda3/bin/python
CONDA_SHLVL=4
COREPACK_ENABLE_AUTO_PIN=0
DEBUGINFOD_URLS=https://debuginfod.ubuntu.com 
DISABLE_PANDERA_IMPORT_WARNING=True
GIT_EDITOR=true
GSETTINGS_SCHEMA_DIR=/home/bviggiano/miniconda3/envs/proto-tools/share/glib-2.0/schemas
GSETTINGS_SCHEMA_DIR_CONDA_BACKUP=
HOME=/home/bviggiano
LANG=en_US.utf8
LESSCLOSE=/usr/bin/lesspipe %s %s
LESSOPEN=| /usr/bin/lesspipe %s
LOGNAME=bviggiano
LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=00:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.arj=...
NoDefaultCurrentDirectoryInExePath=1
OLDPWD=/home/bviggiano/codebases/proto
OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta
PATH=/home/bviggiano/.local/bin:/home/bviggiano/.local/bin:/home/bviggiano/.local/bin:/usr/local/cuda/bin:/opt/bin:/home/bviggiano/.local/bin:/home/bviggiano/.local/bin:/home/bviggiano/.local/bin:/home/bvi...
PWD=/home/bviggiano/codebases/proto/proto-tools
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.3
RDBASE=/home/bviggiano/miniconda3/envs/proto-tools/lib/python3.12/site-packages/rdkit
SHELL=/bin/bash
SHLVL=3
TERM=tmux-256color
TERM_PROGRAM=tmux
TERM_PROGRAM_VERSION=3.4
TMUX=/tmp/tmux-1003/default,70674,0
TMUX_PANE=%0
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
CONDA_PREFIX=/home/bviggiano/.proto/proto_tool_envs/pyhmmer_env
DETECTED_COMPUTE_PLATFORM=cuda
DETECTED_CUDA_VERSION=13
DETECTED_DRIVER_VERSION=580
HF_HOME=/home/bviggiano/.proto/proto_model_cache/huggingface
HOME=/home/bviggiano
LANG=en_US.utf8
LD_LIBRARY_PATH=/home/bviggiano/miniconda3/envs/proto-tools/lib
LOGNAME=bviggiano
PATH=/home/bviggiano/.proto/proto_tool_envs/pyhmmer_env/bin:/home/bviggiano/.local/bin:/usr/local/cuda/bin:/opt/bin:/home/bviggiano/miniconda3/envs/proto-tools/bin:/home/bviggiano/miniconda3/condabin:/usr/...
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
UV_HTTP_TIMEOUT=300
VIRTUAL_ENV=/home/bviggiano/.proto/proto_tool_envs/pyhmmer_env
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Results by Category

### Causal Models (0/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `evo1-sample` | yes | ✅ | 49.5s | `16714ef` ✱ | ❌ Fail |
| `evo2-sample` | yes | ✅ | 2.0s | `16714ef` ✱ | ❌ Fail |
| `progen2-sample` | yes | ✅ | 1.8s | `16714ef` ✱ | ❌ Fail |
| `progen3-sample` | yes | ✅ | 0.1s | `16714ef` ✱ | ❌ Fail |

### Gene Annotation (4/5)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `blast-create-db` | no | ✅ | 70.8s | `16714ef` ✱ | ✅ Pass |
| `crispr-tracr` | no | ✅ | 13.7s | `16714ef` ✱ | ❌ Fail |
| `minced-crispr` | no | ✅ | 3.1s | `16714ef` ✱ | ✅ Pass |
| `mmseqs-clustering` | no | ✅ | 5.3s | `16714ef` ✱ | ✅ Pass |
| `pyhmmer-hmmscan` | no | ✅ | 3.9s | `16714ef` ✱ | ✅ Pass |

### Inverse Folding (3/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `esm-if1-sample` | yes | ✅ | 197.2s | `16714ef` ✱ | ✅ Pass |
| `fampnn-pack` | yes | ✅ | 47.9s | `16714ef` ✱ | ✅ Pass |
| `ligandmpnn-sample` | yes | ✅ | 32.8s | `16714ef` ✱ | ❌ Fail |
| `proteinmpnn-sample` | yes | ✅ | 113.3s | `16714ef` ✱ | ✅ Pass |

### Masked Models (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `ablang-embedding` | yes | ✅ | 130.0s | `16714ef` ✱ | ✅ Pass |
| `esm2-embedding` | yes | ✅ | 53.4s | `16714ef` ✱ | ✅ Pass |
| `esm3-embedding` | yes | ✅ | 61.9s | `16714ef` ✱ | ✅ Pass |

### Mutagenesis (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `random-nucleotide-sample` | no | - | 0.0s | `16714ef` ✱ | ✅ Pass |
| `random-protein-sample` | no | - | 0.0s | `16714ef` ✱ | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `orfipy-prediction` | no | ✅ | 31.8s | `16714ef` ✱ | ✅ Pass |
| `prodigal-prediction` | no | ✅ | 3.7s | `16714ef` ✱ | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `splice-transformer-prediction` | yes | ✅ | 20.8s | `16714ef` ✱ | ✅ Pass |

### Sequence Alignment (1/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `colabfold-search` | no | ✅ | 33.9s | `16714ef` ✱ | ✅ Pass |
| `mafft-align` | no | ✅ | 7.1s | `16714ef` ✱ | ❌ Fail |

### Sequence Scoring (3/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphagenome-predict-intervals` | yes | ✅ | 99.3s | `16714ef` ✱ | ✅ Pass |
| `borzoi-ensemble` | yes | ✅ | 116.2s | `16714ef` ✱ | ✅ Pass |
| `enformer-prediction` | yes | ✅ | 30.5s | `16714ef` ✱ | ✅ Pass |
| `segmasker-score` | no | ✅ | 61.7s | `16714ef` ✱ | ❌ Fail |

### Structure Alignment (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `tmalign-alignment` | no | ✅ | 9.8s | `16714ef` ✱ | ✅ Pass |
| `usalign-alignment` | no | ✅ | 16.2s | `16714ef` ✱ | ✅ Pass |

### Structure Design (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `rfdiffusion3-design` | yes | ✅ | 1548.7s | `16714ef` ✱ | ✅ Pass |

### Structure Dynamics (0/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `bioemu-sample` | yes | ✅ | 82.2s | `16714ef` ✱ | ❌ Fail |

### Structure Prediction (3/6)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `alphafold2-prediction` | yes | ✅ | 90.6s | `16714ef` ✱ | ✅ Pass |
| `alphafold3-prediction` | yes | - | - | `16714ef` ✱ | ⏭️ Skip |
| `boltz2-prediction` | yes | ✅ | 150.1s | `16714ef` ✱ | ❌ Fail |
| `chai1-prediction` | yes | ✅ | 1.9s | `16714ef` ✱ | ❌ Fail |
| `esmfold-prediction` | yes | ✅ | 159.7s | `16714ef` ✱ | ✅ Pass |
| `protenix-prediction` | yes | ✅ | 33.7s | `16714ef` ✱ | ❌ Fail |
| `viennarna-prediction` | no | ✅ | 3.8s | `16714ef` ✱ | ✅ Pass |

### Structure Scoring (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `pyrosetta-energy` | no | ✅ | 360.8s | `16714ef` ✱ | ✅ Pass |
| `structure-metrics` | no | ✅ | 15.2s | `16714ef` ✱ | ✅ Pass |

### Testing (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |
|------|--------------|----------------------|----------|-----------|--------|
| `mock-cli-multi-gpu-tool-run` | yes | - | - | `16714ef` ✱ | ⏭️ Skip |
| `mock-cli-tool-run` | yes | ✅ | 2.1s | `16714ef` ✱ | ✅ Pass |
| `mock-jax-multi-gpu-tool-run` | yes | - | - | `16714ef` ✱ | ⏭️ Skip |
| `mock-jax-tool-run` | yes | ✅ | 8.9s | `16714ef` ✱ | ✅ Pass |
| `mock-pytorch-multi-gpu-tool-run` | yes | - | - | `16714ef` ✱ | ⏭️ Skip |
| `mock-pytorch-tool-run` | yes | ✅ | 18.1s | `16714ef` ✱ | ✅ Pass |

## Failure Details

### ❌ `segmasker-score`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[segmasker-score]`

```
tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool segmasker-score failed: ["Command '['/home/bviggiano/.proto/proto_tool_envs/segmasker_env/bin/python', '/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_scoring/segmasker/standalone/run.py', '/tmp/tmp5iyjxewg/input.json', '/tmp/tmp5iyjxewg/output.json']' returned non-zero exit status 1.", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 495, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_scoring/segmasker/segmasker.py", line 235, in run_segmasker\n    output_data = ToolInstance.dispatch(\n                  ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 252, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 298, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 991, in _run_oneshot\n    subprocess.run(\n  File "/home/bviggiano/miniconda3/envs/proto-tools/lib/python3.12/subprocess.py", line 571, in run\n    raise CalledProcessError(retcode, process.args,\nsubprocess.CalledProcessError: Command \'[\'/home/bviggiano/.proto/proto_tool_envs/segmasker_env/bin/python\', \'/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_scoring/segmasker/standalone/run.py\', \'/tmp/tmp5iyjxewg/input.json\', \'/tmp/tmp5iyjxewg/output.json\']\' returned non-zero exit status 1.\n']
E   assert False
E    +  where False = SegmaskerOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success
```

### ❌ `mafft-align`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mafft-align]`

```
tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool mafft-align failed: ['\'mafft\' may not be compatible with your system. setup.sh failed (exit 1).\n    install_binary(sys.argv[1])\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/install_binary.py", line 179, in install_binary\n    config.extract(archive_path, bin_dir)\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_alignment/mafft/standalone/binary_config.py", line 163, in extract\n    _extract_from_source(archive_path, bin_dir)\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_alignment/mafft/standalone/binary_config.py", line 130, in _extract_from_source\n    subprocess.check_call(\n  File "/home/bviggiano/.proto/proto_tool_envs/mafft_env/lib/python3.12/subprocess.py", line 413, in check_call\n    raise CalledProcessError(retcode, cmd)\nsubprocess.CalledProcessError: Command \'[\'make\', \'-j4\', \'PREFIX=/home/bviggiano/.proto/proto_tool_envs/mafft_env\', \'BINDIR=/home/bviggiano/.proto/proto_tool_envs/mafft_env/bin\', \'LIBDIR=/home/bviggiano/.proto/proto_tool_envs/mafft_env/libexec/mafft\']\' returned non-zero exit status 2.', 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 495, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_alignment/mafft/mafft.py", line 195, in run_mafft_align\n    output_data = ToolInstance.dispatch(\n                  ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 252, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 298, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 949, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 510, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1473, in _create_env\n    raise RuntimeError(\nRuntimeError: \'mafft\' may not be compatible with your system. setup.sh failed (exit 1).\n    install_binary(sys.argv[1])\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/install_binary.py", line 179, in install_binary\n    config.extract(archive_path, bin_dir)\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_alignment/mafft/standalone/binary_config.py", line 163, in extract\n    _extract_from_source(archive_path, bin_dir)\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_alignment/mafft/standalone/binary_config.py", line 130, in _extract_from_source\n    subprocess.check_call(\n  File "/home/bviggiano/.proto/proto_tool_envs/mafft_env/lib/python3.12/subprocess.py", line 413, in check_call\n    raise CalledProcessError(retcode, cmd)\nsubprocess.CalledProcessError: Command \'[\'make\', \'-j4\', \'PREFIX=/home/bviggiano/.proto/proto_tool_envs/mafft_env\', \'BINDIR=/home/bviggiano/.proto/proto_tool_envs/mafft_env/bin\', \'LIBDIR=/home/bviggiano/.proto/proto_tool_envs/mafft_env/libexec/mafft\']\' returned non-zero exit status 2.\n']
E   assert False
E    +  where False = MafftOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success
```

### ❌ `protenix-prediction`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[protenix-prediction]`

```
tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool protenix-prediction failed: ["'protenix' may not be compatible with your system. setup.sh failed (exit 1).\n      Model& model) {\n            |\n      ^\n      ninja: build stopped: subcommand failed.\n      [stderr]\n      *** CMake build failed\n      hint: This usually indicates a problem with the package or the build\n      environment.\n  help: `gemmi` (v0.6.7) was included because `protenix` (v2.0.0) depends\n        on `gemmi`", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 495, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/protenix/protenix.py", line 422, in run_protenix\n    output_data = ToolInstance.dispatch(\n                  ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 252, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 291, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 949, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 510, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1473, in _create_env\n    raise RuntimeError(\nRuntimeError: \'protenix\' may not be compatible with your system. setup.sh failed (exit 1).\n      Model& model) {\n            |\n      ^\n      ninja: build stopped: subcommand failed.\n      [stderr]\n      *** CMake build failed\n      hint: This usually indicates a problem with the package or the build\n      environment.\n  help: `gemmi` (v0.6.7) was included because `protenix` (v2.0.0) depends\n        on `gemmi`\n']
E   assert False
E    +  where False = <[ToolExecutionError('Attempt to access field of tool output after failure:         on `gemmi`\n\nError Messages:\n\'protenix\' may not be compatible with your system. setup.sh failed (exit 1).\n      Model& model) {\n            |\n      ^\n      ninja: build stopped: subcommand failed.\n      [stderr]\n      *** CMake build failed\n      hint: This usually indicates a problem with the package or the build\n      environment.\n  help: `gemmi` (v0.6.7) was included because `protenix` (v2.0.0) depends\n        on `gemmi`\nTraceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 495, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/protenix/protenix.py", line 422, in run_protenix\n    output_data = ToolInstance.dispatch(\n                  ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 252, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 291, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 949, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 510, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1473, in _create_env\n    raise RuntimeError(\nRuntimeError: \'protenix\' may not be compatible with your system. setup.sh failed (exit 1).\n      Model& model) {\n            |\n      ^\n      ninja: build stopped: subcommand failed.\n      [stderr]\n      *** CMake build failed\n      hint: This usually indicates a problem with the package or the build\n      environment.\n  help: `gemmi` (v0.6.7) was included because `protenix` (v2.0.0) depends\n        on `gemmi`\n') raised in repr()] ProtenixOutput object at 0xe58fa9e52030>.success
```

### ❌ `chai1-prediction`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[chai1-prediction]`

```
tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool chai1-prediction failed: ["'chai1' may not be compatible with your system. setup.sh failed (exit 1).\nERROR: Chai is not supported on aarch64.\nchai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its\npre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 495, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py", line 291, in run_chai1\n    run_chai1_on_complex(comp=comp, config=config, msas=inputs.msas, instance=instance)\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py", line 374, in run_chai1_on_complex\n    result = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 252, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 291, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 949, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 510, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1473, in _create_env\n    raise RuntimeError(\nRuntimeError: \'chai1\' may not be compatible with your system. setup.sh failed (exit 1).\nERROR: Chai is not supported on aarch64.\nchai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its\npre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\n']
E   assert False
E    +  where False = <[ToolExecutionError('Attempt to access field of tool output after failure: pre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\n\nError Messages:\n\'chai1\' may not be compatible with your system. setup.sh failed (exit 1).\nERROR: Chai is not supported on aarch64.\nchai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its\npre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\nTraceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 495, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py", line 291, in run_chai1\n    run_chai1_on_complex(comp=comp, config=config, msas=inputs.msas, instance=instance)\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py", line 374, in run_chai1_on_complex\n    result = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 252, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 291, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 949, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 510, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1473, in _create_env\n    raise RuntimeError(\nRuntimeError: \'chai1\' may not be compatible with your system. setup.sh failed (exit 1).\nERROR: Chai is not supported on aarch64.\nchai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its\npre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\n') raised in repr()] Chai1Output object at 0xe58fa93546e0>.success
```

### ❌ `ligandmpnn-sample`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ligandmpnn-sample]`

```
tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool ligandmpnn-sample failed: ['\'ligandmpnn\' may not be compatible with your system. setup.sh failed (exit 1).\n          getattr(self, attribute)\n        File\n      "/home/bviggiano/.cache/uv/builds-v0/.tmpa8JMuK/lib/python3.12/site-packages/hatchling/metadata/core.py",\n      line 750, in license_files\n          raise TypeError(message)\n      TypeError: Field `project.license-files` must be a table\n      hint: This usually indicates a problem with the package or the build\n      environment.\n  help: `biotite` (v1.4.0) was included because `rc-foundry` (v0.1.12) depends\n        on `atomworks` (v2.2.0) which depends on `biotite`', 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 495, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/inverse_folding/ligandmpnn/ligandmpnn_sample.py", line 123, in run_ligandmpnn_sample\n    result = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 252, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 291, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 949, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 510, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1473, in _create_env\n    raise RuntimeError(\nRuntimeError: \'ligandmpnn\' may not be compatible with your system. setup.sh failed (exit 1).\n          getattr(self, attribute)\n        File\n      "/home/bviggiano/.cache/uv/builds-v0/.tmpa8JMuK/lib/python3.12/site-packages/hatchling/metadata/core.py",\n      line 750, in license_files\n          raise TypeError(message)\n      TypeError: Field `project.license-files` must be a table\n      hint: This usually indicates a problem with the package or the build\n      environment.\n  help: `biotite` (v1.4.0) was included because `rc-foundry` (v0.1.12) depends\n        on `atomworks` (v2.2.0) which depends on `biotite`\n']
E   assert False
E    +  where False = <[ToolExecutionError('Attempt to access field of tool output after failure:         on `atomworks` (v2.2.0) which depends on `biotite`\n\nError Messages:\n\'ligandmpnn\' may not be compatible with your system. setup.sh failed (exit 1).\n          getattr(self, attribute)\n        File\n      "/home/bviggiano/.cache/uv/builds-v0/.tmpa8JMuK/lib/python3.12/site-packages/hatchling/metadata/core.py",\n      line 750, in license_files\n          raise TypeError(message)\n      TypeError: Field `project.license-files` must be a table\n      hint: This usually indicates a problem with the package or the build\n      environment.\n  help: `biotite` (v1.4.0) was included because `rc-foundry` (v0.1.12) depends\n        on `atomworks` (v2.2.0) which depends on `biotite`\nTraceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 495, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/inverse_folding/ligandmpnn/ligandmpnn_sample.py", line 123, in run_ligandmpnn_sample\n    result = ToolInstance.dis...ne 291, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 949, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 510, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1473, in _create_env\n    raise RuntimeError(\nRuntimeError: \'ligandmpnn\' may not be compatible with your system. setup.sh failed (exit 1).\n          getattr(self, attribute)\n        File\n      "/home/bviggiano/.cache/uv/builds-v0/.tmpa8JMuK/lib/python3.12/site-packages/hatchling/metadata/core.py",\n      line 750, in license_files\n          raise TypeError(message)\n      TypeError: Field `project.license-files` must be a table\n      hint: This usually indicates a problem with the package or the build\n      environment.\n  help: `biotite` (v1.4.0) was included because `rc-foundry` (v0.1.12) depends\n        on `atomworks` (v2.2.0) which depends on `biotite`\n') raised in repr()] InverseFoldingOutput object at 0xe58fa9354c30>.success
```

### ❌ `crispr-tracr`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[crispr-tracr]`

```
tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool crispr-tracr failed: ["'crispr_tracr' may not be compatible with your system. setup.sh failed (exit 1).\nCloning into '/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA'...\nCloning CRISPRidentify into CRISPRtracrRNA tools directory...\nCloning into '/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA/tools/CRISPRidentify/CRISPRidentify'...\nCloning CRISPRcasIdentifier into CRISPRtracrRNA tools directory...\nCloning into '/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA/tools/CRISPRcasIdentifier/CRISPRcasIdentifier'...\nCreating isolated conda environment (Python 3.8 + scikit-learn 0.22)...\nCRISPRidentify's pickled models require sklearn 0.22 (incompatible with 3.12).\nUsing /home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/conda_deps to avoid polluting base env...\nERROR: CRISPRtracrRNA requires x86_64 bioconda packages (vmatch, etc.)\n       that are not available on Linux aarch64.", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 495, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/gene_annotation/crispr_tracr/crispr_tracr.py", line 214, in run_crispr_tracr\n    output_data = ToolInstance.dispatch(\n                  ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 252, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 298, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 949, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 510, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1473, in _create_env\n    raise RuntimeError(\nRuntimeError: \'crispr_tracr\' may not be compatible with your system. setup.sh failed (exit 1).\nCloning into \'/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA\'...\nCloning CRISPRidentify into CRISPRtracrRNA tools directory...\nCloning into \'/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA/tools/CRISPRidentify/CRISPRidentify\'...\nCloning CRISPRcasIdentifier into CRISPRtracrRNA tools directory...\nCloning into \'/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA/tools/CRISPRcasIdentifier/CRISPRcasIdentifier\'...\nCreating isolated conda environment (Python 3.8 + scikit-learn 0.22)...\nCRISPRidentify\'s pickled models require sklearn 0.22 (incompatible with 3.12).\nUsing /home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/conda_deps to avoid polluting base env...\nERROR: CRISPRtracrRNA requires x86_64 bioconda packages (vmatch, etc.)\n       that are not available on Linux aarch64.\n']
E   assert False
E    +  where False = CrisprTracrOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, predictions).success
```

### ❌ `boltz2-prediction`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[boltz2-prediction]`

```
tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool boltz2-prediction failed: ["Command '['/home/bviggiano/.proto/proto_tool_envs/boltz2_env/bin/python', '/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py', '/tmp/tmplfk3mp4m/input.json', '/tmp/tmplfk3mp4m/output.json']' returned non-zero exit status 1.", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 495, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/boltz2.py", line 301, in run_boltz2\n    run_boltz2_on_complex(\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/boltz2.py", line 390, in run_boltz2_on_complex\n    output_data = ToolInstance.dispatch(\n                  ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 252, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 291, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 991, in _run_oneshot\n    subprocess.run(\n  File "/home/bviggiano/miniconda3/envs/proto-tools/lib/python3.12/subprocess.py", line 571, in run\n    raise CalledProcessError(retcode, process.args,\nsubprocess.CalledProcessError: Command \'[\'/home/bviggiano/.proto/proto_tool_envs/boltz2_env/bin/python\', \'/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\', \'/tmp/tmplfk3mp4m/input.json\', \'/tmp/tmplfk3mp4m/output.json\']\' returned non-zero exit status 1.\n']
E   assert False
E    +  where False = <[ToolExecutionError('Attempt to access field of tool output after failure: subprocess.CalledProcessError: Command \'[\'/home/bviggiano/.proto/proto_tool_envs/boltz2_env/bin/python\', \'/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\', \'/tmp/tmplfk3mp4m/input.json\', \'/tmp/tmplfk3mp4m/output.json\']\' returned non-zero exit status 1.\n\nError Messages:\nCommand \'[\'/home/bviggiano/.proto/proto_tool_envs/boltz2_env/bin/python\', \'/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\', \'/tmp/tmplfk3mp4m/input.json\', \'/tmp/tmplfk3mp4m/output.json\']\' returned non-zero exit status 1.\nTraceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 495, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/boltz2.py", line 301, in run_boltz2\n    run_boltz2_on_complex(\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/boltz2.py", line 390, in run_boltz2_on_complex\n    output_data = ToolInstance.dispatch(\n                  ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 252, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 291, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 991, in _run_oneshot\n    subprocess.run(\n  File "/home/bviggiano/miniconda3/envs/proto-tools/lib/python3.12/subprocess.py", line 571, in run\n    raise CalledProcessError(retcode, process.args,\nsubprocess.CalledProcessError: Command \'[\'/home/bviggiano/.proto/proto_tool_envs/boltz2_env/bin/python\', \'/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\', \'/tmp/tmplfk3mp4m/input.json\', \'/tmp/tmplfk3mp4m/output.json\']\' returned non-zero exit status 1.\n') raised in repr()] Boltz2Output object at 0xe58fa9355810>.success
```

### ❌ `progen2-sample`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen2-sample]`

```
tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool progen2-sample failed: ["'progen2' may not be compatible with your system. setup.sh failed (exit 1).\nERROR: ProGen2 is not supported on aarch64.\nProGen2 pins torch==2.2.2 which has no aarch64 CUDA wheel available.", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 495, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/causal_models/progen2/progen2_sample.py", line 256, in run_progen2_sample\n    result = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 252, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 291, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 949, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 510, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1473, in _create_env\n    raise RuntimeError(\nRuntimeError: \'progen2\' may not be compatible with your system. setup.sh failed (exit 1).\nERROR: ProGen2 is not supported on aarch64.\nProGen2 pins torch==2.2.2 which has no aarch64 CUDA wheel available.\n']
E   assert False
E    +  where False = ProGen2SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, logits).success
```

### ❌ `evo2-sample`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo2-sample]`

```
tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool evo2-sample failed: ["'evo2' may not be compatible with your system. setup.sh failed (exit 1).\nERROR: Evo2 is not supported on aarch64.\nEvo2 requires transformer-engine and flash-attn which only provide x86_64 pre-built wheels.", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 495, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/causal_models/evo2/evo2_sample.py", line 259, in run_evo2_sample\n    result = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 252, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 291, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 949, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 510, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1473, in _create_env\n    raise RuntimeError(\nRuntimeError: \'evo2\' may not be compatible with your system. setup.sh failed (exit 1).\nERROR: Evo2 is not supported on aarch64.\nEvo2 requires transformer-engine and flash-attn which only provide x86_64 pre-built wheels.\n']
E   assert False
E    +  where False = Evo2SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, logits, kv_caches).success
```

### ❌ `progen3-sample`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen3-sample]`

```
tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool progen3-sample failed: ["'progen3' may not be compatible with your system. Check logs for details.\nERROR: ProGen3 is not supported on aarch64.\nProGen3 requires flash-attn which has no aarch64 wheels.", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 495, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/causal_models/progen3/progen3_sample.py", line 226, in run_progen3_sample\n    result = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 252, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 291, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 949, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 464, in _ensure_env\n    raise RuntimeError(\nRuntimeError: \'progen3\' may not be compatible with your system. Check logs for details.\nERROR: ProGen3 is not supported on aarch64.\nProGen3 requires flash-attn which has no aarch64 wheels.\n']
E   assert False
E    +  where False = CausalModelSampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success
```

### ❌ `bioemu-sample`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[bioemu-sample]`

```
tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool bioemu-sample failed: ["Command '['/home/bviggiano/.proto/proto_tool_envs/bioemu_env/bin/python', '/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py', '/tmp/tmpg4gkvgi4/input.json', '/tmp/tmpg4gkvgi4/output.json']' returned non-zero exit status 1.", 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 495, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/bioemu_sample.py", line 262, in run_bioemu\n    output = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 252, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 291, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 991, in _run_oneshot\n    subprocess.run(\n  File "/home/bviggiano/miniconda3/envs/proto-tools/lib/python3.12/subprocess.py", line 571, in run\n    raise CalledProcessError(retcode, process.args,\nsubprocess.CalledProcessError: Command \'[\'/home/bviggiano/.proto/proto_tool_envs/bioemu_env/bin/python\', \'/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py\', \'/tmp/tmpg4gkvgi4/input.json\', \'/tmp/tmpg4gkvgi4/output.json\']\' returned non-zero exit status 1.\n']
E   assert False
E    +  where False = BioEmuOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success
```

### ❌ `evo1-sample`

**Test**: `tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo1-sample]`

```
tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
E   AssertionError: Tool evo1-sample failed: ['\'evo1\' may not be compatible with your system. setup.sh failed (exit 1).\n      on `psutil`, but doesn\'t declare it as a build dependency. If\n      `flash-attn` is a first-party package, consider adding `psutil`\n      to its `build-system.requires`. Otherwise, either add it to your\n      `pyproject.toml` under:\n      [tool.uv.extra-build-dependencies]\n      flash-attn = ["psutil"]\n      or `uv pip install psutil` into the environment and re-run with\n      `--no-build-isolation`.\n  help: `flash-attn` (v2.8.3) was included because `evo-model` (v0.5) depends\n        on `stripedhyena` (v0.2.2) which depends on `flash-attn`', 'Traceback (most recent call last):\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py", line 495, in _wrapper_body\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/causal_models/evo1/evo1_sample.py", line 148, in run_evo1_sample\n    result = ToolInstance.dispatch(\n             ^^^^^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 252, in dispatch\n    return cls._oneshot(\n           ^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 291, in _oneshot\n    return inst._run_oneshot(\n           ^^^^^^^^^^^^^^^^^^\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 949, in _run_oneshot\n    self._ensure_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 510, in _ensure_env\n    self._create_env()\n  File "/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py", line 1473, in _create_env\n    raise RuntimeError(\nRuntimeError: \'evo1\' may not be compatible with your system. setup.sh failed (exit 1).\n      on `psutil`, but doesn\'t declare it as a build dependency. If\n      `flash-attn` is a first-party package, consider adding `psutil`\n      to its `build-system.requires`. Otherwise, either add it to your\n      `pyproject.toml` under:\n      [tool.uv.extra-build-dependencies]\n      flash-attn = ["psutil"]\n      or `uv pip install psutil` into the environment and re-run with\n      `--no-build-isolation`.\n  help: `flash-attn` (v2.8.3) was included because `evo-model` (v0.5) depends\n        on `stripedhyena` (v0.2.2) which depends on `flash-attn`\n']
E   assert False
E    +  where False = Evo1SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, scores).success
```

---
*Generated at 2026-04-12 23:31:44 by `pytest --env-report`*

<!-- env-report-data
[
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
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "ablang-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ablang-embedding]",
    "status": "passed",
    "duration_seconds": 129.96,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/ablang_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "structure-metrics",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[structure-metrics]",
    "status": "passed",
    "duration_seconds": 15.19,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/structure_metrics_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "usalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[usalign-alignment]",
    "status": "passed",
    "duration_seconds": 16.25,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/usalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
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
    "error_message": "('/home/bviggiano/codebases/proto/proto-tools/tests/tool_infra_tests/test_env_report.py', 72, 'Skipped: --env-report: requires Chimera cluster')",
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "enformer-prediction",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[enformer-prediction]",
    "status": "passed",
    "duration_seconds": 30.49,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/enformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "mock-pytorch-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-pytorch-tool-run]",
    "status": "passed",
    "duration_seconds": 18.09,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/mock_pytorch_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
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
    "error_message": "('/home/bviggiano/codebases/proto/proto-tools/tests/tool_infra_tests/test_env_report.py', 72, 'Skipped: --env-report: requires 2 GPUs, only 1 visible')",
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "segmasker-score",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[segmasker-score]",
    "status": "failed",
    "duration_seconds": 61.69,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/segmasker_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool segmasker-score failed: [\"Command '['/home/bviggiano/.proto/proto_tool_envs/segmasker_env/bin/python', '/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_scoring/segmasker/standalone/run.py', '/tmp/tmp5iyjxewg/input.json', '/tmp/tmp5iyjxewg/output.json']' returned non-zero exit status 1.\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 495, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_scoring/segmasker/segmasker.py\", line 235, in run_segmasker\\n    output_data = ToolInstance.dispatch(\\n                  ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 252, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 298, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 991, in _run_oneshot\\n    subprocess.run(\\n  File \"/home/bviggiano/miniconda3/envs/proto-tools/lib/python3.12/subprocess.py\", line 571, in run\\n    raise CalledProcessError(retcode, process.args,\\nsubprocess.CalledProcessError: Command \\'[\\'/home/bviggiano/.proto/proto_tool_envs/segmasker_env/bin/python\\', \\'/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_scoring/segmasker/standalone/run.py\\', \\'/tmp/tmp5iyjxewg/input.json\\', \\'/tmp/tmp5iyjxewg/output.json\\']\\' returned non-zero exit status 1.\\n']\nE   assert False\nE    +  where False = SegmaskerOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success",
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "esmfold-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esmfold-prediction]",
    "status": "passed",
    "duration_seconds": 159.74,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/esmfold_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "mafft-align",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mafft-align]",
    "status": "failed",
    "duration_seconds": 7.08,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/mafft_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool mafft-align failed: ['\\'mafft\\' may not be compatible with your system. setup.sh failed (exit 1).\\n    install_binary(sys.argv[1])\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/install_binary.py\", line 179, in install_binary\\n    config.extract(archive_path, bin_dir)\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_alignment/mafft/standalone/binary_config.py\", line 163, in extract\\n    _extract_from_source(archive_path, bin_dir)\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_alignment/mafft/standalone/binary_config.py\", line 130, in _extract_from_source\\n    subprocess.check_call(\\n  File \"/home/bviggiano/.proto/proto_tool_envs/mafft_env/lib/python3.12/subprocess.py\", line 413, in check_call\\n    raise CalledProcessError(retcode, cmd)\\nsubprocess.CalledProcessError: Command \\'[\\'make\\', \\'-j4\\', \\'PREFIX=/home/bviggiano/.proto/proto_tool_envs/mafft_env\\', \\'BINDIR=/home/bviggiano/.proto/proto_tool_envs/mafft_env/bin\\', \\'LIBDIR=/home/bviggiano/.proto/proto_tool_envs/mafft_env/libexec/mafft\\']\\' returned non-zero exit status 2.', 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 495, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_alignment/mafft/mafft.py\", line 195, in run_mafft_align\\n    output_data = ToolInstance.dispatch(\\n                  ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 252, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 298, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 949, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 510, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1473, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'mafft\\' may not be compatible with your system. setup.sh failed (exit 1).\\n    install_binary(sys.argv[1])\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/install_binary.py\", line 179, in install_binary\\n    config.extract(archive_path, bin_dir)\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_alignment/mafft/standalone/binary_config.py\", line 163, in extract\\n    _extract_from_source(archive_path, bin_dir)\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/sequence_alignment/mafft/standalone/binary_config.py\", line 130, in _extract_from_source\\n    subprocess.check_call(\\n  File \"/home/bviggiano/.proto/proto_tool_envs/mafft_env/lib/python3.12/subprocess.py\", line 413, in check_call\\n    raise CalledProcessError(retcode, cmd)\\nsubprocess.CalledProcessError: Command \\'[\\'make\\', \\'-j4\\', \\'PREFIX=/home/bviggiano/.proto/proto_tool_envs/mafft_env\\', \\'BINDIR=/home/bviggiano/.proto/proto_tool_envs/mafft_env/bin\\', \\'LIBDIR=/home/bviggiano/.proto/proto_tool_envs/mafft_env/libexec/mafft\\']\\' returned non-zero exit status 2.\\n']\nE   assert False\nE    +  where False = MafftOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success",
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "pyrosetta-energy",
    "category": "structure_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pyrosetta-energy]",
    "status": "passed",
    "duration_seconds": 360.83,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/pyrosetta_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "mmseqs-clustering",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mmseqs-clustering]",
    "status": "passed",
    "duration_seconds": 5.32,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/mmseqs_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
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
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "proteinmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[proteinmpnn-sample]",
    "status": "passed",
    "duration_seconds": 113.31,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/proteinmpnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "protenix-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[protenix-prediction]",
    "status": "failed",
    "duration_seconds": 33.74,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/protenix_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool protenix-prediction failed: [\"'protenix' may not be compatible with your system. setup.sh failed (exit 1).\\n      Model& model) {\\n            |\\n      ^\\n      ninja: build stopped: subcommand failed.\\n      [stderr]\\n      *** CMake build failed\\n      hint: This usually indicates a problem with the package or the build\\n      environment.\\n  help: `gemmi` (v0.6.7) was included because `protenix` (v2.0.0) depends\\n        on `gemmi`\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 495, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/protenix/protenix.py\", line 422, in run_protenix\\n    output_data = ToolInstance.dispatch(\\n                  ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 252, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 291, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 949, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 510, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1473, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'protenix\\' may not be compatible with your system. setup.sh failed (exit 1).\\n      Model& model) {\\n            |\\n      ^\\n      ninja: build stopped: subcommand failed.\\n      [stderr]\\n      *** CMake build failed\\n      hint: This usually indicates a problem with the package or the build\\n      environment.\\n  help: `gemmi` (v0.6.7) was included because `protenix` (v2.0.0) depends\\n        on `gemmi`\\n']\nE   assert False\nE    +  where False = <[ToolExecutionError('Attempt to access field of tool output after failure:         on `gemmi`\\n\\nError Messages:\\n\\'protenix\\' may not be compatible with your system. setup.sh failed (exit 1).\\n      Model& model) {\\n            |\\n      ^\\n      ninja: build stopped: subcommand failed.\\n      [stderr]\\n      *** CMake build failed\\n      hint: This usually indicates a problem with the package or the build\\n      environment.\\n  help: `gemmi` (v0.6.7) was included because `protenix` (v2.0.0) depends\\n        on `gemmi`\\nTraceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 495, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/protenix/protenix.py\", line 422, in run_protenix\\n    output_data = ToolInstance.dispatch(\\n                  ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 252, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 291, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 949, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 510, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1473, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'protenix\\' may not be compatible with your system. setup.sh failed (exit 1).\\n      Model& model) {\\n            |\\n      ^\\n      ninja: build stopped: subcommand failed.\\n      [stderr]\\n      *** CMake build failed\\n      hint: This usually indicates a problem with the package or the build\\n      environment.\\n  help: `gemmi` (v0.6.7) was included because `protenix` (v2.0.0) depends\\n        on `gemmi`\\n') raised in repr()] ProtenixOutput object at 0xe58fa9e52030>.success",
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "esm2-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm2-embedding]",
    "status": "passed",
    "duration_seconds": 53.43,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/esm2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "chai1-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[chai1-prediction]",
    "status": "failed",
    "duration_seconds": 1.92,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/chai1_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool chai1-prediction failed: [\"'chai1' may not be compatible with your system. setup.sh failed (exit 1).\\nERROR: Chai is not supported on aarch64.\\nchai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its\\npre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 495, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py\", line 291, in run_chai1\\n    run_chai1_on_complex(comp=comp, config=config, msas=inputs.msas, instance=instance)\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py\", line 374, in run_chai1_on_complex\\n    result = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 252, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 291, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 949, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 510, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1473, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'chai1\\' may not be compatible with your system. setup.sh failed (exit 1).\\nERROR: Chai is not supported on aarch64.\\nchai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its\\npre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\\n']\nE   assert False\nE    +  where False = <[ToolExecutionError('Attempt to access field of tool output after failure: pre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\\n\\nError Messages:\\n\\'chai1\\' may not be compatible with your system. setup.sh failed (exit 1).\\nERROR: Chai is not supported on aarch64.\\nchai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its\\npre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\\nTraceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 495, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py\", line 291, in run_chai1\\n    run_chai1_on_complex(comp=comp, config=config, msas=inputs.msas, instance=instance)\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/chai1/chai1.py\", line 374, in run_chai1_on_complex\\n    result = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 252, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 291, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 949, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 510, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1473, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'chai1\\' may not be compatible with your system. setup.sh failed (exit 1).\\nERROR: Chai is not supported on aarch64.\\nchai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its\\npre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.\\n') raised in repr()] Chai1Output object at 0xe58fa93546e0>.success",
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "ligandmpnn-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[ligandmpnn-sample]",
    "status": "failed",
    "duration_seconds": 32.8,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/ligandmpnn_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool ligandmpnn-sample failed: ['\\'ligandmpnn\\' may not be compatible with your system. setup.sh failed (exit 1).\\n          getattr(self, attribute)\\n        File\\n      \"/home/bviggiano/.cache/uv/builds-v0/.tmpa8JMuK/lib/python3.12/site-packages/hatchling/metadata/core.py\",\\n      line 750, in license_files\\n          raise TypeError(message)\\n      TypeError: Field `project.license-files` must be a table\\n      hint: This usually indicates a problem with the package or the build\\n      environment.\\n  help: `biotite` (v1.4.0) was included because `rc-foundry` (v0.1.12) depends\\n        on `atomworks` (v2.2.0) which depends on `biotite`', 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 495, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/inverse_folding/ligandmpnn/ligandmpnn_sample.py\", line 123, in run_ligandmpnn_sample\\n    result = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 252, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 291, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 949, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 510, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1473, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'ligandmpnn\\' may not be compatible with your system. setup.sh failed (exit 1).\\n          getattr(self, attribute)\\n        File\\n      \"/home/bviggiano/.cache/uv/builds-v0/.tmpa8JMuK/lib/python3.12/site-packages/hatchling/metadata/core.py\",\\n      line 750, in license_files\\n          raise TypeError(message)\\n      TypeError: Field `project.license-files` must be a table\\n      hint: This usually indicates a problem with the package or the build\\n      environment.\\n  help: `biotite` (v1.4.0) was included because `rc-foundry` (v0.1.12) depends\\n        on `atomworks` (v2.2.0) which depends on `biotite`\\n']\nE   assert False\nE    +  where False = <[ToolExecutionError('Attempt to access field of tool output after failure:         on `atomworks` (v2.2.0) which depends on `biotite`\\n\\nError Messages:\\n\\'ligandmpnn\\' may not be compatible with your system. setup.sh failed (exit 1).\\n          getattr(self, attribute)\\n        File\\n      \"/home/bviggiano/.cache/uv/builds-v0/.tmpa8JMuK/lib/python3.12/site-packages/hatchling/metadata/core.py\",\\n      line 750, in license_files\\n          raise TypeError(message)\\n      TypeError: Field `project.license-files` must be a table\\n      hint: This usually indicates a problem with the package or the build\\n      environment.\\n  help: `biotite` (v1.4.0) was included because `rc-foundry` (v0.1.12) depends\\n        on `atomworks` (v2.2.0) which depends on `biotite`\\nTraceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 495, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/inverse_folding/ligandmpnn/ligandmpnn_sample.py\", line 123, in run_ligandmpnn_sample\\n    result = ToolInstance.dis...ne 291, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 949, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 510, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1473, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'ligandmpnn\\' may not be compatible with your system. setup.sh failed (exit 1).\\n          getattr(self, attribute)\\n        File\\n      \"/home/bviggiano/.cache/uv/builds-v0/.tmpa8JMuK/lib/python3.12/site-packages/hatchling/metadata/core.py\",\\n      line 750, in license_files\\n          raise TypeError(message)\\n      TypeError: Field `project.license-files` must be a table\\n      hint: This usually indicates a problem with the package or the build\\n      environment.\\n  help: `biotite` (v1.4.0) was included because `rc-foundry` (v0.1.12) depends\\n        on `atomworks` (v2.2.0) which depends on `biotite`\\n') raised in repr()] InverseFoldingOutput object at 0xe58fa9354c30>.success",
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "minced-crispr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[minced-crispr]",
    "status": "passed",
    "duration_seconds": 3.11,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/minced_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "crispr-tracr",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[crispr-tracr]",
    "status": "failed",
    "duration_seconds": 13.65,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool crispr-tracr failed: [\"'crispr_tracr' may not be compatible with your system. setup.sh failed (exit 1).\\nCloning into '/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA'...\\nCloning CRISPRidentify into CRISPRtracrRNA tools directory...\\nCloning into '/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA/tools/CRISPRidentify/CRISPRidentify'...\\nCloning CRISPRcasIdentifier into CRISPRtracrRNA tools directory...\\nCloning into '/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA/tools/CRISPRcasIdentifier/CRISPRcasIdentifier'...\\nCreating isolated conda environment (Python 3.8 + scikit-learn 0.22)...\\nCRISPRidentify's pickled models require sklearn 0.22 (incompatible with 3.12).\\nUsing /home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/conda_deps to avoid polluting base env...\\nERROR: CRISPRtracrRNA requires x86_64 bioconda packages (vmatch, etc.)\\n       that are not available on Linux aarch64.\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 495, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/gene_annotation/crispr_tracr/crispr_tracr.py\", line 214, in run_crispr_tracr\\n    output_data = ToolInstance.dispatch(\\n                  ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 252, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 298, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 949, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 510, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1473, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'crispr_tracr\\' may not be compatible with your system. setup.sh failed (exit 1).\\nCloning into \\'/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA\\'...\\nCloning CRISPRidentify into CRISPRtracrRNA tools directory...\\nCloning into \\'/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA/tools/CRISPRidentify/CRISPRidentify\\'...\\nCloning CRISPRcasIdentifier into CRISPRtracrRNA tools directory...\\nCloning into \\'/home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/CRISPRtracrRNA/tools/CRISPRcasIdentifier/CRISPRcasIdentifier\\'...\\nCreating isolated conda environment (Python 3.8 + scikit-learn 0.22)...\\nCRISPRidentify\\'s pickled models require sklearn 0.22 (incompatible with 3.12).\\nUsing /home/bviggiano/.proto/proto_tool_envs/crispr_tracr_env/conda_deps to avoid polluting base env...\\nERROR: CRISPRtracrRNA requires x86_64 bioconda packages (vmatch, etc.)\\n       that are not available on Linux aarch64.\\n']\nE   assert False\nE    +  where False = CrisprTracrOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, predictions).success",
    "git_commit": "16714ef3b882",
    "git_dirty": true
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
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "viennarna-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[viennarna-prediction]",
    "status": "passed",
    "duration_seconds": 3.8,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/viennarna_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "orfipy-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[orfipy-prediction]",
    "status": "passed",
    "duration_seconds": 31.75,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/orfipy_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "boltz2-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[boltz2-prediction]",
    "status": "failed",
    "duration_seconds": 150.09,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/boltz2_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool boltz2-prediction failed: [\"Command '['/home/bviggiano/.proto/proto_tool_envs/boltz2_env/bin/python', '/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py', '/tmp/tmplfk3mp4m/input.json', '/tmp/tmplfk3mp4m/output.json']' returned non-zero exit status 1.\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 495, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/boltz2.py\", line 301, in run_boltz2\\n    run_boltz2_on_complex(\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/boltz2.py\", line 390, in run_boltz2_on_complex\\n    output_data = ToolInstance.dispatch(\\n                  ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 252, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 291, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 991, in _run_oneshot\\n    subprocess.run(\\n  File \"/home/bviggiano/miniconda3/envs/proto-tools/lib/python3.12/subprocess.py\", line 571, in run\\n    raise CalledProcessError(retcode, process.args,\\nsubprocess.CalledProcessError: Command \\'[\\'/home/bviggiano/.proto/proto_tool_envs/boltz2_env/bin/python\\', \\'/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\\', \\'/tmp/tmplfk3mp4m/input.json\\', \\'/tmp/tmplfk3mp4m/output.json\\']\\' returned non-zero exit status 1.\\n']\nE   assert False\nE    +  where False = <[ToolExecutionError('Attempt to access field of tool output after failure: subprocess.CalledProcessError: Command \\'[\\'/home/bviggiano/.proto/proto_tool_envs/boltz2_env/bin/python\\', \\'/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\\', \\'/tmp/tmplfk3mp4m/input.json\\', \\'/tmp/tmplfk3mp4m/output.json\\']\\' returned non-zero exit status 1.\\n\\nError Messages:\\nCommand \\'[\\'/home/bviggiano/.proto/proto_tool_envs/boltz2_env/bin/python\\', \\'/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\\', \\'/tmp/tmplfk3mp4m/input.json\\', \\'/tmp/tmplfk3mp4m/output.json\\']\\' returned non-zero exit status 1.\\nTraceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 495, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/boltz2.py\", line 301, in run_boltz2\\n    run_boltz2_on_complex(\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/boltz2.py\", line 390, in run_boltz2_on_complex\\n    output_data = ToolInstance.dispatch(\\n                  ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 252, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 291, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 991, in _run_oneshot\\n    subprocess.run(\\n  File \"/home/bviggiano/miniconda3/envs/proto-tools/lib/python3.12/subprocess.py\", line 571, in run\\n    raise CalledProcessError(retcode, process.args,\\nsubprocess.CalledProcessError: Command \\'[\\'/home/bviggiano/.proto/proto_tool_envs/boltz2_env/bin/python\\', \\'/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_prediction/boltz2/standalone/inference.py\\', \\'/tmp/tmplfk3mp4m/input.json\\', \\'/tmp/tmplfk3mp4m/output.json\\']\\' returned non-zero exit status 1.\\n') raised in repr()] Boltz2Output object at 0xe58fa9355810>.success",
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "tmalign-alignment",
    "category": "structure_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[tmalign-alignment]",
    "status": "passed",
    "duration_seconds": 9.8,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/tmalign_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "alphagenome-predict-intervals",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphagenome-predict-intervals]",
    "status": "passed",
    "duration_seconds": 99.27,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/alphagenome_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
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
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "esm-if1-sample",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm-if1-sample]",
    "status": "passed",
    "duration_seconds": 197.25,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/esm_if1_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "mock-cli-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-cli-tool-run]",
    "status": "passed",
    "duration_seconds": 2.07,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/mock_cli_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "splice-transformer-prediction",
    "category": "rna_splicing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[splice-transformer-prediction]",
    "status": "passed",
    "duration_seconds": 20.83,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/splice_transformer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "mock-jax-tool-run",
    "category": "testing",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[mock-jax-tool-run]",
    "status": "passed",
    "duration_seconds": 8.86,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/mock_jax_tool_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "progen2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen2-sample]",
    "status": "failed",
    "duration_seconds": 1.83,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/progen2_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool progen2-sample failed: [\"'progen2' may not be compatible with your system. setup.sh failed (exit 1).\\nERROR: ProGen2 is not supported on aarch64.\\nProGen2 pins torch==2.2.2 which has no aarch64 CUDA wheel available.\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 495, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/causal_models/progen2/progen2_sample.py\", line 256, in run_progen2_sample\\n    result = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 252, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 291, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 949, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 510, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1473, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'progen2\\' may not be compatible with your system. setup.sh failed (exit 1).\\nERROR: ProGen2 is not supported on aarch64.\\nProGen2 pins torch==2.2.2 which has no aarch64 CUDA wheel available.\\n']\nE   assert False\nE    +  where False = ProGen2SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, logits).success",
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "evo2-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo2-sample]",
    "status": "failed",
    "duration_seconds": 2.04,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/evo2_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool evo2-sample failed: [\"'evo2' may not be compatible with your system. setup.sh failed (exit 1).\\nERROR: Evo2 is not supported on aarch64.\\nEvo2 requires transformer-engine and flash-attn which only provide x86_64 pre-built wheels.\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 495, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/causal_models/evo2/evo2_sample.py\", line 259, in run_evo2_sample\\n    result = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 252, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 291, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 949, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 510, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1473, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'evo2\\' may not be compatible with your system. setup.sh failed (exit 1).\\nERROR: Evo2 is not supported on aarch64.\\nEvo2 requires transformer-engine and flash-attn which only provide x86_64 pre-built wheels.\\n']\nE   assert False\nE    +  where False = Evo2SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, logits, kv_caches).success",
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "alphafold2-prediction",
    "category": "structure_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[alphafold2-prediction]",
    "status": "passed",
    "duration_seconds": 90.63,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/alphafold2_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "blast-create-db",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[blast-create-db]",
    "status": "passed",
    "duration_seconds": 70.8,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/blast_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "borzoi-ensemble",
    "category": "sequence_scoring",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[borzoi-ensemble]",
    "status": "passed",
    "duration_seconds": 116.23,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/borzoi_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "progen3-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[progen3-sample]",
    "status": "failed",
    "duration_seconds": 0.05,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/progen3_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool progen3-sample failed: [\"'progen3' may not be compatible with your system. Check logs for details.\\nERROR: ProGen3 is not supported on aarch64.\\nProGen3 requires flash-attn which has no aarch64 wheels.\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 495, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/causal_models/progen3/progen3_sample.py\", line 226, in run_progen3_sample\\n    result = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 252, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 291, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 949, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 464, in _ensure_env\\n    raise RuntimeError(\\nRuntimeError: \\'progen3\\' may not be compatible with your system. Check logs for details.\\nERROR: ProGen3 is not supported on aarch64.\\nProGen3 requires flash-attn which has no aarch64 wheels.\\n']\nE   assert False\nE    +  where False = CausalModelSampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success",
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "prodigal-prediction",
    "category": "orf_prediction",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[prodigal-prediction]",
    "status": "passed",
    "duration_seconds": 3.66,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/prodigal_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "bioemu-sample",
    "category": "structure_dynamics",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[bioemu-sample]",
    "status": "failed",
    "duration_seconds": 82.22,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/bioemu_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool bioemu-sample failed: [\"Command '['/home/bviggiano/.proto/proto_tool_envs/bioemu_env/bin/python', '/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py', '/tmp/tmpg4gkvgi4/input.json', '/tmp/tmpg4gkvgi4/output.json']' returned non-zero exit status 1.\", 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 495, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/bioemu_sample.py\", line 262, in run_bioemu\\n    output = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 252, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 291, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 991, in _run_oneshot\\n    subprocess.run(\\n  File \"/home/bviggiano/miniconda3/envs/proto-tools/lib/python3.12/subprocess.py\", line 571, in run\\n    raise CalledProcessError(retcode, process.args,\\nsubprocess.CalledProcessError: Command \\'[\\'/home/bviggiano/.proto/proto_tool_envs/bioemu_env/bin/python\\', \\'/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/structure_dynamics/bioemu/standalone/inference.py\\', \\'/tmp/tmpg4gkvgi4/input.json\\', \\'/tmp/tmpg4gkvgi4/output.json\\']\\' returned non-zero exit status 1.\\n']\nE   assert False\nE    +  where False = BioEmuOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success",
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "colabfold-search",
    "category": "sequence_alignment",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[colabfold-search]",
    "status": "passed",
    "duration_seconds": 33.91,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/colabfold_search_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "rfdiffusion3-design",
    "category": "structure_design",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[rfdiffusion3-design]",
    "status": "passed",
    "duration_seconds": 1548.68,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/rfdiffusion3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "esm3-embedding",
    "category": "masked_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[esm3-embedding]",
    "status": "passed",
    "duration_seconds": 61.87,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/esm3_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "evo1-sample",
    "category": "causal_models",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[evo1-sample]",
    "status": "failed",
    "duration_seconds": 49.55,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/evo1_env",
    "env_status": "success",
    "error_message": "tests/tool_infra_tests/test_env_report.py:78: in test_tool_env_report\n    assert result.success, f\"Tool {spec.key} failed: {result.errors}\"\nE   AssertionError: Tool evo1-sample failed: ['\\'evo1\\' may not be compatible with your system. setup.sh failed (exit 1).\\n      on `psutil`, but doesn\\'t declare it as a build dependency. If\\n      `flash-attn` is a first-party package, consider adding `psutil`\\n      to its `build-system.requires`. Otherwise, either add it to your\\n      `pyproject.toml` under:\\n      [tool.uv.extra-build-dependencies]\\n      flash-attn = [\"psutil\"]\\n      or `uv pip install psutil` into the environment and re-run with\\n      `--no-build-isolation`.\\n  help: `flash-attn` (v2.8.3) was included because `evo-model` (v0.5) depends\\n        on `stripedhyena` (v0.2.2) which depends on `flash-attn`', 'Traceback (most recent call last):\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/tool_registry.py\", line 495, in _wrapper_body\\n    result = func(inputs, config, instance)\\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/tools/causal_models/evo1/evo1_sample.py\", line 148, in run_evo1_sample\\n    result = ToolInstance.dispatch(\\n             ^^^^^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 252, in dispatch\\n    return cls._oneshot(\\n           ^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 291, in _oneshot\\n    return inst._run_oneshot(\\n           ^^^^^^^^^^^^^^^^^^\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 949, in _run_oneshot\\n    self._ensure_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 510, in _ensure_env\\n    self._create_env()\\n  File \"/home/bviggiano/codebases/proto/proto-tools/proto_tools/utils/tool_instance.py\", line 1473, in _create_env\\n    raise RuntimeError(\\nRuntimeError: \\'evo1\\' may not be compatible with your system. setup.sh failed (exit 1).\\n      on `psutil`, but doesn\\'t declare it as a build dependency. If\\n      `flash-attn` is a first-party package, consider adding `psutil`\\n      to its `build-system.requires`. Otherwise, either add it to your\\n      `pyproject.toml` under:\\n      [tool.uv.extra-build-dependencies]\\n      flash-attn = [\"psutil\"]\\n      or `uv pip install psutil` into the environment and re-run with\\n      `--no-build-isolation`.\\n  help: `flash-attn` (v2.8.3) was included because `evo-model` (v0.5) depends\\n        on `stripedhyena` (v0.2.2) which depends on `flash-attn`\\n']\nE   assert False\nE    +  where False = Evo1SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, scores).success",
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "fampnn-pack",
    "category": "inverse_folding",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[fampnn-pack]",
    "status": "passed",
    "duration_seconds": 47.87,
    "uses_gpu": true,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/fampnn_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
  },
  {
    "tool_key": "pyhmmer-hmmscan",
    "category": "gene_annotation",
    "test_name": "tests/tool_infra_tests/test_env_report.py::test_tool_env_report[pyhmmer-hmmscan]",
    "status": "passed",
    "duration_seconds": 3.91,
    "uses_gpu": false,
    "env_path": "/home/bviggiano/.proto/proto_tool_envs/pyhmmer_env",
    "env_status": "success",
    "error_message": null,
    "git_commit": "16714ef3b882",
    "git_dirty": true
  }
]
-->