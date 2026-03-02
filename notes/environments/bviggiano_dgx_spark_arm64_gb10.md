# DGX Spark Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-72%25-yellow) ![Passed](https://img.shields.io/badge/passed-24-brightgreen) ![Failed](https://img.shields.io/badge/failed-9-red) ![Skipped](https://img.shields.io/badge/skipped-0-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Linux Linux 6.11.0-1016-nvidia |
| **Architecture** | aarch64 |
| **Hostname** | `spark-3b18` |
| **Python** | 3.12.12 |
| **RAM** | 119.7 GB |
| **GPU** | 1× NVIDIA GB10 |
| **CUDA** | 13.0 |
| **Conda Env** | `bio_tools` |

## Git

- **Commit**: `b75009443392`
- **Branch**: `device_management`
- **Dirty**: Yes

## Environment Variables

### Parent Process Environment

```
CLAUDECODE=1
CLAUDE_CODE_ENTRYPOINT=cli
CONDA_DEFAULT_ENV=bio_tools
CONDA_EXE=/home/bviggiano/miniconda3/bin/conda
CONDA_PREFIX=/home/bviggiano/miniconda3/envs/bio_tools
CONDA_PREFIX_1=/home/bviggiano/miniconda3
CONDA_PROMPT_MODIFIER=(bio_tools) 
CONDA_PYTHON_EXE=/home/bviggiano/miniconda3/bin/python
CONDA_SHLVL=2
COREPACK_ENABLE_AUTO_PIN=0
DEBUGINFOD_URLS=https://debuginfod.ubuntu.com 
DISABLE_PANDERA_IMPORT_WARNING=True
GIT_EDITOR=true
HOME=/home/bviggiano
LANG=en_US.utf8
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64:
LESSCLOSE=/usr/bin/lesspipe %s %s
LESSOPEN=| /usr/bin/lesspipe %s
LOGNAME=bviggiano
LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=00:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.arj=...
NoDefaultCurrentDirectoryInExePath=1
OLDPWD=/home/bviggiano/codebases/bio-programming
OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta
PATH=/home/bviggiano/.local/bin:/home/bviggiano/.local/bin:/usr/local/cuda/bin:/usr/local/cuda/bin:/opt/bin:/home/bviggiano/.local/bin:/home/bviggiano/.local/bin:/usr/local/cuda/bin:/home/bviggiano/minicon...
PWD=/home/bviggiano/codebases/bio-programming/bio-programming-tools
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
RDBASE=/home/bviggiano/miniconda3/envs/bio_tools/lib/python3.12/site-packages/rdkit
SHELL=/bin/bash
SHLVL=3
TERM=tmux-256color
TERM_PROGRAM=tmux
TERM_PROGRAM_VERSION=3.4
TMUX=/tmp/tmux-1001/default,734518,0
TMUX_PANE=%0
USER=bviggiano
XDG_DATA_DIRS=/usr/share/gnome:/usr/local/share:/usr/share:/var/lib/snapd/desktop
XDG_RUNTIME_DIR=/run/user/1001
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
_=/home/bviggiano/miniconda3/envs/bio_tools/bin/pytest
_CE_CONDA=
_CE_M=
_CONDA_EXE=/home/bviggiano/miniconda3/bin/conda
_CONDA_ROOT=/home/bviggiano/miniconda3
```

### Subprocess Environment (passed to tools)

```
CONDA_PREFIX=/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/splice_transformer_env
DETECTED_COMPUTE_PLATFORM=cuda
DETECTED_CUDA_VERSION=13
DETECTED_DRIVER_VERSION=580
HOME=/home/bviggiano
LANG=en_US.utf8
LD_LIBRARY_PATH=/home/bviggiano/codebases/bio-programming/bio-programming-tools/.micromamba/foundation_env/lib:/lib/aarch64-linux-gnu
LOGNAME=bviggiano
PATH=/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/splice_transformer_env/bin:/usr/local/cuda/bin:/home/bviggiano/codebases/bio-programming/bio-programming-tools/.micromamba/fou...
RECOMMENDED_JAX_SPEC=jax[cuda13]>=0.4.20,<1
RECOMMENDED_JAX_VARIANT=cuda13
RECOMMENDED_TORCH_SPEC=torch>=2.8,<3
SHELL=/bin/bash
TORCH_CUDA_ARCH_LIST=12.0
TORCH_HOME=/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/splice_transformer_env/cache/torch
USER=bviggiano
VIRTUAL_ENV=/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/splice_transformer_env
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Results by Category

### Causal Models (0/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `evo1` | yes | ✅ | 58.1s | ❌ Fail |
| `evo2` | yes | ✅ | 2.6s | ❌ Fail |
| `progen2` | yes | ✅ | 2.6s | ❌ Fail |

### Gene Annotation (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `blast` | no | ✅ | 61.7s | ✅ Pass |
| `minced` | no | ✅ | 4.2s | ✅ Pass |
| `mmseqs` | no | ✅ | 6.2s | ✅ Pass |
| `pyhmmer` | no | ✅ | 3.9s | ✅ Pass |

### Inverse Folding (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `ligandmpnn` | yes | ✅ | 121.7s | ✅ Pass |
| `proteinmpnn` | yes | ✅ | 31.4s | ✅ Pass |

### Masked Models (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `esm2` | yes | ✅ | 39.0s | ✅ Pass |
| `esm3` | yes | ✅ | 17.5s | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `orfipy` | no | ✅ | 4.2s | ✅ Pass |
| `prodigal` | no | ✅ | 3.5s | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `splice_transformer` | yes | ✅ | 10.5s | ✅ Pass |

### Sequence Alignment (0/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `mafft` | no | ✅ | 11.7s | ❌ Fail |

### Sequence Scoring (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `borzoi` | yes | ✅ | 22.5s | ✅ Pass |
| `enformer` | yes | ✅ | 31.2s | ✅ Pass |

### Structure Alignment (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `tmalign` | no | ✅ | 13.5s | ✅ Pass |
| `tmalign` | no | ✅ | 0.0s | ✅ Pass |
| `usalign` | no | ✅ | 21.1s | ✅ Pass |
| `usalign` | no | ✅ | 0.1s | ✅ Pass |

### Structure Design (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `rfdiffusion3` | yes | ✅ | 227.1s | ✅ Pass |

### Structure Dynamics (0/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `bioemu` | yes | ✅ | 28.8s | ❌ Fail |

### Structure Prediction (3/6)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `alphafold2` | yes | ✅ | 111.1s | ✅ Pass |
| `boltz2` | yes | ✅ | 2411.7s | ❌ Fail |
| `chai1` | yes | ✅ | 6.8s | ❌ Fail |
| `esmfold` | yes | ✅ | 56.8s | ✅ Pass |
| `protenix` | yes | ✅ | 2530.9s | ❌ Fail |
| `viennarna` | no | ✅ | 7.2s | ✅ Pass |

### Unknown (3/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `alphagenome` | yes | ✅ | 134.6s | ✅ Pass |
| `crispr_tracr` | no | ✅ | 14.2s | ❌ Fail |
| `local_colabfold_search` | no | — | 40.3s | ✅ Pass |
| `structure_metrics` | no | ✅ | 4.4s | ✅ Pass |

## Failure Details

### ❌ `crispr_tracr`

**Test**: `tests/gene_annotation_tests/test_crispr_tracr.py::TestCrisprTracrIntegration::test_run_crispr_tracr`

```
tests/gene_annotation_tests/test_crispr_tracr.py:320: in test_run_crispr_tracr
    assert len(result.predictions) == 1
E   assert 0 == 1
E    +  where 0 = len([])
E    +    where [] = CrisprTracrOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, predictions).predictions
```

### ❌ `evo1`

**Test**: `tests/language_model_tests/test_evo1.py::test_evo1_sample_tool`

```
tests/language_model_tests/test_evo1.py:100: in test_evo1_sample_tool
    validate_output(result)
tests/tool_infra_tests/test_export_functionality.py:102: in validate_output
    assert output.success is True, f"Tool execution failed: {output}"
E   AssertionError: Tool execution failed: 
E     ================================================================================
E     evo1-sample: TOOL FAILURE after 58.0870s
E     ================================================================================
E     
E     Error 1:
E     'evo1' may not be compatible with your system. setup.sh failed (exit 1).
E           on `psutil`, but doesn't declare it as a build dependency. If
E           `flash-attn` is a first-party package, consider adding `psutil`
E           to its `build-system.requires`. Otherwise, either add it to your
E           `pyproject.toml` under:
E           [tool.uv.extra-build-dependencies]
E           flash-attn = ["psutil"]
E           or `uv pip install psutil` into the environment and re-run with
E           `--no-build-isolation`.
E       help: `flash-attn` (v2.8.3) was included because `evo-model` (v0.5) depends
E             on `stripedhyena` (v0.2.2) which depends on `flash-attn`
E     
E     Error 2:
E     Traceback (most recent call last):
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 245, in wrapper
E         result = func(inputs, config, instance)
E                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/causal_models/evo1/evo1_sample.py", line 215, in run_evo1_sample
E         result = ToolInstance.dispatch(
E                  ^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 245, in dispatch
E         return cached.run(
E                ^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 552, in run
E         return self._run_persistent(
E                ^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 847, in _run_persistent
E         self._ensure_env()
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 513, in _ensure_env
E         self._create_env()
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 1407, in _create_env
E         raise RuntimeError(
E     RuntimeError: 'evo1' may not be compatible with your system. setup.sh failed (exit 1).
E           on `psutil`, but doesn't declare it as a build dependency. If
E           `flash-attn` is a first-party package, consider adding `psutil`
E           to its `build-system.requires`. Otherwise, either add it to your
E           `pyproject.toml` under:
E           [tool.uv.extra-build-dependencies]
E           flash-attn = ["psutil"]
E           or `uv pip install psutil` into the environment and re-run with
E           `--no-build-isolation`.
E       help: `flash-attn` (v2.8.3) was included because `evo-model` (v0.5) depends
E             on `stripedhyena` (v0.2.2) which depends on `flash-attn`
E     
E     ================================================================================
E   assert False is True
E    +  where False = Evo1SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, scores).success
```

### ❌ `evo2`

**Test**: `tests/language_model_tests/test_evo2.py::test_evo2_sample_tool`

```
tests/language_model_tests/test_evo2.py:102: in test_evo2_sample_tool
    validate_output(result)
tests/tool_infra_tests/test_export_functionality.py:102: in validate_output
    assert output.success is True, f"Tool execution failed: {output}"
E   AssertionError: Tool execution failed: 
E     ================================================================================
E     evo2-sample: TOOL FAILURE after 2.6299s
E     ================================================================================
E     
E     Error 1:
E     'evo2' may not be compatible with your system. setup.sh failed (exit 1).
E     ERROR: Evo2 is not supported on aarch64.
E     Evo2 requires transformer-engine and flash-attn which only provide x86_64 pre-built wheels.
E     
E     Error 2:
E     Traceback (most recent call last):
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 245, in wrapper
E         result = func(inputs, config, instance)
E                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/causal_models/evo2/evo2_sample.py", line 425, in run_evo2_sample
E         result = ToolInstance.dispatch(
E                  ^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 245, in dispatch
E         return cached.run(
E                ^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 552, in run
E         return self._run_persistent(
E                ^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 847, in _run_persistent
E         self._ensure_env()
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 513, in _ensure_env
E         self._create_env()
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 1407, in _create_env
E         raise RuntimeError(
E     RuntimeError: 'evo2' may not be compatible with your system. setup.sh failed (exit 1).
E     ERROR: Evo2 is not supported on aarch64.
E     Evo2 requires transformer-engine and flash-attn which only provide x86_64 pre-built wheels.
E     
E     ================================================================================
E   assert False is True
E    +  where False = Evo2SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, logits, kv_caches).success
```

### ❌ `progen2`

**Test**: `tests/language_model_tests/test_progen2.py::test_progen2_sample_basic`

```
tests/language_model_tests/test_progen2.py:50: in test_progen2_sample_basic
    validate_output(result)
tests/tool_infra_tests/test_export_functionality.py:102: in validate_output
    assert output.success is True, f"Tool execution failed: {output}"
E   AssertionError: Tool execution failed: 
E     ================================================================================
E     progen2-sample: TOOL FAILURE after 2.6366s
E     ================================================================================
E     
E     Error 1:
E     'progen2' may not be compatible with your system. setup.sh failed (exit 1).
E     ERROR: ProGen2 is not supported on aarch64.
E     ProGen2 pins torch==2.2.2 which has no aarch64 CUDA wheel available.
E     
E     Error 2:
E     Traceback (most recent call last):
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 245, in wrapper
E         result = func(inputs, config, instance)
E                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/causal_models/progen2/progen2_sample.py", line 366, in run_progen2_sample
E         result = ToolInstance.dispatch(
E                  ^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 245, in dispatch
E         return cached.run(
E                ^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 552, in run
E         return self._run_persistent(
E                ^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 847, in _run_persistent
E         self._ensure_env()
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 513, in _ensure_env
E         self._create_env()
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 1407, in _create_env
E         raise RuntimeError(
E     RuntimeError: 'progen2' may not be compatible with your system. setup.sh failed (exit 1).
E     ERROR: ProGen2 is not supported on aarch64.
E     ProGen2 pins torch==2.2.2 which has no aarch64 CUDA wheel available.
E     
E     ================================================================================
E   assert False is True
E    +  where False = ProGen2SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, logits).success
```

### ❌ `mafft`

**Test**: `tests/sequence_alignment_tests/test_mafft.py::TestMafftIntegration::test_protein_alignment_with_internal_gap`

```
tests/sequence_alignment_tests/test_mafft.py:207: in test_protein_alignment_with_internal_gap
    validate_output(result)
tests/tool_infra_tests/test_export_functionality.py:102: in validate_output
    assert output.success is True, f"Tool execution failed: {output}"
E   AssertionError: Tool execution failed: 
E     ================================================================================
E     mafft-align: TOOL FAILURE after 11.7171s
E     ================================================================================
E     
E     Error 1:
E     'mafft' may not be compatible with your system. setup.sh failed (exit 1).
E         install_binary(sys.argv[1])
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/install_binary.py", line 178, in install_binary
E         config.extract(archive_path, bin_dir)
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/sequence_alignment/mafft/standalone/binary_config.py", line 165, in extract
E         _extract_from_source(archive_path, bin_dir)
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/sequence_alignment/mafft/standalone/binary_config.py", line 134, in _extract_from_source
E         subprocess.check_call(
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/mafft_env/lib/python3.12/subprocess.py", line 413, in check_call
E         raise CalledProcessError(retcode, cmd)
E     subprocess.CalledProcessError: Command '['make', '-j4', 'PREFIX=/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/mafft_env', 'BINDIR=/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/mafft_env/bin', 'LIBDIR=/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/mafft_env/libexec/mafft']' returned non-zero exit status 2.
E     
E     Error 2:
E     Traceback (most recent call last):
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 245, in wrapper
E         result = func(inputs, config, instance)
E                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/sequence_alignment/mafft/mafft.py", line 194, in run_mafft_align
E         output_data = ToolInstance.dispatch(
E                       ^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 267, in dispatch
E         return cls._oneshot(
E                ^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 306, in _oneshot
E         return inst._run_oneshot(
E                ^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 983, in _run_oneshot
E         self._ensure_env()
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 513, in _ensure_env
E         self._create_env()
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 1407, in _create_env
E         raise RuntimeError(
E     RuntimeError: 'mafft' may not be compatible with your system. setup.sh failed (exit 1).
E         install_binary(sys.argv[1])
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/install_binary.py", line 178, in install_binary
E         config.extract(archive_path, bin_dir)
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/sequence_alignment/mafft/standalone/binary_config.py", line 165, in extract
E         _extract_from_source(archive_path, bin_dir)
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/sequence_alignment/mafft/standalone/binary_config.py", line 134, in _extract_from_source
E         subprocess.check_call(
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/mafft_env/lib/python3.12/subprocess.py", line 413, in check_call
E         raise CalledProcessError(retcode, cmd)
E     subprocess.CalledProcessError: Command '['make', '-j4', 'PREFIX=/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/mafft_env', 'BINDIR=/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/mafft_env/bin', 'LIBDIR=/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/mafft_env/libexec/mafft']' returned non-zero exit status 2.
E     
E     ================================================================================
E   assert False is True
E    +  where False = MafftOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success
```

### ❌ `bioemu`

**Test**: `tests/structure_dynamics_tests/test_bioemu.py::TestRunBioEmu::test_bioemu_sample_tool`

```
tests/structure_dynamics_tests/test_bioemu.py:139: in test_bioemu_sample_tool
    validate_output(result)
tests/tool_infra_tests/test_export_functionality.py:102: in validate_output
    assert output.success is True, f"Tool execution failed: {output}"
E   AssertionError: Tool execution failed: 
E     ================================================================================
E     bioemu-sample: TOOL FAILURE after 28.8139s
E     ================================================================================
E     
E     Error 1:
E     Command '['/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/bioemu_env/bin/python', '/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/structure_dynamics/bioemu/standalone/inference.py', '/tmp/tmpdd1f286r/input.json', '/tmp/tmpdd1f286r/output.json']' returned non-zero exit status 1.
E     
E     Error 2:
E     Traceback (most recent call last):
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 245, in wrapper
E         result = func(inputs, config, instance)
E                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/structure_dynamics/bioemu/bioemu_sample.py", line 210, in run_bioemu
E         output = ToolInstance.dispatch(
E                  ^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 267, in dispatch
E         return cls._oneshot(
E                ^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 299, in _oneshot
E         return inst._run_oneshot(
E                ^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 1014, in _run_oneshot
E         subprocess.run(
E       File "/home/bviggiano/miniconda3/envs/bio_tools/lib/python3.12/subprocess.py", line 571, in run
E         raise CalledProcessError(retcode, process.args,
E     subprocess.CalledProcessError: Command '['/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/bioemu_env/bin/python', '/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/structure_dynamics/bioemu/standalone/inference.py', '/tmp/tmpdd1f286r/input.json', '/tmp/tmpdd1f286r/output.json']' returned non-zero exit status 1.
E     
E     ================================================================================
E   assert False is True
E    +  where False = BioEmuOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success
```

### ❌ `chai1`

**Test**: `tests/structure_prediction_tests/test_structure_prediction.py::test_folding[gfp-chai1-without_msa]`

```
tests/structure_prediction_tests/test_structure_prediction.py:336: in test_folding
    validate_output(output)
tests/tool_infra_tests/test_export_functionality.py:102: in validate_output
    assert output.success is True, f"Tool execution failed: {output}"
                                                            ^^^^^^^^
bio_programming_tools/tools/structure_prediction/shared_data_models.py:801: in __str__
    return f"StructurePredictionOutput(structures={self.structures})"
                                                   ^^^^^^^^^^^^^^^
bio_programming_tools/utils/tool_io.py:129: in __getattr__
    raise ToolExecutionError("\nError Messages:\n" + "\n".join(errors))
E   bio_programming_tools.utils.tool_io.ToolExecutionError: Attempt to access field of tool output after failure: pre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.
E   
E   Error Messages:
E   'chai1' may not be compatible with your system. setup.sh failed (exit 1).
E   ERROR: Chai is not supported on aarch64.
E   chai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its
E   pre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.
E   Traceback (most recent call last):
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 245, in wrapper
E       result = func(inputs, config, instance)
E                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_cache.py", line 460, in wrapper
E       return func(*args, **kwargs)
E              ^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/structure_prediction/chai1/chai1.py", line 300, in run_chai1
E       results.append(run_chai1_on_complex(comp=comp, config=config, instance=instance))
E                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/structure_prediction/chai1/chai1.py", line 545, in run_chai1_on_complex
E       result = ToolInstance.dispatch(
E                ^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 245, in dispatch
E       return cached.run(
E              ^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 552, in run
E       return self._run_persistent(
E              ^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 847, in _run_persistent
E       self._ensure_env()
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 513, in _ensure_env
E       self._create_env()
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 1407, in _create_env
E       raise RuntimeError(
E   RuntimeError: 'chai1' may not be compatible with your system. setup.sh failed (exit 1).
E   ERROR: Chai is not supported on aarch64.
E   chai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its
E   pre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.
```

### ❌ `boltz2`

**Test**: `tests/structure_prediction_tests/test_structure_prediction.py::test_folding[gfp-boltz2-without_msa]`

```
tests/structure_prediction_tests/test_structure_prediction.py:336: in test_folding
    validate_output(output)
tests/tool_infra_tests/test_export_functionality.py:102: in validate_output
    assert output.success is True, f"Tool execution failed: {output}"
                                                            ^^^^^^^^
bio_programming_tools/tools/structure_prediction/shared_data_models.py:801: in __str__
    return f"StructurePredictionOutput(structures={self.structures})"
                                                   ^^^^^^^^^^^^^^^
bio_programming_tools/utils/tool_io.py:129: in __getattr__
    raise ToolExecutionError("\nError Messages:\n" + "\n".join(errors))
E   bio_programming_tools.utils.tool_io.ToolExecutionError: Attempt to access field of tool output after failure: TimeoutError: Worker for boltz2 timed out after 2400s
E   
E   Error Messages:
E   Worker for boltz2 timed out after 2400s
E   Traceback (most recent call last):
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 245, in wrapper
E       result = func(inputs, config, instance)
E                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_cache.py", line 460, in wrapper
E       return func(*args, **kwargs)
E              ^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/structure_prediction/boltz2/boltz2.py", line 329, in run_boltz2
E       run_boltz2_on_complex(
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/structure_prediction/boltz2/boltz2.py", line 578, in run_boltz2_on_complex
E       output_data = ToolInstance.dispatch(
E                     ^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 245, in dispatch
E       return cached.run(
E              ^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 552, in run
E       return self._run_persistent(
E              ^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 958, in _run_persistent
E       result = self._worker.send(input_dict, timeout=effective_timeout)
E                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/persistent_worker.py", line 530, in send
E       raise TimeoutError(
E   TimeoutError: Worker for boltz2 timed out after 2400s
```

### ❌ `protenix`

**Test**: `tests/structure_prediction_tests/test_structure_prediction.py::test_folding[gfp-protenix-without_msa]`

```
tests/structure_prediction_tests/test_structure_prediction.py:336: in test_folding
    validate_output(output)
tests/tool_infra_tests/test_export_functionality.py:102: in validate_output
    assert output.success is True, f"Tool execution failed: {output}"
                                                            ^^^^^^^^
bio_programming_tools/tools/structure_prediction/shared_data_models.py:801: in __str__
    return f"StructurePredictionOutput(structures={self.structures})"
                                                   ^^^^^^^^^^^^^^^
bio_programming_tools/utils/tool_io.py:129: in __getattr__
    raise ToolExecutionError("\nError Messages:\n" + "\n".join(errors))
E   bio_programming_tools.utils.tool_io.ToolExecutionError: Attempt to access field of tool output after failure: TimeoutError: Worker for protenix timed out after 2400s
E   
E   Error Messages:
E   Worker for protenix timed out after 2400s
E   Traceback (most recent call last):
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 245, in wrapper
E       result = func(inputs, config, instance)
E                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_cache.py", line 460, in wrapper
E       return func(*args, **kwargs)
E              ^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/structure_prediction/protenix/protenix.py", line 426, in run_protenix
E       output_data = ToolInstance.dispatch(
E                     ^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 245, in dispatch
E       return cached.run(
E              ^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 552, in run
E       return self._run_persistent(
E              ^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 958, in _run_persistent
E       result = self._worker.send(input_dict, timeout=effective_timeout)
E                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/persistent_worker.py", line 530, in send
E       raise TimeoutError(
E   TimeoutError: Worker for protenix timed out after 2400s
```

---
*Generated at 2026-03-01 11:41:48 by `pytest --env-report`*