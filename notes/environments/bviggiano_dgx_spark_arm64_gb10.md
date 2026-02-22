# DGX Spark Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-75%25-yellow) ![Passed](https://img.shields.io/badge/passed-22-brightgreen) ![Failed](https://img.shields.io/badge/failed-7-red) ![Skipped](https://img.shields.io/badge/skipped-0-lightgrey)

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

- **Commit**: `7f69fce0ae34`
- **Branch**: `bv/deps_iso_pass`
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
CONDA_PREFIX_2=/home/bviggiano/miniconda3/envs/bio_tools
CONDA_PREFIX_3=/home/bviggiano/miniconda3
CONDA_PROMPT_MODIFIER=(bio_tools) 
CONDA_PYTHON_EXE=/home/bviggiano/miniconda3/bin/python
CONDA_SHLVL=4
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
TMUX=/tmp/tmux-1001/default,4076373,0
TMUX_PANE=%0
USER=bviggiano
XDG_DATA_DIRS=/usr/share/gnome:/usr/local/share:/usr/share:/var/lib/snapd/desktop
XDG_RUNTIME_DIR=/run/user/1001
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
_=/home/bviggiano/miniconda3/envs/bio_tools/bin/python
_CE_CONDA=
_CE_M=
_CONDA_EXE=/home/bviggiano/miniconda3/bin/conda
_CONDA_ROOT=/home/bviggiano/miniconda3
```

### Subprocess Environment (passed to tools)

```
CONDA_DEFAULT_ENV=bio_tools
CONDA_PREFIX=/home/bviggiano/miniconda3/envs/bio_tools
CONDA_SHLVL=4
CUDA_VISIBLE_DEVICES=0
DETECTED_COMPUTE_PLATFORM=cuda
DETECTED_CUDA_VERSION=13
DETECTED_DRIVER_VERSION=580
HOME=/home/bviggiano
LANG=en_US.utf8
LD_LIBRARY_PATH=/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/splice_transformer_env/lib/python3.1/site-packages/nvidia/cusparselt/lib:/home/bviggiano/codebases/bio-programming/bio-program...
LOGNAME=bviggiano
PATH=/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/splice_transformer_env/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RECOMMENDED_JAX_SPEC=jax[cuda13]>=0.4.20,<1
RECOMMENDED_JAX_VARIANT=cuda13
RECOMMENDED_TORCH_SPEC=torch>=2.8,<3
SHELL=/bin/bash
TORCH_CUDA_ARCH_LIST=12.0
TORCH_HOME=/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/splice_transformer_env/cache/torch
USER=bviggiano
```

## Results by Category

### Causal Models (0/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `evo1` | yes | ✅ | 16.2s | ❌ Fail |
| `evo2` | yes | ✅ | 2.7s | ❌ Fail |
| `progen2` | yes | ✅ | 3.2s | ❌ Fail |

### Gene Annotation (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `blast` | no | ✅ | 62.8s | ✅ Pass |
| `minced` | no | ✅ | 4.0s | ✅ Pass |
| `mmseqs` | no | ✅ | 6.1s | ✅ Pass |
| `pyhmmer` | no | ✅ | 3.7s | ✅ Pass |

### Inverse Folding (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `ligandmpnn` | yes | ✅ | 25.7s | ✅ Pass |
| `proteinmpnn` | yes | ✅ | 16.2s | ✅ Pass |

### Masked Models (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `esm2` | yes | ✅ | 35.1s | ✅ Pass |
| `esm3` | yes | ✅ | 18.7s | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `orfipy` | no | ✅ | 4.5s | ✅ Pass |
| `prodigal` | no | ✅ | 3.7s | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `splice_transformer` | yes | ✅ | 23.6s | ✅ Pass |

### Sequence Alignment (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `mafft` | no | ✅ | 17.4s | ✅ Pass |

### Sequence Scoring (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `borzoi` | yes | ✅ | 31.0s | ✅ Pass |
| `enformer` | yes | ✅ | 20.8s | ✅ Pass |

### Structure Design (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `rfdiffusion3` | yes | ✅ | 216.4s | ✅ Pass |

### Structure Dynamics (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `bioemu` | no | — | 0.0s | ✅ Pass |

### Structure Prediction (3/6)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `alphafold3` | yes | — | 0.1s | ❌ Fail |
| `boltz2` | yes | ✅ | 610.2s | ❌ Fail |
| `chai1` | yes | ✅ | 5.9s | ❌ Fail |
| `esmfold` | yes | ✅ | 63.9s | ✅ Pass |
| `protenix` | yes | ✅ | 567.7s | ✅ Pass |
| `viennarna` | no | ✅ | 11.2s | ✅ Pass |

### Unknown (3/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `alphagenome` | yes | ✅ | 134.8s | ✅ Pass |
| `crispr_tracr` | no | ✅ | 17.4s | ❌ Fail |
| `local_colabfold_search` | no | — | 37.7s | ✅ Pass |
| `structure_metrics` | no | ✅ | 9.6s | ✅ Pass |

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
E     evo1-sample: TOOL FAILURE after 16.1473s
E     ================================================================================
E     
E     Error 1:
E     Worker for evo1 returned an error:
E     Traceback (most recent call last):
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/_worker_bootstrap.py", line 158, in main
E         result = dispatch(input_dict)
E                  ^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/causal_models/evo1/standalone/inference.py", line 276, in dispatch
E         return _model.sample(
E                ^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/causal_models/evo1/standalone/inference.py", line 82, in sample
E         self.load(self.device, verbose=verbose)
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/causal_models/evo1/standalone/inference.py", line 227, in load
E         evo_obj = Evo(self.model_name)
E                   ^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/evo1_env/lib/python3.12/site-packages/evo/models.py", line 54, in __init__
E         self.model = load_checkpoint(
E                      ^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/evo1_env/lib/python3.12/site-packages/evo/models.py", line 146, in load_checkpoint
E         model = StripedHyena(global_config)
E                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/evo1_env/lib/python3.12/site-packages/stripedhyena/model.py", line 354, in __init__
E         self.blocks = nn.ModuleList(
E                       ^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/evo1_env/lib/python3.12/site-packages/torch/nn/modules/container.py", line 300, in __init__
E         self += modules
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/evo1_env/lib/python3.12/site-packages/torch/nn/modules/container.py", line 349, in __iadd__
E         return self.extend(modules)
E                ^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/evo1_env/lib/python3.12/site-packages/torch/nn/modules/container.py", line 432, in extend
E         for i, module in enumerate(modules):
E                          ^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/evo1_env/lib/python3.12/site-packages/stripedhyena/model.py", line 355, in <genexpr>
E         get_block(config, layer_idx, flash_fft=self.flash_fft) for layer_idx in range(config.num_layers)
E         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/evo1_env/lib/python3.12/site-packages/stripedhyena/model.py", line 326, in get_block
E         return AttentionBlock(config, layer_idx)
E                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/evo1_env/lib/python3.12/site-packages/stripedhyena/model.py", line 39, in __init__
E         self.inner_mha_cls = MHA(
E                              ^^^
E     NameError: name 'MHA' is not defined
E     
E     
E     Error 2:
E     Traceback (most recent call last):
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 162, in wrapper
E         result = func(inputs, config, instance)
E                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/causal_models/evo1/evo1_sample.py", line 209, in run_evo1_sample
E         result = ToolInstance.dispatch(
E                  ^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 236, in dispatch
E         return cached.run(
E                ^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 527, in run
E         return self._run_persistent(
E                ^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 629, in _run_persistent
E         return self._worker.send(input_dict, timeout=timeout)
E                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/persistent_worker.py", line 446, in send
E         raise RuntimeError(
E     RuntimeError: Worker for evo1 returned an error:
E     Traceback (most recent call last):
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/_worker_bootstrap.py", line 158, in main
E         result = dispatch(input_dict)
E                  ^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/causal_models/evo1/standalone/inference.py", line 276, in dispatch
E         return _model.sample(
E                ^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/causal_models/evo1/standalone/inference.py", line 82, in sample
E         self.load(self.device, verbose=verbose)
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/causal_models/evo1/standalone/inference.py", line 227, in load
E         evo_obj = Evo(self.model_name)
E                   ^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/evo1_env/lib/python3.12/site-packages/evo/models.py", line 54, in __init__
E         self.model = load_checkpoint(
E                      ^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/evo1_env/lib/python3.12/site-packages/evo/models.py", line 146, in load_checkpoint
E         model = StripedHyena(global_config)
E                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/evo1_env/lib/python3.12/site-packages/stripedhyena/model.py", line 354, in __init__
E         self.blocks = nn.ModuleList(
E                       ^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/evo1_env/lib/python3.12/site-packages/torch/nn/modules/container.py", line 300, in __init__
E         self += modules
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/evo1_env/lib/python3.12/site-packages/torch/nn/modules/container.py", line 349, in __iadd__
E         return self.extend(modules)
E                ^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/evo1_env/lib/python3.12/site-packages/torch/nn/modules/container.py", line 432, in extend
E         for i, module in enumerate(modules):
E                          ^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/evo1_env/lib/python3.12/site-packages/stripedhyena/model.py", line 355, in <genexpr>
E         get_block(config, layer_idx, flash_fft=self.flash_fft) for layer_idx in range(config.num_layers)
E         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/evo1_env/lib/python3.12/site-packages/stripedhyena/model.py", line 326, in get_block
E         return AttentionBlock(config, layer_idx)
E                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/tool_envs/evo1_env/lib/python3.12/site-packages/stripedhyena/model.py", line 39, in __init__
E         self.inner_mha_cls = MHA(
E                              ^^^
E     NameError: name 'MHA' is not defined
E     
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
E     evo2-sample: TOOL FAILURE after 2.6537s
E     ================================================================================
E     
E     Error 1:
E     'evo2' may not be compatible with your system. setup.sh failed (exit 1).
E     ERROR: Evo2 is not supported on aarch64.
E     Evo2 requires transformer-engine and flash-attn which only provide x86_64 pre-built wheels.
E     
E     Error 2:
E     Traceback (most recent call last):
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 162, in wrapper
E         result = func(inputs, config, instance)
E                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/causal_models/evo2/evo2_sample.py", line 455, in run_evo2_sample
E         result = ToolInstance.dispatch(
E                  ^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 236, in dispatch
E         return cached.run(
E                ^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 527, in run
E         return self._run_persistent(
E                ^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 589, in _run_persistent
E         self._ensure_env()
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 486, in _ensure_env
E         self._create_env()
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 1056, in _create_env
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
E     progen2-sample: TOOL FAILURE after 3.1666s
E     ================================================================================
E     
E     Error 1:
E     'progen2' may not be compatible with your system. setup.sh failed (exit 1).
E     ERROR: ProGen2 is not supported on aarch64.
E     ProGen2 pins torch==2.2.2 which has no aarch64 CUDA wheel available.
E     
E     Error 2:
E     Traceback (most recent call last):
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 162, in wrapper
E         result = func(inputs, config, instance)
E                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/causal_models/progen2/progen2_sample.py", line 374, in run_progen2_sample
E         result = ToolInstance.dispatch(
E                  ^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 236, in dispatch
E         return cached.run(
E                ^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 527, in run
E         return self._run_persistent(
E                ^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 589, in _run_persistent
E         self._ensure_env()
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 486, in _ensure_env
E         self._create_env()
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 1056, in _create_env
E         raise RuntimeError(
E     RuntimeError: 'progen2' may not be compatible with your system. setup.sh failed (exit 1).
E     ERROR: ProGen2 is not supported on aarch64.
E     ProGen2 pins torch==2.2.2 which has no aarch64 CUDA wheel available.
E     
E     ================================================================================
E   assert False is True
E    +  where False = ProGen2SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, logits).success
```

### ❌ `alphafold3`

**Test**: `tests/structure_prediction_tests/test_structure_prediction.py::test_folding[gfp-alphafold3-without_msa]`

```
tests/structure_prediction_tests/test_structure_prediction.py:337: in test_folding
    validate_output(output)
tests/tool_infra_tests/test_export_functionality.py:102: in validate_output
    assert output.success is True, f"Tool execution failed: {output}"
                                                            ^^^^^^^^
bio_programming_tools/tools/structure_prediction/shared_data_models.py:801: in __str__
    return f"StructurePredictionOutput(structures={self.structures})"
                                                   ^^^^^^^^^^^^^^^
bio_programming_tools/utils/tool_io.py:129: in __getattr__
    raise ToolExecutionError("\nError Messages:\n" + "\n".join(errors))
E   bio_programming_tools.utils.tool_io.ToolExecutionError: Attempt to access field of tool output after failure: bio_programming_tools.tools.structure_prediction.alphafold3.inference.AlphaFold3ExecutionError: Path does not exist for repo_path: /large_storage/hielab/brk/models/alphafold3
E   
E   Error Messages:
E   Path does not exist for repo_path: /large_storage/hielab/brk/models/alphafold3
E   Traceback (most recent call last):
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 162, in wrapper
E       result = func(inputs, config, instance)
E                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_cache.py", line 460, in wrapper
E       return func(*args, **kwargs)
E              ^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/structure_prediction/alphafold3/alphafold3.py", line 232, in run_alphafold3
E       pdb_path, alphafold3_scores = alphafold3_inference(
E                                     ^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/structure_prediction/alphafold3/inference.py", line 255, in alphafold3_inference
E       raise AlphaFold3ExecutionError(f"Path does not exist for {name}: {path}")
E   bio_programming_tools.tools.structure_prediction.alphafold3.inference.AlphaFold3ExecutionError: Path does not exist for repo_path: /large_storage/hielab/brk/models/alphafold3
```

### ❌ `chai1`

**Test**: `tests/structure_prediction_tests/test_structure_prediction.py::test_folding[gfp-chai1-without_msa]`

```
tests/structure_prediction_tests/test_structure_prediction.py:337: in test_folding
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
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 162, in wrapper
E       result = func(inputs, config, instance)
E                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_cache.py", line 460, in wrapper
E       return func(*args, **kwargs)
E              ^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/structure_prediction/chai1/chai1.py", line 306, in run_chai1
E       results.append(run_chai1_on_complex(comp=comp, config=config, instance=instance))
E                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/structure_prediction/chai1/chai1.py", line 586, in run_chai1_on_complex
E       result = ToolInstance.dispatch(
E                ^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 236, in dispatch
E       return cached.run(
E              ^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 527, in run
E       return self._run_persistent(
E              ^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 589, in _run_persistent
E       self._ensure_env()
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 486, in _ensure_env
E       self._create_env()
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 1056, in _create_env
E       raise RuntimeError(
E   RuntimeError: 'chai1' may not be compatible with your system. setup.sh failed (exit 1).
E   ERROR: Chai is not supported on aarch64.
E   chai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its
E   pre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures.
```

### ❌ `boltz2`

**Test**: `tests/structure_prediction_tests/test_structure_prediction.py::test_folding[gfp-boltz2-without_msa]`

```
tests/structure_prediction_tests/test_structure_prediction.py:337: in test_folding
    validate_output(output)
tests/tool_infra_tests/test_export_functionality.py:102: in validate_output
    assert output.success is True, f"Tool execution failed: {output}"
                                                            ^^^^^^^^
bio_programming_tools/tools/structure_prediction/shared_data_models.py:801: in __str__
    return f"StructurePredictionOutput(structures={self.structures})"
                                                   ^^^^^^^^^^^^^^^
bio_programming_tools/utils/tool_io.py:129: in __getattr__
    raise ToolExecutionError("\nError Messages:\n" + "\n".join(errors))
E   bio_programming_tools.utils.tool_io.ToolExecutionError: Attempt to access field of tool output after failure: TimeoutError: Worker for boltz2 timed out after 600s
E   
E   Error Messages:
E   Worker for boltz2 timed out after 600s
E   Traceback (most recent call last):
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 162, in wrapper
E       result = func(inputs, config, instance)
E                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_cache.py", line 460, in wrapper
E       return func(*args, **kwargs)
E              ^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/structure_prediction/boltz2/boltz2.py", line 335, in run_boltz2
E       run_boltz2_on_complex(
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/structure_prediction/boltz2/boltz2.py", line 621, in run_boltz2_on_complex
E       output_data = ToolInstance.dispatch(
E                     ^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 236, in dispatch
E       return cached.run(
E              ^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 527, in run
E       return self._run_persistent(
E              ^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 629, in _run_persistent
E       return self._worker.send(input_dict, timeout=timeout)
E              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/persistent_worker.py", line 390, in send
E       raise TimeoutError(
E   TimeoutError: Worker for boltz2 timed out after 600s
```

---
*Generated at 2026-02-20 23:17:14 by `pytest --env-report`*