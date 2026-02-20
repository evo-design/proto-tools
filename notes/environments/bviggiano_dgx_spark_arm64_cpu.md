# DGX Spark Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-68%25-yellow) ![Passed](https://img.shields.io/badge/passed-20-brightgreen) ![Failed](https://img.shields.io/badge/failed-9-red) ![Skipped](https://img.shields.io/badge/skipped-0-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Linux Linux 6.11.0-1016-nvidia |
| **Architecture** | aarch64 |
| **Hostname** | `spark-3b18` |
| **Python** | 3.12.12 |
| **RAM** | 119.7 GB |
| **GPU** | None |
| **Conda Env** | `bio_tools` |

## Git

- **Commit**: `9c23c400a192`
- **Branch**: `bv/add_setup_hashes`
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
TMUX=/tmp/tmux-1001/default,3802034,0
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
CLAUDECODE=1
CLAUDE_CODE_ENTRYPOINT=cli
CONDA_PREFIX_1=/home/bviggiano/miniconda3
CONDA_PREFIX_2=/home/bviggiano/miniconda3/envs/bio_tools
CONDA_PREFIX_3=/home/bviggiano/miniconda3
COREPACK_ENABLE_AUTO_PIN=0
CUDA_VISIBLE_DEVICES=0
DEBUGINFOD_URLS=https://debuginfod.ubuntu.com 
DISABLE_PANDERA_IMPORT_WARNING=True
GIT_EDITOR=true
HOME=/home/bviggiano
LANG=en_US.utf8
LD_LIBRARY_PATH=/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/splice_transformer_env/lib/python3.12/site-packages/nvidia/cusparselt/lib:/home/bviggiano/codebases/bio-programming/bio-programmi...
LESSCLOSE=/usr/bin/lesspipe %s %s
LESSOPEN=| /usr/bin/lesspipe %s
LOGNAME=bviggiano
LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=00:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.arj=...
NoDefaultCurrentDirectoryInExePath=1
OLDPWD=/home/bviggiano/codebases/bio-programming
OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta
PATH=/home/bviggiano/.local/bin:/home/bviggiano/.local/bin:/usr/local/cuda/bin:/usr/local/cuda/bin:/opt/bin:/home/bviggiano/.local/bin:/home/bviggiano/.local/bin:/usr/local/cuda/bin:/home/bviggiano/minicon...
PWD=/home/bviggiano/codebases/bio-programming/bio-programming-tools
PYTEST_CURRENT_TEST=tests/test_rna_splicing.py::test_splice_transformer_gpu (call)
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
RDBASE=/home/bviggiano/miniconda3/envs/bio_tools/lib/python3.12/site-packages/rdkit
SHELL=/bin/bash
SHLVL=3
TERM=tmux-256color
TERM_PROGRAM=tmux
TERM_PROGRAM_VERSION=3.4
TMUX=/tmp/tmux-1001/default,3802034,0
TMUX_PANE=%0
USER=bviggiano
XDG_DATA_DIRS=/usr/share/gnome:/usr/local/share:/usr/share:/var/lib/snapd/desktop
XDG_RUNTIME_DIR=/run/user/1001
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
_=/home/bviggiano/miniconda3/envs/bio_tools/bin/python
_CE_M=
```

## Results by Category

### Causal Models (0/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `evo1` | yes | ✅ | 38.6s | ❌ Fail |
| `evo2` | yes | ✅ | 4.6s | ❌ Fail |
| `progen2` | yes | ✅ | 12.5s | ❌ Fail |

### Gene Annotation (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `blast` | no | ✅ | 48.3s | ✅ Pass |
| `minced` | no | ✅ | 7.1s | ✅ Pass |
| `mmseqs` | no | ✅ | 17.0s | ✅ Pass |
| `pyhmmer` | no | ✅ | 9.0s | ✅ Pass |

### Inverse Folding (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `ligandmpnn` | yes | ✅ | 80.4s | ✅ Pass |
| `proteinmpnn` | yes | ✅ | 56.0s | ✅ Pass |

### Masked Models (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `esm2` | yes | ✅ | 48.6s | ✅ Pass |
| `esm3` | yes | ✅ | 56.3s | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `orfipy` | no | ✅ | 14.1s | ✅ Pass |
| `prodigal` | no | ✅ | 20.8s | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `splice_transformer` | yes | ✅ | 26.0s | ✅ Pass |

### Sequence Alignment (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `mafft` | no | ✅ | 47.0s | ✅ Pass |

### Sequence Scoring (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `borzoi` | yes | ✅ | 53.6s | ✅ Pass |
| `enformer` | yes | ✅ | 34.3s | ✅ Pass |

### Structure Design (0/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `rfdiffusion3` | yes | ✅ | 28.0s | ❌ Fail |

### Structure Dynamics (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `bioemu` | no | — | 0.0s | ✅ Pass |

### Structure Prediction (2/6)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `alphafold3` | yes | — | 5.6s | ❌ Fail |
| `boltz2` | yes | ✅ | 613.4s | ❌ Fail |
| `chai1` | yes | ✅ | 12.6s | ❌ Fail |
| `esmfold` | yes | ✅ | 144.5s | ✅ Pass |
| `protenix` | yes | ✅ | 257.0s | ❌ Fail |
| `viennarna` | no | ✅ | 11.4s | ✅ Pass |

### Unknown (3/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `alphagenome` | yes | ✅ | 345.6s | ✅ Pass |
| `crispr_tracr` | no | ✅ | 24.9s | ❌ Fail |
| `local_colabfold_search` | no | — | 115.9s | ✅ Pass |
| `structure_metrics` | no | ✅ | 13.3s | ✅ Pass |

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
E     evo1-sample: TOOL FAILURE after 38.6285s
E     ================================================================================
E     
E     Error 1:
E     Worker for evo1 returned an error:
E     Traceback (most recent call last):
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/_worker_bootstrap.py", line 148, in main
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
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/evo1_env/lib/python3.12/site-packages/evo/models.py", line 54, in __init__
E         self.model = load_checkpoint(
E                      ^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/evo1_env/lib/python3.12/site-packages/evo/models.py", line 146, in load_checkpoint
E         model = StripedHyena(global_config)
E                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/evo1_env/lib/python3.12/site-packages/stripedhyena/model.py", line 354, in __init__
E         self.blocks = nn.ModuleList(
E                       ^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/evo1_env/lib/python3.12/site-packages/torch/nn/modules/container.py", line 300, in __init__
E         self += modules
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/evo1_env/lib/python3.12/site-packages/torch/nn/modules/container.py", line 349, in __iadd__
E         return self.extend(modules)
E                ^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/evo1_env/lib/python3.12/site-packages/torch/nn/modules/container.py", line 432, in extend
E         for i, module in enumerate(modules):
E                          ^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/evo1_env/lib/python3.12/site-packages/stripedhyena/model.py", line 355, in <genexpr>
E         get_block(config, layer_idx, flash_fft=self.flash_fft) for layer_idx in range(config.num_layers)
E         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/evo1_env/lib/python3.12/site-packages/stripedhyena/model.py", line 326, in get_block
E         return AttentionBlock(config, layer_idx)
E                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/evo1_env/lib/python3.12/site-packages/stripedhyena/model.py", line 39, in __init__
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
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 232, in dispatch
E         return cached.run(
E                ^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 520, in run
E         return self._run_persistent(
E                ^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 621, in _run_persistent
E         return self._worker.send(input_dict, timeout=timeout)
E                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/persistent_worker.py", line 298, in send
E         raise RuntimeError(
E     RuntimeError: Worker for evo1 returned an error:
E     Traceback (most recent call last):
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/_worker_bootstrap.py", line 148, in main
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
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/evo1_env/lib/python3.12/site-packages/evo/models.py", line 54, in __init__
E         self.model = load_checkpoint(
E                      ^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/evo1_env/lib/python3.12/site-packages/evo/models.py", line 146, in load_checkpoint
E         model = StripedHyena(global_config)
E                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/evo1_env/lib/python3.12/site-packages/stripedhyena/model.py", line 354, in __init__
E         self.blocks = nn.ModuleList(
E                       ^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/evo1_env/lib/python3.12/site-packages/torch/nn/modules/container.py", line 300, in __init__
E         self += modules
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/evo1_env/lib/python3.12/site-packages/torch/nn/modules/container.py", line 349, in __iadd__
E         return self.extend(modules)
E                ^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/evo1_env/lib/python3.12/site-packages/torch/nn/modules/container.py", line 432, in extend
E         for i, module in enumerate(modules):
E                          ^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/evo1_env/lib/python3.12/site-packages/stripedhyena/model.py", line 355, in <genexpr>
E         get_block(config, layer_idx, flash_fft=self.flash_fft) for layer_idx in range(config.num_layers)
E         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/evo1_env/lib/python3.12/site-packages/stripedhyena/model.py", line 326, in get_block
E         return AttentionBlock(config, layer_idx)
E                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/evo1_env/lib/python3.12/site-packages/stripedhyena/model.py", line 39, in __init__
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
E     evo2-sample: TOOL FAILURE after 4.5577s
E     ================================================================================
E     
E     Error 1:
E     'evo2' may not be compatible with your system. setup.sh failed (exit 1).
E     
E     Error 2:
E     Traceback (most recent call last):
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 162, in wrapper
E         result = func(inputs, config, instance)
E                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/causal_models/evo2/evo2_sample.py", line 455, in run_evo2_sample
E         result = ToolInstance.dispatch(
E                  ^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 232, in dispatch
E         return cached.run(
E                ^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 520, in run
E         return self._run_persistent(
E                ^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 582, in _run_persistent
E         self._ensure_venv()
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 479, in _ensure_venv
E         self._create_venv()
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 882, in _create_venv
E         raise RuntimeError(
E     RuntimeError: 'evo2' may not be compatible with your system. setup.sh failed (exit 1).
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
E     progen2-sample: TOOL FAILURE after 12.5053s
E     ================================================================================
E     
E     Error 1:
E     'progen2' may not be compatible with your system. setup.sh failed (exit 1).
E     
E     Error 2:
E     Traceback (most recent call last):
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 162, in wrapper
E         result = func(inputs, config, instance)
E                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/causal_models/progen2/progen2_sample.py", line 374, in run_progen2_sample
E         result = ToolInstance.dispatch(
E                  ^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 232, in dispatch
E         return cached.run(
E                ^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 520, in run
E         return self._run_persistent(
E                ^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 582, in _run_persistent
E         self._ensure_venv()
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 479, in _ensure_venv
E         self._create_venv()
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 882, in _create_venv
E         raise RuntimeError(
E     RuntimeError: 'progen2' may not be compatible with your system. setup.sh failed (exit 1).
E     
E     ================================================================================
E   assert False is True
E    +  where False = ProGen2SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, logits).success
```

### ❌ `rfdiffusion3`

**Test**: `tests/structure_design_tests/test_rfdiffusion3.py::test_rfdiffusion3_unconditional_design`

```
tests/structure_design_tests/test_rfdiffusion3.py:38: in test_rfdiffusion3_unconditional_design
    assert len(output.output_structures) > 0
E   assert 0 > 0
E    +  where 0 = len([])
E    +    where [] = RFdiffusion3Output(output_structures=[0 structures]).output_structures
```

### ❌ `alphafold3`

**Test**: `tests/structure_prediction_tests/test_structure_prediction.py::test_folding[gfp-alphafold3-without_msa]`

```
Setup failed: tests/structure_prediction_tests/test_structure_prediction.py:283: in _release_between_predictors
    ToolInstance.get(predictor_name)
bio_programming_tools/utils/tool_instance.py:130: in get
    new_inst = cls(tool_name)
               ^^^^^^^^^^^^^^
bio_programming_tools/utils/tool_instance.py:410: in __init__
    self.tool_name = self._validate_tool_name(tool_name)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bio_programming_tools/utils/tool_instance.py:729: in _validate_tool_name
    raise ValueError(
E   ValueError: Invalid tool name: 'alphafold3'. Available tools with standalone dirs: ['alphagenome', 'bioemu', 'blast', 'boltz2', 'borzoi', 'chai1', 'colabfold_search', 'crispr_tracr', 'enformer', 'esm2', 'esm3', 'esmfold', 'evo1', 'evo2', 'ligandmpnn', 'mafft', 'minced', 'mmseqs', 'orfipy', 'prodigal', 'progen2', 'proteinmpnn', 'protenix', 'pyhmmer', 'rfdiffusion3', 'segmasker', 'splice_transformer', 'structure_metrics', 'viennarna']
```

### ❌ `chai1`

**Test**: `tests/structure_prediction_tests/test_structure_prediction.py::test_folding[gfp-chai1-without_msa]`

```
tests/structure_prediction_tests/test_structure_prediction.py:331: in test_folding
    validate_output(output)
tests/tool_infra_tests/test_export_functionality.py:102: in validate_output
    assert output.success is True, f"Tool execution failed: {output}"
                                                            ^^^^^^^^
bio_programming_tools/tools/structure_prediction/shared_data_models.py:801: in __str__
    return f"StructurePredictionOutput(structures={self.structures})"
                                                   ^^^^^^^^^^^^^^^
bio_programming_tools/utils/tool_io.py:129: in __getattr__
    raise ToolExecutionError("\nError Messages:\n" + "\n".join(errors))
E   bio_programming_tools.utils.tool_io.ToolExecutionError: Attempt to access field of tool output after failure: RuntimeError: 'chai1' may not be compatible with your system. setup.sh failed (exit 1).
E   
E   Error Messages:
E   'chai1' may not be compatible with your system. setup.sh failed (exit 1).
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
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/structure_prediction/chai1/chai1.py", line 585, in run_chai1_on_complex
E       result = ToolInstance.dispatch(
E                ^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 232, in dispatch
E       return cached.run(
E              ^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 520, in run
E       return self._run_persistent(
E              ^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 582, in _run_persistent
E       self._ensure_venv()
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 479, in _ensure_venv
E       self._create_venv()
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 882, in _create_venv
E       raise RuntimeError(
E   RuntimeError: 'chai1' may not be compatible with your system. setup.sh failed (exit 1).
```

### ❌ `boltz2`

**Test**: `tests/structure_prediction_tests/test_structure_prediction.py::test_folding[gfp-boltz2-without_msa]`

```
tests/structure_prediction_tests/test_structure_prediction.py:331: in test_folding
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
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/structure_prediction/boltz2/boltz2.py", line 620, in run_boltz2_on_complex
E       output_data = ToolInstance.dispatch(
E                     ^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 232, in dispatch
E       return cached.run(
E              ^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 520, in run
E       return self._run_persistent(
E              ^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 621, in _run_persistent
E       return self._worker.send(input_dict, timeout=timeout)
E              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/persistent_worker.py", line 272, in send
E       raise TimeoutError(
E   TimeoutError: Worker for boltz2 timed out after 600s
```

### ❌ `protenix`

**Test**: `tests/structure_prediction_tests/test_structure_prediction.py::test_folding[gfp-protenix-without_msa]`

```
tests/structure_prediction_tests/test_structure_prediction.py:331: in test_folding
    validate_output(output)
tests/tool_infra_tests/test_export_functionality.py:102: in validate_output
    assert output.success is True, f"Tool execution failed: {output}"
                                                            ^^^^^^^^
bio_programming_tools/tools/structure_prediction/shared_data_models.py:801: in __str__
    return f"StructurePredictionOutput(structures={self.structures})"
                                                   ^^^^^^^^^^^^^^^
bio_programming_tools/utils/tool_io.py:129: in __getattr__
    raise ToolExecutionError("\nError Messages:\n" + "\n".join(errors))
E   bio_programming_tools.utils.tool_io.ToolExecutionError: Attempt to access field of tool output after failure: RuntimeError: Error building extension 'fast_layer_norm_cuda_v2'
E   
E   Error Messages:
E   'protenix' may not be compatible with your system. setup.sh failed (exit 1).
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/protenix_env/lib/python3.12/site-packages/torch/utils/cpp_extension.py", line 1623, in load
E       return _jit_compile(
E              ^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/protenix_env/lib/python3.12/site-packages/torch/utils/cpp_extension.py", line 2076, in _jit_compile
E       _write_ninja_file_and_build_library(
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/protenix_env/lib/python3.12/site-packages/torch/utils/cpp_extension.py", line 2222, in _write_ninja_file_and_build_library
E       _run_ninja_build(
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/protenix_env/lib/python3.12/site-packages/torch/utils/cpp_extension.py", line 2522, in _run_ninja_build
E       raise RuntimeError(message) from e
E   RuntimeError: Error building extension 'fast_layer_norm_cuda_v2'
E   Traceback (most recent call last):
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 162, in wrapper
E       result = func(inputs, config, instance)
E                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_cache.py", line 460, in wrapper
E       return func(*args, **kwargs)
E              ^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/structure_prediction/protenix/protenix.py", line 423, in run_protenix
E       output_data = ToolInstance.dispatch(
E                     ^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 232, in dispatch
E       return cached.run(
E              ^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 520, in run
E       return self._run_persistent(
E              ^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 582, in _run_persistent
E       self._ensure_venv()
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 479, in _ensure_venv
E       self._create_venv()
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 882, in _create_venv
E       raise RuntimeError(
E   RuntimeError: 'protenix' may not be compatible with your system. setup.sh failed (exit 1).
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/protenix_env/lib/python3.12/site-packages/torch/utils/cpp_extension.py", line 1623, in load
E       return _jit_compile(
E              ^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/protenix_env/lib/python3.12/site-packages/torch/utils/cpp_extension.py", line 2076, in _jit_compile
E       _write_ninja_file_and_build_library(
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/protenix_env/lib/python3.12/site-packages/torch/utils/cpp_extension.py", line 2222, in _write_ninja_file_and_build_library
E       _run_ninja_build(
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/protenix_env/lib/python3.12/site-packages/torch/utils/cpp_extension.py", line 2522, in _run_ninja_build
E       raise RuntimeError(message) from e
E   RuntimeError: Error building extension 'fast_layer_norm_cuda_v2'
```

---
*Generated at 2026-02-19 17:29:00 by `pytest --env-report`*