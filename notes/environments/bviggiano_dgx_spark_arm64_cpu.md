# DGX Spark Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-64%25-yellow) ![Passed](https://img.shields.io/badge/passed-18-brightgreen) ![Failed](https://img.shields.io/badge/failed-10-red) ![Skipped](https://img.shields.io/badge/skipped-0-lightgrey)

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

- **Commit**: `064faae518b2`
- **Branch**: `bv/env_testing`
- **Dirty**: No

## Environment Variables

### Parent Process Environment

```
CONDA_DEFAULT_ENV=bio_tools
CONDA_EXE=/home/bviggiano/miniconda3/bin/conda
CONDA_PREFIX=/home/bviggiano/miniconda3/envs/bio_tools
CONDA_PREFIX_1=/home/bviggiano/miniconda3
CONDA_PROMPT_MODIFIER=(bio_tools) 
CONDA_PYTHON_EXE=/home/bviggiano/miniconda3/bin/python
CONDA_SHLVL=2
DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1001/bus
DEBUGINFOD_URLS=https://debuginfod.ubuntu.com 
DISABLE_PANDERA_IMPORT_WARNING=True
HOME=/home/bviggiano
LANG=en_US.utf8
LD_LIBRARY_PATH=/usr/local/cuda/lib64:
LESSCLOSE=/usr/bin/lesspipe %s %s
LESSOPEN=| /usr/bin/lesspipe %s
LOGNAME=bviggiano
LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=00:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.arj=...
OLDPWD=/home/bviggiano/codebases/bio-programming
PATH=/home/bviggiano/.local/bin:/home/bviggiano/.local/bin:/usr/local/cuda/bin:/home/bviggiano/miniconda3/envs/bio_tools/bin:/home/bviggiano/miniconda3/condabin:/usr/local/cuda/bin:/opt/bin:/usr/local/sbin...
PWD=/home/bviggiano/codebases/bio-programming/bio-programming-tools
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
RDBASE=/home/bviggiano/miniconda3/envs/bio_tools/lib/python3.12/site-packages/rdkit
SHELL=/bin/bash
SHLVL=1
TERM=xterm-256color
USER=bviggiano
XDG_DATA_DIRS=/usr/share/gnome:/usr/local/share:/usr/share:/var/lib/snapd/desktop
XDG_RUNTIME_DIR=/run/user/1001
XDG_SESSION_CLASS=user
XDG_SESSION_ID=19398
XDG_SESSION_TYPE=tty
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
CONDA_PREFIX_1=/home/bviggiano/miniconda3
CUDA_VISIBLE_DEVICES=0
DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1001/bus
DEBUGINFOD_URLS=https://debuginfod.ubuntu.com 
DISABLE_PANDERA_IMPORT_WARNING=True
HOME=/home/bviggiano
LANG=en_US.utf8
LESSCLOSE=/usr/bin/lesspipe %s %s
LESSOPEN=| /usr/bin/lesspipe %s
LOGNAME=bviggiano
LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=00:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.arj=...
OLDPWD=/home/bviggiano/codebases/bio-programming
PATH=/home/bviggiano/.local/bin:/home/bviggiano/.local/bin:/usr/local/cuda/bin:/home/bviggiano/miniconda3/envs/bio_tools/bin:/home/bviggiano/miniconda3/condabin:/usr/local/cuda/bin:/opt/bin:/usr/local/sbin...
PWD=/home/bviggiano/codebases/bio-programming/bio-programming-tools
PYTEST_CURRENT_TEST=tests/test_rna_splicing.py::test_splice_transformer_gpu (call)
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
RDBASE=/home/bviggiano/miniconda3/envs/bio_tools/lib/python3.12/site-packages/rdkit
SHELL=/bin/bash
SHLVL=1
TERM=xterm-256color
USER=bviggiano
XDG_DATA_DIRS=/usr/share/gnome:/usr/local/share:/usr/share:/var/lib/snapd/desktop
XDG_RUNTIME_DIR=/run/user/1001
XDG_SESSION_CLASS=user
XDG_SESSION_ID=19398
XDG_SESSION_TYPE=tty
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
_=/home/bviggiano/miniconda3/envs/bio_tools/bin/pytest
_CE_M=
```

## Results by Category

### Causal Models (0/3)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `evo1` | ❌ Fail | yes | ✅ | 13.7s |
| `evo2` | ❌ Fail | yes | ✅ | 1.4s |
| `progen2` | ❌ Fail | yes | ✅ | 1.4s |

### Gene Annotation (4/5)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `blast` | ✅ Pass | no | ✅ | 64.7s |
| `crispr_tracr` | ❌ Fail | no | ✅ | 15.3s |
| `minced` | ✅ Pass | no | ✅ | 2.5s |
| `mmseqs` | ✅ Pass | no | ✅ | 4.2s |
| `pyhmmer` | ✅ Pass | no | ✅ | 2.2s |

### Inverse Folding (2/2)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `ligandmpnn` | ✅ Pass | yes | ✅ | 22.3s |
| `proteinmpnn` | ✅ Pass | yes | ✅ | 14.1s |

### Masked Models (2/2)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `esm2` | ✅ Pass | yes | ✅ | 31.3s |
| `esm3` | ✅ Pass | yes | ✅ | 15.8s |

### Orf Prediction (2/2)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `orfipy` | ✅ Pass | no | ✅ | 2.7s |
| `prodigal` | ✅ Pass | no | ✅ | 2.2s |

### Rna Splicing (1/1)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `splice_transformer` | ✅ Pass | yes | ✅ | 9.4s |

### Sequence Alignment (1/2)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `colabfold_search` | ❌ Fail | no | ✅ | 3.6s |
| `mafft` | ✅ Pass | no | ✅ | 12.7s |

### Sequence Scoring (2/3)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `alphagenome` | ❌ Fail | yes | ✅ | 1.4s |
| `borzoi` | ✅ Pass | yes | ✅ | 23.2s |
| `enformer` | ✅ Pass | yes | ✅ | 19.9s |

### Structure Design (0/1)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `rfdiffusion3` | ❌ Fail | yes | ✅ | 6.9s |

### Structure Dynamics (1/1)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `bioemu` | ✅ Pass | no | — | 0.0s |

### Structure Prediction (3/6)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `protenix` | ❌ Fail | yes | ✅ | 18.1s |
| `structure_metrics` | ✅ Pass | no | ✅ | 2.8s |
| `structure_prediction` | ✅ Pass | yes | — | 83.6s |
| `structure_prediction` | ❌ Fail | yes | — | 4.6s |
| `structure_prediction` | ❌ Fail | yes | — | 607.9s |
| `viennarna` | ✅ Pass | no | ✅ | 2.3s |

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
tests/language_model_tests/test_evo1.py:138: in test_evo1_sample_tool
    validate_output(result)
tests/tool_infra_tests/test_export_functionality.py:102: in validate_output
    assert output.success is True, f"Tool execution failed: {output}"
E   AssertionError: Tool execution failed: 
E     ================================================================================
E     evo1-sample: TOOL FAILURE after 13.6350s
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
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/causal_models/evo1/standalone/inference.py", line 81, in sample
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
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 159, in wrapper
E         result = func(inputs, config, instance)
E                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/causal_models/evo1/evo1_sample.py", line 201, in run_evo1_sample
E         result = ToolInstance.dispatch(
E                  ^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 232, in dispatch
E         return cached.run(
E                ^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 515, in run
E         return self._run_persistent(
E                ^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 616, in _run_persistent
E         return self._worker.send(input_dict, timeout=timeout)
E                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/persistent_worker.py", line 242, in send
E         raise RuntimeError(
E     RuntimeError: Worker for evo1 returned an error:
E     Traceback (most recent call last):
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/_worker_bootstrap.py", line 148, in main
E         result = dispatch(input_dict)
E                  ^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/causal_models/evo1/standalone/inference.py", line 276, in dispatch
E         return _model.sample(
E                ^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/causal_models/evo1/standalone/inference.py", line 81, in sample
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
E     evo2-sample: TOOL FAILURE after 1.3968s
E     ================================================================================
E     
E     Error 1:
E     'evo2' may not be compatible with your system. setup.sh failed (exit 1).
E     
E     Error 2:
E     Traceback (most recent call last):
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 159, in wrapper
E         result = func(inputs, config, instance)
E                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/causal_models/evo2/evo2_sample.py", line 453, in run_evo2_sample
E         result = ToolInstance.dispatch(
E                  ^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 232, in dispatch
E         return cached.run(
E                ^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 515, in run
E         return self._run_persistent(
E                ^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 577, in _run_persistent
E         self._ensure_venv()
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 474, in _ensure_venv
E         self._create_venv()
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 864, in _create_venv
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
E     progen2-sample: TOOL FAILURE after 1.3903s
E     ================================================================================
E     
E     Error 1:
E     'progen2' may not be compatible with your system. setup.sh failed (exit 1).
E     
E     Error 2:
E     Traceback (most recent call last):
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 159, in wrapper
E         result = func(inputs, config, instance)
E                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/causal_models/progen2/progen2_sample.py", line 372, in run_progen2_sample
E         result = ToolInstance.dispatch(
E                  ^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 232, in dispatch
E         return cached.run(
E                ^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 515, in run
E         return self._run_persistent(
E                ^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 577, in _run_persistent
E         self._ensure_venv()
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 474, in _ensure_venv
E         self._create_venv()
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 864, in _create_venv
E         raise RuntimeError(
E     RuntimeError: 'progen2' may not be compatible with your system. setup.sh failed (exit 1).
E     
E     ================================================================================
E   assert False is True
E    +  where False = ProGen2SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, logits).success
```

### ❌ `colabfold_search`

**Test**: `tests/sequence_alignment_tests/test_local_colabfold_search.py::TestColabfoldSearchExecutionDebugDatabase::test_finding_self_in_database`

```
tests/sequence_alignment_tests/test_local_colabfold_search.py:275: in test_finding_self_in_database
    validate_output(result)
tests/tool_infra_tests/test_export_functionality.py:102: in validate_output
    assert output.success is True, f"Tool execution failed: {output}"
E   AssertionError: Tool execution failed: 
E     ================================================================================
E     colabfold-search: TOOL FAILURE after 3.5810s
E     ================================================================================
E     
E     Error 1:
E     colabfold_search failed: colabfold_search failed with exit code 1
E     STDERR: INFO:colabfold.mmseqs.search:Running mmseqs createdb /tmp/tmpu1gq60e9/msas/query.fas /tmp/tmpu1gq60e9/msas/qdb --shuffle 0 --dbtype 1
E     Traceback (most recent call last):
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/colabfold_search_env/bin/colabfold_search", line 10, in <module>
E         sys.exit(main())
E                  ^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/colabfold_search_env/lib/python3.12/site-packages/colabfold/mmseqs/search.py", line 449, in main
E         run_mmseqs(
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/colabfold_search_env/lib/python3.12/site-packages/colabfold/mmseqs/search.py", line 46, in run_mmseqs
E         subprocess.check_call([mmseqs] + params)
E       File "/home/bviggiano/miniconda3/envs/bio_tools/lib/python3.12/subprocess.py", line 408, in check_call
E         retcode = call(*popenargs, **kwargs)
E                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/miniconda3/envs/bio_tools/lib/python3.12/subprocess.py", line 389, in call
E         with Popen(*popenargs, **kwargs) as p:
E              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/miniconda3/envs/bio_tools/lib/python3.12/subprocess.py", line 1026, in __init__
E         self._execute_child(args, executable, preexec_fn, close_fds,
E       File "/home/bviggiano/miniconda3/envs/bio_tools/lib/python3.12/subprocess.py", line 1955, in _execute_child
E         raise child_exception_type(errno_num, err_msg, err_filename)
E     FileNotFoundError: [Errno 2] No such file or directory: PosixPath('mmseqs')
E     
E     
E     Error 2:
E     Traceback (most recent call last):
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 159, in wrapper
E         result = func(inputs, config, instance)
E                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/sequence_alignment/colabfold_search/colabfold_search.py", line 485, in run_colabfold_search
E         return _local_search(sequences, sequence_ids, config, msa_out_dir, instance=instance)
E                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/sequence_alignment/colabfold_search/colabfold_search.py", line 676, in _local_search
E         raise RuntimeError(f"colabfold_search failed: {error_msg}")
E     RuntimeError: colabfold_search failed: colabfold_search failed with exit code 1
E     STDERR: INFO:colabfold.mmseqs.search:Running mmseqs createdb /tmp/tmpu1gq60e9/msas/query.fas /tmp/tmpu1gq60e9/msas/qdb --shuffle 0 --dbtype 1
E     Traceback (most recent call last):
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/colabfold_search_env/bin/colabfold_search", line 10, in <module>
E         sys.exit(main())
E                  ^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/colabfold_search_env/lib/python3.12/site-packages/colabfold/mmseqs/search.py", line 449, in main
E         run_mmseqs(
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/colabfold_search_env/lib/python3.12/site-packages/colabfold/mmseqs/search.py", line 46, in run_mmseqs
E         subprocess.check_call([mmseqs] + params)
E       File "/home/bviggiano/miniconda3/envs/bio_tools/lib/python3.12/subprocess.py", line 408, in check_call
E         retcode = call(*popenargs, **kwargs)
E                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/miniconda3/envs/bio_tools/lib/python3.12/subprocess.py", line 389, in call
E         with Popen(*popenargs, **kwargs) as p:
E              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/miniconda3/envs/bio_tools/lib/python3.12/subprocess.py", line 1026, in __init__
E         self._execute_child(args, executable, preexec_fn, close_fds,
E       File "/home/bviggiano/miniconda3/envs/bio_tools/lib/python3.12/subprocess.py", line 1955, in _execute_child
E         raise child_exception_type(errno_num, err_msg, err_filename)
E     FileNotFoundError: [Errno 2] No such file or directory: PosixPath('mmseqs')
E     
E     
E     ================================================================================
E   assert False is True
E    +  where False = ColabfoldSearchOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success
```

### ❌ `alphagenome`

**Test**: `tests/sequence_scoring_tests/test_alphagenome.py::TestAlphaGenome::test_interval_prediction`

```
tests/sequence_scoring_tests/test_alphagenome.py:56: in test_interval_prediction
    validate_output(result)
tests/tool_infra_tests/test_export_functionality.py:102: in validate_output
    assert output.success is True, f"Tool execution failed: {output}"
E   AssertionError: Tool execution failed: 
E     ================================================================================
E     alphagenome-predict-interval: TOOL FAILURE after 1.3608s
E     ================================================================================
E     
E     Error 1:
E     'alphagenome' may not be compatible with your system. setup.sh failed (exit 1).
E     
E     Error 2:
E     Traceback (most recent call last):
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 159, in wrapper
E         result = func(inputs, config, instance)
E                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/sequence_scoring/alphagenome/alphagenome_predict_interval.py", line 50, in run_alphagenome_predict_interval
E         result = ToolInstance.dispatch(
E                  ^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 232, in dispatch
E         return cached.run(
E                ^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 515, in run
E         return self._run_persistent(
E                ^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 577, in _run_persistent
E         self._ensure_venv()
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 474, in _ensure_venv
E         self._create_venv()
E       File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 864, in _create_venv
E         raise RuntimeError(
E     RuntimeError: 'alphagenome' may not be compatible with your system. setup.sh failed (exit 1).
E     
E     ================================================================================
E   assert False is True
E    +  where False = AlphaGenomePredictOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, variant).success
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

### ❌ `protenix`

**Test**: `tests/structure_prediction_tests/test_protenix.py::test_protenix_model_variants[protenix_base_default_v1.0.0]`

```
tests/structure_prediction_tests/test_protenix.py:77: in test_protenix_model_variants
    assert len(output.structures) == 1, f"Expected 1 structure, got {len(output.structures)}"
               ^^^^^^^^^^^^^^^^^
bio_programming_tools/utils/tool_io.py:129: in __getattr__
    raise ToolExecutionError("\nError Messages:\n" + "\n".join(errors))
E   bio_programming_tools.utils.tool_io.ToolExecutionError: Attempt to access field of tool output after failure: tar: Error is not recoverable: exiting now
E   
E   Error Messages:
E   'protenix' may not be compatible with your system. setup.sh failed (exit 1).
E   [notice] A new release of pip is available: 25.0.1 -> 26.0.1
E   [notice] To update, run: pip install --upgrade pip
E   bzip2: (stdin) is not a bzip2 file.
E   tar: Child returned status 2
E   tar: Error is not recoverable: exiting now
E   Traceback (most recent call last):
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 159, in wrapper
E       result = func(inputs, config, instance)
E                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_cache.py", line 460, in wrapper
E       return func(*args, **kwargs)
E              ^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/structure_prediction/protenix/protenix.py", line 422, in run_protenix
E       output_data = ToolInstance.dispatch(
E                     ^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 232, in dispatch
E       return cached.run(
E              ^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 515, in run
E       return self._run_persistent(
E              ^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 577, in _run_persistent
E       self._ensure_venv()
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 474, in _ensure_venv
E       self._create_venv()
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 864, in _create_venv
E       raise RuntimeError(
E   RuntimeError: 'protenix' may not be compatible with your system. setup.sh failed (exit 1).
E   [notice] A new release of pip is available: 25.0.1 -> 26.0.1
E   [notice] To update, run: pip install --upgrade pip
E   bzip2: (stdin) is not a bzip2 file.
E   tar: Child returned status 2
E   tar: Error is not recoverable: exiting now
```

### ❌ `structure_prediction`

**Test**: `tests/structure_prediction_tests/test_structure_prediction.py::test_folding[two_complex-chai1-without_msa]`

```
tests/structure_prediction_tests/test_structure_prediction.py:334: in test_folding
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
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 159, in wrapper
E       result = func(inputs, config, instance)
E                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_cache.py", line 460, in wrapper
E       return func(*args, **kwargs)
E              ^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/structure_prediction/chai1/chai1.py", line 305, in run_chai1
E       results.append(run_chai1_on_complex(comp=comp, config=config, instance=instance))
E                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/structure_prediction/chai1/chai1.py", line 584, in run_chai1_on_complex
E       result = ToolInstance.dispatch(
E                ^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 232, in dispatch
E       return cached.run(
E              ^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 515, in run
E       return self._run_persistent(
E              ^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 577, in _run_persistent
E       self._ensure_venv()
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 474, in _ensure_venv
E       self._create_venv()
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 864, in _create_venv
E       raise RuntimeError(
E   RuntimeError: 'chai1' may not be compatible with your system. setup.sh failed (exit 1).
```

### ❌ `structure_prediction`

**Test**: `tests/structure_prediction_tests/test_structure_prediction.py::test_folding[two_complex-boltz2-without_msa]`

```
tests/structure_prediction_tests/test_structure_prediction.py:334: in test_folding
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
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 159, in wrapper
E       result = func(inputs, config, instance)
E                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_cache.py", line 460, in wrapper
E       return func(*args, **kwargs)
E              ^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/structure_prediction/boltz2/boltz2.py", line 334, in run_boltz2
E       run_boltz2_on_complex(
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/structure_prediction/boltz2/boltz2.py", line 619, in run_boltz2_on_complex
E       output_data = ToolInstance.dispatch(
E                     ^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 232, in dispatch
E       return cached.run(
E              ^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 515, in run
E       return self._run_persistent(
E              ^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 616, in _run_persistent
E       return self._worker.send(input_dict, timeout=timeout)
E              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/persistent_worker.py", line 216, in send
E       raise TimeoutError(
E   TimeoutError: Worker for boltz2 timed out after 600s
```

---
*Generated at 2026-02-17 08:53:20 by `pytest --env-report`*