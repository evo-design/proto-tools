# Chimera x86_64 Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-82%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-23-brightgreen) ![Failed](https://img.shields.io/badge/failed-5-red) ![Skipped](https://img.shields.io/badge/skipped-0-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Linux Linux 5.15.0-164-generic |
| **Architecture** | x86_64 |
| **Hostname** | `GPUC870` |
| **Python** | 3.12.12 |
| **RAM** | 1007.4 GB |
| **GPU** | 1× NVIDIA H100 80GB HBM3 |
| **CUDA** | 12.2 |
| **Mamba Env** | `bio_tools` |

## Git

- **Commit**: `064faae518b2`
- **Branch**: `bv/env_testing`
- **Dirty**: No

## Environment Variables

### Parent Process Environment

```
BLASTDB=/common_datasets/external/databases/blast
CONDA_DEFAULT_ENV=bio_tools
CONDA_EXE=/home/bviggiano/miniforge3/bin/conda
CONDA_PREFIX=/home/bviggiano/miniforge3/envs/bio_tools
CONDA_PREFIX_1=/home/bviggiano/miniforge3
CONDA_PROMPT_MODIFIER=(bio_tools)
CONDA_PYTHON_EXE=/home/bviggiano/miniforge3/bin/python
CONDA_SHLVL=2
DISABLE_PANDERA_IMPORT_WARNING=True
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
OLDPWD=/home/bviggiano/main/codebases/bio-programming
PATH=/home/bviggiano/.local/bin:/home/bviggiano/bin:/home/bviggiano/miniforge3/envs/bio_tools/bin:/home/bviggiano/miniforge3/condabin:/usr/local/ncbi/sra-tools/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/...
PWD=/home/bviggiano/main/codebases/bio-programming/bio-programming-tools
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
RDBASE=/home/bviggiano/miniforge3/envs/bio_tools/lib/python3.12/site-packages/rdkit
SHELL=/bin/bash
SHLVL=1
SLURM_JOB_ID=1700008
TERM=xterm-256color
USER=bviggiano
XDG_DATA_DIRS=/usr/local/share:/usr/share:/var/lib/snapd/desktop
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
XML_CATALOG_FILES=file:///home/bviggiano/miniforge3/etc/xml/catalog file:///etc/xml/catalog
_=/home/bviggiano/miniforge3/envs/bio_tools/bin/pytest
_CE_CONDA=
_CE_M=
_CONDA_EXE=/home/bviggiano/miniforge3/bin/conda
_CONDA_ROOT=/home/bviggiano/miniforge3
```

### Subprocess Environment (passed to tools)

```
BLASTDB=/common_datasets/external/databases/blast
CONDA_PREFIX_1=/home/bviggiano/miniforge3
CUDA_VISIBLE_DEVICES=0
DISABLE_PANDERA_IMPORT_WARNING=True
HOME=/home/bviggiano
LANG=C.UTF-8
LESSCLOSE=/usr/bin/lesspipe %s %s
LESSOPEN=| /usr/bin/lesspipe %s
LOGNAME=bviggiano
LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.a...
MAMBA_EXE=/home/bviggiano/miniforge3/bin/mamba
MAMBA_ROOT_PREFIX=/home/bviggiano/.local/share/mamba
MOTD_SHOWN=pam
OLDPWD=/home/bviggiano/main/codebases/bio-programming
PATH=/home/bviggiano/.local/bin:/home/bviggiano/bin:/home/bviggiano/miniforge3/envs/bio_tools/bin:/home/bviggiano/miniforge3/condabin:/usr/local/ncbi/sra-tools/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/...
PWD=/home/bviggiano/main/codebases/bio-programming/bio-programming-tools
PYTEST_CURRENT_TEST=tests/test_rna_splicing.py::test_splice_transformer_gpu (call)
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
RDBASE=/home/bviggiano/miniforge3/envs/bio_tools/lib/python3.12/site-packages/rdkit
SHELL=/bin/bash
SHLVL=1
SLURM_JOB_ID=1700008
TERM=xterm-256color
USER=bviggiano
XDG_DATA_DIRS=/usr/local/share:/usr/share:/var/lib/snapd/desktop
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
XML_CATALOG_FILES=file:///home/bviggiano/miniforge3/etc/xml/catalog file:///etc/xml/catalog
_=/home/bviggiano/miniforge3/envs/bio_tools/bin/pytest
_CE_M=
```

## Results by Category

### Causal Models (2/3)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `evo1` | ✅ Pass | yes | ✅ | 91.2s |
| `evo2` | ❌ Fail | yes | ✅ | 203.5s |
| `progen2` | ✅ Pass | yes | ✅ | 28.8s |

### Gene Annotation (5/5)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `blast` | ✅ Pass | no | ✅ | 22.6s |
| `crispr_tracr` | ✅ Pass | no | ✅ | 444.5s |
| `minced` | ✅ Pass | no | ✅ | 7.0s |
| `mmseqs` | ✅ Pass | no | ✅ | 30.7s |
| `pyhmmer` | ✅ Pass | no | ✅ | 7.8s |

### Inverse Folding (2/2)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `ligandmpnn` | ✅ Pass | yes | ✅ | 62.1s |
| `proteinmpnn` | ✅ Pass | yes | ✅ | 27.3s |

### Masked Models (2/2)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `esm2` | ✅ Pass | yes | ✅ | 43.3s |
| `esm3` | ✅ Pass | yes | ✅ | 41.3s |

### Orf Prediction (2/2)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `orfipy` | ✅ Pass | no | ✅ | 7.9s |
| `prodigal` | ✅ Pass | no | ✅ | 5.2s |

### Rna Splicing (1/1)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `splice_transformer` | ✅ Pass | yes | ✅ | 29.1s |

### Sequence Alignment (1/2)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `colabfold_search` | ❌ Fail | no | ✅ | 27.4s |
| `mafft` | ✅ Pass | no | ✅ | 10.9s |

### Sequence Scoring (3/3)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `alphagenome` | ✅ Pass | yes | ✅ | 188.1s |
| `borzoi` | ✅ Pass | yes | ✅ | 51.3s |
| `enformer` | ✅ Pass | yes | ✅ | 39.9s |

### Structure Design (0/1)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `rfdiffusion3` | ❌ Fail | yes | ✅ | 36.6s |

### Structure Dynamics (1/1)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `bioemu` | ✅ Pass | no | — | 0.0s |

### Structure Prediction (4/6)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `protenix` | ❌ Fail | yes | ✅ | 828.6s |
| `structure_metrics` | ✅ Pass | no | ✅ | 8.3s |
| `structure_prediction` | ✅ Pass | yes | — | 49.4s |
| `structure_prediction` | ❌ Fail | yes | — | 282.8s |
| `structure_prediction` | ✅ Pass | yes | — | 220.6s |
| `viennarna` | ✅ Pass | no | ✅ | 5.1s |

## Failure Details

### ❌ `evo2`

**Test**: `tests/language_model_tests/test_evo2.py::test_evo2_sample_tool`

```
tests/language_model_tests/test_evo2.py:102: in test_evo2_sample_tool
    validate_output(result)
tests/tool_infra_tests/test_export_functionality.py:102: in validate_output
    assert output.success is True, f"Tool execution failed: {output}"
E   AssertionError: Tool execution failed: 
E     ================================================================================
E     evo2-sample: TOOL FAILURE after 203.1512s
E     ================================================================================
E     
E     Error 1:
E     Worker for evo2 returned invalid JSON: '[02/17/26 09:02:38] WARNING  transformer_engine.pytorch.attention backends.py:98\n'
E     
E     Error 2:
E     Traceback (most recent call last):
E       File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/persistent_worker.py", line 228, in send
E         response = json.loads(response_line)
E                    ^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/miniforge3/envs/bio_tools/lib/python3.12/json/__init__.py", line 346, in loads
E         return _default_decoder.decode(s)
E                ^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/miniforge3/envs/bio_tools/lib/python3.12/json/decoder.py", line 338, in decode
E         obj, end = self.raw_decode(s, idx=_w(s, 0).end())
E                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/miniforge3/envs/bio_tools/lib/python3.12/json/decoder.py", line 354, in raw_decode
E         obj, end = self.scan_once(s, idx)
E                    ^^^^^^^^^^^^^^^^^^^^^^
E     json.decoder.JSONDecodeError: Expecting ',' delimiter: line 1 column 3 (char 2)
E     
E     The above exception was the direct cause of the following exception:
E     
E     Traceback (most recent call last):
E       File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 159, in wrapper
E         result = func(inputs, config, instance)
E                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/causal_models/evo2/evo2_sample.py", line 453, in run_evo2_sample
E         result = ToolInstance.dispatch(
E                  ^^^^^^^^^^^^^^^^^^^^^^
E       File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 232, in dispatch
E         return cached.run(
E                ^^^^^^^^^^^
E       File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 515, in run
E         return self._run_persistent(
E                ^^^^^^^^^^^^^^^^^^^^^
E       File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 616, in _run_persistent
E         return self._worker.send(input_dict, timeout=timeout)
E                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/persistent_worker.py", line 230, in send
E         raise RuntimeError(
E     RuntimeError: Worker for evo2 returned invalid JSON: '[02/17/26 09:02:38] WARNING  transformer_engine.pytorch.attention backends.py:98\n'
E     
E     
E     Warnings:
E       - unclosed file <_io.TextIOWrapper name=16 encoding='UTF-8'>
E     ================================================================================
E   assert False is True
E    +  where False = Evo2SampleOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata, logits, kv_caches).success
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
E     colabfold-search: TOOL FAILURE after 27.3539s
E     ================================================================================
E     
E     Error 1:
E     colabfold_search failed: colabfold_search failed with exit code 1
E     STDERR: INFO:colabfold.mmseqs.search:Running mmseqs createdb /tmp/tmp533v2w83/msas/query.fas /tmp/tmp533v2w83/msas/qdb --shuffle 0 --dbtype 1
E     Traceback (most recent call last):
E       File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/colabfold_search_env/bin/colabfold_search", line 10, in <module>
E         sys.exit(main())
E                  ^^^^^^
E       File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/colabfold_search_env/lib/python3.12/site-packages/colabfold/mmseqs/search.py", line 449, in main
E         run_mmseqs(
E       File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/colabfold_search_env/lib/python3.12/site-packages/colabfold/mmseqs/search.py", line 46, in run_mmseqs
E         subprocess.check_call([mmseqs] + params)
E       File "/home/bviggiano/miniforge3/envs/bio_tools/lib/python3.12/subprocess.py", line 408, in check_call
E         retcode = call(*popenargs, **kwargs)
E                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/miniforge3/envs/bio_tools/lib/python3.12/subprocess.py", line 389, in call
E         with Popen(*popenargs, **kwargs) as p:
E              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/miniforge3/envs/bio_tools/lib/python3.12/subprocess.py", line 1026, in __init__
E         self._execute_child(args, executable, preexec_fn, close_fds,
E       File "/home/bviggiano/miniforge3/envs/bio_tools/lib/python3.12/subprocess.py", line 1955, in _execute_child
E         raise child_exception_type(errno_num, err_msg, err_filename)
E     FileNotFoundError: [Errno 2] No such file or directory: PosixPath('mmseqs')
E     
E     
E     Error 2:
E     Traceback (most recent call last):
E       File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 159, in wrapper
E         result = func(inputs, config, instance)
E                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/sequence_alignment/colabfold_search/colabfold_search.py", line 485, in run_colabfold_search
E         return _local_search(sequences, sequence_ids, config, msa_out_dir, instance=instance)
E                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/sequence_alignment/colabfold_search/colabfold_search.py", line 676, in _local_search
E         raise RuntimeError(f"colabfold_search failed: {error_msg}")
E     RuntimeError: colabfold_search failed: colabfold_search failed with exit code 1
E     STDERR: INFO:colabfold.mmseqs.search:Running mmseqs createdb /tmp/tmp533v2w83/msas/query.fas /tmp/tmp533v2w83/msas/qdb --shuffle 0 --dbtype 1
E     Traceback (most recent call last):
E       File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/colabfold_search_env/bin/colabfold_search", line 10, in <module>
E         sys.exit(main())
E                  ^^^^^^
E       File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/colabfold_search_env/lib/python3.12/site-packages/colabfold/mmseqs/search.py", line 449, in main
E         run_mmseqs(
E       File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/.venvs/colabfold_search_env/lib/python3.12/site-packages/colabfold/mmseqs/search.py", line 46, in run_mmseqs
E         subprocess.check_call([mmseqs] + params)
E       File "/home/bviggiano/miniforge3/envs/bio_tools/lib/python3.12/subprocess.py", line 408, in check_call
E         retcode = call(*popenargs, **kwargs)
E                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/miniforge3/envs/bio_tools/lib/python3.12/subprocess.py", line 389, in call
E         with Popen(*popenargs, **kwargs) as p:
E              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/home/bviggiano/miniforge3/envs/bio_tools/lib/python3.12/subprocess.py", line 1026, in __init__
E         self._execute_child(args, executable, preexec_fn, close_fds,
E       File "/home/bviggiano/miniforge3/envs/bio_tools/lib/python3.12/subprocess.py", line 1955, in _execute_child
E         raise child_exception_type(errno_num, err_msg, err_filename)
E     FileNotFoundError: [Errno 2] No such file or directory: PosixPath('mmseqs')
E     
E     
E     ================================================================================
E   assert False is True
E    +  where False = ColabfoldSearchOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success
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
E   bio_programming_tools.utils.tool_io.ToolExecutionError: Attempt to access field of tool output after failure: TimeoutError: Worker for protenix timed out after 600s
E   
E   Error Messages:
E   Worker for protenix timed out after 600s
E   Traceback (most recent call last):
E     File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 159, in wrapper
E       result = func(inputs, config, instance)
E                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_cache.py", line 460, in wrapper
E       return func(*args, **kwargs)
E              ^^^^^^^^^^^^^^^^^^^^^
E     File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/structure_prediction/protenix/protenix.py", line 422, in run_protenix
E       output_data = ToolInstance.dispatch(
E                     ^^^^^^^^^^^^^^^^^^^^^^
E     File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 232, in dispatch
E       return cached.run(
E              ^^^^^^^^^^^
E     File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 515, in run
E       return self._run_persistent(
E              ^^^^^^^^^^^^^^^^^^^^^
E     File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 616, in _run_persistent
E       return self._worker.send(input_dict, timeout=timeout)
E              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/persistent_worker.py", line 216, in send
E       raise TimeoutError(
E   TimeoutError: Worker for protenix timed out after 600s
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
E   bio_programming_tools.utils.tool_io.ToolExecutionError: Attempt to access field of tool output after failure: RuntimeError: Worker for chai1 returned invalid JSON: 'Score=0.0477, writing output to /tmp/tmpti1i3x_l/output/pred.model_idx_0.cif\n'
E   
E   Error Messages:
E   Worker for chai1 returned invalid JSON: 'Score=0.0477, writing output to /tmp/tmpti1i3x_l/output/pred.model_idx_0.cif\n'
E   Traceback (most recent call last):
E     File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/persistent_worker.py", line 228, in send
E       response = json.loads(response_line)
E                  ^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/miniforge3/envs/bio_tools/lib/python3.12/json/__init__.py", line 346, in loads
E       return _default_decoder.decode(s)
E              ^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/miniforge3/envs/bio_tools/lib/python3.12/json/decoder.py", line 338, in decode
E       obj, end = self.raw_decode(s, idx=_w(s, 0).end())
E                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/home/bviggiano/miniforge3/envs/bio_tools/lib/python3.12/json/decoder.py", line 356, in raw_decode
E       raise JSONDecodeError("Expecting value", s, err.value) from None
E   json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
E   
E   The above exception was the direct cause of the following exception:
E   
E   Traceback (most recent call last):
E     File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 159, in wrapper
E       result = func(inputs, config, instance)
E                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_cache.py", line 460, in wrapper
E       return func(*args, **kwargs)
E              ^^^^^^^^^^^^^^^^^^^^^
E     File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/structure_prediction/chai1/chai1.py", line 305, in run_chai1
E       results.append(run_chai1_on_complex(comp=comp, config=config, instance=instance))
E                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/tools/structure_prediction/chai1/chai1.py", line 584, in run_chai1_on_complex
E       result = ToolInstance.dispatch(
E                ^^^^^^^^^^^^^^^^^^^^^^
E     File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 232, in dispatch
E       return cached.run(
E              ^^^^^^^^^^^
E     File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 515, in run
E       return self._run_persistent(
E              ^^^^^^^^^^^^^^^^^^^^^
E     File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 616, in _run_persistent
E       return self._worker.send(input_dict, timeout=timeout)
E              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/large_storage/hielab/bviggiano/codebases/bio-programming/bio-programming-tools/bio_programming_tools/utils/persistent_worker.py", line 230, in send
E       raise RuntimeError(
E   RuntimeError: Worker for chai1 returned invalid JSON: 'Score=0.0477, writing output to /tmp/tmpti1i3x_l/output/pred.model_idx_0.cif\n'
```

---
*Generated at 2026-02-17 09:33:01 by `pytest --env-report`*