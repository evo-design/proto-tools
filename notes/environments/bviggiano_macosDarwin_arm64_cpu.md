# Darwin arm64 Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-91%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-11-brightgreen) ![Failed](https://img.shields.io/badge/failed-1-red) ![Skipped](https://img.shields.io/badge/skipped-0-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Darwin Darwin 25.2.0 |
| **Architecture** | arm64 |
| **Hostname** | `spock-2.local` |
| **Python** | 3.12.12 |
| **RAM** | 64.0 GB |
| **GPU** | None |
| **Conda Env** | `bio_tools` |

## Git

- **Commit**: `064faae518b2`
- **Branch**: `bv/env_testing`
- **Dirty**: No

## Environment Variables

### Parent Process Environment

```
CLICOLOR=1
COLORTERM=truecolor
CONDA_DEFAULT_ENV=bio_tools
CONDA_EXE=/Users/bviggiano/miniconda3/bin/conda
CONDA_PREFIX=/Users/bviggiano/miniconda3/envs/bio_tools
CONDA_PREFIX_1=/Users/bviggiano/miniconda3
CONDA_PROMPT_MODIFIER=(bio_tools) 
CONDA_PYTHON_EXE=/Users/bviggiano/miniconda3/bin/python
CONDA_SHLVL=2
DISABLE_PANDERA_IMPORT_WARNING=True
DISPLAY=/private/tmp/com.apple.launchd.IuPqzb7ya8/org.xquartz:0
HOME=/Users/bviggiano
HOMEBREW_CELLAR=/opt/homebrew/Cellar
HOMEBREW_PREFIX=/opt/homebrew
HOMEBREW_REPOSITORY=/opt/homebrew
INFOPATH=/opt/homebrew/share/info:
LANG=en_US.UTF-8
LOGNAME=bviggiano
LSCOLORS=ExFxBxDxCxegedabagacad
OLDPWD=/Users/bviggiano/Projects
OSLogRateLimit=64
PATH=/Users/bviggiano/.local/bin:/Users/bviggiano/.juliaup/bin:/Users/bviggiano/miniconda3/envs/bio_tools/bin:/Users/bviggiano/miniconda3/condabin:/Library/Frameworks/Python.framework/Versions/3.10/bin:/op...
PWD=/Users/bviggiano/Projects/bio-programming-tools
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
RDBASE=/Users/bviggiano/miniconda3/envs/bio_tools/lib/python3.12/site-packages/rdkit
SHELL=/bin/zsh
SHLVL=1
TERM=xterm-256color
TERM_PROGRAM=Apple_Terminal
TERM_PROGRAM_VERSION=466
TERM_SESSION_ID=FC8C6BB1-4C1F-47A3-82BE-71CFD02F9B9E
TMPDIR=/var/folders/6f/gqwcqlqn3sxdz7rlzjxk7pgh0000gn/T/
USER=bviggiano
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
XPC_FLAGS=0x0
XPC_SERVICE_NAME=0
_=/Users/bviggiano/miniconda3/envs/bio_tools/bin/pytest
__CFBundleIdentifier=com.apple.Terminal
```

### Subprocess Environment (passed to tools)

```
CLICOLOR=1
COLORTERM=truecolor
CONDA_PREFIX_1=/Users/bviggiano/miniconda3
CUDA_VISIBLE_DEVICES=
DISABLE_PANDERA_IMPORT_WARNING=True
HOME=/Users/bviggiano
HOMEBREW_CELLAR=/opt/homebrew/Cellar
HOMEBREW_PREFIX=/opt/homebrew
HOMEBREW_REPOSITORY=/opt/homebrew
INFOPATH=/opt/homebrew/share/info:
LANG=en_US.UTF-8
LOGNAME=bviggiano
LSCOLORS=ExFxBxDxCxegedabagacad
OLDPWD=/Users/bviggiano/Projects
OSLogRateLimit=64
PATH=/Users/bviggiano/.local/bin:/Users/bviggiano/.juliaup/bin:/Users/bviggiano/miniconda3/envs/bio_tools/bin:/Users/bviggiano/miniconda3/condabin:/Library/Frameworks/Python.framework/Versions/3.10/bin:/op...
PWD=/Users/bviggiano/Projects/bio-programming-tools
PYTEST_CURRENT_TEST=tests/structure_prediction_tests/test_viennarna_secondary_structure_prediction.py::test_basic_folding (call)
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
RDBASE=/Users/bviggiano/miniconda3/envs/bio_tools/lib/python3.12/site-packages/rdkit
SHELL=/bin/zsh
SHLVL=1
SSH_AUTH_SOCK=/private/tmp/com.apple.launchd.UFtJ0GFSBp/Listeners
TERM=xterm-256color
TERM_PROGRAM=Apple_Terminal
TERM_PROGRAM_VERSION=466
TERM_SESSION_ID=FC8C6BB1-4C1F-47A3-82BE-71CFD02F9B9E
TMPDIR=/var/folders/6f/gqwcqlqn3sxdz7rlzjxk7pgh0000gn/T/
USER=bviggiano
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
XPC_FLAGS=0x0
XPC_SERVICE_NAME=0
_=/Users/bviggiano/miniconda3/envs/bio_tools/bin/pytest
__CFBundleIdentifier=com.apple.Terminal
```

## Results by Category

### Gene Annotation (5/5)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `blast` | ✅ Pass | no | ✅ | 26.2s |
| `crispr_tracr` | ✅ Pass | no | ✅ | 304.5s |
| `minced` | ✅ Pass | no | ✅ | 7.6s |
| `mmseqs` | ✅ Pass | no | ✅ | 12.6s |
| `pyhmmer` | ✅ Pass | no | ✅ | 9.6s |

### Orf Prediction (2/2)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `orfipy` | ✅ Pass | no | ✅ | 11.7s |
| `prodigal` | ✅ Pass | no | ✅ | 7.2s |

### Sequence Alignment (1/2)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `colabfold_search` | ❌ Fail | no | ✅ | 11.3s |
| `mafft` | ✅ Pass | no | ✅ | 16.0s |

### Structure Dynamics (1/1)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `bioemu` | ✅ Pass | no | — | 0.0s |

### Structure Prediction (2/2)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `structure_metrics` | ✅ Pass | no | ✅ | 15.1s |
| `viennarna` | ✅ Pass | no | ✅ | 7.5s |

## Failure Details

### ❌ `colabfold_search`

**Test**: `tests/sequence_alignment_tests/test_local_colabfold_search.py::TestColabfoldSearchExecutionDebugDatabase::test_finding_self_in_database`

```
tests/sequence_alignment_tests/test_local_colabfold_search.py:275: in test_finding_self_in_database
    validate_output(result)
tests/tool_infra_tests/test_export_functionality.py:102: in validate_output
    assert output.success is True, f"Tool execution failed: {output}"
E   AssertionError: Tool execution failed: 
E     ================================================================================
E     colabfold-search: TOOL FAILURE after 11.1661s
E     ================================================================================
E     
E     Error 1:
E     colabfold_search failed: colabfold_search failed with exit code 1
E     STDERR: INFO:colabfold.mmseqs.search:Running mmseqs createdb /var/folders/6f/gqwcqlqn3sxdz7rlzjxk7pgh0000gn/T/tmpoxkizlbx/msas/query.fas /var/folders/6f/gqwcqlqn3sxdz7rlzjxk7pgh0000gn/T/tmpoxkizlbx/msas/qdb --shuffle 0 --dbtype 1
E     Traceback (most recent call last):
E       File "/Users/bviggiano/Projects/bio-programming-tools/.venvs/colabfold_search_env/bin/colabfold_search", line 10, in <module>
E         sys.exit(main())
E                  ^^^^^^
E       File "/Users/bviggiano/Projects/bio-programming-tools/.venvs/colabfold_search_env/lib/python3.12/site-packages/colabfold/mmseqs/search.py", line 449, in main
E         run_mmseqs(
E       File "/Users/bviggiano/Projects/bio-programming-tools/.venvs/colabfold_search_env/lib/python3.12/site-packages/colabfold/mmseqs/search.py", line 46, in run_mmseqs
E         subprocess.check_call([mmseqs] + params)
E       File "/Users/bviggiano/miniconda3/envs/bio_tools/lib/python3.12/subprocess.py", line 408, in check_call
E         retcode = call(*popenargs, **kwargs)
E                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/Users/bviggiano/miniconda3/envs/bio_tools/lib/python3.12/subprocess.py", line 389, in call
E         with Popen(*popenargs, **kwargs) as p:
E              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/Users/bviggiano/miniconda3/envs/bio_tools/lib/python3.12/subprocess.py", line 1026, in __init__
E         self._execute_child(args, executable, preexec_fn, close_fds,
E       File "/Users/bviggiano/miniconda3/envs/bio_tools/lib/python3.12/subprocess.py", line 1955, in _execute_child
E         raise child_exception_type(errno_num, err_msg, err_filename)
E     FileNotFoundError: [Errno 2] No such file or directory: PosixPath('mmseqs')
E     
E     
E     Error 2:
E     Traceback (most recent call last):
E       File "/Users/bviggiano/Projects/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 159, in wrapper
E         result = func(inputs, config, instance)
E                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/Users/bviggiano/Projects/bio-programming-tools/bio_programming_tools/tools/sequence_alignment/colabfold_search/colabfold_search.py", line 485, in run_colabfold_search
E         return _local_search(sequences, sequence_ids, config, msa_out_dir, instance=instance)
E                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/Users/bviggiano/Projects/bio-programming-tools/bio_programming_tools/tools/sequence_alignment/colabfold_search/colabfold_search.py", line 676, in _local_search
E         raise RuntimeError(f"colabfold_search failed: {error_msg}")
E     RuntimeError: colabfold_search failed: colabfold_search failed with exit code 1
E     STDERR: INFO:colabfold.mmseqs.search:Running mmseqs createdb /var/folders/6f/gqwcqlqn3sxdz7rlzjxk7pgh0000gn/T/tmpoxkizlbx/msas/query.fas /var/folders/6f/gqwcqlqn3sxdz7rlzjxk7pgh0000gn/T/tmpoxkizlbx/msas/qdb --shuffle 0 --dbtype 1
E     Traceback (most recent call last):
E       File "/Users/bviggiano/Projects/bio-programming-tools/.venvs/colabfold_search_env/bin/colabfold_search", line 10, in <module>
E         sys.exit(main())
E                  ^^^^^^
E       File "/Users/bviggiano/Projects/bio-programming-tools/.venvs/colabfold_search_env/lib/python3.12/site-packages/colabfold/mmseqs/search.py", line 449, in main
E         run_mmseqs(
E       File "/Users/bviggiano/Projects/bio-programming-tools/.venvs/colabfold_search_env/lib/python3.12/site-packages/colabfold/mmseqs/search.py", line 46, in run_mmseqs
E         subprocess.check_call([mmseqs] + params)
E       File "/Users/bviggiano/miniconda3/envs/bio_tools/lib/python3.12/subprocess.py", line 408, in check_call
E         retcode = call(*popenargs, **kwargs)
E                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/Users/bviggiano/miniconda3/envs/bio_tools/lib/python3.12/subprocess.py", line 389, in call
E         with Popen(*popenargs, **kwargs) as p:
E              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       File "/Users/bviggiano/miniconda3/envs/bio_tools/lib/python3.12/subprocess.py", line 1026, in __init__
E         self._execute_child(args, executable, preexec_fn, close_fds,
E       File "/Users/bviggiano/miniconda3/envs/bio_tools/lib/python3.12/subprocess.py", line 1955, in _execute_child
E         raise child_exception_type(errno_num, err_msg, err_filename)
E     FileNotFoundError: [Errno 2] No such file or directory: PosixPath('mmseqs')
E     
E     
E     ================================================================================
E   assert False is True
E    +  where False = ColabfoldSearchOutput(tool_id, execution_time, timestamp, success, warnings, errors, metadata).success
```

---
*Generated at 2026-02-17 08:44:11 by `pytest --env-report`*