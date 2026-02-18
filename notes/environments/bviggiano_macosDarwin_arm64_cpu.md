# Darwin arm64 Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-100%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-12-brightgreen) ![Failed](https://img.shields.io/badge/failed-0-red) ![Skipped](https://img.shields.io/badge/skipped-0-lightgrey)

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

- **Commit**: `808fb567d7ce`
- **Branch**: `env-report-improvements`
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
TERM=xterm-256color
TERM_PROGRAM=Apple_Terminal
TERM_PROGRAM_VERSION=466
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

### Gene Annotation (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `blast` | no | ✅ | 91.0s | ✅ Pass |
| `minced` | no | ✅ | 7.6s | ✅ Pass |
| `mmseqs` | no | ✅ | 14.6s | ✅ Pass |
| `pyhmmer` | no | ✅ | 8.9s | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `orfipy` | no | ✅ | 12.4s | ✅ Pass |
| `prodigal` | no | ✅ | 7.9s | ✅ Pass |

### Sequence Alignment (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `mafft` | no | ✅ | 18.2s | ✅ Pass |

### Structure Dynamics (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `bioemu` | no | — | 0.0s | ✅ Pass |

### Structure Prediction (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `viennarna` | no | ✅ | 7.0s | ✅ Pass |

### Unknown (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `crispr_tracr` | no | ✅ | 369.9s | ✅ Pass |
| `local_colabfold_search` | no | — | 39.9s | ✅ Pass |
| `structure_metrics` | no | ✅ | 15.2s | ✅ Pass |

---
*Generated at 2026-02-17 23:12:15 by `pytest --env-report`*