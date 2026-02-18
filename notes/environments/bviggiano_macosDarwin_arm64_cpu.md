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

- **Commit**: `68eabf62a825`
- **Branch**: `macos_colabfold_search`
- **Dirty**: Yes

## Environment Variables

### Parent Process Environment

```
CLAUDECODE=1
CLAUDE_CODE_ENTRYPOINT=cli
CLICOLOR=1
COLORTERM=truecolor
CONDA_DEFAULT_ENV=bio_tools
CONDA_EXE=/Users/bviggiano/miniconda3/bin/conda
CONDA_PREFIX=/Users/bviggiano/miniconda3/envs/bio_tools
CONDA_PREFIX_1=/Users/bviggiano/miniconda3
CONDA_PROMPT_MODIFIER=(bio_tools) 
CONDA_PYTHON_EXE=/Users/bviggiano/miniconda3/bin/python
CONDA_SHLVL=2
COREPACK_ENABLE_AUTO_PIN=0
DISABLE_PANDERA_IMPORT_WARNING=True
DISPLAY=/private/tmp/com.apple.launchd.IuPqzb7ya8/org.xquartz:0
GIT_EDITOR=true
HOME=/Users/bviggiano
HOMEBREW_CELLAR=/opt/homebrew/Cellar
HOMEBREW_PREFIX=/opt/homebrew
HOMEBREW_REPOSITORY=/opt/homebrew
INFOPATH=/opt/homebrew/share/info:/opt/homebrew/share/info:
LANG=en_US.UTF-8
LOGNAME=bviggiano
LSCOLORS=ExFxBxDxCxegedabagacad
NoDefaultCurrentDirectoryInExePath=1
OLDPWD=/Users/bviggiano/Projects/bio-programming-tools
OSLogRateLimit=64
OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta
PATH=/Users/bviggiano/.local/bin:/Users/bviggiano/.juliaup/bin:/Users/bviggiano/miniconda3/envs/bio_tools/bin:/Users/bviggiano/miniconda3/condabin:/Library/Frameworks/Python.framework/Versions/3.10/bin:/op...
PWD=/Users/bviggiano/Projects/bio-programming-tools
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
RDBASE=/Users/bviggiano/miniconda3/envs/bio_tools/lib/python3.12/site-packages/rdkit
SHELL=/bin/zsh
SHLVL=2
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
CLAUDECODE=1
CLAUDE_CODE_ENTRYPOINT=cli
CLICOLOR=1
COLORTERM=truecolor
CONDA_PREFIX_1=/Users/bviggiano/miniconda3
COREPACK_ENABLE_AUTO_PIN=0
CUDA_VISIBLE_DEVICES=
DISABLE_PANDERA_IMPORT_WARNING=True
GIT_EDITOR=true
HOME=/Users/bviggiano
HOMEBREW_CELLAR=/opt/homebrew/Cellar
HOMEBREW_PREFIX=/opt/homebrew
HOMEBREW_REPOSITORY=/opt/homebrew
INFOPATH=/opt/homebrew/share/info:/opt/homebrew/share/info:
LANG=en_US.UTF-8
LOGNAME=bviggiano
LSCOLORS=ExFxBxDxCxegedabagacad
NoDefaultCurrentDirectoryInExePath=1
OLDPWD=/Users/bviggiano/Projects/bio-programming-tools
OSLogRateLimit=64
OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta
PATH=/Users/bviggiano/.local/bin:/Users/bviggiano/.juliaup/bin:/Users/bviggiano/miniconda3/envs/bio_tools/bin:/Users/bviggiano/miniconda3/condabin:/Library/Frameworks/Python.framework/Versions/3.10/bin:/op...
PWD=/Users/bviggiano/Projects/bio-programming-tools
PYTEST_CURRENT_TEST=tests/structure_prediction_tests/test_viennarna_secondary_structure_prediction.py::test_basic_folding (call)
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
RDBASE=/Users/bviggiano/miniconda3/envs/bio_tools/lib/python3.12/site-packages/rdkit
SHELL=/bin/zsh
SHLVL=2
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

### Gene Annotation (5/5)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `blast` | ✅ Pass | no | ✅ | 21.6s |
| `crispr_tracr` | ✅ Pass | no | ✅ | 262.3s |
| `minced` | ✅ Pass | no | ✅ | 7.5s |
| `mmseqs` | ✅ Pass | no | ✅ | 10.4s |
| `pyhmmer` | ✅ Pass | no | ✅ | 7.9s |

### Orf Prediction (2/2)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `orfipy` | ✅ Pass | no | ✅ | 9.9s |
| `prodigal` | ✅ Pass | no | ✅ | 6.4s |

### Sequence Alignment (2/2)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `colabfold_search` | ✅ Pass | no | ✅ | 31.9s |
| `mafft` | ✅ Pass | no | ✅ | 13.5s |

### Structure Dynamics (1/1)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `bioemu` | ✅ Pass | no | — | 0.0s |

### Structure Prediction (2/2)

| Tool | Status | Requires GPU | Venv build succeeded | Duration |
|------|--------|--------------|----------------------|----------|
| `structure_metrics` | ✅ Pass | no | ✅ | 13.5s |
| `viennarna` | ✅ Pass | no | ✅ | 6.0s |

---
*Generated at 2026-02-17 20:46:02 by `pytest --env-report`*