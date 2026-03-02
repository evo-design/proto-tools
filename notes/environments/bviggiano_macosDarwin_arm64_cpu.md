# Darwin arm64 Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-100%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-15-brightgreen) ![Failed](https://img.shields.io/badge/failed-0-red) ![Skipped](https://img.shields.io/badge/skipped-0-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Darwin Darwin 25.3.0 |
| **Architecture** | arm64 |
| **Hostname** | `spock-2.local` |
| **Python** | 3.10.13 |
| **RAM** | 64.0 GB |
| **GPU** | None |
| **Conda Env** | `base` |

## Git

- **Commit**: `681dcfff0315`
- **Branch**: `device_management`
- **Dirty**: No

## Environment Variables

### Parent Process Environment

```
CLICOLOR=1
COLORTERM=truecolor
CONDA_DEFAULT_ENV=base
CONDA_EXE=/Users/bviggiano/miniconda3/bin/conda
CONDA_PREFIX=/Users/bviggiano/miniconda3
CONDA_PROMPT_MODIFIER=(base) 
CONDA_PYTHON_EXE=/Users/bviggiano/miniconda3/bin/python
CONDA_SHLVL=1
DISABLE_PANDERA_IMPORT_WARNING=True
DISPLAY=/private/tmp/com.apple.launchd.xYoeIqcqTo/org.xquartz:0
HOME=/Users/bviggiano
HOMEBREW_CELLAR=/opt/homebrew/Cellar
HOMEBREW_PREFIX=/opt/homebrew
HOMEBREW_REPOSITORY=/opt/homebrew
INFOPATH=/opt/homebrew/share/info:/opt/homebrew/share/info:
LANG=en_US.UTF-8
LOGNAME=bviggiano
LSCOLORS=ExFxBxDxCxegedabagacad
OLDPWD=/Users/bviggiano/Projects/bio-programming-tools
OSLogRateLimit=64
PATH=/Users/bviggiano/.local/bin:/Users/bviggiano/.juliaup/bin:/Library/Frameworks/Python.framework/Versions/3.10/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/System/Cryptexes/App/usr/bin:/usr/...
PWD=/Users/bviggiano/Projects/bio-programming-tools
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
RDBASE=/Users/bviggiano/miniconda3/lib/python3.10/site-packages/rdkit
SHELL=/bin/zsh
SHLVL=2
TERM=tmux-256color
TERM_PROGRAM=tmux
TERM_PROGRAM_VERSION=3.6a
TMPDIR=/var/folders/6f/gqwcqlqn3sxdz7rlzjxk7pgh0000gn/T/
TMUX=/private/tmp/tmux-501/default,98298,0
TMUX_PANE=%0
USER=bviggiano
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
XML_CATALOG_FILES=file:///Users/bviggiano/miniconda3/etc/xml/catalog file:///etc/xml/catalog
XPC_FLAGS=0x0
XPC_SERVICE_NAME=0
_=/Users/bviggiano/miniconda3/bin/pytest
__CFBundleIdentifier=com.apple.Terminal
```

### Subprocess Environment (passed to tools)

```
CONDA_PREFIX=/Users/bviggiano/Projects/bio-programming-tools/tool_envs/viennarna_env
DETECTED_COMPUTE_PLATFORM=cpu
HOME=/Users/bviggiano
LANG=en_US.UTF-8
LD_LIBRARY_PATH=/Users/bviggiano/Projects/bio-programming-tools/.micromamba/foundation_env/lib
LOGNAME=bviggiano
PATH=/Users/bviggiano/Projects/bio-programming-tools/tool_envs/viennarna_env/bin:/Users/bviggiano/Projects/bio-programming-tools/.micromamba/foundation_env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr...
RECOMMENDED_JAX_SPEC=jax
RECOMMENDED_TORCH_SPEC=torch
SHELL=/bin/zsh
TMPDIR=/var/folders/6f/gqwcqlqn3sxdz7rlzjxk7pgh0000gn/T/
TORCH_HOME=/Users/bviggiano/Projects/bio-programming-tools/tool_envs/viennarna_env/cache/torch
USER=bviggiano
VIRTUAL_ENV=/Users/bviggiano/Projects/bio-programming-tools/tool_envs/viennarna_env
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Results by Category

### Gene Annotation (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `blast` | no | ✅ | 59.7s | ✅ Pass |
| `minced` | no | ✅ | 14.1s | ✅ Pass |
| `mmseqs` | no | ✅ | 17.5s | ✅ Pass |
| `pyhmmer` | no | ✅ | 15.7s | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `orfipy` | no | ✅ | 18.9s | ✅ Pass |
| `prodigal` | no | ✅ | 15.0s | ✅ Pass |

### Sequence Alignment (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `mafft` | no | ✅ | 21.8s | ✅ Pass |

### Structure Alignment (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `tmalign` | no | ✅ | 18.1s | ✅ Pass |
| `tmalign` | no | ✅ | 0.1s | ✅ Pass |
| `usalign` | no | ✅ | 22.2s | ✅ Pass |
| `usalign` | no | ✅ | 0.1s | ✅ Pass |

### Structure Prediction (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `viennarna` | no | ✅ | 14.1s | ✅ Pass |

### Unknown (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `crispr_tracr` | no | ✅ | 221.1s | ✅ Pass |
| `local_colabfold_search` | no | — | 40.9s | ✅ Pass |
| `structure_metrics` | no | ✅ | 20.6s | ✅ Pass |

---
*Generated at 2026-03-01 12:58:58 by `pytest --env-report`*