# Darwin arm64 Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-100%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-15-brightgreen) ![Failed](https://img.shields.io/badge/failed-0-red) ![Skipped](https://img.shields.io/badge/skipped-0-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Darwin Darwin 25.3.0 |
| **Architecture** | arm64 |
| **Hostname** | `Daniels-MacBook-Pro-5.local` |
| **Python** | 3.12.12 |
| **RAM** | 16.0 GB |
| **GPU** | None |
| **Conda Env** | `bio-programming` |

## Git

- **Commit**: `cbc0b961f0df`
- **Branch**: `dguo/bioemu-py311-fix`
- **Dirty**: No

## Environment Variables

### Parent Process Environment

```
COLORTERM=truecolor
COMMAND_MODE=unix2003
CONDA_CHANGEPS1=false
CONDA_DEFAULT_ENV=bio-programming
CONDA_EXE=/opt/miniconda3/bin/conda
CONDA_PREFIX=/opt/miniconda3/envs/bio-programming
CONDA_PREFIX_1=/opt/miniconda3
CONDA_PROMPT_MODIFIER=
CONDA_PYTHON_EXE=/opt/miniconda3/bin/python
CONDA_SHLVL=2
DISABLE_PANDERA_IMPORT_WARNING=True
DISPLAY=/private/tmp/com.apple.launchd.4rcYEp7e88/org.xquartz:0
HOME=/Users/danielguo
HOMEBREW_CELLAR=/opt/homebrew/Cellar
HOMEBREW_PREFIX=/opt/homebrew
HOMEBREW_REPOSITORY=/opt/homebrew
INFOPATH=/opt/homebrew/share/info:/opt/homebrew/share/info:
LANG=en_US.UTF-8
LOGNAME=danielguo
NVM_BIN=/Users/danielguo/.nvm/versions/node/v22.14.0/bin
NVM_CD_FLAGS=-q
NVM_DIR=/Users/danielguo/.nvm
NVM_INC=/Users/danielguo/.nvm/versions/node/v22.14.0/include/node
OLDPWD=/Users/danielguo/Research/darwin/bio-programming
OSLogRateLimit=64
PATH=/Users/danielguo/.local/bin:/opt/homebrew/opt/gnu-getopt/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/Users/danielguo/.local/bin:/Users/danielguo/.nvm/versions/node/v22.14.0/bin:/opt/homebrew/opt/gnu-get...
PWD=/Users/danielguo/Research/darwin/bio-programming/bio-programming-tools
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
RDBASE=/opt/miniconda3/envs/bio-programming/lib/python3.12/site-packages/rdkit
SHELL=/bin/zsh
SHLVL=2
TERM=xterm-256color
TERM_PROGRAM=WarpTerminal
TERM_PROGRAM_VERSION=v0.2026.02.18.08.22.stable_02
TMPDIR=/var/folders/rs/6dqw0_k1125fl7f7_9h85hgh0000gn/T/
USER=danielguo
WARP_HONOR_PS1=0
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
XPC_FLAGS=0x0
XPC_SERVICE_NAME=0
_=/opt/miniconda3/envs/bio-programming/bin/pytest
_CE_CONDA=
_CE_M=
__CFBundleIdentifier=dev.warp.Warp-Stable
__CF_USER_TEXT_ENCODING=0x1F5:0x0:0x0
```

### Subprocess Environment (passed to tools)

```
CONDA_DEFAULT_ENV=bio-programming
CONDA_PREFIX=/opt/miniconda3/envs/bio-programming
CONDA_SHLVL=2
CUDA_VISIBLE_DEVICES=
DETECTED_COMPUTE_PLATFORM=cpu
HOME=/Users/danielguo
JAX_PLATFORMS=cpu
LANG=en_US.UTF-8
LOGNAME=danielguo
PATH=/Users/danielguo/Research/darwin/bio-programming/bio-programming-tools/tool_envs/viennarna_env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RECOMMENDED_JAX_SPEC=jax
RECOMMENDED_TORCH_SPEC=torch
SHELL=/bin/zsh
TMPDIR=/var/folders/rs/6dqw0_k1125fl7f7_9h85hgh0000gn/T/
TORCH_HOME=/Users/danielguo/Research/darwin/bio-programming/bio-programming-tools/tool_envs/viennarna_env/cache/torch
USER=danielguo
```

## Results by Category

### Gene Annotation (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `blast` | no | ✅ | 56.6s | ✅ Pass |
| `minced` | no | ✅ | 13.8s | ✅ Pass |
| `mmseqs` | no | ✅ | 18.2s | ✅ Pass |
| `pyhmmer` | no | ✅ | 15.0s | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `orfipy` | no | ✅ | 17.2s | ✅ Pass |
| `prodigal` | no | ✅ | 14.3s | ✅ Pass |

### Sequence Alignment (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `mafft` | no | ✅ | 19.1s | ✅ Pass |

### Structure Alignment (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `tmalign` | no | ✅ | 18.7s | ✅ Pass |
| `tmalign` | no | ✅ | 0.1s | ✅ Pass |
| `usalign` | no | ✅ | 21.9s | ✅ Pass |
| `usalign` | no | ✅ | 0.1s | ✅ Pass |

### Structure Prediction (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `viennarna` | no | ✅ | 13.8s | ✅ Pass |

### Unknown (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `crispr_tracr` | no | ✅ | 230.3s | ✅ Pass |
| `local_colabfold_search` | no | — | 38.2s | ✅ Pass |
| `structure_metrics` | no | ✅ | 19.2s | ✅ Pass |

---
*Generated at 2026-02-25 01:05:37 by `pytest --env-report`*