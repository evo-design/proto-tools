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
| **Conda Env** | `proto-language` |

## Git

- **Commit**: `ab8a09d645b0`
- **Branch**: `main`
- **Dirty**: No

## Environment Variables

### Parent Process Environment

```
AR=arm64-apple-darwin20.0.0-ar
AS=arm64-apple-darwin20.0.0-as
CHECKSYMS=arm64-apple-darwin20.0.0-checksyms
CLAUDECODE=1
CLAUDE_CODE_ENTRYPOINT=cli
CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
COLORTERM=truecolor
COMMAND_MODE=unix2003
CONDA_CHANGEPS1=false
CONDA_DEFAULT_ENV=proto-language
CONDA_EXE=/opt/miniconda3/bin/conda
CONDA_PREFIX=/opt/miniconda3/envs/proto-language
CONDA_PREFIX_1=/opt/miniconda3
CONDA_PROMPT_MODIFIER=
CONDA_PYTHON_EXE=/opt/miniconda3/bin/python
CONDA_SHLVL=2
COREPACK_ENABLE_AUTO_PIN=0
DISABLE_PANDERA_IMPORT_WARNING=True
DISPLAY=/private/tmp/com.apple.launchd.4rcYEp7e88/org.xquartz:0
GIT_EDITOR=true
GSETTINGS_SCHEMA_DIR=/opt/miniconda3/envs/proto-language/share/glib-2.0/schemas
GSETTINGS_SCHEMA_DIR_CONDA_BACKUP=
HOME=/Users/danielguo
HOMEBREW_CELLAR=/opt/homebrew/Cellar
HOMEBREW_PREFIX=/opt/homebrew
HOMEBREW_REPOSITORY=/opt/homebrew
INFOPATH=/opt/homebrew/share/info:/opt/homebrew/share/info:
INSTALL_NAME_TOOL=arm64-apple-darwin20.0.0-install_name_tool
LANG=en_US.UTF-8
LD=arm64-apple-darwin20.0.0-ld
LIBTOOL=arm64-apple-darwin20.0.0-libtool
LIPO=arm64-apple-darwin20.0.0-lipo
LOGNAME=danielguo
NM=arm64-apple-darwin20.0.0-nm
NMEDIT=arm64-apple-darwin20.0.0-nmedit
NVM_BIN=/Users/danielguo/.nvm/versions/node/v22.14.0/bin
NVM_CD_FLAGS=-q
NVM_DIR=/Users/danielguo/.nvm
NVM_INC=/Users/danielguo/.nvm/versions/node/v22.14.0/include/node
NoDefaultCurrentDirectoryInExePath=1
OLDPWD=/Users/danielguo/Research/darwin/proto-language/proto-tools
OSLogRateLimit=64
OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta
OTOOL=arm64-apple-darwin20.0.0-otool
PAGESTUFF=arm64-apple-darwin20.0.0-pagestuff
PATH=/Users/danielguo/.local/bin:/opt/homebrew/opt/gnu-getopt/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/Users/danielguo/.local/bin:/Users/danielguo/.nvm/versions/node/v22.14.0/bin:/opt/homebrew/opt/gnu-get...
PWD=/Users/danielguo/Research/darwin/proto-language/proto-tools
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
RANLIB=arm64-apple-darwin20.0.0-ranlib
RDBASE=/opt/miniconda3/envs/proto-language/lib/python3.12/site-packages/rdkit
REDO_PREBINDING=arm64-apple-darwin20.0.0-redo_prebinding
SEGEDIT=arm64-apple-darwin20.0.0-segedit
SEG_ADDR_TABLE=arm64-apple-darwin20.0.0-seg_addr_table
SEG_HACK=arm64-apple-darwin20.0.0-seg_hack
SHELL=/bin/zsh
SHLVL=3
SIZE=arm64-apple-darwin20.0.0-size
STRINGS=arm64-apple-darwin20.0.0-strings
STRIP=arm64-apple-darwin20.0.0-strip
TERM=xterm-256color
TERM_PROGRAM=WarpTerminal
TERM_PROGRAM_VERSION=v0.2026.02.18.08.22.stable_02
TMPDIR=/var/folders/rs/6dqw0_k1125fl7f7_9h85hgh0000gn/T/
USER=danielguo
WARP_HONOR_PS1=0
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
XML_CATALOG_FILES=file:///opt/miniconda3/envs/proto-language/etc/xml/catalog file:///etc/xml/catalog
XPC_FLAGS=0x0
XPC_SERVICE_NAME=0
_=/opt/miniconda3/envs/proto-language/bin/python
_CE_CONDA=
_CE_M=
__CFBundleIdentifier=dev.warp.Warp-Stable
__CF_USER_TEXT_ENCODING=0x1F5:0x0:0x0
```

### Subprocess Environment (passed to tools)

```
CONDA_PREFIX=/Users/danielguo/Research/darwin/proto-language/proto-tools/tool_envs/viennarna_env
CUDA_VISIBLE_DEVICES=
DETECTED_COMPUTE_PLATFORM=cpu
HOME=/Users/danielguo
JAX_PLATFORMS=cpu
LANG=en_US.UTF-8
LD_LIBRARY_PATH=/opt/miniconda3/envs/proto-language/lib
LOGNAME=danielguo
PATH=/Users/danielguo/Research/darwin/proto-language/proto-tools/tool_envs/viennarna_env/bin:/Users/danielguo/.local/bin:/opt/homebrew/opt/gnu-getopt/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/Us...
RECOMMENDED_JAX_SPEC=jax
RECOMMENDED_TORCH_SPEC=torch
SHELL=/bin/zsh
TMPDIR=/var/folders/rs/6dqw0_k1125fl7f7_9h85hgh0000gn/T/
TORCH_HOME=/Users/danielguo/Research/darwin/proto-language/proto-tools/tool_envs/viennarna_env/cache/torch
USER=danielguo
VIRTUAL_ENV=/Users/danielguo/Research/darwin/proto-language/proto-tools/tool_envs/viennarna_env
```

## Results by Category

### Gene Annotation (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `blast` | no | ✅ | 79.6s | ✅ Pass |
| `minced` | no | ✅ | 12.5s | ✅ Pass |
| `mmseqs` | no | ✅ | 15.8s | ✅ Pass |
| `pyhmmer` | no | ✅ | 12.1s | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `orfipy` | no | ✅ | 14.0s | ✅ Pass |
| `prodigal` | no | ✅ | 11.3s | ✅ Pass |

### Sequence Alignment (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `mafft` | no | ✅ | 22.5s | ✅ Pass |

### Structure Alignment (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `tmalign` | no | ✅ | 14.2s | ✅ Pass |
| `tmalign` | no | ✅ | 0.1s | ✅ Pass |
| `usalign` | no | ✅ | 17.1s | ✅ Pass |
| `usalign` | no | ✅ | 0.1s | ✅ Pass |

### Structure Prediction (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `viennarna` | no | ✅ | 10.2s | ✅ Pass |

### Unknown (3/3)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `crispr_tracr` | no | ✅ | 244.1s | ✅ Pass |
| `local_colabfold_search` | no | — | 34.6s | ✅ Pass |
| `structure_metrics` | no | ✅ | 16.8s | ✅ Pass |

---
*Generated at 2026-02-26 23:49:41 by `pytest --env-report`*