# Linux x86_64 Environment Report

![Pass Rate](https://img.shields.io/badge/pass_rate-97%25-brightgreen) ![Passed](https://img.shields.io/badge/passed-41-brightgreen) ![Failed](https://img.shields.io/badge/failed-1-red) ![Skipped](https://img.shields.io/badge/skipped-0-lightgrey)

## Platform

| Property | Value |
|----------|-------|
| **OS** | Linux Linux 5.15.0-1086-nvidia |
| **Architecture** | x86_64 |
| **Hostname** | `ashleylab-h100` |
| **Python** | 3.12.13 |
| **RAM** | 2015.6 GB |
| **GPU** | 8× NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3, NVIDIA H100 80GB HBM3 |
| **CUDA** | 12.2 |
| **Conda Env** | `bio-tools` |

## Git

- **Commit**: `afcf77285f71`
- **Branch**: `bv/parallel`
- **Dirty**: No

## Environment Variables

### Parent Process Environment

```
CONDA_DEFAULT_ENV=bio-tools
CONDA_EXE=/home/viggiano/miniconda3/bin/conda
CONDA_PREFIX=/projects/viggiano/envs/bio-tools
CONDA_PREFIX_1=/home/viggiano/miniconda3
CONDA_PREFIX_2=/projects/viggiano/envs/bio-tools
CONDA_PREFIX_3=/home/viggiano/miniconda3
CONDA_PROMPT_MODIFIER=(bio-tools) 
CONDA_PYTHON_EXE=/home/viggiano/miniconda3/bin/python
CONDA_SHLVL=4
CUDA_VISIBLE_DEVICES=5,6,7
DISABLE_PANDERA_IMPORT_WARNING=True
HOME=/home/viggiano
LANG=en_US.UTF-8
LESSCLOSE=/usr/bin/lesspipe %s %s
LESSOPEN=| /usr/bin/lesspipe %s
LOGNAME=viggiano
LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.a...
MIG_PARTED_CHECKPOINT_FILE=/var/lib/nvidia-mig-manager/checkpoint.json
MIG_PARTED_CONFIG_FILE=/etc/nvidia-mig-manager/config.yaml
MIG_PARTED_HOOKS_FILE=/etc/nvidia-mig-manager/hooks.yaml
MOTD_SHOWN=pam
OLDPWD=/home/viggiano/main/codebases
PATH=/home/viggiano/.local/bin:/usr/local/cuda/bin:/opt/bin:/home/viggiano/.local/bin:/home/viggiano/.cargo/bin:/projects/viggiano/envs/bio-tools/bin:/home/viggiano/miniconda3/condabin:/usr/local/cuda/bin:...
PWD=/home/viggiano/main/codebases/bio-programming-tools
PYTEST_RUNNING=1
PYTEST_VERSION=9.0.2
RDBASE=/projects/viggiano/envs/bio-tools/lib/python3.12/site-packages/rdkit
SHELL=/bin/bash
SHLVL=2
TERM=tmux-256color
TERMINFO_DIRS=/home/viggiano/.terminfo:/home/viggiano/.terminfo:/home/viggiano/.terminfo:/home/viggiano/.terminfo:/home/viggiano/.terminfo:/home/viggiano/.terminfo:
TERM_PROGRAM=tmux
TERM_PROGRAM_VERSION=3.2a
TMUX=/tmp/tmux-1013/default,582927,0
TMUX_PANE=%0
USER=viggiano
XDG_DATA_DIRS=/usr/local/share:/usr/share:/var/lib/snapd/desktop
XDG_RUNTIME_DIR=/run/user/1013
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
_=/projects/viggiano/envs/bio-tools/bin/pytest
```

### Subprocess Environment (passed to tools)

```
CONDA_PREFIX=/raid/projects/viggiano/codebases/bio-programming-tools/tool_envs/viennarna_env
CUDA_VISIBLE_DEVICES=5,6,7
DETECTED_COMPUTE_PLATFORM=cuda
DETECTED_CUDA_VERSION=12
DETECTED_DRIVER_VERSION=535
HOME=/home/viggiano
LANG=en_US.UTF-8
LD_LIBRARY_PATH=/projects/viggiano/envs/bio-tools/lib
LOGNAME=viggiano
PATH=/raid/projects/viggiano/codebases/bio-programming-tools/tool_envs/viennarna_env/bin:/home/viggiano/.local/bin:/usr/local/cuda/bin:/opt/bin:/home/viggiano/.cargo/bin:/projects/viggiano/envs/bio-tools/b...
RECOMMENDED_JAX_SPEC=jax[cuda12]>=0.4.20,<1
RECOMMENDED_JAX_VARIANT=cuda12
RECOMMENDED_TORCH_SPEC=torch>=2.4,<2.7
SHELL=/bin/bash
TORCH_CUDA_ARCH_LIST=9.0
TORCH_HOME=/raid/projects/viggiano/codebases/bio-programming-tools/tool_envs/viennarna_env/cache/torch
USER=viggiano
VIRTUAL_ENV=/raid/projects/viggiano/codebases/bio-programming-tools/tool_envs/viennarna_env
XLA_PYTHON_CLIENT_ALLOCATOR=platform
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Results by Category

### Causal Models (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `evo1` | yes | ✅ | 355.3s | ✅ Pass |
| `evo2` | yes | ✅ | 255.3s | ✅ Pass |
| `evo2` | yes | ✅ | 428.4s | ✅ Pass |
| `progen2` | yes | ✅ | 78.2s | ✅ Pass |

### Gene Annotation (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `blast` | no | ✅ | 97.0s | ✅ Pass |
| `minced` | no | ✅ | 21.1s | ✅ Pass |
| `mmseqs` | no | ✅ | 35.1s | ✅ Pass |
| `pyhmmer` | no | ✅ | 25.5s | ✅ Pass |

### Inverse Folding (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `ligandmpnn` | yes | ✅ | 116.7s | ✅ Pass |
| `proteinmpnn` | yes | ✅ | 158.3s | ✅ Pass |

### Masked Models (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `esm2` | yes | ✅ | 51.0s | ✅ Pass |
| `esm3` | yes | ✅ | 88.5s | ✅ Pass |

### Orf Prediction (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `orfipy` | no | ✅ | 22.1s | ✅ Pass |
| `prodigal` | no | ✅ | 21.8s | ✅ Pass |

### Rna Splicing (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `splice_transformer` | yes | ✅ | 52.5s | ✅ Pass |

### Sequence Alignment (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `mafft` | no | ✅ | 28.4s | ✅ Pass |

### Sequence Scoring (2/2)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `borzoi` | yes | ✅ | 105.4s | ✅ Pass |
| `enformer` | yes | ✅ | 70.3s | ✅ Pass |

### Structure Alignment (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `tmalign` | no | ✅ | 50.9s | ✅ Pass |
| `tmalign` | no | ✅ | 0.0s | ✅ Pass |
| `usalign` | no | ✅ | 85.3s | ✅ Pass |
| `usalign` | no | ✅ | 0.1s | ✅ Pass |

### Structure Design (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `rfdiffusion3` | yes | ✅ | 96.8s | ✅ Pass |

### Structure Dynamics (1/1)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `bioemu` | yes | ✅ | 104.1s | ✅ Pass |

### Structure Prediction (13/14)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `alphafold2` | yes | ✅ | 174.0s | ✅ Pass |
| `boltz2` | yes | ✅ | 419.9s | ✅ Pass |
| `chai1` | yes | ✅ | 392.1s | ✅ Pass |
| `esmfold` | yes | ✅ | 74.0s | ✅ Pass |
| `protenix` | yes | ✅ | 773.6s | ✅ Pass |
| `protenix` | yes | ✅ | 471.6s | ✅ Pass |
| `protenix` | yes | ✅ | 504.2s | ✅ Pass |
| `protenix` | yes | ✅ | 2400.8s | ❌ Fail |
| `protenix` | yes | ✅ | 202.0s | ✅ Pass |
| `protenix` | yes | ✅ | 838.9s | ✅ Pass |
| `protenix` | yes | ✅ | 1294.2s | ✅ Pass |
| `protenix` | yes | ✅ | 182.9s | ✅ Pass |
| `protenix` | yes | ✅ | 408.6s | ✅ Pass |
| `viennarna` | no | ✅ | 14.1s | ✅ Pass |

### Unknown (4/4)

| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |
|------|--------------|----------------------|----------|--------|
| `alphagenome` | yes | ✅ | 343.3s | ✅ Pass |
| `crispr_tracr` | no | ✅ | 272.5s | ✅ Pass |
| `local_colabfold_search` | no | — | 150.5s | ✅ Pass |
| `structure_metrics` | no | ✅ | 33.1s | ✅ Pass |

## Failure Details

### ❌ `protenix`

**Test**: `tests/structure_prediction_tests/test_protenix.py::test_protenix_model_variants[protenix_base_constraint_v0.5.0]`

**Note**: Checkpoint download for this model variant is flakey and fails intermittently. The downloaded `.pt` file was corrupted (`PytorchStreamReader failed reading zip archive: failed finding central directory`), causing the worker to hang until the 2400s timeout. Deleting the cached checkpoint and re-running usually fixes it.

```
tests/structure_prediction_tests/test_protenix.py:123: in test_protenix_model_variants
    assert output.success
E   assert False
E    +  where False = <[ToolExecutionError('Attempt to access field of tool output after failure: TimeoutError: Worker for protenix timed out after 2400s\n\nError Messages:\nWorker for protenix timed out after 2400s\nTraceback (most recent call last):\n  File "/raid/projects/viggiano/codebases/bio-programming-tools/bio_programming_tools/tools/tool_registry.py", line 345, in wrapper\n    result = func(inputs, config, instance)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/raid/projects/viggiano/codebases/bio-programming-tools/bio_programming_tools/tools/structure_prediction/protenix/protenix.py", line 422, in run_protenix\n    output_data = ToolInstance.dispatch(\n                  ^^^^^^^^^^^^^^^^^^^^^^\n  File "/raid/projects/viggiano/codebases/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 241, in dispatch\n    return cached.run(\n           ^^^^^^^^^^^\n  File "/raid/projects/viggiano/codebases/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 549, in run\n    return self._run_persistent(\n           ^^^^^^^^^^^^^^^^^^^^^\n  File "/raid/projects/viggiano/codebases/bio-programming-tools/bio_programming_tools/utils/tool_instance.py", line 955, in _run_persistent\n    result = self._worker.send(input_dict, timeout=effective_timeout)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/raid/projects/viggiano/codebases/bio-programming-tools/bio_programming_tools/utils/persistent_worker.py", line 463, in send\n    raise TimeoutError(\nTimeoutError: Worker for protenix timed out after 2400s\n') raised in repr()] StructurePredictionOutput object at 0x7f9539d2b5c0>.success
```

---
*Generated at 2026-03-08 22:05:29 by `pytest --env-report`* (boltz2/chai1 re-verified 2026-03-08 after triton upgrade + SIGPIPE fix)