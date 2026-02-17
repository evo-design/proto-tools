# Evo2 Standalone Environment: Architecture & Fixes

## How the Standalone Env Works

### The Problem It Solves

Evo2 requires a very specific set of dependencies (torch 2.6.0, flash-attn, transformer-engine, vortex, CUDA toolkit + cuDNN) that would conflict with the user's base environment. The standalone env isolates all of this into a self-contained venv so the user only needs `pip install bio_programming_tools` in their base env.

### Architecture Overview

```
User's base env (conda or venv)
  │
  │  from bio_programming_tools.tools.causal_models.evo2 import run_evo2_sample
  │  run_evo2_sample(inputs, config)
  │
  ▼
EnvManager("evo2")                          # bio_programming_tools/utils/env_manager.py
  │
  ├─ _get_venvs_root()                     # Finds project root via pyproject.toml marker
  │   └─ .venvs/evo2_env/                  # The isolated venv lives here
  │
  ├─ _create_env()  [if first run]
  │   ├─ python -m venv --copies           # Creates venv with COPIED Python binary
  │   │                                    # (--copies makes it independent of base env)
  │   ├─ _get_clean_subprocess_env()       # Strips LD_LIBRARY_PATH, CONDA_*, Jupyter vars
  │   └─ bash setup.sh                     # Installs everything inside the venv
  │
  └─ call_standalone_script_in_venv()
      ├─ Serializes inputs to JSON          # input.json in temp dir
      ├─ Calls: evo2_env/bin/python inference.py input.json output.json
      │   └─ This runs in a SUBPROCESS with clean env vars
      └─ Reads output.json back             # Deserializes results
```

### What setup.sh Does (in order)

```
1. Install uv (fast pip alternative)
2. Install micromamba (conda-like package manager, installed INTO the venv)
3. micromamba install CUDA toolkit + cuDNN → .venvs/evo2_env/cuda_env/
4. Create header symlinks (nvtx3, cuda/thrust/cub headers)
5. Fix broken libcudart.so symlink
6. Set compilation env vars (CUDA_HOME, CPATH, LDFLAGS, etc.)
7. uv pip install torch==2.6.0
8. uv pip install build deps (ninja, numpy, setuptools, etc.)
9. uv pip install flash-attn==2.8.3 (--no-build-isolation, builds from source)
10. uv pip install transformer_engine[pytorch]==2.5.0 (--no-build-isolation, builds from source)
11. uv pip install vtx (vortex)
12. uv pip install -r requirements.txt (evo2 + tqdm)
13. uv pip install --upgrade triton (override torch's pinned 3.2.0)
14. Generate sitecustomize.py (CUDA lib preloader)
```

### Why sitecustomize.py Exists

When EnvManager runs inference.py, it **strips LD_LIBRARY_PATH** from the subprocess environment. This prevents the user's system CUDA from interfering with the venv's bundled CUDA.

But transformer-engine was compiled against the venv's local CUDA (from micromamba), not torch's bundled CUDA. Without LD_LIBRARY_PATH, transformer-engine can't find the right CUDA libs at runtime.

The fix: `sitecustomize.py` runs automatically when Python starts (before any imports). It uses `ctypes.CDLL(path, mode=RTLD_GLOBAL)` to pre-load the correct CUDA libs into the process's global symbol table. This way, when transformer-engine later tries to load CUDA symbols, they're already available.

```
Python process starts
  → sitecustomize.py runs (before any imports)
    → Pre-loads libcudnn, libcublas, libcusparse, libnvrtc from cuda_env/lib/
  → import torch (loads torch's bundled CUDA 12.6 - but critical libs already loaded)
  → import transformer_engine (uses pre-loaded CUDA 12.9 libs - matches what it was compiled against)
```

### How the JSON Subprocess Protocol Works

```python
# In the user's base env (evo2_sample.py):
venv_manager = EnvManager("evo2")
result = venv_manager.call_standalone_script_in_venv(
    script_path="standalone/inference.py",
    input_dict={"operation": "sample", "prompts": ["ATGCGTAAA"], ...},
    device="cuda",
)

# This becomes:
# 1. Write input_dict → /tmp/xxx/input.json
# 2. subprocess.run([".venvs/evo2_env/bin/python", "inference.py", "input.json", "output.json"])
# 3. Read /tmp/xxx/output.json → result dict
```

The subprocess uses the **venv's Python** (not the user's), so it has access to torch, evo2, etc. The user's base env never needs to install any of these.

---

## What Was Broken (Conda Base Env)

When a user did `conda create -n bio_tools python=3.12 && pip install .`, three things broke:

### Issue 1: transformer_engine 2.3.0 Build Failure

**Symptom**: `ModuleNotFoundError: No module named 'build_tools'` during setup.sh

**Root cause**: TE 2.3.0's `setup.py` does `from build_tools import ...` where `build_tools/` is a directory in the source distribution (sdist). When setuptools builds the package, it `exec()`s setup.py as a string. The standard `setuptools.build_meta` backend does NOT add the source directory to `sys.path`, so the import fails.

**Why it seemed conda-specific**: The Python 3.13 venv approach had a **cached prebuilt wheel** for TE-torch in uv's cache (`~/.cache/uv/sdists-v9/`) from a previous session. When using Python 3.12 (conda), there was no compatible cached wheel (wheels are Python-version-specific: cp313 vs cp312), so uv had to build from source, hitting the bug.

**Fix**: Upgraded to `transformer_engine[pytorch]==2.5.0`. TE 2.5.0 adds a `pyproject.toml` with:
```toml
build-backend = "setuptools.build_meta:__legacy__"
```
The `__legacy__` backend adds `sys.path.insert(0, source_dir)` before exec'ing setup.py, so `from build_tools import ...` works. The TE developers even added a comment: *"Use legacy backend to import local packages in setup.py"*.

### Issue 2: Triton 3.2.0 Runtime Bug

**Symptom**: `SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#' formats` when running inference

**Root cause**: `torch==2.6.0` pins `triton==3.2.0`. Triton 3.2.0 has a C extension (`load_binary`) that doesn't define the `PY_SSIZE_T_CLEAN` macro, which is required by Python 3.12's C API. Conda-forge's Python 3.12.0 enforces this strictly, while some other Python builds are more lenient.

**Fix**: Added `uv pip install --upgrade triton` at the **end** of setup.sh (after all other installs). It must be last because `uv pip install -r requirements.txt` would otherwise reinstall triton 3.2.0 via torch's dependency pin. The upgrade to triton >=3.6.0 fixes the C API issue.

### Issue 3: .venvs Created Inside site-packages

**Symptom**: `.venvs/evo2_env` created at `miniconda/envs/bio_tools/lib/python3.12/site-packages/.venvs/` instead of the project root

**Root cause**: `EnvManager.__init__` used `Path(__file__).parent.parent.parent` to find the project root. This assumes `env_manager.py` is at `project_root/bio_programming_tools/utils/env_manager.py`. For editable installs (`pip install -e .`), `__file__` points to the source tree and this works. For non-editable installs (`pip install .`), `__file__` is inside site-packages, so the calculation resolves to `site-packages/` — putting multi-GB venvs inside the conda env's site-packages directory.

**Fix**: Replaced with `_get_venvs_root()` which walks up from `__file__` looking for `pyproject.toml` as a project root marker. If found (editable install or running from source), uses `project_root/.venvs/`. If not found (non-editable install from arbitrary location), falls back to `~/.cache/bio_programming_tools/.venvs/`.

### Issue 4: uv sdist Cache Corruption by TE's setup.py

**Symptom**: `ModuleNotFoundError: No module named 'build_tools'` during setup.sh, even with TE 2.5.0

**Root cause**: TE's `setup.py` has a cleanup step that runs `shutil.rmtree("build_tools")` after a successful build. When uv builds TE from source, it caches the unpacked sdist at `~/.cache/uv/sdists-v9/pypi/transformer-engine-torch/2.5.0/`. After the first successful build (e.g., for Python 3.12/cp312), `build_tools/` is deleted from this cached source directory. If a subsequent build is triggered for a different Python version (e.g., Python 3.13/cp313), uv reuses the now-corrupted cached sdist (missing `build_tools/`), and the build fails.

**Why it's intermittent**: Only manifests when the same machine builds TE for multiple Python versions. The first build succeeds; the second fails. Clearing `~/.cache/uv/` between builds masks the issue.

**Fix**: Added `uv cache clean transformer-engine-torch` before the TE install in setup.sh. This ensures each build starts from a fresh sdist extraction.

### Bonus: Conda Env Var Leakage

**What was happening**: When the user's base env is conda, variables like `CONDA_PREFIX`, `CONDA_EXE`, `CONDA_PYTHON_EXE` etc. leak into the subprocess environment. While these didn't cause the specific failures above, they could confuse pip/uv about which Python environment to target.

**Fix**: Added `CONDA_BLOCKLIST` to `_get_clean_subprocess_env()` that strips all `CONDA_*` variables from the subprocess environment, just like it already strips `LD_LIBRARY_PATH` and Jupyter variables.

---

## Key Files

| File | Role |
|------|------|
| `utils/env_manager.py` | Creates venvs, runs setup.sh, calls inference subprocess with clean env |
| `tools/causal_models/evo2/standalone/setup.sh` | Installs everything in the venv (CUDA, torch, TE, flash-attn, evo2) |
| `tools/causal_models/evo2/standalone/inference.py` | Runs inside the venv subprocess; loads model, runs sample/score, writes JSON |
| `tools/causal_models/evo2/evo2_sample.py` | Public API; routes to EnvManager (local GPU) or Modal (remote GPU) |
| `.venvs/evo2_env/` | The isolated venv with all evo2 deps |
| `.venvs/evo2_env/cuda_env/` | Micromamba-managed CUDA toolkit + cuDNN |
| `.venvs/evo2_env/lib/.../sitecustomize.py` | Auto-generated; pre-loads CUDA libs at Python startup |

## Testing

Both approaches now work identically:

```bash
# Venv approach
python -m venv .venv && source .venv/bin/activate && pip install .
python -c "from bio_programming_tools.tools.causal_models.evo2 import run_evo2_sample, Evo2SampleInput, Evo2SampleConfig; ..."

# Conda approach
conda create -n bio_tools python=3.12 -y && conda activate bio_tools && pip install .
python -c "from bio_programming_tools.tools.causal_models.evo2 import run_evo2_sample, Evo2SampleInput, Evo2SampleConfig; ..."
```

First run takes ~10 minutes (downloads CUDA toolkit, builds flash-attn and transformer-engine from source). Subsequent runs reuse the cached venv and take ~10 seconds for model loading + inference.
