# fix-env: Detailed Failure Patterns

Reference file for the `fix-env` skill. Contains detailed failure patterns with full bash examples and root cause analysis.

---

## Infrastructure Failures

Failures in the centralized environment setup infrastructure — hardware detection, env variable isolation, sitecustomize.py injection. These are **root causes** that produce misleading downstream symptoms. Always check these first.

### 1. Compute Detection Failure

**Symptoms:** Wrong torch version installed (CPU-only on GPU machine), `DETECTED_COMPUTE_PLATFORM=cpu` when GPU is present.

**Root Cause:** `detect_compute_environment()` in `utils/compute_deps.py` failed to parse `nvidia-smi` output. The function is `@lru_cache(maxsize=1)` — a bad result persists for the process lifetime.

**Debugging:**
```bash
# Check if nvidia-smi works
nvidia-smi

# Check what the subprocess sees
python -c "
import subprocess, os
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, env=os.environ)
print('returncode:', result.returncode)
print('stdout:', result.stdout[:500])
"

# Check DETECTED_* vars from STATUS.txt
cat tool_envs/{tool}_env/STATUS.txt

# Test detection directly
python -c "
from bio_programming_tools.utils.compute_deps import detect_compute_environment
env = detect_compute_environment()
for k, v in sorted(env.items()): print(f'{k}={v}')
"
```

**Solution:** If nvidia-smi works but detection fails:
1. Check `nvidia-smi` is on the PATH that `_build_subprocess_env()` constructs
2. Verify the GPU query returns parseable CSV: `nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv,noheader,nounits` (this is the exact query used in `utils/system_info.py`; CUDA version is parsed separately from bare `nvidia-smi` output)
3. Verify driver major version maps correctly in compatibility matrices

See `utils/compute_deps.py` (detection logic), `tests/tool_infra_tests/test_compute_deps.py` (all hardware configs).

---

### 2. Environment Variable Isolation Failure

**Symptoms:** `uv pip install` installs into parent conda env, `ModuleNotFoundError` for installed packages, `LD_LIBRARY_PATH` missing entries, wrong Python version.

**Root Cause:** `_build_subprocess_env()` in `utils/persistent_worker.py` uses a whitelist approach. If CONDA_PREFIX or VIRTUAL_ENV don't point to the tool env path, uv/pip install into the wrong location.

**Debugging:**
```bash
# Check env_vars.txt
cat bio_programming_tools/tools/{category}/{tool}/standalone/env_vars.txt

# Inspect what the subprocess environment actually looks like
python -c "
from bio_programming_tools.utils.persistent_worker import _build_subprocess_env
from pathlib import Path
tool_env = Path('tool_envs/{tool}_env')
env = _build_subprocess_env(tool_env, tool_env / 'standalone')
print('CONDA_PREFIX:', env.get('CONDA_PREFIX'))
print('VIRTUAL_ENV:', env.get('VIRTUAL_ENV'))
print('LD_LIBRARY_PATH:', env.get('LD_LIBRARY_PATH', ''))
"

# Verify Python resolves to tool env (run inside subprocess)
# python -c "import sys; print(sys.prefix)"  # should print tool_env path
```

**Solution:**
- Wrong install target: verify CONDA_PREFIX and VIRTUAL_ENV point to tool env in `_build_subprocess_env()` step 7
- Missing LD_LIBRARY_PATH: add `LD_LIBRARY_PATH=${VENV_PATH}/cuda_env/lib` to `env_vars.txt` `[set]` section
- Leaked parent vars: check `_BASE_PASSTHROUGH` whitelist

**env_vars.txt format:**
```
[passthrough]
HF_TOKEN

[set]
LD_LIBRARY_PATH=${VENV_PATH}/cuda_env/lib
```

See `utils/persistent_worker.py` (`_build_subprocess_env`, steps 1-8), `evo1/standalone/env_vars.txt`, `alphagenome/standalone/env_vars.txt`.

---

### 3. sitecustomize.py Injection Failure

**Symptoms:** `OSError: libcudnn.so.9: cannot open shared object file` (evo2), `RuntimeError: Error building extension` / `gcc: command not found` (protenix).

**Root Cause:** evo2 generates `sitecustomize.py` to preload CUDA libs with `RTLD_GLOBAL`. protenix generates it to set `CC`/`CXX` to conda-forge GCC 12. If missing or malformed, tools fail at import time.

**Debugging:**
```bash
# Check if sitecustomize.py exists
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
cat "$SITE_PACKAGES/sitecustomize.py"

# For evo2 — verify CUDA libs exist
ls tool_envs/evo2_env/cuda_env/lib/libcudnn*.so*

# For protenix — verify GCC exists
ls tool_envs/protenix_env/cuda_env/bin/gcc
```

**Solution:** Re-run `setup.sh` (generates the file at the end). If lib paths are wrong, check that `cuda_env` was created correctly by micromamba. See `evo2/standalone/setup.sh` (ctypes.CDLL preloading), `protenix/standalone/setup.sh` (CC/CXX setup).

---

### 4. Micromamba Download/Install Failure

**Symptoms:** `Failed to download/extract micromamba`, `Could not resolve host: micro.mamba.pm`, `Connection timed out`, `Unsupported Linux architecture`.

**Root Cause:** `ToolInstance._ensure_micromamba()` downloads micromamba via `curl | tar` on first use. Fails before any tool-specific setup.

**Debugging:**
```bash
# Check if already cached
ls ~/.cache/bio_programming_tools/.micromamba/bin/micromamba

# Test download URL
ARCH=$(uname -m); SYSTEM=$(uname -s)
# Map: Linux x86_64→linux-64, Linux aarch64→linux-aarch64, Darwin arm64→osx-arm64
curl -Ls "https://micro.mamba.pm/api/micromamba/${PLATFORM}/latest" | tar -xvj bin/micromamba

# Check connectivity
curl -Is https://micro.mamba.pm | head -5
```

**Solution:**
- **Network failure:** Check connectivity. If behind proxy, set `https_proxy`. If blocked, manually place binary at `~/.cache/bio_programming_tools/.micromamba/bin/micromamba`.
- **Corrupt download:** Delete `~/.cache/bio_programming_tools/.micromamba/` and retry.
- **Unsupported platform:** Only Linux x86_64/aarch64 and Darwin x86_64/arm64 supported.

See `utils/tool_instance.py` (`_ensure_micromamba`).

---

## PyTorch / CUDA Failures

### 5. ABI Mismatch Errors (flash-attn, transformer-engine, torch)

**Symptoms:** `undefined symbol: _ZN3c105ErrorC2E...`, `ImportError: flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol`.

**Root Cause:** Cached wheels built against different PyTorch/CUDA/compiler versions.

**Standard Pattern (add to setup.sh):**
```bash
#!/bin/bash
set -euo pipefail

pip install uv

# Clear caches BEFORE installing ABI-sensitive packages
uv cache clean torch 2>/dev/null || true
uv cache clean flash-attn 2>/dev/null || true
uv cache clean transformer-engine 2>/dev/null || true

# Install with --refresh
uv pip install torch==X.Y.Z --extra-index-url "${RECOMMENDED_TORCH_INDEX}" --refresh
uv pip install --no-build-isolation flash-attn==A.B.C --refresh

# Validate deep imports
if ! python -c "import flash_attn_2_cuda" 2>/dev/null; then
    echo "ABI mismatch — rebuilding from source (30+ min)..."
    uv pip install --no-build-isolation --no-binary flash-attn --reinstall-package flash-attn flash-attn==A.B.C
fi
```

**Key Requirements:**
1. Clear caches early (after uv, before packages)
2. `--refresh` on all ABI-sensitive installs
3. Validate deep C++ extension imports, not Python wrappers
4. `2>/dev/null || true` for missing caches

See `evo1/standalone/setup.sh`, `evo2/standalone/setup.sh`.

---

### 6. Broken CUDA Library Symlinks

**Symptoms:** `libcudart.so: No such file or directory`, `libcublas.so.12: cannot open shared object file`.

**Root Cause:** micromamba installs versioned files (e.g., `libcudart.so.12.1.55`) but tools expect unversioned or major-versioned symlinks.

**Debugging:**
```bash
ls -la tool_envs/{tool}_env/cuda_env/lib/libcudart*
find tool_envs/{tool}_env/cuda_env/lib/ -xtype l  # broken symlinks
```

**Solution:** Auto-repair symlinks:
```bash
CUDA_LIB="$CUDA_HOME/lib"
if [ -d "$CUDA_LIB" ]; then
    for lib in "$CUDA_LIB"/lib*.so.*.*; do
        base=$(basename "$lib")
        major_link=$(echo "$base" | sed 's/\(\.so\.[0-9]*\).*/\1/')
        [ ! -e "$CUDA_LIB/$major_link" ] && ln -sf "$base" "$CUDA_LIB/$major_link"
        bare_link=$(echo "$base" | sed 's/\.so\..*/\.so/')
        [ ! -e "$CUDA_LIB/$bare_link" ] && ln -sf "$base" "$CUDA_LIB/$bare_link"
    done
fi
```

**See also:** Pattern 14 (Missing CUDA Headers) for a similar symlink issue with header files. See `evo1/standalone/setup.sh`, `evo2/standalone/setup.sh` for cuda_env setup.

---

### 7. Triton Version Coordination

**Symptoms:** `PY_SSIZE_T_CLEAN macro must be defined`, segfault on `import triton`, `triton/runtime/driver.py: ImportError`.

**Root Cause:** torch 2.6.0 ships bundled triton that conflicts with separately-installed versions. `PY_SSIZE_T_CLEAN` error = triton <3.2.0 with Python 3.12+.

**Solution:** Install triton AFTER all other packages — let torch's dependency resolver pick the right version:
```bash
uv pip install torch==2.6.0 --extra-index-url "${RECOMMENDED_TORCH_INDEX}" --refresh
uv pip install -r requirements.txt
# triton resolved by torch's dependency — don't pin separately
```

If already wrong: `uv pip install --reinstall-package triton triton==3.2.0`

See `evo2/standalone/setup.sh` (torch 2.6.0 + triton coordination).

---

## JAX Failures

### 8. JAX Setup / Version Downgrade

**Symptoms:** JAX uses CPU on GPU machine, `jaxlib version mismatch`, wrong CUDA plugin installed.

**Root Cause:** JAX needs `jax` + `jaxlib` + CUDA plugin. If a dependency reinstalls jax as a transitive dep, it can downgrade or drop the CUDA plugin.

**Solution:** Re-apply JAX spec AFTER all dependency installs:
```bash
JAX_VARIANT="${TOOL_JAX_VARIANT:-${RECOMMENDED_JAX_VARIANT:-cuda12}}"
JAX_SPEC="${TOOL_JAX_SPEC:-${RECOMMENDED_JAX_SPEC:-jax[cuda12]>=0.5,<1}}"

uv pip install "${JAX_SPEC}"
uv pip install -r requirements.txt
# RE-APPLY in case a dependency downgraded it
uv pip install "${JAX_SPEC}"

# Validate
if [ "${DETECTED_COMPUTE_PLATFORM:-cpu}" = "cuda" ]; then
    python -c "import jax; devs = jax.devices(); assert any('gpu' in str(d).lower() for d in devs), f'JAX GPU not found: {devs}'"
fi
```

See `alphagenome/standalone/setup.sh`.

---

## Compilation Failures

### 9. GCC/nvcc Version Mismatch

**Symptoms:** `_Float32 was not declared`, `_Float16 undeclared`, `nvcc fatal: unsupported compiler 'gcc-14'`.

**Root Cause:** CUDA JIT tools need GCC compatible with their CUDA toolkit. conda-forge sysroot 2.34+ adds typedefs that older nvcc can't parse.

**CUDA → max GCC mapping:**
| CUDA | Max GCC | Sysroot needed? |
|------|---------|-----------------|
| 12.1 | GCC ≤12 | Yes: `sysroot_linux-64=2.17` |
| 12.4 | GCC ≤13.2 | Depends on nvcc |
| 12.6+ | GCC ≤14 | No |

**Solution:** Match GCC to CUDA version in micromamba create:
```bash
# CUDA 12.1 (evo1, protenix):
micromamba create -p "$CUDA_HOME" -c conda-forge -c nvidia \
    "cuda-toolkit=12.1.*" "gcc=12.*" "gxx=12.*" "sysroot_linux-64=2.17" -y

# CUDA 12.8 (evo2):
micromamba create -p "$CUDA_HOME" -c conda-forge -c nvidia \
    "cuda-toolkit=12.8.*" "gcc=14.*" "gxx=14.*" -y
```

For runtime JIT tools (protenix), also set `CC`/`CXX` in `sitecustomize.py`. See `evo1/standalone/setup.sh` (GCC 12 + sysroot), `protenix/standalone/setup.sh` (GCC 12 + CC/CXX), `evo2/standalone/setup.sh` (GCC 14).

---

### 10. Platform Detection Issues

**Symptoms:** `No CUDA target directory found`, `not supported on aarch64`, `Tool X requires NVIDIA GPU but none detected`.

**Debugging:**
```bash
uname -m              # x86_64 or aarch64
nvidia-smi            # CUDA driver version
ls $CUDA_HOME/targets/ # Available CUDA targets
```

**Solutions:**
```bash
# Platform guard
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    echo "ERROR: Tool X is not supported on aarch64."
    exit 1
fi

# Robust target detection
CUDA_TARGET=$(ls "$CUDA_HOME/targets/" 2>/dev/null | head -1)
if [ -z "$CUDA_TARGET" ]; then
    echo "WARNING: No CUDA target directory found, attempting fallback..."
fi
```

**Platform notes:** DGX Spark (aarch64 + GPU) works for some tools (esm2, esmfold) but not others (evo2 — no aarch64 flash-attn wheels). See `notes/environments/` for per-machine reports.

---

### 12. Out of Memory (OOM) During Source Builds

**Symptoms:** `Killed signal terminated program cc1plus`, exit code -9.

**Root Cause:** Source builds of flash-attn/transformer-engine consume 20+ GB RAM.

**Solution:** Prefer wheels. If source build needed:
```bash
if ! python -c "import flash_attn_2_cuda" 2>/dev/null; then
    export MAX_JOBS=1
    uv pip install --no-build-isolation --no-binary flash-attn --reinstall-package flash-attn flash-attn==2.8.3
fi
```

---

## Binary / Network Failures

### 11. Network Failures with GitHub Release Wheels

**Symptoms:** `HTTP Error 404: Not Found`, `HTTP Error 502: Bad Gateway`.

**Root Cause:** Direct GitHub release URLs are unreliable (rate limiting, deleted releases).

**Solution:** Switch to PyPI:
```bash
# Bad:
pip install --force-reinstall https://github.com/.../flash_attn-2.8.0+....whl

# Good (in requirements.txt):
flash-attn==2.8.0.post2

# In setup.sh:
uv pip install -r requirements.txt --no-build-isolation-package flash-attn --refresh
```

---

### 13. Binary Installation Failure

**Symptoms:** `Failed to download blast after 3 attempts`, `No binary available for (Linux, x86_64)`, `No binary_config.py found`.

**Root Cause:** `utils/install_binary.py` downloads external binaries during `setup.sh`. Failures from network issues, platform not in `binary_config.py`, or truncated downloads.

**Debugging:**
```bash
# Check binary_config.py
cat bio_programming_tools/tools/{category}/{tool}/standalone/binary_config.py

# Check platform
python -c "import platform; print(platform.system(), platform.machine())"
# Note: install_binary.py normalizes aarch64→arm64, AMD64→x86_64

# Test URL manually
curl -L -o /dev/null -w '%{http_code}' "URL_FROM_BINARY_CONFIG"
```

**Solution:**
- Network: retry (3 retries with 5s delay built in)
- Platform missing: add `(system, machine)` tuple to `URLS` in `binary_config.py` — use `"arm64"` not `"aarch64"`
- Truncated: validates Content-Length — try a different mirror URL

See `utils/install_binary.py`, `blast/standalone/binary_config.py`, `mmseqs/standalone/binary_config.py`.

---

## Platform Failures

### 14. Missing CUDA Headers for JIT Compilation

**Symptoms:** `cuda_runtime.h: No such file or directory`, `Error building extension 'fused_dense'`.

**Root Cause:** PyTorch's `cpp_extension.load()` expects headers in `CUDA_HOME/include/`, but micromamba puts them in `CUDA_HOME/targets/{arch}/include/`.

**Solution:** Symlink headers if missing:
```bash
CUDA_TARGET=$(ls "$CUDA_HOME/targets/" 2>/dev/null | head -1)
CUDA_TARGETS_DIR="$CUDA_HOME/targets/${CUDA_TARGET}/include"
if [ -d "$CUDA_TARGETS_DIR" ]; then
    for item in "$CUDA_TARGETS_DIR"/*; do
        name=$(basename "$item")
        [ ! -e "$CUDA_HOME/include/$name" ] && ln -s "$item" "$CUDA_HOME/include/$name"
    done
fi
```

**See also:** Pattern 6 (Broken CUDA Library Symlinks) for a similar issue with runtime libraries.

---

### 15. Python Version Mismatch

**Symptoms:** `No matching distribution found for scikit-learn-extra==0.3.0 (Python 3.12)`, `Could not find a version that satisfies the requirement`.

**Root Cause:** Some packages lack wheels for newer Python. `standalone/python_version.txt` controls tool env Python version; if missing, defaults to 3.12.

**Debugging:**
```bash
cat bio_programming_tools/tools/{category}/{tool}/standalone/python_version.txt
tool_envs/{tool}_env/bin/python --version
```

**Solution:** Create/update `python_version.txt`:
```
3.11
```
Format: single line, `major.minor` or `major.minor.patch`, `>=3.8`. Delete tool env and rebuild after changing. See `protenix/standalone/python_version.txt`, `tests/tool_infra_tests/test_python_version_files.py`.

---

## Device Management Failures

Failures related to the DeviceManager infrastructure — specifically environment/setup failures, not runtime scheduling issues.

### 16. Standalone Helpers Import Failure

**Symptoms:** `ImportError: cannot import name 'get_subprocess_device_env'`, `ModuleNotFoundError: No module named 'standalone_helpers'`.

**Root Cause:** `_worker_bootstrap.py` copies `standalone_helpers.py` from `utils/` to each tool's `standalone/` at runtime. If copy fails (permissions, disk full, race condition), worker scripts crash.

**Debugging:**
```bash
# Check if file exists in tool's standalone dir
ls bio_programming_tools/tools/{category}/{tool}/standalone/standalone_helpers.py

# Check source exists
ls bio_programming_tools/utils/standalone_helpers_source/standalone_helpers.py

# Manual copy to test
cp bio_programming_tools/utils/standalone_helpers_source/standalone_helpers.py \
   bio_programming_tools/tools/{category}/{tool}/standalone/
```

**Solution:** Bootstrap copy failed — check `_worker_bootstrap.py` stderr. If source missing, reinstall with `pip install -e ".[dev]"`. Race conditions resolve on retry. See `utils/_worker_bootstrap.py` (`_copy_standalone_helpers`), `utils/standalone_helpers_source/`.

---

### 17. Device Configuration / CUDA Visibility Mismatch

**Symptoms:** `No available device for allocation`, tools hang waiting for GPU, wrong GPU used, device index out of range.

**Root Cause:** Confusion between `CUDA_VISIBLE_DEVICES` (physical GPU visibility) and `BIO_TOOLS_MANAGED_DEVICES` (logical device management). `CUDA_VISIBLE_DEVICES` remaps physical indices to zero-based logical indices.

**Example:**
```
Physical GPUs: 0, 1, 2, 3
CUDA_VISIBLE_DEVICES=2,3       → Logical: cuda:0 (phys 2), cuda:1 (phys 3)
BIO_TOOLS_MANAGED_DEVICES=cuda:2,cuda:3   → WRONG: logical 2,3 don't exist
BIO_TOOLS_MANAGED_DEVICES=cuda:0,cuda:1   → CORRECT
```

**Debugging:**
```bash
python -c "
from bio_programming_tools.utils.device import number_of_visible_gpus, number_of_physical_gpus
print(f'Physical: {number_of_physical_gpus()}, Visible: {number_of_visible_gpus()}')
"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "BIO_TOOLS_MANAGED_DEVICES=${BIO_TOOLS_MANAGED_DEVICES:-not set}"
```

**Solution:**
- Fix `BIO_TOOLS_MANAGED_DEVICES` to use logical indices (0-based after remapping), or unset for auto-detect
- SLURM sets `CUDA_VISIBLE_DEVICES` automatically — don't also set `BIO_TOOLS_MANAGED_DEVICES` unless restricting to a subset
- Empty `CUDA_VISIBLE_DEVICES` = no GPUs

See `utils/device_manager.py`, `utils/device.py`.
