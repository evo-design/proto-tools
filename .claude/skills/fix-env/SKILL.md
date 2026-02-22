# fix-env

**When to use:** Debugging and fixing tool environment setup failures on new systems or after system updates.

## Core Principle

**You will only be testing on the current machine.** Assume the existing setup works on other clusters. Your goal is to make surgical changes to `standalone/setup.sh` (and other standalone/ files, but **NEVER** `run.py` or `inference.py`) that fix the current machine while maintaining compatibility with other platforms.

## Strategy

1. **Identify the failure** on the current machine
2. **Make targeted fixes** to environment setup files only
3. **Use defensive patterns** that don't break existing platforms
4. **Test the fix** on the current machine only

**Files you can modify:**
- `standalone/setup.sh` ✓
- `standalone/requirements.txt` ✓
- `standalone/env_vars.txt` ✓
- `standalone/binary_config.py` ✓
- `standalone/python_version.txt` ✓

**Files you MUST NOT modify:**
- `standalone/run.py` ✗
- `standalone/inference.py` ✗
- `{tool_name}.py` (core implementation) ✗

## Common Failure Patterns

### 1. ABI Mismatch Errors (flash-attn, transformer-engine, torch)

**Symptoms:**
```
RuntimeError: Error compiling objects for extension
undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationESs
ImportError: flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol
```

**Root Cause:** Cached wheels built against different PyTorch/CUDA/compiler versions being reused across multiple users or after system updates.

**Solution:** Clear package manager caches before installation.

**Standard Pattern (add to setup.sh):**
```bash
#!/bin/bash
set -euo pipefail

echo "Installing uv package manager..."
pip install uv

# CRITICAL: Clear caches BEFORE installing any ABI-sensitive packages
echo "Clearing package caches for ABI-sensitive dependencies..."
uv cache clean torch 2>/dev/null || true
uv cache clean flash-attn 2>/dev/null || true
uv cache clean transformer-engine 2>/dev/null || true  # if used

# Install with --refresh flag as defense-in-depth
echo "Installing torch..."
uv pip install torch==X.Y.Z --torch-backend=auto --refresh

echo "Installing flash-attn..."
uv pip install --no-build-isolation flash-attn==A.B.C --refresh

# Validate the deepest import used by runtime code
if ! python -c "import flash_attn_2_cuda" 2>/dev/null; then
    echo "WARNING: flash-attn wheel has ABI mismatch (flash_attn_2_cuda import failed)"
    echo "Rebuilding from source... This can take 30+ minutes."
    uv pip install --no-build-isolation --no-binary flash-attn --reinstall-package flash-attn flash-attn==A.B.C
fi
```

**Why this is safe for other machines:**
- `uv cache clean` gracefully handles missing caches (`2>/dev/null || true`)
- `--refresh` flag just forces revalidation (doesn't break working setups)
- Validation only triggers source build if wheel actually fails to import

**Key Requirements:**
1. **Clear caches early**: Call `uv cache clean <package>` after installing uv, before installing packages
2. **Add --refresh flag**: Use `--refresh` on all ABI-sensitive installs (torch, flash-attn, transformer-engine)
3. **Validate deep imports**: Test the actual C++ extension import (e.g., `flash_attn_2_cuda`), not just the Python wrapper
4. **Graceful failure**: Use `2>/dev/null || true` to handle missing cache entries

**Reference Implementations:**
- `bio_programming_tools/tools/causal_models/evo1/standalone/setup.sh`
- `bio_programming_tools/tools/causal_models/evo2/standalone/setup.sh`
- `bio_programming_tools/tools/sequence_scoring/borzoi/standalone/setup.sh`

### 2. Network Failures with GitHub Release Wheels

**Symptoms:**
```
urllib.error.HTTPError: HTTP Error 404: Not Found
urllib.error.HTTPError: HTTP Error 502: Bad Gateway
```

**Root Cause:** Direct GitHub release URLs can be unreliable (rate limiting, temporary unavailability, deleted releases).

**Solution:** Prefer PyPI installation via requirements.txt over direct GitHub URLs.

**Bad (fragile):**
```bash
pip install --force-reinstall https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0/flash_attn-2.8.0+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

**Good (reliable):**
```bash
# In requirements.txt:
flash-attn==2.8.0.post2

# In setup.sh:
uv pip install -r requirements.txt --no-build-isolation-package flash-attn --refresh
```

**Why this is safe for other machines:**
- PyPI mirrors work everywhere
- `--no-build-isolation-package` ensures torch compatibility on all platforms
- Removes network-specific failure points

### 3. Out of Memory (OOM) During Source Builds

**Symptoms:**
```
c++: fatal error: Killed signal terminated program cc1plus
error: command '/usr/bin/gcc' failed with exit code -9
```

**Root Cause:** Source builds of flash-attn/transformer-engine can consume 20+ GB RAM during parallel compilation.

**Solutions:**
1. **Prefer pre-built wheels**: Use PyPI installation (see above) - wheels work on all machines
2. **Limit build parallelism**: Set `MAX_JOBS=1` before source builds (safe everywhere)
3. **Validate before source build**: Test wheel import first (see ABI pattern above)

**Example (safe for all machines):**
```bash
# Only build from source if wheel failed validation
if ! python -c "import flash_attn_2_cuda" 2>/dev/null; then
    export MAX_JOBS=1  # Prevents OOM, safe everywhere
    uv pip install --no-build-isolation --no-binary flash-attn --reinstall-package flash-attn flash-attn==2.8.3
fi
```

### 4. Platform Detection Issues

**Symptoms:**
```
ERROR: No CUDA target directory found in $CUDA_HOME/targets/
ERROR: Evo2 is not supported on aarch64
```

**Debugging:**
```bash
uname -m              # x86_64 or aarch64
nvidia-smi            # CUDA driver version
ls $CUDA_HOME/targets/  # Available CUDA targets (if CUDA_HOME set)
```

**Solutions:**
- **Add platform guards** (doesn't affect other machines):
```bash
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    echo "ERROR: Tool X is not supported on aarch64."
    exit 1
fi
```

- **Make target detection robust** (graceful fallback):
```bash
CUDA_TARGET=$(ls "$CUDA_HOME/targets/" 2>/dev/null | head -1)
if [ -z "$CUDA_TARGET" ]; then
    echo "WARNING: No CUDA target directory found, attempting fallback..."
    # Try platform-specific defaults
fi
```

### 5. Missing CUDA Headers for JIT Compilation

**Symptoms:**
```
fatal error: cuda_runtime.h: No such file or directory
RuntimeError: Error building extension 'fused_dense'
```

**Root Cause:** PyTorch's cpp_extension.load() expects CUDA headers in `CUDA_HOME/include/`, but micromamba installs them in `CUDA_HOME/targets/{arch}/include/`.

**Solution (safe for all machines):** Symlink headers if missing:
```bash
CUDA_TARGET=$(ls "$CUDA_HOME/targets/" 2>/dev/null | head -1)
CUDA_TARGETS_DIR="$CUDA_HOME/targets/${CUDA_TARGET}/include"
if [ -d "$CUDA_TARGETS_DIR" ]; then
    for item in "$CUDA_TARGETS_DIR"/*; do
        name=$(basename "$item")
        # Only create symlink if missing (idempotent, safe)
        if [ ! -e "$CUDA_HOME/include/$name" ]; then
            ln -s "$item" "$CUDA_HOME/include/$name"
        fi
    done
fi
```

## Debugging Workflow

### 1. Identify the Failure

```bash
# Check environment setup logs for the current machine
cat bio_programming_tools/tool_envs/{tool}_env/STATUS.txt

# Look for specific error patterns:
# - "undefined symbol" → ABI mismatch
# - "HTTP Error" → network failure
# - "Killed signal" → OOM
# - "No such file" → missing headers/deps
```

### 2. Test Specific Component

```bash
# Delete environment to force rebuild
rm -rf bio_programming_tools/tool_envs/{tool}_env

# Test just the failing tool
pytest -k "tool_name" --all -sv
```

### 3. Apply Fixes to setup.sh

Based on error pattern, apply appropriate solution to `standalone/setup.sh`:

- **ABI issues** → Add cache clearing + --refresh flags (safe everywhere)
- **Network failures** → Switch to PyPI from GitHub URLs (more reliable)
- **OOM** → Prefer wheels, add MAX_JOBS=1 to source build fallback (safe everywhere)
- **Platform issues** → Add graceful platform detection with fallbacks
- **Missing headers** → Add conditional symlink logic (idempotent)

**Key principle:** Use defensive patterns (`|| true`, conditional checks, graceful fallbacks) that fix the current machine without breaking others.

### 4. Validate Fix on Current Machine

```bash
# Delete environment again
rm -rf bio_programming_tools/tool_envs/{tool}_env

# Run affected tests
pytest -k "tool_name" --all -sv

# If that passes, run broader suite to check for regressions
pytest --cpu --skip-ci
pytest --gpu --all  # if GPU available
```

### 5. Document What You Changed

When updating setup.sh, add comments explaining:
- **What** the change does
- **Why** it was needed (which error it fixes)
- **Why it's safe** for other platforms

**Example:**
```bash
# Clear package caches to prevent ABI mismatches from cached wheels built
# against different PyTorch/CUDA versions. Safe for all platforms (gracefully
# handles missing caches). Fixes: undefined symbol errors on multi-user systems.
uv cache clean torch 2>/dev/null || true
uv cache clean flash-attn 2>/dev/null || true
```

## Cross-Platform Safety Checklist

When modifying setup.sh, ensure your changes:

- [ ] Use `|| true` for operations that might fail on some platforms
- [ ] Use conditional checks (`if [ -d ... ]`) before filesystem operations
- [ ] Use `2>/dev/null` to suppress expected errors
- [ ] Don't hardcode platform-specific paths (use detection)
- [ ] Add comments explaining why the change is safe for other platforms
- [ ] Only modify standalone/ files (never run.py/inference.py)
- [ ] Test on current machine after deleting tool_env

## Reference Documentation

- **Cache management**: See "Cache Management for ABI-Sensitive Packages" in CLAUDE.md
- **Compute deps**: See "Compute Dependency Management" in CLAUDE.md for hardware detection
- **Python versions**: See "Python Version Specification" in CLAUDE.md for python_version.txt
- **Binary installation**: See "Binary Installation" in CLAUDE.md for install_binary.py usage
