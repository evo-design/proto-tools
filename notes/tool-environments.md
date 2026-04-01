# Tool Environments Reference

Detailed reference for standalone tool environment setup, compute dependency management, and environment isolation. For model weight storage and `PROTO_HOME` configuration, see [model-weights.md](model-weights.md).

## Compute Dependency Management

Tools with PyTorch/JAX dependencies use **centralized hardware detection** (`utils/compute_deps.py`) to automatically select compatible package versions based on the host's NVIDIA driver and CUDA versions.

**How it works:**

1. `persistent_worker.py` calls `detect_compute_environment()` when building the subprocess environment
2. Detection inspects `nvidia-smi` output to extract driver and CUDA versions
3. Compatibility matrices map driver major versions to package version constraints
4. Environment variables are injected into the subprocess before `setup.sh` runs
5. Tool setup scripts consume these variables to install the right package versions

**Environment variables injected:**

| Variable | Example Value | Description |
|---|---|---|
| `DETECTED_COMPUTE_PLATFORM` | `"nvidia-gpu"` or `"cpu"` | Hardware platform detected |
| `DETECTED_DRIVER_VERSION` | `"570.122.3"` | NVIDIA driver version |
| `DETECTED_CUDA_VERSION` | `"12.6"` | CUDA version |
| `RECOMMENDED_TORCH_SPEC` | `"torch>=2.10,<3"` | PyTorch version constraint for detected driver |
| `RECOMMENDED_JAX_SPEC` | `"jax[cuda12]>=0.5,<1"` | JAX version constraint with CUDA plugin |
| `RECOMMENDED_JAX_VARIANT` | `"cuda12"` | JAX CUDA variant (cuda12, cuda13) |

### Standard PyTorch Setup Pattern

See `esm2`, `esmfold`, `boltz2` for reference implementations:

```bash
#!/bin/bash
set -euo pipefail

echo "Installing uv package manager..."
pip install uv

# Install hardware-aware PyTorch version (from centralized detection)
echo "Installing PyTorch: ${RECOMMENDED_TORCH_SPEC:-torch} (platform: ${DETECTED_COMPUTE_PLATFORM:-unknown})"
uv pip install "${RECOMMENDED_TORCH_SPEC:-torch}" --extra-index-url "${RECOMMENDED_TORCH_INDEX}"

echo "Installing remaining dependencies..."
uv pip install -r requirements.txt
```

### Standard JAX Setup Pattern

See `alphagenome` for reference:

```bash
JAX_VARIANT="${TOOL_JAX_VARIANT:-${RECOMMENDED_JAX_VARIANT:-cuda12}}"
JAX_SPEC="${TOOL_JAX_SPEC:-${RECOMMENDED_JAX_SPEC:-jax[cuda12]>=0.5,<1}}"

echo "Detected platform: ${DETECTED_COMPUTE_PLATFORM:-unknown}"
echo "Installing JAX: ${JAX_SPEC}"
uv pip install "${JAX_SPEC}"
```

### Tool-Specific Overrides

Tools can override centralized recommendations via env_vars.txt or tool-specific environment variables (e.g., `SPLICE_TRANSFORMER_TORCH_SPEC`, `ALPHAGENOME_JAX_SPEC`).

### Pinned-Version Tools

Some tools have hard version pins for ABI compatibility with pre-built wheels (flash-attn, transformer-engine). These tools explicitly pin torch versions in their setup.sh and should NOT be migrated to use dynamic version selection:
- `evo1`: `torch==2.7.1` (flash-attn ABI compatibility)
- `evo2`: `torch==2.6.0` (flash-attn + transformer-engine compatibility)
- `borzoi`: `torch==2.7.1` (flash-attn wheel compatibility)

### Compatibility Matrices

Based on official sources (PyTorch RELEASE.md, JAX docs, NVIDIA CUDA compatibility):

**PyTorch** (driver → torch version):
- Driver 570+: torch 2.8+ (CUDA 12.8 native support)
- Driver 550-569: torch 2.5+ (CUDA 12.4 native support)
- Driver 535-549: torch 2.4-2.6.x (CUDA 12.2; 2.7+ ships CUDA 12.8 runtime libs)
- Driver 525-534: torch 2.4-2.6.x (CUDA 12.0-12.1)
- Driver &lt;525: torch 2.1-2.3 (CUDA 11.x era)

**JAX** (driver + CUDA → jax version + variant):
- Driver 525+: jax[cuda12] 0.4.20+ (all CUDA 12.x)
- Driver 580+: jax[cuda13] 0.4.20+ (CUDA 13.x, not yet used)
- Driver &lt;525: jax[cuda11] 0.4.20+ (CUDA 11.x)

See `tests/tool_infra_tests/test_compute_deps.py` for comprehensive test coverage.

## env_vars.txt

Each tool's `standalone/env_vars.txt` supports two sections:
- `[passthrough]`: Variable names copied from the parent environment (e.g., `HF_TOKEN`)
- `[set]`: Literal `KEY=VALUE` assignments, with `${VENV_PATH}` interpolation

**Auto-set environment variables** (always injected by `_build_subprocess_env()`):
- `CONDA_PREFIX`: set to the **tool env path** (not the parent conda env) so uv/pip install into the correct environment
- `VIRTUAL_ENV`: set to the **tool env path** for uv >=0.10 compatibility
- `PATH`: `tool_env/bin` > `cuda/bin` (GPU) > parent PATH entries > system dirs
- `LD_LIBRARY_PATH`: tool-specific `[set]` paths > parent `LD_LIBRARY_PATH` entries > `$CONDA_PREFIX/lib`

## GCC/nvcc Compatibility for CUDA JIT Tools

Tools that JIT-compile CUDA C++ extensions install a compatible GCC from conda-forge. The GCC version is chosen per-tool based on the CUDA toolkit version.

**CUDA → max GCC mapping:**
- CUDA 12.1: GCC ≤12
- CUDA 12.4: GCC ≤13.2
- CUDA 12.6: GCC ≤14
- CUDA 12.8: GCC ≤14

**Per-tool versions:**
- **evo1** (CUDA 12.1): `"gcc=12.*" "gxx=12.*" "sysroot_linux-64=2.17"`
- **protenix** (CUDA 12.1): `"gcc=12.*" "gxx=12.*" "sysroot_linux-64=2.17"`
- **evo2** (latest CUDA ~12.8): `"gcc=14.*" "gxx=14.*"`

**Why sysroot 2.17 for GCC 12 tools:** conda-forge GCC packages pull in the latest sysroot by default. Sysroot 2.34+ adds `_Float32`/`_Float16` typedefs in `<stdlib.h>` that nvcc 12.1's EDG parser rejects. Pinning to glibc 2.17 avoids this.

**Pattern:**
1. Add `gcc`/`gxx` (+ `sysroot_linux-64` if needed) to the micromamba create command
2. For runtime JIT tools (protenix): also set `CC`/`CXX` in `sitecustomize.py`

## Cache Management for ABI-Sensitive Packages

Tools that install C++ extensions with ABI dependencies (torch, flash-attn, transformer-engine) must clear package manager caches to prevent compatibility issues. Cached wheels may be built against different PyTorch/CUDA/compiler versions, causing runtime failures with symbols like `undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationESs`.

**Standard pattern for ABI-sensitive tools** (evo1, evo2, borzoi):

```bash
echo "Installing uv package manager..."
pip install uv

# Clear caches BEFORE installing any ABI-sensitive packages
echo "Clearing package caches for ABI-sensitive dependencies..."
uv cache clean torch 2>/dev/null || true
uv cache clean flash-attn 2>/dev/null || true
uv cache clean transformer-engine 2>/dev/null || true  # if used

# Install with --refresh flag as defense-in-depth
echo "Installing torch..."
uv pip install torch==X.Y.Z --extra-index-url "${RECOMMENDED_TORCH_INDEX}" --refresh

echo "Installing flash-attn..."
uv pip install --no-build-isolation flash-attn==A.B.C --refresh

# Validate the deepest import used by runtime code
if ! python -c "import flash_attn_2_cuda" 2>/dev/null; then
    echo "flash-attn wheel has ABI mismatch, rebuilding from source..."
    uv pip install --no-build-isolation --no-binary flash-attn --reinstall-package flash-attn flash-attn==A.B.C
fi
```

**Key requirements:**
1. Clear caches early (`uv cache clean <package>`)
2. Use `--refresh` on all ABI-sensitive installs
3. Validate deep imports (e.g., `flash_attn_2_cuda`), not just Python wrappers
4. Use `2>/dev/null || true` for graceful failure

**For direct URL installs** (e.g., GitHub release wheels), use pip's `--force-reinstall`.

**Reference implementations:**
- `proto_tools/tools/causal_models/evo1/standalone/setup.sh`
- `proto_tools/tools/causal_models/evo2/standalone/setup.sh`
- `proto_tools/tools/sequence_scoring/borzoi/standalone/setup.sh`

## Python Version Specification

Tools can specify their required Python version via `standalone/python_version.txt`.

**Format:** Single line, `major.minor` or `major.minor.patch` (e.g., `3.11`)

**Validation:** Version ≥3.8, numeric components only, no comments/prefixes. Missing file defaults to Python 3.12.

**Rebuilds:** Content is included in the environment setup hash; changing version triggers rebuild.

See `tests/tool_infra_tests/test_python_version_files.py` for validation tests.

## Binary Installation

Tools needing external binaries must use `utils/install_binary.py`; never raw `curl`/`wget` in `setup.sh`.

1. Create `standalone/binary_config.py` with:
   - `URLS`: dict mapping `(system, machine)` tuples to download URLs (use `"arm64"` not `"aarch64"`)
   - `extract(archive_path: Path, bin_dir: Path)`: extracts/copies binaries into bin/
2. In `setup.sh`, call: `python "$SEARCH_DIR/utils/install_binary.py" <tool_name>`

See blast or mmseqs for the standard pattern. For platform-independent tools (e.g., Java JARs), use the same URL for all platform keys.

## Compile-from-Source Tools

Tools distributed as C/C++ source compile during `setup.sh`. No `binary_config.py` or `requirements.txt` needed; just check for the compiler (`g++`), clone the source, compile into the venv's `bin/`, and clean up. Use `BUILD_DIR` (not `TMPDIR`) for the temporary clone directory. See TMalign/USalign (`tools/structure_alignment/`) as canonical examples.

## Standalone Helpers for CLI Subprocess Device Routing

For tools that spawn CLI subprocesses (Boltz2, RFDiffusion3, Protenix, AlphaFold3), `get_subprocess_device_env()` ensures correct device routing when the parent process has `CUDA_VISIBLE_DEVICES` set.

**The problem:** When DeviceManager allocates a logical device (e.g., `cuda:2`), CLI subprocesses need the physical GPU index mapped from the logical index.

**The solution:** `utils/standalone_helpers_source/standalone_helpers.py` provides `get_subprocess_device_env(device: str) -> Dict[str, str]`:

```python
from standalone_helpers import get_subprocess_device_env

env = get_subprocess_device_env("cuda:2")  # Maps to physical GPU
subprocess.run(cmd, env=env)
```

**Auto-copy mechanism:**
- Source: `proto_tools/utils/standalone_helpers_source/standalone_helpers.py` (tracked in git)
- Destination: `{tool}/standalone/standalone_helpers.py` (not tracked, auto-generated at runtime by `_worker_bootstrap.py`)
- Exception: AlphaFold3's `standalone_helpers.py` is manually copied and tracked

**Consistency enforcement:** The test `tests/tool_infra_tests/test_device_manager.py::test_gpu_cli_tools_use_subprocess_device_helper` verifies all GPU tools with subprocess calls properly use `get_subprocess_device_env()`.

## The `to_device()` Protocol

All standalone scripts (`standalone/inference.py` or `standalone/run.py`) implement a `to_device(device: str) -> dict` function that DeviceManager calls to move models between devices. This function is invoked via the worker bootstrap protocol when `ToolInstance.to(device)` is called.

**PyTorch tools** (ESMFold, Evo2, ESM2, ProteinMPNN, etc.):
```python
def to_device(device: str) -> dict:
    global _model
    if _model is not None and _model._loaded:
        _model.to_device(device)
        return {"success": True, "device": device}
    else:
        return {"success": True, "device": device, "note": "model not loaded yet"}
```

**JAX tools** (AlphaGenome):
```python
def to_device(device: str) -> dict:
    global _model
    if _model is not None and hasattr(_model, "to_device"):
        _model.to_device(device)
    return {"success": True, "device": device}
```

**CLI tools** (Boltz2, RFDiffusion3, BLAST, etc.):
```python
def to_device(device: str) -> dict:
    return {"success": True, "device": device, "note": "CLI tool, auto-unloads"}
```

When implementing new tools, add `to_device()` to `standalone/inference.py` or `standalone/run.py` following the pattern above. The model wrapper class should have a `to_device(device: str)` method for actual device moves.
