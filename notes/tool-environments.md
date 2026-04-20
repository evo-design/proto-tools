# Tool Environments Reference

Detailed reference for standalone tool environment setup, compute dependency management, and environment isolation. For model weight storage and `PROTO_HOME` configuration, see [storage.md](storage.md).

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

## Debugging Env Setup (`PROTO_ENV_VERBOSE`, `PROTO_ENV_LOG_DIR`)

When `ToolInstance._create_env()` runs `standalone/setup.sh` to build a tool's venv, the subprocess output is normally captured quietly and only surfaced on failure (via `STATUS.txt` and the raised `RuntimeError`). Two env vars opt into richer visibility:

- `PROTO_ENV_VERBOSE=1` — streams each line of `setup.sh`'s output live to the caller's stderr as the subprocess runs. Useful for watching long installs (PyTorch, flash-attn, transformer-engine, etc.) in real time and for diagnosing hangs.
- `PROTO_ENV_LOG_DIR=<path>` — after `setup.sh` exits (success or failure), copies the complete log to `<PROTO_ENV_LOG_DIR>/<tool_name>_setup.log`. Useful when the env directory itself is ephemeral and you want the log to survive a rollback.

Regardless of either flag, the combined output is always written to `<env_path>/setup.log` during setup, so you can inspect it after the fact from any env that still has its files on disk.

Both variables default to off — setup output stays quiet unless a caller opts in. See `tests/tool_infra_tests/test_tool_instance.py::test_run_setup_script_*` for the behavior contract.

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

Every tool with a `standalone/` directory must ship a `standalone/python_version.txt` that pins its Python version. The consistency tests fail if any tool is missing the file, and `ToolInstance._get_python_version` raises `FileNotFoundError` on setup for a tool whose file is missing.

**Format:** keyed lines, with a required `default` and optional per-platform overrides. Comments (`#` to end of line) and blank lines are ignored. Whitespace around `:` is stripped; keys are lowercased.

```text
# Comments and blank lines are allowed.
default: 3.11
linux: 3.10            # OS-only fallback (any Linux)
linux-aarch64: 3.10    # specific arch override (most specific)
darwin-arm64: 3.10
```

**Lookup (most specific wins):**

1. Exact `{system}-{machine}` key — e.g. `linux-aarch64`
2. OS-only `{system}` key — e.g. `linux`
3. `default` (required catch-all)

The lookup key is built as `f"{platform.system().lower()}-{platform.machine()}"`:

| Platform | Specific key | OS key |
|---|---|---|
| Linux x86_64 | `linux-x86_64` | `linux` |
| Linux ARM | `linux-aarch64` | `linux` |
| macOS Intel | `darwin-x86_64` | `darwin` |
| macOS Apple Silicon | `darwin-arm64` | `darwin` |

**Validation:** every value must be `major.minor[.patch]` with `major == 3` and `minor >= 8`. All values are validated up front, so a typo in any override fails on any developer's machine — not just the affected platform.

**When to use overrides:** declare a per-platform override only when a tool's upstream dependency is unavailable for the default Python on that platform (e.g., PyRosetta on `linux-aarch64` only ships py39/py310 builds, so it pins `linux-aarch64: 3.10`). Use the OS-only tier when an entire OS family needs a different version. The reference example is `proto_tools/tools/structure_scoring/pyrosetta/standalone/python_version.txt`.

**Rebuilds:** the file content **and the resolved version** are both included in the environment setup hash, so any edit triggers a rebuild and two platforms with different resolved versions get distinct hashes (matters when `PROTO_HOME` is on shared storage).

**Consistency tests:** every shipped `python_version.txt` is validated by `tests/style_consistency_tests/test_python_version_consistency.py` (one parametrized result per tool). Parser unit tests live in `tests/tool_infra_tests/test_python_version_files.py`.

## Shared Environments

Multiple tools may rely on the same dependencies. For example, ESM3 and ESM C, which both come from `evolutionaryscale/esm` can share a single micromamba environment on disk. This avoids duplicating environments when adding a sibling model.

**Layout:**

```
proto_tools/
├── shared_envs/
│   └── evolutionaryscale_esm/      # one env definition
│       ├── setup.sh
│       ├── requirements.txt
│       └── python_version.txt
└── tools/
    └── masked_models/
        ├── esm3/
        │   └── standalone/
        │       ├── shared_env.txt   # contents: "evolutionaryscale_esm"
        │       └── inference.py     # ESM3-specific dispatch
        └── esmc/
            └── standalone/
                ├── shared_env.txt   # contents: "evolutionaryscale_esm"
                └── inference.py     # ESM C-specific dispatch
```

**How it resolves:** When a tool's `standalone/` contains a `shared_env.txt` marker, `ToolInstance._resolve_env_def()` reads the marker, looks up `proto_tools/shared_envs/<name>/`, and uses that directory's `setup.sh` / `requirements.txt` / `python_version.txt` / `env_vars.txt` for env construction. The on-disk env path becomes `PROTO_HOME/proto_tool_envs/<name>_env/` so all tools opting into the same shared env collide on the same physical directory and skip redundant setup.

**`inference.py` always lives per-tool** — only env-construction inputs are shared. Each tool ships its own dispatch logic.

**Validation:**

- A tool with both `shared_env.txt` and `setup.sh` raises (ambiguous).
- A `shared_env.txt` pointing to a non-existent shared env raises with a clear error at dispatch time.
- An empty `shared_env.txt` raises.

**Concurrency:** Existing setup-lock at `<env_path>/.setup.lock` serializes concurrent setup attempts from different tools.

**When to use a shared env:**

- Two or more tools install the same heavy Python package and the same Python version.
- A new tool ships in an upstream package that already has a wrapper using the shared-env pattern.

**When NOT to use:** Tools with conflicting Python versions, conflicting framework version pins, or genuinely independent dependency sets.

**Migration note:** When a tool adopts a shared env (e.g. `esm3` migrated to `evolutionaryscale_esm`), its on-disk env directory changes from `<tool_name>_env/` to `<shared_name>_env/`. The old directory is orphaned but harmless; users can manually delete `PROTO_HOME/proto_tool_envs/<old_name>_env/` to reclaim disk.

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

JAX tools come in two flavors depending on what the upstream library gives you. The difference matters because it determines whether your tool can be moved between devices, or has to be thrown away and rebuilt.

**Pattern 1 — move-based** (`mock_jax_tool`, ProteinMPNN):
```python
def to_device(device: str) -> dict:
    global _model
    if _model is not None and _model._loaded:
        _model.to_device(device)  # jax.device_put(params, device) via move_model_to_device
    return {"success": True, "device": device}
```
Use this when the library hands you the model weights as plain data — a dict of arrays you can hold and inspect. You can physically move those arrays between GPU and CPU with `jax.device_put`, and the compiled forward pass runs against them wherever they live. `move_model_to_device()` in `standalone_helpers.py` does the transfer and frees GPU memory via `jax.clear_caches()` when moving off CUDA.

**Pattern 2 — reload-based** (AlphaGenome):
```python
def to_device(device: str) -> dict:
    global _model
    if _model is not None and hasattr(_model, "to_device"):
        _model.to_device(device)
    return {"success": True, "device": device}
```
Use this when the library hands you a black-box model object (e.g. from `dna_model.create()`) with its weights and compiled forward pass hidden behind a wrapper. You can't reach in to move anything, so your only option is to destroy the model and create a new one on the target device.

For CPU eviction specifically, even "create a new one on CPU" is a bad trade: compiling a large model like AlphaGenome for the CPU backend takes 10+ minutes, and CPU inference is too slow to actually use. So the tool's `to_device("cpu")` just **unloads** the model (frees the GPU memory and drops the reference) without putting anything on CPU. The next dispatch triggers a fresh load back onto GPU.

There's no registry flag to tell you which pattern to use — it's determined by what the library exposes. If the weights are reachable as plain data, use move-based. Otherwise, use reload-based.

**Neither pattern works — use `gpu_only=True`**

Some tools can't even do the reload-based pattern safely: the first run works, but the second run crashes the worker (either in the same process or in a fresh subprocess). This is usually an upstream bug in how the library tracks CUDA or XLA state across runs. Mark these with `gpu_only=True` in the `@tool()` decorator:

```python
@tool(
    key="alphagenome-predict-variants",
    ...
    uses_gpu=True,
    gpu_only=True,
    ...
)
```

The framework then changes two things for this tool:

1. **It refuses CPU dispatch up front.** Calling the tool with `config.device="cpu"` raises `ValueError` immediately, so misconfigurations fail clearly instead of crashing deep inside the worker.
2. **On LRU eviction, it kills the worker outright.** Instead of sending `to_device("cpu")` (which would hit the broken reload path), the framework calls `worker.stop()`, logs a warning, and drops the reference. The next time this tool is dispatched, a brand-new subprocess is spawned on GPU from scratch — fresh imports, fresh model load, fresh compile. It's slow, but it's always correct.

`gpu_only=True` implies `uses_gpu=True` — the framework checks this at registration. Currently only `alphagenome-predict-variants` opts in; the other alphagenome variants use plain reload-based because they don't exhibit the consecutive-dispatch crash.

**CLI tools** (Boltz2, RFDiffusion3, BLAST, etc.):
```python
def to_device(device: str) -> dict:
    return {"success": True, "device": device, "note": "CLI tool, auto-unloads"}
```

When implementing new tools, add `to_device()` to `standalone/inference.py` or `standalone/run.py` following the pattern above. The model wrapper class should have a `to_device(device: str)` method for actual device moves.
