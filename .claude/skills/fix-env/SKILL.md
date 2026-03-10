---
name: fix-env
description: >
  Debugs and fixes tool environment setup failures in bio-programming-tools.
  Covers infrastructure failures (compute detection, env variable isolation,
  sitecustomize.py injection, micromamba install), PyTorch/CUDA issues (ABI
  mismatch, broken symlinks, triton coordination), JAX setup/downgrades,
  compilation failures (GCC/nvcc mismatch, source builds), network/binary
  download failures, platform issues (aarch64, Python version), and device
  management setup failures (standalone helpers, CUDA visibility). Use when
  setup.sh fails, tool environments break after system updates, or standalone
  venvs need cross-platform fixes.
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
---

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
- `standalone/setup.sh`
- `standalone/requirements.txt`
- `standalone/env_vars.txt`
- `standalone/binary_config.py`
- `standalone/python_version.txt`

**Files you MUST NOT modify:**
- `standalone/run.py`
- `standalone/inference.py`
- `{tool_name}.py` (core implementation)

## Common Failure Patterns

| Category | Pattern | Symptoms | Solution |
|----------|---------|----------|---------|
| **Infra** | Compute Detection (#1) | Wrong torch installed, `No GPU detected` | Verify nvidia-smi, check DETECTED_* vars |
| **Infra** | Env Var Isolation (#2) | `uv installs to wrong env`, missing libs | Check CONDA_PREFIX, VIRTUAL_ENV, LD_LIBRARY_PATH |
| **Infra** | sitecustomize.py (#3) | `ctypes.CDLL` errors, `CC not found` | Verify generated file, check lib paths |
| **Infra** | Micromamba Install (#4) | `Failed to download/extract micromamba` | Check network, manual install to cache |
| **PyTorch** | ABI Mismatch (#5) | `undefined symbol`, `ImportError: *.so` | Cache clear + `--refresh` + validate deep imports |
| **PyTorch** | Broken CUDA Symlinks (#6) | `libcudart.so: No such file` | Auto-repair symlinks in cuda_env |
| **PyTorch** | Triton Version (#7) | `PY_SSIZE_T_CLEAN` crash | Upgrade triton AFTER all other installs |
| **JAX** | Version Downgrade (#8) | JAX uses CPU on GPU, wrong CUDA plugin | Re-apply JAX spec after dependency install |
| **Compile** | GCC/nvcc Mismatch (#9) | `_Float32 undeclared`, nvcc errors | Match GCC to CUDA version, pin sysroot |
| **Compile** | Platform Detection (#10) | `No CUDA target`, `not supported on aarch64` | Platform guards + graceful fallbacks |
| **Network** | GitHub Wheel 404 (#11) | `HTTP Error 404/502` | Switch to PyPI |
| **Compile** | OOM Source Build (#12) | `Killed signal`, exit -9 | Prefer wheels, `MAX_JOBS=1` |
| **Network** | Binary Download (#13) | `Failed after 3 attempts` | Check network, platform support in binary_config.py |
| **Platform** | CUDA Headers (#14) | `cuda_runtime.h: No such file` | Conditional symlinks |
| **Platform** | Python Version (#15) | `No wheel for Python 3.12` | python_version.txt |
| **Device Mgmt** | Standalone Helpers Import (#16) | `ImportError: standalone_helpers` | Check bootstrap copy, verify source exists |
| **Device Mgmt** | CUDA Visibility Mismatch (#17) | `No available device`, wrong GPU | Check CUDA_VISIBLE_DEVICES vs BIO_TOOLS_MANAGED_DEVICES |

For detailed patterns with full bash examples: Read `.claude/skills/fix-env/PATTERNS.md`

## Debugging Workflow

### 1. Read STATUS.txt
```bash
cat tool_envs/{tool}_env/STATUS.txt
```
Check for error messages, DETECTED_* var values, and the exit point.

### 2. Verify Compute Detection (Pre-Flight)
```bash
# Check actual hardware
nvidia-smi

# Check what the detection set
python -c "
from bio_programming_tools.utils.compute_deps import detect_compute_environment
env = detect_compute_environment()
for k, v in sorted(env.items()):
    print(f'{k}={v}')
"
```
If nvidia-smi works but `DETECTED_COMPUTE_PLATFORM=cpu`, compute detection failed — see Pattern 1. **Fix this first** — everything downstream (torch, JAX) depends on correct detection.

### 3. Verify Env Isolation (Pre-Flight)
```bash
# Inside a tool subprocess, check critical env vars:
# CONDA_PREFIX and VIRTUAL_ENV should point to tool_envs/{tool}_env
# LD_LIBRARY_PATH should include cuda_env/lib for CUDA JIT tools
# python sys.prefix should match tool env path
```
If env vars point to the parent conda env instead of the tool env, see Pattern 2. **Fix this before pattern-matching** — wrong env isolation causes misleading package-not-found errors.

### 4. Match Error to Pattern
Match the error in STATUS.txt to patterns in the table above. Read the detailed pattern in PATTERNS.md for full debugging steps and bash examples.

### 5. Apply Fix to setup.sh
Use defensive patterns (`|| true`, conditional checks, graceful fallbacks) that fix the current machine without breaking others.

### 6. Validate Fix on Current Machine
```bash
rm -rf tool_envs/{tool}_env
pytest -k "tool_name" --all -sv
pytest --cpu --skip-ci
pytest --gpu --all  # if GPU available
```

### 7. Document What You Changed
Add comments explaining what, why, and why it's safe for other platforms.

## Validation Checklist

- [ ] Root cause identified (not just symptoms)
- [ ] Only standalone/ files modified (never run.py/inference.py)
- [ ] DETECTED_COMPUTE_PLATFORM matches actual hardware
- [ ] CONDA_PREFIX and VIRTUAL_ENV point to tool env path inside subprocess
- [ ] LD_LIBRARY_PATH includes cuda_env/lib for CUDA JIT tools (evo1, evo2, protenix)
- [ ] Uses `|| true` for operations that might fail on some platforms
- [ ] Uses conditional checks (`if [ -d ... ]`) before filesystem operations
- [ ] Uses `2>/dev/null` to suppress expected errors
- [ ] No hardcoded platform-specific paths (uses detection)
- [ ] Comments explain why changes are safe for other platforms
- [ ] Tool environment deleted and rebuilt successfully
- [ ] Tool tests pass on current machine
- [ ] Broader test suite checked for regressions

## Reference Documentation

- **Cache management**: See "Cache Management for ABI-Sensitive Packages" in `docs/tool-environments.md`
- **Compute deps**: See "Compute Dependency Management" in `docs/tool-environments.md` for hardware detection
- **GCC/nvcc compat**: See "GCC/nvcc Compatibility for CUDA JIT Tools" in `docs/tool-environments.md` for version mapping
- **Compile-from-source**: See "Compile-from-Source Tools" in `docs/tool-environments.md` for TMalign/USalign pattern
- **Python versions**: See "Python Version Specification" in `docs/tool-environments.md` for python_version.txt
- **Binary installation**: See "Binary Installation" in `docs/tool-environments.md` for install_binary.py usage
- **env_vars.txt format**: See "env_vars.txt sections" under "Compute Dependency Management" in `docs/tool-environments.md`
- **Device management**: See `docs/device-management.mdx` for GPU allocation and LRU eviction; see `docs/tool-environments.md` for `to_device()` protocol
- **Standalone helpers**: See "Standalone Helpers for CLI Subprocess Device Routing" in `docs/tool-environments.md` for `get_subprocess_device_env()`
- **Platform reports**: See `notes/environments/` for per-machine compatibility reports
