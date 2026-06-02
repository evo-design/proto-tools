---
name: fix-env
description: >
  Fixes tool environment setup failures in proto-tools, either just for the
  current machine (eject the tool's standalone dir, patch it, and point
  PROTO_<TOOLKIT>_STANDALONE_DIR at it; works for any install, including a
  non-editable pip install) or as a cross-platform fix contributed back to the
  repo. Same diagnosis for both: infrastructure failures (compute detection, env
  variable isolation, sitecustomize.py injection, micromamba install),
  PyTorch/CUDA issues (ABI mismatch, broken symlinks, triton coordination), JAX
  setup/downgrades, compilation failures (GCC/nvcc mismatch, source builds),
  network/binary download failures, platform issues (aarch64, Python version),
  and device management setup failures (standalone helpers, CUDA visibility).
  Use when a tool's setup.sh fails, an env breaks after a system update, or a
  standalone venv needs a fix — for your own use or to upstream.
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
---

# fix-env

**When to use:** a tool's environment fails to build or work, and you need to fix it — either just on the current machine, or as a fix the whole project should ship.

## Two ways to fix an env

The diagnosis is identical; what differs is **where you apply the fix** and **how compatible it has to be**. Decide the mode first; the failure patterns and workflow below apply to both.

### Local fix — "I just need this tool working on my machine"

For when you (or a user you're helping) hit a broken env and want it working now, **without modifying the installed package**. Works for any install, including a non-editable `pip install` where the packaged files sit in read-only site-packages.

1. **Eject the tool's setup** into an editable copy and set the variable it prints:
   ```bash
   proto-tools eject-standalone <toolkit>          # -> ./proto_standalone/<toolkit>/
   export PROTO_<TOOLKIT>_STANDALONE_DIR=$PWD/proto_standalone/<toolkit>
   ```
2. **See the failure live** by re-running the tool with `PROTO_ENV_VERBOSE=1` (streams `setup.sh` output to your terminal); diagnose with the patterns below.
3. **Patch the ejected files** (`setup.sh`, etc.) under `./proto_standalone/<toolkit>/`.
4. **Rebuild by re-running the tool** — with the variable set, the next call builds from your copy under an isolated env name. Iterate until it runs.

Never edit the installed package. Tell the user to export the variable per project (e.g. a `direnv` `.envrc`) so it applies only where they want. Reference: "Overriding a tool's standalone env" in `notes/tool-environments.md`.

### Contributed fix — "the packaged setup should change for everyone"

For when you have the repo (editable install) and the fix belongs upstream so every platform benefits.

**You will only be testing on the current machine.** Assume the existing setup works on other clusters. Make surgical changes to `standalone/setup.sh` (and other standalone/ files) that fix the current machine while maintaining compatibility with other platforms, using defensive patterns (`|| true`, conditional checks, graceful fallbacks), then commit them.

## Files you can modify

Both modes edit the same files — the ejected copy (local) or the repo copy (contributed):

- `standalone/setup.sh`
- `standalone/requirements.txt`
- `standalone/env_vars.txt`
- `standalone/binary_config.py`
- `standalone/python_version.txt`

**Never** modify `standalone/run.py`, `standalone/inference.py`, or `{toolkit}.py` (core implementation).

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
from proto_tools.utils.compute_deps import detect_compute_environment
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

### 5. Apply the Fix to setup.sh
Edit the **ejected copy** (local fix) or the repo's `standalone/setup.sh` (contributed fix). For a contributed fix, use defensive patterns (`|| true`, conditional checks, graceful fallbacks) that fix the current machine without breaking others; a local fix only has to work here.

### 6. Validate the Fix
**Local fix:** re-run the tool with `PROTO_<TOOLKIT>_STANDALONE_DIR` set — the env rebuilds from your patched copy; confirm the tool runs.

**Contributed fix:** rebuild the packaged env and run the tests:
```bash
rm -rf tool_envs/{tool}_env
pytest -k "tool_key" --all -sv
pytest --cpu-only --skip-ci
pytest --gpu-only --all  # if GPU available
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

- **Cache management**: See "Cache Management for ABI-Sensitive Packages" in `notes/tool-environments.md`
- **Compute deps**: See "Compute Dependency Management" in `notes/tool-environments.md` for hardware detection
- **GCC/nvcc compat**: See "GCC/nvcc Compatibility for CUDA JIT Tools" in `notes/tool-environments.md` for version mapping
- **Compile-from-source**: See "Compile-from-Source Tools" in `notes/tool-environments.md` for TMalign/USalign pattern
- **Python versions**: See "Python Version Specification" in `notes/tool-environments.md` for python_version.txt
- **Binary installation**: See "Binary Installation" in `notes/tool-environments.md` for install_binary.py usage
- **env_vars.txt format**: See "env_vars.txt sections" under "Compute Dependency Management" in `notes/tool-environments.md`
- **Device management**: GPU allocation, LRU eviction, and persistence are documented in `proto_tools/utils/device_manager.py` and `proto_tools/utils/tool_instance.py` docstrings (auto-generated reference pages); see `notes/tool-environments.md` for `to_device()` protocol
- **Standalone helpers**: See "Standalone Helpers for CLI Subprocess Device Routing" in `notes/tool-environments.md` for `get_subprocess_device_env()`
