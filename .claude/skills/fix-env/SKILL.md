---
name: fix-env
description: >
  Debugs and fixes tool environment setup failures in bio-programming-tools.
  Covers ABI mismatch errors (flash-attn, transformer-engine, torch),
  network failures with GitHub release wheels, OOM during source builds,
  platform detection issues (CUDA, aarch64), and missing CUDA headers for
  JIT compilation. Use when setup.sh fails, tool environments break after
  system updates, or standalone venvs need cross-platform fixes.
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

| Pattern | Symptoms | Solution |
|---------|----------|----------|
| ABI Mismatch | `undefined symbol`, `ImportError: *.so` | Cache clear + `--refresh` + validate deep imports |
| Network Failure | `HTTP Error 404/502` | Switch GitHub URLs to PyPI |
| OOM Source Build | `Killed signal`, exit code -9 | Prefer wheels, `MAX_JOBS=1` fallback |
| Platform Detection | `No CUDA target`, `not supported on aarch64` | Platform guards + graceful fallbacks |
| Missing CUDA Headers | `cuda_runtime.h: No such file` | Conditional symlinks |

For detailed patterns with full bash examples: Read `.claude/skills/fix-env/PATTERNS.md`

## Debugging Workflow

### 1. Identify the Failure
```bash
cat bio_programming_tools/tool_envs/{tool}_env/STATUS.txt
```
Match error patterns: `undefined symbol` -> ABI, `HTTP Error` -> network, `Killed signal` -> OOM, `No such file` -> missing headers.

### 2. Test Specific Component
```bash
rm -rf bio_programming_tools/tool_envs/{tool}_env
pytest -k "tool_name" --all -sv
```

### 3. Apply Fixes to setup.sh
Match the error pattern to the solution in the table above. Use defensive patterns (`|| true`, conditional checks, graceful fallbacks) that fix the current machine without breaking others.

### 4. Validate Fix on Current Machine
```bash
rm -rf bio_programming_tools/tool_envs/{tool}_env
pytest -k "tool_name" --all -sv
pytest --cpu --skip-ci
pytest --gpu --all  # if GPU available
```

### 5. Document What You Changed
Add comments explaining what, why, and why it's safe for other platforms.

## Validation Checklist

- [ ] Root cause identified (not just symptoms)
- [ ] Only standalone/ files modified (never run.py/inference.py)
- [ ] Uses `|| true` for operations that might fail on some platforms
- [ ] Uses conditional checks (`if [ -d ... ]`) before filesystem operations
- [ ] Uses `2>/dev/null` to suppress expected errors
- [ ] No hardcoded platform-specific paths (uses detection)
- [ ] Comments explain why changes are safe for other platforms
- [ ] Tool environment deleted and rebuilt successfully
- [ ] Tool tests pass on current machine
- [ ] Broader test suite checked for regressions

## Reference Documentation

- **Cache management**: See "Cache Management for ABI-Sensitive Packages" in CLAUDE.md
- **Compute deps**: See "Compute Dependency Management" in CLAUDE.md for hardware detection
- **Python versions**: See "Python Version Specification" in CLAUDE.md for python_version.txt
- **Binary installation**: See "Binary Installation" in CLAUDE.md for install_binary.py usage
