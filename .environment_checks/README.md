# Environment Compatibility Reports

This directory contains machine-generated Markdown reports documenting which tools work on which platforms.

## Generating Reports

```bash
# Run all venv smoke tests and generate a compatibility report
pytest --env-report

# Custom output path
pytest --env-report=custom_report.md

# List tests without running (dry run)
pytest --env-report --collect-only

# Re-test specific tools incrementally (merges into existing report)
pytest --env-report -k "bioemu"
```

The `--env-report` flag:
1. Auto-discovers all standalone-environment tools from `ToolRegistry` and runs each representative tool's `example_input()` — one test per standalone tool directory
2. Cleans tool envs for fresh rebuilds (with `-k`, only cleans envs for selected tools)
3. Skips GPU tools if no GPU is available, skips multi-GPU tools if insufficient GPUs
4. Captures parent process and subprocess environment variables
5. Generates a Markdown report in this directory with an embedded data block for incremental merging

## How It Works

Tests are defined in `tests/tool_infra_tests/test_env_report.py`. A single parametrized test function auto-discovers tools via `ToolRegistry.list_all()` — no manual markers needed. When a new standalone-environment tool is added to the registry with an `example_input`, its tool directory is automatically included in env-report runs.

Config overrides for self-contained testing:
- `verbose=True` on all tools for diagnostic logging
- `use_msa=False` for structure prediction tools (avoids ColabFold dependency)
- Pre-computed MSA fixture for BioEmu (always requires MSAs)

## Incremental Retesting

Use `-k` to re-test specific tools without re-running everything:

```bash
pytest --env-report -k "esmfold"
```

This:
- Only cleans and rebuilds the env for the selected tool(s)
- Merges new results into the existing report (via embedded data block)
- Preserves all other tool results from previous runs

## Report Naming

Reports are named: `{platform_id}.md`

Platform ID format: `[{user}_]{cluster_or_os}[_{hostname}]_{arch}_{gpu_or_cpu}`

The filename uses a shortened platform ID (no date or commit hash). Known hostname
patterns are mapped to friendly names (e.g. Sherlock `sh*` nodes → `sherlock`).
For unrecognized Linux hosts, the sanitized hostname is appended to disambiguate.
Named clusters (chimera, dgx_spark) and macOS already have unique OS parts. Examples:
- `bob_chimera_x86_64_h100.md` (Chimera cluster with H100 GPU)
- `alice_dgx_spark_arm64_gb10.md` (DGX Spark with GB10 GPU)
- `viggiano_sherlock_x86_64_h100.md` (Sherlock compute node)
- `alice_linux_myhost_x86_64_a100.md` (Unknown Linux machine, hostname included)

Reports are overwritten on each full run to keep the latest results per platform/user.
Incremental runs (`-k`) merge into the existing report.

## Report Contents

Each report includes:

1. **Summary badges** — Pass rate, passed/failed/skipped counts
2. **Platform info** — OS, architecture, hostname, Python version, RAM, GPU details
3. **Git info** — Commit hash, branch, dirty status
4. **Environment variables** — Both parent process env and subprocess env (what gets passed to tools)
5. **Results by category** — Table per tool category with status, GPU requirement, venv build status, duration
6. **Failure details** — Full error messages for any failed tests
7. **Embedded data block** — JSON data block for incremental merging across runs

## Interpreting Reports

A tool is considered **working** on a platform if:
- `status` is "passed"
- `venv_status` is "success" (✅)

A tool **failed** if:
- `status` is "failed" (test assertion failed or error during execution)
- `venv_status` is "build_failed" (❌) — `setup.sh` failed during venv creation

A tool was **skipped** if:
- Test was marked skip for any reason (missing deps, platform-specific, GPU not available, etc.)
