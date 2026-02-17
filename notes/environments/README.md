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
```

The `--env-report` flag:
1. Cleans the `.venvs/` directory to force fresh venv rebuilds
2. Runs ALL tests marked `@pytest.mark.run_all_venvs` (overrides `--cpu`, `--gpu`, `--slow`, `skip_ci`)
3. Skips GPU tests if no GPU is available
4. Captures parent process and subprocess environment variables
5. Generates a Markdown report in this directory

## Report Naming

Reports are named: `{platform_id}.md`

Platform ID format: `[{user}_]{cluster_or_os}_{arch}_{gpu_or_cpu}`

The filename uses a shortened platform ID (no date or commit hash). Examples:
- `bviggiano_macosDarwin_arm64_cpu.md` (Mac M-series, no GPU)
- `bob_chimera_x86_64_h100.md` (Chimera cluster with H100 GPU)
- `alice_dgx_spark_arm64_gb10.md` (DGX Spark with GB10 GPU)

Reports are overwritten on each run to keep the latest results per platform/user.

## Report Contents

Each report includes:

1. **Summary badges** — Pass rate, passed/failed/skipped counts
2. **Platform info** — OS, architecture, hostname, Python version, RAM, GPU details
3. **Git info** — Commit hash, branch, dirty status
4. **Environment variables** — Both parent process env and subprocess env (what gets passed to tools)
5. **Results by category** — Table per tool category with status, GPU requirement, venv build status, duration
6. **Failure details** — Full error messages for any failed tests

## Interpreting Reports

A tool is considered **working** on a platform if:
- `status` is "passed"
- `venv_status` is "success" (✅)

A tool **failed** if:
- `status` is "failed" (test assertion failed or error during execution)
- `venv_status` is "build_failed" (❌) — `setup.sh` failed during venv creation

A tool was **skipped** if:
- Test was marked skip for any reason (missing deps, platform-specific, GPU not available, etc.)
