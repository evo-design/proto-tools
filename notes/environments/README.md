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
1. Cleans the `tool_envs/` directory to force fresh venv rebuilds
2. Runs ALL tests marked `@pytest.mark.include_in_env_report` (overrides `--cpu`, `--gpu`, `--slow`, `skip_ci`)
3. Skips GPU tests if no GPU is available
4. Captures parent process and subprocess environment variables
5. Generates a Markdown report in this directory

## Marking Tests for Reports

Mark one smoke test per tool with `@pytest.mark.include_in_env_report`:

```python
@pytest.mark.include_in_env_report
def test_my_tool_basic():
    ...
```

For tools where the test file name doesn't match the tool name, or where auto-detection fails, you can explicitly specify the tool name and category:

```python
@pytest.mark.include_in_env_report(tool="my_tool", category="my_category")
def test_my_tool_basic():
    ...
```

Categories should match the directory structure in `proto_tools/tools/` (e.g., `sequence_scoring`, `gene_annotation`, `structure_prediction`, etc.).

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
