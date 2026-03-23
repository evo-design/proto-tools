"""
Test configuration for bio_programming_tools test suite.

Supports the same CLI options and markers as the main bio-programming tests:
  --cpu        Run only CPU tests
  --gpu        Run only GPU tests (skip CPU tests)
  --all        Include slow and GPU tests
  --slow       Run only slow tests
  --exhaustive Include exhaustive combinatorial tests (e.g., every tool × device)
  --skip-ci    Skip tests marked skip_ci (mimics CI)
  --no-log-console  Disable console logging during tests
  --env-report[=PATH]  Run venv smoke tests and generate compatibility report
"""

from __future__ import annotations

import functools
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pytest

from bio_programming_tools import setup_logging
from bio_programming_tools.tools.tool_registry import ToolRegistry

# Register testing tools so consistency tests cover them.
# These are NOT exported via tools/__init__.py and are invisible outside tests.
import bio_programming_tools.tools.testing  # noqa: F401
from bio_programming_tools.utils.system_info import (
    capture_parent_env,
    collect_system_info,
    get_captured_env,
    get_platform_id,
)
from bio_programming_tools.utils.device import number_of_visible_gpus
from bio_programming_tools.utils.tool_instance import ToolInstance


def is_on_chimera() -> bool:
    """Check if running on the Chimera (arc-slurm) cluster."""
    # First check environment variable (set by some SLURM configs)
    cluster_name = os.environ.get("SLURM_CLUSTER_NAME")
    if cluster_name == "arc-slurm":
        return True

    # If SLURM_JOB_ID is set, we're on SLURM - query scontrol for cluster name
    if os.environ.get("SLURM_JOB_ID"):
        try:
            result = subprocess.run(
                ["scontrol", "show", "config"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # Parse "ClusterName = arc-slurm" from output
                import re

                match = re.search(r"ClusterName\s*=\s*(\S+)", result.stdout)
                if match:
                    return match.group(1) == "arc-slurm"
        except Exception:
            pass

    return False


@functools.cache
def _gpu_available() -> bool:
    """Check if a GPU is likely available (without importing torch)."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() == "":
        return False
    return shutil.which("nvidia-smi") is not None


# ============================================================================
# Environment Report Collector
# ============================================================================
@dataclass
class ToolTestResult:
    """Result of a single tool smoke test."""

    tool_name: str
    category: str
    test_name: str
    status: str  # "passed", "failed", "skipped"
    duration_seconds: float
    uses_gpu: bool
    env_path: str | None
    env_status: str  # "success", "build_failed", "not_found"
    error_message: str | None


@dataclass
class EnvReportCollector:
    """Collects test results for environment compatibility report."""

    output_path: Path | None = None
    results: list[ToolTestResult] = field(default_factory=list)
    _test_start_times: dict[str, float] = field(default_factory=dict)

    def record_start(self, nodeid: str) -> None:
        """Record when a test starts."""
        import time

        self._test_start_times[nodeid] = time.time()

    def record_result(
        self,
        nodeid: str,
        outcome: str,
        *,
        has_gpu_marker: bool = False,
        error_message: str | None = None,
        tool_name: str | None = None,
        category: str | None = None,
    ) -> None:
        """Record a test result."""
        import time

        # Calculate duration
        start_time = self._test_start_times.pop(nodeid, time.time())
        duration = time.time() - start_time

        # Use explicit tool_name/category if provided, otherwise parse from nodeid
        if tool_name:
            parsed_tool = tool_name
            # Use explicit category if provided, otherwise lookup
            if not category:
                category = self._get_category_for_tool(tool_name)
        else:
            parsed_tool, parsed_category = self._parse_tool_from_nodeid(nodeid)
            if not category:
                category = parsed_category

        if not parsed_tool:
            return  # Not a recognizable tool test

        # Determine venv path and status
        env_path, env_status = self._get_venv_info(parsed_tool)

        self.results.append(
            ToolTestResult(
                tool_name=parsed_tool,
                category=category,
                test_name=nodeid,
                status=outcome,
                duration_seconds=round(duration, 2),
                uses_gpu=has_gpu_marker,
                env_path=env_path,
                env_status=env_status,
                error_message=error_message,
            )
        )

    def _get_category_for_tool(self, tool_name: str) -> str:
        """Get category for a tool name from the ToolRegistry."""
        tool_categories = ToolRegistry.get_tool_categories()
        return tool_categories.get(tool_name, "unknown")

    def _parse_tool_from_nodeid(self, nodeid: str) -> tuple[str | None, str | None]:
        """Extract tool name and category from test nodeid."""
        # Pattern: tests/{category}_tests/test_{tool}.py::...
        # or: tests/test_{category}.py::...
        match = re.search(r"test_(\w+)\.py", nodeid)
        if not match:
            return None, None

        test_file = match.group(1)

        # Map test file names to tool names (when they differ)
        file_to_tool = {
            "local_colabfold_search": "colabfold_search",
            "rna_splicing": "splice_transformer",
            "viennarna_secondary_structure_prediction": "viennarna",
        }

        tool_name = file_to_tool.get(test_file, test_file)
        tool_categories = ToolRegistry.get_tool_categories()
        category = tool_categories.get(tool_name)

        if category:
            return tool_name, category

        # Fallback: use test file name as tool name
        return test_file, "unknown"

    def _get_venv_info(self, tool_name: str) -> tuple[str | None, str]:
        """Get venv path and status for a tool."""
        project_root = Path(__file__).parent.parent
        venvs_dir = project_root / "tool_envs"

        # Look for venv with tool name
        for venv_dir in venvs_dir.glob(f"*{tool_name}*"):
            if venv_dir.is_dir():
                # Check if venv has python executable
                python_path = venv_dir / "bin" / "python"
                if python_path.exists():
                    return str(venv_dir), "success"
                return str(venv_dir), "build_failed"

        return None, "not_found"

    def finalize_and_write(self) -> Path | None:
        """Write the report to disk as a README.md and return the path."""
        if not self.results:
            return None

        # Collect system info
        system_info = collect_system_info()
        git_info = system_info["git_info"]
        platform = system_info["platform"]
        gpu = system_info["gpu"]
        parent_env = system_info["parent_process_env"]

        # Build summary
        passed = sum(1 for r in self.results if r.status == "passed")
        failed = sum(1 for r in self.results if r.status == "failed")
        skipped = sum(1 for r in self.results if r.status == "skipped")
        total = len(self.results)

        # Group results by category
        by_category: dict[str, list[ToolTestResult]] = {}
        for r in self.results:
            by_category.setdefault(r.category, []).append(r)

        # Build README content
        lines: list[str] = []

        # Header with badges
        pass_rate = int(100 * passed / total) if total > 0 else 0
        if pass_rate >= 80:
            rate_color = "brightgreen"
        elif pass_rate >= 50:
            rate_color = "yellow"
        else:
            rate_color = "red"

        # Determine cluster/environment name for title
        cluster_name = os.environ.get("SLURM_CLUSTER_NAME")
        if cluster_name == "arc-slurm":
            env_name = "Chimera"
        elif "dgx" in platform['hostname'].lower() or "spark" in platform['hostname'].lower():
            env_name = "DGX Spark"
        else:
            env_name = f"{platform['os']} {platform['architecture']}"

        lines.append(f"# {env_name} Environment Report")
        lines.append("")
        lines.append(
            f"![Pass Rate](https://img.shields.io/badge/pass_rate-{pass_rate}%25-{rate_color})"
            f" ![Passed](https://img.shields.io/badge/passed-{passed}-brightgreen)"
            f" ![Failed](https://img.shields.io/badge/failed-{failed}-red)"
            f" ![Skipped](https://img.shields.io/badge/skipped-{skipped}-lightgrey)"
        )
        lines.append("")

        # Platform info table
        lines.append("## Platform")
        lines.append("")
        lines.append("| Property | Value |")
        lines.append("|----------|-------|")
        lines.append(f"| **OS** | {platform['os']} {platform['os_version']} |")
        lines.append(f"| **Architecture** | {platform['architecture']} |")
        lines.append(f"| **Hostname** | `{platform['hostname']}` |")
        lines.append(f"| **Python** | {platform['python_version']} |")
        lines.append(f"| **RAM** | {platform['ram_gb']:.1f} GB |")

        # GPU info
        if gpu["available"]:
            devices = ", ".join(d["name"] for d in gpu["devices"]) or "Unknown"
            lines.append(f"| **GPU** | {gpu['count']}× {devices} |")
            if gpu["cuda_version"]:
                lines.append(f"| **CUDA** | {gpu['cuda_version']} |")
        else:
            lines.append("| **GPU** | None |")

        # Parent environment
        if parent_env["type"] == "mamba":
            lines.append(f"| **Mamba Env** | `{parent_env['name']}` |")
        elif parent_env["type"] == "conda":
            lines.append(f"| **Conda Env** | `{parent_env['name']}` |")
        elif parent_env["type"] == "venv":
            lines.append(f"| **Venv** | `{parent_env['prefix']}` |")

        lines.append("")

        # Git info
        lines.append("## Git")
        lines.append("")
        lines.append(f"- **Commit**: `{git_info['commit']}`")
        lines.append(f"- **Branch**: `{git_info['branch']}`")
        lines.append(f"- **Dirty**: {'Yes' if git_info['dirty'] else 'No'}")
        lines.append("")

        # Environment variables
        captured_env = get_captured_env()
        if captured_env["parent_env"] or captured_env["subprocess_env"]:
            lines.append("## Environment Variables")
            lines.append("")

            if captured_env["parent_env"]:
                lines.append("### Parent Process Environment")
                lines.append("")
                lines.append("```")
                for key in sorted(captured_env["parent_env"].keys()):
                    value = captured_env["parent_env"][key]
                    # Truncate long values (like PATH)
                    if len(value) > 200:
                        value = value[:200] + "..."
                    lines.append(f"{key}={value}")
                lines.append("```")
                lines.append("")

            if captured_env["subprocess_env"]:
                lines.append("### Subprocess Environment (passed to tools)")
                lines.append("")
                lines.append("```")
                for key in sorted(captured_env["subprocess_env"].keys()):
                    value = captured_env["subprocess_env"][key]
                    # Truncate long values (like PATH)
                    if len(value) > 200:
                        value = value[:200] + "..."
                    lines.append(f"{key}={value}")
                lines.append("```")
                lines.append("")

        # Results by category
        lines.append("## Results by Category")
        lines.append("")

        for category in sorted(by_category.keys()):
            results = by_category[category]
            cat_passed = sum(1 for r in results if r.status == "passed")
            cat_total = len(results)

            # Category header with mini badge
            lines.append(f"### {category.replace('_', ' ').title()} ({cat_passed}/{cat_total})")
            lines.append("")
            lines.append("| Tool | Requires GPU | Venv Build Succeeded | Duration | Status |")
            lines.append("|------|--------------|----------------------|----------|--------|")

            for r in sorted(results, key=lambda x: x.tool_name):
                # Status emoji
                if r.status == "passed":
                    status = "✅ Pass"
                elif r.status == "failed":
                    status = "❌ Fail"
                else:
                    status = "⏭️ Skip"

                # Duration formatting
                duration = f"{r.duration_seconds:.1f}s" if r.status != "skipped" else "—"

                # GPU indicator
                gpu_req = "yes" if r.uses_gpu else "no"

                # Venv status
                if r.env_status == "success":
                    venv = "✅"
                elif r.env_status == "build_failed":
                    venv = "❌"
                else:
                    venv = "—"

                lines.append(f"| `{r.tool_name}` | {gpu_req} | {venv} | {duration} | {status} |")

            lines.append("")

        # Failure details section
        failures = [r for r in self.results if r.status == "failed" and r.error_message]
        if failures:
            lines.append("## Failure Details")
            lines.append("")

            for r in failures:
                lines.append(f"### ❌ `{r.tool_name}`")
                lines.append("")
                lines.append(f"**Test**: `{r.test_name}`")
                lines.append("")
                lines.append("```")
                lines.append(r.error_message)
                lines.append("```")
                lines.append("")

        # Footer
        lines.append("---")
        lines.append(
            f"*Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"by `pytest --env-report`*"
        )

        # Determine output path
        if self.output_path:
            output_path = self.output_path
            # Ensure .md extension
            if output_path.suffix != ".md":
                output_path = output_path.with_suffix(".md")
        else:
            # Default: notes/environments/{platform_id}.md
            project_root = Path(__file__).parent.parent
            env_dir = project_root / "notes" / "environments"
            env_dir.mkdir(parents=True, exist_ok=True)
            platform_id = get_platform_id(include_date=False, include_commit=False)
            output_path = env_dir / f"{platform_id}.md"

        # Write report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        return output_path


# Global collector instance (set in pytest_configure if --env-report is used)
_env_report_collector: EnvReportCollector | None = None


# ============================================================================
# Pytest Hooks
# ============================================================================
def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--cpu",
        action="store_true",
        default=False,
        help="Run only CPU tests, skip GPU tests",
    )
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="Run only GPU tests, skip CPU tests",
    )
    parser.addoption(
        "--all",
        action="store_true",
        default=False,
        help="Run all tests including slow and GPU tests",
    )
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run only slow tests",
    )
    parser.addoption(
        "--skip-ci",
        action="store_true",
        default=False,
        help="Skip tests marked with skip_ci (mimics CI environment behavior)",
    )
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests (hit external APIs/services). Skipped by default.",
    )
    parser.addoption(
        "--exhaustive",
        action="store_true",
        default=False,
        help="Include exhaustive combinatorial tests (e.g., every tool × device transition)",
    )
    parser.addoption(
        "--no-log-console",
        action="store_true",
        default=False,
        help="Disable console logging during tests",
    )
    parser.addoption(
        "--env-report",
        nargs="?",
        const=True,
        default=False,
        help=(
            "Run venv smoke tests and generate platform compatibility report. "
            "Cleans tool_envs/ first, overrides --cpu/--gpu/--slow/skip_ci filters. "
            "Optionally specify output path: --env-report=path/to/report.md"
        ),
    )


def pytest_configure(config):
    """Configure pytest with custom markers and options."""
    global _env_report_collector

    config.addinivalue_line("markers", "uses_gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "uses_cpu: mark test as CPU-only")
    config.addinivalue_line(
        "markers", "include_in_env_report: Include test in --env-report (one smoke test per tool). "
        "Accepts optional kwargs: tool='tool_name', category='category_name'"
    )

    # Set environment variable to indicate we're in pytest
    # This prevents setup_logging() from creating timestamped files during test imports
    os.environ["PYTEST_RUNNING"] = "1"

    # Hide CUDA devices when --skip-ci is specified to simulate CI environment
    if config.getoption("--skip-ci"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Handle --env-report
    env_report_opt = config.getoption("--env-report")
    if env_report_opt:
        # Parse output path if provided
        output_path = None
        if isinstance(env_report_opt, str):
            output_path = Path(env_report_opt)

        # Initialize collector
        _env_report_collector = EnvReportCollector(output_path=output_path)

        # Capture parent process environment before any tools run
        capture_parent_env()

        # Clean tool_envs/ directory to force fresh rebuilds
        project_root = Path(__file__).parent.parent
        venvs_dir = project_root / "tool_envs"
        if venvs_dir.exists():
            logger = logging.getLogger("bio_programming_tools.tests")
            logger.warning(
                f"Cleaning {venvs_dir} for fresh environment rebuilds "
                "(this may take a while on network filesystems)..."
            )
            # Use subprocess instead of shutil.rmtree to avoid NFS hang issues
            subprocess.run(
                ["rm", "-rf", str(venvs_dir)],
                check=False,
                capture_output=True,
            )
            logger.warning(f"Finished cleaning {venvs_dir}")


def pytest_runtest_logstart(nodeid, location):
    """Log when a test starts (DEBUG level, file only)."""
    logger = logging.getLogger("bio_programming_tools.tests")
    logger.debug(f"TEST START: {nodeid}")

    # Record start time for env report
    if _env_report_collector is not None:
        _env_report_collector.record_start(nodeid)


def pytest_runtest_logreport(report):
    """Log test results (DEBUG level to avoid console output)."""
    logger = logging.getLogger("bio_programming_tools.tests")

    # Only log on the call phase (not setup/teardown)
    if report.when == "call":
        if report.passed:
            logger.debug(f"TEST PASSED: {report.nodeid}")
        elif report.failed:
            logger.error(f"TEST FAILED: {report.nodeid}")
            if report.longrepr:
                # Use DEBUG level to keep it file-only, but prefix with ERROR for visibility in logs
                logger.debug(f"Error Traceback:\n{report.longreprtext}")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture test results for env report."""
    outcome = yield
    report = outcome.get_result()

    # Only record final outcome (call phase for pass/fail, setup/teardown for errors)
    if _env_report_collector is not None:
        # Only record tests marked with include_in_env_report
        if "include_in_env_report" not in item.keywords:
            return

        # Extract tool name and category from marker if provided
        tool_name = None
        category = None
        marker = item.get_closest_marker("include_in_env_report")
        if marker:
            tool_name = marker.kwargs.get("tool")
            category = marker.kwargs.get("category")

        # Determine outcome
        if report.when == "call":
            if report.passed:
                status = "passed"
                error_msg = None
            elif report.failed:
                status = "failed"
                error_msg = str(report.longrepr) if report.longrepr else None
            elif report.skipped:
                status = "skipped"
                error_msg = str(report.longrepr) if report.longrepr else None
            else:
                return

            # Check for GPU marker
            has_gpu = "uses_gpu" in item.keywords

            _env_report_collector.record_result(
                item.nodeid,
                status,
                has_gpu_marker=has_gpu,
                error_message=error_msg,
                tool_name=tool_name,
                category=category,
            )
        elif report.when == "setup" and report.failed:
            # Setup failure
            _env_report_collector.record_result(
                item.nodeid,
                "failed",
                has_gpu_marker="uses_gpu" in item.keywords,
                error_message=f"Setup failed: {report.longrepr}",
                tool_name=tool_name,
                category=category,
            )


def pytest_sessionfinish(session, exitstatus):
    """Log test session summary at the end and write env report if requested."""
    logger = logging.getLogger("bio_programming_tools.tests")

    # Get test statistics from the session
    test_reports = session.items
    num_collected = len(test_reports)

    # Count passed and failed tests from the terminal reporter
    if hasattr(session.config, "pluginmanager"):
        terminalreporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if terminalreporter:
            stats = terminalreporter.stats

            passed = len(stats.get("passed", []))
            failed = len(stats.get("failed", []))
            skipped = len(stats.get("skipped", []))
            errors = len(stats.get("error", []))

            # Build summary message
            summary_lines = [
                "\n" + "=" * 80,
                "TEST SESSION SUMMARY",
                "=" * 80,
                f"Tests collected: {num_collected}",
                f"Tests passed:    {passed}",
                f"Tests failed:    {failed}",
                f"Tests skipped:   {skipped}",
                f"Tests errors:    {errors}",
            ]

            # Add list of failed tests if any
            if failed > 0:
                summary_lines.append("\nFailed tests:")
                failed_reports = stats.get("failed", [])
                for report in failed_reports:
                    summary_lines.append(f"  - {report.nodeid}")

            # Add list of error tests if any
            if errors > 0:
                summary_lines.append("\nTests with errors:")
                error_reports = stats.get("error", [])
                for report in error_reports:
                    summary_lines.append(f"  - {report.nodeid}")

            summary_lines.append("=" * 80)

            # Log the summary at INFO level so it appears in both console and file
            summary_message = "\n".join(summary_lines)
            logger.info(summary_message)

    # Write env report if collector is active
    if _env_report_collector is not None:
        report_path = _env_report_collector.finalize_and_write()
        if report_path:
            logger.info(f"\n--env-report: Compatibility report written to {report_path}")
            print(f"\n[env-report] Report written to: {report_path}")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options and auto-mark tests."""
    # --env-report: keep only venv smoke tests, skip everything else
    if config.getoption("--env-report"):
        skip_not_venv = pytest.mark.skip(reason="--env-report: not a venv smoke test")
        skip_no_gpu = pytest.mark.skip(reason="--env-report: GPU not available")
        skip_not_chimera = pytest.mark.skip(
            reason="--env-report: requires Chimera cluster"
        )
        gpu_available = _gpu_available()
        on_chimera = is_on_chimera()

        for item in items:
            if "include_in_env_report" not in item.keywords:
                item.add_marker(skip_not_venv)
            elif "only_chimera" in item.keywords and not on_chimera:
                item.add_marker(skip_not_chimera)
            elif "uses_gpu" in item.keywords and not gpu_available:
                # Skip GPU tests on platforms without GPU, but still include in report
                item.add_marker(skip_no_gpu)
        return

    # Auto-mark all tests as CPU-only unless explicitly marked as GPU
    for item in items:
        # If no GPU marker found, mark as CPU
        if not any(mark.name == "uses_gpu" for mark in item.iter_markers()):
            item.add_marker(pytest.mark.uses_cpu)

    # Skip tests marked with skip_ci when running in GitHub Actions or --skip-ci is specified
    if os.getenv("GITHUB_ACTIONS") == "true" or config.getoption("--skip-ci"):
        skip_ci = pytest.mark.skip(
            reason="Skipped in CI environment (GitHub Actions or --skip-ci)"
        )
        for item in items:
            if "skip_ci" in item.keywords:
                item.add_marker(skip_ci)

    # Skip GPU tests when --cpu is specified
    if config.getoption("--cpu"):
        skip_gpu = pytest.mark.skip(reason="--cpu specified")
        for item in items:
            if "uses_gpu" in item.keywords:
                item.add_marker(skip_gpu)

    # Skip CPU tests when --gpu is specified
    elif config.getoption("--gpu"):
        skip_cpu = pytest.mark.skip(reason="--gpu specified")
        for item in items:
            if "uses_cpu" in item.keywords and "uses_gpu" not in item.keywords:
                item.add_marker(skip_cpu)

    # Default: skip GPU tests unless --integration (with GPU available) or --all
    elif not (
        config.getoption("--all")
        or (config.getoption("--integration") and _gpu_available())
    ):
        skip_gpu = pytest.mark.skip(
            reason="GPU test (use --gpu, --integration with GPU, or --all to run)"
        )
        for item in items:
            if "uses_gpu" in item.keywords:
                item.add_marker(skip_gpu)

    # Handle slow test filtering
    run_all = config.getoption("--all")
    run_slow_only = config.getoption("--slow")

    if run_slow_only:
        # When --slow is specified, skip tests NOT marked as slow
        skip_non_slow = pytest.mark.skip(
            reason="--slow specified, skipping non-slow tests"
        )
        for item in items:
            if "slow" not in item.keywords:
                item.add_marker(skip_non_slow)
    elif not run_all and not config.getoption("--exhaustive"):
        # By default, skip slow tests (--all or --exhaustive bypass this)
        skip_slow = pytest.mark.skip(
            reason="slow test (use --all to run, or --slow to run only slow tests)"
        )
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # Skip integration tests unless --integration or --all is specified
    if not config.getoption("--integration") and not run_all:
        skip_integration = pytest.mark.skip(
            reason="integration test (use --integration to run, or --all for everything)"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

    # Skip exhaustive tests unless --exhaustive is specified
    if not config.getoption("--exhaustive"):
        skip_exhaustive = pytest.mark.skip(
            reason="exhaustive test (use --exhaustive to run)"
        )
        for item in items:
            if "exhaustive" in item.keywords:
                item.add_marker(skip_exhaustive)

    # Skip only_chimera tests when not on Chimera cluster
    if not is_on_chimera():
        skip_not_chimera = pytest.mark.skip(
            reason="Test requires Chimera cluster (SLURM_CLUSTER_NAME != 'arc-slurm')"
        )
        for item in items:
            if "only_chimera" in item.keywords:
                item.add_marker(skip_not_chimera)

    # Skip uses_gpu(n) tests when fewer than n GPUs are visible
    visible_gpus = number_of_visible_gpus()
    for item in items:
        for marker in item.iter_markers("uses_gpu"):
            required = marker.args[0] if marker.args else 1
            if visible_gpus < required:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"Requires {required} GPUs, only {visible_gpus} visible"
                    )
                )


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging(request):
    """Set up logging for the test session. Runs early to prevent timestamped log files."""
    # Use same log directory as application logs (logs/ in project root)
    project_root = Path(__file__).parent.parent
    log_dir = os.environ.get(
        "BIO_PROGRAMMING_TOOLS_LOG_DIR", str(project_root / "logs")
    )

    # Get options from command line
    no_log_console = request.config.getoption("--no-log-console")
    k_expression = request.config.getoption("-k", default=None)

    # Clear any existing handlers first to prevent duplicate log files
    bio_programming_tools_logger = logging.getLogger("bio_programming_tools")
    bio_programming_tools_logger.handlers.clear()

    # Create header with pytest command and timestamp
    pytest_command = " ".join(sys.argv)
    now = datetime.now()
    timestamp = now.strftime("%H:%M:%S")
    datestamp = now.strftime("%m/%d/%Y")
    header = f"Pytest Run Command: `{pytest_command}`\nRun Started: {timestamp} on {datestamp}\n{'=' * 80}\n\n"

    # Create log filename based on -k parameter or timestamp
    if k_expression:
        # Sanitize the -k expression to make it filename-safe
        # Replace spaces with underscores, remove special characters
        sanitized = re.sub(
            r"[^\w\s-]", "", k_expression
        )  # Remove special chars except spaces, hyphens, underscores
        sanitized = re.sub(r"\s+", "_", sanitized)  # Replace spaces with underscores
        sanitized = sanitized.strip("_")  # Remove leading/trailing underscores
        log_filename = f"pytest_{sanitized}.log"
    else:
        # Use timestamp for the log file
        file_timestamp = now.strftime("%Y%m%d_%H%M%S")
        log_filename = f"pytest_{file_timestamp}.log"

    # Configure logging (use pytest's --log-cli-level for level control)
    setup_logging(
        level=logging.INFO,
        log_dir=log_dir,
        log_filename=log_filename,
        log_to_file=True,
        log_to_console=not no_log_console,
        log_file_header=header,
    )

    # Suppress noisy third-party loggers that aren't suppressed by setup_logging
    noisy_test_loggers = [
        "httpcore",
        "httpx",
        "urllib3",
        "requests",
        "asyncio",
    ]
    for logger_name in noisy_test_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    yield


@pytest.fixture(scope="session", autouse=True)
def _cleanup_tool_instances():
    """Final safety net — kill any stray ToolInstance workers at session end."""
    yield
    ToolInstance.clear_all()


# ============================================================================
# Persistent tool fixture factory
# ============================================================================
def make_persistent_fixture(tool_name: str, *, gpu: bool = True):
    """Create a module-scoped autouse fixture that wraps tests in persistence.

    Parameters
    ----------
    tool_name : str
        Tool name passed to ``ToolInstance.persist_tool()``.
    gpu : bool
        When *True* (default), the fixture skips persistence when no
        GPU is available: ``--cpu`` flag, ``CUDA_VISIBLE_DEVICES=""``,
        or ``nvidia-smi`` not found.  When *False* (CPU-only tools),
        persistence is always active.
    """

    @pytest.fixture(scope="module", autouse=True)
    def _persistent_tool(request):
        if gpu and (request.config.getoption("--cpu") or not _gpu_available()):
            yield
            return
        with ToolInstance.persist_tool(tool_name):
            yield

    return _persistent_tool
