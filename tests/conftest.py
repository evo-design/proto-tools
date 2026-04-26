"""tests/conftest.py.

Supports the same CLI options and markers as the main proto-language tests:
  --cpu        Run only CPU tests
  --gpu        Run only GPU tests (skip CPU tests)
  --all        Include slow and GPU tests
  --slow       Run only slow tests
  --ext        Include extensive combinatorial tests (e.g., every tool x device). Long form: --extensive
  --skip-ci    Skip tests marked skip_ci (mimics CI)
  --no-log-console  Disable console logging during tests
  --env-report[=PATH]  Run venv smoke tests and generate compatibility report
                       (combine with -k to re-test specific tools incrementally)
"""

import functools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Register testing tools so consistency tests cover them.
# These are NOT exported via tools/__init__.py and are invisible outside tests.
import proto_tools.tools.testing  # noqa: F401
from proto_tools import setup_logging
from proto_tools.tools.tool_registry import ToolRegistry
from proto_tools.utils.device import number_of_visible_gpus
from proto_tools.utils.system_info import (
    capture_parent_env,
    collect_system_info,
    get_captured_env,
    get_platform_id,
)
from proto_tools.utils.tool_instance import ToolInstance


@functools.cache
def _gpu_available() -> bool:
    """Check if a GPU is likely available (without importing torch)."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() == "":
        return False
    return shutil.which("nvidia-smi") is not None


def _all_subclasses(cls: type) -> set[type]:
    """Recursively collect every subclass of ``cls``."""
    result: set[type] = set()
    for sub in cls.__subclasses__():
        result.add(sub)
        result.update(_all_subclasses(sub))
    return result


# ============================================================================
# Environment Report Collector
# ============================================================================
@dataclass
class ToolTestResult:
    """Result of a single tool smoke test."""

    tool_key: str
    category: str
    test_name: str
    status: str  # "passed", "failed", "skipped"
    duration_seconds: float
    uses_gpu: bool
    env_path: str | None
    env_status: str  # "success", "build_failed", "not_found"
    error_message: str | None
    git_commit: str | None = None
    git_dirty: bool = False


@dataclass
class EnvReportCollector:
    """Collects test results for environment compatibility report."""

    output_path: Path | None = None
    results: list[ToolTestResult] = field(default_factory=list)
    _test_start_times: dict[str, float] = field(default_factory=dict)
    selected_tools: set[str] = field(default_factory=set)
    expected_count: int = 0
    is_filtered: bool = False

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
        tool_key: str | None = None,
        category: str | None = None,
    ) -> None:
        """Record a test result.

        Args:
            nodeid (str): Pytest node ID for the test.
            outcome (str): Test outcome (``"passed"``, ``"failed"``, ``"skipped"``).
            has_gpu_marker (bool): Whether the test is marked with ``uses_gpu``.
            error_message (str | None): Error message if the test failed.
            tool_key (str | None): Tool key from the ``include_in_env_report`` marker.
            category (str | None): Tool category from the marker.
        """
        import time

        if not tool_key:
            return  # Marker must provide tool name

        # Calculate duration
        start_time = self._test_start_times.pop(nodeid, time.time())
        duration = time.time() - start_time

        if not category:
            category = "unknown"

        # Determine venv path and status
        env_path, env_status = self._get_venv_info(tool_key)

        # Capture git state at test time
        from proto_tools.utils.system_info import get_git_info

        git_info = get_git_info()

        self.results.append(
            ToolTestResult(
                tool_key=tool_key,
                category=category,
                test_name=nodeid,
                status=outcome,
                duration_seconds=round(duration, 2),
                uses_gpu=has_gpu_marker,
                env_path=env_path,
                env_status=env_status,
                error_message=error_message,
                git_commit=git_info.get("commit"),
                git_dirty=git_info.get("dirty", False),
            )
        )

    def _get_venv_info(self, tool_key: str) -> tuple[str | None, str]:
        """Get venv path and status for a tool.

        Args:
            tool_key (str): Registry key (e.g., ``"fampnn-sample"``).
        """
        from proto_tools.utils.tool_instance import ToolInstance

        venvs_dir = ToolInstance._get_tool_envs_root()

        # Map registry key → env name via ToolInstance, so shared-env tools
        # report the actual env path instead of a non-existent per-tool path.
        spec = ToolRegistry.get(tool_key)
        if not spec:
            return None, "not_found"
        try:
            _, env_name = ToolInstance._resolve_env_def(spec.source_file.parent.name)
        except ValueError:
            return None, "not_found"

        venv_dir = venvs_dir / f"{env_name}_env"
        if venv_dir.is_dir():
            python_path = venv_dir / "bin" / "python"
            if python_path.exists():
                return str(venv_dir), "success"
            return str(venv_dir), "build_failed"
        return None, "not_found"

    def _get_output_path(self) -> Path:
        """Resolve the output path for the report."""
        if self.output_path:
            path = self.output_path
            if path.suffix != ".md":
                path = path.with_suffix(".md")
            return path
        project_root = Path(__file__).parent.parent
        env_dir = project_root / "notes" / "environments"
        env_dir.mkdir(parents=True, exist_ok=True)
        platform_id = get_platform_id(include_date=False, include_commit=False)
        return env_dir / f"{platform_id}.md"

    @staticmethod
    def _load_embedded_data(path: Path) -> list[ToolTestResult]:
        """Load ToolTestResult entries from the embedded data block in a report."""
        if not path.exists():
            return []
        text = path.read_text()
        match = re.search(r"<!-- env-report-data\n(.*?)\n-->", text, re.DOTALL)
        if not match:
            return []
        try:
            raw = json.loads(match.group(1))
            return [ToolTestResult(**entry) for entry in raw]
        except (json.JSONDecodeError, TypeError):
            return []

    @staticmethod
    def _serialize_embedded_data(results: list[ToolTestResult]) -> str:
        """Serialize results as an HTML comment block."""
        data = [asdict(r) for r in results]
        return "<!-- env-report-data\n" + json.dumps(data, indent=2) + "\n-->"

    def finalize_and_write(self) -> Path | None:
        """Write the report to disk and return the path.

        Filtered runs (``-k``) merge new results into existing data.
        Full runs only write if all expected tests completed.
        """
        if not self.results:
            return None

        output_path = self._get_output_path()

        # Full run safety: don't overwrite with partial results
        if not self.is_filtered and len(self.results) < self.expected_count:
            logger = logging.getLogger("proto_tools.tests")
            logger.warning(
                f"--env-report: only {len(self.results)}/{self.expected_count} "
                f"tests completed — skipping report write to preserve existing data"
            )
            return None

        # Merge results
        if self.is_filtered:
            existing = self._load_embedded_data(output_path)
            # Build lookup: new results override by tool_key
            merged = {r.tool_key: r for r in existing}
            for r in self.results:
                merged[r.tool_key] = r
            all_results = list(merged.values())
        else:
            all_results = self.results

        # Collect system info
        system_info = collect_system_info()
        git_info = system_info["git_info"]
        platform = system_info["platform"]
        gpu = system_info["gpu"]
        parent_env = system_info["parent_process_env"]

        # Build summary (skipped tools excluded from pass rate)
        passed = sum(1 for r in all_results if r.status == "passed")
        failed = sum(1 for r in all_results if r.status == "failed")
        skipped = sum(1 for r in all_results if r.status == "skipped")
        tested = passed + failed

        # Group results by category
        by_category: dict[str, list[ToolTestResult]] = {}
        for r in all_results:
            by_category.setdefault(r.category, []).append(r)

        # Build README content
        lines: list[str] = []

        # Header with badges
        pass_rate = int(100 * passed / tested) if tested > 0 else 0
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
        elif "dgx" in platform["hostname"].lower() or "spark" in platform["hostname"].lower():
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
            lines.append(f"| **GPU** | {gpu['count']}x {devices} |")
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
            cat_tested = sum(1 for r in results if r.status != "skipped")

            # Category header with mini badge
            lines.append(f"### {category.replace('_', ' ').title()} ({cat_passed}/{cat_tested})")
            lines.append("")
            lines.append("| Tool | Requires GPU | Venv Build Succeeded | Duration | Tested At | Status |")
            lines.append("|------|--------------|----------------------|----------|-----------|--------|")

            for r in sorted(results, key=lambda x: x.tool_key):
                # Status emoji
                if r.status == "passed":
                    status = "✅ Pass"
                elif r.status == "failed":
                    status = "❌ Fail"
                else:
                    status = "⏭️ Skip"

                # Duration formatting
                duration = f"{r.duration_seconds:.1f}s" if r.status != "skipped" else "-"

                # GPU indicator
                gpu_req = "yes" if r.uses_gpu else "no"

                # Venv status
                if r.env_status == "success":
                    venv = "✅"
                elif r.env_status == "build_failed":
                    venv = "❌"
                else:
                    venv = "-"

                # Commit tag
                if r.git_commit:
                    commit_tag = f"`{r.git_commit[:7]}`"
                    if r.git_dirty:
                        commit_tag += " ✱"
                else:
                    commit_tag = "—"

                lines.append(f"| `{r.tool_key}` | {gpu_req} | {venv} | {duration} | {commit_tag} | {status} |")

            lines.append("")

        # Failure details section
        failures = [r for r in all_results if r.status == "failed" and r.error_message]
        if failures:
            lines.append("## Failure Details")
            lines.append("")

            for r in failures:
                lines.append(f"### ❌ `{r.tool_key}`")
                lines.append("")
                lines.append(f"**Test**: `{r.test_name}`")
                lines.append("")
                lines.append("```")
                lines.append(r.error_message)
                lines.append("```")
                lines.append("")

        # Footer
        lines.append("---")
        lines.append(f"*Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by `pytest --env-report`*")
        lines.append("")

        # Embedded data block (source of truth for merging)
        lines.append(self._serialize_embedded_data(all_results))

        # Write report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        return output_path


# Global collector instance (set in pytest_configure if --env-report is used)
_env_report_collector: EnvReportCollector | None = None


# ============================================================================
# Benchmark Report Collector
# ============================================================================
@dataclass
class BenchmarkResult:
    """Outcome of a single @pytest.mark.benchmark test."""

    tool_key: str
    toolkit: str
    test_nodeid: str
    status: str  # "passed", "failed", "skipped"
    duration_seconds: float
    error_message: str | None
    backend_url: str | None
    parametrize_summary: dict[str, str] | None  # callspec.params for parametrized tests
    timestamp: str  # ISO-8601 UTC


def _resolve_toolkit(tool_key: str) -> str:
    """Map a registered tool_key to its toolkit (parent directory) name.

    Falls back to the part before the first '-' if the registry doesn't know
    the key (e.g. test-only registration).
    """
    spec = ToolRegistry.get(tool_key)
    if spec is not None:
        return spec.source_file.parent.name
    return tool_key.split("-", 1)[0]


def _summarize_callspec(item: pytest.Item) -> dict[str, str] | None:
    """Compact one-line summary of pytest parametrize values, if any.

    Returns ``{"test_name": "trp_heterodimer", "predictor_name": "esmfold", ...}``
    so the report can identify which row inside a parametrized test was run.
    """
    callspec = getattr(item, "callspec", None)
    if callspec is None or not getattr(callspec, "params", None):
        return None
    return {k: repr(v) if not isinstance(v, str) else v for k, v in callspec.params.items()}


@dataclass
class BenchmarkReportCollector:
    """Collects benchmark test results and writes per-tool markdown reports.

    Layout: ``<output_dir>/{toolkit}/{tool_key}.md``. Each run overwrites the
    canonical file for that tool — no merging, no embedded JSON.

    The ``results`` mapping is keyed by ``tool_key`` and uses last-write-wins
    semantics. Today only one parametrize row per tool carries the benchmark
    marker, so this is unambiguous; if a future change marks two rows with the
    same tool_key, only the last to finish will appear in the report.
    """

    output_dir: Path
    backend_url: str | None = None
    results: dict[str, BenchmarkResult] = field(default_factory=dict)

    def record_result(
        self,
        item: pytest.Item,
        tool_key: str,
        outcome: str,
        duration_seconds: float,
        *,
        error_message: str | None = None,
    ) -> None:
        self.results[tool_key] = BenchmarkResult(
            tool_key=tool_key,
            toolkit=_resolve_toolkit(tool_key),
            test_nodeid=item.nodeid,
            status=outcome,
            duration_seconds=round(duration_seconds, 2),
            error_message=error_message,
            backend_url=self.backend_url,
            parametrize_summary=_summarize_callspec(item),
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    def write_reports(self) -> list[Path]:
        """Render one markdown file per recorded tool. Returns the paths written."""
        written: list[Path] = []
        for result in self.results.values():
            target = self.output_dir / result.toolkit / f"{result.tool_key}.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(_render_benchmark_markdown(result))
            written.append(target)
        return written


def _render_benchmark_markdown(result: BenchmarkResult) -> str:
    """Render a BenchmarkResult as a markdown report (one file per tool)."""
    if result.status == "passed":
        status_emoji, color = "✅", "brightgreen"
    elif result.status == "failed":
        status_emoji, color = "❌", "red"
    else:
        status_emoji, color = "⏭️", "lightgrey"

    backend_display = result.backend_url or "local (default device)"

    lines: list[str] = [
        f"# `{result.tool_key}`",
        "",
        f"![status](https://img.shields.io/badge/status-{result.status}-{color})",
        "",
        "| | |",
        "|---|---|",
        f"| **Toolkit** | `{result.toolkit}` |",
        f"| **Backend** | `{backend_display}` |",
        f"| **Test** | `{result.test_nodeid}` |",
        f"| **Duration** | {result.duration_seconds:.2f}s |",
        f"| **Status** | {status_emoji} {result.status} |",
        f"| **Run at** | {result.timestamp} |",
        "",
        "## Parametrize values",
        "",
    ]
    if result.parametrize_summary:
        lines += ["| Parameter | Value |", "|---|---|"]
        lines += [f"| `{k}` | `{v}` |" for k, v in result.parametrize_summary.items()]
    else:
        lines.append("_(test is not parametrized)_")

    if result.error_message:
        lines += ["", "## Error", "", "```", result.error_message, "```"]

    lines += ["", "---", f"*Generated by `pytest --benchmark-report` at {result.timestamp}.*", ""]
    return "\n".join(lines)


# Global collector (set in pytest_configure if --benchmark-report is used)
_benchmark_report_collector: BenchmarkReportCollector | None = None


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
        "--extensive",
        "--ext",
        action="store_true",
        default=False,
        dest="extensive",
        help="Include extensive combinatorial tests (e.g., every tool x device transition)",
    )
    parser.addoption(
        "--benchmark",
        action="store_true",
        default=False,
        help="Run only @pytest.mark.benchmark tests. In this mode the 'slow' gate is bypassed (no need "
        "to also pass --slow); hardware gates (uses_gpu, GPU count) are bypassed only when --use-cloud "
        "is also set. Outside this mode, marked tests are gated normally — they run under --all but "
        "are skipped under plain pytest if also marked slow / uses_gpu.",
    )
    parser.addoption(
        "--use-cloud",
        action="store_true",
        default=False,
        help="Route every tool run through device='cloud' (proto_tools.cloud.use_api_backend). "
        "Requires PROTO_API_KEY in the environment.",
    )
    parser.addoption(
        "--benchmark-report",
        default=None,
        help="Write per-tool benchmark markdown reports under the given directory "
        "(layout: <dir>/{toolkit}/{tool_key}.md). Implies --benchmark.",
    )
    parser.addoption(
        "--benchmark-tool",
        default=None,
        metavar="TOOL_KEY",
        help="Narrow --benchmark selection to a single tool by its marker arg "
        "(e.g. --benchmark-tool=esm2-embedding). Implies --benchmark.",
    )
    parser.addoption(
        "--benchmark-toolkit",
        default=None,
        metavar="TOOLKIT",
        help="Narrow --benchmark selection to one toolkit (e.g. --benchmark-toolkit=esm2 "
        "selects every benchmark whose tool_key resolves to the esm2 toolkit). Implies --benchmark.",
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
            "Combine with -k to re-test specific tools (only their envs are cleaned, "
            "results merge into existing report). Without -k, cleans all tool_envs/ "
            "and writes a fresh report. Optionally specify output path: "
            "--env-report=path/to/report.md"
        ),
    )


def pytest_configure(config):
    """Configure pytest with custom markers and options."""
    global _env_report_collector  # noqa: PLW0603 -- test infrastructure
    global _benchmark_report_collector  # noqa: PLW0603 -- test infrastructure

    config.addinivalue_line("markers", "uses_gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "uses_cpu: mark test as CPU-only")
    config.addinivalue_line(
        "markers",
        "include_in_env_report: Include test in --env-report. "
        "Requires kwargs: tool='tool-key', category='category_name'. "
        "Applied automatically by test_env_report.py parametrize",
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

    # Handle --benchmark-report (implies --benchmark via pytest_collection_modifyitems below)
    benchmark_report_opt = config.getoption("--benchmark-report")
    if benchmark_report_opt:
        backend_url = (
            os.environ.get("PROTO_TOOLS_BASE_URL", "<proto-client default>")
            if config.getoption("--use-cloud")
            else None
        )
        _benchmark_report_collector = BenchmarkReportCollector(
            output_dir=Path(benchmark_report_opt),
            backend_url=backend_url,
        )


def pytest_runtest_logstart(nodeid, location):
    """Log when a test starts (DEBUG level, file only)."""
    logger = logging.getLogger("proto_tools.tests")
    logger.debug(f"TEST START: {nodeid}")

    # Record start time for env report
    if _env_report_collector is not None:
        _env_report_collector.record_start(nodeid)


def pytest_runtest_logreport(report):
    """Log test results (DEBUG level to avoid console output)."""
    logger = logging.getLogger("proto_tools.tests")

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
        tool_key = None
        category = None
        marker = item.get_closest_marker("include_in_env_report")
        if marker:
            tool_key = marker.kwargs.get("tool")
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
                tool_key=tool_key,
                category=category,
            )
        elif report.when == "setup" and report.failed:
            # Setup failure
            _env_report_collector.record_result(
                item.nodeid,
                "failed",
                has_gpu_marker="uses_gpu" in item.keywords,
                error_message=f"Setup failed: {report.longrepr}",
                tool_key=tool_key,
                category=category,
            )
        elif report.when == "setup" and report.skipped:
            # Skipped via marker (e.g., insufficient GPUs)
            _env_report_collector.record_result(
                item.nodeid,
                "skipped",
                has_gpu_marker="uses_gpu" in item.keywords,
                error_message=str(report.longrepr) if report.longrepr else None,
                tool_key=tool_key,
                category=category,
            )

    # Benchmark report: capture per-tool outcome.
    # On the call phase: record pass/fail/skip with the call duration.
    # On setup-phase failures or skips (e.g. missing weights, insufficient GPUs):
    # record so the report shows "this tool was selected but didn't run", instead of
    # silently leaving the file out.
    if _benchmark_report_collector is not None and "benchmark" in item.keywords:
        marker = item.get_closest_marker("benchmark")
        if marker and marker.args:
            tool_key = marker.args[0]
            longrepr = str(report.longrepr) if report.longrepr else None
            if report.when == "call":
                if report.passed:
                    status, err = "passed", None
                elif report.skipped:
                    status, err = "skipped", longrepr
                else:
                    status, err = "failed", longrepr
                _benchmark_report_collector.record_result(item, tool_key, status, call.duration, error_message=err)
            elif report.when == "setup" and (report.failed or report.skipped):
                status = "skipped" if report.skipped else "failed"
                _benchmark_report_collector.record_result(item, tool_key, status, call.duration, error_message=longrepr)


def pytest_sessionfinish(session, exitstatus):
    """Log test session summary at the end and write env report if requested."""
    logger = logging.getLogger("proto_tools.tests")

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
                summary_lines.extend(f"  - {report.nodeid}" for report in failed_reports)

            # Add list of error tests if any
            if errors > 0:
                summary_lines.append("\nTests with errors:")
                error_reports = stats.get("error", [])
                summary_lines.extend(f"  - {report.nodeid}" for report in error_reports)

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

    # Write benchmark reports if collector is active
    if _benchmark_report_collector is not None:
        for path in _benchmark_report_collector.write_reports():
            logger.info(f"--benchmark-report: wrote {path}")
            print(f"[benchmark-report] {path}")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options and auto-mark tests."""
    # --env-report: keep only env-report smoke tests, deselect everything else
    if config.getoption("--env-report"):
        skip_no_gpu = pytest.mark.skip(reason="--env-report: GPU not available")
        gpu_available = _gpu_available()
        visible_gpus = number_of_visible_gpus() if gpu_available else 0

        selected = []
        deselected = []
        for item in items:
            if "include_in_env_report" not in item.keywords:
                deselected.append(item)
            elif "uses_gpu" in item.keywords and not gpu_available:
                # Skip GPU tests on platforms without GPU, but still include in report
                item.add_marker(skip_no_gpu)
                selected.append(item)
            else:
                # Check multi-GPU requirements (e.g., uses_gpu(2) on a 1-GPU machine)
                for marker in item.iter_markers("uses_gpu"):
                    required = marker.args[0] if marker.args else 1
                    if visible_gpus < required:
                        item.add_marker(
                            pytest.mark.skip(
                                reason=f"--env-report: requires {required} GPUs, only {visible_gpus} visible"
                            )
                        )
                        break
                selected.append(item)
        items[:] = selected
        if deselected:
            config.hook.pytest_deselected(items=deselected)

        # Detect -k filtering. Note: pytest applies -k AFTER this hook,
        # so we detect it from the config option. The actual selected_tools
        # set is populated in the _env_report_clean_envs fixture which runs
        # after all collection hooks (including -k) have completed.
        if _env_report_collector is not None:
            k_expr = config.getoption("-k", default=None)
            _env_report_collector.is_filtered = bool(k_expr)
        return

    # Normal mode: deselect env-report-only tests entirely
    selected = []
    deselected = []
    for item in items:
        if "include_in_env_report" in item.keywords:
            deselected.append(item)
        else:
            selected.append(item)
    if deselected:
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)

    # Auto-mark all tests as CPU-only unless explicitly marked as GPU
    for item in items:
        # If no GPU marker found, mark as CPU
        if not any(mark.name == "uses_gpu" for mark in item.iter_markers()):
            item.add_marker(pytest.mark.uses_cpu)

    # Skip tests marked with skip_ci when running in GitHub Actions or --skip-ci is specified
    if os.getenv("GITHUB_ACTIONS") == "true" or config.getoption("--skip-ci"):
        skip_ci = pytest.mark.skip(reason="Skipped in CI environment (GitHub Actions or --skip-ci)")
        for item in items:
            if "skip_ci" in item.keywords:
                item.add_marker(skip_ci)

    # Two independent carve-outs:
    # - --use-cloud bypasses every hardware-availability gate, since the GPUs
    #   live on the server. Applies to any selected test, not just benchmarks.
    # - @pytest.mark.benchmark bypasses the default 'slow' gate so benchmark
    #   tests still run when invoked under --benchmark, even if they happen
    #   to also be marked slow. Hardware gates (gpu count) are NOT bypassed
    #   by the benchmark marker alone — those still need --use-cloud.
    use_cloud = config.getoption("--use-cloud")

    # Skip GPU tests when --cpu is specified
    if config.getoption("--cpu"):
        skip_gpu = pytest.mark.skip(reason="--cpu specified")
        for item in items:
            if "uses_gpu" in item.keywords and not use_cloud:
                item.add_marker(skip_gpu)

    # Skip CPU tests when --gpu is specified
    elif config.getoption("--gpu"):
        skip_cpu = pytest.mark.skip(reason="--gpu specified")
        for item in items:
            if "uses_cpu" in item.keywords and "uses_gpu" not in item.keywords:
                item.add_marker(skip_cpu)

    # Default: skip GPU tests unless --integration (with GPU available) or --all
    elif not (config.getoption("--all") or (config.getoption("--integration") and _gpu_available())):
        skip_gpu = pytest.mark.skip(reason="GPU test (use --gpu, --integration with GPU, or --all to run)")
        for item in items:
            if "uses_gpu" in item.keywords and not use_cloud:
                item.add_marker(skip_gpu)

    # Handle slow test filtering
    run_all = config.getoption("--all")
    run_slow_only = config.getoption("--slow")
    benchmark_mode = config.getoption("--benchmark") or config.getoption("--benchmark-report")

    if run_slow_only:
        # When --slow is specified, skip tests NOT marked as slow
        skip_non_slow = pytest.mark.skip(reason="--slow specified, skipping non-slow tests")
        for item in items:
            if "slow" not in item.keywords:
                item.add_marker(skip_non_slow)
    elif not run_all and not config.getoption("extensive") and not benchmark_mode:
        # By default, skip slow tests. --all and --ext bypass this; --benchmark
        # also bypasses (benchmark mode is opt-in to running slow workloads).
        skip_slow = pytest.mark.skip(reason="slow test (use --all to run, or --slow to run only slow tests)")
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

    # Skip extensive tests unless --ext (or --extensive) is specified
    if not config.getoption("extensive"):
        skip_extensive = pytest.mark.skip(reason="extensive test (use --ext to run)")
        for item in items:
            if "extensive" in item.keywords:
                item.add_marker(skip_extensive)

    # --benchmark: keep only @pytest.mark.benchmark tests. Without the flag,
    # benchmark-marked tests still run normally — the marker is purely a selection signal.
    # --benchmark-report / --benchmark-tool / --benchmark-toolkit all imply --benchmark.
    tool_filter = config.getoption("--benchmark-tool")
    toolkit_filter = config.getoption("--benchmark-toolkit")
    if config.getoption("--benchmark") or config.getoption("--benchmark-report") or tool_filter or toolkit_filter:
        selected, deselected = [], []
        for item in items:
            if "benchmark" not in item.keywords:
                deselected.append(item)
                continue
            if tool_filter or toolkit_filter:
                marker = item.get_closest_marker("benchmark")
                tool_key = marker.args[0] if marker and marker.args else None
                if tool_filter and tool_key != tool_filter:
                    deselected.append(item)
                    continue
                if toolkit_filter and (tool_key is None or _resolve_toolkit(tool_key) != toolkit_filter):
                    deselected.append(item)
                    continue
            selected.append(item)
        if deselected:
            items[:] = selected
            config.hook.pytest_deselected(items=deselected)

    # Skip test_on_platforms tests when current architecture doesn't match
    import platform as _platform

    current_arch = _platform.machine()
    for item in items:
        for marker in item.iter_markers("test_on_platforms"):
            allowed = marker.args
            if current_arch not in allowed:
                item.add_marker(pytest.mark.skip(reason=f"Requires platform {allowed}, current is {current_arch}"))

    # Skip uses_gpu(n) tests when fewer than n GPUs are visible — bypassed
    # under --use-cloud (the GPUs live on the server).
    visible_gpus = number_of_visible_gpus()
    for item in items:
        if use_cloud:
            continue
        for marker in item.iter_markers("uses_gpu"):
            required = marker.args[0] if marker.args else 1
            if visible_gpus < required:
                item.add_marker(pytest.mark.skip(reason=f"Requires {required} GPUs, only {visible_gpus} visible"))


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging(request):
    """Set up logging for the test session. Runs early to prevent timestamped log files."""
    # Use same log directory as application logs (logs/ in project root)
    project_root = Path(__file__).parent.parent
    log_dir = os.environ.get("PROTO_LOG_DIR", str(project_root / "logs"))

    # Get options from command line
    no_log_console = request.config.getoption("--no-log-console")
    k_expression = request.config.getoption("-k", default=None)

    # Clear any existing handlers first to prevent duplicate log files
    proto_tools_logger = logging.getLogger("proto_tools")
    proto_tools_logger.handlers.clear()

    # Create header with pytest command and timestamp
    pytest_command = " ".join(sys.argv)
    now = datetime.now()
    timestamp = now.strftime("%H:%M:%S")
    datestamp = now.strftime("%m/%d/%Y")
    header = f"Pytest Run Command: `{pytest_command}`\nRun Started: {timestamp} on {datestamp}\n{'=' * 80}\n\n"

    # Create log filename — env-report gets a distinct prefix
    env_report = request.config.getoption("--env-report")
    if env_report:
        file_timestamp = now.strftime("%Y%m%d_%H%M%S")
        log_filename = f"pytest_env_report_{file_timestamp}.log"
    elif k_expression:
        # Sanitize the -k expression to make it filename-safe
        # Replace spaces with underscores, remove special characters
        sanitized = re.sub(r"[^\w\s-]", "", k_expression)  # Remove special chars except spaces, hyphens, underscores
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
def _env_report_clean_envs(request, setup_test_logging):
    """Clean tool envs when --env-report is active (after logging is configured).

    When ``-k`` filtering is active, only deletes envs for selected tools.
    Otherwise deletes the entire ``tool_envs/`` directory for a full rebuild.
    """
    if _env_report_collector is None:
        return

    # Populate selected_tools from the session's final item list (after -k
    # filtering has been applied by pytest's built-in hook).
    for item in request.session.items:
        marker = item.get_closest_marker("include_in_env_report")
        if marker:
            tool = marker.kwargs.get("tool")
            if tool:
                _env_report_collector.selected_tools.add(tool)
    _env_report_collector.expected_count = len(_env_report_collector.selected_tools)

    logger = logging.getLogger("proto_tools.tests")
    venvs_dir = ToolInstance._get_tool_envs_root()
    if not venvs_dir.exists():
        return

    if _env_report_collector.is_filtered:
        # Selective cleanup: only delete envs for the tools being re-tested.
        # Resolve via ToolInstance so shared-env tools target the right physical
        # env (and naturally dedupe — two siblings of one shared env clean once).
        env_names_to_clean: set[str] = set()
        for key in _env_report_collector.selected_tools:
            spec = ToolRegistry.get(key)
            if not spec:
                continue
            try:
                _, env_name = ToolInstance._resolve_env_def(spec.source_file.parent.name)
            except ValueError:
                continue
            env_names_to_clean.add(env_name)
        for env_name in env_names_to_clean:
            env_dir = venvs_dir / f"{env_name}_env"
            if env_dir.exists():
                logger.warning(f"Cleaning {env_dir} for rebuild...")
                subprocess.run(
                    ["rm", "-rf", str(env_dir)],
                    check=False,
                    capture_output=True,
                )
    else:
        # Full run: clean everything
        logger.warning(
            f"Cleaning {venvs_dir} for fresh environment rebuilds (this may take a while on network filesystems)..."
        )
        subprocess.run(
            ["rm", "-rf", str(venvs_dir)],
            check=False,
            capture_output=True,
        )
        logger.warning(f"Finished cleaning {venvs_dir}")


@pytest.fixture(scope="session", autouse=True)
def _route_tests_to_cloud(request):
    """Arm proto_tools.cloud and patch BaseConfig.device default to 'cloud' when --use-cloud is set.

    Lets a test that already passes ``Config()`` without an explicit ``device=`` run
    locally (default ``"cpu"``) or against Proto's remote execution service
    (``--use-cloud``) without any change to the test body.
    """
    if not request.config.getoption("--use-cloud"):
        yield
        return

    from proto_tools.cloud import disable_api_backend, use_api_backend
    from proto_tools.utils.base_config import BaseConfig

    use_api_backend()

    all_classes = {BaseConfig} | _all_subclasses(BaseConfig)
    originals = {}
    for cls in all_classes:
        fi = cls.model_fields.get("device")
        if fi is not None:
            originals[cls] = fi.default
            fi.default = "cloud"
    for cls in all_classes:
        cls.model_rebuild(force=True)

    yield

    for cls, original in originals.items():
        fi = cls.model_fields.get("device")
        if fi is not None:
            fi.default = original
    for cls in all_classes:
        cls.model_rebuild(force=True)
    disable_api_backend()


@pytest.fixture(scope="session", autouse=True)
def _force_verbose_tools():
    """Force verbose=True on all tool configs so subprocess stderr streams into test logs."""
    from proto_tools.utils.base_config import BaseConfig

    all_classes = {BaseConfig} | _all_subclasses(BaseConfig)

    originals = {}
    for cls in all_classes:
        fi = cls.model_fields.get("verbose")
        if fi is not None:
            originals[cls] = fi.default
            fi.default = True

    for cls in all_classes:
        cls.model_rebuild(force=True)

    old_env_verbose = os.environ.get("PROTO_ENV_VERBOSE")
    os.environ["PROTO_ENV_VERBOSE"] = "1"

    yield

    for cls, original in originals.items():
        fi = cls.model_fields.get("verbose")
        if fi is not None:
            fi.default = original
    for cls in all_classes:
        cls.model_rebuild(force=True)

    if old_env_verbose is None:
        os.environ.pop("PROTO_ENV_VERBOSE", None)
    else:
        os.environ["PROTO_ENV_VERBOSE"] = old_env_verbose


@pytest.fixture(scope="session", autouse=True)
def _cleanup_tool_instances():
    """Final safety net: kill any stray ToolInstance workers at session end."""
    yield
    ToolInstance.clear_all()


# ============================================================================
# Persistent tool fixture factory
# ============================================================================
def make_persistent_fixture(toolkit: str, *, gpu: bool = True):
    """Create a module-scoped autouse fixture that wraps tests in persistence.

    Parameters
    ----------
    toolkit : str
        Worker group passed to ``ToolInstance.persist_tool()`` (e.g.
        ``"pyrosetta"``, ``"esm2"``). Accepts a registered tool_key too
        (it will be normalized).
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
        with ToolInstance.persist_tool(toolkit):
            yield

    return _persistent_tool
