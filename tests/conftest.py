"""
Test configuration for bio_programming_tools test suite.

Supports the same CLI options and markers as the main bio-programming tests:
  --cpu        Run only CPU tests
  --gpu        Run only GPU tests
  --all        Include slow tests
  --slow       Run only slow tests
  --skip-ci    Skip tests marked skip_ci (mimics CI)
  --no-log-console  Disable console logging during tests
"""

import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pytest

from bio_programming_tools import setup_logging


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
        help="Run all tests including slow tests",
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
        "--no-log-console",
        action="store_true",
        default=False,
        help="Disable console logging during tests",
    )


def pytest_configure(config):
    """Configure pytest with custom markers and options."""
    config.addinivalue_line("markers", "uses_gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "uses_cpu: mark test as CPU-only")

    # Set environment variable to indicate we're in pytest
    # This prevents setup_logging() from creating timestamped files during test imports
    os.environ["PYTEST_RUNNING"] = "1"

    # Hide CUDA devices when --skip-ci is specified to simulate CI environment
    if config.getoption("--skip-ci"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def pytest_runtest_logstart(nodeid, location):
    """Log when a test starts (DEBUG level, file only)."""
    logger = logging.getLogger("bio_programming_tools.tests")
    logger.debug(f"TEST START: {nodeid}")


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


def pytest_sessionfinish(session, exitstatus):
    """Log test session summary at the end."""
    logger = logging.getLogger("bio_programming_tools.tests")

    # Get test statistics from the session
    test_reports = session.items
    num_collected = len(test_reports)

    # Count passed and failed tests from the terminal reporter
    if hasattr(session.config, 'pluginmanager'):
        terminalreporter = session.config.pluginmanager.get_plugin('terminalreporter')
        if terminalreporter:
            stats = terminalreporter.stats

            passed = len(stats.get('passed', []))
            failed = len(stats.get('failed', []))
            skipped = len(stats.get('skipped', []))
            errors = len(stats.get('error', []))

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
                failed_reports = stats.get('failed', [])
                for report in failed_reports:
                    summary_lines.append(f"  - {report.nodeid}")

            # Add list of error tests if any
            if errors > 0:
                summary_lines.append("\nTests with errors:")
                error_reports = stats.get('error', [])
                for report in error_reports:
                    summary_lines.append(f"  - {report.nodeid}")

            summary_lines.append("=" * 80)

            # Log the summary at INFO level so it appears in both console and file
            summary_message = "\n".join(summary_lines)
            logger.info(summary_message)


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options and auto-mark tests."""
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
    elif not run_all:
        # By default (no --all flag), skip slow tests
        skip_slow = pytest.mark.skip(
            reason="slow test (use --all to run, or --slow to run only slow tests)"
        )
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging(request):
    """Set up logging for the test session. Runs early to prevent timestamped log files."""
    # Use same log directory as application logs (logs/ in project root)
    project_root = Path(__file__).parent.parent
    log_dir = os.environ.get(
        "BIO_PROGRAMMING_TOOLS_LOG_DIR",
        str(project_root / "logs")
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
        sanitized = re.sub(r'[^\w\s-]', '', k_expression)  # Remove special chars except spaces, hyphens, underscores
        sanitized = re.sub(r'\s+', '_', sanitized)  # Replace spaces with underscores
        sanitized = sanitized.strip('_')  # Remove leading/trailing underscores
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
