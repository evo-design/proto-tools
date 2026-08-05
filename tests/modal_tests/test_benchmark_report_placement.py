"""Where benchmark reports land, and what a partial run does to an existing deployment README."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import (
    BenchmarkResult,
    _deployment_dir_for_tool,
    _merge_benchmark_readme,
    _split_readme_sections,
)

_DIR = _deployment_dir_for_tool("spliceai-predict")


def _result(tool_key: str, toolkit: str, duration: float = 1.0) -> BenchmarkResult:
    return BenchmarkResult(
        tool_key=tool_key,
        toolkit=toolkit,
        test_nodeid=f"tests/test_{toolkit}.py::test_{tool_key.replace('-', '_')}_benchmark",
        status="passed",
        duration_seconds=duration,
        error_message=None,
        backend_url="modal environment 'proto-env'",
        parametrize_summary=None,
        timestamp="2026-08-03T00:00:00Z",
        cold_seconds=duration,
        warm_seconds=duration / 2,
    )


def test_deployment_dir_follows_the_serving_app_not_the_toolkit_name() -> None:
    """The report belongs beside the service that serves the tool, and the two names differ."""
    target = _deployment_dir_for_tool("orfipy-prediction")
    assert target is not None
    assert target.name == "orf_deployment", "the orfipy toolkit is served by a differently named app"
    assert target.is_dir(), "resolved directory should exist in the tree"


def test_tools_sharing_an_app_resolve_to_one_directory() -> None:
    """A multi-tool app gets a single README, which is what makes merging necessary."""
    dirs = {_deployment_dir_for_tool(k) for k in ("spliceai-predict", "spliceai-score")}
    assert len(dirs) == 1


def test_tool_without_a_deployment_has_no_directory() -> None:
    """A local_only tool is skipped on a remote device, so it has no deployment cost to record."""
    assert _deployment_dir_for_tool("meme-fimo-scan") is None
    assert _deployment_dir_for_tool("not-a-real-tool") is None


def test_merge_preserves_a_sibling_that_did_not_run() -> None:
    """Benchmarking one tool of an app must not delete its siblings' numbers.

    This is the normal way of iterating on a single failure, so losing the rest would quietly
    destroy the report every time someone re-ran one tool.
    """
    first = _merge_benchmark_readme("", [_result("spliceai-predict", "spliceai", 10.0)], _DIR)
    both = _merge_benchmark_readme(first, [_result("spliceai-score", "spliceai", 20.0)], _DIR)

    assert "## `spliceai-predict`" in both
    assert "## `spliceai-score`" in both
    assert "10.00s" in both, "the untouched sibling keeps its original timing"


def test_merge_replaces_rather_than_appends_a_rerun_tool() -> None:
    """Re-running one tool updates its section instead of adding a second copy."""
    first = _merge_benchmark_readme("", [_result("spliceai-predict", "spliceai", 10.0)], _DIR)
    again = _merge_benchmark_readme(first, [_result("spliceai-predict", "spliceai", 99.0)], _DIR)

    assert again.count("## `spliceai-predict`") == 1
    assert "99.00s" in again
    assert "10.00s" not in again


def test_header_links_to_the_toolkit_readme() -> None:
    """A reader of the deployment report can reach what the tool actually does, and vice versa."""
    merged = _merge_benchmark_readme("", [_result("spliceai-predict", "spliceai")], _DIR)
    assert "[`spliceai` toolkit](" in merged
    link = merged.split("[`spliceai` toolkit](", 1)[1].split(")", 1)[0]
    assert (_DIR / link).resolve().is_file(), f"link should resolve to a real README, got {link}"


def test_header_is_regenerated_so_links_track_the_tools_present() -> None:
    """The header is derived, not preserved: a stale link would outlive the tool it points at."""
    stale = "# Benchmarks\n\nWhat these tools do: [`gone` toolkit](../../../tools/nope/gone/README.md)\n\n"
    merged = _merge_benchmark_readme(stale, [_result("spliceai-predict", "spliceai")], _DIR)
    assert "gone" not in merged
    assert "[`spliceai` toolkit](" in merged


def test_sections_are_ordered_so_the_file_does_not_churn() -> None:
    """Stable ordering keeps a re-run from producing a diff that is only reordering."""
    a = _merge_benchmark_readme(
        "", [_result("spliceai-score", "spliceai"), _result("spliceai-predict", "spliceai")], _DIR
    )
    b = _merge_benchmark_readme(
        "", [_result("spliceai-predict", "spliceai"), _result("spliceai-score", "spliceai")], _DIR
    )
    assert a == b


def test_split_round_trips_a_generated_readme() -> None:
    """Parsing what the writer produced must recover the same sections."""
    text = _merge_benchmark_readme(
        "", [_result("spliceai-predict", "spliceai"), _result("spliceai-score", "spliceai")], _DIR
    )
    _, sections = _split_readme_sections(text)
    assert set(sections) == {"spliceai-predict", "spliceai-score"}


@pytest.mark.parametrize("pattern", ["**/modal/*/*_deployment/README.md"])
def test_image_mount_excludes_the_reports(pattern: str) -> None:
    """The reports must not travel into every container image; nothing there reads them."""
    from proto_tools.modal.base_images import BENCHMARK_REPORT_PATTERNS

    assert pattern in BENCHMARK_REPORT_PATTERNS
    assert BENCHMARK_REPORT_PATTERNS, "an empty list would mount them again"
    # proto_tools/modal/README.md is the Modal setup guide, not a report, and must still ship.
    assert not any(p.endswith("modal/**/README.md") for p in BENCHMARK_REPORT_PATTERNS)


def test_toolkit_readmes_still_ship() -> None:
    """The exclusion is by path: docs extraction reads the toolkit READMEs and needs them packaged."""
    pyproject = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    assert '"*.md"' in pyproject, "toolkit READMEs must remain in package-data"
    assert "[tool.setuptools.exclude-package-data]" in pyproject


def test_deployment_packages_are_excluded_from_the_wheel() -> None:
    """Every deployment package must exclude its README, or benchmark reports ship to installers.

    setuptools requires literal package names in ``exclude-package-data`` — no patterns — so the
    list in pyproject cannot cover a new deployment automatically. This is what notices.
    """
    import tomllib

    from proto_tools.modal.manifest import SERVICE_TO_MODULE

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    excluded = tomllib.loads(pyproject.read_text())["tool"]["setuptools"]["exclude-package-data"]

    expected = {".".join(module.split(".")[:-1]) for module in SERVICE_TO_MODULE.values()}
    missing = sorted(expected - set(excluded))
    assert not missing, (
        f"Deployment packages missing from [tool.setuptools.exclude-package-data]: {missing}. "
        f'Add `"<package>" = ["README.md"]` for each, or its benchmark report ships in the wheel.'
    )
    stale = sorted(set(excluded) - expected)
    assert not stale, f"exclude-package-data names packages that are no longer deployments: {stale}"


def test_a_device_pinned_benchmark_is_not_labelled_with_the_remote_backend() -> None:
    """A pinned benchmark runs here even under --use-modal, so its report must not claim otherwise.

    The run-wide backend is set once from the flag, but a benchmark that pins ``device=`` bypasses
    the routing. Labelling it with the remote backend writes a measurement the deployment never
    made into that deployment's own README.
    """
    from tests.conftest import _PINNED_BACKEND_LABEL, DEVICE_PINNED_BENCHMARKS

    assert "spliceai-score" in DEVICE_PINNED_BENCHMARKS
    result = _result("spliceai-score", "spliceai")
    result.backend_url = _PINNED_BACKEND_LABEL
    rendered = _merge_benchmark_readme("", [result], _DIR)
    assert "modal environment" not in rendered
    assert "local" in rendered
