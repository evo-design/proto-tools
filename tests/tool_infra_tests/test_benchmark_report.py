"""tests/tool_infra_tests/test_benchmark_report.py.

Unit tests for the benchmark report collector wired into ``tests/conftest.py``:
the parametrize summarizer, the per-tool markdown renderer, and the
record-then-write loop. Pytest hooks themselves are exercised end-to-end the
first time a benchmark test runs with ``--benchmark-report`` against a backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from tests.conftest import (
    BenchmarkReportCollector,
    BenchmarkResult,
    _render_benchmark_markdown,
    _summarize_callspec,
)


@dataclass
class _FakeItem:
    nodeid: str
    callspec: object | None = None


def test_summarize_callspec_returns_none_when_no_callspec():
    item = _FakeItem(nodeid="tests/x.py::test_foo")
    assert _summarize_callspec(item) is None


def test_summarize_callspec_returns_string_values_verbatim():
    callspec = SimpleNamespace(params={"test_name": "trp_heterodimer", "predictor": "esmfold"})
    item = _FakeItem(nodeid="tests/x.py::test_foo[...]", callspec=callspec)
    summary = _summarize_callspec(item)
    assert summary == {"test_name": "trp_heterodimer", "predictor": "esmfold"}


def test_summarize_callspec_reprs_non_strings():
    callspec = SimpleNamespace(params={"use_msa": False, "n_samples": 5})
    item = _FakeItem(nodeid="tests/x.py::test_foo[...]", callspec=callspec)
    summary = _summarize_callspec(item)
    assert summary == {"use_msa": "False", "n_samples": "5"}


def test_render_benchmark_markdown_passed():
    result = BenchmarkResult(
        tool_key="esmfold-prediction",
        toolkit="esmfold",
        test_nodeid="tests/x.py::test_folding[trp_heterodimer-esmfold-without_msa]",
        status="passed",
        duration_seconds=33.21,
        error_message=None,
        backend_url="https://staging.example.com",
        parametrize_summary={"test_name": "trp_heterodimer", "predictor_name": "esmfold"},
        timestamp="2026-04-25T19:00:00Z",
    )
    md = _render_benchmark_markdown(result)
    assert "# `esmfold-prediction`" in md
    assert "status-passed" in md
    assert "33.21s" in md
    assert "https://staging.example.com" in md
    assert "trp_heterodimer" in md
    assert "## Error" not in md  # passing tests omit the error section


def test_render_benchmark_markdown_failed_includes_error():
    result = BenchmarkResult(
        tool_key="boltz2-prediction",
        toolkit="boltz2",
        test_nodeid="tests/x.py::test_folding[MfnG_and_ligand-boltz2-without_msa]",
        status="failed",
        duration_seconds=2.10,
        error_message="server returned 500",
        backend_url=None,
        parametrize_summary=None,
        timestamp="2026-04-25T19:00:00Z",
    )
    md = _render_benchmark_markdown(result)
    assert "status-failed" in md
    assert "## Error" in md
    assert "server returned 500" in md
    assert "local (default device)" in md
    assert "_(test is not parametrized)_" in md


def test_collector_records_and_writes(tmp_path):
    """Record a synthetic result and verify the file lands at the right path with the right content."""
    collector = BenchmarkReportCollector(output_dir=tmp_path, backend_url=None)
    item = _FakeItem(nodeid="tests/x.py::test_benchmark_esmfold")
    collector.record_result(item, "esmfold-prediction", "passed", 1.23)
    written = collector.write_reports()

    assert len(written) == 1
    path = written[0]
    assert path == tmp_path / "esmfold" / "esmfold-prediction.md"
    text = path.read_text()
    assert "# `esmfold-prediction`" in text
    assert "passed" in text
    assert "1.23s" in text


def test_collector_overwrites_existing_report(tmp_path):
    """A second run for the same tool overwrites the prior file (no append, no merge)."""
    collector = BenchmarkReportCollector(output_dir=tmp_path, backend_url=None)
    item = _FakeItem(nodeid="tests/x.py::test_benchmark_esmfold")
    collector.record_result(item, "esmfold-prediction", "passed", 1.0)
    collector.write_reports()

    collector.record_result(item, "esmfold-prediction", "failed", 0.5, error_message="timeout")
    collector.write_reports()

    text = (tmp_path / "esmfold" / "esmfold-prediction.md").read_text()
    assert "status-failed" in text
    assert "timeout" in text
    assert "status-passed" not in text


def test_collector_setup_skip_recorded_with_skipped_status(tmp_path):
    """A test skipped at setup phase (e.g. missing weights) still produces a report entry."""
    collector = BenchmarkReportCollector(output_dir=tmp_path, backend_url=None)
    item = _FakeItem(nodeid="tests/x.py::test_benchmark_alphafold3")
    collector.record_result(
        item,
        "alphafold3-prediction",
        "skipped",
        0.01,
        error_message="AlphaFold3 weights not found",
    )
    written = collector.write_reports()
    assert len(written) == 1
    text = written[0].read_text()
    assert "status-skipped" in text
    assert "AlphaFold3 weights not found" in text
