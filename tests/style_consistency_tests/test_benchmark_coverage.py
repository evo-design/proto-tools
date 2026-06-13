"""tests/style_consistency_tests/test_benchmark_coverage.py.

Consistency test: every registered tool must have a benchmark test.

A benchmark is a ``@pytest.mark.benchmark("<tool-key>")`` test exercising the
tool on a realistic workload. This guard fails when a tool is registered without
one, so coverage cannot silently regress.

``_KNOWN_MISSING`` is a shrinking ratchet of tools that predate this requirement.
Delete an entry the moment you add that tool's benchmark — the test enforces that
the list stays exact (no stale entries, no entry that is now covered).
"""

import ast
import pathlib

from proto_tools.tools.tool_registry import ToolRegistry

_TESTS_ROOT = pathlib.Path(__file__).parent.parent

# Categories exempt from the benchmark requirement:
#   testing            — mock tools that ship no real workload.
#   database_retrieval — thin wrappers over external APIs; a benchmark would
#                        measure the remote service's latency, not our code.
_EXEMPT_CATEGORIES = frozenset({"testing", "database_retrieval"})

_EXEMPT_TOOLS = frozenset(
    {
        "blast-create-db",
        "bindcraft-design",
        "freebindcraft-design",
        "germinal-design",
        "mmseqs2-homology-search",
    }
)

# Tools registered before the benchmark requirement existed. Remove an entry
# when you add its benchmark; do NOT add new entries — write the benchmark
# instead. The test below keeps this set honest.
_KNOWN_MISSING = frozenset(
    {
        "mmseqs2-clustering",
        "mmseqs2-search-genomes",
        "mmseqs2-search-proteins",
        "pyhmmer-hmmscan",
        "pyhmmer-hmmsearch",
        "pyhmmer-jackhmmer",
        "pyhmmer-nhmmer",
        "pyhmmer-phmmer",
        "pymol-rmsd-alignment",
        "rf3-prediction",
        "spliceai-predict",
        "spliceai-score",
    }
)


def _benchmarked_tool_keys() -> set[str]:
    """Tool keys carrying a ``@pytest.mark.benchmark("<key>")`` test.

    AST scan rather than pytest collection so the guard runs fast and CPU-only,
    without importing every tool's heavy test module.
    """
    keys: set[str] = set()
    for path in _TESTS_ROOT.rglob("test_*.py"):
        tree = ast.parse(path.read_text(), str(path))
        for node in ast.walk(tree):
            for dec in getattr(node, "decorator_list", []):
                if not isinstance(dec, ast.Call):
                    continue
                func = dec.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "benchmark"
                    and isinstance(func.value, ast.Attribute)
                    and func.value.attr == "mark"
                    and dec.args
                    and isinstance(dec.args[0], ast.Constant)
                    and isinstance(dec.args[0].value, str)
                ):
                    keys.add(dec.args[0].value)
    return keys


def _registered_tool_keys() -> set[str]:
    return {
        spec.key
        for spec in ToolRegistry.list_all()
        if spec.category not in _EXEMPT_CATEGORIES and spec.key not in _EXEMPT_TOOLS
    }


def test_consistency_all_tools_have_benchmark() -> None:
    """Every registered tool must have a benchmark, except the shrinking ratchet."""
    registered = _registered_tool_keys()
    benchmarked = _benchmarked_tool_keys()

    new_gaps = sorted(registered - benchmarked - _KNOWN_MISSING)
    assert not new_gaps, (
        f"Tools registered without a benchmark test: {new_gaps}. Add a "
        '@pytest.mark.benchmark("<tool-key>") test exercising the tool on a '
        "realistic workload. Do not add the tool to _KNOWN_MISSING."
    )


def test_known_missing_is_exact() -> None:
    """The ratchet may not drift: no stale entries, no already-covered entries."""
    registered = _registered_tool_keys()
    benchmarked = _benchmarked_tool_keys()

    unregistered = sorted(_KNOWN_MISSING - registered)
    assert not unregistered, (
        f"_KNOWN_MISSING names tools that are no longer registered: {unregistered}. "
        "Remove them (likely renamed or deleted)."
    )

    now_covered = sorted(_KNOWN_MISSING & benchmarked)
    assert not now_covered, (
        f"_KNOWN_MISSING lists tools that now have a benchmark: {now_covered}. "
        "Delete these entries — the requirement is met for them."
    )

    # Keep the per-tool exemption list honest too.
    all_registered = {spec.key for spec in ToolRegistry.list_all()}
    stale_exempt = sorted(_EXEMPT_TOOLS - all_registered)
    assert not stale_exempt, (
        f"_EXEMPT_TOOLS names tools that are no longer registered: {stale_exempt}. "
        "Remove them (likely renamed or deleted)."
    )
    double_listed = sorted(_EXEMPT_TOOLS & _KNOWN_MISSING)
    assert not double_listed, (
        f"Tools are both exempt and in _KNOWN_MISSING: {double_listed}. "
        "An exempt tool needs no benchmark — remove it from _KNOWN_MISSING."
    )


def test_benchmark_keys_are_registered() -> None:
    """Every benchmark marker must name a real registered tool (catches typos)."""
    registered = _registered_tool_keys()
    benchmarked = _benchmarked_tool_keys()

    unknown = sorted(benchmarked - registered)
    assert not unknown, (
        f"@pytest.mark.benchmark names that match no registered tool: {unknown}. "
        "Fix the marker argument to the tool's registry key."
    )
