"""tests/_gpu_shards.py.

Split one test session across several GPUs without two processes ever sharing a toolkit.

A GPU run spends most of its time loading one toolkit at a time on one GPU while the others
sit idle. ``scripts/run_tests_multi_gpu.py`` runs it in two phases instead:

1. **Parallel**: everything except the serial tests runs as N shards, one pytest per GPU, each
   seeing only its own GPU. A shard holds whole toolkits, so a toolkit's env and weights are only
   ever loaded by one process, and tests that run no tool stay together in a single shard.
2. **Serial**: tests that need more than one GPU, and the DeviceManager tests (which reset the
   process's DeviceManager and lease GPUs themselves), then run once with every GPU visible.

Shards are balanced longest-first on per-test durations from an earlier run's JUnit XML
(``python -m tests._gpu_shards weights <junit.xml> <out.json>``), falling back to a rough
estimate per test. Every collected test lands in exactly one phase and, in the parallel phase,
in exactly one shard.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import pytest

#: Tests under this directory run in the serial phase: they reset DeviceManager and lease GPUs themselves.
SERIAL_DIRS = ("tests/tool_infra_tests/test_device_manager/",)

#: The shard unit for tests that run no tool.
NO_TOOLKIT = "-"

# Fallback cost per test, in seconds, when no weights file covers it.
_GPU_TEST_SECONDS = 60.0
_OTHER_TEST_SECONDS = 0.1


def required_gpus(item: pytest.Item) -> int:
    """GPUs a test needs: the largest ``uses_gpu(n)`` count (1 for bare ``uses_gpu``, 0 if unmarked)."""
    return max((marker.args[0] if marker.args else 1 for marker in item.iter_markers("uses_gpu")), default=0)


def is_serial(item: pytest.Item) -> bool:
    """Whether the test must run with every GPU visible, outside the per-GPU shards."""
    return required_gpus(item) >= 2 or item.nodeid.startswith(SERIAL_DIRS)


def shard_unit(item: pytest.Item) -> str:
    """The unit shards are made of: the test's toolkit, or one shared unit for tests that run none."""
    from tests import _warm_workers

    return _warm_workers.toolkit_of(item) or NO_TOOLKIT


def _estimate(item: pytest.Item, weights: dict[str, float]) -> float:
    if item.nodeid in weights:
        return weights[item.nodeid]
    return _GPU_TEST_SECONDS if required_gpus(item) else _OTHER_TEST_SECONDS


def assign_shards(items: list[pytest.Item], count: int, weights: dict[str, float]) -> dict[str, int]:
    """Map each unit to a shard, longest first onto the lightest shard. Deterministic for given inputs."""
    cost: dict[str, float] = defaultdict(float)
    for item in items:
        cost[shard_unit(item)] += _estimate(item, weights)
    loads = [0.0] * count
    assignment: dict[str, int] = {}
    for unit in sorted(cost, key=lambda u: (-cost[u], u)):
        shard = min(range(count), key=lambda s: (loads[s], s))
        assignment[unit] = shard
        loads[shard] += cost[unit]
    return assignment


def parse_shard(value: str) -> tuple[int, int]:
    """``"K/N"`` -> ``(K, N)`` with ``0 <= K < N``."""
    try:
        index, count = (int(part) for part in value.split("/"))
    except ValueError as e:
        raise pytest.UsageError(f"--shard expects K/N (e.g. 0/2), got {value!r}") from e
    if not 0 <= index < count:
        raise pytest.UsageError(f"--shard {value}: K must be in [0, N)")
    return index, count


def load_weights(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    return {str(k): float(v) for k, v in json.loads(Path(path).read_text()).items()}


def select(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep only the tests for ``--gpu-phase`` (and, in the parallel phase, ``--shard``)."""
    phase = config.getoption("--gpu-phase")
    if phase is None:
        return
    if phase == "serial":
        keep = [item for item in items if is_serial(item)]
    else:
        parallel = [item for item in items if not is_serial(item)]
        shard = config.getoption("--shard")
        if shard is None:
            keep = parallel
        else:
            index, count = parse_shard(shard)
            assignment = assign_shards(parallel, count, load_weights(config.getoption("--shard-weights")))
            keep = [item for item in parallel if assignment[shard_unit(item)] == index]
    kept = {id(item) for item in keep}
    dropped = [item for item in items if id(item) not in kept]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
    items[:] = keep


# ── Weights from an earlier run ───────────────────────────────────────────────


def junit_durations(xml_path: Path, rootdir: Path) -> dict[str, float]:
    """Per-test seconds from JUnit XML, keyed by pytest node ID."""
    durations: dict[str, float] = {}
    # Our own pytest JUnit output, not untrusted input.
    for case in ET.parse(xml_path).getroot().iter("testcase"):  # noqa: S314
        parts = (case.get("classname") or "").split(".")
        # The module is the longest prefix that is a file; anything after it is a class path.
        for cut in range(len(parts), 0, -1):
            module = Path(*parts[:cut]).with_suffix(".py")
            if (rootdir / module).is_file():
                nodeid = "::".join([module.as_posix(), *parts[cut:], case.get("name", "")])
                durations[nodeid] = float(case.get("time") or 0)
                break
    return durations


def main(argv: list[str]) -> int:
    """``weights <junit.xml>... <out.json>``: each test's longest time across the given runs."""
    if len(argv) < 3 or argv[0] != "weights":
        print("usage: python -m tests._gpu_shards weights <junit.xml>... <out.json>", file=sys.stderr)
        return 2
    rootdir = Path(__file__).resolve().parent.parent
    durations: dict[str, float] = {}
    for xml_path in argv[1:-1]:
        for nodeid, seconds in junit_durations(Path(xml_path), rootdir).items():
            durations[nodeid] = max(seconds, durations.get(nodeid, 0.0))
    Path(argv[-1]).write_text(json.dumps(durations, indent=0, sort_keys=True))
    print(f"{len(durations)} test durations, {sum(durations.values()) / 3600:.2f} h, written to {argv[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
