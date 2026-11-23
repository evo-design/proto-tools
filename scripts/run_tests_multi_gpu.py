#!/usr/bin/env python3
r"""scripts/run_tests_multi_gpu.py.

Run the test suite across every visible GPU at once (see ``tests/_gpu_shards.py``).

1. The parallel phase runs as one pytest per GPU, each seeing only its own GPU and holding whole
   toolkits, so no toolkit is ever loaded by two processes. Each shard gets its own ``--basetemp``,
   output and proto log, JUnit file and warm-worker report.
2. The serial phase (tests that need several GPUs, and the DeviceManager tests) then runs once,
   with every GPU visible. It runs last because it is mostly the lifecycle test, which loads every
   toolkit, and by then the shards have left their envs and weights in the file cache.

Environment builds and weight downloads already take cross-process locks under ``PROTO_HOME``, so
shards that share a tool environment wait for each other. Grouping tests by toolkit means
this is rarely needed.

Usage::

    python scripts/run_tests_multi_gpu.py --out runs/gpu \
        --weights runs/shard_weights.json -- tests --all --ext

Everything after ``--`` is passed to pytest. Exits non-zero if any phase or shard fails. Writes
``merged.xml`` (all JUnit results) and ``summary.json`` (per-phase timing and exit codes) to
``--out``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def visible_gpus() -> list[str]:
    """Return GPU indices from CUDA_VISIBLE_DEVICES, or every GPU listed by nvidia-smi."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        return [g for g in cvd.split(",") if g.strip()]
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True, check=True
    ).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def pytest_cmd(pytest_args: list[str], out: Path, name: str, extra: list[str]) -> list[str]:
    """The pytest command line for one phase or shard, with its own temp dir and JUnit file."""
    return [
        sys.executable,
        "-m",
        "pytest",
        *pytest_args,
        *extra,
        "-p",
        "no:cacheprovider",
        f"--basetemp={out / f'tmp_{name}'}",
        f"--junitxml={out / f'{name}.xml'}",
    ]


def launch(cmd: list[str], out: Path, name: str, env: dict[str, str]) -> subprocess.Popen:
    """Start one phase or shard, logging to ``<out>/<name>.log``.

    Its proto log goes to ``<out>/logs_<name>/``. Log files are named by start time
    or ``-k``, so shards started together could otherwise write to the same file.
    """
    log = (out / f"{name}.log").open("w")
    env = {
        **env,
        "PROTO_WARM_WORKERS_REPORT": str(out / f"{name}_pool.json"),
        "PROTO_LOG_DIR": str(out / f"logs_{name}"),
    }
    print(
        f"[{time.strftime('%H:%M:%S')}] start {name}: CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )
    return subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)


def merge_junit(parts: list[Path], target: Path) -> None:
    """Concatenate the testsuites of several JUnit files into one."""
    merged = ET.Element("testsuites")
    for part in parts:
        if not part.exists():
            continue
        root = ET.parse(part).getroot()  # noqa: S314 - our own pytest output
        suites = [root] if root.tag == "testsuite" else list(root)
        for suite in suites:
            suite.set("name", f"{suite.get('name', 'pytest')}:{part.stem}")
            merged.append(suite)
    ET.ElementTree(merged).write(target, encoding="utf-8", xml_declaration=True)


def main() -> int:
    """Run one shard per GPU, then the serial phase, and merge their results."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="directory for logs, JUnit files and the summary")
    parser.add_argument(
        "--weights", type=Path, help="per-test seconds for balancing (python -m tests._gpu_shards weights)"
    )
    parser.add_argument("--gpus", help="comma-separated physical GPU indices (default: all visible)")
    parser.add_argument("--skip-serial", action="store_true", help="run only the parallel phase")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="arguments for pytest, after --")
    args = parser.parse_args()
    pytest_args = args.pytest_args[1:] if args.pytest_args[:1] == ["--"] else args.pytest_args
    gpus = args.gpus.split(",") if args.gpus else visible_gpus()
    if not gpus:
        print("no GPUs visible; run pytest directly", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    base_env = dict(os.environ)
    summary: dict[str, dict] = {}
    started = time.time()

    shard_extra = ["--gpu-phase", "parallel"]
    if args.weights:
        shard_extra += ["--shard-weights", str(args.weights)]
    procs = {}
    t0 = time.time()
    for index, gpu in enumerate(gpus):
        name = f"shard{index}"
        cmd = pytest_cmd(pytest_args, args.out, name, [*shard_extra, "--shard", f"{index}/{len(gpus)}"])
        procs[name] = launch(cmd, args.out, name, {**base_env, "CUDA_VISIBLE_DEVICES": gpu})
    for name, proc in procs.items():
        summary[name] = {"exit": proc.wait(), "seconds": round(time.time() - t0), "gpu": gpus[int(name[5:])]}
        print(f"[{time.strftime('%H:%M:%S')}] {name} done: {summary[name]}", flush=True)

    if not args.skip_serial:
        t0 = time.time()
        proc = launch(
            pytest_cmd(pytest_args, args.out, "serial", ["--gpu-phase", "serial"]),
            args.out,
            "serial",
            {**base_env, "CUDA_VISIBLE_DEVICES": ",".join(gpus)},
        )
        summary["serial"] = {"exit": proc.wait(), "seconds": round(time.time() - t0)}
        print(f"[{time.strftime('%H:%M:%S')}] serial done: {summary['serial']}", flush=True)

    merge_junit([args.out / f"{name}.xml" for name in summary], args.out / "merged.xml")
    summary["total_seconds"] = round(time.time() - started)  # type: ignore[assignment]
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    # pytest exits 5 when a shard collects nothing, which is not a failure here.
    return 0 if all(v["exit"] in (0, 5) for k, v in summary.items() if isinstance(v, dict)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
