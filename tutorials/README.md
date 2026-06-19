# Tutorials

Walkthrough notebooks for the core `proto_tools` features. Start here if you're new to the library — each notebook is standalone and runnable top to bottom.

## Contents

| # | Notebook | What you'll learn | Hardware |
|---|---|---|---|
| 01 | [Getting Started](01_getting_started.ipynb) | The universal `Input` → `Config` → `run_*()` → `Output` pattern. Tool discovery via `ToolRegistry`. | CPU |
| 02 | [Tool Persistence](02_tool_persistence.ipynb) | Keep models warm across calls with `ToolInstance.persist()` / `persist_tool()`. One-shot vs. persistent, named instances, manual lifecycle. | 1 GPU |
| 03 | [Device Management](03_device_management.ipynb) | How `DeviceManager` picks GPUs, LRU eviction, CPU offload, packing multiple models per GPU. | 1+ GPUs |
| 04 | [Parallel Execution](04_parallel_execution.ipynb) | Fan out work across GPUs with `ToolPool`. Auto-partitioning, cost-aware scheduling, warm-worker persistence. | 2+ GPUs |

## Prerequisites

- `proto_tools` installed: `pip install -e ".[dev]"` from the repo root.
- A GPU for notebooks 02–04. Notebook 01 runs on a laptop.
- First-call model downloads can take a few minutes. Subsequent runs are cached under `PROTO_HOME` (default `~/.proto/`).

## How these tutorials relate to the per-tool examples

Each tool also ships an `examples/example.ipynb` under `proto_tools/tools/{category}/{tool}/examples/` that shows the minimum call for that specific tool. The notebooks here are different — they teach the **framework-level** features (persistence, device management, parallelism) that apply to every tool.

## Questions?

- File issues at [evo-design/proto-tools/issues](https://github.com/evo-design/proto-tools/issues).
