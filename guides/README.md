# Guides

Runnable guides for `proto_tools`. Each guide is a notebook that runs top to
bottom; open one and work through it.

| Guide | What it covers | Hardware |
|-------|----------------|----------|
| [`tool_environments.ipynb`](tool_environments.ipynb) | How a tool's isolated environment is built on first call and cached afterward, shown live with a small ESM2 model | CPU |
| [`tool_persistence.ipynb`](tool_persistence.ipynb) | Keeping a model warm across calls (`persist()` / `persist_tool()` / `get()`) | 1 GPU |
| [`device_management.ipynb`](device_management.ipynb) | GPU selection, LRU eviction, CPU offload, and packing models per GPU | 1+ GPU |
| [`parallel_execution.ipynb`](parallel_execution.ipynb) | Fanning work out across GPUs with `ToolPool` | 2+ GPU |
