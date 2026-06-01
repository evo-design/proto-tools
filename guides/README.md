# Guides

User-facing, runnable guides for `proto_tools`. Each guide is a notebook that
runs top to bottom and is the source for its page on
[docs.evodesign.org](https://docs.evodesign.org). Edit the notebook; the docs page is
generated from it. For the deep technical reference behind a guide, follow its
"Go deeper" link into [`../notes/`](../notes).

| Guide | What it covers | Hardware |
|-------|----------------|----------|
| [`tool_environments.ipynb`](tool_environments.ipynb) | How a tool's isolated environment is built on first call and cached afterward, shown live with a small ESM2 model | CPU |
| [`tool_persistence.ipynb`](tool_persistence.ipynb) | Keeping a model warm across calls (`persist()` / `persist_tool()` / `get()`) | 1 GPU |
| [`device_management.ipynb`](device_management.ipynb) | GPU selection, LRU eviction, CPU offload, and packing models per GPU | 1+ GPU |
| [`parallel_execution.ipynb`](parallel_execution.ipynb) | Fanning work out across GPUs with `ToolPool` | 2+ GPU |

## Re-running a guide

Guides carry their executed cell outputs so the rendered docs page can show real
results. After editing a guide's code, re-execute it on a machine with the
required hardware:

```bash
# Re-execute one guide and sanitize its outputs
python scripts/run_example_notebooks.py --only guides/tool_environments

# Re-execute every guide changed since main
python scripts/run_example_notebooks.py --changed
```

## Assets

Figures live under [`assets/`](assets), one subfolder per guide
(`assets/<guide>/<figure>.svg`). They are referenced from the notebook by
relative path and copied into the docs site at generation time.
