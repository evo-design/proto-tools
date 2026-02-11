# `bio_programming_tools`

[![Lint Check](https://github.com/evo-design/bio-programming-tools/actions/workflows/flake8_check.yml/badge.svg)](https://github.com/evo-design/bio-programming-tools/actions/workflows/flake8_check.yml)
[![Run Unit Tests](https://github.com/evo-design/bio-programming-tools/actions/workflows/run-unit-tests.yml/badge.svg)](https://github.com/evo-design/bio-programming-tools/actions/workflows/run-unit-tests.yml)
[![Verify Documentation autogeneration synced](https://github.com/evo-design/bio-programming-tools/actions/workflows/docs_check.yml/badge.svg)](https://github.com/evo-design/bio-programming-tools/actions/workflows/docs_check.yml)
[![Validate Docs](https://github.com/evo-design/bio-programming-tools/actions/workflows/docs.yml/badge.svg)](https://github.com/evo-design/bio-programming-tools/actions/workflows/docs.yml)

This repo contains the tool layer of the [`bio-programming`](https://github.com/evo-design/bio-programming/tree/main) project.


> [!NOTE]
> We currently in the process of transferring all of the infra from the `bio-programming` repo to this one. Note that some tests may not be passing on main at the moment.


## Installation

```bash
# Using conda (recommended)
conda create -n bio_tools python=3.12 -y
conda activate bio_tools
pip install -e .

# For development
pip install -e ".[dev]"
```

## Usage

```python
from bio_programming_tools.tools.infra import EnvManager
from bio_programming_tools.entities.structures import Structure

# Use the tools and entities
```
