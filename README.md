# `bio-tools`

[![Lint Check](https://github.com/evo-design/bio-tools/actions/workflows/flake8_check.yml/badge.svg)](https://github.com/evo-design/bio-tools/actions/workflows/flake8_check.yml)
[![Test Pip Install](https://github.com/evo-design/bio-tools/actions/workflows/test-pip-install.yml/badge.svg)](https://github.com/evo-design/bio-tools/actions/workflows/test-pip-install.yml)

This repo contains the tool layer of the [`bio-programming`](https://github.com/evo-design/bio-programming/tree/main) project.


> [!NOTE]
> We currently in the process of transferring all of the infra from the `bio-programming` repo to this one. Note that some tests may not be passing on main at the moment.


## Set Up Instructions

```bash
conda create -n bio-tools python=3.12 uv -y
conda activate bio-tools
 uv pip install -e ".[dev]"
```