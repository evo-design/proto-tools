# ColabFold Search Standalone Environment

This directory contains the setup for running ColabFold MSA search in an isolated Python environment to avoid dependency conflicts with the main bio-programming package.

## Overview

ColabFold requires AlphaFold dependencies that may conflict with other packages in the main environment. This standalone setup uses the `EnvManager` system to:

1. Create an isolated virtual environment (`.venvs/colabfold_search_env`)
2. Install ColabFold with all its dependencies
3. Run `colabfold_search` commands via the isolated environment

## Setup

The virtual environment is created automatically on first use of the `colabfold_search` tool. The `EnvManager` will:

1. Create a new venv at `.venvs/colabfold_search_env`
2. Run `setup.sh` to install dependencies from `requirements.txt`
3. Create a `STATUS.txt` file indicating setup success/failure
