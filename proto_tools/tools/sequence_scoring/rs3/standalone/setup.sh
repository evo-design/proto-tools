#!/bin/bash
set -euo pipefail
source standalone_helpers.sh
proto_ensure_uv

uv pip install -r requirements.txt
