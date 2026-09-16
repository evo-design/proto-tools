#!/bin/bash
# Standalone environment setup for Paperclip literature retrieval.

set -euo pipefail

source standalone_helpers.sh
proto_ensure_uv

# Paperclip publishes one mutable wheel URL rather than immutable versioned
# artifacts. Keep the last reviewed artifact pinned and fail closed when the
# upstream bytes change; updating these two values is an explicit review step.
PAPERCLIP_VERSION="0.7.52"
PAPERCLIP_SHA256="d25836294c63273f6cebe7d114fbefb2c45766d8cdb582277dd4ece5adb7a9b9"
PAPERCLIP_WHEEL_URL="https://paperclip.gxl.ai/paperclip.whl"

PAPERCLIP_TMPDIR=$(mktemp -d)
trap 'rm -rf "$PAPERCLIP_TMPDIR"' EXIT
PAPERCLIP_WHEEL="${PAPERCLIP_TMPDIR}/gxl_paperclip-${PAPERCLIP_VERSION}-py3-none-any.whl"

proto_download_verified "$PAPERCLIP_WHEEL_URL" "$PAPERCLIP_WHEEL" "$PAPERCLIP_SHA256"

uv pip install --python "$VENV_PATH/bin/python" "$PAPERCLIP_WHEEL"
INSTALLED_VERSION=$("$VENV_PATH/bin/python" -c \
    "from importlib.metadata import version; print(version('gxl-paperclip'))")
if [ "$INSTALLED_VERSION" != "$PAPERCLIP_VERSION" ]; then
    echo "ERROR: expected gxl-paperclip ${PAPERCLIP_VERSION}, installed ${INSTALLED_VERSION}" >&2
    exit 1
fi

echo "[paperclip] Setup complete."
