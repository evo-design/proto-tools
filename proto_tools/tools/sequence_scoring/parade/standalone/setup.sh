#!/bin/bash
# Setup script for PARADE standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up PARADE standalone environment..."

proto_install_pytorch

echo "Installing remaining dependencies..."
uv pip install -r requirements.txt

# Download the bundled "parade" checkpoints from the pinned autosome-ru/parade commit (the files are
# MIT-licensed and published with the paper). Keep in sync with _BUNDLED_CHECKPOINTS in inference.py.
proto_resolve_weights_dir parade
PARADE_SAVED_MODELS="https://raw.githubusercontent.com/autosome-ru/parade/f8e3e02688918e40d85eff0976a984810ee04372/predictor"
proto_download_verified "${PARADE_SAVED_MODELS}/regression_multiple/saved_models/model-utr5-deltas-epoch%3D9-step%3D840.ckpt" \
    "${WEIGHTS_DIR}/parade-model-utr5-deltas-epoch9-step840.ckpt" \
    "efc8785bd860fc144221b00383467d2214958df562936c08a2797cc911810675"
proto_download_verified "${PARADE_SAVED_MODELS}/regression_multiple/saved_models/model-utr3-deltas-epoch%3D9-step%3D1330.ckpt" \
    "${WEIGHTS_DIR}/parade-model-utr3-deltas-epoch9-step1330.ckpt" \
    "61d2f6c5282345f548a334a96088389645697d1d1757a5ca911fa23f0d10d79b"
proto_download_verified "${PARADE_SAVED_MODELS}/regression_stability/saved_models/stability-epoch%3D24-step%3D725.ckpt" \
    "${WEIGHTS_DIR}/parade-stability-epoch24-step725.ckpt" \
    "cfa09ce18e40108a6ecf37c40b37af9cb42d18313116678f6b81f098a227007b"

echo "PARADE setup complete!"
