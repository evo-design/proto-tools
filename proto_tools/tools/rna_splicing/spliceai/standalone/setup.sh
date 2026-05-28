#!/bin/bash
set -euo pipefail
source standalone_helpers.sh

echo "Setting up SpliceAI standalone environment..."

echo "Installing uv package manager..."
pip install uv

# TensorFlow 2.15 bundles Keras 2.15, which is required to load SpliceAI's bundled
# .h5 models (Keras 3 in TF>=2.16 cannot). TF 2.15 wheels support Python 3.9-3.11
# only (see python_version.txt). DETECTED_COMPUTE_PLATFORM is injected by
# compute_deps.py before this script runs.
TF_SPEC="${SPLICEAI_TF_SPEC:-tensorflow~=2.15.0}"
if [ "${DETECTED_COMPUTE_PLATFORM:-cpu}" = "cuda" ]; then
    echo "GPU detected; installing tensorflow[and-cuda]~=2.15.0"
    uv pip install "tensorflow[and-cuda]~=2.15.0"
else
    echo "CPU-only; installing ${TF_SPEC}"
    uv pip install "${TF_SPEC}"
fi

# Install SpliceAI without its (loose) deps so they can't override the TF 2.15
# pin above; its real runtime deps are pinned in requirements.txt.
echo "Installing SpliceAI (no deps) and supporting libraries..."
# Pinned to the tested release (also the current PyPI latest; SpliceAI is stable/unmaintained).
uv pip install --no-deps spliceai==1.3.1
uv pip install -r requirements.txt

# Fail fast if the Keras-2 / .h5 loader path or the SpliceAI import is broken.
python -c "import tensorflow as tf, keras; from keras.models import load_model; from spliceai.utils import Annotator, get_delta_scores; print('SpliceAI OK: TF', tf.__version__, 'Keras', keras.__version__)"

echo "SpliceAI setup complete!"
