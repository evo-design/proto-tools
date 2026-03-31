#!/bin/bash
set -euo pipefail

echo "Setting up TMalign standalone environment..."

# Select C++ compiler: prefer system clang++ on macOS (conda-forge GCC
# cross-compiler produces broken binaries on ARM64), g++ elsewhere.
if [[ "$(uname)" == "Darwin" ]]; then
    CXX="/usr/bin/clang++"
else
    CXX="g++"
fi
if ! command -v "$CXX" &>/dev/null; then
    echo "ERROR: $CXX not found. Install a C++ compiler." >&2
    exit 1
fi

echo "Installing uv package manager..."
pip install uv

echo "Compiling TMalign from source (using $CXX)..."
# Clone USalign repo (contains TMalign.cpp)
BUILD_DIR=$(mktemp -d)
git clone --depth 1 https://github.com/pylelab/USalign.git "$BUILD_DIR/usalign_src"

# Install into the venv's bin directory
BIN_DIR="$(dirname "$(which python)")"
$CXX -O3 -ffast-math -lm -o "$BIN_DIR/TMalign" "$BUILD_DIR/usalign_src/TMalign.cpp"

# Clean up
rm -rf "$BUILD_DIR"

echo "TMalign setup complete!"
