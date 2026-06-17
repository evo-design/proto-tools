#!/bin/bash
set -euo pipefail

echo "Setting up USalign standalone environment..."

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

echo "Compiling USalign from source (using $CXX)..."
# Clone USalign repo
BUILD_DIR=$(mktemp -d)
git clone https://github.com/pylelab/USalign.git "$BUILD_DIR/usalign_src"
git -C "$BUILD_DIR/usalign_src" checkout 177cc8a2bbd3e2a6e9c5faaaa4ff5dfa1e6048f7

# Install into the venv's bin directory
BIN_DIR="$(dirname "$(which python)")"
$CXX -O3 -ffast-math -lm -o "$BIN_DIR/USalign" "$BUILD_DIR/usalign_src/USalign.cpp"

# Clean up
rm -rf "$BUILD_DIR"

echo "USalign setup complete!"
