#!/bin/bash
set -euo pipefail

echo "Setting up TMalign standalone environment..."

# Check for g++ compiler
if ! command -v g++ &>/dev/null; then
    echo "ERROR: g++ not found. Install a C++ compiler (e.g., apt install g++)." >&2
    exit 1
fi

echo "Installing uv package manager..."
pip install uv

echo "Compiling TMalign from source..."
# Clone USalign repo (contains TMalign.cpp)
BUILD_DIR=$(mktemp -d)
git clone --depth 1 https://github.com/pylelab/USalign.git "$BUILD_DIR/usalign_src"

# Install into the venv's bin directory
BIN_DIR="$(dirname "$(which python)")"
g++ -O3 -ffast-math -lm -o "$BIN_DIR/TMalign" "$BUILD_DIR/usalign_src/TMalign.cpp"

# Clean up
rm -rf "$BUILD_DIR"

echo "TMalign setup complete!"
