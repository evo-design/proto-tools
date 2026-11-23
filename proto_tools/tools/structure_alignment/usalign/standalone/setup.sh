#!/bin/bash
set -euo pipefail

echo "Setting up USalign standalone environment..."

# Use system clang++ on macOS because the conda-forge GCC cross-compiler
# produces broken binaries on ARM64. Use g++ on other platforms.
if [[ "$(uname)" == "Darwin" ]]; then
    CXX="/usr/bin/clang++"
else
    CXX="g++"
fi
if ! command -v "$CXX" &>/dev/null; then
    echo "ERROR: $CXX not found. Install a C++ compiler." >&2
    exit 1
fi

echo "Compiling USalign from source (using $CXX)..."
# Clone the USalign source for compilation.
BUILD_DIR=$(mktemp -d)
git clone https://github.com/pylelab/USalign.git "$BUILD_DIR/usalign_src"
git -C "$BUILD_DIR/usalign_src" checkout 177cc8a2bbd3e2a6e9c5faaaa4ff5dfa1e6048f7

# Install the compiled binary into the environment's bin directory.
BIN_DIR="$(dirname "$(which python)")"
$CXX -O3 -ffast-math -lm -o "$BIN_DIR/USalign" "$BUILD_DIR/usalign_src/USalign.cpp"

# Remove the temporary source checkout after installation.
rm -rf "$BUILD_DIR"

echo "USalign setup complete!"
