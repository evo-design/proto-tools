#!/bin/bash
set -euo pipefail

echo "Setting up TMalign standalone environment..."

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

echo "Compiling TMalign from source (using $CXX)..."
# Clone USalign repo (contains TMalign.cpp). The trap removes this build's
# directory on any exit, including a failed compile, which would otherwise
# leave the clone behind.
BUILD_DIR=$(mktemp -d)
trap 'rm -rf -- "$BUILD_DIR"' EXIT
git clone https://github.com/pylelab/USalign.git "$BUILD_DIR/usalign_src"
git -C "$BUILD_DIR/usalign_src" checkout 177cc8a2bbd3e2a6e9c5faaaa4ff5dfa1e6048f7

# Build into the temporary directory, then install on success, so a failed or
# partial compile cannot leave an unusable TMalign in the environment.
BIN_DIR="${VENV_PATH}/bin"
$CXX -O3 -ffast-math "$BUILD_DIR/usalign_src/TMalign.cpp" -lm -o "$BUILD_DIR/TMalign"

# Confirm the binary runs on this host. A compile can succeed and still produce
# an executable this host cannot load.
"$BUILD_DIR/TMalign" -h > "$BUILD_DIR/help.txt" 2>&1 || true
if ! grep -q 'TM-align' "$BUILD_DIR/help.txt"; then
    echo "ERROR: the compiled TMalign did not run. Output:" >&2
    head -20 "$BUILD_DIR/help.txt" >&2
    exit 1
fi

install -m 755 "$BUILD_DIR/TMalign" "$BIN_DIR/TMalign"

echo "TMalign setup complete!"
