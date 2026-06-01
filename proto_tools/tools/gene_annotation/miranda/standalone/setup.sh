#!/bin/bash
set -euo pipefail
source standalone_helpers.sh

echo "Setting up miRanda standalone environment..."

# Need a C compiler (cc/clang on macOS, gcc on Linux) and make.
if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1; then
    echo "ERROR: no C compiler (cc/gcc) found." >&2
    exit 1
fi

echo "Cloning and compiling miRanda from source..."
BUILD_DIR=$(mktemp -d)
git clone --depth 1 https://github.com/hacktrackgnulinux/miranda.git "$BUILD_DIR/src"
cd "$BUILD_DIR/src"

# Delete prebuilt objects + the stale Linux binary so make recompiles natively
# (shallow-clone mtimes otherwise let make skip the rebuild).
find . \( -name '*.o' -o -name '*.a' \) -delete
rm -f src/miranda

# Refresh autotools platform scripts (not the tool binary) so arm64/aarch64 are recognized.
for f in config.guess config.sub; do
    curl -fsSL "https://git.savannah.gnu.org/cgit/config.git/plain/$f" -o "$f" && chmod +x "$f"
done

# Old-C build flags: -fcommon (un-extern'd globals in miranda.h), -fgnu89-inline (RNAlib's
# inline energy fns need GNU89 semantics or they emit no symbol), -w/-Wno-* (silence errors
# modern clang makes fatal).
export CFLAGS="-O2 -fcommon -fgnu89-inline -w -Wno-implicit-function-declaration -Wno-implicit-int -Wno-int-conversion -Wno-return-type"

# Install into the active venv (prefix is the parent of bin/, so miranda lands in venv/bin/miranda).
VENV_ROOT="$(dirname "$(dirname "$(which python)")")"
./configure --prefix="$VENV_ROOT"
make
make install

cd - >/dev/null
rm -rf "$BUILD_DIR"
echo "miRanda setup complete!"
