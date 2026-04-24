#!/bin/bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
BUILD_PATH="$SCRIPT_DIR/build"

rm -rf "$BUILD_PATH"
mkdir -p "$BUILD_PATH"
cd "$BUILD_PATH"
cmake ..
make -j$(nproc)

echo
echo "Build OK. Spouštění:"
echo "  ./build/sinusoid_response --iface eth0 --joint 0 --amplitude 0.15"
