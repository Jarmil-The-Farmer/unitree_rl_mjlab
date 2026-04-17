#!/usr/bin/env bash
# Download the MuJoCo C library for the current platform.
# Usage: ./download_mujoco.sh [version]
#
# Detects x86_64 vs aarch64 and downloads the matching release from GitHub.
# Extracts into simulate/mujoco/ (the path expected by CMakeLists.txt).

set -euo pipefail

VERSION="${1:-3.3.6}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/mujoco"

ARCH="$(uname -m)"
case "$ARCH" in
  x86_64)  PLATFORM="linux-x86_64" ;;
  aarch64) PLATFORM="linux-aarch64" ;;
  *)       echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

TARBALL="mujoco-${VERSION}-${PLATFORM}.tar.gz"
URL="https://github.com/google-deepmind/mujoco/releases/download/${VERSION}/${TARBALL}"

if [ -f "${TARGET_DIR}/lib/libmujoco.so" ]; then
  EXISTING=$(readelf -h "${TARGET_DIR}/lib/libmujoco.so" 2>/dev/null | grep Machine | awk '{print $NF}' || true)
  case "$ARCH" in
    x86_64)  EXPECTED="X86-64" ;;
    aarch64) EXPECTED="AArch64" ;;
  esac
  if [ "$EXISTING" = "$EXPECTED" ]; then
    echo "MuJoCo ${VERSION} (${PLATFORM}) already present in ${TARGET_DIR}"
    exit 0
  else
    echo "Existing MuJoCo is for wrong architecture (${EXISTING}), re-downloading..."
    rm -rf "${TARGET_DIR}"
  fi
fi

echo "Downloading MuJoCo ${VERSION} for ${PLATFORM}..."
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

wget -q --show-progress -O "${TMPDIR}/${TARBALL}" "$URL"
echo "Extracting..."
tar xzf "${TMPDIR}/${TARBALL}" -C "${TMPDIR}"

# The tarball extracts to mujoco-VERSION/ (no platform suffix).
EXTRACTED="${TMPDIR}/mujoco-${VERSION}"
if [ ! -d "$EXTRACTED" ]; then
  echo "Error: expected directory ${EXTRACTED} not found after extraction"
  exit 1
fi

rm -rf "${TARGET_DIR}"
mv "$EXTRACTED" "${TARGET_DIR}"

echo "MuJoCo ${VERSION} (${PLATFORM}) installed to ${TARGET_DIR}"
