#!/usr/bin/env bash

# Script to fix rpath for rrc binary
# Usage: ./fix_rpath.sh <binary_path>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <binary_path>"
    exit 1
fi

BINARY_PATH="$1"

if [ ! -f "$BINARY_PATH" ]; then
    echo "Error: Binary file $BINARY_PATH does not exist"
    exit 1
fi

# Check if patchelf is available
if ! command -v patchelf &> /dev/null; then
    echo "Warning: patchelf not found, skipping rpath patching"
    exit 0
fi

# Remove existing rpath and set new one
patchelf --remove-rpath "$BINARY_PATH"
patchelf --set-rpath '$ORIGIN:$ORIGIN/../lib' "$BINARY_PATH"
