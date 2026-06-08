#!/bin/bash
# Build and run independence test for low-precision-fft
# Usage: bash run-independence-test.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Building Docker image ==="
sudo docker build -t lowp-fft-test "$SCRIPT_DIR"

echo ""
echo "=== Running independence test ==="
sudo docker run --rm \
    -v "$PROJECT_DIR:/test/lowp_fft_src:ro" \
    lowp-fft-test
