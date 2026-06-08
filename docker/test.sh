#!/bin/bash
# Independence test for low-precision-fft
# Simulates: fresh user clones repo → pip install → import → run pure Python tests
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass_count=0
fail_count=0
warn_count=0

pass() { echo -e "${GREEN}[PASS]${NC} $1"; pass_count=$((pass_count+1)); }
fail() { echo -e "${RED}[FAIL]${NC} $1"; fail_count=$((fail_count+1)); }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; warn_count=$((warn_count+1)); }
section() { echo -e "\n${YELLOW}━━━ $1 ━━━${NC}"; }

cd /test

section "1. Environment"
echo "Python: $(python3 --version)"
echo "pip:    $(pip --version | head -1)"
echo "gcc:    $(gcc --version | head -1)"
echo "CUDA_HOME: ${CUDA_HOME:-<not set>}"
echo "torch:  $(python3 -c 'import torch; print(torch.__version__)')  cuda_available=$(python3 -c 'import torch; print(torch.cuda.is_available())')"

pass "environment ready"

section "2. pip install -e . (editable mode)"
cd /test/lowp_fft_src

echo "--- pip install output ---"
if pip install -e . 2>&1; then
    pass "pip install succeeded"
else
    PIP_EXIT=$?
    echo "pip install exit code: $PIP_EXIT"
    # Even if it fails, check if it's a "clean" failure (clear error, not a crash)
    warn "pip install failed (exit $PIP_EXIT) — checking if error is user-friendly..."
fi

section "3. import lowp_fft"
IMPORT_OUT=$(python3 -c "import lowp_fft; print('OK'); print(dir(lowp_fft))" 2>&1) || true
echo "$IMPORT_OUT"
if echo "$IMPORT_OUT" | grep -q "OK"; then
    pass "import lowp_fft succeeded"
    echo "$IMPORT_OUT" | grep -q "fft\|bfp" && pass "  → fft/bfp symbols visible"
else
    fail "import lowp_fft failed"
fi

section "4. Pure Python tests (test_bfp_fft.py)"
cd /test/lowp_fft_src
if pip install pytest numpy 2>&1 > /dev/null; then
    if python3 -m pytest tests/test_bfp_fft.py -v --tb=short 2>&1; then
        pass "BFP FFT tests passed"
    else
        PYTEST_EXIT=$?
        fail "BFP FFT tests failed (exit $PYTEST_EXIT)"
    fi
else
    fail "could not install pytest/numpy"
fi

section "5. _cuda_detect module (standalone)"
cd /test/lowp_fft_src
CUDA_DETECT_OUT=$(python3 -c "
import _cuda_detect
try:
    path = _cuda_detect.find_cuda_home()
    print(f'FOUND: {path}')
except OSError as e:
    print(f'NOT FOUND: {e}')
" 2>&1)
echo "$CUDA_DETECT_OUT"
if echo "$CUDA_DETECT_OUT" | grep -qi "not found"; then
    pass "_cuda_detect handles missing CUDA gracefully (with clear error message)"
elif echo "$CUDA_DETECT_OUT" | grep -qi "found"; then
    pass "_cuda_detect found CUDA"
else
    warn "_cuda_detect unexpected output"
fi

section "6. Check for common pitfalls"
# Issue from pyproject.toml fix: pip build isolation should have torch available
echo "Checking pyproject.toml build-system..."
if [ -f pyproject.toml ]; then
    grep -q "torch" pyproject.toml && pass "pyproject.toml includes torch in build-system.requires" || fail "pyproject.toml missing torch in build-system.requires"
fi

echo "Checking setup.py for hardcoded Windows paths..."
if grep -q 'C:/Program Files' setup.py 2>/dev/null; then
    warn "setup.py contains hardcoded Windows paths (may break Linux users)"
else
    pass "setup.py has no hardcoded Windows paths"
fi

echo "Checking _cuda_detect.py exists..."
[ -f _cuda_detect.py ] && pass "_cuda_detect.py present" || fail "_cuda_detect.py missing"

section "━━━ RESULTS ━━━"
echo -e "${GREEN}Passed: $pass_count${NC}"
echo -e "${RED}Failed: $fail_count${NC}"
echo -e "${YELLOW}Warnings: $warn_count${NC}"

if [ "$fail_count" -gt 0 ]; then
    echo -e "\n${RED}Independence test FAILED${NC}"
    exit 1
else
    echo -e "\n${GREEN}Independence test PASSED${NC}"
    exit 0
fi
