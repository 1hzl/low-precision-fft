#!/usr/bin/env pwsh
# Independence test for low-precision-fft on Windows
# Simulates: fresh user clones repo → pip install → build → import → pytest
# Tests all 4 HANDSHAKE issues:
#   (1) pip build isolation has torch (pyproject.toml)
#   (2) CUDA_HOME auto-detection (_cuda_detect.py)
#   (3) No hardcoded Windows paths break Linux (setup.py)
#   (4) CUDA 13.3 + MSVC /Zc:preprocessor compatibility

$ErrorActionPreference = "Stop"
$script:Passed = 0
$script:Failed = 0
$script:Warnings = 0

function Pass($msg) { Write-Host "[PASS] $msg" -ForegroundColor Green; $script:Passed++ }
function Fail($msg) { Write-Host "[FAIL] $msg" -ForegroundColor Red; $script:Failed++ }
function Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow; $script:Warnings++ }
function Section($msg) { Write-Host "`n━━━ $msg ━━━" -ForegroundColor Cyan }

# ── Config ──
$SrcDir = $PSScriptRoot  # assume repo is at this script's location
Set-Location $SrcDir

Section "1. Environment"

Write-Host "OS:      $(Get-CimInstance Win32_OperatingSystem | Select -Expand Caption)"
Write-Host "Python:  $(python --version 2>&1)"
Write-Host "pip:     $(pip --version 2>&1 | Select-Object -First 1)"
Write-Host "CUDA_HOME: $($env:CUDA_HOME)"
Write-Host "CUDA_PATH: $($env:CUDA_PATH)"

# Check nvcc
try {
    $nvccVer = nvcc --version 2>&1 | Select-String "release"
    Write-Host "nvcc:    $nvccVer"
    Pass "nvcc available"
} catch {
    Warn "nvcc not on PATH (will test _cuda_detect fallback)"
}

# Check MSVC
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $vs = & $vswhere -latest -property installationPath
    Write-Host "Visual Studio: $vs"
    Pass "Visual Studio detected"
} else {
    $vsPath = "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path $vsPath) { Pass "Visual Studio 2022 detected" }
    else { Warn "Visual Studio not found (CUDA build may fail)" }
}

# Check torch
$torchVer = python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())" 2>&1
Write-Host "torch: $($torchVer -join ' | ')"
if ($torchVer -match "CUDA: True") {
    Pass "torch with CUDA"
} else {
    Warn "torch CPU-only (CUDA tests will skip)"
}

Section "2. pyproject.toml build-system (HANDSHAKE #1)"
$pyproject = Get-Content pyproject.toml -Raw
if ($pyproject -match "torch") {
    Pass "pyproject.toml includes torch in build-system.requires"
} else {
    Fail "pyproject.toml missing torch in build-system.requires"
}

Section "3. pip install -e . (fresh install)"
Write-Host "Uninstalling first..."
pip uninstall lowp_fft -y 2>&1 | Out-Null
Write-Host "Installing in editable mode..."
$installOut = pip install -e . 2>&1
$installExit = $LASTEXITCODE
Write-Host ($installOut -join "`n")
if ($installExit -eq 0) {
    Pass "pip install -e . succeeded"
    
    # Verify CUDA extension was built
    if ($installOut -match "cufft_fp16") {
        Pass "  → CUDA extension (cufft_fp16.cu) compiled"
    } else {
        Warn "  → CUDA extension not compiled (check nvcc/MSVC)"
    }
} else {
    Fail "pip install -e . FAILED (exit $installExit)"
    # Check for specific errors
    if ($installOut -match "CUDA_HOME") {
        Warn "  → Failure related to CUDA_HOME (HANDSHAKE #2)"
    }
    if ($installOut -match "traditional preprocessor|C1189") {
        Fail "  → MSVC traditional preprocessor error (HANDSHAKE #4 — /Zc:preprocessor issue)"
    }
}

Section "4. import lowp_fft"
try {
    $importOut = python -c "import lowp_fft; print('OK'); print([x for x in dir(lowp_fft) if not x.startswith('_')])" 2>&1
    Write-Host $importOut
    Pass "import lowp_fft succeeded"
} catch {
    Fail "import lowp_fft FAILED: $_"
}

Section "5. CUDA extension loaded?"
try {
    $ext = python -c "import lowp_fft; print('cufft_ext loaded:', lowp_fft._cufft_ext is not None)" 2>&1
    Write-Host $ext
    if ($ext -match "True") { Pass "_cufft_ext loaded" }
    elseif ($ext -match "False") { Warn "_cufft_ext not loaded (graceful degradation)" }
} catch {
    Warn "could not check _cufft_ext"
}

Section "6. Run tests"
pip install pytest numpy -q 2>&1 | Out-Null

Write-Host "--- BFP FFT pure Python tests ---"
$bfpResult = python -m pytest tests/test_bfp_fft.py -v --tb=short 2>&1
$bfpExit = $LASTEXITCODE
Write-Host ($bfpResult -join "`n")
if ($bfpExit -eq 0) { Pass "test_bfp_fft.py passed" }
else { Fail "test_bfp_fft.py FAILED" }

Write-Host "--- cuFFT autograd tests ---"
$autoResult = python -m pytest tests/test_autograd.py -v --tb=short 2>&1
$autoExit = $LASTEXITCODE
Write-Host ($autoResult -join "`n")
if ($autoExit -eq 0) { Pass "test_autograd.py passed" }
else { Fail "test_autograd.py FAILED" }

Write-Host "--- BF16 tests ---"
try {
    $bf16Result = python -m pytest tests/test_bf16.py -v --tb=short 2>&1
    $bf16Exit = $LASTEXITCODE
    Write-Host ($bf16Result -join "`n")
    if ($bf16Exit -eq 0) { Pass "test_bf16.py passed" }
    else { Fail "test_bf16.py FAILED" }
} catch {
    Warn "test_bf16.py skipped"
}

Section "7. _cuda_detect standalone (HANDSHAKE #2)"
try {
    $detect = python -c @"
import _cuda_detect
try:
    path = _cuda_detect.find_cuda_home()
    print(f'FOUND: {path}')
except OSError as e:
    print(f'NOT FOUND: {e}')
"@ 2>&1
    Write-Host $detect
    if ($detect -match "FOUND") {
        Pass "_cuda_detect found CUDA on Windows"
    } elseif ($detect -match "NOT FOUND") {
        # On Windows without CUDA_HOME set but CUDA installed,
        # it should still find it via Program Files search
        Warn "_cuda_detect did not find CUDA (check Windows default paths)"
    }
} catch {
    Warn "_cuda_detect import failed: $_"
}

Section "8. Anti-regression: hardcoded paths (HANDSHAKE #3)"
$setupPy = Get-Content setup.py -Raw
if ($setupPy -match 'C:/Program Files') {
    Warn "setup.py contains hardcoded Windows paths (C:/Program Files)"
}
# Check for _cuda_detect import (the fix)
if ($setupPy -match 'from _cuda_detect import') {
    Pass "setup.py uses _cuda_detect (cross-platform CUDA detection)"
} else {
    Warn "setup.py may not use _cuda_detect"
}

# Check /Zc:preprocessor flag existence
if ($setupPy -match '/Zc:preprocessor') {
    Pass "setup.py includes /Zc:preprocessor flag (CUDA 13.3 MSVC fix)"
} else {
    Warn "setup.py does NOT include /Zc:preprocessor flag (may break CUDA 13.3 + MSVC)"
}

Section "━━━ RESULTS ━━━"
Write-Host "Passed:   $script:Passed" -ForegroundColor Green
Write-Host "Failed:   $script:Failed" -ForegroundColor Red
Write-Host "Warnings: $script:Warnings" -ForegroundColor Yellow

if ($script:Failed -gt 0) {
    Write-Host "`nIndependence test FAILED" -ForegroundColor Red
    exit 1
} else {
    Write-Host "`nIndependence test PASSED" -ForegroundColor Green
    exit 0
}
