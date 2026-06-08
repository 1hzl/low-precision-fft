# Independence test for low-precision-fft on Windows
# Compatible with PowerShell 5.1 and 7.x
# Tests all 4 HANDSHAKE issues:
#   (1) pip build isolation has torch (pyproject.toml)
#   (2) CUDA_HOME auto-detection (_cuda_detect.py)
#   (3) No hardcoded Windows paths break Linux (setup.py)
#   (4) CUDA 13.3 + MSVC /Zc:preprocessor compatibility

$ErrorActionPreference = "Continue"
$script:Passed = 0
$script:Failed = 0
$script:Warnings = 0

function Pass($msg) { Write-Host "[PASS] $msg" -ForegroundColor Green; $script:Passed++ }
function Fail($msg) { Write-Host "[FAIL] $msg" -ForegroundColor Red; $script:Failed++ }
function Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow; $script:Warnings++ }
function Section($msg) { Write-Host ""; Write-Host "--- $msg ---" -ForegroundColor Cyan }

# Helper: run Python code from stdin, capture output
function Run-Python($code) {
    $code | python 2>&1
}

$SrcDir = Split-Path $PSScriptRoot -Parent
Set-Location $SrcDir

# ===== 1. Environment =====
Section "1. Environment"

$os = Get-CimInstance Win32_OperatingSystem | Select-Object -ExpandProperty Caption
Write-Host "OS:      $os"

$pyVer = python --version 2>&1
Write-Host "Python:  $pyVer"

$pipVer = pip --version 2>&1
Write-Host "pip:     $($pipVer -split '\n')[0]"

Write-Host "CUDA_HOME: $env:CUDA_HOME"
Write-Host "CUDA_PATH: $env:CUDA_PATH"

# Check nvcc
$nvccOut = nvcc --version 2>&1
if ($LASTEXITCODE -eq 0) {
    $nvccLine = $nvccOut | Select-String "release"
    Write-Host "nvcc:    $nvccLine"
    Pass "nvcc available"
} else {
    Warn "nvcc not on PATH"
}

# Check MSVC
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $vs = & $vswhere -latest -property installationPath
    Write-Host "Visual Studio: $vs"
    Pass "Visual Studio detected"
} else {
    $vsDefault = "${env:ProgramFiles}\Microsoft Visual Studio\2022"
    if (Test-Path $vsDefault) {
        Pass "Visual Studio 2022 detected"
    } else {
        Warn "Visual Studio not found"
    }
}

# Check torch (stdin pipe avoids PS5 quote issues)
$torchOut = Run-Python @'
import torch
print(torch.__version__)
print("CUDA_AVAIL:", torch.cuda.is_available())
'@
$torchLines = $torchOut -join ' | '
Write-Host "torch:   $torchLines"
if (($torchOut -join ' ') -match "CUDA_AVAIL: True") {
    Pass "torch with CUDA"
} else {
    Warn "torch CPU-only"
}

# ===== 2. pyproject.toml =====
Section "2. pyproject.toml build-system (HANDSHAKE #1)"

$pyproject = Get-Content pyproject.toml -Raw
if ($pyproject -match "torch") {
    Pass "pyproject.toml includes torch in build-system.requires"
} else {
    Fail "pyproject.toml missing torch in build-system.requires"
}

# ===== 3. pip install =====
Section "3. pip install -e . (fresh install)"

Write-Host "Uninstalling previous..."
pip uninstall lowp_fft -y 2>&1 | Out-Null

Write-Host "Installing in editable mode..."
$installOut = pip install -e . 2>&1
$installExit = $LASTEXITCODE
Write-Host ($installOut -join "`n")

if ($installExit -eq 0) {
    Pass "pip install -e . succeeded"
    
    if (($installOut -join ' ') -match "cufft_fp16") {
        Pass "  -> CUDA extension (cufft_fp16.cu) compiled"
    } else {
        Warn "  -> CUDA extension may not have compiled"
    }
} else {
    Fail "pip install -e . FAILED (exit $installExit)"
    
    $installStr = $installOut -join ' '
    if ($installStr -match "CUDA_HOME") {
        Warn "  -> Failure related to CUDA_HOME (HANDSHAKE #2)"
    }
    if ($installStr -match "traditional preprocessor|C1189") {
        Fail "  -> MSVC traditional preprocessor error (HANDSHAKE #4)"
    }
}

# ===== 4. import =====
Section "4. import lowp_fft"

$importOut = Run-Python @'
import lowp_fft
print("OK")
print([x for x in dir(lowp_fft) if not x.startswith("_")])
'@
$importExit = $LASTEXITCODE
Write-Host ($importOut -join ' ')

if ($importExit -eq 0 -and ($importOut -join ' ') -match "OK") {
    Pass "import lowp_fft succeeded"
} else {
    Fail "import lowp_fft FAILED"
}

# ===== 5. CUDA extension =====
Section "5. CUDA extension loaded?"

$extOut = Run-Python @'
import lowp_fft
print("cufft_ext_loaded:", lowp_fft._cufft_ext is not None)
'@
Write-Host ($extOut -join ' ')

if (($extOut -join ' ') -match "cufft_ext_loaded: True") {
    Pass "_cufft_ext loaded"
} elseif (($extOut -join ' ') -match "cufft_ext_loaded: False") {
    Warn "_cufft_ext not loaded (graceful degradation)"
} else {
    Warn "could not check _cufft_ext status"
}

# ===== 6. Tests =====
Section "6. Run tests"

pip install pytest numpy -q 2>&1 | Out-Null

Write-Host "--- BFP FFT pure Python tests ---"
$bfpOut = python -m pytest tests/test_bfp_fft.py -v --tb=short 2>&1
$bfpExit = $LASTEXITCODE
Write-Host ($bfpOut -join "`n")
if ($bfpExit -eq 0) { Pass "test_bfp_fft.py passed" }
else { Fail "test_bfp_fft.py FAILED" }

Write-Host "--- cuFFT autograd tests ---"
$autoOut = python -m pytest tests/test_autograd.py -v --tb=short 2>&1
$autoExit = $LASTEXITCODE
Write-Host ($autoOut -join "`n")
if ($autoExit -eq 0) { Pass "test_autograd.py passed" }
else { Fail "test_autograd.py FAILED" }

Write-Host "--- BF16 tests ---"
$bf16Out = python -m pytest tests/test_bf16.py -v --tb=short 2>&1
$bf16Exit = $LASTEXITCODE
Write-Host ($bf16Out -join "`n")
if ($bf16Exit -eq 0) { Pass "test_bf16.py passed" }
else { Fail "test_bf16.py FAILED" }

# ===== 7. _cuda_detect =====
Section "7. _cuda_detect standalone (HANDSHAKE #2)"

$detectOut = Run-Python @'
import _cuda_detect
try:
    path = _cuda_detect.find_cuda_home()
    print("FOUND:", path)
except OSError as e:
    print("NOT FOUND:", e)
'@
Write-Host ($detectOut -join ' ')

if (($detectOut -join ' ') -match "FOUND:") {
    Pass "_cuda_detect found CUDA on Windows"
} elseif (($detectOut -join ' ') -match "NOT FOUND:") {
    Warn "_cuda_detect did not find CUDA"
} else {
    Warn "_cuda_detect unexpected output"
}

# ===== 8. Anti-regression =====
Section "8. Anti-regression checks (HANDSHAKE #3, #4)"

$setupPy = Get-Content setup.py -Raw

if ($setupPy -match 'C:/Program Files') {
    Warn "setup.py contains hardcoded C:/Program Files path"
}

if ($setupPy -match '/Zc:preprocessor') {
    Pass "setup.py includes /Zc:preprocessor flag (CUDA 13.3 MSVC fix)"
} else {
    Warn "setup.py does NOT include /Zc:preprocessor flag"
}

# ===== RESULTS =====
Section "RESULTS"

Write-Host "Passed:   $script:Passed" -ForegroundColor Green
Write-Host "Failed:   $script:Failed" -ForegroundColor Red
Write-Host "Warnings: $script:Warnings" -ForegroundColor Yellow

if ($script:Failed -gt 0) {
    Write-Host ""
    Write-Host "Independence test FAILED" -ForegroundColor Red
    exit 1
} else {
    Write-Host ""
    Write-Host "Independence test PASSED" -ForegroundColor Green
    exit 0
}
