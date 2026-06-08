# collect-diagnostics.ps1
# One-click environment capture for low-precision-fft troubleshooting.
# Usage: .\scripts\collect-diagnostics.ps1
# Output: diagnostics-YYYYMMDD-HHMMSS.log (in project root)

$ErrorActionPreference = "Continue"
$ts = Get-Date -Format "yyyyMMdd-HHmmss"
$logDir = Split-Path $PSScriptRoot -Parent
$logFile = Join-Path $logDir "diagnostics-$ts.log"

Set-Location $logDir

function Write-Log($msg) {
    $line = "$msg"
    $line | Out-File -Append -Encoding utf8 $logFile
    Write-Host $line
}

function Run-Cmd($title, $cmd) {
    Write-Host "--- $title ---" -ForegroundColor Yellow
    "`n=== [$title] ===" | Out-File -Append -Encoding utf8 $logFile
    $out = Invoke-Expression $cmd 2>&1
    $exit = $LASTEXITCODE
    $out | Out-File -Append -Encoding utf8 $logFile
    Write-Host ($out -join "`n")
    if ($exit -ne 0) {
        Write-Host "  -> exit code: $exit" -ForegroundColor Red
        "  -> exit code: $exit" | Out-File -Append -Encoding utf8 $logFile
    }
    $out
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " low-precision-fft diagnostics collector" -ForegroundColor Cyan
Write-Host " Log: $logFile" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

"=== low-precision-fft diagnostics ===" | Out-File -Encoding utf8 $logFile
"Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')" | Out-File -Append -Encoding utf8 $logFile
"LogFile: $logFile" | Out-File -Append -Encoding utf8 $logFile

# ===== 1. System Info =====
Run-Cmd "System" '
Write-Output "OS: $(Get-CimInstance Win32_OperatingSystem | Select-Object -ExpandProperty Caption)"
Write-Output "Arch: $env:PROCESSOR_ARCHITECTURE"
Write-Output "Computer: $env:COMPUTERNAME"
'

Run-Cmd "Python" '
python --version 2>&1
Write-Output "Location: $(Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)"
'

Run-Cmd "pip" '
pip --version 2>&1
'

Run-Cmd "pip-list" '
pip list 2>&1 | Select-Object -First 80
'

Run-Cmd "pip-cache" '
pip cache info 2>&1
'

# ===== 2. CUDA Environment =====
Run-Cmd "CUDA-env-vars" '
Write-Output "CUDA_HOME = [$env:CUDA_HOME]"
Write-Output "CUDA_PATH = [$env:CUDA_PATH]"
Write-Output "CUDA_VISIBLE_DEVICES = [$env:CUDA_VISIBLE_DEVICES]"
Get-ChildItem env: | Where-Object { $_.Name -match "CUDA|NVCC|NVIDIA" } | ForEach-Object { "$($_.Name) = [$($_.Value)]" }
'

Run-Cmd "nvidia-smi" '
nvidia-smi 2>&1
'

Run-Cmd "nvcc" '
nvcc --version 2>&1
'

# ===== 3. PyTorch =====
Run-Cmd "torch-info" '
python -c "
import torch
print("Version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}:", torch.cuda.get_device_name(i))
        props = torch.cuda.get_device_properties(i)
        print(f"    Compute: sm_{props.major}{props.minor}, Memory: {props.total_mem // 1024**2} MiB")
" 2>&1
'

# ===== 4. MSVC =====
Run-Cmd "msvc-cl" '
Get-Command cl 2>&1
where cl 2>&1
'

Run-Cmd "msvc-vcvars" '
# Try to detect VS installation
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    & $vswhere -latest -property installationPath
} else {
    Write-Output "vswhere.exe not found — Visual Studio may not be installed"
}
'

# ===== 5. Git status =====
Run-Cmd "git-info" '
git --version 2>&1
git remote -v 2>&1
git log --oneline -5 2>&1
'

Run-Cmd "git-status" '
git status --short 2>&1
'

# ===== 6. pip install attempt =====
Run-Cmd "pip-install" '
pip install -e . --no-build-isolation 2>&1
'

# ===== 7. Import check =====
Run-Cmd "import-check" '
python -c "
try:
    import lowp_fft
    print("import lowp_fft: OK")
    print("Available:", [x for x in dir(lowp_fft) if not x.startswith(\"_\")])
    try:
        import lowp_fft._cufft_ext
        print(\"_cufft_ext: loaded\")
    except ImportError:
        print(\"_cufft_ext: NOT loaded (CUDA extension missing)\")
except Exception as e:
    print(f\"import lowp_fft: FAILED — {e}\")
" 2>&1
'

# ===== 8. Tests =====
Run-Cmd "pytest-install" '
pip install pytest numpy -q 2>&1
'

Run-Cmd "pytest-bfp" '
python -m pytest tests/test_bfp_fft.py -v --tb=short 2>&1
'

Run-Cmd "pytest-autograd" '
python -m pytest tests/test_autograd.py -v --tb=short 2>&1
'

Run-Cmd "pytest-bf16" '
python -m pytest tests/test_bf16.py -v --tb=short 2>&1
'

Run-Cmd "pytest-full" '
python -m pytest tests/ -v --tb=short 2>&1
'

# ===== Summary =====
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " Diagnostics complete" -ForegroundColor Cyan
Write-Host " Log saved to: $logFile" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

"
=== End of diagnostics ===
$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')
" | Out-File -Append -Encoding utf8 $logFile
